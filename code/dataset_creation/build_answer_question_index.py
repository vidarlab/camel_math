"""
Build answer/question indexes and metadata tables.

Reads dataset.json and writes three table-shaped JSON files:
answer_question_index.json, question_metadata.json, and answer_metadata.json.
"""


import json
from pathlib import Path

DATASET_JSON_PATH = Path("../../data/algebra_dataset/dataset.json")
DATA_ROOT = DATASET_JSON_PATH.parent
PROBLEM_CROPS_DIR = DATA_ROOT / "problem_crops"
BLANK_WORKBOOK_PATH = DATA_ROOT / "blank_workbook" / "abePROBLEMworkbook.pdf"
STUDENT_WORKBOOK_DIR = DATA_ROOT / "student_workbook" / "PROBLEMworkbooks"
UNLOCATED_MANIFEST_PATH = DATA_ROOT / "answer_crops_unlocated" / "manifest.json"
ANSWER_QUESTION_INDEX_PATH = DATA_ROOT / "answer_question_index.json"
QUESTION_METADATA_PATH = DATA_ROOT / "question_metadata.json"
ANSWER_METADATA_PATH = DATA_ROOT / "answer_metadata.json"

QUESTION_EXTRACTION_METHOD = "blank_workbook_extraction"
ANSWER_EXTRACTION_METHOD = "referenced_blank_workbook_extraction"

REQUIRED_INDEX_FIELDS = (
    "answer_id",
    "question_id",
)


def question_id_for(pdf_page, problem_label):
    """Build the stable join key for a question."""
    return f"page{pdf_page}_{problem_label.rstrip('.')}"


def load_unlocated_manifest():
    """Load page-location fallback metadata."""
    if not UNLOCATED_MANIFEST_PATH.is_file():
        return {}

    return json.loads(UNLOCATED_MANIFEST_PATH.read_text())


def page_location_method(student_id, pdf_page, unlocated_manifest):
    """Return the page-location method recorded for one answer."""
    manifest_entry = unlocated_manifest.get(student_id, {})


    # Older manifests used {student_id: [failed pages]}; newer manifests use
    # {student_id: {"failed": [...], "vlm_fallback": [...]}}.
    if isinstance(manifest_entry, list):
        manifest_entry = {"failed": manifest_entry}

    if pdf_page in set(manifest_entry.get("vlm_fallback", [])):
        return "vlm_fallback_page_location"

    return "deterministic_topic_header_and_banner_location"


def question_metadata_for(record):
    """Build one row for question_metadata.json."""
    debug = record["_debug"]

    pdf_page = debug["pdf_page"]

    problem_label = debug["problem_label"]

    question_id = question_id_for(pdf_page, problem_label)

    return {
        "question_id": question_id,
        "source_file": str(BLANK_WORKBOOK_PATH),
        "source_pdf_page": pdf_page,
        "problem_label": problem_label,
        "set": debug.get("set"),
        "question_image": record["question_image"],
        "extraction_method": QUESTION_EXTRACTION_METHOD,
    }


def answer_metadata_for(answer_id, record, answer_image, unlocated_manifest):
    """Build one row for answer_metadata.json."""
    debug = record["_debug"]

    pdf_page = debug["pdf_page"]

    problem_label = debug["problem_label"]

    question_id = question_id_for(pdf_page, problem_label)

    student_id = Path(answer_image).stem

    return {
        "answer_id": answer_id,
        "question_id": question_id,
        "student_id": student_id,
        "source_file": str(STUDENT_WORKBOOK_DIR / f"{student_id}.pdf"),
        "source_pdf_page": pdf_page,
        "problem_label": problem_label,
        "answer_image": answer_image,
        "extraction_method": ANSWER_EXTRACTION_METHOD,
        "page_location_method": page_location_method(
            student_id, pdf_page, unlocated_manifest
        ),
    }


def build_tables(dataset, unlocated_manifest):
    """Build all output tables from the nested dataset records."""
    answer_question_index = []

    question_metadata = []

    answer_metadata = []


    # answer_id is intentionally a plain sequence. The stable relationship to
    # the question is stored in answer_question_index.json.
    next_answer_id = 1

    for record in dataset:
        pdf_page = record["_debug"]["pdf_page"]

        problem_label = record["_debug"]["problem_label"]

        question_id = question_id_for(pdf_page, problem_label)

        question_metadata.append(question_metadata_for(record))

        for answer_image in record.get("answer_image") or []:
            answer_question_index.append(
                {
                    "answer_id": next_answer_id,
                    "question_id": question_id,
                }
            )

            answer_metadata.append(
                answer_metadata_for(
                    next_answer_id, record, answer_image, unlocated_manifest
                )
            )

            next_answer_id += 1

    return answer_question_index, question_metadata, answer_metadata


def validate(dataset, answer_question_index, question_metadata, answer_metadata):
    """Check counts, joins, file paths, and answer_id continuity."""
    n_questions = len(dataset)

    n_answers = len(answer_question_index)

    n_original_answer_entries = sum(len(r.get("answer_image") or []) for r in dataset)

    unique_students = {a["student_id"] for a in answer_metadata}

    unique_questions = {r["question_id"] for r in answer_question_index}

    metadata_question_ids = {q["question_id"] for q in question_metadata}

    answer_metadata_question_ids = {a["question_id"] for a in answer_metadata}


    # Count zero-response questions too, so the reported minimum is meaningful.
    responses_per_question = [len(r.get("answer_image") or []) for r in dataset]

    answer_id_counts = {}

    for row in answer_question_index:
        answer_id = row["answer_id"]

        answer_id_counts[answer_id] = answer_id_counts.get(answer_id, 0) + 1

    n_duplicate_answer_ids = sum(c - 1 for c in answer_id_counts.values() if c > 1)


    answer_ids_sorted = sorted(r["answer_id"] for r in answer_question_index)

    # This catches skipped/repeated IDs separately from duplicate detection.
    is_continuous = answer_ids_sorted == list(range(1, n_answers + 1))

    n_missing_answer_files = sum(
        1 for a in answer_metadata if not (DATA_ROOT / a["answer_image"]).is_file()
    )

    n_missing_question_files = sum(
        1
        for q in question_metadata
        if not (PROBLEM_CROPS_DIR / q["question_image"]).is_file()
    )

    n_missing_fields = sum(
        1
        for row in answer_question_index
        if any(not row.get(field) for field in REQUIRED_INDEX_FIELDS)
    )

    n_duplicate_question_metadata_ids = len(question_metadata) - len(
        metadata_question_ids
    )

    n_answer_metadata_without_question = len(
        answer_metadata_question_ids - metadata_question_ids
    )

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"1. Questions in the original dataset:        {n_questions}")

    print(f"2. Individual answers in the index:           {n_answers}")

    print(f"3. Unique students:                            {len(unique_students)}")

    print(
        f"4. Unique questions represented in the index:  {len(unique_questions)} "
        f"(of {n_questions} total -- the rest currently have 0 responses)"
    )

    print(
        "5. Responses per question -- min={}, max={}, mean={:.2f} "
        "(computed over all {} questions in the original dataset)".format(
            min(responses_per_question),
            max(responses_per_question),
            n_answers / n_questions if n_questions else 0.0,
            n_questions,
        )
    )

    print(f"6. Duplicate answer_ids (should be 0):         {n_duplicate_answer_ids}")

    print(
        f"6b. answer_ids run continuously 1..{n_answers} with no gaps: "
        f"{'yes' if is_continuous else 'NO -- see below'}"
    )

    print(
        f"7. answer_image paths that do not exist "
        f"(checked once per answer): {n_missing_answer_files}"
    )

    print(
        f"8. question_image paths that do not exist "
        f"(checked once per question): {n_missing_question_files}"
    )

    print(f"9. Index rows missing answer_id/question_id:  {n_missing_fields}")

    print(f"10. Question metadata rows:                   {len(question_metadata)}")

    print(
        f"11. Duplicate question metadata IDs:          "
        f"{n_duplicate_question_metadata_ids}"
    )

    print(f"12. Answer metadata rows:                     {len(answer_metadata)}")

    print(
        f"13. Answer metadata rows without question:    "
        f"{n_answer_metadata_without_question}"
    )

    print()

    ok = (
        n_answers == n_original_answer_entries
        and len(question_metadata) == n_questions
        and len(answer_metadata) == n_answers
        and n_duplicate_question_metadata_ids == 0
        and n_answer_metadata_without_question == 0
    )

    status = "OK" if ok else "MISMATCH"

    print(
        f"Answer count == total answer_image entries in original JSON? "
        f"{n_answers} == {n_original_answer_entries}  [{status}]"
    )

    return ok


def main():
    """Read dataset.json, validate outputs, and write the three JSON tables."""
    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    unlocated_manifest = load_unlocated_manifest()

    answer_question_index, question_metadata, answer_metadata = build_tables(
        dataset, unlocated_manifest
    )

    ok = validate(dataset, answer_question_index, question_metadata, answer_metadata)

    with open(ANSWER_QUESTION_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(answer_question_index, f, indent=4, ensure_ascii=False)

    with open(QUESTION_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(question_metadata, f, indent=4, ensure_ascii=False)

    with open(ANSWER_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(answer_metadata, f, indent=4, ensure_ascii=False)

    print()
    print(
        f"Wrote {len(answer_question_index)} answer/question index row(s) to:\n"
        f"    {ANSWER_QUESTION_INDEX_PATH}"
    )
    print(
        f"Wrote {len(question_metadata)} question metadata row(s) to:\n"
        f"    {QUESTION_METADATA_PATH}"
    )
    print(
        f"Wrote {len(answer_metadata)} answer metadata row(s) to:\n"
        f"    {ANSWER_METADATA_PATH}"
    )

    if not ok:
        print(
            "\nWARNING: answer count did not match the original JSON's "
            "answer_image entry count -- inspect before using this index."
        )


if __name__ == "__main__":
    main()
