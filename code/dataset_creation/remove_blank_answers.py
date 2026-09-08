"""
Remove already-saved answer crops that are actually blank.

This does not relocate pages or create new crops. It only rereads existing
answer images, compares them to blank reference crops, and refreshes dataset.json.
"""


import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from answer_extraction import (
    ANSWER_CROPS_DIR,
    DATASET_JSON_PATH,
    BLANK_CHECK_MAX_WORKERS,
    VLM_API_KEY,
    vlm_check_answered_page,
)

DEFAULT_MIN_ANSWERS = 100


def discover_groups(min_answers, page_filter=None):
    """Group saved crops by (student, page) for batched VLM checking."""
    label_counts = defaultdict(int)

    for page_dir in ANSWER_CROPS_DIR.glob("page_*"):
        pdf_page_number = int(page_dir.name.removeprefix("page_"))

        for problem_dir in page_dir.glob("problem_*"):
            label = problem_dir.name.removeprefix("problem_") + "."

            label_counts[(pdf_page_number, label)] += sum(
                1 for _ in problem_dir.glob("*.png")
            )

    # Only recheck high-volume questions by default. That is where old blank
    # crops are most likely to inflate answer counts.
    qualifying_labels = {
        key for key, count in label_counts.items() if count >= min_answers
    }

    groups = defaultdict(dict)

    for pdf_page_number, label in qualifying_labels:
        if page_filter is not None and pdf_page_number != page_filter:
            continue

        problem_dir = ANSWER_CROPS_DIR / f"page_{pdf_page_number}" / f"problem_{label.replace('.', '')}"

        for crop_path in problem_dir.glob("*.png"):
            student_id = crop_path.stem

            groups[(student_id, pdf_page_number)][label] = crop_path

    return groups, len(qualifying_labels)


def main():
    """Run the blank-crop cleanup pass."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-answers",
        type=int,
        default=DEFAULT_MIN_ANSWERS,
        help=f"Only check questions with at least this many saved crops right now (default {DEFAULT_MIN_ANSWERS}).",
    )

    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Also restrict to this one PDF page.",
    )

    args = parser.parse_args()

    if not VLM_API_KEY:
        print(
            "VIDAR_LITELLM_KEY is not set -- can't run the blank check at "
            'all. export VIDAR_LITELLM_KEY="sk-..." first.'
        )

        return

    print("Scanning answer_crops/ for qualifying questions...", flush=True)

    groups, n_qualifying_labels = discover_groups(args.min_answers, args.page)

    total_groups = len(groups)

    page_note = f", PDF page {args.page} only" if args.page is not None else ""

    print(
        f"{n_qualifying_labels} question(s) with >= {args.min_answers} answers"
        f"{page_note}: {sum(len(g) for g in groups.values())} crops across "
        f"{total_groups} (student, page) groups to check.\n",
        flush=True,
    )

    if total_groups == 0:
        return

    n_checked = 0

    n_deleted = 0

    n_errors = 0

    run_start = time.monotonic()

    def check_one_group(student_id, pdf_page_number, label_to_path):
        """Check one student's selected crops from one page."""
        label_to_crop = {
            label: Image.open(path) for label, path in label_to_path.items()
        }

        try:
            answered_by_label = vlm_check_answered_page(pdf_page_number, label_to_crop)

        except Exception as e:
            print(
                f"    {student_id} page {pdf_page_number}: check failed, "
                f"keeping all: {e}",
                flush=True,
            )

            return student_id, pdf_page_number, [], False

        deleted_here = []

        for label, path in label_to_path.items():


            # Default to keeping the crop if the VLM omits the label. Deleting
            # is only safe when the model explicitly returns false.
            if not answered_by_label.get(label, True):
                path.unlink()

                deleted_here.append(label)

        return student_id, pdf_page_number, deleted_here, True

    # Each (student, page) group is independent, so these checks can run safely
    # in parallel.
    with ThreadPoolExecutor(max_workers=BLANK_CHECK_MAX_WORKERS) as pool:
        futures = [
            pool.submit(check_one_group, student_id, pdf_page_number, label_to_path)
            for (student_id, pdf_page_number), label_to_path in groups.items()
        ]

        for future in as_completed(futures):
            student_id, pdf_page_number, deleted_here, succeeded = future.result()

            n_checked += 1

            n_deleted += len(deleted_here)

            n_errors += 0 if succeeded else 1

            avg_s = (time.monotonic() - run_start) / n_checked

            eta_s = avg_s * (total_groups - n_checked)

            if deleted_here:
                print(
                    f"    [{n_checked}/{total_groups}] {student_id} "
                    f"page {pdf_page_number}: dropped {deleted_here}  "
                    f"({avg_s:.1f}s/group avg, ~{eta_s / 60:.0f} min left)",
                    flush=True,
                )

    print(
        f"\nChecked {n_checked} group(s), deleted {n_deleted} blank "
        f"crop(s), {n_errors} group(s) errored (left untouched -- rerun "
        "to retry them).",
        flush=True,
    )


    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    for record in dataset:
        pdf_page_number = record["_debug"]["pdf_page"]

        label = record["_debug"]["problem_label"]

        safe_label = label.replace(".", "")

        out_dir = ANSWER_CROPS_DIR / f"page_{pdf_page_number}" / f"problem_{safe_label}"

        answer_files = sorted(out_dir.glob("*.png")) if out_dir.is_dir() else []

        # After deleting files, rebuild answer_image directly from disk so
        # dataset.json matches the current crop folder.
        record["answer_image"] = [
            f"answer_crops/page_{pdf_page_number}/problem_{safe_label}/{p.name}"
            for p in answer_files
        ]

    with open(DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"dataset.json's answer_image fields refreshed.\n    {DATASET_JSON_PATH}")


if __name__ == "__main__":
    main()
