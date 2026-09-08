"""
Review extracted answer crops for one student at a time.

Use this to inspect whether a student's scan was located correctly and whether
the saved crops line up with each question on the reference page.
"""


import json
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_JSON_PATH = PROJECT_ROOT / "data/algebra_dataset/dataset.json"
DATA_ROOT = DATASET_JSON_PATH.parent
ANSWER_CROPS_DIR = DATA_ROOT / "answer_crops"
UNLOCATED_PAGES_DIR = DATA_ROOT / "answer_crops_unlocated"
UNLOCATED_MANIFEST_PATH = UNLOCATED_PAGES_DIR / "manifest.json"

MAX_PROBLEMS_PER_PAGE = 4


def load_dataset():
    """Load question records and attached answer paths."""
    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_unlocated_manifest():
    """Load pages that failed matching or used the VLM fallback."""
    if UNLOCATED_MANIFEST_PATH.is_file():
        return json.loads(UNLOCATED_MANIFEST_PATH.read_text())

    return {}


def discover_students():
    """Find students that have either saved crops or failure metadata."""
    ids = {p.stem for p in ANSWER_CROPS_DIR.glob("*/*/*.png")}

    ids.update(load_unlocated_manifest().keys())

    return sorted(ids)


DATASET = load_dataset()

PAGES = {}

for record in DATASET:
    page_number = record["_debug"]["pdf_page"]

    PAGES.setdefault(page_number, []).append(record)

PAGE_NUMBERS = sorted(PAGES.keys())

UNLOCATED_MANIFEST = load_unlocated_manifest()

STUDENT_IDS = discover_students()


def render(student_id, idx):
    """Render one student's view of one reference page."""
    total = len(PAGE_NUMBERS)

    idx = max(0, min(idx, total - 1)) if total else 0

    if total == 0 or not student_id:
        empty_slots = [(False, "", None)] * MAX_PROBLEMS_PER_PAGE

        return (
            idx,
            "No data available.",
            gr.update(visible=False),
            None,
            "",
            gr.update(visible=True),
            *_flatten(empty_slots),
        )

    page_number = PAGE_NUMBERS[idx]

    records = sorted(PAGES[page_number], key=lambda r: r["_debug"]["problem_label"])


    # The manifest tells the UI whether to show answer crops or an unverified
    # snapshot for pages the extractor could not confidently locate.
    manifest_entry = UNLOCATED_MANIFEST.get(student_id, {})

    if isinstance(manifest_entry, list):
        manifest_entry = {"failed": manifest_entry}

    failed_pages = set(manifest_entry.get("failed", []))

    vlm_fallback_pages = set(manifest_entry.get("vlm_fallback", []))

    page_failed = page_number in failed_pages

    page_vlm_fallback = page_number in vlm_fallback_pages

    if page_failed:
        status = "\U0001f534 COULD NOT LOCATE THIS PAGE"
    elif page_vlm_fallback:
        status = "\U0001f7e1 located via VLM fallback"
    else:
        status = "✅ located"

    header = (
        f"**Page {idx + 1} / {total}**  —  PDF page {page_number}  —  "
        f"student `{student_id}`  —  {status}"
    )

    if page_failed:
        snapshot_path = UNLOCATED_PAGES_DIR / f"page_{page_number}" / f"{student_id}.png"

        image_path = str(snapshot_path) if snapshot_path.is_file() else None

        body = (
            "**This page could not be verified as the right page in this "
            "student's scan** (topic header or SET banner count didn't "
            "match closely enough). The image above is an *unverified* "
            "snapshot of whatever sits at this page's naive expected "
            "position in the scan -- use it to judge whether it's "
            "genuinely a different/missing page, or a false rejection on "
            "a noisy scan.\n\n**Questions on this page (for reference -- "
            "none were checked for an answer):**\n\n"
        )

        body += "\n\n".join(
            f"**{r['_debug']['problem_label']}**  "
            f"{r['question_text']['problem_instructions'] or '*(none)*'}"
            for r in records
        )

        empty_slots = [(False, "", None)] * MAX_PROBLEMS_PER_PAGE

        return (
            idx,
            header,
            gr.update(visible=True),
            image_path,
            body,
            gr.update(visible=False),
            *_flatten(empty_slots),
        )

    slots = []

    for record in records:
        label = record["_debug"]["problem_label"]

        safe_label = label.replace(".", "")

        problem = record["question_text"]["problem_instructions"]

        problem_display = (
            problem
            if problem is not None
            else "*(none -- graph/table/number-line, or a VLM read failure)*"
        )

        crop_path = (
            ANSWER_CROPS_DIR
            / f"page_{page_number}"
            / f"problem_{safe_label}"
            / f"{student_id}.png"
        )

        if crop_path.is_file():
            question_md = f"**{label}**\n\n{problem_display}"

            image_path = str(crop_path)

        else:
            # A missing crop on a located page usually means this workbook has
            # not been extracted since the latest answer-extraction behavior.
            question_md = f"**{label}**\n\n{problem_display}\n\n*(no crop saved yet)*"

            image_path = None

        slots.append((True, question_md, image_path))

    while len(slots) < MAX_PROBLEMS_PER_PAGE:
        slots.append((False, "", None))

    return (
        idx,
        header,
        gr.update(visible=False),
        None,
        "",
        gr.update(visible=True),
        *_flatten(slots),
    )


def _flatten(slots):
    """Flatten repeated slot data into the output list Gradio expects."""
    flat = []

    for visible, question_md, image_path in slots:
        flat.extend([gr.update(visible=visible), question_md, image_path])

    return flat


def go_to_page(student_id, pdf_page_number):
    """Jump to a specific PDF page for the selected student."""
    if pdf_page_number in PAGE_NUMBERS:
        return render(student_id, PAGE_NUMBERS.index(pdf_page_number))

    return render(student_id, 0)


with gr.Blocks(title="Answer extraction review") as demo:
    # idx_state is the position within PAGE_NUMBERS. student_dropdown controls
    # which student's crop files are loaded for that page.
    idx_state = gr.State(0)

    student_dropdown = gr.Dropdown(
        choices=STUDENT_IDS,
        value=STUDENT_IDS[0] if STUDENT_IDS else None,
        label="Student",
    )

    header = gr.Markdown()

    with gr.Column(visible=False) as failed_section:
        failed_image = gr.Image(
            type="filepath",
            label="Raw unverified snapshot (page could not be located)",
            height=650,
        )

        failed_body = gr.Markdown()

    with gr.Column(visible=True) as success_section:
        slot_widgets = []

        with gr.Row():
            for _ in range(MAX_PROBLEMS_PER_PAGE):
                with gr.Column() as slot_col:
                    slot_question = gr.Markdown()

                    slot_image = gr.Image(type="filepath", label="Answer", height=320)

                slot_widgets.append((slot_col, slot_question, slot_image))

    with gr.Row():
        prev_btn = gr.Button("← Prev page")

        next_btn = gr.Button("Next page →", variant="primary")

        goto_input = gr.Number(label="Go to PDF page", precision=0)

        goto_btn = gr.Button("Go")

    outputs = [
        idx_state,
        header,
        failed_section,
        failed_image,
        failed_body,
        success_section,
    ] + [w for triple in slot_widgets for w in triple]

    demo.load(
        fn=lambda sid: render(sid, 0),
        inputs=student_dropdown,
        outputs=outputs,
    )

    student_dropdown.change(
        fn=lambda sid: render(sid, 0), inputs=student_dropdown, outputs=outputs
    )

    prev_btn.click(
        fn=lambda sid, i: render(sid, i - 1),
        inputs=[student_dropdown, idx_state],
        outputs=outputs,
    )

    next_btn.click(
        fn=lambda sid, i: render(sid, i + 1),
        inputs=[student_dropdown, idx_state],
        outputs=outputs,
    )

    goto_btn.click(
        fn=go_to_page, inputs=[student_dropdown, goto_input], outputs=outputs
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(DATA_ROOT)])
