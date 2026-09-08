"""
Review all students' answer crops page by page.

This view is useful while answer_extraction.py is running or after cleanup:
it rescans answer_crops/ on each render so new/deleted crops are reflected.
"""


import json
import shutil
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_JSON_PATH = PROJECT_ROOT / "data/algebra_dataset/dataset.json"
DATA_ROOT = DATASET_JSON_PATH.parent
ANSWER_CROPS_DIR = DATA_ROOT / "answer_crops"
UNLOCATED_PAGES_DIR = DATA_ROOT / "answer_crops_unlocated"
UNLOCATED_MANIFEST_PATH = UNLOCATED_PAGES_DIR / "manifest.json"

MAX_PROBLEMS_PER_PAGE = 4

NOTHING_MARKED = "*(nothing marked -- view a crop above, then Mark/unmark it)*"

NOTHING_VIEWED = "*(nothing viewed yet -- click a crop above)*"


def load_dataset():
    """Load question records and attached answer paths."""
    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


DATASET = load_dataset()

# The question list itself is static while the app runs; the answer crop
# folders are rescanned inside render().
PAGES = {}

for record in DATASET:
    page_number = record["_debug"]["pdf_page"]

    PAGES.setdefault(page_number, []).append(record)

PAGE_NUMBERS = sorted(PAGES.keys())


def load_unlocated_manifest():
    """Load page-location failures and VLM fallback pages."""
    if UNLOCATED_MANIFEST_PATH.is_file():
        return json.loads(UNLOCATED_MANIFEST_PATH.read_text())

    return {}


def students_missing_page(page_number, manifest):
    """List students whose scan failed or used fallback for one page."""
    failed = []

    vlm_fallback = []

    for student_id, entry in manifest.items():
        if isinstance(entry, list):
            entry = {"failed": entry}

        if page_number in entry.get("failed", []):
            failed.append(student_id)

        if page_number in entry.get("vlm_fallback", []):
            vlm_fallback.append(student_id)

    return sorted(failed), sorted(vlm_fallback)


def crop_dir(page_number, label):
    """Return a crop directory path."""
    return ANSWER_CROPS_DIR / f"page_{page_number}" / f"problem_{label.replace('.', '')}"


def crop_path(page_number, label, student_id):
    """Return a student crop path."""
    return crop_dir(page_number, label) / f"{student_id}.png"


def build_gallery_items(path_student_pairs, marked_paths=frozenset()):
    """Build gallery entries, adding a check mark to currently marked crops."""
    return [
        (path_str, f"{'✓ ' if path_str in marked_paths else ''}{student_id}")
        for path_str, student_id in path_student_pairs
    ]


def render(idx, status_override=None):
    """Render all answer galleries for one reference page."""
    total = len(PAGE_NUMBERS)

    idx = max(0, min(idx, total - 1)) if total else 0

    if total == 0:
        empty_slots = [
            (False, "", [], [], gr.update(choices=[], value=None), [], NOTHING_MARKED, None, NOTHING_VIEWED)
        ] * MAX_PROBLEMS_PER_PAGE

        return (idx, "No data available.", "", *_flatten(empty_slots))

    page_number = PAGE_NUMBERS[idx]

    records = sorted(PAGES[page_number], key=lambda r: r["_debug"]["problem_label"])

    all_labels = [r["_debug"]["problem_label"] for r in records]

    manifest = load_unlocated_manifest()

    failed_students, vlm_fallback_students = students_missing_page(page_number, manifest)

    header = f"**Page {idx + 1} / {total}**  —  PDF page {page_number}"

    status_bits = []

    if status_override:
        status_bits.append(f"✅ {status_override}")

    if failed_students:
        status_bits.append(
            f"\U0001f534 {len(failed_students)} student(s) could not be "
            "located on this page (missing entirely, or a genuinely "
            "absent source page)"
        )

    if vlm_fallback_students:
        status_bits.append(
            f"\U0001f7e1 {len(vlm_fallback_students)} student(s) located "
            "via the VLM fallback (less certain than the deterministic path)"
        )

    status = "\n\n".join(status_bits)

    slots = []

    for record in records:
        label = record["_debug"]["problem_label"]

        problem = record["question_text"]["problem_instructions"]

        problem_display = (
            problem
            if problem is not None
            else "*(none -- graph/table/number-line, or a VLM read failure)*"
        )


        # Re-scan the filesystem every render so Refresh picks up crops created
        # after the app was launched.
        crop_paths = sorted(crop_dir(page_number, label).glob("*.png"))

        paths_state = [(str(p), p.stem) for p in crop_paths]

        gallery_items = build_gallery_items(paths_state)

        question_md = (
            f"**{label}**  —  {len(crop_paths)} student answer(s) so far\n\n"
            f"{problem_display}"
        )

        move_choices = [l for l in all_labels if l != label]

        slots.append(
            (
                True,
                question_md,
                gallery_items,
                paths_state,
                gr.update(choices=move_choices, value=None),
                [],
                NOTHING_MARKED,
                None,
                NOTHING_VIEWED,
            )
        )

    while len(slots) < MAX_PROBLEMS_PER_PAGE:
        slots.append(
            (False, "", [], [], gr.update(choices=[], value=None), [], NOTHING_MARKED, None, NOTHING_VIEWED)
        )

    return (idx, header, status, page_number, *_flatten(slots))


def _flatten(slots):
    """Flatten repeated per-question slot values into Gradio outputs."""
    flat = []

    for (
        visible,
        question_md,
        gallery_items,
        paths_state,
        move_update,
        marked_state,
        marked_md,
        current_state,
        current_md,
    ) in slots:
        flat.extend(
            [
                gr.update(visible=visible),
                question_md,
                gallery_items,
                paths_state,
                move_update,
                marked_state,
                marked_md,
                current_state,
                current_md,
            ]
        )

    return flat


def go_to_page(pdf_page_number):
    """Jump to a PDF page."""
    if pdf_page_number in PAGE_NUMBERS:
        return render(PAGE_NUMBERS.index(pdf_page_number))

    return render(0)


def page_crop_count(page_number):
    """Count all saved answer crops for one PDF page."""
    page_dir = ANSWER_CROPS_DIR / f"page_{page_number}"

    return sum(1 for _ in page_dir.glob("*/*.png")) if page_dir.is_dir() else 0


def bulk_page_choices():
    """Build labels for the bulk-delete dropdown."""
    return [
        (f"PDF page {pg}  —  {page_crop_count(pg)} crop(s)", pg)
        for pg in PAGE_NUMBERS
    ]


def refresh_answer_image_for_pages(pdf_page_numbers):
    """Refresh dataset.json answer_image lists after moving/deleting crops."""
    affected = set(pdf_page_numbers)

    for record in DATASET:
        pdf_page_number = record["_debug"]["pdf_page"]

        if pdf_page_number not in affected:
            continue

        label = record["_debug"]["problem_label"]

        answer_files = sorted(crop_dir(pdf_page_number, label).glob("*.png"))

        safe_label = label.replace(".", "")

        # Rebuild from disk so dataset.json mirrors the actual answer_crops
        # folder after review actions.
        record["answer_image"] = [
            f"answer_crops/page_{pdf_page_number}/problem_{safe_label}/{p.name}"
            for p in answer_files
        ]

    with open(DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(DATASET, f, indent=4, ensure_ascii=False)


def delete_pages(selected_pages):
    """Delete every answer crop for selected PDF pages."""
    if not selected_pages:
        return "Nothing selected -- pick one or more pages first.", gr.update(
            choices=bulk_page_choices(), value=[]
        )

    deleted_counts = {}

    for pdf_page_number in selected_pages:
        n = page_crop_count(pdf_page_number)

        page_dir = ANSWER_CROPS_DIR / f"page_{pdf_page_number}"

        if page_dir.is_dir():
            shutil.rmtree(page_dir)

        deleted_counts[pdf_page_number] = n

    refresh_answer_image_for_pages(selected_pages)

    total = sum(deleted_counts.values())

    detail = ", ".join(f"page {pg} ({n})" for pg, n in deleted_counts.items())

    status = f"Deleted {total} crop(s) across {len(selected_pages)} page(s): {detail}"

    return status, gr.update(choices=bulk_page_choices(), value=[])


def marked_display(marked):
    """Render the human-readable list of currently marked crops."""
    if not marked:
        return NOTHING_MARKED

    names = ", ".join(f"`{m['student_id']}`" for m in marked)

    return f"**Marked ({len(marked)}):** {names}  \n*(✓ shows in the gallery above too)*"


def on_view(evt: gr.SelectData, paths):
    """Store the crop the reviewer most recently clicked."""
    if evt.index is None or evt.index >= len(paths):
        return None, NOTHING_VIEWED

    path_str, student_id = paths[evt.index]

    position = f"{evt.index + 1} of {len(paths)}"

    return (
        {"path": path_str, "student_id": student_id},
        f"**Viewing:** `{student_id}`  ({position})",
    )


def toggle_mark(current, marked, paths):
    """Add or remove the viewed crop from this gallery's marked list."""
    if not current:
        return marked, marked_display(marked), gr.update()

    marked = list(marked)

    # Match by path so two crops from the same student on different questions
    # do not collide.
    already_at = next((i for i, m in enumerate(marked) if m["path"] == current["path"]), None)

    if already_at is not None:
        marked.pop(already_at)

    else:
        marked.append(current)

    marked_paths = {m["path"] for m in marked}

    return marked, marked_display(marked), build_gallery_items(paths, marked_paths)


def delete_selected(marked, idx):
    """Delete the marked crops in the active gallery."""
    if not marked:
        return render(idx, status_override="Nothing marked -- click one or more crops first.")

    for m in marked:
        Path(m["path"]).unlink(missing_ok=True)

    ids = ", ".join(m["student_id"] for m in marked)

    return render(idx, status_override=f"Deleted {len(marked)} crop(s): {ids}")


def move_selected(marked, target_label, idx, page_number):
    """Move marked crops to another problem folder on the same page."""
    if not marked:
        return render(idx, status_override="Nothing marked -- click one or more crops first.")

    if not target_label:
        return render(idx, status_override="Pick a target question from the dropdown first.")

    dest_dir = crop_dir(page_number, target_label)

    dest_dir.mkdir(parents=True, exist_ok=True)

    moved_ids = []

    overwrote_ids = []

    for m in marked:
        source = Path(m["path"])

        student_id = m["student_id"]

        dest = dest_dir / f"{student_id}.png"

        if dest.is_file():
            overwrote_ids.append(student_id)

        shutil.move(str(source), str(dest))

        moved_ids.append(student_id)

    note = f"Moved {len(moved_ids)} crop(s) to {target_label}: {', '.join(moved_ids)}"

    if overwrote_ids:
        note += f"  (overwrote existing crops for: {', '.join(overwrote_ids)})"

    return render(idx, status_override=note)


with gr.Blocks(title="Answer extraction review (all students, page by page)") as demo:
    # Each visible problem slot owns its own gallery state, selected crop, and
    # marked crop list. That keeps review actions scoped to one question.
    idx_state = gr.State(0)

    page_number_state = gr.State(0)

    header = gr.Markdown()

    status = gr.Markdown()

    slot_widgets = []

    for _ in range(MAX_PROBLEMS_PER_PAGE):
        with gr.Column() as slot_col:
            slot_question = gr.Markdown()

            slot_gallery = gr.Gallery(
                label="Student answers (click to view -- ✓ shows what's marked)",
                columns=6,
                height=400,
            )

            slot_paths_state = gr.State([])

            slot_current_state = gr.State(None)

            slot_current_md = gr.Markdown(NOTHING_VIEWED)

            slot_mark_btn = gr.Button("☑️ Mark/unmark viewed")

            slot_marked_state = gr.State([])

            slot_marked_md = gr.Markdown(NOTHING_MARKED)

            with gr.Row():
                slot_delete_btn = gr.Button("\U0001f5d1️ Delete marked", variant="stop")

                slot_move_dropdown = gr.Dropdown(label="Move marked to...", choices=[])

                slot_move_btn = gr.Button("➡️ Move")

        slot_widgets.append(
            (
                slot_col,
                slot_question,
                slot_gallery,
                slot_paths_state,
                slot_move_dropdown,
                slot_marked_state,
                slot_marked_md,
                slot_current_state,
                slot_current_md,
                slot_mark_btn,
                slot_delete_btn,
                slot_move_btn,
            )
        )

    with gr.Row():
        prev_btn = gr.Button("← Prev page")

        next_btn = gr.Button("Next page →", variant="primary")

        refresh_btn = gr.Button("\U0001f504 Refresh (pick up new students)")

        goto_input = gr.Number(label="Go to PDF page", precision=0)

        goto_btn = gr.Button("Go")

    with gr.Accordion("Bulk delete pages", open=False):
        gr.Markdown(
            "Deletes every crop, for every student, on each page selected "
            "below -- for a page that's bad enough not to be worth fixing "
            "one crop at a time. This does not undo."
        )

        bulk_page_select = gr.Dropdown(
            label="Pages to delete",
            choices=bulk_page_choices(),
            multiselect=True,
        )

        with gr.Row():
            bulk_delete_btn = gr.Button(
                "\U0001f5d1️ Delete ALL crops for selected pages", variant="stop"
            )

            bulk_refresh_btn = gr.Button("\U0001f504 Refresh list")

        bulk_status = gr.Markdown()

    render_outputs = [idx_state, header, status, page_number_state] + [
        w
        for (
            _col,
            question,
            gallery,
            paths,
            move_dd,
            marked,
            marked_md,
            current,
            current_md,
            _mark_btn,
            _del_btn,
            _move_btn,
        ) in slot_widgets
        for w in (_col, question, gallery, paths, move_dd, marked, marked_md, current, current_md)
    ]

    demo.load(fn=lambda: render(0), inputs=None, outputs=render_outputs)

    prev_btn.click(fn=lambda i: render(i - 1), inputs=idx_state, outputs=render_outputs)

    next_btn.click(fn=lambda i: render(i + 1), inputs=idx_state, outputs=render_outputs)

    refresh_btn.click(fn=lambda i: render(i), inputs=idx_state, outputs=render_outputs)

    goto_btn.click(fn=go_to_page, inputs=goto_input, outputs=render_outputs)

    for (
        _col,
        _question,
        gallery,
        paths,
        move_dd,
        marked,
        marked_md,
        current,
        current_md,
        mark_btn,
        delete_btn,
        move_btn,
    ) in slot_widgets:
        gallery.select(fn=on_view, inputs=[paths], outputs=[current, current_md])

        mark_btn.click(
            fn=toggle_mark, inputs=[current, marked, paths], outputs=[marked, marked_md, gallery]
        )

        delete_btn.click(
            fn=delete_selected, inputs=[marked, idx_state], outputs=render_outputs
        )

        move_btn.click(
            fn=move_selected,
            inputs=[marked, move_dd, idx_state, page_number_state],
            outputs=render_outputs,
        )

    bulk_delete_btn.click(
        fn=delete_pages, inputs=bulk_page_select, outputs=[bulk_status, bulk_page_select]
    )

    bulk_refresh_btn.click(
        fn=lambda: gr.update(choices=bulk_page_choices()), inputs=None, outputs=bulk_page_select
    )


if __name__ == "__main__":
    demo.launch(server_port=7862, allowed_paths=[str(DATA_ROOT)])
