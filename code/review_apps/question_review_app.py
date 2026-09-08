"""
Review extracted question crops and transcribed question text.

Use this after question_extraction.py to confirm each crop box lands on the
right problem and the VLM-read text in dataset.json looks correct.
"""


import json
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_JSON_PATH = PROJECT_ROOT / "data/algebra_dataset/dataset.json"
DATA_ROOT = DATASET_JSON_PATH.parent
PROBLEM_CROPS_DIR = DATA_ROOT / "problem_crops"


def load_dataset():
    """Load the question-level dataset."""
    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


DATASET = load_dataset()


# Keep page grouping in memory. This app reviews existing question extraction
# output and does not need to watch for new dataset rows while running.
PAGES = {}

for record in DATASET:
    page_number = record["_debug"]["pdf_page"]

    PAGES.setdefault(page_number, []).append(record)

PAGE_NUMBERS = sorted(PAGES.keys())


def render(idx):
    """Render one reference workbook page."""
    total = len(PAGE_NUMBERS)

    if total == 0:
        return idx, "No pages found in dataset.json.", None, ""

    idx = max(0, min(idx, total - 1))

    page_number = PAGE_NUMBERS[idx]

    records = sorted(records_for(page_number), key=lambda r: r["_debug"]["problem_label"])

    header = (
        f"**Page {idx + 1} / {total}**  —  PDF page {page_number}  —  "
        f"{len(records)} problem(s) detected"
    )

    debug_path = PROBLEM_CROPS_DIR / f"page_{page_number}" / "debug_crops.png"

    image_path = str(debug_path) if debug_path.is_file() else None

    sections_seen = {}

    blocks = []

    for record in records:
        label = record["_debug"]["problem_label"]

        set_name = record["_debug"]["set"]

        section = record["question_text"]["section_instructions"]

        problem = record["question_text"]["problem_instructions"]


        # SET instructions repeat across several problems, so show each one
        # only when it changes.
        section_md = ""

        if set_name not in sections_seen or sections_seen[set_name] != section:
            sections_seen[set_name] = section

            section_md = f"*{set_name} instruction: {section or '(none)'}*\n\n"

        if problem is None:
            problem_md = (
                "*(none -- either a graph/number-line/table problem with no "
                "printed text, or a VLM read failure; check the crop above)*"
            )

        else:
            problem_md = problem.replace("\n", "  \n")

        blocks.append(f"{section_md}**{label}**  {problem_md}")

    body = "\n\n---\n\n".join(blocks)

    return idx, header, image_path, body


def records_for(page_number):
    """Return all question records on one PDF page."""
    return PAGES[page_number]


def go_to_page(pdf_page_number):
    """Jump to a specific PDF page number from the input box."""
    if pdf_page_number in PAGE_NUMBERS:
        return render(PAGE_NUMBERS.index(pdf_page_number))

    return render(0)


with gr.Blocks(title="Question extraction review") as demo:
    # idx_state stores the current index into PAGE_NUMBERS, not the PDF page
    # number itself. That keeps Prev/Next simple.
    idx_state = gr.State(0)

    header = gr.Markdown()

    page_image = gr.Image(
        type="filepath", label="Detected regions (red boxes, from debug_crops.png)", height=650
    )

    body = gr.Markdown()

    with gr.Row():
        prev_btn = gr.Button("← Prev page")

        next_btn = gr.Button("Next page →", variant="primary")

        goto_input = gr.Number(label="Go to PDF page", precision=0)

        goto_btn = gr.Button("Go")

    outputs = [idx_state, header, page_image, body]

    demo.load(fn=lambda: render(0), outputs=outputs)

    prev_btn.click(fn=lambda i: render(i - 1), inputs=idx_state, outputs=outputs)

    next_btn.click(fn=lambda i: render(i + 1), inputs=idx_state, outputs=outputs)

    goto_btn.click(fn=go_to_page, inputs=goto_input, outputs=outputs)


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(DATA_ROOT)])
