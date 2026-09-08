"""
Extract question crops and printed text from the blank reference workbook.

Output:
  - problem_crops/page_N/problem_X.png
  - problem_crops/page_N/debug_*.png
  - dataset.json with one row per question
"""

import base64
import functools
import io
import os
import fitz
import json
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import ImageDraw

from workbook_layout import (
    assign_problems_to_sets,
    create_problem_regions,
    find_problem_anchors,
    find_set_banners,
    render_page,
)

load_dotenv(override=True)


PDF_PATH = Path("../../data/algebra_dataset/blank_workbook/abePROBLEMworkbook.pdf")
OUTPUT_DIR = Path("../../data/algebra_dataset/problem_crops")
DATASET_JSON_PATH = OUTPUT_DIR.parent / "dataset.json"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


VLM_ENDPOINT_URL = "https://vllm.vidarlab.net"
VLM_MODEL = "qwen3-vl-32b"
VLM_API_KEY = os.environ.get("VIDAR_LITELLM_KEY")

VLM_PROMPT = (
    "This image is one cropped cell from a math worksheet. It contains "
    "printed text and, separately, a student's handwritten work.\n\n"
    "Respond with ONLY the printed text, and nothing else -- no question "
    "number, no letter label, no explanation, and never any handwritten "
    "content.\n\n"
    "Include EVERY printed line in the cell, not just the first sentence "
    "and not just whichever part is phrased as a question. Some problems "
    "print a word problem, THEN a stated answer to evaluate, THEN a "
    "separate instruction -- all of that is printed text and all of it "
    "must be included verbatim, even the parts that are statements "
    "rather than questions. Confirmed as a real failure mode, not a "
    "hypothetical one: asked for only 'the printed question', a model "
    "dropped a printed 'Answer: 4 1/2 hours' line entirely because it "
    "isn't itself phrased as a question, even though it's printed text "
    "in the cell like everything else. For example, if the printed text "
    "reads:\n"
    "    1a. Word Problem: Sam makes $12 per hour at his job. If he made\n"
    "    $54 on Monday, how many hours did he work that day?\n"
    "    Answer: 4 1/2 hours\n"
    "    Is the answer reasonable? Explain why or why not.\n"
    "your entire response should be exactly:\n"
    "    Word Problem: Sam makes $12 per hour at his job. If he made $54 "
    "on Monday, how many hours did he work that day?\n"
    "    Answer: 4 1/2 hours\n"
    "    Is the answer reasonable? Explain why or why not.\n\n"
    "Strip only the leading question number like '1a.', '3b.', or '2.' "
    "before it. For example, if the printed text reads:\n"
    "    3a. |5| - |-7|\n"
    "your entire response should be exactly:\n"
    "    |5| - |-7|\n\n"
    "Use plain-text math notation: |x| for absolute value, ^ for exponents "
    "(e.g. x^2), and a/b for fractions. Do not transcribe or describe any "
    "handwritten content. If there is no printed question visible, "
    "respond with exactly: NONE"
)


_VLM_LABEL_PREFIX_RE = re.compile(r"^\s*\d+[a-zA-Z]?(?:\.(?!\d)|\))\s*")


def extract_problem_instructions_vlm(
    cell_img, endpoint_url, model, api_key=None, timeout=60
):
    """Send one question crop to the VLM and return printed problem text."""
    buf = io.BytesIO()

    cell_img.save(buf, format="PNG")

    # The endpoint accepts OpenAI-style image_url payloads, so local PIL images
    # are encoded as data URLs instead of written to temporary files.
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    headers = {"Content-Type": "application/json"}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = endpoint_url.rstrip("/")

    if not url.endswith("/chat/completions"):
        url = f"{url}/v1/chat/completions"

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)

    resp.raise_for_status()

    text = resp.json()["choices"][0]["message"]["content"].strip()

    # The prompt asks the model to omit labels, but this strips them if the
    # model returns `1a.` or similar anyway.
    text = _VLM_LABEL_PREFIX_RE.sub("", text)

    return None if text.upper() == "NONE" else text


if VLM_API_KEY:
    extract_problem_instructions = functools.partial(
        extract_problem_instructions_vlm,
        endpoint_url=VLM_ENDPOINT_URL,
        model=VLM_MODEL,
        api_key=VLM_API_KEY,
    )
else:
    extract_problem_instructions = None

    print(
        "VIDAR_LITELLM_KEY is not set -- skipping VLM extraction, "
        "problem_instructions will be left as None.\n"
        '    export VIDAR_LITELLM_KEY="sk-..." to enable it.'
    )


def draw_anchor_debug(image, anchors, banners, output_path):
    """Save a page image with problem labels and SET banners marked."""
    debug = image.copy()

    draw = ImageDraw.Draw(debug)


    for item in anchors:
        r = item["rect"]

        draw.rectangle([r.x0, r.y0, r.x1, r.y1], outline="red", width=4)

        draw.text((r.x0, max(0, r.y0 - 25)), item["label"], fill="red")


    for item in banners:
        r = item["rect"]

        draw.rectangle([r.x0, r.y0, r.x1, r.y1], outline="blue", width=4)

        draw.text((r.x0, max(0, r.y0 - 25)), item["label"], fill="blue")

    debug.save(output_path)


def draw_crop_debug(image, regions, output_path):
    """Save a page image with final question crop boxes marked."""
    debug = image.copy()

    draw = ImageDraw.Draw(debug)

    for region in regions:
        left, top, right, bottom = region["box"]

        draw.rectangle([left, top, right, bottom], outline="red", width=4)

        draw.text((left + 10, top + 10), region["label"], fill="red")

    debug.save(output_path)


def save_problem_crops(image, regions, page_folder):
    """Crop and save every detected question region for one page."""
    saved_paths = {}

    for region in regions:
        label = region["label"]

        box = region["box"]

        crop = image.crop(box)

        safe_label = label.replace(".", "")

        filename = f"problem_{safe_label}.png"

        output_path = page_folder / filename

        crop.save(output_path)

        saved_paths[label] = filename

        print(f"    {label:<4} -> {filename:<20} {box}")

    return saved_paths


def main():
    """Run blank-workbook question extraction end to end."""
    dataset = []

    n_vlm_exceptions = 0

    doc = fitz.open(PDF_PATH)

    for page_index in range(len(doc)):
        pdf_page_number = page_index + 1

        page = doc[page_index]


        # Pages without both labels and SET banners are covers, dividers, or
        # back matter, so they do not become dataset rows.
        anchors = find_problem_anchors(page)

        banners = find_set_banners(page)

        if not anchors or not banners:
            continue

        assign_problems_to_sets(anchors, banners)

        print()
        print("=" * 70)
        print(f"PDF PAGE {pdf_page_number} (rotation {page.rotation})")
        print("=" * 70)


        page_folder = OUTPUT_DIR / f"page_{pdf_page_number}"

        page_folder.mkdir(parents=True, exist_ok=True)


        full_page_path = page_folder / "full_page.png"

        image = render_page(page, full_page_path)

        print(f"\nFound {len(anchors)} problem anchors.")

        print(f"Found {len(banners)} SET banners.")


        instructions = {banner["label"]: banner["instruction"] for banner in banners}

        print("\nEXTRACTED SET INSTRUCTIONS:")

        if not instructions:
            print("    NONE FOUND")

        else:
            for set_name, instruction in instructions.items():
                print(f"\n    {set_name}:")

                print(f"        {instruction}")


        draw_anchor_debug(image, anchors, banners, page_folder / "debug_anchors.png")


        regions = create_problem_regions(image, anchors, banners)


        draw_crop_debug(image, regions, page_folder / "debug_crops.png")


        print("\nQUESTION IMAGE CROPS:")

        saved_paths = save_problem_crops(image, regions, page_folder)


        print("\nDATASET ASSOCIATIONS:")

        for region in regions:
            label = region["label"]


            set_name = region["set"]


            section_instruction = instructions.get(set_name, None)


            question_image = f"page_{pdf_page_number}/{saved_paths[label]}"


            # PDF text extraction scrambles some math notation. Reading the
            # rendered crop with a VLM gives cleaner plain-text problem text.
            if extract_problem_instructions is not None:
                crop_img = image.crop(region["box"])

                try:
                    problem_instruction = extract_problem_instructions(crop_img)

                except Exception as e:
                    problem_instruction = None

                    n_vlm_exceptions += 1

                    print(f"        VLM FAILED for {label}: {e}")

            else:
                problem_instruction = None


            # This is the canonical question-level record. Answer paths are
            # filled later by answer_extraction.py after student crops exist.
            record = {
                "question_text": {

                    "section_instructions": section_instruction,

                    "problem_instructions": problem_instruction,
                },

                "question_image": question_image,

                "answer_image": None,

                "evaluation": None,


                "_debug": {
                    "pdf_page": pdf_page_number,
                    "problem_label": label,
                    "set": set_name,
                },
            }

            dataset.append(record)

            print(f"    {label} -> {set_name}")

            print(f"        Section instruction: {section_instruction}")

            print(f"        Problem instruction: {problem_instruction}")

            print(f"        Image: {question_image}")


    dataset_path = DATASET_JSON_PATH

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"\nCreated {len(dataset)} dataset records.")

    print(f"\nDataset saved to:\n    {dataset_path}")

    n_missing_instructions = sum(
        1 for r in dataset if r["question_text"]["problem_instructions"] is None
    )

    print("\nRemember:")

    if extract_problem_instructions is None:
        print("    problem_instructions = None for all records (VLM was skipped)")
    elif n_missing_instructions:


        print(
            f"    problem_instructions = None for {n_missing_instructions} records "
            f"({n_vlm_exceptions} from a real VLM/network error -- see log above; "
            f"the rest are the VLM's own correct answer for a problem with no "
            f"printed text, e.g. a graph, number line, or data table)"
        )

    print("    answer_image = None (student workbook later)")

    print("    evaluation = None (later)")

    print("\n_debug is temporary and will be removed after validation.")


if __name__ == "__main__":
    main()
