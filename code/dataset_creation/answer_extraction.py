"""
Extract student answer crops by referencing the blank workbook.

The blank workbook tells us where each question should be. Student PDFs are
scanned images, so this script first matches each scan page to the reference
page, then maps the reference crop boxes onto the scan and saves nonblank work.
"""


import base64
import glob
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import numpy as np
import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image

from workbook_layout import (
    find_problem_anchors,
    find_set_banners,
    find_header_topic,
    assign_problems_to_sets,
    create_problem_regions,
    render_page,
)

load_dotenv(override=True)


BLANK_PDF_PATH = Path(
    "../../data/algebra_dataset/blank_workbook/abePROBLEMworkbook.pdf"
)
STUDENT_WORKBOOK_DIR = Path(
    "../../data/algebra_dataset/student_workbook/PROBLEMworkbooks"
)
DATASET_JSON_PATH = Path("../../data/algebra_dataset/dataset.json")
ANSWER_CROPS_DIR = Path("../../data/algebra_dataset/answer_crops")


PROBLEM_CROPS_DIR = Path("../../data/algebra_dataset/problem_crops")


UNLOCATED_PAGES_DIR = Path("../../data/algebra_dataset/answer_crops_unlocated")
UNLOCATED_MANIFEST_PATH = UNLOCATED_PAGES_DIR / "manifest.json"


CHECKPOINT_PATH = ANSWER_CROPS_DIR.parent / "answer_extraction_checkpoint.txt"


PAGE_REVIEW_APP_URL = "http://127.0.0.1:7862"


ANSWER_CROPS_DIR.mkdir(parents=True, exist_ok=True)

UNLOCATED_PAGES_DIR.mkdir(parents=True, exist_ok=True)


OFFSET_SEARCH_RANGE = 20
PER_PAGE_CORRECTION_RANGE = 4


BANNER_MIN_WIDTH_FRACTION = 0.6
BANNER_MIN_HEIGHT_PX = 15
BANNER_ROW_DARK_THRESHOLD = 180


BANNER_MERGE_GAP_PX = 15


BANNER_CLEARANCE_PX = 8


DIVIDER_MIN_HEIGHT_PX = 2
DIVIDER_MAX_HEIGHT_PX = 8


VLM_ENDPOINT_URL = "https://vllm.vidarlab.net"
VLM_MODEL = "qwen3-vl-32b"
VLM_API_KEY = os.environ.get("VIDAR_LITELLM_KEY")


VLM_SEARCH_RANGE = 15


BLANK_CHECK_MAX_WORKERS = 8


def compute_blank_regions(blank_doc):
    """Detect and cache every usable question region in the blank workbook."""
    blank_pages = {}

    for page_index in range(len(blank_doc)):
        page = blank_doc[page_index]

        pdf_page_number = page_index + 1

        anchors = find_problem_anchors(page)

        banners = find_set_banners(page)

        # Non-content pages do not have both SET banners and problem labels.
        if not anchors or not banners:
            continue

        assign_problems_to_sets(anchors, banners)

        image = render_page(page)

        regions = create_problem_regions(image, anchors, banners)


        _, header_topic = find_header_topic(page)

        sorted_banners = sorted(banners, key=lambda b: b["rect"].y0)

        banner_bands = [
            (int(b["rect"].y0), int(b["rect"].y1)) for b in sorted_banners
        ]

        blank_pages[pdf_page_number] = {
            "regions": regions,
            "banners": sorted_banners,
            "page_size": image.size,
            "header_topic": header_topic,


            # Thin dividers help split stacked single-column problems.
            "dividers": find_thin_dividers(image, banner_bands),
        }

    return blank_pages


def _levenshtein(a, b):
    """Compute edit distance."""
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        cur = [i]

        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))

        prev = cur

    return prev[-1]


def topic_matches(header_text, expected_topic_text):
    """Fuzzy-match scanned OCR topic text to the blank workbook topic."""
    if not header_text or not expected_topic_text:
        return False

    # `more` distinguishes continuation pages that otherwise share topic words.
    if ("more" in expected_topic_text) != ("more" in header_text):
        return False


    topic = re.sub(r"\b(serp|algebra|more)\b", "", expected_topic_text)


    topic_words = [
        w for w in topic.split() if len(w) > 2 and any(c.isalpha() for c in w)
    ]

    if not topic_words:
        return True

    header_words = header_text.split()

    for word in topic_words:
        # OCR is noisy, so allow a small edit distance for each topic word.
        max_dist = 1 if len(word) <= 6 else 2

        if not any(_levenshtein(word, hw) <= max_dist for hw in header_words):
            return False

    return True


def search_order(max_range):
    """Generate offsets in nearest-first order: 0, -1, +1, -2, +2, ..."""
    yield 0

    for d in range(1, max_range + 1):
        yield -d

        yield d


def header_topic_text(page):
    """OCR only the top-right topic area of a rendered scan page."""
    w, h = page.size

    banner_bands = find_banner_bands(page)

    # Stop before the first SET banner so worksheet body text does not pollute
    # the topic OCR.
    bottom = min(b[0] for b in banner_bands) - 4 if banner_bands else int(h * 0.20)

    crop = page.crop((int(w * 0.40), 0, w, max(1, bottom)))

    return pytesseract.image_to_string(crop, config="--psm 6").strip().lower()


def render_page_if_layout_matches(page, blank_page):
    """Render a scan page and return it only if topic and banner count match."""
    base = render_page(page)

    expected_banner_count = len(blank_page["banners"])

    # Scans may be sideways or upside-down, so test all four orientations.
    for rotation in (0, 90, 180, 270):
        img = base.rotate(-rotation, expand=True) if rotation else base

        if not topic_matches(header_topic_text(img), blank_page["header_topic"]):
            continue

        banner_bands = find_banner_bands(img)

        if len(banner_bands) != expected_banner_count:
            continue

        return img

    return None


def locate_student_page_offset(student_doc, blank_pages, anchor_page=None):
    """Find the global page-number offset between scan and blank workbook."""
    if anchor_page is None:
        anchor_page = min(blank_pages)

    expected_index = anchor_page - 1

    blank_page = blank_pages[anchor_page]

    for offset in search_order(OFFSET_SEARCH_RANGE):
        idx = expected_index + offset

        if idx < 0 or idx >= len(student_doc):
            continue

        if render_page_if_layout_matches(student_doc[idx], blank_page) is not None:
            return offset

    return None


def verify_or_relocate_page(
    student_doc, blank_pages, pdf_page_number, offset, min_index=None
):
    """Find a specific reference page in one student scan."""
    blank_page = blank_pages[pdf_page_number]

    base_index = (pdf_page_number - 1) + offset

    for delta in search_order(PER_PAGE_CORRECTION_RANGE):
        idx = base_index + delta

        if idx < 0 or idx >= len(student_doc):
            continue

        # Pages should move forward through the scan. This prevents two
        # reference pages from claiming the same scanned page.
        if min_index is not None and idx <= min_index:
            continue

        image = render_page_if_layout_matches(student_doc[idx], blank_page)

        if image is not None:
            return idx, image

    return None


def _vlm_questions_for_page(pdf_page_number, blank_page, dataset_by_key):
    """Build the reference question list used by the VLM page fallback."""
    questions = []

    for region in blank_page["regions"]:
        record = dataset_by_key.get((pdf_page_number, region["label"]), {})

        text = record.get("question_text", {}).get("problem_instructions")

        questions.append({"label": region["label"], "text": text})

    return questions


def vlm_locate_boxes(
    image, questions, endpoint_url=VLM_ENDPOINT_URL, model=VLM_MODEL,
    api_key=VLM_API_KEY, timeout=60,
):
    """Ask the VLM to locate question boxes on a hard-to-match scan page."""
    lines = [
        f"{q['label']} {q['text']}" if q["text"] else f"{q['label']} (a graph, "
        "table, or number line -- no printed text)"
        for q in questions
    ]

    prompt = (
        "This is a scanned page from a math workbook. It should contain "
        "the following problems, each printed on the page with blank "
        "space underneath for a student's handwritten answer:\n\n"
        + "\n".join(lines)
        + "\n\nA label alone is NOT enough evidence that a problem is "
        "present: this workbook reuses the same labels ('1a.', '2b.', "
        "etc.) on every page, so a page with a completely different set "
        "of problems will still have something printed next to '1a.'. "
        "Before including a label, read the actual printed mathematical "
        "expression at that position and confirm it genuinely matches "
        "the reference text given above for that exact label (matching "
        "content -- the same numbers and operators -- not just "
        "matching position or label). If what's printed there is a "
        "different problem, leave that label out entirely.\n\n"
        "Do not mentally rotate the image to read it. If the page is "
        "sideways or upside-down, or you cannot confidently read a "
        "problem in its normal reading orientation, treat every label "
        "as not found on this image.\n\n"
        "For each label you're confident genuinely matches, respond "
        "with a bounding box tightly enclosing the entire area from "
        "just below its own printed problem text down through all of "
        "the blank workspace underneath it -- stop before the next "
        "problem's label, the next SET banner, or the page bottom, "
        "whichever comes first.\n\n"
        "Coordinates are integers from 0 to 1000, where (0,0) is this "
        "image's top-left corner and (1000,1000) is its bottom-right "
        "corner, regardless of its real pixel dimensions.\n\n"
        "Respond with ONLY a JSON object, no other text, mapping each "
        'confirmed label to a [left, top, right, bottom] list, e.g. '
        '{"1a.": [520, 100, 1000, 480]}. If none of the listed problems '
        "genuinely appear on this image (e.g. it's the wrong page, or "
        "the image isn't right-side-up), respond with exactly: {}"
    )

    width, height = image.size

    buf = io.BytesIO()

    image.save(buf, format="PNG")

    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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


    # Models sometimes wrap JSON in markdown fences even when asked not to.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        raw_boxes = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}

    valid_labels = {q["label"] for q in questions}

    boxes = {}

    for label, box in raw_boxes.items():
        if label not in valid_labels:
            continue

        if not (isinstance(box, list) and len(box) == 4):
            continue

        left, top, right, bottom = box

        if not (right > left and bottom > top):
            continue

        # Qwen returns 0-1000 normalized coordinates; convert them back to
        # real pixel coordinates before cropping.
        boxes[label] = (
            left / 1000 * width,
            top / 1000 * height,
            right / 1000 * width,
            bottom / 1000 * height,
        )

    return boxes


def vlm_locate_and_crop_page(
    student_doc, pdf_page_number, blank_page, dataset_by_key, offset, min_index
):
    """Use a wider VLM search when deterministic page matching fails."""
    questions = _vlm_questions_for_page(pdf_page_number, blank_page, dataset_by_key)


    # If text exists for a question, require the VLM to find it. This avoids
    # accepting a page just because it has reused labels like `1a.`.
    required_labels = {q["label"] for q in questions if q["text"] is not None}

    base_index = (pdf_page_number - 1) + offset

    for delta in search_order(VLM_SEARCH_RANGE):
        idx = base_index + delta

        if idx < 0 or idx >= len(student_doc):
            continue

        if min_index is not None and idx <= min_index:
            continue

        base = render_page(student_doc[idx])

        for rotation in (0, 90, 180, 270):
            img = base.rotate(-rotation, expand=True) if rotation else base

            banner_bands = find_banner_bands(img)

            if not banner_bands:
                continue

            try:
                vlm_boxes = vlm_locate_boxes(img, questions)
            except Exception as e:
                print(
                    f"        page {pdf_page_number}: VLM fallback error "
                    f"at scan index {idx} rotation {rotation}: {e}"
                )
                continue

            if not vlm_boxes:
                continue

            if required_labels and not required_labels.issubset(vlm_boxes):
                continue


            # If banners line up after the VLM identifies the page, prefer the
            # deterministic crop boxes because they are more consistent.
            if len(banner_bands) == len(blank_page["banners"]):
                boxes = deterministic_boxes_for_page(blank_page, img, banner_bands)


                if all(b[3] > b[1] and b[2] > b[0] for b in boxes.values()):
                    return idx, img, boxes

                continue

            return idx, img, vlm_boxes

    return None


def find_dark_bands(row_means, thresh, min_height):
    """Find continuous dark row ranges in a grayscale page image."""
    dark = row_means < thresh

    bands = []

    start = None

    for i, is_dark in enumerate(dark):
        if is_dark and start is None:
            start = i

        elif not is_dark and start is not None:
            if i - start >= min_height:
                bands.append((start, i))

            start = None

    if start is not None and len(dark) - start >= min_height:
        bands.append((start, len(dark)))

    return bands


def merge_close_bands(bands, max_gap):
    """Merge bands split by tiny white gaps from scan noise."""
    if not bands:
        return bands

    merged = [bands[0]]

    for start, end in bands[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)

        else:
            merged.append((start, end))

    return merged


def find_banner_bands(image):
    """Find SET banner bars on a scanned student page."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    row_means = gray.mean(axis=1)

    bands = find_dark_bands(
        row_means, thresh=BANNER_ROW_DARK_THRESHOLD, min_height=BANNER_MIN_HEIGHT_PX
    )

    bands = merge_close_bands(bands, BANNER_MERGE_GAP_PX)

    # SET banners are dark and span most of the page width. This rejects
    # small marks or local handwriting that happen to be dark.
    banner_bands = [
        (s, e)
        for s, e in bands
        if (gray[s:e, :].mean(axis=0) < BANNER_ROW_DARK_THRESHOLD).mean()
        > BANNER_MIN_WIDTH_FRACTION
    ]

    banner_bands.sort()

    return [(int(s), int(e)) for s, e in banner_bands]


def find_thin_dividers(image, banner_bands):
    """Find thin horizontal rules that split stacked problems."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    row_means = gray.mean(axis=1)

    height = len(row_means)


    # Ignore rows close to SET banners; their edges look like divider lines.
    trailing_edge_buffer = 20

    leading_rule_buffer = 20

    gap_starts = [0] + [b[1] + trailing_edge_buffer for b in banner_bands]

    gap_ends = [b[0] - leading_rule_buffer for b in banner_bands] + [height]

    dividers = []

    for lo, hi in zip(gap_starts, gap_ends):
        if hi <= lo:
            continue

        bands = find_dark_bands(
            row_means[lo:hi], thresh=BANNER_ROW_DARK_THRESHOLD, min_height=DIVIDER_MIN_HEIGHT_PX
        )

        bands = merge_close_bands(bands, BANNER_MERGE_GAP_PX)

        for s, e in bands:
            s, e = s + lo, e + lo

            if (e - s) > DIVIDER_MAX_HEIGHT_PX:
                continue

            width_frac = (gray[s:e, :].mean(axis=0) < BANNER_ROW_DARK_THRESHOLD).mean()

            if width_frac > BANNER_MIN_WIDTH_FRACTION:
                dividers.append((s + e) // 2)

    return dividers


def build_y_mapping(
    blank_banner_bottoms, blank_height, student_banner_bottoms, student_height
):
    """Build a piecewise vertical mapping from blank page to student scan."""
    blank_points = [0.0] + list(blank_banner_bottoms) + [float(blank_height)]

    student_points = [0.0] + list(student_banner_bottoms) + [float(student_height)]

    def map_y(y):
        """Map y."""
        y = min(max(y, blank_points[0]), blank_points[-1])

        for i in range(len(blank_points) - 1):
            b0, b1 = blank_points[i], blank_points[i + 1]

            if y <= b1 or i == len(blank_points) - 2:
                span = b1 - b0

                # Map each section independently so local scan stretch/skew
                # does not drift across the full page height.
                frac = (y - b0) / span if span > 0 else 0.0

                s0, s1 = student_points[i], student_points[i + 1]

                return s0 + frac * (s1 - s0)

        return student_points[-1]

    return map_y


def map_blank_box_to_student(box, map_y, x_scale):
    """Map one reference crop box onto the student scan."""
    left, top, right, bottom = box

    return (left * x_scale, map_y(top), right * x_scale, map_y(bottom))


def deterministic_boxes_for_page(blank_page, student_image, banner_bands):
    """Create answer crop boxes from reference regions and scan banners."""
    blank_w, blank_h = blank_page["page_size"]

    student_w, student_h = student_image.size

    # Horizontal drift is handled with a simple scale; vertical drift needs the
    # banner-anchored piecewise mapping below.
    x_scale = student_w / blank_w


    banner_index_by_label = {
        b["label"]: i for i, b in enumerate(blank_page["banners"])
    }

    blank_anchors = [b["rect"].y1 for b in blank_page["banners"]]

    student_anchors = [band[1] for band in banner_bands]


    if blank_page["dividers"]:
        student_dividers = find_thin_dividers(student_image, banner_bands)

        if len(student_dividers) == len(blank_page["dividers"]):
            combined = sorted(
                zip(
                    blank_anchors + blank_page["dividers"],
                    student_anchors + student_dividers,
                )
            )

            blank_anchors = [c[0] for c in combined]

            student_anchors = [c[1] for c in combined]

    map_y = build_y_mapping(blank_anchors, blank_h, student_anchors, student_h)

    boxes = {}

    for region in blank_page["regions"]:
        box = map_blank_box_to_student(region["box"], map_y, x_scale)


        banner_i = banner_index_by_label.get(region["set"])

        if banner_i is not None:
            # Keep crops clear of SET banners so printed instructions do not
            # bleed into answer images.
            min_top = banner_bands[banner_i][1] + BANNER_CLEARANCE_PX

            left, top, right, bottom = box

            top = max(top, min_top)

            if banner_i + 1 < len(banner_bands):
                max_bottom = banner_bands[banner_i + 1][0] - BANNER_CLEARANCE_PX

                bottom = min(bottom, max_bottom)

            box = (left, top, right, bottom)

        boxes[region["label"]] = box

    return boxes


def extract_student_answers(student_doc, blank_pages, offset, dataset_by_key):
    """Extract answer crops for every reference page in one student workbook."""
    answers = {}

    failed_pages = []

    vlm_fallback_pages = []

    last_claimed_idx = None

    for pdf_page_number, blank_page in blank_pages.items():
        located = verify_or_relocate_page(
            student_doc, blank_pages, pdf_page_number, offset, last_claimed_idx
        )

        student_image = None

        banner_bands = None

        if located is not None:
            idx, candidate_image = located

            candidate_banner_bands = find_banner_bands(candidate_image)

            if len(candidate_banner_bands) != len(blank_page["banners"]):
                print(
                    f"        page {pdf_page_number}: expected "
                    f"{len(blank_page['banners'])} SET banners, found "
                    f"{len(candidate_banner_bands)}, trying VLM fallback"
                )

            else:
                student_image = candidate_image

                banner_bands = candidate_banner_bands

                last_claimed_idx = idx

        if student_image is None:
            # Fall back only after topic/header matching fails or the banner
            # count is wrong. VLM crops are slower and less deterministic.
            vlm_result = vlm_locate_and_crop_page(
                student_doc, pdf_page_number, blank_page, dataset_by_key,
                offset, last_claimed_idx,
            )

            if vlm_result is None:
                print(
                    f"        page {pdf_page_number}: could not locate in "
                    "scan even with VLM fallback, skipped"
                )

                failed_pages.append(pdf_page_number)

                continue

            idx, vlm_image, vlm_boxes = vlm_result

            last_claimed_idx = idx

            vlm_fallback_pages.append(pdf_page_number)

            found_labels = set()

            for region in blank_page["regions"]:
                box = vlm_boxes.get(region["label"])

                if box is not None:
                    answers[(pdf_page_number, region["label"])] = vlm_image.crop(box)

                    found_labels.add(region["label"])

            missing = [
                r["label"] for r in blank_page["regions"] if r["label"] not in found_labels
            ]

            print(
                f"        page {pdf_page_number}: VLM fallback found "
                f"{len(found_labels)}/{len(blank_page['regions'])} question(s)"
                + (f", missing {missing}" if missing else "")
            )

            continue

        boxes = deterministic_boxes_for_page(blank_page, student_image, banner_bands)

        for region in blank_page["regions"]:
            crop = student_image.crop(boxes[region["label"]])

            answers[(pdf_page_number, region["label"])] = crop

    return answers, failed_pages, vlm_fallback_pages


def vlm_check_answered_page(
    pdf_page_number, label_to_crop, endpoint_url=VLM_ENDPOINT_URL,
    model=VLM_MODEL, api_key=VLM_API_KEY, timeout=60,
):
    """Compare blank and student crops and ask which ones contain work."""
    content = [
        {
            "type": "text",
            "text": (
                "Below are one or more problems from a math worksheet. For "
                "each one, you're shown the BLANK version (the printed "
                "problem plus empty workspace, no student work) and then "
                "a specific student's version of that same region.\n\n"
                "For each labeled problem, decide whether the student has "
                "written anything in the workspace that ISN'T already "
                "present in the blank version -- any mark, number, "
                "symbol, or work at all counts, even a single digit or a "
                "crossed-out attempt. Printed content that's already in "
                "the blank version (a graph, a table, grid lines) is NOT "
                "evidence of an answer by itself -- only judge what the "
                "student actually added.\n\n"
                'Respond with ONLY a JSON object mapping each label to '
                'true (the student wrote something) or false (genuinely '
                'blank -- nothing added beyond the printed template), '
                'e.g. {"1a.": true, "1b.": false}.'
            ),
        }
    ]

    labels_included = []

    for label, student_crop in label_to_crop.items():
        blank_path = (
            PROBLEM_CROPS_DIR / f"page_{pdf_page_number}" / f"problem_{label.replace('.', '')}.png"
        )

        if not blank_path.is_file():
            continue

        labels_included.append(label)

        content.append({"type": "text", "text": f"--- {label} ---\nBlank version:"})

        content.append(_image_content(Image.open(blank_path)))

        content.append({"type": "text", "text": "Student's version:"})

        content.append(_image_content(student_crop))

    if not labels_included:
        return {}

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": content}],
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

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}

    return {
        label: bool(raw[label]) for label in labels_included if label in raw
    }


def _image_content(image):
    """Encode a PIL image as an OpenAI-compatible image_url object."""
    buf = io.BytesIO()

    image.save(buf, format="PNG")

    b64 = base64.b64encode(buf.getvalue()).decode()

    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def filter_blank_answers(answers):
    """Drop answer crops that match the blank reference crop."""
    by_page = {}

    for (pdf_page_number, label), crop in answers.items():
        by_page.setdefault(pdf_page_number, {})[label] = crop

    filtered = {}

    n_dropped = 0

    def check_one_page(pdf_page_number, label_to_crop):
        """Check one page."""
        try:
            return pdf_page_number, vlm_check_answered_page(pdf_page_number, label_to_crop)
        except Exception as e:
            print(f"        page {pdf_page_number}: blank-check error, keeping all: {e}")
            return pdf_page_number, {}

    with ThreadPoolExecutor(max_workers=BLANK_CHECK_MAX_WORKERS) as pool:
        futures = [
            pool.submit(check_one_page, pdf_page_number, label_to_crop)
            for pdf_page_number, label_to_crop in by_page.items()
        ]

        for future in as_completed(futures):
            pdf_page_number, answered_by_label = future.result()

            for label, crop in by_page[pdf_page_number].items():


                # If the VLM omits a label or errors, keep the crop. False
                # positives are easier to review than deleted real work.
                is_answered = answered_by_label.get(label, True)

                if is_answered:
                    filtered[(pdf_page_number, label)] = crop

                else:
                    n_dropped += 1

    return filtered, n_dropped


def render_unlocated_page_snapshot(student_doc, pdf_page_number, offset):
    """Render the expected scan page for manual inspection when matching fails."""
    idx = (pdf_page_number - 1) + offset

    if idx < 0 or idx >= len(student_doc):
        return None

    return render_page(student_doc[idx])


def main():
    """Run student answer extraction for all PDFs in the workbook folder."""
    if VLM_API_KEY:
        print("VIDAR_LITELLM_KEY is set -- VLM fallback is active.", flush=True)

    else:
        print(
            "WARNING: VIDAR_LITELLM_KEY is not set -- the VLM fallback will "
            "fail on every page it's needed for (any page verify_or_relocate_page "
            "can't place on its own). Those pages will end up in failed_pages "
            "instead of being rescued.\n"
            '    export VIDAR_LITELLM_KEY="sk-..." to enable it.',
            flush=True,
        )

    print("Detecting blank workbook regions...", flush=True)

    blank_doc = fitz.open(BLANK_PDF_PATH)

    blank_pages = compute_blank_regions(blank_doc)

    print(
        f"    {sum(len(p['regions']) for p in blank_pages.values())} questions across {len(blank_pages)} pages.",
        flush=True,
    )

    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_by_key = {
        (record["_debug"]["pdf_page"], record["_debug"]["problem_label"]): record
        for record in dataset
    }

    student_paths = sorted(glob.glob(str(STUDENT_WORKBOOK_DIR / "*.pdf")))

    total_students = len(student_paths)

    print(f"\nFound {total_students} student workbooks.\n", flush=True)


    # Checkpoint by student ID so long batch runs can resume without redoing
    # every previously processed workbook.
    if CHECKPOINT_PATH.exists():
        done_ids = set(CHECKPOINT_PATH.read_text().splitlines())

    else:
        done_ids = set()

    if done_ids:
        print(
            f"Resuming: {len(done_ids)} student(s) already done, skipping them.\n",
            flush=True,
        )

    checkpoint_file = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    n_students_processed = 0

    n_students_skipped = 0

    n_students_already_done = 0


    run_start = time.monotonic()

    def finish(student_id, outcome):
        """Checkpoint one student and print progress."""
        checkpoint_file.write(student_id + "\n")

        checkpoint_file.flush()

        n_done = n_students_already_done + n_students_processed + n_students_skipped

        n_this_run = n_students_processed + n_students_skipped

        remaining = total_students - n_done

        eta = ""

        if n_this_run > 0 and remaining > 0:
            avg_s = (time.monotonic() - run_start) / n_this_run

            eta_s = avg_s * remaining

            eta = f"  ({avg_s:.0f}s/student avg, ~{eta_s / 60:.0f} min left)"

        print(
            f"    -> [{n_done}/{total_students}] {student_id}: {outcome}{eta}",
            flush=True,
        )

        print(
            f"       See the new crops (page by page, all students): "
            f"{PAGE_REVIEW_APP_URL}  (run code/review_apps/page_review_app.py "
            "if it isn't already; hit its Refresh button to pick this up)",
            flush=True,
        )

    for student_path in student_paths:
        student_id = Path(student_path).stem

        if student_id in done_ids:
            n_students_already_done += 1

            continue

        print(
            f"[{n_students_already_done + n_students_processed + n_students_skipped + 1}/{total_students}] {student_id}",
            flush=True,
        )

        try:
            student_doc = fitz.open(student_path)

        except Exception as e:
            print(f"    FAILED to open: {e}", flush=True)

            n_students_skipped += 1

            finish(student_id, "skipped (failed to open)")

            continue

        offset = locate_student_page_offset(student_doc, blank_pages)

        if offset is None:
            print(
                "    could not locate page layout in this scan, skipping student",
                flush=True,
            )

            n_students_skipped += 1

            finish(student_id, "skipped (unlocatable)")

            continue

        print(f"    page offset: {offset:+d}", flush=True)

        try:
            answers, failed_pages, vlm_fallback_pages = extract_student_answers(
                student_doc, blank_pages, offset, dataset_by_key
            )

        except Exception as e:
            print(f"    FAILED: {e}", flush=True)

            n_students_skipped += 1

            finish(student_id, "skipped (extraction error)")

            continue

        n_located = len(answers)

        answers, n_blank = filter_blank_answers(answers)

        for (pdf_page_number, label), crop in answers.items():
            out_dir = (
                ANSWER_CROPS_DIR
                / f"page_{pdf_page_number}"
                / f"problem_{label.replace('.', '')}"
            )

            out_dir.mkdir(parents=True, exist_ok=True)

            crop.convert("RGB").save(out_dir / f"{student_id}.png")


        # Save a snapshot for failed pages so the review app can show what was
        # at the expected scan location.
        for pdf_page_number in failed_pages:
            snapshot = render_unlocated_page_snapshot(
                student_doc, pdf_page_number, offset
            )

            if snapshot is None:
                continue

            out_dir = UNLOCATED_PAGES_DIR / f"page_{pdf_page_number}"

            out_dir.mkdir(parents=True, exist_ok=True)

            snapshot.convert("RGB").save(out_dir / f"{student_id}.png")

        if failed_pages or vlm_fallback_pages:
            if UNLOCATED_MANIFEST_PATH.exists():
                unlocated_manifest = json.loads(UNLOCATED_MANIFEST_PATH.read_text())

            else:
                unlocated_manifest = {}

            entry = unlocated_manifest.setdefault(student_id, {})


            # Older manifests stored a plain page list; normalize that shape
            # before adding new failed/fallback details.
            if isinstance(entry, list):
                entry = {"failed": entry}

            if failed_pages:
                entry["failed"] = failed_pages

            if vlm_fallback_pages:
                entry["vlm_fallback"] = vlm_fallback_pages

            unlocated_manifest[student_id] = entry

            UNLOCATED_MANIFEST_PATH.write_text(
                json.dumps(unlocated_manifest, indent=2)
            )

        print(
            f"    {n_located} question(s) located, {n_blank} judged blank "
            f"and dropped, {len(answers)} saved "
            f"({len(vlm_fallback_pages)} page(s) via VLM fallback), "
            f"{len(failed_pages)} page(s) could not be located",
            flush=True,
        )

        n_students_processed += 1

        finish(student_id, f"done, {len(answers)} answered")

    checkpoint_file.close()


    print()
    print("=" * 70)
    print("ATTACHING answer_image TO DATASET")
    print("=" * 70)

    n_with_answers = 0

    for key, record in dataset_by_key.items():
        pdf_page_number, label = key

        safe_label = label.replace(".", "")

        out_dir = ANSWER_CROPS_DIR / f"page_{pdf_page_number}" / f"problem_{safe_label}"

        answer_files = sorted(out_dir.glob("*.png")) if out_dir.is_dir() else []


        # Rebuild from disk instead of trusting only this run's memory. That
        # keeps interrupted/resumed runs consistent.
        record["answer_image"] = [
            f"answer_crops/page_{pdf_page_number}/problem_{safe_label}/{p.name}"
            for p in answer_files
        ]

        if answer_files:
            n_with_answers += 1

            print(
                f"    {label} (page {pdf_page_number}): {len(answer_files)} answer(s)"
            )

        else:
            print(f"    {label} (page {pdf_page_number}): 0 answers found")

    with open(DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"\nStudents processed this run: {n_students_processed}")

    print(f"Students already done (resumed past): {n_students_already_done}")

    print(f"Students skipped (unlocatable/failed): {n_students_skipped}")

    print(
        f"\nQuestions with at least one answer: {n_with_answers} / {len(dataset_by_key)}"
    )

    print(f"\nDataset updated in place:\n    {DATASET_JSON_PATH}")


if __name__ == "__main__":
    main()
