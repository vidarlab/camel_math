"""
Shared blank-workbook layout detection.

This module reads the reference workbook PDF text layer to find SET banners,
problem labels, and the crop rectangle for each question. Student scans do not
have reliable text, so answer extraction reuses these reference boxes.
"""


import io
import re
import unicodedata

import fitz
import numpy as np
from PIL import Image


ZOOM = 2  # Render at 2x so text and crop boundaries keep enough detail.


PROBLEM_LABEL_RE = re.compile(r"^\d+[a-zA-Z]\.$")


SET_LABEL_RE = re.compile(r"^SET$", re.IGNORECASE)


def render_page(page, output_path=None):
    """Render one PDF page as a PIL image."""
    matrix = fitz.Matrix(ZOOM, ZOOM)

    pix = page.get_pixmap(matrix=matrix)

    if output_path is not None:
        pix.save(output_path)

        return Image.open(output_path)

    return Image.open(io.BytesIO(pix.tobytes("png")))


def pdf_rect_to_display_rect(page, rect):
    """Convert PDF-space rectangles into rendered-image coordinates."""
    # PyMuPDF text locations are in PDF coordinates. PIL crops use rendered
    # image pixels, so every rectangle goes through rotation and zoom here.
    display_rect = rect * page.rotation_matrix


    return fitz.Rect(
        display_rect.x0 * ZOOM,
        display_rect.y0 * ZOOM,
        display_rect.x1 * ZOOM,
        display_rect.y1 * ZOOM,
    )


def get_text_blocks(page):
    """Group PDF words into block -> line -> word order."""
    grouped = {}

    for x0, y0, x1, y1, text, block_no, line_no, word_no in page.get_text("words"):
        rect = fitz.Rect(x0, y0, x1, y1)

        line = grouped.setdefault(block_no, {}).setdefault(line_no, [])

        line.append((word_no, rect, text))

    blocks = []

    for block_no in sorted(grouped):
        lines = []

        for line_no in sorted(grouped[block_no]):
            words = sorted(grouped[block_no][line_no], key=lambda w: w[0])

            lines.append([(rect, text) for _, rect, text in words])

        blocks.append(lines)

    return blocks


_BARE_NUMBER_LABEL_RE = re.compile(r"^(\d+)\.$")


def _recover_bare_number_anchors(page, anchors, banners):
    """Recover pages that print `1.` instead of explicit `1a.`/`1b.` labels."""
    banner_numbers = {
        m.group()
        for b in banners
        if (m := _SET_NUMBER_RE.search(b["label"]))
    }


    full_display_rect = pdf_rect_to_display_rect(page, page.rect)

    mid_x = (full_display_rect.x0 + full_display_rect.x1) / 2

    for block in get_text_blocks(page):
        for line in block:
            label_rect, label = line[0]

            m = _BARE_NUMBER_LABEL_RE.match(label)

            if not m or m.group(1) not in banner_numbers:
                continue

            n_existing = sum(
                1 for a in anchors if _SET_NUMBER_RE.match(a["label"]).group() == m.group(1)
            )

            if n_existing >= 2:
                continue

            display_rect = pdf_rect_to_display_rect(page, label_rect)

            # Bare labels are split by page column: left column is `a`, right is `b`.
            letter = "a" if display_rect.x0 < mid_x else "b"

            anchors.append({"label": f"{m.group(1)}{letter}.", "rect": display_rect})


def find_problem_anchors(page):
    """Find printed problem labels and their image-space rectangles."""
    anchors = []

    for block in get_text_blocks(page):
        for line in block:
            label_rect, label = line[0]

            if not PROBLEM_LABEL_RE.match(label):
                continue

            anchors.append(
                {"label": label, "rect": pdf_rect_to_display_rect(page, label_rect)}
            )

    _recover_bare_number_anchors(page, anchors, find_set_banners(page))

    return anchors


def find_set_banners(page):
    """Find SET banners and the instruction text printed beside them."""
    banners = []

    for block in get_text_blocks(page):
        first_line = block[0]
        _, first_word = first_line[0]

        if not SET_LABEL_RE.match(first_word) or len(first_line) < 2:
            continue

        set_number_rect, set_number = first_line[1]

        label = f"SET {set_number}"

        banner_rect = first_line[0][0] | set_number_rect

        instruction = " ".join(text for _, text in first_line[2:]).strip()

        banners.append(
            {
                "label": label,
                "rect": pdf_rect_to_display_rect(page, banner_rect),
                "instruction": instruction,
            }
        )

    return banners


_ID_ROW_MARKERS = ("id code", "date:", "teacher:", "section:", "not your name")


def find_header_topic(page, top_limit_display=300):
    """Extract the top-right topic header used to match scanned pages."""
    page_width = page.rect.width if page.rotation in (0, 180) else page.rect.height

    display_width = page_width * ZOOM

    candidates = []

    for block in get_text_blocks(page):
        for line in block:
            text = " ".join(t for _, t in line)

            if not text.strip():
                continue

            if any(marker in text.lower() for marker in _ID_ROW_MARKERS):
                continue

            rect = line[0][0]

            for r, _ in line[1:]:
                rect = rect | r

            display_rect = pdf_rect_to_display_rect(page, rect)

            # The student ID/date row is also near the top. The topic header is
            # consistently on the right side, so this filters out the ID area.
            if display_rect.y1 <= top_limit_display and display_rect.x0 > display_width * 0.35:
                candidates.append((display_rect, text))

    if not candidates:
        return None, ""

    candidates.sort(key=lambda c: c[0].y0)

    header = [candidates[0]]

    for display_rect, text in candidates[1:]:
        if display_rect.y0 - header[-1][0].y1 < 20:
            header.append((display_rect, text))

        else:
            break

    left = min(d.x0 for d, _ in header)


    joined = unicodedata.normalize("NFKD", "\n".join(t for _, t in header).lower())

    return left, joined


_LABEL_LETTER_RE = re.compile(r"[a-zA-Z]")

_SET_NUMBER_RE = re.compile(r"\d+")


def assign_problems_to_sets(anchors, banners):
    """Attach each problem to the nearest SET banner above it."""
    for anchor in anchors:
        anchor_y = anchor["rect"].y0

        candidates = [b for b in banners if b["rect"].y0 <= anchor_y]

        nearest = max(candidates, key=lambda b: b["rect"].y0, default=None)

        anchor["set"] = nearest["label"] if nearest else None

        if nearest is None:
            continue

        set_number_match = _SET_NUMBER_RE.search(nearest["label"])

        letter_match = _LABEL_LETTER_RE.search(anchor["label"])

        if not set_number_match or not letter_match:
            continue

        # SET number plus label letter gives a stable label like `3b.` even
        # when the printed label text was recovered imperfectly.
        corrected_label = f"{set_number_match.group()}{letter_match.group()}."

        if corrected_label != anchor["label"]:
            anchor["label"] = corrected_label


PRINT_ROW_INK_FRACTION = 0.01
PRINT_ROW_DARK_THRESHOLD = 200


PRINT_TEXT_LINE_GAP_PX = 25


PRINT_TEXT_SEARCH_HEIGHT_PX = 400


def find_print_text_bottom(image, box):
    """Estimate where printed problem text ends inside a question crop."""
    left, top, right, bottom = box

    search_bottom = min(bottom, top + PRINT_TEXT_SEARCH_HEIGHT_PX)

    crop = image.crop((left, top, right, search_bottom))

    arr = np.asarray(crop.convert("L"), dtype=np.float32)

    # Work row-by-row: printed text creates dense dark bands near the top.
    row_dark_fraction = (arr < PRINT_ROW_DARK_THRESHOLD).mean(axis=1)

    bands = _bands_above_fraction(row_dark_fraction, PRINT_ROW_INK_FRACTION)

    if not bands:
        return top


    merged_end = bands[0][1]

    for start, end in bands[1:]:
        if start - merged_end < PRINT_TEXT_LINE_GAP_PX:
            merged_end = end

        else:
            break

    return top + merged_end


def _bands_above_fraction(row_fraction, min_fraction):
    """Find continuous row ranges whose dark-pixel fraction is high enough."""
    has_ink = row_fraction > min_fraction

    bands = []

    start = None

    for i, ink in enumerate(has_ink):
        if ink and start is None:
            start = i

        elif not ink and start is not None:
            bands.append((start, i))

            start = None

    if start is not None:
        bands.append((start, len(has_ink)))

    return bands


def create_problem_regions(image, anchors, banners):
    """Create final crop boxes from labels, columns, and next-stop boundaries."""
    width, height = image.size

    mid_x = width / 2

    # Pages are either single-column or split into left/right question columns.
    two_column = any(a["rect"].x0 < mid_x for a in anchors) and any(
        a["rect"].x0 >= mid_x for a in anchors
    )


    banner_y = sorted([item["rect"].y0 for item in banners])

    regions = []

    for anchor in anchors:
        rect = anchor["rect"]

        anchor_x = rect.x0
        anchor_y = rect.y0


        if not two_column:
            left = 0
            right = width

        elif anchor_x < mid_x:
            left = 0
            right = mid_x

        else:
            left = mid_x
            right = width


        same_column_anchor_y = [
            a["rect"].y0
            for a in anchors
            if a is not anchor
            and a["rect"].y0 > anchor_y
            and (not two_column or (a["rect"].x0 < mid_x) == (anchor_x < mid_x))
        ]

        # A crop ends at the next SET banner or next problem in the same column.
        stop_candidates = [y for y in banner_y if y > anchor_y] + same_column_anchor_y

        next_stop = min(stop_candidates) if stop_candidates else None


        padding_above = 20

        top = max(0, anchor_y - padding_above)


        if next_stop is not None:
            bottom = next_stop - 10

        else:
            bottom = height

        box = (int(left), int(top), int(right), int(bottom))

        regions.append(
            {
                "label": anchor["label"],
                "set": anchor["set"],
                "box": box,
                "print_bottom": find_print_text_bottom(image, box),
            }
        )

    return regions
