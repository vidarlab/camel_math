"""
Review frozen-representation question classifiers one held-out answer at a time.

Three pretrained encoders (ImageNet ResNet50, Vectorization ResNet50 and
Pix2Struct) were frozen and only a linear question-ID classifier was trained on
top of each. Use this to read, qualitatively, what each representation thinks a
student's handwritten answer belongs to: the student crop and the true question
crop side by side, then the Top-5 predicted question crops per encoder with the
ground-truth prediction marked.

Served with Flask over plain HTTP GETs. Every URL in the template is relative,
so the app works unchanged when it is reached through a reverse proxy that
mounts it under a path prefix (an OnDemand /rnode/<host>/<job>/proxy/7860/ URL,
for instance) as well as when it is opened directly on localhost.
"""


import argparse
import json
import socket
from pathlib import Path

import torch
from flask import Flask, abort, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/algebra_dataset"
ANSWER_CROPS_DIR = DATA_ROOT / "answer_crops"
PROBLEM_CROPS_DIR = DATA_ROOT / "problem_crops"
QUESTION_METADATA_PATH = DATA_ROOT / "question_metadata.json"
DATASET_JSON_PATH = DATA_ROOT / "dataset.json"

RESULT_ROOT = PROJECT_ROOT / "data/representation_analysis"

RESULT_FILES = {
    "ImageNet": RESULT_ROOT / "imagenet_q_classifier.pt",
    "Vectorization": RESULT_ROOT / "vectorization_q_classifier.pt",
    "Pix2Struct": RESULT_ROOT / "pix2struct_q_classifier.pt",
}

MODEL_NAMES = list(RESULT_FILES)

# Every category the classifier results can be sliced into, in menu order.
CATEGORIES = [
    "All test answers",
    "All models correct",
    "Pix2Struct correct",
    "Pix2Struct Top-1 wrong, actual in Top-5",
    "Pix2Struct Top-5 failure",
    "Vectorization correct, Pix2Struct wrong",
    "Pix2Struct correct, Vectorization wrong",
    "ImageNet wrong, both specialized correct",
    "All models Top-1 wrong",
]


# ---------------------------------------------------------------------------
# Classifier results
# ---------------------------------------------------------------------------


def load_results():
    """Load one classifier result file per encoder."""
    loaded = {}

    for name, path in RESULT_FILES.items():
        print(f"  Loading {name}: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Missing classifier result file: {path}")

        loaded[name] = torch.load(path, map_location="cpu")

    return loaded


print("Loading classifier results...")

RESULTS = load_results()

REFERENCE = RESULTS["Pix2Struct"]

for name, result in RESULTS.items():
    if result["answer_id"] != REFERENCE["answer_id"]:
        raise RuntimeError(f"{name} answer ordering does not match Pix2Struct.")


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


def load_question_metadata():
    """Map question ID to its metadata record."""
    with open(QUESTION_METADATA_PATH, encoding="utf-8") as f:
        records = json.load(f)

    return {record["question_id"]: record for record in records}


def load_question_text():
    """Map question ID to its transcribed prompt, when dataset.json has one."""
    if not DATASET_JSON_PATH.is_file():
        return {}

    with open(DATASET_JSON_PATH, encoding="utf-8") as f:
        records = json.load(f)

    # dataset.json is keyed by question_image, so bridge through the metadata.
    by_image = {
        record["question_image"]: record.get("question_text", {})
        for record in records
        if record.get("question_image")
    }

    text = {}

    for question_id, metadata in QUESTION_METADATA.items():
        entry = by_image.get(metadata.get("question_image"))

        if entry:
            text[question_id] = entry

    return text


print("Loading question metadata...")

QUESTION_METADATA = load_question_metadata()
QUESTION_TEXT = load_question_text()

print(
    f"Loaded {len(QUESTION_METADATA)} question metadata entries "
    f"({len(QUESTION_TEXT)} with prompt text)."
)


# ---------------------------------------------------------------------------
# Path resolution
#
# Verified against the real dataset:
#   classifier "answer_image"          -> "answer_crops/page_4/problem_1a/AbE_x.png"
#                                    relative to data/algebra_dataset/
#   metadata "question_image"     -> "page_4/problem_1a.png"
#                                    relative to data/algebra_dataset/problem_crops/
# The extra candidates below only cover the same file written a different way;
# whichever one exists on disk is used, and every miss is reported.
# ---------------------------------------------------------------------------


def first_existing(candidates, label):
    """Return the first candidate that exists, printing all misses otherwise."""
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    print(f"IMAGE NOT FOUND for {label}. Tried:")

    for candidate in candidates:
        print(f"    {candidate}")

    return None


def resolve_answer_path(stored_path):
    """Resolve a classifier 'answer_image' entry to a file on disk."""
    if stored_path is None:
        print("NO ANSWER IMAGE RECORDED for this example.")

        return None

    path = Path(stored_path)

    if path.is_absolute():
        return first_existing([path], f"answer image {stored_path!r}")

    return first_existing(
        [
            DATA_ROOT / path,
            ANSWER_CROPS_DIR / path,
            PROJECT_ROOT / path,
        ],
        f"answer image {stored_path!r}",
    )


def resolve_question_path(question_id):
    """Resolve a question ID to its problem crop on disk."""
    metadata = QUESTION_METADATA.get(question_id)

    if metadata is None:
        print(f"QUESTION NOT IN METADATA: {question_id}")

        return None

    stored_path = metadata.get("question_image")

    if stored_path is None:
        print(f"NO QUESTION IMAGE RECORDED FOR: {question_id}")

        return None

    path = Path(stored_path)

    if path.is_absolute():
        return first_existing([path], f"question {question_id}")

    return first_existing(
        [
            PROBLEM_CROPS_DIR / path,
            DATA_ROOT / path,
            PROJECT_ROOT / path,
        ],
        f"question {question_id} ({stored_path!r})",
    )


def preflight():
    """Check every question crop and a sample of answer crops up front."""
    missing_questions = [
        question_id
        for question_id in REFERENCE["question_classes"]
        if resolve_question_path(question_id) is None
    ]

    total = len(REFERENCE["answer_id"])
    sample = range(0, total, max(1, total // 25))

    missing_answers = [
        REFERENCE["answer_image"][i]
        for i in sample
        if resolve_answer_path(REFERENCE["answer_image"][i]) is None
    ]

    print(
        f"Path check: {len(REFERENCE['question_classes']) - len(missing_questions)}"
        f"/{len(REFERENCE['question_classes'])} question crops resolve, "
        f"{len(list(sample)) - len(missing_answers)}/{len(list(sample))} "
        "sampled answer crops resolve."
    )

    if missing_questions or missing_answers:
        print("Some crops are missing; the paths tried are listed above.")


# ---------------------------------------------------------------------------
# Case categories
# ---------------------------------------------------------------------------


def build_cases():
    """Group test answers into the qualitative categories worth browsing."""
    cases = {name: [] for name in CATEGORIES}

    imagenet = RESULTS["ImageNet"]
    vectorization = RESULTS["Vectorization"]
    pix2struct = RESULTS["Pix2Struct"]

    for i in range(len(REFERENCE["answer_id"])):
        img_correct = bool(imagenet["top1_correct"][i])
        vec_correct = bool(vectorization["top1_correct"][i])
        pix_correct = bool(pix2struct["top1_correct"][i])
        pix_top5 = bool(pix2struct["actual_in_top5"][i])

        cases["All test answers"].append(i)

        if img_correct and vec_correct and pix_correct:
            cases["All models correct"].append(i)

        if pix_correct:
            cases["Pix2Struct correct"].append(i)

        if not pix_correct and pix_top5:
            cases["Pix2Struct Top-1 wrong, actual in Top-5"].append(i)

        if not pix_top5:
            cases["Pix2Struct Top-5 failure"].append(i)

        if vec_correct and not pix_correct:
            cases["Vectorization correct, Pix2Struct wrong"].append(i)

        if pix_correct and not vec_correct:
            cases["Pix2Struct correct, Vectorization wrong"].append(i)

        if not img_correct and vec_correct and pix_correct:
            cases["ImageNet wrong, both specialized correct"].append(i)

        if not img_correct and not vec_correct and not pix_correct:
            cases["All models Top-1 wrong"].append(i)

    return cases


CASES = build_cases()

preflight()

print()
print("=" * 70)
print("CASE COUNTS")
print("=" * 70)

for name, indices in CASES.items():
    print(f"{name:<55}{len(indices)}")

print("=" * 70)

for name, result in RESULTS.items():
    print(
        f"{name:<20}top-1 {result['top1_accuracy'] * 100:6.2f}%   "
        f"top-5 {result['top5_accuracy'] * 100:6.2f}%"
    )

print("=" * 70)
print()

NON_EMPTY_CATEGORIES = [name for name in CATEGORIES if CASES[name]]

DEFAULT_CATEGORY = (
    "Pix2Struct Top-1 wrong, actual in Top-5"
    if CASES["Pix2Struct Top-1 wrong, actual in Top-5"]
    else NON_EMPTY_CATEGORIES[0]
)


# ---------------------------------------------------------------------------
# Prediction interpretation
# ---------------------------------------------------------------------------


def top5_for(model_name, index):
    """Return (question_ids, probabilities, actual_question_id) for one example."""
    result = RESULTS[model_name]

    probabilities = result["top5_probabilities"][index]

    if torch.is_tensor(probabilities):
        probabilities = probabilities.tolist()

    return (
        list(result["top5_question_ids"][index]),
        list(probabilities),
        result["actual_question_id"][index],
    )


def status_for(question_ids, actual_question_id):
    """Describe how a model's Top-5 relates to the ground-truth question."""
    if question_ids and question_ids[0] == actual_question_id:
        return "Top-1 correct", "correct"

    if actual_question_id in question_ids:
        return "Top-1 wrong, actual in Top-5", "partial"

    return "Top-5 failure", "failure"


def model_block(model_name, index):
    """Build one encoder's panel: status plus its five predicted questions."""
    question_ids, probabilities, actual_question_id = top5_for(model_name, index)

    status, status_class = status_for(question_ids, actual_question_id)

    predictions = []

    for rank, (question_id, probability) in enumerate(
        zip(question_ids, probabilities), start=1
    ):
        predictions.append(
            {
                "rank": rank,
                "question_id": question_id,
                "probability": probability * 100,
                "is_actual": question_id == actual_question_id,
                "available": resolve_question_path(question_id) is not None,
            }
        )

    return {
        "name": model_name,
        "status": status,
        "status_class": status_class,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def selected_category():
    """Read the category from the query string, falling back to the default."""
    category = request.args.get("category", DEFAULT_CATEGORY)

    return category if category in CASES and CASES[category] else DEFAULT_CATEGORY


def selected_position(category):
    """Read the 1-based example number, clamped into the category."""
    total = len(CASES[category])

    # A fresh category selection restarts at the first example.
    if request.args.get("previous_category") not in (None, category):
        return 1

    try:
        position = int(request.args.get("example", 1))
    except (TypeError, ValueError):
        position = 1

    return max(1, min(position, total))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    """Render one example: the answer, the true question, and every encoder."""
    category = selected_category()
    position = selected_position(category)
    total = len(CASES[category])

    index = CASES[category][position - 1]

    actual_question_id = REFERENCE["actual_question_id"][index]
    stored_answer_path = REFERENCE["answer_image"][index]

    print()
    print("-" * 70)
    print(f"Example {position}/{total}  ({category})")
    print(f"  Answer ID       : {REFERENCE['answer_id'][index]}")
    print(f"  Student         : {REFERENCE['student_id'][index]}")
    print(f"  Actual question : {actual_question_id}")

    answer_path = resolve_answer_path(stored_answer_path)
    question_path = resolve_question_path(actual_question_id)

    print(f"  Answer image    : {answer_path}")
    print(f"  Question image  : {question_path}")

    text = QUESTION_TEXT.get(actual_question_id) or {}

    return render_template(
        "q_classifier_review.html",
        categories=NON_EMPTY_CATEGORIES,
        counts={name: len(CASES[name]) for name in NON_EMPTY_CATEGORIES},
        category=category,
        position=position,
        total=total,
        index=index,
        answer_id=REFERENCE["answer_id"][index],
        student_id=REFERENCE["student_id"][index],
        actual_question_id=actual_question_id,
        problem_instructions=text.get("problem_instructions"),
        section_instructions=text.get("section_instructions"),
        answer_available=answer_path is not None,
        question_available=question_path is not None,
        models=[model_block(name, index) for name in MODEL_NAMES],
    )


@app.route("/answer-image/<int:index>")
def answer_image(index):
    """Serve the student answer crop for one test index, straight from disk."""
    # The index is only ever an offset into the loaded results, so no path from
    # the URL reaches the filesystem.
    if not 0 <= index < len(REFERENCE["answer_image"]):
        abort(404)

    path = resolve_answer_path(REFERENCE["answer_image"][index])

    if path is None:
        abort(404)

    return send_file(str(path), mimetype="image/png")


@app.route("/question-image/<question_id>")
def question_image(question_id):
    """Serve a question crop, looked up through the loaded metadata only."""
    if question_id not in QUESTION_METADATA:
        abort(404)

    path = resolve_question_path(question_id)

    if path is None:
        abort(404)

    return send_file(str(path), mimetype="image/png")


def local_addresses(port):
    """Addresses that actually resolve in a browser, unlike 0.0.0.0."""
    addresses = [f"http://localhost:{port}"]

    # Ask the routing table which address this host uses to reach the network;
    # gethostname() often maps straight back to loopback on cluster nodes.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))

            host_ip = probe.getsockname()[0]

        if host_ip and not host_ip.startswith("127."):
            addresses.append(f"http://{host_ip}:{port}")
    except OSError:
        pass

    return addresses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier review app.")

    parser.add_argument("--host", default="0.0.0.0", help="interface to bind")
    parser.add_argument("--port", type=int, default=7860, help="port to bind")

    args = parser.parse_args()

    print("Serving. Open one of these directly (NOT the 0.0.0.0 address):")

    for address in local_addresses(args.port):
        print(f"    {address}")

    print("Through the OnDemand proxy, the /proxy/<port>/ URL works as well.")
    print()

    app.run(host=args.host, port=args.port, debug=False)
