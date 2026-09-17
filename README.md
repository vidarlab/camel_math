# Camel Math Dataset Pipeline

Tools for extracting algebra workbook questions from a blank reference PDF, aligning student workbook scans to that reference, and building answer/question metadata tables.

## Source Layout

- `code/dataset_creation/workbook_layout.py` - shared blank-workbook layout detection.
- `code/dataset_creation/question_extraction.py` - creates question crops and `dataset.json`.
- `code/dataset_creation/answer_extraction.py` - creates student answer crops using the blank workbook reference.
- `code/dataset_creation/remove_blank_answers.py` - cleanup pass for crops judged blank.
- `code/dataset_creation/build_answer_question_index.py` - writes answer/question index and metadata JSON files.
- `code/review_apps/question_review_app.py` - review extracted questions.
- `code/review_apps/answer_review_app.py` - review one student's answers.
- `code/review_apps/page_review_app.py` - review all answers on selected pages.
- `code/models/sketch_encoder.py` - ResNet-50 sketch encoder definition and loader for the QuickDraw Vectorization checkpoint.
- `models/pretrained/` - pretrained checkpoints (gitignored; see `quickdraw_vectorization_resnet50.pth`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Commands

Run from `code/dataset_creation`:

```bash
python3 question_extraction.py
python3 answer_extraction.py
python3 build_answer_question_index.py
```
