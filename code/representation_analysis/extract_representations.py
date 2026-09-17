"""
Extract pretrained visual representations for CAMEL answer images.

Encoders:
    1. ImageNet-pretrained ResNet-50
    2. QuickDraw Vectorization-pretrained ResNet-50
    3. Pix2Struct pretrained visual encoder

Each saved representation is associated with:
    - answer_id
    - question_id
    - student_id
    - answer_image

For Pix2Struct, this initial extraction saves the masked mean-pooled
768-D representation. Full patch features are not saved yet.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    REPO_ROOT
    / "data"
    / "algebra_dataset"
)

ANSWER_METADATA_PATH = (
    DATA_ROOT
    / "answer_metadata.json"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "representations"
)


# Allow imports from code/
CODE_ROOT = REPO_ROOT / "code"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


from models.pretrained_encoders import (
    ImageNetEncoder,
    Pix2StructEncoder,
    load_vectorization_encoder,
)


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Official preprocessing associated with the ImageNet ResNet-50 weights.
IMAGENET_TRANSFORM = (
    ResNet50_Weights.IMAGENET1K_V2.transforms()
)


# Exact test preprocessing from the Vectorization repository:
#
# Resize(256)
# ToTensor()
# Normalize(mean=.5, std=.5)
#
# Resize(256) preserves aspect ratio.
VECTORIZATION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def load_answer_metadata():
    """Load answer_metadata.json."""

    if not ANSWER_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {ANSWER_METADATA_PATH}"
        )

    with open(
        ANSWER_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return metadata


def get_answer_path(entry):
    """
    Convert the relative answer_image path in the JSON
    into its actual path on disk.
    """

    return DATA_ROOT / entry["answer_image"]


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_rgb_image(path):
    """Load an image and convert it to RGB."""

    with Image.open(path) as image:
        return image.convert("RGB")


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def extract_imagenet(
    metadata,
    device,
):
    """Extract ImageNet ResNet-50 representations."""

    print("\nLoading ImageNet ResNet-50...")

    model = ImageNetEncoder().to(device)
    model.eval()

    features = []
    valid_metadata = []

    for entry in tqdm(
        metadata,
        desc="ImageNet",
    ):
        image_path = get_answer_path(entry)

        if not image_path.is_file():
            print(
                f"\nMissing image: {image_path}"
            )
            continue

        image = load_rgb_image(image_path)

        x = IMAGENET_TRANSFORM(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(x)

        features.append(
            feature.squeeze(0).cpu()
        )

        valid_metadata.append(entry)

    if not features:
        raise RuntimeError(
            "No ImageNet features were extracted."
        )

    features = torch.stack(features)

    return features, valid_metadata


def extract_vectorization(
    metadata,
    device,
):
    """Extract Vectorization ResNet-50 representations."""

    print("\nLoading Vectorization ResNet-50...")

    model = load_vectorization_encoder(
        device=device
    )

    features = []
    valid_metadata = []

    for entry in tqdm(
        metadata,
        desc="Vectorization",
    ):
        image_path = get_answer_path(entry)

        if not image_path.is_file():
            print(
                f"\nMissing image: {image_path}"
            )
            continue

        image = load_rgb_image(image_path)

        x = VECTORIZATION_TRANSFORM(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(x)

        features.append(
            feature.squeeze(0).cpu()
        )

        valid_metadata.append(entry)

    if not features:
        raise RuntimeError(
            "No Vectorization features were extracted."
        )

    features = torch.stack(features)

    return features, valid_metadata


def extract_pix2struct(
    metadata,
    device,
):
    """Extract pooled Pix2Struct representations."""

    print("\nLoading Pix2Struct...")

    model = Pix2StructEncoder().to(device)
    model.eval()

    features = []
    valid_metadata = []

    for entry in tqdm(
        metadata,
        desc="Pix2Struct",
    ):
        image_path = get_answer_path(entry)

        if not image_path.is_file():
            print(
                f"\nMissing image: {image_path}"
            )
            continue

        image = load_rgb_image(image_path)

        inputs = model.preprocess(
            [image]
        )

        flattened_patches = inputs[
            "flattened_patches"
        ].to(device)

        attention_mask = inputs[
            "attention_mask"
        ].to(device)

        with torch.no_grad():
            output = model(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
            )

        feature = output[
            "pooled_features"
        ]

        features.append(
            feature.squeeze(0).cpu()
        )

        valid_metadata.append(entry)

    if not features:
        raise RuntimeError(
            "No Pix2Struct features were extracted."
        )

    features = torch.stack(features)

    return features, valid_metadata


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_representations(
    encoder_name,
    features,
    metadata,
):
    """Save representations and associated metadata."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / f"{encoder_name}.pt"
    )

    output = {
        "encoder": encoder_name,
        "features": features,
        "answer_id": [
            entry["answer_id"]
            for entry in metadata
        ],
        "question_id": [
            entry["question_id"]
            for entry in metadata
        ],
        "student_id": [
            entry["student_id"]
            for entry in metadata
        ],
        "answer_image": [
            entry["answer_image"]
            for entry in metadata
        ],
    }

    torch.save(
        output,
        output_path,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Feature shape: {features.shape}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract pretrained representations "
            "for CAMEL answer images."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process the first N answers. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--encoder",
        choices=[
            "imagenet",
            "vectorization",
            "pix2struct",
            "all",
        ],
        default="all",
        help="Which encoder to run.",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Dataset root:", DATA_ROOT)
    print(
        "Metadata:",
        ANSWER_METADATA_PATH,
    )

    metadata = load_answer_metadata()

    print(
        "Total answers in metadata:",
        len(metadata),
    )

    if args.limit is not None:
        metadata = metadata[
            :args.limit
        ]

        print(
            "Testing with:",
            len(metadata),
            "answers",
        )

    # -------------------------------------------------------------
    # ImageNet
    # -------------------------------------------------------------

    if args.encoder in (
        "imagenet",
        "all",
    ):
        features, valid_metadata = (
            extract_imagenet(
                metadata,
                device,
            )
        )

        save_representations(
            "imagenet_resnet50",
            features,
            valid_metadata,
        )

        del features

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Vectorization
    # -------------------------------------------------------------

    if args.encoder in (
        "vectorization",
        "all",
    ):
        features, valid_metadata = (
            extract_vectorization(
                metadata,
                device,
            )
        )

        save_representations(
            "vectorization_resnet50",
            features,
            valid_metadata,
        )

        del features

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Pix2Struct
    # -------------------------------------------------------------

    if args.encoder in (
        "pix2struct",
        "all",
    ):
        features, valid_metadata = (
            extract_pix2struct(
                metadata,
                device,
            )
        )

        save_representations(
            "pix2struct",
            features,
            valid_metadata,
        )

        del features

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "\nRepresentation extraction complete!"
    )


if __name__ == "__main__":
    main()