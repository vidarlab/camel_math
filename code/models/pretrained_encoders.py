"""
Pretrained visual encoders for CAMEL representation analysis.

Encoders:
    1. ImageNet-pretrained ResNet-50
    2. QuickDraw Vectorization-pretrained ResNet-50
    3. Pix2Struct pretrained visual encoder

For the ResNet models, each image is represented by a single 2048-D vector.

Pix2Struct produces patch-level representations. We preserve those patch
representations and also compute a masked mean-pooled vector for initial
image-level analysis.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from transformers import (
    AutoProcessor,
    Pix2StructForConditionalGeneration,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

VECTORIZATION_CHECKPOINT = (
    REPO_ROOT
    / "models"
    / "pretrained"
    / "quickdraw_vectorization_resnet50.pth"
)


# ---------------------------------------------------------------------
# ImageNet ResNet-50
# ---------------------------------------------------------------------

class ImageNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        model = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # Remove the ImageNet classification layer.
        # The standard ResNet average pooling is kept.
        model.fc = nn.Identity()

        self.encoder = model
        self.feature_dim = 2048

    def forward(self, x):
        """
        Input:
            x: (B, 3, H, W)

        Output:
            (B, 2048)
        """

        return self.encoder(x)


# ---------------------------------------------------------------------
# Vectorization ResNet-50
# ---------------------------------------------------------------------

class VectorizationEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=None)

        self.features = nn.Sequential()

        for name, module in backbone.named_children():
            if name not in ("avgpool", "fc"):
                self.features.add_module(name, module)

        # Match the pooling used by the Vectorization model.
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.feature_dim = 2048

    def forward(self, x):
        """
        Input:
            x: (B, 3, H, W)

        Output:
            (B, 2048)
        """

        x = self.features(x)
        x = self.pool(x)

        return torch.flatten(x, 1)


def load_vectorization_encoder(
    checkpoint_path=VECTORIZATION_CHECKPOINT,
    device="cpu",
):
    """
    Load the QuickDraw Vectorization-pretrained ResNet-50.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Vectorization checkpoint not found: {checkpoint_path}"
        )

    model = VectorizationEncoder()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Pix2Struct
# ---------------------------------------------------------------------

class Pix2StructEncoder(nn.Module):

    def __init__(
        self,
        model_name="google/pix2struct-base",
    ):
        super().__init__()

        self.model_name = model_name

        # Pix2Struct-specific image preprocessing.
        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        # Load the COMPLETE pretrained Pix2Struct checkpoint.
        full_model = (
            Pix2StructForConditionalGeneration.from_pretrained(
                model_name
            )
        )

        # Keep the pretrained visual encoder.
        self.encoder = full_model.encoder

        # We do not need the text decoder for representation extraction.
        del full_model

        self.feature_dim = self.encoder.config.hidden_size

    def preprocess(self, images):
        """
        Convert images into the flattened patches and attention masks
        expected by the Pix2Struct vision encoder.
        """

        return self.processor(
            images=images,
            return_tensors="pt",
        )

    def forward(
        self,
        flattened_patches,
        attention_mask,
    ):
        """
        Returns:
            patch_features:
                Patch-level Pix2Struct representations.

            pooled_features:
                One masked mean-pooled vector per image.

            attention_mask:
                Indicates valid vs padded patches.
        """

        outputs = self.encoder(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
        )

        # Shape:
        # (B, number_of_patches, 768)
        patch_features = outputs.last_hidden_state

        # -------------------------------------------------------------
        # Temporary whole-image representation
        # -------------------------------------------------------------
        #
        # Pix2Struct naturally gives us patch representations.
        # For simple image-to-image comparisons, we also create one
        # vector by averaging only the valid patches.
        #
        # This pooled vector is for initial analysis. We still preserve
        # the full patch-level representation.
        # -------------------------------------------------------------

        mask = attention_mask.unsqueeze(-1).to(
            patch_features.dtype
        )

        summed_features = (
            patch_features * mask
        ).sum(dim=1)

        number_of_valid_patches = (
            mask.sum(dim=1).clamp(min=1)
        )

        pooled_features = (
            summed_features
            / number_of_valid_patches
        )

        return {
            "patch_features": patch_features,
            "pooled_features": pooled_features,
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------

def main():
    """
    Verify that all three pretrained encoders load and
    successfully process an image.
    """

    from PIL import Image

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # =================================================================
    # 1. ImageNet ResNet-50
    # =================================================================

    print("\nLoading ImageNet ResNet50...")

    imagenet = ImageNetEncoder().to(device)
    imagenet.eval()

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        features = imagenet(x)

    print(
        "ImageNet output:",
        features.shape,
    )

    del imagenet
    del x
    del features

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =================================================================
    # 2. Vectorization ResNet-50
    # =================================================================

    print("\nLoading Vectorization ResNet50...")

    vectorization = load_vectorization_encoder(
        device=device
    )

    x = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():
        features = vectorization(x)

    print(
        "Vectorization output:",
        features.shape,
    )

    del vectorization
    del x
    del features

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =================================================================
    # 3. Pix2Struct
    # =================================================================

    print("\nLoading Pix2Struct...")

    pix2struct = Pix2StructEncoder().to(device)
    pix2struct.eval()

    # Fake white image.
    # This is ONLY checking that preprocessing + forward pass work.
    image = Image.new(
        "RGB",
        (800, 600),
        "white",
    )

    inputs = pix2struct.preprocess(
        [image]
    )

    flattened_patches = inputs[
        "flattened_patches"
    ].to(device)

    attention_mask = inputs[
        "attention_mask"
    ].to(device)

    with torch.no_grad():
        outputs = pix2struct(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
        )

    print(
        "Pix2Struct patch features:",
        outputs["patch_features"].shape,
    )

    print(
        "Pix2Struct pooled features:",
        outputs["pooled_features"].shape,
    )

    print(
        "Valid patches:",
        outputs["attention_mask"].sum().item(),
    )

    print(
        "\nAll three pretrained encoders loaded successfully!"
    )


if __name__ == "__main__":
    main()