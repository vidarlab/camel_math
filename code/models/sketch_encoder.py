"""
ResNet-50 sketch encoder pretrained on QuickDraw with the Vectorization objective.

The checkpoint in `models/pretrained/quickdraw_vectorization_resnet50.pth` is the
raster encoder half of the Vectorization pretraining model. Its architecture must
be reproduced exactly here or the state dict will not load: a torchvision
ResNet-50 with `avgpool` and `fc` dropped, followed by AdaptiveMaxPool2d(1) and a
flatten to a 2048-dimensional feature vector.

The saved parameter names are prefixed with `features.` (for example
`features.conv1.weight`), which is why the backbone stages are held in a
`self.features` Sequential rather than being inlined.
"""


from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as backbone_


FEATURE_DIM = 2048


# code/models/sketch_encoder.py -> code/models -> code -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "pretrained" / "quickdraw_vectorization_resnet50.pth"


# Layers discarded from the torchvision ResNet-50 during pretraining.
DISCARDED_LAYERS = ("avgpool", "fc")


class SketchEncoder(nn.Module):
    """ResNet-50 trunk that maps a sketch image batch to 2048-d features."""

    def __init__(self):
        super().__init__()

        backbone = backbone_.resnet50(weights=None)

        self.features = nn.Sequential()

        for name, module in backbone.named_children():
            if name not in DISCARDED_LAYERS:
                self.features.add_module(name, module)

        self.pool_method = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        """Take a (B, 3, H, W) image batch and return (B, 2048) features."""
        x = self.features(x)

        return self.pool_method(x).view(-1, FEATURE_DIM)


def load_pretrained_encoder(checkpoint_path=DEFAULT_CHECKPOINT, device="cpu", eval_mode=True):
    """Build a SketchEncoder and load the QuickDraw Vectorization weights into it.

    Args:
        checkpoint_path: `.pth` state dict to load. Defaults to the checkpoint
            committed under `models/pretrained/`.
        device: Device string or `torch.device` to move the encoder onto.
        eval_mode: Put the encoder in eval mode, which is what feature
            extraction wants. Pass False to fine-tune.

    Returns:
        The loaded `SketchEncoder`.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Pretrained encoder checkpoint not found: " + str(checkpoint_path)
        )

    encoder = SketchEncoder()

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # The pretraining script saved the encoder without a classifier head, so the
    # keys should line up one to one. Keep this strict to catch silent mismatches.
    encoder.load_state_dict(state_dict)

    encoder = encoder.to(device)

    if eval_mode:
        encoder.eval()

    return encoder
