"""
Frozen linear classifier for CAMEL question identity.

Tests whether question identity is linearly accessible from each
pretrained representation before any CAMEL-specific encoder training.

The pretrained encoders are NOT trained or fine-tuned.
Only a single linear classification layer is trained.

Train/test split is by student, so test answers come from students
that the linear classifier did not see during training.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

REPRESENTATION_ROOT = REPO_ROOT / "data" / "representations"

ENCODER_FILES = {
    "ImageNet": REPRESENTATION_ROOT / "imagenet_resnet50.pt",
    "Vectorization": REPRESENTATION_ROOT / "vectorization_resnet50.pt",
    "Pix2Struct (mean pooled)": REPRESENTATION_ROOT / "pix2struct.pt",
}

TEST_FRACTION = 0.20
SEED = 42

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Load saved representations
# ---------------------------------------------------------------------

def load_representation(path):

    data = torch.load(
        path,
        map_location="cpu",
    )

    return {
        "features": data["features"].float(),
        "answer_id": np.array(data["answer_id"]),
        "question_id": np.array(data["question_id"]),
        "student_id": np.array(data["student_id"]),
    }


# ---------------------------------------------------------------------
# Student-held-out split
# ---------------------------------------------------------------------

def make_student_split(student_ids):

    rng = np.random.default_rng(SEED)

    students = np.unique(student_ids)
    rng.shuffle(students)

    num_test = max(
        1,
        int(len(students) * TEST_FRACTION),
    )

    test_students = set(
        students[:num_test]
    )

    test_mask = np.array([
        student in test_students
        for student in student_ids
    ])

    train_mask = ~test_mask

    return train_mask, test_mask


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def standardize_features(X_train, X_test):

    # Statistics are calculated ONLY from training data.
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)

    # Avoid division by zero for constant dimensions.
    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std,
    )

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        # Top-1
        predictions = logits.argmax(dim=1)

        correct_top1 += (
            predictions == labels
        ).sum().item()

        # Top-5
        k = min(5, logits.shape[1])

        top5_predictions = logits.topk(
            k,
            dim=1,
        ).indices

        correct_top5 += (
            top5_predictions
            == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        total += labels.size(0)

    top1 = correct_top1 / total
    top5 = correct_top5 / total

    return top1, top5


# ---------------------------------------------------------------------
# Train one linear classifier
# ---------------------------------------------------------------------

def train_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes,
    device,
):

    feature_dim = X_train.shape[1]

    # -------------------------------------------------------------
    # Standardize frozen features
    # -------------------------------------------------------------

    X_train, X_test = standardize_features(
        X_train,
        X_test,
    )

    # -------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train,
    )

    test_dataset = TensorDataset(
        X_test,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -------------------------------------------------------------
    # LINEAR classifier
    #
    # No hidden layers.
    # No nonlinearities.
    # -------------------------------------------------------------

    model = nn.Linear(
        feature_dim,
        num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for features, labels in train_loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            total_samples += (
                labels.size(0)
            )

        # Print occasionally rather than every epoch.
        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            train_loss = (
                running_loss
                / total_samples
            )

            top1, top5 = evaluate(
                model,
                test_loader,
                device,
            )

            print(
                f"Epoch {epoch:>3}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Test Top-1: {top1 * 100:.2f}% | "
                f"Test Top-5: {top5 * 100:.2f}%"
            )

    # -------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------

    top1, top5 = evaluate(
        model,
        test_loader,
        device,
    )

    return top1, top5


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print()

    # -------------------------------------------------------------
    # Load reference metadata
    # -------------------------------------------------------------

    reference = load_representation(
        ENCODER_FILES["ImageNet"]
    )

    answer_ids = reference["answer_id"]
    question_ids = reference["question_id"]
    student_ids = reference["student_id"]

    print(
        "Total answers:",
        len(answer_ids),
    )

    print(
        "Unique questions:",
        len(np.unique(question_ids)),
    )

    print(
        "Unique students:",
        len(np.unique(student_ids)),
    )

    # -------------------------------------------------------------
    # Create ONE student split shared by every encoder
    # -------------------------------------------------------------

    train_mask, test_mask = make_student_split(
        student_ids
    )

    print()
    print(
        "Training students:",
        len(np.unique(
            student_ids[train_mask]
        )),
    )

    print(
        "Test students:",
        len(np.unique(
            student_ids[test_mask]
        )),
    )

    print(
        "Training answers:",
        train_mask.sum(),
    )

    print(
        "Test answers:",
        test_mask.sum(),
    )

    # -------------------------------------------------------------
    # Convert question IDs into integer class labels
    # -------------------------------------------------------------

    unique_questions = sorted(
        np.unique(question_ids)
    )

    question_to_index = {
        question: index
        for index, question
        in enumerate(unique_questions)
    }

    labels = torch.tensor(
        [
            question_to_index[q]
            for q in question_ids
        ],
        dtype=torch.long,
    )

    num_classes = len(
        unique_questions
    )

    print(
        "Question classes:",
        num_classes,
    )

    print(
        f"Uniform random Top-1 chance: "
        f"{100 / num_classes:.2f}%"
    )

    # -------------------------------------------------------------
    # Verify all test questions also occur in training
    # -------------------------------------------------------------

    train_question_ids = set(
        question_ids[train_mask]
    )

    test_question_ids = set(
        question_ids[test_mask]
    )

    missing_questions = (
        test_question_ids
        - train_question_ids
    )

    if missing_questions:

        raise RuntimeError(
            f"{len(missing_questions)} test questions "
            "do not occur in the training split."
        )

    # Convert masks for tensor indexing.
    train_indices = torch.tensor(
        np.where(train_mask)[0],
        dtype=torch.long,
    )

    test_indices = torch.tensor(
        np.where(test_mask)[0],
        dtype=torch.long,
    )

    y_train = labels[train_indices]
    y_test = labels[test_indices]

    # -------------------------------------------------------------
    # Evaluate each representation
    # -------------------------------------------------------------

    results = []

    for encoder_name, path in ENCODER_FILES.items():

        print()
        print("=" * 75)
        print(encoder_name)
        print("=" * 75)

        data = load_representation(path)

        # Ensure representations correspond to exactly
        # the same answers in exactly the same order.
        if not np.array_equal(
            data["answer_id"],
            answer_ids,
        ):

            raise RuntimeError(
                f"{encoder_name} answer ordering "
                "does not match ImageNet."
            )

        X = data["features"]

        print(
            "Feature shape:",
            tuple(X.shape),
        )

        X_train = X[train_indices]
        X_test = X[test_indices]

        top1, top5 = train_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            num_classes,
            device,
        )

        results.append(
            (
                encoder_name,
                top1,
                top5,
            )
        )

        # Free GPU memory before next classifier.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print()
    print("=" * 75)
    print("FROZEN QUESTION-ID LINEAR CLASSIFIER")
    print("=" * 75)

    print(
        f"{'Representation':<32}"
        f"{'Top-1':>12}"
        f"{'Top-5':>12}"
    )

    print("-" * 56)

    for name, top1, top5 in results:

        print(
            f"{name:<32}"
            f"{top1 * 100:>11.2f}%"
            f"{top5 * 100:>11.2f}%"
        )


if __name__ == "__main__":
    main()