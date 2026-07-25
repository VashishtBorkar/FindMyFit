"""Offline metric-model training entry point.

The current pair splitting and evaluation behavior is intentionally retained for
the later ML-correctness phase.
"""

from __future__ import annotations

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from findmyfit.config import Settings
from findmyfit.models.metric import FashionCompatibilityModel
from training.metric_learning.data import PairDataset, load_embeddings, load_pairs
from training.metric_learning.loss import ContrastiveLoss
from training.metric_learning.trainer import Trainer
from training.metric_learning.tune import run_optimization_study


EMBEDDING_DIM = 512


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    total = len(dataset)
    train_size = int(train_ratio * total)
    validation_size = int(val_ratio * total)
    test_size = total - train_size - validation_size
    return random_split(dataset, [train_size, validation_size, test_size])


def main() -> None:
    settings = Settings.from_env()
    embeddings, _ = load_embeddings(settings.clip_embeddings_dir)
    pairs = load_pairs(
        embeddings,
        settings.compatibility_outfits_file,
        settings.compatibility_pairs_path,
    )
    train_set, validation_set, test_set = split_dataset(PairDataset(pairs, embeddings))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parameters = run_optimization_study(
        train_set,
        validation_set,
        EMBEDDING_DIM,
        settings.optuna_storage_path,
        device,
    )

    model = FashionCompatibilityModel(
        EMBEDDING_DIM,
        parameters["hidden_dim"],
        parameters["output_dim"],
    )
    optimizer = optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    criterion = ContrastiveLoss(parameters["margin"])
    trainer = Trainer(model, device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    train_loader = DataLoader(
        train_set, batch_size=parameters["batch_size"], shuffle=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=parameters["batch_size"]
    )

    best_loss = float("inf")
    no_improvement_epochs = 0
    for epoch in range(50):
        train_loss, _ = trainer.train_epoch(train_loader, optimizer, criterion)
        validation_loss, _ = trainer.validate(validation_loader, criterion)
        scheduler.step(validation_loss)
        print(
            f"Epoch {epoch}: train={train_loss:.4f}, validation={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            no_improvement_epochs = 0
            settings.metric_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    **parameters,
                },
                settings.metric_checkpoint_path,
            )
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= 5:
                break

    test_loader = DataLoader(test_set, batch_size=parameters["batch_size"])
    test_loss, test_metrics = trainer.test(test_loader, criterion)
    print(f"Test loss={test_loss:.4f}; metrics={test_metrics}")


if __name__ == "__main__":
    main()
