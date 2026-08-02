"""Optuna search for metric-learning hyperparameters."""

from __future__ import annotations

from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader

from findmyfit.models.metric import FashionCompatibilityModel
from training.metric_learning.loss import ContrastiveLoss
from training.metric_learning.trainer import Trainer

STUDY_NAME = "fashion_compatibility_search"
NUM_TUNING_EPOCHS = 5


def objective(trial, train_dataset, val_dataset, embedding_dim, device):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    output_dim = trial.suggest_int("output_dim", 32, 256, step=32)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    margin = trial.suggest_float("margin", 0.5, 2.0)

    model = FashionCompatibilityModel(embedding_dim, hidden_dim, output_dim).to(device)
    criterion = ContrastiveLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    for epoch in range(NUM_TUNING_EPOCHS):
        trainer.train_epoch(train_loader, optimizer, criterion)
        validation_loss, _ = trainer.validate(val_loader, criterion)
        best_loss = min(best_loss, validation_loss)
        trial.report(validation_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_loss


def run_optimization_study(
    train_dataset,
    val_dataset,
    embedding_dim: int,
    storage_path: Path,
    device: str = "cuda",
    force_reload: bool = False,
) -> dict:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{storage_path.as_posix()}",
        load_if_exists=True,
    )
    if not study.trials or force_reload:
        study.optimize(
            lambda trial: objective(
                trial, train_dataset, val_dataset, embedding_dim, device
            ),
            n_trials=40,
            show_progress_bar=True,
        )
    return study.best_trial.params
