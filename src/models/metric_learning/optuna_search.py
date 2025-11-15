import optuna
from torch.utils.data import DataLoader
import torch
from pathlib import Path

from src.models.metric_learning.model import FashionCompatibilityModel
from src.models.metric_learning.trainer import Trainer
from src.models.metric_learning.loss_functions import ContrastiveLoss

OPTUNA_STORAGE = "checkpoints/metric_learning/optuna_study.db"
STUDY_NAME = "fashion_compatibility_search"
NUM_TUNING_EPOCHS = 5

def run_optimization_study(train_dataset, val_dataset, embedding_dim, device="cuda", force_reload=False):
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{OPTUNA_STORAGE}",
        load_if_exists=True,
    )

    if len(study.trials) == 0 or force_reload:
        print("Running Optuna search...")
        study.optimize(
            lambda trial: objective(
                trial,
                train_dataset,
                val_dataset,
                embedding_dim,
                device
            ),
            n_trials=40,
            show_progress_bar=True
        )
    else:
        print("Loaded existing Optuna study, skipping optimization.")

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_trial.params}")
    return study.best_trial.params

def get_hyperparameters():
    try:
        storage = f"sqlite:///{OPTUNA_STORAGE}"
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
        )
    except Exception:
        return None

    # No trials have been run
    if len(study.trials) == 0:
        return None

    return study.best_trial.params

def objective(trial, train_dataset, val_dataset, embedding_dim, device):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    output_dim = trial.suggest_int("output_dim", 32, 256, step=32)

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    margin = trial.suggest_float("margin", 0.5, 2.0)

    model = FashionCompatibilityModel(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        device=device
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(NUM_TUNING_EPOCHS):
        train_loss, train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_metrics = trainer.validate(val_loader, criterion)

        trial.report(val_loss, epoch)

        if trial.should_prune(): # early prune for bad trials
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


def run_optuna_search(train_dataset, val_dataset, embedding_dim, device="cuda"):
    study = optuna.create_study(
        direction="minimize",    # minimizing loss
        study_name=STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
        storage= "sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            val_dataset,
            embedding_dim,
            device
        ),
        n_trials=40,
        show_progress_bar=True
    )

    print("Optuna search completed.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Best value: {study.best_trial.value}")


    return study

