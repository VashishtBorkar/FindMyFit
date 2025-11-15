from src.data_manager.pair_manager import PairDataset, load_pairs
from src.data_manager.embedding_manager import load_embeddings
from src.models.metric_learning.model import FashionCompatibilityModel
from src.models.metric_learning.loss_functions import ContrastiveLoss
from src.models.metric_learning.trainer import Trainer
from src.models.metric_learning.optuna_search import run_optimization_study

import optuna
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dotenv import load_dotenv
from torch import optim
import os
from pathlib import Path
import torch

EMBEDDING_DIM = 512  # CLIP embedding dimension

def load_data():
    load_dotenv()
    embeddings_dir = os.getenv("CLIP_EMBEDDINGS_DIR", "data/clip_embeddings") 
    embeddings_pickle = os.getenv("EMBEDDINGS_PICKLE", None)
    outfit_file = Path(os.getenv("COMPATIBILE_OUTFITS_FILE"), "data/fashion_compatibility_predictions.txt")
    pairs_pickle = Path(os.getenv("PAIRS_PICKLE", "data/compatibility_pairs.pkl"))

    checkpoint_dir = Path("checkpoints/metric_learning")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    embeddings, _ = load_embeddings(embeddings_dir)
    pairs = load_pairs(embeddings, outfit_file, pairs_pickle)
    dataset = PairDataset(pairs, embeddings)
    print(f"Total pairs in dataset: {len(dataset)}")
    
    return dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

def train_model(train_set, val_set, best_params):
    train_set, val_set, test_set = split_dataset(dataset)

    batch_size = best_params['batch_size']
    lr = best_params['learning_rate']
    margin = best_params['margin']
    hidden_dim = best_params['hidden_dim']
    output_dim = best_params['output_dim']

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCompatibilityModel(
        embedding_dim=512,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=margin)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    trainer = Trainer(model=model, device=device)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = 5
    no_improve_epochs = 0
    max_epochs = 50 

    print(f"Training on device: {device}")

    for epoch in range(max_epochs):
        train_loss, train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_metrics = trainer.validate(val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            checkpoint_path = Path("checkpoints/metric_learning/best_model.pt")
            print(f"Saving best model at epoch {epoch} with val loss {val_loss:.4f} to {checkpoint_path}")
            #torch.save(model.state_dict(), checkpoint_path)
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "batch_size": batch_size,
                "learning_rate": lr,
                "margin": margin
            }, checkpoint_path)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Stopping early due to no improvement in validation loss.")
            break

    test_loader = DataLoader(test_set, batch_size=best_params["batch_size"], shuffle=False)
    test_loss, test_metrics = trainer.validate(test_loader, ContrastiveLoss(margin=best_params["margin"]))
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    print("Loading data...")
    dataset = load_data()

    print("Starting hyperparameter optimization...")
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    best_params = run_optimization_study(train_dataset, val_dataset, EMBEDDING_DIM, device=device)
    print(f"Training final model with hyperparameters: {best_params}")
    
    print("Training model...")
    train_model(train_dataset, val_dataset, best_params)
    print("Training complete.")