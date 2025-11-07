from src.data_manager.pair_manager import PairManager, PairDataset
from src.data_manager.embedding_manager import EmbeddingManager
from src.models.metric_learning.model import FashionCompatibilityModel
from src.models.metric_learning.loss_functions import ContrastiveLoss

from typing import Dict, Tuple
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn as nn
from dotenv import load_dotenv
from torch import optim
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch

def calculate_metrics(distances: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Simple metrics for metric learning."""
    # Lower distance = compatible (predict 1)
    pred_binary = (distances < threshold).float()
    accuracy = (pred_binary == labels).float().mean().item()
    
    return {'accuracy': accuracy}


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_distances = []
    all_labels = []
    
    for emb_a, emb_b, labels in tqdm(train_loader, desc="Training", leave=False):
        emb_a = emb_a.to(device)
        emb_b = emb_b.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()

        # Forward pass
        feat_a = model(emb_a)
        feat_b = model(emb_b)

        loss = criterion(feat_a, feat_b, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            distance = torch.sqrt(torch.sum((feat_a - feat_b) ** 2, dim=1) + 1e-8)
            all_distances.append(distance.cpu())
            all_labels.append(labels.cpu())
    
    """
    TODO: Add metrics calculation
    """
    # Calculate metrics
    all_distances = torch.cat(all_distances)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_distances, all_labels)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, metrics


def validate(model: nn.Module, 
             val_loader: DataLoader, 
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for emb_a, emb_b, labels in tqdm(val_loader, desc="Validating", leave=False):
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            labels = labels.to(device).float()
            
            feat_a = model(emb_a)
            feat_b = model(emb_b)
            
            loss = criterion(feat_a, feat_b, labels)

            total_loss += loss.item()

            distance = torch.sqrt(torch.sum((feat_a - feat_b) ** 2, dim=1) + 1e-8)
            all_distances.append(distance.cpu())
            all_labels.append(labels.cpu())
    
    """
    TODO: Add metrics calculation
    """
    # Calculate metrics
    all_distances = torch.cat(all_distances)
    all_labels = torch.cat(all_labels)

    metrics = calculate_metrics(all_distances, all_labels)
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, metrics


def train_model(train_dataset,
                embedding_dim: int,
                num_epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                val_split: float = 0.15,
                save_dir: str = "checkpoints",
                device: str = None,
                resume_from : str = None):
    """
    Complete training pipeline for fashion compatibility model.
    
    Args:
        train_dataset: Your PairDataset instance
        embedding_dim: Dimension of CLIP embeddings
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data to use for validation
        save_dir: Directory to save model checkpoints
        device: Device to train on ('cuda', 'cpu', or None for auto)
    """
    # Setup
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Split dataset
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    model = FashionCompatibilityModel(embedding_dim=embedding_dim)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    start_epoch = 0
    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'schduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', best_val_loss)
        print(f"Resumed from epoch {start_epoch} with val_loss {best_val_loss:.4f}")
    
    print("\nStarting training...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for key, val in train_metrics.items():
            print(f"  Train {key.capitalize()}: {val:.4f} | Val {key.capitalize()}: {val_metrics[key]:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'embedding_dim': embedding_dim
            }
            torch.save(checkpoint, save_path / 'best_model.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_path}")
    
    return model, history


def load_model(checkpoint_path: str, device: str = None):
    """Load a trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = FashionCompatibilityModel(embedding_dim=checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

# Model hyperparameters
EMBEDDING_DIM = 512  # 512 for CLIP ViT-B/32, 768 for ViT-L/14

# Training hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Load data
    load_dotenv()
    embeddings_dir = os.getenv("CLIP_EMBEDDINGS_DIR", "data/clip_embeddings") 
    outfit_file = Path(os.getenv("COMPATIBILE_OUTFITS_FILE"), "data/fashion_compatibility_predictions.txt")
    output_pickle = Path(os.getenv("PAIRS_PICKLE", "data/compatibility_pairs.pkl"))
    checkpoint_dir = Path("checkpoints/metric_learning")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    embedding_manager = EmbeddingManager(embeddings_dir)
    embeddings, category_index = embedding_manager.load_embeddings()

    pair_manager = PairManager(embeddings, outfit_file, output_pickle)
    pairs = pair_manager.load_pairs()

    # Create dataset
    dataset = PairDataset(pairs, embeddings)
    print(f"Total pairs in dataset: {len(dataset)}")

    print("\nStarting training...")
    model, history = train_model(
        train_dataset=dataset,
        embedding_dim=EMBEDDING_DIM,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_split=VAL_SPLIT,
        save_dir=checkpoint_dir,
        device=DEVICE,
        #resume_from = checkpoint_dir / 'best_model.pt' if (checkpoint_dir / 'best_model.pt').exists() else None
    )

    print("Training Complete!")

    final_train_metrics = history['train_metrics'][-1]
    final_val_metrics = history['val_metrics'][-1]

    final_train_metrics = history['train_metrics'][-1]
    final_val_metrics = history['val_metrics'][-1]
    
    print(f"\nFinal Training Metrics:")
    for key, val in final_train_metrics.items():
        print(f"  {key.capitalize()}: {val:.4f}")
    
    
    print(f"\nFinal Validation Metrics:")
    for key, val in final_val_metrics.items():
        print(f"  {key.capitalize()}: {val:.4f}")
    
    print(f"\nModel saved to: {checkpoint_dir}/best_model.pt")
    print(f"Training history saved to: {checkpoint_dir}/training_history.json")

if __name__ == "__main__":
    main()