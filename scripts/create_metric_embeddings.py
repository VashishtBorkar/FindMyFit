"""
Script to generate metric embeddings from CLIP embeddings using trained model.
This converts CLIP embeddings into the learned metric space where compatible items are close.
"""

from pathlib import Path
import os
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.models.metric_learning.model import FashionCompatibilityModel
from src.models.metric_learning.optuna_search import get_hyperparameters
from src.utils.logging import get_logger, setup_logging


def load_trained_model(checkpoint_path: str, embedding_dim: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FashionCompatibilityModel(
        embedding_dim=embedding_dim,
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, device


def generate_metric_embeddings(clip_embeddings_dir: Path, 
                               metric_embeddings_dir: Path,
                               model: torch.nn.Module,
                               device: torch.device,
                               batch_size: int = 64,
                               force_reload: bool = False):
    """
    Generate metric embeddings from CLIP embeddings.
    
    Args:
        clip_embeddings_dir: Directory containing CLIP embeddings (.npy files)
        metric_embeddings_dir: Directory to save metric embeddings
        model: Trained metric learning model
        device: Device for computation
        batch_size: Batch size for processing
        force_reload: regenerate embeddings even if they exist
    """
    logger = get_logger(__name__)
    
    metric_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    # Process each category
    for category_dir in clip_embeddings_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_metric_dir = metric_embeddings_dir / category_dir.name
        category_metric_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all embedding files in this category
        embedding_files = list(category_dir.glob("*.npy"))
        
        logger.info(f"Processing category: {category_dir.name} ({len(embedding_files)} items)")
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(embedding_files), batch_size), 
                     desc=f"Processing {category_dir.name}"):
            batch_files = embedding_files[i:i + batch_size]
            batch_clip_embeddings = []
            batch_output_paths = []
            batch_valid_indices = []
            
            # Load batch of CLIP embeddings
            for idx, emb_file in enumerate(batch_files):
                metric_emb_file = category_metric_dir / f"{emb_file.stem}.npy"
                
                # Skip if already exists
                if metric_emb_file.exists() and not force_reload:
                    skipped_count += 1
                    continue
                
                try:
                    clip_emb = np.load(emb_file)
                    batch_clip_embeddings.append(clip_emb)
                    batch_output_paths.append(metric_emb_file)
                    batch_valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Failed to load {emb_file}: {e}")
                    continue
            
            # Skip if nothing to process in this batch
            if len(batch_clip_embeddings) == 0:
                continue
            
            # Convert to tensor and generate metric embeddings
            try:
                with torch.no_grad():
                    clip_tensor = torch.tensor(np.array(batch_clip_embeddings), 
                                              dtype=torch.float32).to(device)
                    metric_embeddings = model(clip_tensor)
                    metric_embeddings = metric_embeddings.cpu().numpy()
                
                # Save each metric embedding
                for metric_emb, output_path in zip(metric_embeddings, batch_output_paths):
                    np.save(output_path, metric_emb)
                    processed_count += 1
            
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                continue
    
    return processed_count, skipped_count


def main():
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    # Load environment variables
    load_dotenv()
    
    # Paths
    clip_embeddings_dir = Path("data/clip_embeddings").resolve()
    metric_embeddings_dir = Path("data/metric_embeddings").resolve()
    checkpoint_path = Path("checkpoints/metric_learning/best_model.pt").resolve()
    
    # Validate paths
    if not clip_embeddings_dir.exists():
        raise FileNotFoundError(f"CLIP embeddings directory not found: {clip_embeddings_dir}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    logger.info(f"CLIP embeddings directory: {clip_embeddings_dir}")
    logger.info(f"Metric embeddings directory: {metric_embeddings_dir}")
    logger.info(f"Model checkpoint: {checkpoint_path}")


    # Load model
    logger.info("Loading trained model...")
    model, device = load_trained_model(
        embedding_dim=512,
        checkpoint_path=str(checkpoint_path),
    )
    logger.info(f"Model loaded on {device}")
    
    # Generate metric embeddings
    logger.info("Generating metric embeddings...")
    processed_count, skipped_count = generate_metric_embeddings(
        clip_embeddings_dir=clip_embeddings_dir,
        metric_embeddings_dir=metric_embeddings_dir,
        model=model,
        device=device,
        batch_size=64,
        force_reload=True
    )
    
    logger.info("Finished generating metric embeddings!")
    logger.info(f"Processed: {processed_count} items")
    logger.info(f"Skipped: {skipped_count} items (already existed)")
    logger.info(f"Saved to: {metric_embeddings_dir}")


if __name__ == "__main__":
    main()