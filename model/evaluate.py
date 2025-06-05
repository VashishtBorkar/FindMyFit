# model/evaluate.py

import torch
from torch.utils.data import DataLoader
from scripts.data_pipeline import get_all_pairs
from scripts.utils import split_pairs
from train import EmbeddingPairDataset, CompatibilityModel  # reuse

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for emb1, emb2, labels in dataloader:
            outputs = model(emb1, emb2).squeeze()
            preds = outputs > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    # Load and split data
    all_pairs = get_all_pairs()
    _, _, test_pairs = split_pairs(all_pairs)

    # Load model
    embedding_dim = len(test_pairs[0]["embedding_1"])
    model = CompatibilityModel(embedding_dim)
    model.load_state_dict(torch.load("compatibility_model.pt"))

    # Run evaluation
    test_loader = DataLoader(EmbeddingPairDataset(test_pairs), batch_size=64)
    evaluate(model, test_loader)
