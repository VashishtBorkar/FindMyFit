import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.data_loader import load_raw_data
from utils.data_preprocessor import create_pairs


class EmbeddingPairDataset(Dataset):
    def __init__(self, pair_list):
        self.pairs = pair_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        emb1 = torch.tensor(item["embedding_1"], dtype=torch.float32)
        emb2 = torch.tensor(item["embedding_2"], dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.float32)
        return emb1, emb2, label


class CompatibilityModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        x = torch.cat([emb1, emb2], dim=1)
        return self.fc(x)


if __name__ == "__main__":
    # Load raw dataframe (must include precomputed embeddings)
    df = load_raw_data(
        images_dir="data/raw/images",
        segments_dir="data/raw/segm",
        captions_path="data/raw/captions.json",
        shapes_path="data/raw/labels/shape_anno_all.txt",
        fabrics_path="data/raw/labels/fabric_ann.txt",
        patterns_path="data/raw/labels/pattern_ann.txt"
    )

    # Generate compatibility pairs
    pair_list = create_pairs(df)

    # Dataset and DataLoader
    dataset = EmbeddingPairDataset(pair_list)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model
    embedding_dim = len(pair_list[0]["embedding_1"])
    model = CompatibilityModel(embedding_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for emb1, emb2, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(emb1, emb2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
