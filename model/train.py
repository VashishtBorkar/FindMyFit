import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from scripts.data_pipeline import get_training_pairs
from scripts.utils import split_pairs

# Dataset class for embedding pairs
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

# Compatibility model
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

# Main training loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get training pairs via the pipeline
    pair_list = get_training_pairs()
    train_pairs, val_pairs, test_pairs = split_pairs(pair_list)

    train_loader = DataLoader(EmbeddingPairDataset(train_pairs), batch_size=64, shuffle=True)
    val_loader = DataLoader(EmbeddingPairDataset(val_pairs), batch_size=64, shuffle=True)

    # Model, loss, optimizer
    embedding_dim = len(pair_list[0]["embedding_1"])
    model = CompatibilityModel(embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float("inf")
    save_path = Path("compatibility_model.pth")
    # Training
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for emb1, emb2, labels in train_loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(emb1, emb2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)

                outputs = model(emb1, emb2).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
