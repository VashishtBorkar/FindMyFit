from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch

from src.models.metric_learning.model import FashionCompatibilityModel
from src.utils.logging import get_logger
from src.models.metric_learning.loss_functions import ContrastiveLoss

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @staticmethod
    def calculate_metrics(distances: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        # Lower distance = compatible (predict 1)
        pred_binary = (distances < threshold).float()
        accuracy = (pred_binary == labels).float().mean().item()
        
        return {'accuracy': accuracy}

    def train_epoch(self, train_loader: DataLoader, optimizer: nn.Module, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        all_distances = []
        all_labels = []
        
        for emb_a, emb_b, labels in tqdm(train_loader, desc="Training", leave=False):
            emb_a = emb_a.to(self.device)
            emb_b = emb_b.to(self.device)
            labels = labels.to(self.device).float()
            
            optimizer.zero_grad()

            # Forward pass
            feat_a = self.model(emb_a)
            feat_b = self.model(emb_b)

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
        metrics = self.calculate_metrics(all_distances, all_labels)
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, metrics
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_distances = []
        all_labels = []
        
        with torch.no_grad():
            for emb_a, emb_b, labels in tqdm(val_loader, desc="Validating", leave=False):
                emb_a = emb_a.to(self.device)
                emb_b = emb_b.to(self.device)
                labels = labels.to(self.device).float()
                
                feat_a = self.model(emb_a)
                feat_b = self.model(emb_b)
                
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

        metrics = self.calculate_metrics(all_distances, all_labels)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def test(self, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_distances = []
        all_labels = []
        
        with torch.no_grad():
            for emb_a, emb_b, labels in tqdm(test_loader, desc="Testing", leave=False):
                emb_a = emb_a.to(self.device)
                emb_b = emb_b.to(self.device)
                labels = labels.to(self.device).float()
                
                feat_a = self.model(emb_a)
                feat_b = self.model(emb_b)
                
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

        metrics = self.calculate_metrics(all_distances, all_labels)
        
        avg_loss = total_loss / len(test_loader)
        return avg_loss, metrics
        