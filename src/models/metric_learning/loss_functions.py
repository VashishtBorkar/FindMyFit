import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

class ContrastiveLoss(nn.Module):

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for negative pairs
            distance_type: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, emb_a, emb_b, labels):
        """
        Args:
            emb_a: First item metric embeddings (batch_size, output_dim)
            emb_b: Second item metric embeddings (batch_size, output_dim)
            labels: Binary labels (1=compatible, 0=incompatible)
        """
        # Compute euclidean distance in metric space
        distance = torch.sqrt(torch.sum((emb_a - emb_b) ** 2, dim=1) + 1e-8)
        
        # Compatible pairs: minimize distance
        loss_positive = labels * torch.pow(distance, 2)
        
        # Incompatible pairs: maximize distance up to margin
        loss_negative = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )
        
        return torch.mean(loss_positive + loss_negative)
    

    