import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity


class FashionCompatibilityModel(nn.Module):
    """
    Metric learning model that projects fashion items into an embedding space
    where compatible items are close together and incompatible items are far apart.
    """    
    def __init__(self, embedding_dim: int, hidden_dims: list = [512, 256, 128], output_dim: int = 128):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Compatibility prediction layers
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final projection to metric space
        layers.append(nn.Linear(prev_dim, 1))
        
        self.embedding_net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Project input to metric embedding space.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
        
        Returns:
            Projected embeddings (batch_size, output_dim)
        """
        return self.embedding_net(x)
    
    def embed_pair(self, emb_a, emb_b):
        """
        Embed a pair of items.
        
        Args:
            emb_a: First item embeddings (batch_size, embedding_dim)
            emb_b: Second item embeddings (batch_size, embedding_dim)
        
        Returns:
            Tuple of embedded items
        """
        feat_a = self.forward(emb_a)
        feat_b = self.forward(emb_b)
        return feat_a, feat_b

    def compute_distance(self, emb_a, emb_b, distance_type='euclidean'):
        """
        Compute distance between two items in metric space.
        
        Args:
            emb_a: First item embeddings
            emb_b: Second item embeddings
            distance_type: 'euclidean' or 'cosine'
        
        Returns:
            Distances (batch_size,)
        """
        feat_a, feat_b = self.embed_pair(emb_a, emb_b)
        
        if distance_type == 'euclidean':
            distance = torch.sqrt(torch.sum((feat_a - feat_b) ** 2, dim=1) + 1e-8)
        elif distance_type == 'cosine':
            distance = 1 - cosine_similarity(feat_a, feat_b)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
        
        return distance
    
    def predict_compatibility(self, emb_a, emb_b, threshold=0.5):
        """
        Predict if two items are compatible based on distance.
        
        Args:
            emb_a: First item embeddings
            emb_b: Second item embeddings
            threshold: Distance threshold for compatibility
        
        Returns:
            compatibility_score: Score between 0 and 1 (higher = more compatible)
            is_compatible: Boolean prediction
        """
        distance = self.compute_distance(emb_a, emb_b, distance_type='euclidean')
        
        # Convert distance to similarity score (0-1 range)
        # Lower distance = higher compatibility
        compatibility_score = torch.exp(-distance)
        
        is_compatible = distance < threshold
        
        return compatibility_score, is_compatible
    