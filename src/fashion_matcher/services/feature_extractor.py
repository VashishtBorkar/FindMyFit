import numpy as np
from typing import Optional
import logging

from ..core.interfaces import FeatureExtractor
from ..utils.logging import get_logger

class CLIPFeatureExtractor(FeatureExtractor):
    """CLIP-based feature extractor for fashion items."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.device = None
        self.feature_dim = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model."""
        try:
            import clip
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            if self.device == "cuda":
                self.model = self.model.half()
            
            self.feature_dim = 512  # ViT-B/32
            self.logger.info(f"Initialized CLIP {self.model_name} on {self.device}")
            
        except ImportError:
            self.logger.warning("CLIP not available, falling back to simple features")
            self.simple_extractor = SimpleFeatureExtractor()
            self.feature_dim = self.simple_extractor.get_feature_dimension()
        except Exception as e:
            self.logger.error(f"Error initializing CLIP: {str(e)}")
            raise
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using CLIP."""
        if self.model is None:
            return self.simple_extractor.extract_features(image)
        
        try:
            import torch
            from PIL import Image
            
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = features.squeeze(0).cpu().numpy()
                features = features / np.linalg.norm(features)  # Normalize
            
            self.logger.debug(f"Extracted CLIP features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting CLIP features: {str(e)}")
            raise
    
    def get_feature_dimension(self) -> int:
        return self.feature_dim

class ResNetFeatureExtractor(FeatureExtractor):
    """ResNet-based feature extractor using pre-trained model."""
    def __init__(self, model_name: str = 'resnet50', use_pretrained: bool = True):
        self.logger = get_logger(__name__)
        self.model = None
        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ResNet model."""
        try:
            import torch
            import torchvision.models as models
            import torch.nn as nn
            
            self.torch = torch
            
            # Load pre-trained ResNet
            if self.model_name == 'resnet50':
                self.model = models.resnet50(pretrained=self.use_pretrained)
            elif self.model_name == 'resnet18':
                self.model = models.resnet18(pretrained=self.use_pretrained)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Remove the final classification layer to get features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            
            # Get feature dimension
            with self.torch.no_grad():
                dummy_input = self.torch.randn(1, 3, 224, 224)
                dummy_output = self.model(dummy_input)
                self.feature_dim = dummy_output.numel()
            
            self.logger.info(f"Initialized {self.model_name} with feature dim: {self.feature_dim}")
            
        except ImportError:
            self.logger.warning("PyTorch not available, falling back to simple features")
            # Fallback to simple extractor
            self.simple_extractor = SimpleFeatureExtractor()
            self.feature_dim = self.simple_extractor.get_feature_dimension()
        except Exception as e:
            self.logger.error(f"Error initializing ResNet model: {str(e)}")
            raise
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using ResNet."""
        if self.model is None:
            # Fallback to simple extractor
            return self.simple_extractor.extract_features(image)
        
        try:
            # Preprocess image for ResNet
            image_tensor = self._preprocess_for_resnet(image)
            
            # Extract features
            with self.torch.no_grad():
                features = self.model(image_tensor)
                features = features.flatten().numpy()
            
            self.logger.debug(f"Extracted ResNet features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting ResNet features: {str(e)}")
            raise
    
    def _preprocess_for_resnet(self, image: np.ndarray) -> 'torch.Tensor':
        """Preprocess image for ResNet input."""
        # Resize to 224x224
        if image.shape[:2] != (224, 224):
            import cv2
            image = cv2.resize(image, (224, 224))
        
        # Convert to tensor and normalize
        image_tensor = self.torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Normalize using ImageNet stats
        mean = self.torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = self.torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor
    
    def get_feature_dimension(self) -> int:
        """Return the dimension of extracted features."""
        return self.feature_dim

class SimpleFeatureExtractor(FeatureExtractor):
    """Simple feature extractor using basic image statistics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.feature_dim = None  # Will be set when extracting features

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the image into a 1D vector to use as features.
        Expects image shape: (H, W, C), e.g., (224, 224, 3)
        """
        try:
            feature_vector = image.flatten()

            self.feature_dim = len(feature_vector)

            return feature_vector

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise

    def get_feature_dimension(self) -> int:
        if self.feature_dim is None:
            raise ValueError("Feature dimension not set. Call extract_features first.")
        return self.feature_dim
