import numpy as np
from typing import Optional
import logging

from ..core.interfaces import FeatureExtractor
from ..utils.logging import get_logger


class SimpleFeatureExtractor(FeatureExtractor):
    """Simple feature extractor using basic image statistics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.feature_dim = 64  # Fixed dimension for this simple extractor
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract simple statistical features from image."""
        try:
            features = []
            
            # Color histogram features (RGB channels)
            for channel in range(3):
                hist, _ = np.histogram(image[:, :, channel], bins=8, range=(0, 1))
                hist = hist / np.sum(hist)  # Normalize
                features.extend(hist)
            
            # Texture features using local standard deviation
            gray = np.mean(image, axis=2)
            
            # Calculate local standard deviations in 4x4 blocks
            h, w = gray.shape
            block_size = 4
            texture_features = []
            
            for i in range(0, h - block_size + 1, block_size):
                for j in range(0, w - block_size + 1, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    texture_features.append(np.std(block))
            
            # Pad or truncate to fixed size
            if len(texture_features) > 40:
                texture_features = texture_features[:40]
            else:
                texture_features.extend([0.0] * (40 - len(texture_features)))
            
            features.extend(texture_features)
            
            # Ensure fixed dimension
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                features.extend([0.0] * (self.feature_dim - len(features)))
            
            feature_vector = np.array(features, dtype=np.float32)
            self.logger.debug(f"Extracted {len(feature_vector)} features")
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def get_feature_dimension(self) -> int:
        """Return the dimension of extracted features."""
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
        # Resize to 224x224 if needed
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


class CLIPFeatureExtractor(FeatureExtractor):
    """CLIP-based feature extractor for fashion items."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.logger = get_logger(__name__)
        self.model = None
        self.preprocess = None
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model."""
        try:
            import clip
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.model_name, device=device)
            self.device = device
            
            # CLIP image features are 512-dimensional for ViT-B/32
            self.feature_dim = 512
            
            self.logger.info(f"Initialized CLIP {self.model_name} on {device}")
            
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
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Preprocess for CLIP
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = features.cpu().numpy().flatten()
            
            self.logger.debug(f"Extracted CLIP features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting CLIP features: {str(e)}")
            raise
    
    def get_feature_dimension(self) -> int:
        """Return the dimension of extracted features."""
        return self.feature_dim