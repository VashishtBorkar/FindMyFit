import cv2
import numpy as np
from pathlib import Path
from typing import List
from sklearn.cluster import KMeans
import logging

from ..core.interfaces import ImageProcessor
from ..core.models import Color
from ..utils.logging import get_logger

class PILImageProcessor(ImageProcessor):
    """PIL-based image processor implementation."""
    
    def __init__(self, target_size: tuple = (224, 224)):
        self.target_size = target_size
        self.logger = get_logger(__name__)
        
        try:
            from PIL import Image, ImageFilter
            self.Image = Image
            self.ImageFilter = ImageFilter
        except ImportError:
            raise ImportError("PIL/Pillow is required for PILImageProcessor")
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image using PIL."""
        try:
            with self.Image.open(image_path) as img:
                img = img.convert('RGB')
                image_array = np.array(img)
            
            self.logger.debug(f"Successfully loaded image with PIL: {image_path}")
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image using PIL."""
        pil_image = self.Image.fromarray((image * 255).astype(np.uint8))
        resized = pil_image.resize(target_size, self.Image.Resampling.LANCZOS)
        return np.array(resized)


class OpenCVImageProcessor(ImageProcessor):
    """
    OpenCV-based image processor implementation for more advanced preprocessing
    used for user uploaded images in the future
    """
    
    def __init__(self, target_size: tuple = (224, 224)):
        self.target_size = target_size
        self.logger = get_logger(__name__)
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess an image using OpenCV."""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic preprocessing
            # image = self._preprocess_image(image)
            
            self.logger.debug(f"Successfully loaded image: {image_path}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to the image."""
        # Remove background (simple approach - can be enhanced)
        image = self._remove_background_simple(image)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _remove_background_simple(self, image: np.ndarray) -> np.ndarray:
        """Simple background removal using edge detection and morphology."""
        # This is a basic implementation - can be enhanced with more sophisticated methods
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the clothing item)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Apply mask to original image
            result = cv2.bitwise_and(image, image, mask=mask)
            return result
        
        return image
    
    def resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image to target dimensions."""
        try:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            self.logger.debug(f"Resized image to {target_size}")
            return resized
        except Exception as e:
            self.logger.error(f"Error resizing image: {str(e)}")
            raise
