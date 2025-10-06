"""
Logging utilities for the Fashion Matcher system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = 'DEBUG', log_file: Optional[Path] = None):
    """
    Setup logging configuration for the entire system.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file path for logging output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# src/utils/helpers.py
"""
Helper utilities for the Fashion Matcher system.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np


def generate_item_id(image_path: Path) -> str:
    """
    Generate a unique ID for a clothing item based on image path and content.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Unique item ID
    """
    # Use file path and modification time to generate ID
    path_str = str(image_path.absolute())
    try:
        mtime = image_path.stat().st_mtime
        id_string = f"{path_str}_{mtime}"
    except OSError:
        id_string = path_str
    
    # Generate hash
    hash_object = hashlib.md5(id_string.encode())
    return hash_object.hexdigest()[:16]  # 16 character ID


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON, handling numpy types.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON string
    """
    def json_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'value'):  # For enums
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(obj, default=json_serializer, indent=2)


def validate_image_file(image_path: Path) -> bool:
    """
    Validate that a file is a supported image format.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        True if valid image, False otherwise
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not image_path.exists():
        return False
    
    if image_path.suffix.lower() not in supported_extensions:
        return False
    
    # Try to open with PIL to verify it's a valid image
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def calculate_color_distance(color1: tuple, color2: tuple) -> float:
    """
    Calculate Euclidean distance between two RGB colors.
    
    Args:
        color1: RGB tuple (r, g, b)
        color2: RGB tuple (r, g, b)
    
    Returns:
        Color distance (0-441, lower is more similar)
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    return np.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)


def normalize_features(features: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize feature vector using specified method.
    
    Args:
        features: Feature vector to normalize
        method: Normalization method ('l2', 'l1', 'max', 'minmax')
    
    Returns:
        Normalized feature vector
    """
    if method == 'l2':
        norm = np.linalg.norm(features, ord=2)
        return features / norm if norm > 0 else features
    elif method == 'l1':
        norm = np.linalg.norm(features, ord=1)
        return features / norm if norm > 0 else features
    elif method == 'max':
        max_val = np.max(np.abs(features))
        return features / max_val if max_val > 0 else features
    elif method == 'minmax':
        min_val, max_val = np.min(features), np.max(features)
        if max_val > min_val:
            return (features - min_val) / (max_val - min_val)
        return features
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_color_palette_image(colors: List[tuple], size: tuple = (400, 100)) -> np.ndarray:
    """
    Create a color palette image from a list of RGB colors.
    
    Args:
        colors: List of RGB tuples
        size: Image size (width, height)
    
    Returns:
        RGB image array
    """
    if not colors:
        # Return black image if no colors
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    width, height = size
    stripe_width = width // len(colors)
    
    # Create image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        start_x = i * stripe_width
        end_x = start_x + stripe_width if i < len(colors) - 1 else width
        image[:, start_x:end_x] = color
    
    return image


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    size = size_bytes
    i = 0
    
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"


def ensure_directory(directory: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    
    Returns:
        The directory path
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory