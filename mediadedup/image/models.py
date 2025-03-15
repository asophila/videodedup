"""
Image-specific data models.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import imagehash
from PIL import Image
import numpy as np

from ..common.models import MediaFile

logger = logging.getLogger(__name__)

class ImageFile(MediaFile):
    """Represents an image file with its metadata and analysis results."""
    def __init__(self, path: Path):
        super().__init__(path)
        self.dimensions: Tuple[int, int] = (0, 0)
        self.format: str = ""
        self.mode: str = ""  # Color mode (RGB, RGBA, etc.)
        self.perceptual_hash: Optional[Any] = None
        self.color_hash: Optional[Any] = None  # For color-based similarity
    
    def load_complete_metadata(self) -> bool:
        """Load full image metadata using PIL."""
        try:
            with Image.open(self.path) as img:
                self.dimensions = img.size
                self.format = img.format
                self.mode = img.mode
                return True
        except Exception as e:
            logger.warning(f"Error loading metadata for {self.path}: {e}")
            return False
    
    def calculate_content_score(self) -> float:
        """Calculate a quality score based on resolution and file size."""
        # Basic scoring based on resolution and file size
        resolution_score = self.dimensions[0] * self.dimensions[1] / 1000000  # Normalized by millions of pixels
        size_score = self.size / (1024 * 1024)  # Normalized by MB
        
        # Combine scores with weights
        self.content_score = (resolution_score * 0.7) + (size_score * 0.3)
        return self.content_score
    
    def calculate_hashes(self, hash_algorithm: str = 'phash') -> Tuple[Any, Any]:
        """Calculate perceptual and color hashes for the image."""
        try:
            with Image.open(self.path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate perceptual hash
                if hash_algorithm == 'phash':
                    self.perceptual_hash = imagehash.phash(img)
                elif hash_algorithm == 'dhash':
                    self.perceptual_hash = imagehash.dhash(img)
                elif hash_algorithm == 'whash':
                    self.perceptual_hash = imagehash.whash(img)
                elif hash_algorithm == 'average_hash':
                    self.perceptual_hash = imagehash.average_hash(img)
                else:
                    self.perceptual_hash = imagehash.phash(img)
                
                # Calculate color hash (average color values)
                img_array = np.array(img.resize((8, 8)))  # Downscale for color analysis
                self.color_hash = imagehash.average_hash(Image.fromarray(img_array))
                
                return self.perceptual_hash, self.color_hash
                
        except Exception as e:
            logger.warning(f"Error calculating hashes for {self.path}: {e}")
            return None, None
    
    def get_metadata(self) -> Dict:
        """Get complete metadata including image-specific information."""
        metadata = super().get_metadata()
        metadata.update({
            'dimensions': f"{self.dimensions[0]}x{self.dimensions[1]}",
            'format': self.format,
            'mode': self.mode
        })
        return metadata
