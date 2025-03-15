"""
Common data models for media deduplication.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import hashlib
import json

@dataclass
class MediaFile:
    """Base class for media files (images and videos)."""
    path: Path
    size: int = 0
    content_score: float = 0.0  # Quality score
    hash_id: str = ""  # Unique hash for this file
    
    def __post_init__(self):
        """Initialize size from path if not provided."""
        if self.size == 0 and self.path.exists():
            self.size = self.path.stat().st_size
        
        # Generate a unique hash ID for this file
        if not self.hash_id:
            self.hash_id = hashlib.md5(str(self.path).encode()).hexdigest()

    def get_metadata(self) -> Dict:
        """Get basic metadata without loading the file."""
        return {
            'path': str(self.path),
            'size': self.size,
            'extension': self.path.suffix.lower(),
            'last_modified': self.path.stat().st_mtime
        }
    
    def calculate_content_score(self) -> float:
        """Calculate a quality score. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement calculate_content_score")

@dataclass
class DuplicateGroup:
    """Represents a group of duplicate media files."""
    files: List[MediaFile] = field(default_factory=list)
    similarity_score: float = 0.0
    best_version: Optional[MediaFile] = None
    
    def add_file(self, file: MediaFile):
        """Add a file to the duplicate group."""
        self.files.append(file)
        
    def determine_best_version(self) -> MediaFile:
        """Determine the best version based on content score."""
        if not self.files:
            return None
            
        # Calculate content scores if not already done
        for file in self.files:
            if file.content_score == 0:
                file.calculate_content_score()
                
        # Sort by content score (higher is better)
        sorted_files = sorted(self.files, key=lambda v: v.content_score, reverse=True)
        self.best_version = sorted_files[0]
        return self.best_version
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'similarity_score': self.similarity_score,
            'file_count': len(self.files),
            'best_version': str(self.best_version.path) if self.best_version else None,
            'files': [str(v.path) for v in self.files],
            'total_size': sum(v.size for v in self.files),
            'wasted_space': sum(v.size for v in self.files[1:]) if self.files else 0
        }
