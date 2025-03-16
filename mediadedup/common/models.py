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
            'files': [{
                'path': str(v.path),
                'size': v.size,
                'content_score': v.content_score,
                'hash_id': v.hash_id,
                'metadata': v.get_metadata()
            } for v in self.files],
            'best_version_hash': self.best_version.hash_id if self.best_version else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DuplicateGroup':
        """Create a DuplicateGroup instance from dictionary data."""
        group = cls()
        group.similarity_score = data['similarity_score']
        
        # Restore files
        for file_data in data['files']:
            # Determine file type from metadata
            path = Path(file_data['path'])
            metadata = file_data['metadata']
            
            # Import here to avoid circular imports
            from ..video.models import VideoFile
            
            # Create appropriate file type
            if path.suffix.lower() in {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}:
                file = VideoFile.from_dict(file_data)
            else:
                file = MediaFile(path)
                file.size = file_data['size']
                file.content_score = file_data['content_score']
                file.hash_id = file_data['hash_id']
            
            group.files.append(file)
        
        # Restore best version by matching hash_id
        if data['best_version_hash'] and group.files:
            for file in group.files:
                if file.hash_id == data['best_version_hash']:
                    group.best_version = file
                    break
        
        return group
