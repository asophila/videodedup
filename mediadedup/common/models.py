"""
Common data models for media deduplication.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import zlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class MediaFile:
    """Base class for media files (images and videos)."""
    path: Path
    size: int = 0
    content_score: float = 0.0  # Quality score
    hash_id: str = ""  # Unique hash for this file
    original_path: Optional[Path] = None  # Original path when using RAM disk
    crc32: int = 0  # CRC32 hash of file content
    sha256: str = ""  # SHA-256 hash of file content
    
    def __post_init__(self):
        """Initialize size and hashes from path if not provided."""
        if self.size == 0 and self.path.exists():
            self.size = self.path.stat().st_size
        
        # Store original path if not set
        if not self.original_path:
            self.original_path = self.path
        
        # Generate a unique hash ID from the original path
        if not self.hash_id:
            self.hash_id = hashlib.md5(str(self.original_path).encode()).hexdigest()
        
        # Calculate CRC and SHA-256 if not provided
        if self.crc32 == 0 or not self.sha256:
            self.crc32, self.sha256 = self.calculate_crc()
    
    def calculate_crc(self, chunk_size: int = 65536) -> Tuple[int, str]:
        """Calculate CRC32 and SHA-256 hash of the file content.
        
        Args:
            chunk_size: Size of chunks to read from file (default: 64KB)
            
        Returns:
            Tuple of (CRC32 value, SHA-256 hex digest)
        """
        crc32 = 0
        sha256 = hashlib.sha256()
        
        try:
            with open(self.path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    crc32 = zlib.crc32(chunk, crc32)
                    sha256.update(chunk)
            
            return crc32 & 0xFFFFFFFF, sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating CRC for {self.path}: {e}")
            return 0, ''

    def get_metadata(self) -> Dict:
        """Get basic metadata without loading the file."""
        # Always use original path for metadata
        return {
            'path': str(self.original_path),
            'size': self.size,
            'extension': self.original_path.suffix.lower(),
            'last_modified': self.original_path.stat().st_mtime if self.original_path.exists() else 0
        }
    
    def get_display_path(self) -> Path:
        """Get the path to display in reports and use for operations."""
        return self.original_path or self.path
    
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
                'path': str(v.get_display_path()),  # Use display path for serialization
                'size': v.size,
                'content_score': v.content_score,
                'hash_id': v.hash_id,
                'crc32': v.crc32,
                'sha256': v.sha256,
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
                file.crc32 = file_data.get('crc32', 0)
                file.sha256 = file_data.get('sha256', '')
            
            group.files.append(file)
        
        # Restore best version by matching hash_id
        if data['best_version_hash'] and group.files:
            for file in group.files:
                if file.hash_id == data['best_version_hash']:
                    group.best_version = file
                    break
        
        return group
