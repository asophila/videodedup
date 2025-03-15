"""
Cache management utilities for media deduplication.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of analysis results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager with optional custom cache directory."""
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "mediadedup"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, cache_name: str) -> Path:
        """Get the path for a cache file."""
        return self.cache_dir / f"{cache_name}.json"
    
    def save(self, data: Any, cache_name: str) -> None:
        """Save data to cache file."""
        cache_file = self._get_cache_path(cache_name)
        
        try:
            # Handle special types that need conversion for JSON serialization
            serializable_data = self._prepare_for_serialization(data)
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_data, f)
                
            logger.debug(f"Saved cache: {cache_name}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_name}: {e}")
    
    def load(self, cache_name: str) -> Optional[Any]:
        """Load data from cache file."""
        cache_file = self._get_cache_path(cache_name)
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Convert back from JSON-serializable format if needed
            return self._process_loaded_data(data)
            
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_name}: {e}")
            return None
    
    def clear(self, cache_name: Optional[str] = None) -> None:
        """Clear cache files. If no name provided, clear all cache."""
        try:
            if cache_name:
                cache_file = self._get_cache_path(cache_name)
                if cache_file.exists():
                    cache_file.unlink()
                    logger.debug(f"Cleared cache: {cache_name}")
            else:
                # Clear all cache files
                for cache_file in self.cache_dir.glob('*.json'):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                logger.debug("Cleared all cache files")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
    
    def _prepare_for_serialization(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]
        elif isinstance(data, Path):
            return str(data)
        elif hasattr(data, 'to_dict'):  # For custom classes with to_dict method
            return data.to_dict()
        elif hasattr(data, '__dict__'):  # For other custom classes
            return self._prepare_for_serialization(data.__dict__)
        return data
    
    def _process_loaded_data(self, data: Any) -> Any:
        """Process loaded data to restore special types."""
        if isinstance(data, dict):
            # Check if this is a serialized Path
            if len(data) == 1 and "path" in data:
                return Path(data["path"])
            return {k: self._process_loaded_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_loaded_data(item) for item in data]
        return data
