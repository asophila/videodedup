"""
Main entry point for video deduplication.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

from .common.utils import setup_logging
from .video.models import VideoFile
from .common.cli import VideoArgumentParser
from .common.actions import handle_duplicates
from .common.cache import CacheManager
from .video import FFMPEG_AVAILABLE, get_available_hw_accelerators
from .video.analysis import find_video_files, analyze_videos

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = VideoArgumentParser()
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Check dependencies and capabilities
    if not FFMPEG_AVAILABLE:
        logger.error("ffmpeg and ffprobe are required but not found in the system PATH.")
        logger.error("Please install them and make sure they are available in your PATH.")
        return 1
    
    # Check hardware acceleration support
    hw_accelerators = get_available_hw_accelerators()
    if hw_accelerators:
        logger.info("Available hardware accelerators:")
        for accel in hw_accelerators:
            logger.info(f"  - {accel}")
    else:
        logger.warning("No hardware acceleration methods available")
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Clear cache if requested
    if args.clear_cache:
        cache_manager.clear()
    
    # Start the analysis process
    logger.info("Starting video deduplication...")
    
    # Convert directories to Path objects
    directories = [Path(d).resolve() for d in args.directories]
    
    # Find all video files
    video_files = find_video_files(directories, args.recursive)
    
    if not video_files:
        logger.error("No video files found in the specified directories")
        return 1
    
    # Create a cache key based on files and parameters
    cache_key = _create_cache_key(video_files, args)
    
    # Try to load from cache
    duplicate_groups = None
    if not args.clear_cache:
        logger.info("Checking cache for previous analysis results...")
        cached_data = cache_manager.load(cache_key)
        if cached_data:
            logger.info("Found cached analysis results")
            duplicate_groups = _restore_from_cache(cached_data)
            if duplicate_groups is None:
                logger.info("Cache was invalid or files were modified, will re-analyze videos")
    
    # If no cache or cache cleared, perform analysis
    if duplicate_groups is None:
        logger.info("Analyzing videos...")
        duplicate_groups = analyze_videos(video_files, args)
        # Cache the results
        logger.info("Caching analysis results...")
        cache_manager.save(_prepare_for_cache(duplicate_groups), cache_key)
    else:
        logger.info("Using cached analysis results")
    
    # Handle duplicates according to the specified action
    handle_duplicates(duplicate_groups, args)
    
    # Clear cache if delete action was performed
    if args.action == 'delete':
        logger.info("Clearing cache after delete operation...")
        cache_manager.clear()
        # Also clear any other cached results for these directories
        for cache_file in cache_manager.cache_dir.glob('analysis_*.json'):
            try:
                cache_file.unlink()
                logger.debug(f"Cleared cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to clear cache file {cache_file}: {e}")
    
    return 0

def _create_cache_key(video_files: List[VideoFile], args) -> str:
    """Create a cache key based on files and parameters."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Sort paths to ensure consistent order
    paths = sorted(str(f.path) for f in video_files)
    
    # Create a hash of paths and relevant parameters
    import hashlib
    m = hashlib.sha256()
    
    # Add analysis parameters first for better logging
    params = {
        'duration_threshold': args.duration_threshold,
        'similarity_threshold': args.similarity_threshold,
        'hash_algorithm': args.hash_algorithm,
        'skip_crc': getattr(args, 'skip_crc', False),
        'use_ramdisk': getattr(args, 'use_ramdisk', False),
        'recursive': args.recursive,
        'file_count': len(paths)
    }
    param_str = json.dumps(params, sort_keys=True)
    m.update(param_str.encode())
    
    # Add file paths
    paths_str = json.dumps(paths, sort_keys=True)
    m.update(paths_str.encode())
    
    cache_key = f"analysis_{m.hexdigest()}"
    logger.debug(f"Cache key parameters: {param_str}")
    logger.debug(f"Cache key file count: {len(paths)}")
    logger.debug(f"Generated cache key: {cache_key}")
    return cache_key

def _prepare_for_cache(duplicate_groups: List['DuplicateGroup']) -> List[Dict]:
    """Convert duplicate groups to a cacheable format."""
    return [group.to_dict() for group in duplicate_groups]

def _is_valid_cache_format(cached_data: List[Dict]) -> bool:
    """Check if the cached data has the expected format."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        for group_data in cached_data:
            # Check required group fields
            if not all(k in group_data for k in ['similarity_score', 'files', 'best_version_hash']):
                logger.debug("Missing required group fields")
                return False
            
            # Check each file in the group
            for file_data in group_data['files']:
                # Check required file fields
                required_fields = ['path', 'size', 'content_score', 'hash_id', 'metadata']
                if not all(k in file_data for k in required_fields):
                    logger.debug(f"Missing required file fields: {[f for f in required_fields if f not in file_data]}")
                    return False
                
                # Check video-specific fields for video files
                path = Path(file_data['path'])
                if path.suffix.lower() in {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}:
                    video_fields = ['duration', 'resolution', 'bitrate', 'frame_rate']
                    if not all(k in file_data for k in video_fields):
                        logger.debug(f"Missing video fields: {[f for f in video_fields if f not in file_data]}")
                        return False
                
                # Verify the file still exists
                if not Path(file_data['path']).exists():
                    logger.debug(f"File no longer exists: {file_data['path']}")
                    return False
                
                # Verify file size hasn't changed
                current_size = Path(file_data['path']).stat().st_size
                if current_size != file_data['size']:
                    logger.debug(f"File size changed: {file_data['path']} ({file_data['size']} -> {current_size})")
                    return False
        
        return True
    except (KeyError, TypeError, AttributeError) as e:
        logger.debug(f"Cache validation error: {e}")
        return False

def _restore_from_cache(cached_data: List[Dict]) -> Optional[List['DuplicateGroup']]:
    """Restore duplicate groups from cached data."""
    from .common.models import DuplicateGroup
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate cache format
    if not _is_valid_cache_format(cached_data):
        logger.warning("Cache format is invalid or incompatible, will re-analyze videos")
        return None
    
    try:
        groups = []
        for group_data in cached_data:
            group = DuplicateGroup.from_dict(group_data)
            groups.append(group)
        return groups
    except Exception as e:
        logger.warning(f"Error restoring from cache: {e}")
        return None

if __name__ == "__main__":
    sys.exit(main())
