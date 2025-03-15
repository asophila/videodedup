"""
Main entry point for video deduplication.
"""

import sys
from pathlib import Path

from .common.utils import setup_logging
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
    
    # Find duplicates
    duplicate_groups = analyze_videos(video_files, args)
    
    # Handle duplicates according to the specified action
    handle_duplicates(duplicate_groups, args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
