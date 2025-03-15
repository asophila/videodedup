"""
Main entry point for image deduplication.
"""

import sys
from pathlib import Path

from .common.utils import setup_logging
from .common.cli import ImageArgumentParser
from .common.actions import handle_duplicates
from .common.cache import CacheManager
from .image import IMAGEHASH_AVAILABLE, PIL_AVAILABLE
from .image.analysis import find_image_files, analyze_images

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = ImageArgumentParser()
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Check dependencies
    if not IMAGEHASH_AVAILABLE:
        logger.error("imagehash is required but not installed")
        logger.error("Install with: pip install imagehash")
        return 1
    
    if not PIL_AVAILABLE:
        logger.error("Pillow (PIL) is required but not installed")
        logger.error("Install with: pip install Pillow")
        return 1
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Clear cache if requested
    if args.clear_cache:
        cache_manager.clear()
    
    # Start the analysis process
    logger.info("Starting image deduplication...")
    
    # Convert directories to Path objects
    directories = [Path(d).resolve() for d in args.directories]
    
    # Find all image files
    image_files = find_image_files(directories, args.recursive)
    
    if not image_files:
        logger.error("No image files found in the specified directories")
        return 1
    
    # Find duplicates
    duplicate_groups = analyze_images(image_files, args)
    
    # Handle duplicates according to the specified action
    handle_duplicates(duplicate_groups, args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
