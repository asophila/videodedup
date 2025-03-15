"""
Common utilities for media deduplication.
"""

import logging
from pathlib import Path
from typing import List, Set

# Add custom VERBOSE level between INFO and DEBUG
VERBOSE = 15  # Between INFO (20) and DEBUG (10)
logging.addLevelName(VERBOSE, "VERBOSE")

def setup_logging(verbose_level: int = 0) -> logging.Logger:
    """Set up logging with configurable verbosity."""
    # Set up basic logging format
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('mediadedup')
    
    # Set log level based on verbosity
    if verbose_level == 0:
        log_level = logging.INFO
    elif verbose_level == 1:
        log_level = VERBOSE
    else:
        log_level = logging.DEBUG
    
    logger.setLevel(log_level)
    
    if verbose_level >= 1:
        logger.log(VERBOSE, "Verbose logging enabled")
        if verbose_level >= 2:
            logger.debug("Debug logging enabled")
    
    return logger

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def find_files(directories: List[Path], 
               extensions: Set[str], 
               recursive: bool = True,
               logger: logging.Logger = None) -> List[Path]:
    """Find files with specified extensions in given directories."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    found_files = []
    
    for directory in directories:
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue
            
        logger.info(f"Scanning directory: {directory}")
        
        # Process all files in the directory
        if recursive:
            file_iterator = directory.rglob('*')
        else:
            file_iterator = directory.glob('*')
            
        for file_path in file_iterator:
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                found_files.append(file_path)
    
    logger.info(f"Found {len(found_files)} files")
    return found_files

def get_file_metadata(file_path: Path) -> dict:
    """Get basic file metadata."""
    stat = file_path.stat()
    return {
        'path': str(file_path),
        'size': stat.st_size,
        'extension': file_path.suffix.lower(),
        'last_modified': stat.st_mtime,
        'created': stat.st_ctime
    }

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def is_same_file(path1: Path, path2: Path) -> bool:
    """Check if two paths point to the same file."""
    try:
        return path1.resolve() == path2.resolve()
    except Exception:
        return False

# Constants
VERSION = "1.0.0"

# Common file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.tiff', '.bmp',
    '.heic', '.heif', '.jfif', '.jp2', '.j2k'
}

VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
    '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.mxf', '.ts', 
    '.m2ts', '.vob', '.ogv', '.mts', '.m2v', '.divx', '.rmvb', '.rm'
}
