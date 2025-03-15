"""
Video deduplication functionality.
"""

from ..common.utils import VIDEO_EXTENSIONS

# Check if ffmpeg and ffprobe are installed
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Check if ffmpeg and ffprobe are installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    logger.error("ffmpeg and ffprobe are required but not found in the system PATH.")
    logger.error("Please install them and make sure they are available in your PATH.")
    sys.exit(1)

def get_available_hw_accelerators():
    """Get list of available hardware accelerators."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"], 
            capture_output=True, text=True
        )
        # Parse the output to get list of accelerators
        accelerators = []
        for line in result.stdout.splitlines():
            line = line.strip().lower()
            if line and not line.startswith('hardware'):  # Skip header line
                accelerators.append(line)
        return accelerators
    except Exception:
        return []

def check_quicksync_available():
    """Check if Intel QuickSync is available."""
    try:
        accelerators = get_available_hw_accelerators()
        return "qsv" in accelerators
    except Exception:
        return False

# Export constants and functions
__all__ = [
    'VIDEO_EXTENSIONS',
    'FFMPEG_AVAILABLE',
    'get_available_hw_accelerators',
    'check_quicksync_available',
]
