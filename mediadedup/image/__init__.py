"""
Image deduplication functionality.
"""

from ..common.utils import IMAGE_EXTENSIONS

# Check if required dependencies are available
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.error("imagehash is required but not installed")
    logger.error("Install with: pip install imagehash")
    sys.exit(1)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.error("Pillow (PIL) is required but not installed")
    logger.error("Install with: pip install Pillow")
    sys.exit(1)

# Export constants
__all__ = [
    'IMAGE_EXTENSIONS',
    'IMAGEHASH_AVAILABLE',
    'PIL_AVAILABLE',
]
