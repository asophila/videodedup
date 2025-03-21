"""
Common utilities and shared functionality for media deduplication.
"""

from .models import MediaFile, DuplicateGroup
from .utils import setup_logging, format_size, find_files, VERSION
from .actions import handle_duplicates
from .cli import BaseArgumentParser, VideoArgumentParser, ImageArgumentParser

__all__ = [
    'MediaFile',
    'DuplicateGroup',
    'setup_logging',
    'format_size',
    'find_files',
    'VERSION',
    'handle_duplicates',
    'BaseArgumentParser',
    'VideoArgumentParser',
    'ImageArgumentParser',
]
