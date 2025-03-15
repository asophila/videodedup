"""
Common CLI argument handling for media deduplication.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Sequence
from . import utils

class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter to improve the display of argument choices."""
    
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        
        # Format choices with proper spacing
        if action.choices:
            args_string = '{' + ', '.join(str(c) for c in action.choices) + '}'
            
        return ', '.join(action.option_strings) + ' ' + args_string

class BaseArgumentParser:
    """Base argument parser with common options."""
    
    def __init__(self, description: str):
        """Initialize parser with description."""
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=CustomHelpFormatter
        )
        self._add_common_arguments()
    
    def _add_common_arguments(self):
        """Add arguments common to all media deduplication tools."""
        # Input directories
        self.parser.add_argument(
            'directories', 
            nargs='+', 
            type=str, 
            help='Directories to scan for media files'
        )
        
        # Analysis options
        analysis_group = self.parser.add_argument_group('Analysis Options')
        analysis_group.add_argument(
            '--similarity-threshold', 
            type=float, 
            default=85.0,
            help='Percentage threshold for similarity detection'
        )
        analysis_group.add_argument(
            '--hash-algorithm', 
            type=str, 
            choices=['phash', 'dhash', 'whash', 'average_hash'],
            default='phash',
            help='Perceptual hash algorithm to use. Options: phash, dhash, whash, average_hash (default: phash)'
        )
        analysis_group.add_argument(
            '--recursive', 
            action='store_true', 
            default=True,
            help='Scan directories recursively'
        )
        analysis_group.add_argument(
            '--no-recursive', 
            dest='recursive', 
            action='store_false',
            help='Do not scan directories recursively'
        )
        
        # Action options
        action_group = self.parser.add_argument_group('Action Options')
        action_group.add_argument(
            '--action', 
            type=str, 
            choices=['report', 'interactive', 'move', 'symlink', 'hardlink', 'delete', 'script'],
            default='report',
            help='Action to take for duplicates. Options:\n' +
                 '  report      - Display duplicate files without modifying them\n' +
                 '  interactive - Interactively choose which duplicates to handle\n' +
                 '  move        - Move duplicate files to target directory\n' +
                 '  symlink     - Replace duplicates with symbolic links\n' +
                 '  hardlink    - Replace duplicates with hard links\n' +
                 '  delete      - Delete duplicate files (requires --force-delete)\n' +
                 '  script      - Generate a script to handle duplicates\n' +
                 '(default: report)'
        )
        action_group.add_argument(
            '--target-dir', 
            type=str, 
            help='Target directory for move/symlink/hardlink actions'
        )
        action_group.add_argument(
            '--force-delete', 
            action='store_true', 
            help='Force deletion of duplicates (required for delete action)'
        )
        action_group.add_argument(
            '--script-type', 
            type=str, 
            choices=['bash', 'powershell', 'python'],
            default='bash',
            help='Type of script to generate. Options: bash, powershell, python (default: bash)'
        )
        
        # Output options
        output_group = self.parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--output-format', 
            type=str, 
            choices=['text', 'json', 'csv', 'html'],
            default='text',
            help='Output format for report. Options: text, json, csv, html (default: text)'
        )
        output_group.add_argument(
            '--html-report-dir',
            type=str,
            help='Directory to store HTML report with thumbnails (only used with --output-format html)'
        )
        output_group.add_argument(
            '--output-file', 
            type=str, 
            help='Output file path (default: stdout)'
        )
        
        # Misc options
        misc_group = self.parser.add_argument_group('Miscellaneous')
        misc_group.add_argument(
            '-v', '--verbose',
            action='count',
            default=0,
            help='Increase verbosity level (-v for detailed, -vv for debug)'
        )
        misc_group.add_argument(
            '--version', 
            action='version',
            version=f'mediadedup {utils.VERSION}'
        )
        misc_group.add_argument(
            '--clear-cache', 
            action='store_true', 
            help='Clear cache before running'
        )
    
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        args = self.parser.parse_args()
        
        # Convert directories to Path objects
        args.directories = [Path(d).resolve() for d in args.directories]
        
        # Convert target_dir to Path if provided
        if args.target_dir:
            args.target_dir = Path(args.target_dir).resolve()
        
        # Convert output_file to Path if provided
        if args.output_file:
            args.output_file = Path(args.output_file).resolve()
        
        # Convert html_report_dir to Path if provided
        if args.html_report_dir:
            args.html_report_dir = Path(args.html_report_dir).resolve()
        
        return args

class VideoArgumentParser(BaseArgumentParser):
    """Argument parser for video deduplication."""
    
    def __init__(self):
        super().__init__(
            description="""videodedup - Intelligent Video Deduplication Tool

This tool helps you find and manage duplicate videos, even if they have different names,
resolutions, or encodings. It uses perceptual hashing and content analysis to identify
duplicates with high accuracy."""
        )
        self._add_video_arguments()
    
    def _add_video_arguments(self):
        """Add video-specific arguments."""
        video_group = self.parser.add_argument_group('Video Options')
        video_group.add_argument(
            '--duration-threshold', 
            type=float, 
            default=1.0,
            help='Percentage threshold for duration matching'
        )

class ImageArgumentParser(BaseArgumentParser):
    """Argument parser for image deduplication."""
    
    def __init__(self):
        super().__init__(
            description="""imagededup - Intelligent Image Deduplication Tool

This tool helps you find and manage duplicate images, even if they have different names,
resolutions, or encodings. It uses perceptual hashing and content analysis to identify
duplicates with high accuracy."""
        )
        self._add_image_arguments()
    
    def _add_image_arguments(self):
        """Add image-specific arguments."""
        image_group = self.parser.add_argument_group('Image Options')
        image_group.add_argument(
            '--dimension-threshold', 
            type=float, 
            default=0.1,
            help='Percentage threshold for dimension matching'
        )
