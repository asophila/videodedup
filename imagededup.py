#!/usr/bin/env python3
"""
imagededup - Intelligent Image Deduplication Tool

A content-aware image deduplicator that finds duplicate images
even if they have different names, resolutions, or encodings.
"""

import os
import sys
import argparse
import hashlib
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict

# Required imports
import imagehash
from PIL import Image
import numpy as np
import gc  # For memory management

# Optional imports - gracefully handle if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Check if imagehash is available
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

# Configure logging
# Add custom VERBOSE level between INFO and DEBUG
logging.VERBOSE = 15  # Between INFO (20) and DEBUG (10)
logging.addLevelName(logging.VERBOSE, "VERBOSE")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('imagededup')

# Constants
VERSION = "1.0.0"
CACHE_DIR = Path.home() / ".cache" / "imagededup"
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.tiff', '.bmp',
    '.heic', '.heif', '.jfif', '.jp2', '.j2k'
}

# ======= Data Models =======

@dataclass
class ImageFile:
    """Represents an image file with its metadata and analysis results."""
    path: Path
    size: int = 0
    dimensions: Tuple[int, int] = (0, 0)
    format: str = ""
    mode: str = ""  # Color mode (RGB, RGBA, etc.)
    perceptual_hash: Optional[Any] = None
    color_hash: Optional[Any] = None  # For color-based similarity
    content_score: float = 0.0  # Quality score
    hash_id: str = ""  # Unique hash for this image
    
    def __post_init__(self):
        """Initialize size from path if not provided."""
        if self.size == 0 and self.path.exists():
            self.size = self.path.stat().st_size
        
        # Generate a unique hash ID for this file
        if not self.hash_id:
            self.hash_id = hashlib.md5(str(self.path).encode()).hexdigest()

    def get_metadata(self) -> Dict:
        """Get basic metadata without loading the image."""
        return {
            'path': str(self.path),
            'size': self.size,
            'extension': self.path.suffix.lower(),
            'last_modified': datetime.fromtimestamp(self.path.stat().st_mtime).isoformat()
        }
    
    def load_complete_metadata(self) -> bool:
        """Load full image metadata using PIL."""
        try:
            with Image.open(self.path) as img:
                self.dimensions = img.size
                self.format = img.format
                self.mode = img.mode
                return True
        except Exception as e:
            logger.warning(f"Error loading metadata for {self.path}: {e}")
            return False
    
    def calculate_content_score(self) -> float:
        """Calculate a quality score based on resolution and file size."""
        # Basic scoring based on resolution and file size
        resolution_score = self.dimensions[0] * self.dimensions[1] / 1000000  # Normalized by millions of pixels
        size_score = self.size / (1024 * 1024)  # Normalized by MB
        
        # Combine scores with weights
        self.content_score = (resolution_score * 0.7) + (size_score * 0.3)
        return self.content_score
    
    def calculate_hashes(self, hash_algorithm: str = 'phash') -> Tuple[Any, Any]:
        """Calculate perceptual and color hashes for the image."""
        if not IMAGEHASH_AVAILABLE:
            logger.warning("ImageHash not available, can't calculate hashes")
            return None, None
            
        # Try to load from cache first
        cache_key = f"image_hashes_{self.hash_id}"
        cached_hashes = load_cache(cache_key)
        if cached_hashes:
            logger.debug(f"Using cached hashes for {self.path}")
            self.perceptual_hash = cached_hashes.get('perceptual')
            self.color_hash = cached_hashes.get('color')
            return self.perceptual_hash, self.color_hash
        
        try:
            with Image.open(self.path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate perceptual hash
                if hash_algorithm == 'phash':
                    self.perceptual_hash = imagehash.phash(img)
                elif hash_algorithm == 'dhash':
                    self.perceptual_hash = imagehash.dhash(img)
                elif hash_algorithm == 'whash':
                    self.perceptual_hash = imagehash.whash(img)
                elif hash_algorithm == 'average_hash':
                    self.perceptual_hash = imagehash.average_hash(img)
                else:
                    self.perceptual_hash = imagehash.phash(img)
                
                # Calculate color hash (average color values)
                img_array = np.array(img.resize((8, 8)))  # Downscale for color analysis
                self.color_hash = imagehash.average_hash(Image.fromarray(img_array))
                
                # Cache the results
                save_cache({
                    'perceptual': self.perceptual_hash,
                    'color': self.color_hash
                }, cache_key)
                
                return self.perceptual_hash, self.color_hash
                
        except Exception as e:
            logger.warning(f"Error calculating hashes for {self.path}: {e}")
            return None, None

@dataclass
class DuplicateGroup:
    """Represents a group of duplicate images."""
    images: List[ImageFile] = field(default_factory=list)
    similarity_score: float = 0.0
    best_version: Optional[ImageFile] = None
    
    def add_image(self, image: ImageFile):
        """Add an image to the duplicate group."""
        self.images.append(image)
        
    def determine_best_version(self, 
                             prefer_resolution: bool = True,
                             prefer_size: bool = True) -> ImageFile:
        """Determine the best version based on configurable criteria."""
        if not self.images:
            return None
            
        # Calculate content scores if not already done
        for image in self.images:
            if image.content_score == 0:
                image.calculate_content_score()
                
        # Sort by content score (higher is better)
        sorted_images = sorted(self.images, key=lambda v: v.content_score, reverse=True)
        self.best_version = sorted_images[0]
        return self.best_version
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'similarity_score': self.similarity_score,
            'image_count': len(self.images),
            'best_version': str(self.best_version.path) if self.best_version else None,
            'images': [str(v.path) for v in self.images],
            'total_size': sum(v.size for v in self.images),
            'wasted_space': sum(v.size for v in self.images[1:]) if self.images else 0
        }


# ======= Core Analysis Functions =======

def find_image_files(directories: List[Path], recursive: bool = True) -> List[ImageFile]:
    """Find all image files in the specified directories."""
    image_files = []
    
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
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(ImageFile(path=file_path))
    
    logger.info(f"Found {len(image_files)} image files")
    return image_files

def group_by_dimensions(images: List[ImageFile], threshold: float = 0.1) -> Dict[str, List[ImageFile]]:
    """Group images by similar dimensions with a percentage threshold."""
    dimension_groups = {}
    
    logger.log(logging.VERBOSE, f"Grouping {len(images)} images by dimensions (threshold: {threshold}%)")
    
    # First, load metadata for all images
    with ProcessPoolExecutor() as executor:
        for i, image in enumerate(executor.map(load_image_metadata, images)):
            if i % 10 == 0:  # Log progress every 10 images
                logger.debug(f"Processing image {i+1}/{len(images)}")
            if image and image.dimensions != (0, 0):
                # Create a dimension key with a precision relative to the threshold
                width, height = image.dimensions
                key_width = int(width / (width * threshold))
                key_height = int(height / (height * threshold))
                dimension_key = f"{key_width}x{key_height}"
                
                if dimension_key not in dimension_groups:
                    dimension_groups[dimension_key] = []
                    
                dimension_groups[dimension_key].append(image)
    
    # Filter out groups with only one image (no potential duplicates)
    filtered_groups = {k: v for k, v in dimension_groups.items() if len(v) > 1}
    
    logger.info(f"Grouped images into {len(filtered_groups)} dimension groups")
    return filtered_groups

def load_image_metadata(image: ImageFile) -> ImageFile:
    """Load image metadata - this is a wrapper for multiprocessing."""
    try:
        logger.debug(f"Loading metadata for {image.path}")
        image.load_complete_metadata()
        logger.log(logging.VERBOSE, f"Loaded metadata: {image.path} ({image.dimensions[0]}x{image.dimensions[1]})")
        return image
    except Exception as e:
        logger.warning(f"Error loading metadata for {image.path}: {e}")
        return None

def _process_image_hashes(image: ImageFile, hash_algorithm: str) -> ImageFile:
    """Process a single image to calculate hashes."""
    try:
        logger.debug(f"Processing hashes for {image.path}")
        image.calculate_hashes(hash_algorithm)
        return image
    except Exception as e:
        logger.warning(f"Error calculating hashes for {image.path}: {e}")
        return image

def process_images(images: List[ImageFile], hash_algorithm: str = 'phash') -> List[ImageFile]:
    """Process images to calculate their hashes."""
    # Create a function with preset arguments
    process_func = partial(_process_image_hashes, hash_algorithm=hash_algorithm)
    
    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        processed_images = list(executor.map(process_func, images))
    
    return processed_images

def compare_images(image1: ImageFile, image2: ImageFile, threshold: int = 10) -> float:
    """Compare two images based on their hashes and return a similarity score."""
    if not image1.perceptual_hash or not image2.perceptual_hash:
        return 0.0
    
    # Calculate perceptual hash distance
    perceptual_distance = image1.perceptual_hash - image2.perceptual_hash
    
    # Calculate color hash distance
    color_distance = image1.color_hash - image2.color_hash if (image1.color_hash and image2.color_hash) else 0
    
    # Combine distances with weights (perceptual hash is more important)
    avg_distance = (perceptual_distance * 0.7) + (color_distance * 0.3)
    
    # Convert distance to similarity score (0-100)
    # Lower distance means higher similarity
    similarity = max(0, 100 - (avg_distance * (100 / threshold)))
    
    return similarity

def find_duplicates(images: List[ImageFile], similarity_threshold: float = 80.0) -> List[DuplicateGroup]:
    """Find duplicate images based on similarity threshold."""
    duplicate_groups = []
    processed = set()
    
    # Compare each image with every other image
    for i, image1 in enumerate(images):
        if image1.hash_id in processed:
            continue
            
        current_group = DuplicateGroup()
        current_group.add_image(image1)
        
        for j, image2 in enumerate(images):
            if i == j or image2.hash_id in processed:
                continue
                
            similarity = compare_images(image1, image2)
            if similarity >= similarity_threshold:
                current_group.add_image(image2)
                processed.add(image2.hash_id)
                
        if len(current_group.images) > 1:
            current_group.similarity_score = similarity
            current_group.determine_best_version()
            duplicate_groups.append(current_group)
            
        processed.add(image1.hash_id)
    
    return duplicate_groups

def analyze_images(image_files: List[ImageFile], args: argparse.Namespace) -> List[DuplicateGroup]:
    """Main analysis pipeline to find duplicate images."""
    logger.log(logging.VERBOSE, "Starting image analysis pipeline")
    logger.log(logging.VERBOSE, f"Analysis parameters: dimension_threshold={args.dimension_threshold}%, "
               f"similarity_threshold={args.similarity_threshold}%, hash_algorithm={args.hash_algorithm}")
    
    # Step 1: Load metadata
    logger.info("Loading image metadata...")
    images_with_metadata = []
    
    if TQDM_AVAILABLE:
        for image in tqdm(image_files, desc="Loading metadata"):
            if image.load_complete_metadata():
                images_with_metadata.append(image)
    else:
        with ProcessPoolExecutor() as executor:
            images_with_metadata = list(filter(None, executor.map(load_image_metadata, image_files)))
    
    logger.info(f"Successfully loaded metadata for {len(images_with_metadata)} images")
    
    # Step 2: Group by dimensions
    logger.debug("Grouping images by dimensions...")
    dimension_groups = group_by_dimensions(images_with_metadata, args.dimension_threshold)
    
    # Step 3: Process each dimension group
    all_duplicate_groups = []
    total_comparisons = 0
    
    for dimensions, images in dimension_groups.items():
        if len(images) < 2:
            logger.debug(f"Skipping dimension group {dimensions} (only {len(images)} image)")
            continue
            
        logger.info(f"Processing {len(images)} images with dimensions ~{dimensions}")
        logger.log(logging.VERBOSE, f"Calculating hashes for {len(images)} images...")
        
        # Calculate hashes for this group
        images_with_hashes = process_images(
            images,
            hash_algorithm=args.hash_algorithm
        )
        
        # Find duplicates within this group
        logger.debug(f"Comparing images in dimension group {dimensions}...")
        duplicate_groups = find_duplicates(
            images_with_hashes,
            similarity_threshold=args.similarity_threshold
        )
        
        if duplicate_groups:
            logger.log(logging.VERBOSE, f"Found {len(duplicate_groups)} duplicate groups in dimension group {dimensions}")
            for group in duplicate_groups:
                logger.debug(f"Duplicate group: {len(group.images)} images, {group.similarity_score:.1f}% similarity")
        
        all_duplicate_groups.extend(duplicate_groups)
        total_comparisons += (len(images) * (len(images) - 1)) // 2
    
    # Sort duplicate groups by wasted space (descending)
    all_duplicate_groups.sort(
        key=lambda g: sum(v.size for v in g.images[1:]),
        reverse=True
    )
    
    logger.log(logging.VERBOSE, f"Analysis complete: {len(all_duplicate_groups)} duplicate groups found")
    logger.log(logging.VERBOSE, f"Total image comparisons performed: {total_comparisons}")
    
    return all_duplicate_groups


# ======= Action Functions =======

def handle_duplicates(duplicate_groups: List[DuplicateGroup], args: argparse.Namespace) -> None:
    """Handle duplicate images according to the specified action."""
    if not duplicate_groups:
        logger.info("No duplicates found.")
        return
        
    # Calculate total statistics
    total_images = sum(len(group.images) for group in duplicate_groups)
    total_duplicates = sum(len(group.images) - 1 for group in duplicate_groups)
    total_wasted_space = sum(
        sum(v.size for v in group.images[1:]) for group in duplicate_groups
    )
    
    logger.info(f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicates")
    logger.info(f"Total wasted space: {total_wasted_space / (1024**2):.2f} MB")
    
    # Create output directory if needed
    output_dir = None
    if args.action in ['move', 'symlink', 'hardlink'] and args.target_dir:
        output_dir = Path(args.target_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle according to action
    if args.action == 'report':
        generate_report(duplicate_groups, args.output_format, args.output_file, args.html_report_dir)
    
    elif args.action == 'interactive':
        interactive_deduplication(duplicate_groups)
    
    elif args.action == 'move':
        if not output_dir:
            logger.error("Target directory required for 'move' action")
            return
        move_duplicates(duplicate_groups, output_dir)
    
    elif args.action == 'symlink':
        if not output_dir:
            logger.error("Target directory required for 'symlink' action")
            return
        create_symlinks(duplicate_groups, output_dir)
    
    elif args.action == 'hardlink':
        if not output_dir:
            logger.error("Target directory required for 'hardlink' action")
            return
        create_hardlinks(duplicate_groups, output_dir)
    
    elif args.action == 'delete':
        if args.force_delete:
            delete_duplicates(duplicate_groups)
        else:
            logger.warning("Delete action requires --force-delete flag for safety")
    
    elif args.action == 'script':
        generate_action_script(duplicate_groups, args.script_type, args.output_file)


def generate_report(duplicate_groups: List[DuplicateGroup], 
                   format_type: str = 'text', 
                   output_file: Optional[str] = None,
                   html_report_dir: Optional[str] = None) -> None:
    """Generate a report of duplicate images."""
    if format_type == 'html' and html_report_dir:
        # Create HTML report with thumbnails
        report_dir = Path(html_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create thumbnails directory
        thumbnails_dir = report_dir / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)
        
        # Generate thumbnails for each image
        thumbnails = {}
        
        for group_idx, group in enumerate(duplicate_groups):
            for image_idx, image in enumerate(group.images):
                thumbnail_path = thumbnails_dir / f"group_{group_idx}_image_{image_idx}.jpg"
                
                try:
                    with Image.open(image.path) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Create thumbnail
                        img.thumbnail((320, 320))
                        img.save(thumbnail_path, "JPEG")
                        thumbnails[(group_idx, image_idx)] = str(thumbnail_path.relative_to(report_dir))
                except Exception as e:
                    logger.warning(f"Failed to create thumbnail: {e}")
        
        # Generate HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>imagededup - Duplicate Image Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .group { border: 1px solid #ccc; margin-bottom: 20px; padding: 15px; border-radius: 5px; }",
            "        .group-header { background-color: #f0f0f0; padding: 10px; margin-bottom: 15px; border-radius: 3px; }",
            "        .image { margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px dashed #eee; }",
            "        .best-version { background-color: #e6ffe6; padding: 10px; border-radius: 3px; }",
            "        .thumbnail { margin-right: 10px; text-align: center; }",
            "        .thumbnail img { border: 1px solid #ddd; border-radius: 3px; max-width: 320px; }",
            "        .metadata { margin-top: 10px; font-size: 0.9em; color: #555; }",
            "        .summary { background-color: #f8f8f8; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>imagededup - Duplicate Image Report</h1>",
            f"    <p>Generated on: {datetime.now().isoformat()}</p>"
        ]
        
        # Add summary statistics
        total_images = sum(len(group.images) for group in duplicate_groups)
        total_duplicates = sum(len(group.images) - 1 for group in duplicate_groups)
        total_original_size = sum(sum(v.size for v in group.images) for group in duplicate_groups)
        total_optimized_size = sum(group.best_version.size for group in duplicate_groups)
        total_savings = total_original_size - total_optimized_size
        
        html_content.extend([
            "    <div class='summary'>",
            "        <h2>Summary</h2>",
            f"        <p>Total duplicate groups: {len(duplicate_groups)}</p>",
            f"        <p>Total images analyzed: {total_images}</p>",
            f"        <p>Total duplicate images: {total_duplicates}</p>",
            f"        <p>Total original size: {total_original_size / (1024**2):.2f} MB</p>",
            f"        <p>Size after deduplication: {total_optimized_size / (1024**2):.2f} MB</p>",
            f"        <p>Potential space savings: {total_savings / (1024**2):.2f} MB ({total_savings/total_original_size*100:.1f}%)</p>",
            "    </div>"
        ])
        
        # Add each duplicate group
        for group_idx, group in enumerate(duplicate_groups):
            html_content.extend([
                f"    <div class='group' id='group-{group_idx}'>",
                f"        <div class='group-header'>",
                f"            <h2>Group {group_idx + 1} - Similarity: {group.similarity_score:.1f}%</h2>",
                f"        </div>"
            ])
            
            # Add best version first
            best_image = group.best_version
            best_idx = group.images.index(best_image)
            
            html_content.extend([
                "        <div class='image best-version'>",
                f"            <h3>Best Version: {best_image.path.name}</h3>",
                "            <div class='thumbnail'>"
            ])
            
            # Add thumbnail for best version
            if (group_idx, best_idx) in thumbnails:
                html_content.extend([
                    f"                <img src='{thumbnails[(group_idx, best_idx)]}' alt='Best version'>",
                ])
            
            html_content.extend([
                "            </div>",
                "            <div class='metadata'>",
                f"                <p>Path: {best_image.path}</p>",
                f"                <p>Size: {best_image.size / (1024**2):.2f} MB</p>",
                f"                <p>Dimensions: {best_image.dimensions[0]}x{best_image.dimensions[1]}</p>",
                f"                <p>Format: {best_image.format}</p>",
                "            </div>",
                "        </div>"
            ])
            
            # Add duplicates
            for image_idx, image in enumerate(group.images):
                if image != best_image:
                    html_content.extend([
                        "        <div class='image'>",
                        f"            <h3>Duplicate {image_idx + 1}: {image.path.name}</h3>",
                        "            <div class='thumbnail'>"
                    ])
                    
                    # Add thumbnail for this duplicate
                    if (group_idx, image_idx) in thumbnails:
                        html_content.extend([
                            f"                <img src='{thumbnails[(group_idx, image_idx)]}' alt='Duplicate'>",
                        ])
                    
                    html_content.extend([
                        "            </div>",
                        "            <div class='metadata'>",
                        f"                <p>Path: {image.path}</p>",
                        f"                <p>Size: {image.size / (1024**2):.2f} MB</p>",
                        f"                <p>Dimensions: {image.dimensions[0]}x{image.dimensions[1]}</p>",
                        f"                <p>Format: {image.format}</p>",
                        "            </div>",
                        "        </div>"
                    ])
            
            html_content.append("    </div>")
        
        # Close HTML
        html_content.extend([
            "</body>",
            "</html>"
        ])
        
        # Write HTML file
        html_path = report_dir / "report.html"
        with open(html_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        logger.info(f"HTML report generated at {html_path}")
        return
        
    elif format_type == 'json':
        # Convert to JSON
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_groups': len(duplicate_groups),
            'total_duplicates': sum(len(group.images) - 1 for group in duplicate_groups),
            'total_wasted_space': sum(
                sum(v.size for v in group.images[1:]) for group in duplicate_groups
            ),
            'duplicate_groups': [group.to_dict() for group in duplicate_groups]
        }
        
        # Output to file or stdout
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"Report saved to {output_file}")
        else:
            print(json.dumps(json_data, indent=2))
    
    elif format_type == 'csv':
        # Generate CSV content
        csv_lines = ['group_id,similarity,file_path,is_best,size,dimensions']
        
        for i, group in enumerate(duplicate_groups):
            for image in group.images:
                is_best = 'yes' if image == group.best_version else 'no'
                dims = f"{image.dimensions[0]}x{image.dimensions[1]}"
                line = f"{i},{group.similarity_score:.1f},\"{image.path}\",{is_best},{image.size},{dims}"
                csv_lines.append(line)
        
        # Output to file or stdout
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(csv_lines))
            logger.info(f"Report saved to {output_file}")
        else:
            print('\n'.join(csv_lines))
    
    else:  # Default to text format
        # Generate human-readable text report
        report_lines = [
            "=== Image Deduplication Report ===",
            f"Generated on: {datetime.now().isoformat()}",
            f"Total duplicate groups: {len(duplicate_groups)}",
            f"Total duplicate images: {sum(len(group.images) - 1 for group in duplicate_groups)}",
            f"Total wasted space: {sum(sum(v.size for v in group.images[1:]) for group in duplicate_groups) / (1024**2):.2f} MB",
            "\n=== Duplicate Groups ==="
        ]
        
        for i, group in enumerate(duplicate_groups):
            report_lines.append(f"\nGroup {i+1} - Similarity: {group.similarity_score:.1f}%")
            report_lines.append(f"Best Version: {group.best_version.path}")
            report_lines.append(f"Dimensions: {group.best_version.dimensions[0]}x{group.best_version.dimensions[1]}")
            report_lines.append(f"Size: {group.best_version.size / (1024**2):.2f} MB")
            report_lines.append("\nDuplicates:")
            
            for image in group.images:
                if image != group.best_version:
                    report_lines.append(f"- {image.path}")
                    report_lines.append(f"  Dimensions: {image.dimensions[0]}x{image.dimensions[1]}")
                    report_lines.append(f"  Size: {image.size / (1024**2):.2f} MB")
        
        # Output to file or stdout
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"Report saved to {output_file}")
        else:
            print('\n'.join(report_lines))


def interactive_deduplication(duplicate_groups: List[DuplicateGroup]) -> None:
    """Interactive CLI interface for handling duplicates."""
    if not duplicate_groups:
        print("No duplicates found.")
        return
        
    print(f"\nFound {len(duplicate_groups)} duplicate groups")
    
    # Process each group
    for i, group in enumerate(duplicate_groups):
        print(f"\n=== Group {i+1}/{len(duplicate_groups)} ===")
        print(f"Similarity: {group.similarity_score:.1f}%")
        
        # Display images with numbering
        images = group.images
        best_image = group.best_version
        
        print("\n[KEEP] Best version:")
        print(f"* {best_image.path}")
        print(f"  Size: {best_image.size / (1024**2):.2f} MB")
        print(f"  Dimensions: {best_image.dimensions[0]}x{best_image.dimensions[1]}")
        
        print("\n[DUPLICATES]")
        for j, image in enumerate(images):
            if image != best_image:
                print(f"{j+1}. {image.path}")
                print(f"   Size: {image.size / (1024**2):.2f} MB")
                print(f"   Dimensions: {image.dimensions[0]}x{image.dimensions[1]}")
        
        # Ask for action
        while True:
            print("\nActions:")
            print("k - Keep all (skip this group)")
            print("d - Delete all duplicates")
            print("s - Select different image to keep")
            print("n - Next group")
            print("q - Quit")
            
            choice = input("\nEnter action: ").strip().lower()
            
            if choice == 'k':
                print("Keeping all images in this group")
                break
                
            elif choice == 'd':
                confirm = input("Are you sure you want to delete all duplicates? (y/n): ")
                if confirm.lower() == 'y':
                    for image in images:
                        if image != best_image:
                            try:
                                print(f"Deleting {image.path}...")
                                image.path.unlink()
                            except Exception as e:
                                print(f"Error deleting {image.path}: {e}")
                    print("Duplicates deleted")
                    break
                else:
                    print("Deletion cancelled")
                    
            elif choice == 's':
                print("\nSelect image to keep:")
                for j, image in enumerate(images):
                    print(f"{j+1}. {image.path}")
                
                try:
                    selection = int(input("\nEnter number: "))
                    if 1 <= selection <= len(images):
                        best_image = images[selection-1]
                        print(f"Selected: {best_image.path}")
                        
                        confirm = input("Delete all other images? (y/n): ")
                        if confirm.lower() == 'y':
                            for image in images:
                                if image != best_image:
                                    try:
                                        print(f"Deleting {image.path}...")
                                        image.path.unlink()
                                    except Exception as e:
                                        print(f"Error deleting {image.path}: {e}")
                            print("Duplicates deleted")
                            break
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == 'n':
                print("Moving to next group")
                break
                
            elif choice == 'q':
                print("Quitting interactive mode")
                return
                
            else:
                print("Invalid choice, please try again")


def move_duplicates(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Move duplicate images to a target directory."""
    for i, group in enumerate(duplicate_groups):
        # Create a subdirectory for each group
        group_dir = target_dir / f"group_{i+1}"
        group_dir.mkdir(exist_ok=True)
        
        # Move duplicates, keep the best version in place
        best_image = group.best_version
        for image in group.images:
            if image != best_image:
                try:
                    target_path = group_dir / image.path.name
                    # Ensure unique name in target directory
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = group_dir / f"{stem}_{image.hash_id[:8]}{suffix}"
                    
                    # Move the file
                    shutil.move(str(image.path), str(target_path))
                    logger.info(f"Moved: {image.path} -> {target_path}")
                except Exception as e:
                    logger.error(f"Error moving {image.path}: {e}")


def create_symlinks(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Create symbolic links for duplicates."""
    for i, group in enumerate(duplicate_groups):
        best_image = group.best_version
        
        # Create links for duplicates
        for image in group.images:
            if image != best_image:
                try:
                    # Create target path
                    target_path = target_dir / image.path.name
                    # Ensure unique name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = target_dir / f"{stem}_{image.hash_id[:8]}{suffix}"
                    
                    # Create symbolic link
                    target_path.symlink_to(image.path.absolute())
                    logger.info(f"Created symlink: {target_path} -> {image.path}")
                except Exception as e:
                    logger.error(f"Error creating symlink for {image.path}: {e}")


def create_hardlinks(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Create hard links for duplicates."""
    for i, group in enumerate(duplicate_groups):
        best_image = group.best_version
        
        # Create hard links for duplicates
        for image in group.images:
            if image != best_image:
                try:
                    # Create hard link path
                    target_path = target_dir / image.path.name
                    # Ensure unique name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = target_dir / f"{stem}_{image.hash_id[:8]}{suffix}"
                    
                    # Create hard link
                    os.link(str(image.path.absolute()), str(target_path))
                    logger.info(f"Created hardlink: {target_path} -> {image.path}")
                except Exception as e:
                    logger.error(f"Error creating hardlink for {image.path}: {e}")


def delete_duplicates(duplicate_groups: List[DuplicateGroup]) -> None:
    """Delete duplicate images, keeping the best version."""
    total_deleted = 0
    total_freed = 0
    
    for group in duplicate_groups:
        best_image = group.best_version
        
        for image in group.images:
            if image != best_image:
                try:
                    logger.info(f"Deleting: {image.path}")
                    image.path.unlink()
                    total_deleted += 1
                    total_freed += image.size
                except Exception as e:
                    logger.error(f"Error deleting {image.path}: {e}")
    
    logger.info(f"Deleted {total_deleted} duplicate images")
    logger.info(f"Freed {total_freed / (1024**2):.2f} MB")


def generate_action_script(duplicate_groups: List[DuplicateGroup], 
                          script_type: str = 'bash',
                          output_file: Optional[str] = None) -> None:
    """Generate a script to handle duplicates."""
    if script_type == 'bash':
        # Generate bash script
        script_lines = [
            "#!/bin/bash",
            "# Generated by imagededup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate images",
            "# Review carefully before executing!",
            "",
            "# Execute with: bash script.sh",
            "# To execute dry-run (no actual deletion): bash script.sh --dry-run",
            "",
            "DRY_RUN=false",
            "if [ \"$1\" = \"--dry-run\" ]; then",
            "    DRY_RUN=true",
            "    echo \"Running in dry-run mode (no files will be deleted)\"",
            "fi",
            "",
            "echo \"Starting duplicate removal...\"",
            ""
        ]
        
        for i, group in enumerate(duplicate_groups):
            best_image = group.best_version
            
            script_lines.append(f"echo \"Group {i+1}/{len(duplicate_groups)}\"")
            script_lines.append(f"echo \"Keeping: {best_image.path}\"")
            
            for image in group.images:
                if image != best_image:
                    script_lines.append("if [ \"$DRY_RUN\" = \"true\" ]; then")
                    script_lines.append(f"    echo \"Would delete: {image.path}\"")
                    script_lines.append("else")
                    script_lines.append(f"    echo \"Deleting: {image.path}\"")
                    script_lines.append(f"    rm \"{image.path}\"")
                    script_lines.append("fi")
            
            script_lines.append("")
        
        script_lines.append("echo \"Finished!\"")
        
    elif script_type == 'powershell':
        # Generate PowerShell script
        script_lines = [
            "# Generated by imagededup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate images",
            "# Review carefully before executing!",
            "",
            "# Execute with: ./script.ps1",
            "# To execute dry-run (no actual deletion): ./script.ps1 -DryRun",
            "",
            "param(",
            "    [switch]$DryRun",
            ")",
            "",
            "if ($DryRun) {",
            "    Write-Host \"Running in dry-run mode (no files will be deleted)\" -ForegroundColor Yellow",
            "}",
            "",
            "Write-Host \"Starting duplicate removal...\" -ForegroundColor Green",
            ""
        ]
        
        for i, group in enumerate(duplicate_groups):
            best_image = group.best_version
            
            script_lines.append(f"Write-Host \"Group {i+1}/{len(duplicate_groups)}\" -ForegroundColor Cyan")
            script_lines.append(f"Write-Host \"Keeping: {best_image.path}\" -ForegroundColor Green")
            
            for image in group.images:
                if image != best_image:
                    script_lines.append("if ($DryRun) {")
                    script_lines.append(f"    Write-Host \"Would delete: {image.path}\" -ForegroundColor Yellow")
                    script_lines.append("} else {")
                    script_lines.append(f"    Write-Host \"Deleting: {image.path}\" -ForegroundColor Red")
                    script_lines.append(f"    Remove-Item -Path \"{image.path}\" -Force")
                    script_lines.append("}")
            
            script_lines.append("")
        
        script_lines.append("Write-Host \"Finished!\" -ForegroundColor Green")
        
    else:  # Python script
        # Generate Python script
        script_lines = [
            "#!/usr/bin/env python3",
            "# Generated by imagededup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate images",
            "# Review carefully before executing!",
            "",
            "# Execute with: python script.py",
            "# To execute dry-run (no actual deletion): python script.py --dry-run",
            "",
            "import os",
            "import sys",
            "import argparse",
            "",
            "def main():",
            "    parser = argparse.ArgumentParser(description='Delete duplicate images')",
            "    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no deletion)')",
            "    args = parser.parse_args()",
            "",
            "    print('Starting duplicate removal...')",
            ""
        ]
        
        for i, group in enumerate(duplicate_groups):
            best_image = group.best_version
            
            script_lines.append(f"    print(f'Group {i+1}/{len(duplicate_groups)}')")
            script_lines.append(f"    print(f'Keeping: {best_image.path}')")
            
            for image in group.images:
                if image != best_image:
                    script_lines.append("    if args.dry_run:")
                    script_lines.append(f"        print(f'Would delete: {image.path}')")
                    script_lines.append("    else:")
                    script_lines.append(f"        print(f'Deleting: {image.path}')")
                    script_lines.append(f"        try:")
                    script_lines.append(f"            os.remove(r'{image.path}')")
                    script_lines.append(f"        except Exception as e:")
                    script_lines.append(f"            print(f'Error: {{e}}')")
            
            script_lines.append("")
        
        script_lines.append("    print('Finished!')")
        script_lines.append("")
        script_lines.append("if __name__ == '__main__':")
        script_lines.append("    main()")
    
    # Output script to file or stdout
    script_content = '\n'.join(script_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(script_content)
        logger.info(f"Script saved to {output_file}")
        
        # Make script executable on Unix-like systems
        if script_type == 'bash' and os.name == 'posix':
            os.chmod(output_file, 0o755)
    else:
        print(script_content)


# ======= Cache Management =======

def save_cache(data: Any, cache_name: str) -> None:
    """Save data to cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_name}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def load_cache(cache_name: str) -> Any:
    """Load data from cache file."""
    cache_file = CACHE_DIR / f"{cache_name}.json"
    
    if not cache_file.exists():
        return None
        
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def clear_cache() -> None:
    """Clear all cache files."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob('*.json'):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="""imagededup - Intelligent Image Deduplication Tool

This tool helps you find and manage duplicate images, even if they have different names,
resolutions, or encodings. It uses perceptual hashing and content analysis to identify
duplicates with high accuracy.

Examples:
  # Basic usage - generate a report of duplicates
  imagededup.py /path/to/images

  # Scan multiple directories and generate HTML report
  imagededup.py /images/photos /images/downloads --output-format html --html-report-dir ./report

  # Interactive mode to review and handle duplicates
  imagededup.py /path/to/images --action interactive

  # Move duplicates to a separate directory, keeping originals
  imagededup.py /path/to/images --action move --target-dir ./duplicates

  # Generate a script to handle duplicates later
  imagededup.py /path/to/images --action script --script-type bash --output-file cleanup.sh

  # Delete duplicates (requires --force-delete for safety)
  imagededup.py /path/to/images --action delete --force-delete""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input directories
    parser.add_argument(
        'directories', 
        nargs='+', 
        type=str, 
        help='Directories to scan for images'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--dimension-threshold', 
        type=float, 
        default=0.1,
        help='Percentage threshold for dimension matching'
    )
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
        metavar='ALGORITHM',
        default='phash',
        help='Perceptual hash algorithm to use (default: phash)'
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
    action_group = parser.add_argument_group('Action Options')
    action_group.add_argument(
        '--action', 
        type=str, 
        choices=['report', 'interactive', 'move', 'symlink', 'hardlink', 'delete', 'script'],
        metavar='ACTION',
        default='report',
        help='Action to take for duplicates (default: report)'
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
        metavar='TYPE',
        default='bash',
        help='Type of script to generate (default: bash)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-format', 
        type=str, 
        choices=['text', 'json', 'csv', 'html'],
        metavar='FORMAT',
        default='text',
        help='Output format for report (default: text)'
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
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity level (-v for detailed, -vv for debug)'
    )
    misc_group.add_argument(
        '--version', 
        action='version', 
            version=f'imagededup {VERSION}'
    )
    misc_group.add_argument(
        '--clear-cache', 
        action='store_true', 
        help='Clear cache before running'
    )
    
    args = parser.parse_args()
    
    # Set up logging based on verbosity
    if args.verbose == 0:
        log_level = logging.INFO
    elif args.verbose == 1:
        # Add custom VERBOSE level between INFO and DEBUG
        logging.VERBOSE = 15  # Between INFO (20) and DEBUG (10)
        logging.addLevelName(logging.VERBOSE, "VERBOSE")
        log_level = logging.VERBOSE
    else:
        log_level = logging.DEBUG
    
    logger.setLevel(log_level)
    
    if args.verbose >= 1:
        logger.log(logging.VERBOSE, "Verbose logging enabled")
        if args.verbose >= 2:
            logger.debug("Debug logging enabled")
    
    # Check dependencies
    if not IMAGEHASH_AVAILABLE:
        logger.error("imagehash is required but not installed")
        logger.error("Install with: pip install imagehash")
        return 1
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    # Start the analysis process
    start_time = time.time()
    
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
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
