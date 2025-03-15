"""
Core image analysis functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from ..common.utils import find_files, IMAGE_EXTENSIONS
from .models import ImageFile

logger = logging.getLogger(__name__)

def find_image_files(directories: List[Path], recursive: bool = True) -> List[ImageFile]:
    """Find all image files in the specified directories."""
    paths = find_files(directories, IMAGE_EXTENSIONS, recursive, logger)
    return [ImageFile(path) for path in paths]

def group_by_dimensions(images: List[ImageFile], threshold: float = 0.1) -> Dict[str, List[ImageFile]]:
    """Group images by similar dimensions with a percentage threshold."""
    dimension_groups = {}
    
    logger.debug(f"Grouping {len(images)} images by dimensions (threshold: {threshold}%)")
    
    # First, load metadata for all images
    with ProcessPoolExecutor() as executor:
        for i, image in enumerate(executor.map(load_image_metadata, images)):
            if i % 10 == 0:  # Log progress every 10 images
                logger.debug(f"Processing image {i+1}/{len(images)}")
            if image and image.dimensions != (0, 0):
                # Create dimension key by rounding to nearest multiple of threshold
                width, height = image.dimensions
                key_width = round(width / (width * threshold)) * (width * threshold)
                key_height = round(height / (height * threshold)) * (height * threshold)
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
        logger.debug(f"Loaded metadata: {image.path} ({image.dimensions[0]}x{image.dimensions[1]})")
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

def find_duplicates(images: List[ImageFile], similarity_threshold: float = 80.0) -> List['DuplicateGroup']:
    """Find duplicate images based on similarity threshold."""
    from ..common.models import DuplicateGroup
    
    duplicate_groups = []
    processed = set()
    
    # Compare each image with every other image
    for i, image1 in enumerate(images):
        if image1.hash_id in processed:
            continue
            
        current_group = DuplicateGroup()
        current_group.add_file(image1)
        max_similarity = 0.0
        
        for j, image2 in enumerate(images):
            if i == j or image2.hash_id in processed:
                continue
                
            similarity = compare_images(image1, image2)
            if similarity >= similarity_threshold:
                current_group.add_file(image2)
                processed.add(image2.hash_id)
                max_similarity = max(max_similarity, similarity)
                
        if len(current_group.files) > 1:
            current_group.similarity_score = max_similarity
            current_group.determine_best_version()
            duplicate_groups.append(current_group)
            
        processed.add(image1.hash_id)
    
    return duplicate_groups

def analyze_images(image_files: List[ImageFile], args) -> List['DuplicateGroup']:
    """Main analysis pipeline to find duplicate images."""
    logger.info("Starting image analysis pipeline")
    logger.debug(f"Analysis parameters: dimension_threshold={args.dimension_threshold}%, "
                f"similarity_threshold={args.similarity_threshold}%, hash_algorithm={args.hash_algorithm}")
    
    # Step 1: Load metadata
    logger.info("Loading image metadata...")
    images_with_metadata = []
    
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
        logger.debug(f"Calculating hashes for {len(images)} images...")
        
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
            logger.debug(f"Found {len(duplicate_groups)} duplicate groups in dimension group {dimensions}")
            for group in duplicate_groups:
                logger.debug(f"Duplicate group: {len(group.files)} images, {group.similarity_score:.1f}% similarity")
        
        all_duplicate_groups.extend(duplicate_groups)
        total_comparisons += (len(images) * (len(images) - 1)) // 2
    
    # Sort duplicate groups by wasted space (descending)
    all_duplicate_groups.sort(
        key=lambda g: sum(v.size for v in g.files[1:]),
        reverse=True
    )
    
    logger.debug(f"Analysis complete: {len(all_duplicate_groups)} duplicate groups found")
    logger.debug(f"Total image comparisons performed: {total_comparisons}")
    
    return all_duplicate_groups
