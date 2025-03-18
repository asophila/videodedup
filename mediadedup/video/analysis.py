"""
Core video analysis functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from ..common.utils import find_files, VIDEO_EXTENSIONS
from .models import VideoFile

logger = logging.getLogger(__name__)

def find_video_files(directories: List[Path], recursive: bool = True) -> List[VideoFile]:
    """Find all video files in the specified directories."""
    paths = find_files(directories, VIDEO_EXTENSIONS, recursive, logger)
    return [VideoFile(path) for path in paths]

def group_by_duration(videos: List[VideoFile], threshold: float = 1.0) -> Dict[str, List[VideoFile]]:
    """Group videos by similar duration with a percentage threshold."""
    duration_groups = {}
    
    logger.debug(f"Grouping {len(videos)} videos by duration (threshold: {threshold}%)")
    
    # First, load metadata for all videos
    with ProcessPoolExecutor() as executor:
        for i, video in enumerate(executor.map(load_video_metadata, videos)):
            if i % 10 == 0:  # Log progress every 10 videos
                logger.debug(f"Processing video {i+1}/{len(videos)}")
            if video and video.duration > 0:
                # Find or create a suitable duration group
                duration_found = False
                for key in duration_groups.keys():
                    base_duration = float(key)
                    # Calculate allowed duration range based on threshold
                    min_duration = base_duration * (1 - threshold/100)
                    max_duration = base_duration * (1 + threshold/100)
                    
                    if min_duration <= video.duration <= max_duration:
                        duration_groups[key].append(video)
                        duration_found = True
                        break
                
                # If no suitable group found, create a new one
                if not duration_found:
                    key = f"{video.duration:.1f}"
                    duration_groups[key] = [video]
    
    # Filter out groups with only one video (no potential duplicates)
    filtered_groups = {k: v for k, v in duration_groups.items() if len(v) > 1}
    
    logger.info(f"Grouped videos into {len(filtered_groups)} duration groups")
    return filtered_groups

def load_video_metadata(video: VideoFile) -> VideoFile:
    """Load video metadata - this is a wrapper for multiprocessing."""
    try:
        logger.debug(f"Loading metadata for {video.path}")
        video.load_complete_metadata()
        logger.debug(f"Loaded metadata: {video.path} ({video.resolution[0]}x{video.resolution[1]}, {video.duration:.1f}s)")
        return video
    except Exception as e:
        logger.warning(f"Error loading metadata for {video.path}: {e}")
        return None

def _process_video_frames(video: VideoFile, frame_positions: List[float], hash_algorithm: str) -> VideoFile:
    """Process a single video to extract frame hashes."""
    try:
        logger.debug(f"Processing frames for {video.path}")
        # Adjust frame positions based on video duration
        actual_positions = [pos * video.duration for pos in frame_positions 
                          if 0 <= pos <= 1.0]
        logger.debug(f"Extracting frames at positions: {[f'{pos:.1f}s' for pos in actual_positions]}")
        
        # Extract frame hashes
        video.extract_frame_hashes(actual_positions, hash_algorithm)
        return video
    except Exception as e:
        logger.warning(f"Error extracting fingerprints from {video.path}: {e}")
        return video

def extract_video_fingerprints(videos: List[VideoFile], 
                             frame_positions: List[float] = None,
                             hash_algorithm: str = 'phash') -> List[VideoFile]:
    """Extract fingerprints (frame hashes) from videos using RAM disk for better performance."""
    from ..common.ramdisk import RAMDiskManager
    
    if frame_positions is None:
        # Default to sampling at beginning, 25%, 50%, 75% and end
        frame_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Create a function with preset arguments
    process_func = partial(_process_video_frames, 
                          frame_positions=frame_positions,
                          hash_algorithm=hash_algorithm)
    
    # Group videos by size to optimize RAM disk usage
    small_videos = []
    large_videos = []
    
    # Create RAM disk manager to check sizes
    with RAMDiskManager() as ram_disk:
        ram_disk_size = ram_disk.size_mb * 1024 * 1024  # Convert to bytes
        max_single_file = ram_disk_size * 0.8  # Single file shouldn't use more than 80% of RAM disk
        
        # Sort videos into small and large groups
        for video in videos:
            if video.size <= max_single_file:
                small_videos.append(video)
            else:
                large_videos.append(video)
        
        if large_videos:
            logger.info(f"Found {len(large_videos)} videos too large for RAM disk, will process directly")
        
        # Sort small videos by size for optimal batching
        small_videos.sort(key=lambda v: v.size)
        processed_videos = []
        
        # Process small videos in RAM disk batches
        batch = []
        batch_size = 0
        
        for video in small_videos:
            # Check if this video will fit with current batch
            if not ram_disk.will_file_fit(video.size) and batch:
                # Process current batch
                logger.info(f"Processing batch of {len(batch)} videos in RAM disk")
                with ProcessPoolExecutor() as executor:
                    processed_batch = list(executor.map(process_func, batch))
                processed_videos.extend(processed_batch)
                # Clear batch and RAM disk
                batch = []
                batch_size = 0
                ram_disk.clear()
            
            # Try to add video to current batch
            try:
                # Copy video to RAM disk
                ram_path = ram_disk.copy_to_ramdisk(video.path)
                # Update video path to RAM disk path
                orig_path = video.path
                video.path = ram_path
                batch.append(video)
                batch_size += video.size
                logger.debug(f"Added {orig_path.name} to RAM disk batch")
            except ValueError as e:
                # File too large for current RAM disk state
                logger.warning(str(e))
                if batch:
                    # Process current batch first
                    logger.info(f"Processing batch of {len(batch)} videos in RAM disk")
                    with ProcessPoolExecutor() as executor:
                        processed_batch = list(executor.map(process_func, batch))
                    processed_videos.extend(processed_batch)
                    # Clear batch and RAM disk
                    batch = []
                    batch_size = 0
                    ram_disk.clear()
                    # Try again with empty RAM disk
                    try:
                        ram_path = ram_disk.copy_to_ramdisk(video.path)
                        video.path = ram_path
                        batch.append(video)
                        batch_size += video.size
                        logger.debug(f"Added {orig_path.name} to RAM disk batch after clearing")
                    except ValueError:
                        # Still too large, process directly
                        logger.warning(f"Video {video.path.name} too large for RAM disk, processing directly")
                        processed_videos.append(process_func(video))
            except Exception as e:
                logger.warning(f"Failed to copy {video.path} to RAM disk: {e}")
                # Process this video directly from disk
                processed_videos.append(process_func(video))
        
        # Process final batch if any
        if batch:
            logger.info(f"Processing final batch of {len(batch)} videos in RAM disk")
            with ProcessPoolExecutor() as executor:
                processed_batch = list(executor.map(process_func, batch))
            processed_videos.extend(processed_batch)
        
        # Process large videos directly
        if large_videos:
            logger.info(f"Processing {len(large_videos)} large videos directly from disk")
            with ProcessPoolExecutor() as executor:
                processed_large = list(executor.map(process_func, large_videos))
            processed_videos.extend(processed_large)
    
    # Store original paths before processing
    original_paths = {video.hash_id: video.path for video in videos}
    
    # Restore original paths after processing
    for video in processed_videos:
        if str(video.path).startswith(str(ram_disk.mount_point)):
            # Store RAM disk path
            video.original_path = video.path
            # Restore the full original path
            video.path = original_paths[video.hash_id]
    
    return processed_videos

def compare_videos(video1: VideoFile, video2: VideoFile, threshold: int = 10) -> float:
    """Compare two videos based on their frame hashes and return a similarity score."""
    if not video1.frame_hashes or not video2.frame_hashes:
        return 0.0
    
    # Find common timestamp positions
    common_positions = set(video1.frame_hashes.keys()) & set(video2.frame_hashes.keys())
    if not common_positions:
        return 0.0
    
    # Calculate hamming distances between frame hashes
    distances = []
    for pos in common_positions:
        hash1 = video1.frame_hashes[pos]
        hash2 = video2.frame_hashes[pos]
        distance = hash1 - hash2
        distances.append(distance)
    
    # Calculate average distance
    avg_distance = sum(distances) / len(distances)
    
    # Convert distance to similarity score (0-100)
    # Lower distance means higher similarity
    similarity = max(0, 100 - (avg_distance * (100 / threshold)))
    
    return similarity

def find_duplicates(videos: List[VideoFile], similarity_threshold: float = 80.0) -> List['DuplicateGroup']:
    """Find duplicate videos based on similarity threshold."""
    from ..common.models import DuplicateGroup
    
    duplicate_groups = []
    processed = set()
    
    # Compare each video with every other video
    for i, video1 in enumerate(videos):
        if video1.hash_id in processed:
            continue
            
        current_group = DuplicateGroup()
        current_group.add_file(video1)
        max_similarity = 0.0
        
        for j, video2 in enumerate(videos):
            if i == j or video2.hash_id in processed:
                continue
                
            similarity = compare_videos(video1, video2)
            if similarity >= similarity_threshold:
                current_group.add_file(video2)
                processed.add(video2.hash_id)
                max_similarity = max(max_similarity, similarity)
                
        if len(current_group.files) > 1:
            current_group.similarity_score = max_similarity
            current_group.determine_best_version()
            duplicate_groups.append(current_group)
            
        processed.add(video1.hash_id)
    
    return duplicate_groups

def analyze_videos(video_files: List[VideoFile], args) -> List['DuplicateGroup']:
    """Main analysis pipeline to find duplicate videos."""
    logger.info("Starting video analysis pipeline")
    logger.debug(f"Analysis parameters: duration_threshold={args.duration_threshold}%, "
                f"similarity_threshold={args.similarity_threshold}%, hash_algorithm={args.hash_algorithm}")
    
    # Step 1: Load metadata
    logger.info("Loading video metadata...")
    videos_with_metadata = []
    
    with ProcessPoolExecutor() as executor:
        videos_with_metadata = list(filter(None, executor.map(load_video_metadata, video_files)))
    
    logger.info(f"Successfully loaded metadata for {len(videos_with_metadata)} videos")
    
    # Step 2: Group by duration
    logger.debug("Grouping videos by duration...")
    duration_groups = group_by_duration(videos_with_metadata, args.duration_threshold)
    
    # Step 3: Process each duration group
    all_duplicate_groups = []
    total_comparisons = 0
    
    for duration, videos in duration_groups.items():
        if len(videos) < 2:
            logger.debug(f"Skipping duration group {duration}s (only {len(videos)} video)")
            continue
            
        logger.info(f"Processing {len(videos)} videos with duration ~{duration}s")
        logger.debug(f"Extracting frame hashes for {len(videos)} videos...")
        
        # Extract frame hashes for this group
        videos_with_hashes = extract_video_fingerprints(
            videos, 
            frame_positions=[0.1, 0.3, 0.5, 0.7, 0.9],  # More sample points for better accuracy
            hash_algorithm=args.hash_algorithm
        )
        
        # Find duplicates within this group
        logger.debug(f"Comparing videos in duration group {duration}s...")
        duplicate_groups = find_duplicates(
            videos_with_hashes, 
            similarity_threshold=args.similarity_threshold
        )
        
        if duplicate_groups:
            logger.debug(f"Found {len(duplicate_groups)} duplicate groups in duration {duration}s")
            for group in duplicate_groups:
                logger.debug(f"Duplicate group: {len(group.files)} videos, {group.similarity_score:.1f}% similarity")
        
        all_duplicate_groups.extend(duplicate_groups)
        total_comparisons += (len(videos) * (len(videos) - 1)) // 2
    
    # Sort duplicate groups by wasted space (descending)
    all_duplicate_groups.sort(
        key=lambda g: sum(v.size for v in g.files[1:]), 
        reverse=True
    )
    
    logger.debug(f"Analysis complete: {len(all_duplicate_groups)} duplicate groups found")
    logger.debug(f"Total video comparisons performed: {total_comparisons}")
    
    return all_duplicate_groups
