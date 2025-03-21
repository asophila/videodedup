"""
Core video analysis functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from ..common.utils import find_files, VIDEO_EXTENSIONS
from ..common.models import DuplicateGroup
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
        futures = list(executor.map(load_video_metadata, videos))
        for i, video in enumerate(tqdm(futures, desc="Loading metadata", total=len(videos))):
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
                             hash_algorithm: str = 'phash',
                             progress_callback: Optional[Callable] = None) -> List[VideoFile]:
    """Extract fingerprints (frame hashes) from videos.
    
    Args:
        videos: List of videos to process
        frame_positions: List of positions to extract frames from (0.0-1.0)
        hash_algorithm: Hash algorithm to use for frame hashing
        progress_callback: Optional callback to update progress
    """
    if frame_positions is None:
        # Default to sampling at beginning, 25%, 50%, 75% and end
        frame_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Create a function with preset arguments
    process_func = partial(_process_video_frames, 
                         frame_positions=frame_positions,
                         hash_algorithm=hash_algorithm)
    
    # Process all videos using multiprocessing
    processed_videos = []
    with ProcessPoolExecutor() as executor:
        futures = list(executor.map(process_func, videos))
        for video in futures:
            processed_videos.append(video)
            if progress_callback:
                progress_callback()
    
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

def find_duplicates(videos: List[VideoFile], 
                   similarity_threshold: float = 80.0,
                   progress_callback: Optional[Callable] = None) -> List['DuplicateGroup']:
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
            
            if progress_callback:
                progress_callback()
                
        if len(current_group.files) > 1:
            current_group.similarity_score = max_similarity
            current_group.determine_best_version()
            duplicate_groups.append(current_group)
            
        processed.add(video1.hash_id)
    
    return duplicate_groups

def group_by_crc(videos: List[VideoFile]) -> Dict[str, List[VideoFile]]:
    """Group videos by CRC32 and SHA-256 hash to find exact duplicates."""
    crc_groups = {}
    
    logger.debug(f"Grouping {len(videos)} videos by CRC hash")
    
    for video in tqdm(videos, desc="Checking CRC hashes"):
        # Use both CRC32 and SHA-256 as key to avoid collisions
        key = f"{video.crc32}_{video.sha256}"
        if key not in crc_groups:
            crc_groups[key] = []
        crc_groups[key].append(video)
    
    # Filter out groups with only one video (no duplicates)
    filtered_groups = {k: v for k, v in crc_groups.items() if len(v) > 1}
    
    if filtered_groups:
        logger.info(f"Found {len(filtered_groups)} groups of exact duplicates")
        for key, group in filtered_groups.items():
            logger.debug(f"CRC group {key[:8]}: {len(group)} videos")
    
    return filtered_groups

def analyze_videos(video_files: List[VideoFile], args) -> List['DuplicateGroup']:
    """Main analysis pipeline to find duplicate videos."""
    logger.info("Starting video analysis pipeline")
    logger.debug(f"Analysis parameters: duration_threshold={args.duration_threshold}%, "
                f"similarity_threshold={args.similarity_threshold}%, hash_algorithm={args.hash_algorithm}")
    
    # Step 1: Load metadata
    logger.info("Loading video metadata...")
    videos_with_metadata = []
    
    with ProcessPoolExecutor() as executor:
        futures = list(executor.map(load_video_metadata, video_files))
        for i, video in enumerate(tqdm(futures, desc="Loading metadata", total=len(video_files))):
            if video:
                videos_with_metadata.append(video)
    
    logger.info(f"Successfully loaded metadata for {len(videos_with_metadata)} videos")
    
    # Initialize variables
    all_duplicate_groups = []
    videos_for_analysis = videos_with_metadata
    processed_videos = set()

    # Step 2: Find exact duplicates using CRC (unless skipped)
    if not args.skip_crc:
        logger.debug("Finding exact duplicates by CRC...")
        crc_groups = group_by_crc(videos_with_metadata)
        
        # Create duplicate groups for exact matches
        for videos in crc_groups.values():
            group = DuplicateGroup()
            for video in videos:
                group.add_file(video)
                processed_videos.add(video.hash_id)
            group.similarity_score = 100.0  # Exact match
            group.determine_best_version()
            all_duplicate_groups.append(group)
        
        # Update videos for perceptual analysis
        videos_for_analysis = [v for v in videos_with_metadata if v.hash_id not in processed_videos]
    
    # Step 3: Group remaining videos by duration for perceptual analysis
    if videos_for_analysis:
        logger.debug("Grouping videos by duration...")
        duration_groups = group_by_duration(videos_for_analysis, args.duration_threshold)
        
        # Step 4: Process each duration group
        total_comparisons = 0
    
        for duration, videos in duration_groups.items():
            if len(videos) < 2:
                logger.debug(f"Skipping duration group {duration}s (only {len(videos)} video)")
                continue
                
            logger.info(f"Processing {len(videos)} videos with duration ~{duration}s")
            
            # Extract frame hashes for this group
            with tqdm(total=len(videos), desc=f"Processing videos in group {duration}s") as pbar:
                videos_with_hashes = extract_video_fingerprints(
                    videos, 
                    frame_positions=[0.1, 0.3, 0.5, 0.7, 0.9],  # More sample points for better accuracy
                    hash_algorithm=args.hash_algorithm,
                    progress_callback=lambda: pbar.update(1)
                )
            
            # Find duplicates within this group
            logger.debug(f"Comparing videos in duration group {duration}s...")
            with tqdm(total=(len(videos) * (len(videos) - 1)) // 2, desc="Comparing videos") as pbar:
                duplicate_groups = find_duplicates(
                    videos_with_hashes, 
                    similarity_threshold=args.similarity_threshold,
                    progress_callback=lambda: pbar.update(1)
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
    if total_comparisons > 0:
        logger.debug(f"Total perceptual comparisons performed: {total_comparisons}")
    
    return all_duplicate_groups
