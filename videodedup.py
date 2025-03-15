#!/usr/bin/env python3
"""
videodedup - Intelligent Video Deduplication Tool

A content-aware video deduplicator that finds duplicate videos
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
import subprocess
import tempfile
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

# FFmpeg-based scene detection and audio fingerprinting
def detect_scenes_ffmpeg(video_path, threshold=0.3):
    """Detect scene changes using FFmpeg."""
    cmd = [
        "ffmpeg", "-i", str(video_path), 
        "-filter:v", f"select='gt(scene,{threshold})',showinfo", 
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output to extract scene change timestamps
        scenes = []
        for line in result.stderr.splitlines():
            if "pts_time:" in line and "scene:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("pts_time:"):
                        timestamp = float(parts[i].split(':')[1])
                        scenes.append(timestamp)
        
        return scenes
    except Exception as e:
        logger.warning(f"Error detecting scenes in {video_path}: {e}")
        return []

def extract_audio_fingerprint_ffmpeg(video_path):
    """Extract audio fingerprint using FFmpeg."""
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
            # Extract audio to WAV
            subprocess.run([
                "ffmpeg", "-i", str(video_path), 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", "16000", "-ac", "1", temp_audio.name
            ], check=True, capture_output=True)
            
            # Extract audio features (volume levels at intervals)
            cmd = [
                "ffmpeg", "-i", temp_audio.name,
                "-af", "volumedetect", "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse volume levels
            volume_levels = []
            for line in result.stderr.splitlines():
                if "mean_volume:" in line:
                    vol = float(line.split(':')[1].strip().split()[0])
                    volume_levels.append(vol)
            
            return volume_levels
    except Exception as e:
        logger.warning(f"Error extracting audio fingerprint from {video_path}: {e}")
        return []

def check_quicksync_available():
    """Check if Intel QuickSync is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"], 
            capture_output=True, text=True
        )
        return "qsv" in result.stdout.lower()
    except Exception:
        return False

def extract_frames_quicksync(video_path, timestamps):
    """Extract frames using QuickSync hardware acceleration."""
    frames = []
    
    try:
        for ts in timestamps:
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_img:
                cmd = [
                    "ffmpeg", "-hwaccel", "qsv", "-i", str(video_path),
                    "-ss", str(ts), "-frames:v", "1",
                    "-c:v", "mjpeg_qsv", temp_img.name
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Load the extracted frame
                with Image.open(temp_img.name) as img:
                    frames.append(img.copy())
        
        return frames
    except Exception as e:
        logger.warning(f"Error extracting frames with QuickSync from {video_path}: {e}")
        return []

# Check if ffmpeg and ffprobe are installed
def check_ffmpeg():
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('videodedup')

# Constants
VERSION = "1.0.0"
CACHE_DIR = Path.home() / ".cache" / "videodedup"
VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
    '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.mxf', '.ts', 
    '.m2ts', '.vob', '.ogv', '.mts', '.m2v', '.divx', '.rmvb', '.rm'
}

# ======= Data Models =======

@dataclass
class VideoFile:
    """Represents a video file with its metadata and analysis results."""
    path: Path
    size: int = 0
    duration: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    bitrate: int = 0
    frame_rate: float = 0.0
    frame_hashes: Dict[float, Any] = field(default_factory=dict)
    audio_fingerprint: Optional[Any] = None
    content_score: float = 0.0  # Quality score
    hash_id: str = ""  # Unique hash for this video
    
    def __post_init__(self):
        """Initialize size from path if not provided."""
        if self.size == 0 and self.path.exists():
            self.size = self.path.stat().st_size
        
        # Generate a unique hash ID for this file
        if not self.hash_id:
            self.hash_id = hashlib.md5(str(self.path).encode()).hexdigest()

    def get_metadata(self) -> Dict:
        """Get basic metadata without loading the video."""
        return {
            'path': str(self.path),
            'size': self.size,
            'extension': self.path.suffix.lower(),
            'last_modified': datetime.fromtimestamp(self.path.stat().st_mtime).isoformat()
        }
    
    def load_complete_metadata(self) -> bool:
        """Load full video metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe", 
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(self.path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Could not probe video file: {self.path}")
                return False
            
            probe_data = json.loads(result.stdout)
            
            # Get video stream
            video_stream = next(
                (stream for stream in probe_data['streams'] 
                 if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                logger.warning(f"No video stream found in {self.path}")
                return False
            
            # Extract metadata
            self.frame_rate = float(eval(video_stream.get('r_frame_rate', '0/1')))
            self.duration = float(video_stream.get('duration', 0))
            self.resolution = (
                int(video_stream.get('width', 0)),
                int(video_stream.get('height', 0))
            )
            
            # Get bitrate from format section if available, otherwise from video stream
            format_data = probe_data.get('format', {})
            self.bitrate = int(format_data.get('bit_rate', 0)) or int(video_stream.get('bit_rate', 0))
            
            return True
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {self.path}: {e}")
            return False
    
    def calculate_content_score(self) -> float:
        """Calculate a quality score based on resolution, bitrate, etc."""
        # Basic scoring based on resolution and bitrate
        resolution_score = self.resolution[0] * self.resolution[1] / 1000000  # Normalized by millions of pixels
        bitrate_score = self.bitrate / 1000000  # Normalized by mbps
        
        # Combine scores with weights
        self.content_score = (resolution_score * 0.7) + (bitrate_score * 0.3)
        return self.content_score
    
    def extract_frame_hashes(self, 
                            frame_positions: List[float], 
                            hash_algorithm: str = 'phash',
                            use_quicksync: bool = False) -> Dict[float, Any]:
        """Extract perceptual hashes from frames at specified positions using FFmpeg."""
        if not IMAGEHASH_AVAILABLE:
            logger.warning("ImageHash not available, can't extract frame hashes")
            return {}
            
        # Clear previous hashes if any
        self.frame_hashes = {}
        
        try:
            # Use QuickSync if available and requested
            if use_quicksync and check_quicksync_available():
                frames = extract_frames_quicksync(self.path, frame_positions)
            else:
                frames = []
                for pos in frame_positions:
                    with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_img:
                        cmd = [
                            "ffmpeg", "-ss", str(pos), 
                            "-i", str(self.path),
                            "-frames:v", "1",
                            "-f", "image2", temp_img.name
                        ]
                        try:
                            subprocess.run(cmd, check=True, capture_output=True)
                            with Image.open(temp_img.name) as img:
                                frames.append(img.copy())
                        except Exception as e:
                            logger.warning(f"Error extracting frame at {pos}s: {e}")
            
            # Generate hashes for extracted frames
            for pos, frame in zip(frame_positions, frames):
                if frame:
                    # Generate perceptual hash
                    if hash_algorithm == 'phash':
                        frame_hash = imagehash.phash(frame)
                    elif hash_algorithm == 'dhash':
                        frame_hash = imagehash.dhash(frame)
                    elif hash_algorithm == 'whash':
                        frame_hash = imagehash.whash(frame)
                    elif hash_algorithm == 'average_hash':
                        frame_hash = imagehash.average_hash(frame)
                    else:
                        frame_hash = imagehash.phash(frame)
                        
                    self.frame_hashes[pos] = frame_hash
            
            return self.frame_hashes
            
        except Exception as e:
            logger.warning(f"Error extracting frame hashes from {self.path}: {e}")
            return {}
    
    def detect_scenes(self, threshold: float = 0.3) -> List[float]:
        """Detect scene changes in the video."""
        return detect_scenes_ffmpeg(self.path, threshold)
    
    def extract_audio_fingerprint(self) -> List[float]:
        """Extract audio fingerprint from the video."""
        self.audio_fingerprint = extract_audio_fingerprint_ffmpeg(self.path)
        return self.audio_fingerprint

@dataclass
class DuplicateGroup:
    """Represents a group of duplicate videos."""
    videos: List[VideoFile] = field(default_factory=list)
    similarity_score: float = 0.0
    best_version: Optional[VideoFile] = None
    
    def add_video(self, video: VideoFile):
        """Add a video to the duplicate group."""
        self.videos.append(video)
        
    def determine_best_version(self, 
                              prefer_resolution: bool = True, 
                              prefer_bitrate: bool = True,
                              prefer_size: bool = False) -> VideoFile:
        """Determine the best version based on configurable criteria."""
        if not self.videos:
            return None
            
        # Calculate content scores if not already done
        for video in self.videos:
            if video.content_score == 0:
                video.calculate_content_score()
                
        # Sort by content score (higher is better)
        sorted_videos = sorted(self.videos, key=lambda v: v.content_score, reverse=True)
        self.best_version = sorted_videos[0]
        return self.best_version
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'similarity_score': self.similarity_score,
            'video_count': len(self.videos),
            'best_version': str(self.best_version.path) if self.best_version else None,
            'videos': [str(v.path) for v in self.videos],
            'total_size': sum(v.size for v in self.videos),
            'wasted_space': sum(v.size for v in self.videos[1:]) if self.videos else 0
        }


# ======= Core Analysis Functions =======

def find_video_files(directories: List[Path], recursive: bool = True) -> List[VideoFile]:
    """Find all video files in the specified directories."""
    video_files = []
    
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
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(VideoFile(path=file_path))
    
    logger.info(f"Found {len(video_files)} video files")
    return video_files

def group_by_duration(videos: List[VideoFile], threshold: float = 1.0) -> Dict[str, List[VideoFile]]:
    """Group videos by similar duration with a percentage threshold."""
    duration_groups = {}
    
    logger.log(logging.VERBOSE, f"Grouping {len(videos)} videos by duration (threshold: {threshold}%)")
    
    # First, load metadata for all videos
    with ProcessPoolExecutor() as executor:
        for i, video in enumerate(executor.map(load_video_metadata, videos)):
            if i % 10 == 0:  # Log progress every 10 videos
                logger.debug(f"Processing video {i+1}/{len(videos)}")
            if video and video.duration > 0:
                # Create a duration key with a precision relative to the threshold
                # e.g., with 1% threshold, round to the nearest 1% of the duration
                duration_key = f"{video.duration:.1f}"
                
                if duration_key not in duration_groups:
                    duration_groups[duration_key] = []
                    
                duration_groups[duration_key].append(video)
    
    # Filter out groups with only one video (no potential duplicates)
    filtered_groups = {k: v for k, v in duration_groups.items() if len(v) > 1}
    
    logger.info(f"Grouped videos into {len(filtered_groups)} duration groups")
    return filtered_groups

def load_video_metadata(video: VideoFile) -> VideoFile:
    """Load video metadata - this is a wrapper for multiprocessing."""
    try:
        logger.debug(f"Loading metadata for {video.path}")
        video.load_complete_metadata()
        logger.log(logging.VERBOSE, f"Loaded metadata: {video.path} ({video.resolution[0]}x{video.resolution[1]}, {video.duration:.1f}s)")
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
        logger.log(logging.VERBOSE, f"Extracting frames at positions: {[f'{pos:.1f}s' for pos in actual_positions]}")
        
        # Extract frame hashes
        video.extract_frame_hashes(actual_positions, hash_algorithm)
        return video
    except Exception as e:
        logger.warning(f"Error extracting fingerprints from {video.path}: {e}")
        return video

def extract_video_fingerprints(videos: List[VideoFile], 
                              frame_positions: List[float] = None,
                              hash_algorithm: str = 'phash') -> List[VideoFile]:
    """Extract fingerprints (frame hashes) from videos."""
    if frame_positions is None:
        # Default to sampling at beginning, 25%, 50%, 75% and end
        frame_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Create a function with preset arguments
    process_func = partial(_process_video_frames, 
                          frame_positions=frame_positions,
                          hash_algorithm=hash_algorithm)
    
    # Process videos in parallel
    with ProcessPoolExecutor() as executor:
        processed_videos = list(executor.map(process_func, videos))
    
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

def find_duplicates(videos: List[VideoFile], similarity_threshold: float = 80.0) -> List[DuplicateGroup]:
    """Find duplicate videos based on similarity threshold."""
    duplicate_groups = []
    processed = set()
    
    # Compare each video with every other video
    for i, video1 in enumerate(videos):
        if video1.hash_id in processed:
            continue
            
        current_group = DuplicateGroup()
        current_group.add_video(video1)
        
        for j, video2 in enumerate(videos):
            if i == j or video2.hash_id in processed:
                continue
                
            similarity = compare_videos(video1, video2)
            if similarity >= similarity_threshold:
                current_group.add_video(video2)
                processed.add(video2.hash_id)
                
        if len(current_group.videos) > 1:
            current_group.similarity_score = similarity
            current_group.determine_best_version()
            duplicate_groups.append(current_group)
            
        processed.add(video1.hash_id)
    
    return duplicate_groups

def analyze_videos(video_files: List[VideoFile], args: argparse.Namespace) -> List[DuplicateGroup]:
    """Main analysis pipeline to find duplicate videos."""
    logger.log(logging.VERBOSE, "Starting video analysis pipeline")
    logger.log(logging.VERBOSE, f"Analysis parameters: duration_threshold={args.duration_threshold}%, "
               f"similarity_threshold={args.similarity_threshold}%, hash_algorithm={args.hash_algorithm}")
    
    # Step 1: Load metadata
    logger.info("Loading video metadata...")
    videos_with_metadata = []
    
    if TQDM_AVAILABLE:
        for video in tqdm(video_files, desc="Loading metadata"):
            if video.load_complete_metadata():
                videos_with_metadata.append(video)
    else:
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
        logger.log(logging.VERBOSE, f"Extracting frame hashes for {len(videos)} videos...")
        
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
            logger.log(logging.VERBOSE, f"Found {len(duplicate_groups)} duplicate groups in duration {duration}s")
            for group in duplicate_groups:
                logger.debug(f"Duplicate group: {len(group.videos)} videos, {group.similarity_score:.1f}% similarity")
        
        all_duplicate_groups.extend(duplicate_groups)
        total_comparisons += (len(videos) * (len(videos) - 1)) // 2
    
    # Sort duplicate groups by wasted space (descending)
    all_duplicate_groups.sort(
        key=lambda g: sum(v.size for v in g.videos[1:]), 
        reverse=True
    )
    
    logger.log(logging.VERBOSE, f"Analysis complete: {len(all_duplicate_groups)} duplicate groups found")
    logger.log(logging.VERBOSE, f"Total video comparisons performed: {total_comparisons}")
    
    return all_duplicate_groups


# ======= Action Functions =======

def handle_duplicates(duplicate_groups: List[DuplicateGroup], args: argparse.Namespace) -> None:
    """Handle duplicate videos according to the specified action."""
    if not duplicate_groups:
        logger.info("No duplicates found.")
        return
        
    # Calculate total statistics
    total_videos = sum(len(group.videos) for group in duplicate_groups)
    total_duplicates = sum(len(group.videos) - 1 for group in duplicate_groups)
    total_wasted_space = sum(
        sum(v.size for v in group.videos[1:]) for group in duplicate_groups
    )
    
    logger.info(f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicates")
    logger.info(f"Total wasted space: {total_wasted_space / (1024**3):.2f} GB")
    
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
    """Generate a report of duplicate videos."""
    if format_type == 'html' and html_report_dir:
        # Create HTML report with thumbnails
        report_dir = Path(html_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create thumbnails directory
        thumbnails_dir = report_dir / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)
        
        # Generate thumbnails for each video
        thumbnails = {}
        
        for group_idx, group in enumerate(duplicate_groups):
            for video_idx, video in enumerate(group.videos):
                # Extract thumbnails at key positions (beginning, middle, end)
                positions = [0.1, 0.5, 0.9]  # 10%, 50%, 90% of video
                
                video_thumbnails = []
                for pos_idx, pos in enumerate(positions):
                    timestamp = pos * video.duration
                    thumbnail_path = thumbnails_dir / f"group_{group_idx}_video_{video_idx}_pos_{pos_idx}.jpg"
                    
                    # Extract thumbnail using ffmpeg
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(timestamp), 
                        "-i", str(video.path),
                        "-frames:v", "1", 
                        "-vf", "scale=320:-1",
                        str(thumbnail_path)
                    ]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        video_thumbnails.append(str(thumbnail_path.relative_to(report_dir)))
                    except Exception as e:
                        logger.warning(f"Failed to extract thumbnail: {e}")
                
                thumbnails[(group_idx, video_idx)] = video_thumbnails
        
        # Generate HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>videodedup - Duplicate Video Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .group { border: 1px solid #ccc; margin-bottom: 20px; padding: 15px; border-radius: 5px; }",
            "        .group-header { background-color: #f0f0f0; padding: 10px; margin-bottom: 15px; border-radius: 3px; }",
            "        .video { margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px dashed #eee; }",
            "        .best-version { background-color: #e6ffe6; padding: 10px; border-radius: 3px; }",
            "        .thumbnails { display: flex; margin-top: 10px; }",
            "        .thumbnail { margin-right: 10px; text-align: center; }",
            "        .thumbnail img { border: 1px solid #ddd; border-radius: 3px; }",
            "        .metadata { margin-top: 10px; font-size: 0.9em; color: #555; }",
            "        .summary { background-color: #f8f8f8; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>videodedup - Duplicate Video Report</h1>",
            f"    <p>Generated on: {datetime.now().isoformat()}</p>"
        ]
        
        # Add summary statistics
        total_videos = sum(len(group.videos) for group in duplicate_groups)
        total_duplicates = sum(len(group.videos) - 1 for group in duplicate_groups)
        total_original_size = sum(sum(v.size for v in group.videos) for group in duplicate_groups)
        total_optimized_size = sum(group.best_version.size for group in duplicate_groups)
        total_savings = total_original_size - total_optimized_size
        
        html_content.extend([
            "    <div class='summary'>",
            "        <h2>Summary</h2>",
            f"        <p>Total duplicate groups: {len(duplicate_groups)}</p>",
            f"        <p>Total videos analyzed: {total_videos}</p>",
            f"        <p>Total duplicate videos: {total_duplicates}</p>",
            f"        <p>Total original size: {total_original_size / (1024**3):.2f} GB</p>",
            f"        <p>Size after deduplication: {total_optimized_size / (1024**3):.2f} GB</p>",
            f"        <p>Potential space savings: {total_savings / (1024**3):.2f} GB ({total_savings/total_original_size*100:.1f}%)</p>",
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
            best_video = group.best_version
            best_idx = group.videos.index(best_video)
            
            html_content.extend([
                "        <div class='video best-version'>",
                f"            <h3>Best Version: {best_video.path.name}</h3>",
                "            <div class='thumbnails'>"
            ])
            
            # Add thumbnails for best version
            for i, thumb in enumerate(thumbnails.get((group_idx, best_idx), [])):
                position = ["Beginning", "Middle", "End"][i]
                html_content.extend([
                    f"                <div class='thumbnail'>",
                    f"                    <img src='{thumb}' alt='Thumbnail {i}' width='320'>",
                    f"                    <div>{position}</div>",
                    f"                </div>"
                ])
            
            html_content.extend([
                "            </div>",
                "            <div class='metadata'>",
                f"                <p>Path: {best_video.path}</p>",
                f"                <p>Size: {best_video.size / (1024**2):.2f} MB</p>",
                f"                <p>Resolution: {best_video.resolution[0]}x{best_video.resolution[1]}</p>",
                f"                <p>Bitrate: {best_video.bitrate / 1000:.0f} kbps</p>",
                f"                <p>Duration: {best_video.duration:.2f} seconds</p>",
                "            </div>",
                "        </div>"
            ])
            
            # Add duplicates
            for video_idx, video in enumerate(group.videos):
                if video != best_video:
                    html_content.extend([
                        "        <div class='video'>",
                        f"            <h3>Duplicate {video_idx + 1}: {video.path.name}</h3>",
                        "            <div class='thumbnails'>"
                    ])
                    
                    # Add thumbnails for this duplicate
                    for i, thumb in enumerate(thumbnails.get((group_idx, video_idx), [])):
                        position = ["Beginning", "Middle", "End"][i]
                        html_content.extend([
                            f"                <div class='thumbnail'>",
                            f"                    <img src='{thumb}' alt='Thumbnail {i}' width='320'>",
                            f"                    <div>{position}</div>",
                            f"                </div>"
                        ])
                    
                    html_content.extend([
                        "            </div>",
                        "            <div class='metadata'>",
                        f"                <p>Path: {video.path}</p>",
                        f"                <p>Size: {video.size / (1024**2):.2f} MB</p>",
                        f"                <p>Resolution: {video.resolution[0]}x{video.resolution[1]}</p>",
                        f"                <p>Bitrate: {video.bitrate / 1000:.0f} kbps</p>",
                        f"                <p>Duration: {video.duration:.2f} seconds</p>",
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
            'total_duplicates': sum(len(group.videos) - 1 for group in duplicate_groups),
            'total_wasted_space': sum(
                sum(v.size for v in group.videos[1:]) for group in duplicate_groups
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
        csv_lines = ['group_id,similarity,file_path,is_best,size,resolution']
        
        for i, group in enumerate(duplicate_groups):
            for video in group.videos:
                is_best = 'yes' if video == group.best_version else 'no'
                res = f"{video.resolution[0]}x{video.resolution[1]}"
                line = f"{i},{group.similarity_score:.1f},\"{video.path}\",{is_best},{video.size},{res}"
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
            "=== Video Deduplication Report ===",
            f"Generated on: {datetime.now().isoformat()}",
            f"Total duplicate groups: {len(duplicate_groups)}",
            f"Total duplicate videos: {sum(len(group.videos) - 1 for group in duplicate_groups)}",
            f"Total wasted space: {sum(sum(v.size for v in group.videos[1:]) for group in duplicate_groups) / (1024**3):.2f} GB",
            "\n=== Duplicate Groups ==="
        ]
        
        for i, group in enumerate(duplicate_groups):
            report_lines.append(f"\nGroup {i+1} - Similarity: {group.similarity_score:.1f}%")
            report_lines.append(f"Best Version: {group.best_version.path}")
            report_lines.append(f"Resolution: {group.best_version.resolution[0]}x{group.best_version.resolution[1]}")
            report_lines.append(f"Size: {group.best_version.size / (1024**2):.2f} MB")
            report_lines.append("\nDuplicates:")
            
            for video in group.videos:
                if video != group.best_version:
                    report_lines.append(f"- {video.path}")
                    report_lines.append(f"  Resolution: {video.resolution[0]}x{video.resolution[1]}")
                    report_lines.append(f"  Size: {video.size / (1024**2):.2f} MB")
        
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
        
        # Display videos with numbering
        videos = group.videos
        best_video = group.best_version
        
        print("\n[KEEP] Best version:")
        print(f"* {best_video.path}")
        print(f"  Size: {best_video.size / (1024**2):.2f} MB")
        print(f"  Resolution: {best_video.resolution[0]}x{best_video.resolution[1]}")
        
        print("\n[DUPLICATES]")
        for j, video in enumerate(videos):
            if video != best_video:
                print(f"{j+1}. {video.path}")
                print(f"   Size: {video.size / (1024**2):.2f} MB")
                print(f"   Resolution: {video.resolution[0]}x{video.resolution[1]}")
        
        # Ask for action
        while True:
            print("\nActions:")
            print("k - Keep all (skip this group)")
            print("d - Delete all duplicates")
            print("s - Select different video to keep")
            print("n - Next group")
            print("q - Quit")
            
            choice = input("\nEnter action: ").strip().lower()
            
            if choice == 'k':
                print("Keeping all videos in this group")
                break
                
            elif choice == 'd':
                confirm = input("Are you sure you want to delete all duplicates? (y/n): ")
                if confirm.lower() == 'y':
                    for video in videos:
                        if video != best_video:
                            try:
                                print(f"Deleting {video.path}...")
                                video.path.unlink()
                            except Exception as e:
                                print(f"Error deleting {video.path}: {e}")
                    print("Duplicates deleted")
                    break
                else:
                    print("Deletion cancelled")
                    
            elif choice == 's':
                print("\nSelect video to keep:")
                for j, video in enumerate(videos):
                    print(f"{j+1}. {video.path}")
                
                try:
                    selection = int(input("\nEnter number: "))
                    if 1 <= selection <= len(videos):
                        best_video = videos[selection-1]
                        print(f"Selected: {best_video.path}")
                        
                        confirm = input("Delete all other videos? (y/n): ")
                        if confirm.lower() == 'y':
                            for video in videos:
                                if video != best_video:
                                    try:
                                        print(f"Deleting {video.path}...")
                                        video.path.unlink()
                                    except Exception as e:
                                        print(f"Error deleting {video.path}: {e}")
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
    """Move duplicate videos to a target directory."""
    for i, group in enumerate(duplicate_groups):
        # Create a subdirectory for each group
        group_dir = target_dir / f"group_{i+1}"
        group_dir.mkdir(exist_ok=True)
        
        # Move duplicates, keep the best version in place
        best_video = group.best_version
        for video in group.videos:
            if video != best_video:
                try:
                    target_path = group_dir / video.path.name
                    # Ensure unique name in target directory
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = group_dir / f"{stem}_{video.hash_id[:8]}{suffix}"
                    
                    # Move the file
                    shutil.move(str(video.path), str(target_path))
                    logger.info(f"Moved: {video.path} -> {target_path}")
                except Exception as e:
                    logger.error(f"Error moving {video.path}: {e}")


def create_symlinks(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Create symbolic links for duplicates."""
    for i, group in enumerate(duplicate_groups):
        best_video = group.best_version
        
        # Create links for duplicates
        for video in group.videos:
            if video != best_video:
                try:
                    # Create target path
                    target_path = target_dir / video.path.name
                    # Ensure unique name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = target_dir / f"{stem}_{video.hash_id[:8]}{suffix}"
                    
                    # Create symbolic link
                    target_path.symlink_to(video.path.absolute())
                    logger.info(f"Created symlink: {target_path} -> {video.path}")
                except Exception as e:
                    logger.error(f"Error creating symlink for {video.path}: {e}")


def create_hardlinks(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Create hard links for duplicates."""
    for i, group in enumerate(duplicate_groups):
        best_video = group.best_version
        
        # Create hard links for duplicates
        for video in group.videos:
            if video != best_video:
                try:
                    # Create hard link path
                    target_path = target_dir / video.path.name
                    # Ensure unique name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = target_dir / f"{stem}_{video.hash_id[:8]}{suffix}"
                    
                    # Create hard link
                    os.link(str(video.path.absolute()), str(target_path))
                    logger.info(f"Created hardlink: {target_path} -> {video.path}")
                except Exception as e:
                    logger.error(f"Error creating hardlink for {video.path}: {e}")


def delete_duplicates(duplicate_groups: List[DuplicateGroup]) -> None:
    """Delete duplicate videos, keeping the best version."""
    total_deleted = 0
    total_freed = 0
    
    for group in duplicate_groups:
        best_video = group.best_version
        
        for video in group.videos:
            if video != best_video:
                try:
                    logger.info(f"Deleting: {video.path}")
                    video.path.unlink()
                    total_deleted += 1
                    total_freed += video.size
                except Exception as e:
                    logger.error(f"Error deleting {video.path}: {e}")
    
    logger.info(f"Deleted {total_deleted} duplicate videos")
    logger.info(f"Freed {total_freed / (1024**3):.2f} GB of space")


def generate_action_script(duplicate_groups: List[DuplicateGroup], 
                          script_type: str = 'bash',
                          output_file: Optional[str] = None) -> None:
    """Generate a script to handle duplicates."""
    if script_type == 'bash':
        # Generate bash script
        script_lines = [
            "#!/bin/bash",
            "# Generated by videodedup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate videos",
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
            best_video = group.best_version
            
            script_lines.append(f"echo \"Group {i+1}/{len(duplicate_groups)}\"")
            script_lines.append(f"echo \"Keeping: {best_video.path}\"")
            
            for video in group.videos:
                if video != best_video:
                    script_lines.append("if [ \"$DRY_RUN\" = \"true\" ]; then")
                    script_lines.append(f"    echo \"Would delete: {video.path}\"")
                    script_lines.append("else")
                    script_lines.append(f"    echo \"Deleting: {video.path}\"")
                    script_lines.append(f"    rm \"{video.path}\"")
                    script_lines.append("fi")
            
            script_lines.append("")
        
        script_lines.append("echo \"Finished!\"")
        
    elif script_type == 'powershell':
        # Generate PowerShell script
        script_lines = [
            "# Generated by videodedup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate videos",
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
            best_video = group.best_version
            
            script_lines.append(f"Write-Host \"Group {i+1}/{len(duplicate_groups)}\" -ForegroundColor Cyan")
            script_lines.append(f"Write-Host \"Keeping: {best_video.path}\" -ForegroundColor Green")
            
            for video in group.videos:
                if video != best_video:
                    script_lines.append("if ($DryRun) {")
                    script_lines.append(f"    Write-Host \"Would delete: {video.path}\" -ForegroundColor Yellow")
                    script_lines.append("} else {")
                    script_lines.append(f"    Write-Host \"Deleting: {video.path}\" -ForegroundColor Red")
                    script_lines.append(f"    Remove-Item -Path \"{video.path}\" -Force")
                    script_lines.append("}")
            
            script_lines.append("")
        
        script_lines.append("Write-Host \"Finished!\" -ForegroundColor Green")
        
    else:  # Python script
        # Generate Python script
        script_lines = [
            "#!/usr/bin/env python3",
            "# Generated by videodedup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate videos",
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
            "    parser = argparse.ArgumentParser(description='Delete duplicate videos')",
            "    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no deletion)')",
            "    args = parser.parse_args()",
            "",
            "    print('Starting duplicate removal...')",
            ""
        ]
        
        for i, group in enumerate(duplicate_groups):
            best_video = group.best_version
            
            script_lines.append(f"    print(f'Group {i+1}/{len(duplicate_groups)}')")
            script_lines.append(f"    print(f'Keeping: {best_video.path}')")
            
            for video in group.videos:
                if video != best_video:
                    script_lines.append("    if args.dry_run:")
                    script_lines.append(f"        print(f'Would delete: {video.path}')")
                    script_lines.append("    else:")
                    script_lines.append(f"        print(f'Deleting: {video.path}')")
                    script_lines.append(f"        try:")
                    script_lines.append(f"            os.remove(r'{video.path}')")
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


# ======= Main Function and CLI =======

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="""videodedup - Intelligent Video Deduplication Tool

This tool helps you find and manage duplicate videos, even if they have different names,
resolutions, or encodings. It uses perceptual hashing and content analysis to identify
duplicates with high accuracy.

Examples:
  # Basic usage - generate a report of duplicates
  videodedup.py /path/to/videos

  # Scan multiple directories and generate HTML report
  videodedup.py /videos/movies /videos/shows --output-format html --html-report-dir ./report

  # Interactive mode to review and handle duplicates
  videodedup.py /path/to/videos --action interactive

  # Move duplicates to a separate directory, keeping originals
  videodedup.py /path/to/videos --action move --target-dir ./duplicates

  # Generate a script to handle duplicates later
  videodedup.py /path/to/videos --action script --script-type bash --output-file cleanup.sh

  # Delete duplicates (requires --force-delete for safety)
  videodedup.py /path/to/videos --action delete --force-delete""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input directories
    parser.add_argument(
        'directories', 
        nargs='+', 
        type=str, 
        help='Directories to scan for videos'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--duration-threshold', 
        type=float, 
        default=1.0,
        help='Percentage threshold for duration matching'
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
            version=f'videodedup {VERSION}'
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
    missing_deps = []
    if not IMAGEHASH_AVAILABLE:
        missing_deps.append("imagehash")
    if not TQDM_AVAILABLE:
        missing_deps.append("tqdm")
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.warning("Install with: pip install " + " ".join(missing_deps))
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    # Start the analysis process
    start_time = time.time()
    
    # Convert directories to Path objects
    directories = [Path(d).resolve() for d in args.directories]
    
    # Find all video files
    video_files = find_video_files(directories, args.recursive)
    
    if not video_files:
        logger.error("No video files found in the specified directories")
        return 1
    
    # Find duplicates
    duplicate_groups = analyze_videos(video_files, args)
    
    # Handle duplicates according to the specified action
    handle_duplicates(duplicate_groups, args)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
