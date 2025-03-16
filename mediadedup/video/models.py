"""
Video-specific data models.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from PIL import Image
import imagehash

from ..common.models import MediaFile
from . import check_quicksync_available

logger = logging.getLogger(__name__)

class VideoFile(MediaFile):
    """Represents a video file with its metadata and analysis results."""
    def __init__(self, path: Path):
        super().__init__(path)
        self.duration: float = 0.0
        self.resolution: Tuple[int, int] = (0, 0)
        self.bitrate: int = 0
        self.frame_rate: float = 0.0
        self.frame_hashes: Dict[float, Any] = {}
        self.audio_fingerprint: Optional[List[float]] = None
    
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
        """Extract perceptual hashes from frames at specified positions."""
        try:
            # Create a temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Build the select filter for all frames at once
                select_expr = '+'.join([f"eq(t,{pos})" for pos in frame_positions])
                
                # Build FFmpeg command with hardware acceleration and downscaling
                cmd = [
                    "ffmpeg",
                    "-hwaccel", "auto" if use_quicksync else "none",  # Enable hardware acceleration
                    "-i", str(self.path),
                    "-vf", f"select='{select_expr}',scale=320:-1",  # Select frames and downscale
                    "-vsync", "0",
                    "-frame_pts", "1",  # Include presentation timestamps
                    "-f", "image2",
                    f"{temp_dir}/frame_%d.jpg"
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Create a mapping of frame numbers to positions
                    frame_map = {}
                    for i, pos in enumerate(frame_positions):
                        frame_map[i + 1] = pos  # frame numbers start at 1
                    
                    # Load and hash all extracted frames
                    frames = {}
                    extracted_frames = sorted(Path(temp_dir).glob("frame_*.jpg"))
                    
                    # Verify we got the expected number of frames
                    if len(extracted_frames) != len(frame_positions):
                        logger.warning(f"Expected {len(frame_positions)} frames but got {len(extracted_frames)}")
                    
                    # Process each extracted frame
                    for frame_file in extracted_frames:
                        try:
                            with Image.open(frame_file) as img:
                                # Get the frame number from the filename
                                frame_num = int(frame_file.stem.split('_')[1])
                                frames[frame_num] = img.copy()
                        except Exception as e:
                            logger.warning(f"Error loading frame {frame_file}: {e}")
                            continue
            
                    # Generate hashes for extracted frames
                    for frame_num, frame in frames.items():
                        try:
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
                            
                            # Map frame number to position using our frame map
                            if frame_num in frame_map:
                                pos = frame_map[frame_num]
                                self.frame_hashes[pos] = frame_hash
                            else:
                                logger.warning(f"Frame number {frame_num} not found in frame map")
                        except Exception as e:
                            logger.warning(f"Error hashing frame {frame_num}: {e}")
                            continue
                    
                    return self.frame_hashes
                    
                except Exception as e:
                    logger.warning(f"Error extracting frames: {e}")
                    return {}
            
        except Exception as e:
            logger.warning(f"Error extracting frame hashes from {self.path}: {e}")
            return {}
    
    def detect_scenes(self, threshold: float = 0.3) -> List[float]:
        """Detect scene changes in the video."""
        cmd = [
            "ffmpeg", "-i", str(self.path), 
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
            logger.warning(f"Error detecting scenes in {self.path}: {e}")
            return []
    
    def extract_audio_fingerprint(self) -> List[float]:
        """Extract audio fingerprint from the video."""
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                # Extract audio to WAV
                subprocess.run([
                    "ffmpeg", "-i", str(self.path), 
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
                
                self.audio_fingerprint = volume_levels
                return volume_levels
        except Exception as e:
            logger.warning(f"Error extracting audio fingerprint from {self.path}: {e}")
            return []
    
    def get_metadata(self) -> Dict:
        """Get complete metadata including video-specific information."""
        metadata = super().get_metadata()
        metadata.update({
            'duration': self.duration,
            'resolution': f"{self.resolution[0]}x{self.resolution[1]}",
            'bitrate': self.bitrate,
            'frame_rate': self.frame_rate,
            'frame_hashes': {str(k): str(v) for k, v in self.frame_hashes.items()},
            'audio_fingerprint': self.audio_fingerprint
        })
        return metadata

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'path': str(self.path),
            'size': self.size,
            'content_score': self.content_score,
            'hash_id': self.hash_id,
            'duration': self.duration,
            'resolution': list(self.resolution),
            'bitrate': self.bitrate,
            'frame_rate': self.frame_rate,
            'frame_hashes': {str(k): str(v) for k, v in self.frame_hashes.items()},
            'audio_fingerprint': self.audio_fingerprint
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoFile':
        """Create a VideoFile instance from dictionary data."""
        video = cls(Path(data['path']))
        video.size = data['size']
        video.content_score = data['content_score']
        video.hash_id = data['hash_id']
        video.duration = data['duration']
        video.resolution = tuple(data['resolution'])
        video.bitrate = data['bitrate']
        video.frame_rate = data['frame_rate']
        video.frame_hashes = {float(k): imagehash.hex_to_hash(v) for k, v in data['frame_hashes'].items()}
        video.audio_fingerprint = data['audio_fingerprint']
        return video
