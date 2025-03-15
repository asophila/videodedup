# Media Deduplication Tools 🎥 🖼️

A powerful suite of intelligent media deduplication tools that help you find and manage duplicate files, even when they have different names, resolutions, or encodings.

## Tools in this Repository

### VideoDeDup 🎥

A video deduplication tool that uses advanced perceptual hashing and scene detection to find duplicate videos and help you clean up your video library.

### ImageDeDup 🖼️

An image deduplication tool that uses perceptual hashing and color analysis to find duplicate images, even when they have different names, formats, or dimensions.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

### Common Features

- 🔍 **Smart Detection**: Identifies duplicates even with different names, formats, or resolutions
- 📊 **Rich Reports**: Generate HTML reports with thumbnails for visual comparison
- 🎨 **Quality-Aware**: Automatically determines the best version based on quality metrics
- 💪 **Efficient Processing**: Parallel processing and smart filtering for better performance
- 🛠️ **Flexible Actions**: Multiple ways to handle duplicates
  - Generate reports (Text, JSON, CSV, HTML)
  - Move duplicates to a separate directory
  - Create symbolic or hard links
  - Interactive review and deletion
  - Generate cleanup scripts

### VideoDeDup Features

- 🎬 **Scene Detection**: Uses FFmpeg's scene detection for accurate video matching
- 🔊 **Audio Analysis**: Includes audio fingerprinting for better accuracy
- 🚀 **Hardware Acceleration**: Supports Intel QuickSync for faster frame extraction
- ⏱️ **Duration-based Filtering**: Groups videos by similar duration for efficient comparison

### ImageDeDup Features

- 🌈 **Color Analysis**: Uses both perceptual and color-based hashing
- 📐 **Dimension Matching**: Groups images by similar dimensions for efficient comparison
- 🔄 **Format Support**: Handles all common image formats (JPEG, PNG, WebP, etc.)
- 🎯 **High Precision**: Specialized algorithms for image-specific comparison

## 📦 Installation

### Required Dependencies

1. **Python 3.7+**
   - Required for both tools
   - Download from [python.org](https://www.python.org/downloads/)

2. **FFmpeg** (required for VideoDeDup)
   Ubuntu/Debian:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   macOS:
   ```bash
   brew install ffmpeg
   ```

   Windows:
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add to system PATH

### Install from Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/videodedup.git
cd videodedup

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test VideoDeDup
python videodedup.py --version

# Test ImageDeDup
python imagededup.py --version
```

## 🔧 How They Work

### VideoDeDup - Video Deduplication

VideoDeDup uses a sophisticated multi-stage approach to find duplicate videos:

1. **Pre-filtering** 📥
   ```
   Videos/
   ├── movie-1080p.mkv (2h 1m 30s)
   ├── movie-720p.mp4  (2h 1m 30s)  ┐
   └── other.mp4       (1h 30m 20s) ┘ Different duration, skipped
   ```
   Groups videos by similar duration (default: ±1%) to reduce comparison load

2. **Frame Analysis** 🖼️
   - Extracts frames at strategic points (10%, 30%, 50%, 70%, 90% of duration)
   - Generates perceptual hashes (pHash) of frames
   - Compares hash distances to identify potential matches
   ```python
   Frame Positions: [0.1, 0.3, 0.5, 0.7, 0.9]
   Hash Distance < 10 = Potential Match
   ```

3. **Scene Detection** 🎬
   - Uses FFmpeg's scene detection to identify content changes
   - Helps verify matches across different encodings
   ```
   Scene Changes:
   Video 1: [0:30, 1:45, 2:15, ...]
   Video 2: [0:31, 1:44, 2:16, ...]  ← Similar pattern
   ```

4. **Quality Assessment** ⭐
   - Evaluates video quality based on:
     - Resolution (e.g., 1920x1080 vs 1280x720)
     - Bitrate (e.g., 8000 kbps vs 2000 kbps)
     - Frame rate (e.g., 60 fps vs 30 fps)
   - Scores each version to determine the best copy

### ImageDeDup - Image Deduplication

ImageDeDup uses an efficient approach to find duplicate images:

1. **Pre-filtering** 📥
   ```
   Images/
   ├── photo-4k.jpg     (3840x2160)
   ├── photo-2k.jpg     (2560x1440)  ┐
   └── other.jpg        (1920x1080)  ┘ Different dimensions, grouped separately
   ```
   Groups images by similar dimensions to reduce comparison load

2. **Perceptual Analysis** 🖼️
   - Calculates perceptual hash (pHash) of each image
   - Generates color hash for additional comparison
   - Compares hash distances to identify potential matches
   ```python
   Perceptual Hash Weight: 70%
   Color Hash Weight: 30%
   Similarity Score >= 85% = Potential Match
   ```

3. **Quality Assessment** ⭐
   - Evaluates image quality based on:
     - Resolution (e.g., 3840x2160 vs 1920x1080)
     - File size and format
     - Color depth and mode
   - Scores each version to determine the best copy

## 📦 Installation

1. **Required Dependencies**

   Ubuntu/Debian:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   macOS:
   ```bash
   brew install ffmpeg
   ```

   Windows:
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add to system PATH

2. **Install VideoDeDup**

   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/videodedup.git
   cd videodedup

   # Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Verify Installation**

   ```bash
   python videodedup.py --version
   ```

## 📚 Usage Examples

### VideoDeDup Examples

#### Basic Video Scan and Report

```bash
python videodedup.py /path/to/videos
```

Example output:
```
=== Video Deduplication Report ===
Generated on: 2025-03-14T21:30:00
Total duplicate groups: 3
Total duplicate videos: 5
Total wasted space: 8.45 GB

=== Duplicate Groups ===
Group 1 - Similarity: 98.5%
Best Version: /videos/movie-1080p.mkv
Resolution: 1920x1080
Size: 4.2 GB

Duplicates:
- /videos/movie-720p.mp4
  Resolution: 1280x720
  Size: 2.1 GB
```

#### Generate Video HTML Report with Thumbnails

```bash
python videodedup.py /path/to/videos --output-format html --html-report-dir ./report
```

Creates a rich HTML report with:
- Video thumbnails from beginning, middle, and end
- Side-by-side visual comparison
- Detailed metadata
- Quality comparisons
- Interactive layout

#### Interactive Video Mode

```bash
python videodedup.py /path/to/videos --action interactive
```

Provides an interactive CLI interface to:
- Review each duplicate group
- Compare video properties
- Choose which version to keep
- Safely delete duplicates

#### Move Video Duplicates to Separate Directory

```bash
python videodedup.py /path/to/videos --action move --target-dir /path/to/duplicates
```

Directory structure after moving:
```
/path/to/duplicates/
├── group_1/
│   ├── movie-720p.mp4
│   └── movie-480p.avi
├── group_2/
│   └── video2-copy.mkv
└── group_3/
    ├── duplicate1.mp4
    └── duplicate2.webm
```

#### Generate Video Cleanup Script

```bash
python videodedup.py /path/to/videos --action script --script-type bash --output-file cleanup.sh
```

Creates a script for manual review and execution:
```bash
#!/bin/bash
# Generated by VideoDeDup
# Date: 2025-03-14T21:30:00

# Execute with: bash cleanup.sh
# For dry-run: bash cleanup.sh --dry-run

echo "Group 1"
echo "Keeping: /videos/movie-1080p.mkv"
rm "/videos/movie-720p.mp4"
...
```

### ImageDeDup Examples

#### Basic Image Scan and Report

```bash
python imagededup.py /path/to/images
```

Example output:
```
=== Image Deduplication Report ===
Generated on: 2025-03-14T21:30:00
Total duplicate groups: 5
Total duplicate images: 12
Total wasted space: 145.2 MB

=== Duplicate Groups ===
Group 1 - Similarity: 99.2%
Best Version: /photos/image-4k.jpg
Dimensions: 3840x2160
Size: 8.5 MB

Duplicates:
- /photos/image-2k.jpg
  Dimensions: 2560x1440
  Size: 4.2 MB
```

#### Generate Image HTML Report with Thumbnails

```bash
python imagededup.py /path/to/images --output-format html --html-report-dir ./report
```

Creates a rich HTML report with:
- Image thumbnails for easy comparison
- Side-by-side visual comparison
- Detailed metadata (dimensions, format, color mode)
- Quality comparisons
- Interactive layout

#### Interactive Image Mode

```bash
python imagededup.py /path/to/images --action interactive
```

Provides an interactive CLI interface to:
- Review each duplicate group
- Compare image properties
- Choose which version to keep
- Safely delete duplicates

#### Move Image Duplicates to Separate Directory

```bash
python imagededup.py /path/to/images --action move --target-dir /path/to/duplicates
```

Directory structure after moving:
```
/path/to/duplicates/
├── group_1/
│   ├── photo-2k.jpg
│   └── photo-1080p.jpg
├── group_2/
│   └── image-copy.png
└── group_3/
    ├── duplicate1.jpg
    └── duplicate2.webp
```

#### Generate Image Cleanup Script

```bash
python imagededup.py /path/to/images --action script --script-type bash --output-file cleanup.sh
```

Creates a script for manual review and execution:
```bash
#!/bin/bash
# Generated by ImageDeDup
# Date: 2025-03-14T21:30:00

# Execute with: bash cleanup.sh
# For dry-run: bash cleanup.sh --dry-run

echo "Group 1"
echo "Keeping: /photos/image-4k.jpg"
rm "/photos/image-2k.jpg"
...
```

## 🎮 Command Reference

### VideoDeDup Options

#### Analysis Options

```bash
--duration-threshold FLOAT   # Percentage threshold for duration matching (default: 1.0)
--similarity-threshold FLOAT # Percentage threshold for similarity detection (default: 85.0)
--hash-algorithm ALGORITHM   # Hash algorithm: phash, dhash, whash, average_hash (default: phash)
--recursive                  # Scan directories recursively (default: True)
--no-recursive              # Do not scan directories recursively
```

#### Action Options

```bash
--action ACTION             # Action to take: report, interactive, move, symlink, hardlink, delete, script
--target-dir DIR           # Target directory for move/symlink/hardlink actions
--force-delete             # Force deletion of duplicates (required for delete action)
--script-type TYPE         # Type of script to generate: bash, powershell, python
```

#### Output Options

```bash
--output-format FORMAT     # Output format: text, json, csv, html
--html-report-dir DIR      # Directory to store HTML report with thumbnails
--output-file FILE        # Output file path (default: stdout)
```

#### Miscellaneous

### ImageDeDup Options

#### Analysis Options

```bash
--dimension-threshold FLOAT # Percentage threshold for dimension matching (default: 0.1)
--similarity-threshold FLOAT # Percentage threshold for similarity detection (default: 85.0)
--hash-algorithm ALGORITHM   # Hash algorithm: phash, dhash, whash, average_hash (default: phash)
--recursive                  # Scan directories recursively (default: True)
--no-recursive              # Do not scan directories recursively
```

#### Action Options

```bash
--action ACTION             # Action to take: report, interactive, move, symlink, hardlink, delete, script
--target-dir DIR           # Target directory for move/symlink/hardlink actions
--force-delete             # Force deletion of duplicates (required for delete action)
--script-type TYPE         # Type of script to generate: bash, powershell, python
```

#### Output Options

```bash
--output-format FORMAT     # Output format: text, json, csv, html
--html-report-dir DIR      # Directory to store HTML report with thumbnails
--output-file FILE        # Output file path (default: stdout)
```

#### Miscellaneous

```bash
--log-level LEVEL         # Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
--version                 # Show program version
--clear-cache            # Clear cache before running
```

## 💡 Performance Tips

### VideoDeDup Tips

1. **Use Hardware Acceleration**
   - Intel QuickSync is automatically used when available
   - Significantly speeds up frame extraction

2. **Optimize Memory Usage**
   - Process videos in batches if memory is limited
   - Use duration grouping to reduce comparison load

3. **Adjust Thresholds**
   - Lower `--similarity-threshold` for more matches (may increase false positives)
   - Increase `--duration-threshold` for videos with slight duration differences

### ImageDeDup Tips

1. **Optimize Memory Usage**
   - Process images in batches for large collections
   - Use dimension grouping to reduce comparison load

2. **Adjust Thresholds**
   - Lower `--similarity-threshold` for more matches (may increase false positives)
   - Adjust `--dimension-threshold` for scaled/resized images

3. **Color Analysis**
   - Color hash helps identify similar images with different compression
   - Particularly useful for photos and artwork

### Common Tips

1. **Cache Management**
   - Use `--clear-cache` when changing analysis parameters
   - Cache speeds up subsequent runs on the same files

2. **File Organization**
   - Group similar files in directories for faster processing
   - Use recursive mode carefully with large directory trees

## ❗ Troubleshooting

### VideoDeDup Issues

1. **FFmpeg Not Found**
   - Ensure FFmpeg is installed and in system PATH
   - Test with `ffmpeg -version` and `ffprobe -version`

2. **Video Processing Errors**
   - Check video file integrity
   - Ensure sufficient disk space for frame extraction

### ImageDeDup Issues

1. **Image Format Errors**
   - Ensure image files are not corrupted
   - Check for proper file extensions
   - Convert non-standard formats to JPEG/PNG

2. **Color Mode Issues**
   - Some images may need conversion to RGB
   - CMYK images are automatically converted

### Common Issues

1. **Memory Issues**
   - Process smaller directories at a time
   - Increase grouping thresholds for smaller batches
   - Use 64-bit Python for large collections

2. **False Positives/Negatives**
   - Adjust `--similarity-threshold` (default: 85.0)
   - Try different hash algorithms with `--hash-algorithm`
   - Fine-tune grouping thresholds

3. **Slow Processing**
   - Use SSD for temporary files
   - Process files on local drives rather than network storage
   - Enable parallel processing with multiple cores

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
