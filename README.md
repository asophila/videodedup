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

- 🔍 **Multi-tier Detection**: Uses CRC32/SHA-256 for exact matches, then perceptual analysis for similar content
- 🎬 **Scene Detection**: Uses FFmpeg's scene detection for accurate video matching
- 🔊 **Audio Analysis**: Includes audio fingerprinting for better accuracy
- 🚀 **Hardware Acceleration**: Supports Intel QuickSync for faster frame extraction
- ⏱️ **Duration-based Filtering**: Groups videos by similar duration for efficient comparison
- ⚡ **Fast Exact Matching**: Instantly identifies bit-for-bit identical files using CRC

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

# Install package and dependencies
pip install -e .
```

### Verify Installation

```bash
# Test VideoDeDup
videodedup --version

# Test ImageDeDup
imagededup --version
```

## 🔧 Usage Examples

### VideoDeDup Examples

#### Basic Video Scan and Report

```bash
videodedup /path/to/videos
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
videodedup /path/to/videos --output-format html --html-report-dir ./report
```

Creates a rich HTML report with:
- Video thumbnails from beginning, middle, and end
- Side-by-side visual comparison
- Detailed metadata
- Quality comparisons
- Interactive layout

#### Interactive Video Mode

```bash
videodedup /path/to/videos --action interactive
```

Provides an interactive CLI interface to:
- Review each duplicate group
- Compare video properties
- Choose which version to keep
- Safely delete duplicates

#### Move Video Duplicates to Separate Directory

```bash
videodedup /path/to/videos --action move --target-dir /path/to/duplicates
```

### ImageDeDup Examples

#### Basic Image Scan and Report

```bash
imagededup /path/to/images
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
imagededup /path/to/images --output-format html --html-report-dir ./report
```

Creates a rich HTML report with:
- Image thumbnails for easy comparison
- Side-by-side visual comparison
- Detailed metadata
- Quality comparisons
- Interactive layout

#### Interactive Image Mode

```bash
imagededup /path/to/images --action interactive
```

Provides an interactive CLI interface to:
- Review each duplicate group
- Compare image properties
- Choose which version to keep
- Safely delete duplicates

#### Move Image Duplicates to Separate Directory

```bash
imagededup /path/to/images --action move --target-dir /path/to/duplicates
```

## 🔄 Running Both Tools Together

### Sequential Execution

You can run both tools on the same directory to find duplicates of both images and videos. The tools are designed to work together:

- Each tool only processes its relevant file types
- Actions and reports are compatible
- Cache is managed separately for each tool

Basic sequential execution:
```bash
# Generate reports for both images and videos
imagededup /path/to/directory --output-file images-report.txt
videodedup /path/to/directory --output-file videos-report.txt

# Interactive mode for both types
imagededup /path/to/directory --action interactive
videodedup /path/to/directory --action interactive

# Move duplicates to separate directories
imagededup /path/to/directory --action move --target-dir /path/to/image-duplicates
videodedup /path/to/directory --action move --target-dir /path/to/video-duplicates
```

### Shell Script Helper

Here's a helpful shell script (`mediadedup.sh`) that runs both tools with the same parameters:

```bash
#!/bin/bash

# Check if at least a directory is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 DIRECTORY [OPTIONS]"
    echo "Example: $0 /path/to/media --action move --target-dir /path/to/duplicates"
    exit 1
fi

# Get the directory from first argument
DIR="$1"
shift  # Remove the directory from arguments

# Create separate target directories if needed
if [[ "$*" == *"--target-dir"* ]]; then
    for arg in "$@"; do
        if [ "$prev_arg" = "--target-dir" ]; then
            TARGET_DIR="$arg"
            # Replace the target dir in the arguments for each tool
            IMAGE_ARGS=${@//$TARGET_DIR/$TARGET_DIR\/images}
            VIDEO_ARGS=${@//$TARGET_DIR/$TARGET_DIR\/videos}
            break
        fi
        prev_arg="$arg"
    done
else
    IMAGE_ARGS="$@"
    VIDEO_ARGS="$@"
fi

# Run image deduplication
echo "🖼️ Running image deduplication..."
imagededup "$DIR" $IMAGE_ARGS

# Run video deduplication
echo "🎥 Running video deduplication..."
videodedup "$DIR" $VIDEO_ARGS
```

Make the script executable and use it:
```bash
chmod +x mediadedup.sh

# Basic usage
./mediadedup.sh /path/to/media

# With options
./mediadedup.sh /path/to/media --action move --target-dir /path/to/duplicates
```

When using the script with `--target-dir`, it automatically creates separate subdirectories for images and videos:
```
/path/to/duplicates/
├── images/
│   ├── group_1/
│   └── group_2/
└── videos/
    ├── group_1/
    └── group_2/
```

### Report Compatibility

Both tools support the same report formats with media-specific information:

- **Text Reports**: Human-readable format with media-specific details
- **JSON Reports**: Can be combined programmatically for unified analysis
- **CSV Reports**: Compatible format for spreadsheet analysis
- **HTML Reports**: Separate reports with appropriate thumbnails and previews

## 📚 Command Reference

### Common Options

```bash
--output-format FORMAT     # Output format: text, json, csv, html
--html-report-dir DIR      # Directory to store HTML report with thumbnails
--output-file FILE        # Output file path (default: stdout)
--action ACTION           # Action: report, interactive, move, symlink, hardlink, delete, script
--target-dir DIR         # Target directory for move/symlink/hardlink actions
--force-delete           # Force deletion of duplicates (required for delete action)
--script-type TYPE       # Type of script to generate: bash, powershell, python
--recursive              # Scan directories recursively (default: True)
--no-recursive          # Do not scan directories recursively
--verbose, -v           # Increase verbosity level (-v for detailed, -vv for debug)
--version              # Show program version
--clear-cache          # Clear cache before running
```

### VideoDeDup Options

```bash
--duration-threshold FLOAT   # Percentage threshold for duration matching (default: 1.0)
--similarity-threshold FLOAT # Percentage threshold for similarity detection (default: 85.0)
--hash-algorithm ALGORITHM   # Hash algorithm: phash, dhash, whash, average_hash (default: phash)
--skip-crc                  # Skip CRC-based exact duplicate detection (use only perceptual analysis)
```

### ImageDeDup Options

```bash
--dimension-threshold FLOAT  # Percentage threshold for dimension matching (default: 0.1)
--similarity-threshold FLOAT # Percentage threshold for similarity detection (default: 85.0)
--hash-algorithm ALGORITHM   # Hash algorithm: phash, dhash, whash, average_hash (default: phash)
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

4. **CRC Detection**
   - CRC-based detection is enabled by default for instant exact match detection
   - Use `--skip-crc` if you only want to find similar videos, not exact duplicates
   - CRC detection is extremely fast and can significantly reduce processing time

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

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
