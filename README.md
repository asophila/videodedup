# VideoDeDup 🎥

A powerful, intelligent video deduplication tool that finds duplicate videos even when they have different names, resolutions, or encodings. Using advanced perceptual hashing and scene detection, VideoDeDup helps you clean up your video library and reclaim valuable disk space.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🔍 **Smart Detection**: Identifies duplicates even with different names, formats, or resolutions
- 🚀 **Hardware Acceleration**: Supports Intel QuickSync for faster frame extraction
- 📊 **Rich Reports**: Generate HTML reports with video thumbnails for visual comparison
- 🎯 **Multi-stage Analysis**: Progressive refinement for accurate results
  - Duration-based pre-filtering
  - Perceptual hash comparison
  - Scene change detection
  - Audio fingerprinting
- 🎨 **Quality-Aware**: Automatically determines the best version based on resolution, bitrate, and other metrics
- 💪 **Efficient Processing**: Parallel processing and smart filtering for better performance
- 🛠️ **Flexible Actions**: Multiple ways to handle duplicates
  - Generate reports (Text, JSON, CSV, HTML)
  - Move duplicates to a separate directory
  - Create symbolic or hard links
  - Interactive review and deletion
  - Generate cleanup scripts

## 🔧 How It Works

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

## 📦 Installation

1. **Install FFmpeg** (required)

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

### Basic Scan and Report

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

### Generate HTML Report with Thumbnails

```bash
python videodedup.py /path/to/videos --output-format html --html-report-dir ./report
```

Creates a rich HTML report with:
- Video thumbnails from beginning, middle, and end
- Side-by-side visual comparison
- Detailed metadata
- Quality comparisons
- Interactive layout

### Interactive Mode

```bash
python videodedup.py /path/to/videos --action interactive
```

Provides an interactive CLI interface to:
- Review each duplicate group
- Compare video properties
- Choose which version to keep
- Safely delete duplicates

### Move Duplicates to Separate Directory

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

### Generate Cleanup Script

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

## 🎮 Command Reference

### Analysis Options

```bash
--duration-threshold FLOAT   # Percentage threshold for duration matching (default: 1.0)
--similarity-threshold FLOAT # Percentage threshold for similarity detection (default: 85.0)
--hash-algorithm ALGORITHM   # Hash algorithm: phash, dhash, whash, average_hash (default: phash)
--recursive                  # Scan directories recursively (default: True)
--no-recursive              # Do not scan directories recursively
```

### Action Options

```bash
--action ACTION             # Action to take: report, interactive, move, symlink, hardlink, delete, script
--target-dir DIR           # Target directory for move/symlink/hardlink actions
--force-delete             # Force deletion of duplicates (required for delete action)
--script-type TYPE         # Type of script to generate: bash, powershell, python
```

### Output Options

```bash
--output-format FORMAT     # Output format: text, json, csv, html
--html-report-dir DIR      # Directory to store HTML report with thumbnails
--output-file FILE        # Output file path (default: stdout)
```

### Miscellaneous

```bash
--log-level LEVEL         # Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
--version                 # Show program version
--clear-cache            # Clear cache before running
```

## 💡 Performance Tips

1. **Use Hardware Acceleration**
   - Intel QuickSync is automatically used when available
   - Significantly speeds up frame extraction

2. **Optimize Memory Usage**
   - Process videos in batches if memory is limited
   - Use duration grouping to reduce comparison load

3. **Adjust Thresholds**
   - Lower `--similarity-threshold` for more matches (may increase false positives)
   - Increase `--duration-threshold` for videos with slight duration differences

4. **Cache Management**
   - Use `--clear-cache` when changing analysis parameters
   - Cache speeds up subsequent runs on the same files

## ❗ Troubleshooting

1. **FFmpeg Not Found**
   - Ensure FFmpeg is installed and in system PATH
   - Test with `ffmpeg -version` and `ffprobe -version`

2. **Memory Issues**
   - Process smaller directories at a time
   - Increase `--duration-threshold` to create smaller groups

3. **False Positives/Negatives**
   - Adjust `--similarity-threshold` (default: 85.0)
   - Try different hash algorithms with `--hash-algorithm`

4. **Slow Processing**
   - Enable hardware acceleration
   - Use SSD for temporary files
   - Process files on local drives rather than network storage

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
