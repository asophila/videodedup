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
