"""
Common actions for handling duplicate media files.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from . import utils
from .models import DuplicateGroup, MediaFile

logger = logging.getLogger(__name__)

def handle_duplicates(duplicate_groups: List[DuplicateGroup], args) -> None:
    """Handle duplicate files according to the specified action."""
    # Calculate total statistics
    total_files = sum(len(group.files) for group in duplicate_groups)
    total_duplicates = sum(len(group.files) - 1 for group in duplicate_groups)
    total_wasted_space = sum(
        sum(v.size for v in group.files[1:]) for group in duplicate_groups
    )
    
    # Always generate duplicates.txt
    duplicates_txt = Path('duplicates.txt')
    with open(duplicates_txt, 'w') as f:
        f.write("=== Files that can be safely deleted ===\n")
        f.write(f"Total duplicate files: {total_duplicates}\n")
        f.write(f"Total wasted space: {utils.format_size(total_wasted_space)}\n\n")
        
        for i, group in enumerate(duplicate_groups):
            f.write(f"\nGroup {i+1} - Similarity: {group.similarity_score:.1f}%\n")
            f.write(f"Keep: {group.best_version.get_display_path()}\n")
            f.write("Delete:\n")
            for file in group.files:
                if file != group.best_version:
                    f.write(f"- {file.get_display_path()}\n")
    
    if not duplicate_groups:
        if args.action == 'report':
            report_path = generate_report([], args.output_format, args.output_file, args.html_report_dir)
            if report_path:
                print(f"\nReport saved to: {report_path}")
        logger.info("No duplicates found.")
        return
    
    logger.info(f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicates")
    logger.info(f"Total wasted space: {utils.format_size(total_wasted_space)}")
    logger.info(f"List of duplicate files saved to: {duplicates_txt}")
    
    # Create output directory if needed
    output_dir = None
    if args.action in ['move', 'symlink', 'hardlink'] and args.target_dir:
        output_dir = Path(args.target_dir)
        utils.ensure_dir(output_dir)
    
    # Handle according to action
    if args.action == 'report':
        report_path = generate_report(duplicate_groups, args.output_format, args.output_file, args.html_report_dir)
        if report_path:
            print(f"\nReport saved to: {report_path}")
    
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
        delete_duplicates(duplicate_groups)
    
    elif args.action == 'script':
        script_path = generate_action_script(duplicate_groups, args.script_type, args.output_file)
        if script_path:
            print(f"\nScript saved to: {script_path}")

def generate_report(duplicate_groups: List[DuplicateGroup], 
                   format_type: str = 'text', 
                   output_file: Optional[Path] = None,
                   html_report_dir: Optional[Path] = None) -> Optional[Path]:
    """Generate a report of duplicate files."""
    if format_type == 'json':
        # Convert to JSON
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_groups': len(duplicate_groups),
            'total_duplicates': sum(len(group.files) - 1 for group in duplicate_groups),
            'total_wasted_space': sum(
                sum(v.size for v in group.files[1:]) for group in duplicate_groups
            ),
            'duplicate_groups': [group.to_dict() for group in duplicate_groups]
        }
        
        # Output to file or stdout
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.debug(f"JSON report saved to: {output_file.absolute()}")
            return output_file.absolute()
        else:
            print(json.dumps(json_data, indent=2))
            return None
    
    elif format_type == 'csv':
        # Generate CSV content
        csv_lines = ['group_id,similarity,file_path,is_best,size']
        
        for i, group in enumerate(duplicate_groups):
            for file in group.files:
                is_best = 'yes' if file == group.best_version else 'no'
                line = f"{i},{group.similarity_score:.1f},\"{file.path}\",{is_best},{file.size}"
                csv_lines.append(line)
        
        # Output to file or stdout
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(csv_lines))
            logger.debug(f"CSV report saved to: {output_file.absolute()}")
            return output_file.absolute()
        else:
            print('\n'.join(csv_lines))
            return None
    
    elif format_type == 'html':
        if not html_report_dir:
            logger.error("HTML report directory is required for HTML format")
            return None
            
        # Create report directory and frames subdirectory
        html_report_dir = Path(html_report_dir)
        frames_dir = html_report_dir / "frames"
        html_report_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(exist_ok=True)
        
        # Generate HTML content
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Media Deduplication Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".group { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }",
            ".best-version { background: #e8f5e9; padding: 10px; margin: 5px 0; }",
            ".duplicate { background: #ffebee; padding: 10px; margin: 5px 0; }",
            ".stats { font-weight: bold; margin: 10px 0; }",
            ".frames { display: flex; gap: 10px; margin: 10px 0; }",
            ".frame { text-align: center; }",
            ".frame img { max-width: 150px; height: auto; }",
            ".frame p { margin: 5px 0; font-size: 0.9em; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Media Deduplication Report</h1>",
            f"<p>Generated on: {datetime.now().isoformat()}</p>",
            "<div class='stats'>",
            f"<p>Total duplicate groups: {len(duplicate_groups)}</p>",
            f"<p>Total duplicate files: {sum(len(group.files) - 1 for group in duplicate_groups)}</p>",
            f"<p>Total wasted space: {utils.format_size(sum(sum(v.size for v in group.files[1:]) for group in duplicate_groups))}</p>",
            "</div>",
            "<h2>Duplicate Groups</h2>"
        ]
        
        # Group videos by similarity score
        similarity_groups = {}
        for group in duplicate_groups:
            score = group.similarity_score
            if score not in similarity_groups:
                similarity_groups[score] = []
            similarity_groups[score].append(group)
        
        # Process groups by similarity score (highest first)
        for score in sorted(similarity_groups.keys(), reverse=True):
            groups = similarity_groups[score]
            html_lines.append(f"<h3>Similarity Score: {score:.1f}%</h3>")
            
            for i, group in enumerate(groups):
                html_lines.extend([
                    f"<div class='group'>",
                    f"<h4>Group {i+1}</h4>"
                ])
                
                # Extract frames for the best version
                frames_path = frames_dir / f"group_{i}"
                frames_path.mkdir(exist_ok=True)
                
                # Extract frames at beginning, middle, and end
                frame_positions = [0.1, 0.5, 0.9]  # 10%, 50%, 90%
                group.best_version.extract_frame_hashes(
                    frame_positions,
                    save_frames=True,
                    output_dir=frames_path,
                    frame_width=150
                )
                
                # Show frames
                frame_positions = [0.1, 0.5, 0.9]  # 10%, 50%, 90%
                frame_times = [int(pos * group.best_version.duration) for pos in frame_positions]
                frame_labels = ['Start', 'Middle', 'End']
                
                html_lines.append("<div class='frames'>")
                for time, label in zip(frame_times, frame_labels):
                    frame_path = f"frames/group_{i}/{group.best_version.path.stem}_frame_{time + 1}.jpg"
                    html_lines.extend([
                        "<div class='frame'>",
                        f"<img src='{frame_path}' alt='{label} frame'>",
                        f"<p>{label}</p>",
                        "</div>"
                    ])
                html_lines.append("</div>")
                
                html_lines.extend([
                    "<div class='best-version'>",
                    "<h4>Best Version:</h4>",
                    f"<p>Path: {group.best_version.get_display_path()}</p>",
                    f"<p>Size: {utils.format_size(group.best_version.size)}</p>",
                    f"<p>Resolution: {group.best_version.resolution[0]}x{group.best_version.resolution[1]}</p>",
                    "</div>",
                    "<h4>Duplicates:</h4>"
                ])
                
                for file in group.files:
                    if file != group.best_version:
                        html_lines.extend([
                            "<div class='duplicate'>",
                            f"<p>Path: {file.get_display_path()}</p>",
                            f"<p>Size: {utils.format_size(file.size)}</p>",
                            f"<p>Resolution: {file.resolution[0]}x{file.resolution[1]}</p>",
                            "</div>"
                        ])
                
                html_lines.append("</div>")
        
        html_lines.extend([
            "</body>",
            "</html>"
        ])
        
        # Write HTML report
        report_path = html_report_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_lines))
        
        logger.debug(f"HTML report saved to: {report_path.absolute()}")
        return report_path.absolute()
    
    else:  # Default to text format
        # Generate human-readable text report
        report_lines = [
            "=== Media Deduplication Report ===",
            f"Generated on: {datetime.now().isoformat()}",
            f"Total duplicate groups: {len(duplicate_groups)}",
            f"Total duplicate files: {sum(len(group.files) - 1 for group in duplicate_groups)}",
            f"Total wasted space: {utils.format_size(sum(sum(v.size for v in group.files[1:]) for group in duplicate_groups))}",
            "\n=== Duplicate Groups ==="
        ]
        
        for i, group in enumerate(duplicate_groups):
            report_lines.append(f"\nGroup {i+1} - Similarity: {group.similarity_score:.1f}%")
            report_lines.append(f"Best Version: {group.best_version.get_display_path()}")
            report_lines.append(f"Size: {utils.format_size(group.best_version.size)}")
            report_lines.append("\nDuplicates:")
            
            for file in group.files:
                if file != group.best_version:
                    report_lines.append(f"- {file.get_display_path()}")
                    report_lines.append(f"  Size: {utils.format_size(file.size)}")
        
        # Output to file or stdout
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            logger.debug(f"Text report saved to: {output_file.absolute()}")
            return output_file.absolute()
        else:
            print('\n'.join(report_lines))
            return None

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
        
        # Display files with numbering
        files = group.files
        best_file = group.best_version
        
        print("\n[KEEP] Best version:")
        print(f"* {best_file.get_display_path()}")
        print(f"  Size: {utils.format_size(best_file.size)}")
        
        print("\n[DUPLICATES]")
        for j, file in enumerate(files):
            if file != best_file:
                print(f"{j+1}. {file.get_display_path()}")
                print(f"   Size: {utils.format_size(file.size)}")
        
        # Ask for action
        while True:
            print("\nActions:")
            print("k - Keep all (skip this group)")
            print("d - Delete all duplicates")
            print("s - Select different file to keep")
            print("n - Next group")
            print("q - Quit")
            
            choice = input("\nEnter action: ").strip().lower()
            
            if choice == 'k':
                print("Keeping all files in this group")
                break
                
            elif choice == 'd':
                confirm = input("Are you sure you want to delete all duplicates? (y/n): ")
                if confirm.lower() == 'y':
                    for file in files:
                        if file != best_file:
                            try:
                                print(f"Deleting {file.path}...")
                                file.path.unlink()
                            except Exception as e:
                                print(f"Error deleting {file.path}: {e}")
                    print("Duplicates deleted")
                    break
                else:
                    print("Deletion cancelled")
                    
            elif choice == 's':
                print("\nSelect file to keep:")
                for j, file in enumerate(files):
                    print(f"{j+1}. {file.get_display_path()}")
                
                try:
                    selection = int(input("\nEnter number: "))
                    if 1 <= selection <= len(files):
                        best_file = files[selection-1]
                        print(f"Selected: {best_file.path}")
                        
                        confirm = input("Delete all other files? (y/n): ")
                        if confirm.lower() == 'y':
                            for file in files:
                                if file != best_file:
                                    try:
                                        print(f"Deleting {file.get_display_path()}...")
                                        file.get_display_path().unlink()
                                    except Exception as e:
                                        print(f"Error deleting {file.get_display_path()}: {e}")
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
    """Move duplicate files to a target directory."""
    for i, group in enumerate(duplicate_groups):
        # Create a subdirectory for each group
        group_dir = target_dir / f"group_{i+1}"
        utils.ensure_dir(group_dir)
        
        # Move duplicates, keep the best version in place
        best_file = group.best_version
        for file in group.files:
            if file != best_file:
                try:
                    display_path = file.get_display_path()
                    target_path = group_dir / display_path.name
                    # Ensure unique name in target directory
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = group_dir / f"{stem}_{file.hash_id[:8]}{suffix}"
                    
                    # Move the file
                    shutil.move(str(display_path), str(target_path))
                    logger.info(f"Moved: {display_path} -> {target_path}")
                except Exception as e:
                    logger.error(f"Error moving {file.path}: {e}")

def create_symlinks(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Create symbolic links for duplicates."""
    for i, group in enumerate(duplicate_groups):
        best_file = group.best_version
        
        # Create links for duplicates
        for file in group.files:
            if file != best_file:
                try:
                    # Create target path
                    display_path = file.get_display_path()
                    target_path = target_dir / display_path.name
                    # Ensure unique name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = target_dir / f"{stem}_{file.hash_id[:8]}{suffix}"
                    
                    # Create symbolic link
                    target_path.symlink_to(display_path.absolute())
                    logger.info(f"Created symlink: {target_path} -> {display_path}")
                except Exception as e:
                    logger.error(f"Error creating symlink for {file.path}: {e}")

def create_hardlinks(duplicate_groups: List[DuplicateGroup], target_dir: Path) -> None:
    """Create hard links for duplicates."""
    for i, group in enumerate(duplicate_groups):
        best_file = group.best_version
        
        # Create hard links for duplicates
        for file in group.files:
            if file != best_file:
                try:
                    # Create hard link path
                    display_path = file.get_display_path()
                    target_path = target_dir / display_path.name
                    # Ensure unique name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        target_path = target_dir / f"{stem}_{file.hash_id[:8]}{suffix}"
                    
                    # Create hard link
                    os.link(str(display_path.absolute()), str(target_path))
                    logger.info(f"Created hardlink: {target_path} -> {display_path}")
                except Exception as e:
                    logger.error(f"Error creating hardlink for {file.path}: {e}")

def delete_duplicates(duplicate_groups: List[DuplicateGroup]) -> None:
    """Delete duplicate files, keeping the best version."""
    total_deleted = 0
    total_freed = 0
    
    for group in duplicate_groups:
        best_file = group.best_version
        
        for file in group.files:
            if file != best_file:
                try:
                    display_path = file.get_display_path()
                    logger.info(f"Deleting: {display_path}")
                    display_path.unlink()
                    total_deleted += 1
                    total_freed += file.size
                except Exception as e:
                    logger.error(f"Error deleting {file.path}: {e}")
    
    logger.info(f"Deleted {total_deleted} duplicate files")
    logger.info(f"Freed {utils.format_size(total_freed)}")

def generate_action_script(duplicate_groups: List[DuplicateGroup], 
                          script_type: str = 'bash',
                          output_file: Optional[Path] = None) -> Optional[Path]:
    """Generate a script to handle duplicates."""
    if script_type == 'bash':
        # Generate bash script
        script_lines = [
            "#!/bin/bash",
            "# Generated by mediadedup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate files",
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
            best_file = group.best_version
            
            script_lines.append(f"echo \"Group {i+1}/{len(duplicate_groups)}\"")
            script_lines.append(f"echo \"Keeping: {best_file.get_display_path()}\"")
            
            for file in group.files:
                if file != best_file:
                    script_lines.append("if [ \"$DRY_RUN\" = \"true\" ]; then")
                    script_lines.append(f"    echo \"Would delete: {file.get_display_path()}\"")
                    script_lines.append("else")
                    script_lines.append(f"    echo \"Deleting: {file.get_display_path()}\"")
                    script_lines.append(f"    rm \"{file.get_display_path()}\"")
                    script_lines.append("fi")
            
            script_lines.append("")
        
        script_lines.append("echo \"Finished!\"")
        
    elif script_type == 'powershell':
        # Generate PowerShell script
        script_lines = [
            "# Generated by mediadedup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate files",
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
            best_file = group.best_version
            
            script_lines.append(f"Write-Host \"Group {i+1}/{len(duplicate_groups)}\" -ForegroundColor Cyan")
            script_lines.append(f"Write-Host \"Keeping: {best_file.get_display_path()}\" -ForegroundColor Green")
            
            for file in group.files:
                if file != best_file:
                    script_lines.append("if ($DryRun) {")
                    script_lines.append(f"    Write-Host \"Would delete: {file.get_display_path()}\" -ForegroundColor Yellow")
                    script_lines.append("} else {")
                    script_lines.append(f"    Write-Host \"Deleting: {file.get_display_path()}\" -ForegroundColor Red")
                    script_lines.append(f"    Remove-Item -Path \"{file.get_display_path()}\" -Force")
                    script_lines.append("}")
            
            script_lines.append("")
        
        script_lines.append("Write-Host \"Finished!\" -ForegroundColor Green")
        
    else:  # Python script
        # Generate Python script
        script_lines = [
            "#!/usr/bin/env python3",
            "# Generated by mediadedup",
            f"# Date: {datetime.now().isoformat()}",
            "",
            "# This script will delete duplicate files",
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
            "    parser = argparse.ArgumentParser(description='Delete duplicate files')",
            "    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no deletion)')",
            "    args = parser.parse_args()",
            "",
            "    print('Starting duplicate removal...')",
            ""
        ]
        
        for i, group in enumerate(duplicate_groups):
            best_file = group.best_version
            
            script_lines.append(f"    print(f'Group {i+1}/{len(duplicate_groups)}')")
            script_lines.append(f"    print(f'Keeping: {best_file.get_display_path()}')")
            
            for file in group.files:
                if file != best_file:
                    script_lines.append("    if args.dry_run:")
                    script_lines.append(f"        print(f'Would delete: {file.get_display_path()}')")
                    script_lines.append("    else:")
                    script_lines.append(f"        print(f'Deleting: {file.get_display_path()}')")
                    script_lines.append(f"        try:")
                    script_lines.append(f"            os.remove(r'{file.get_display_path()}')")
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
        logger.debug(f"Script saved to: {output_file.absolute()}")
        
        # Make script executable on Unix-like systems
        if script_type == 'bash' and os.name == 'posix':
            os.chmod(output_file, 0o755)
        
        return output_file.absolute()
    else:
        print(script_content)
        return None
