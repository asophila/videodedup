"""
Common actions for handling duplicate media files.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from . import utils
from .models import DuplicateGroup, MediaFile

logger = logging.getLogger(__name__)

def handle_duplicates(duplicate_groups: List[DuplicateGroup], args) -> None:
    """Handle duplicate files according to the specified action."""
    if not duplicate_groups:
        logger.info("No duplicates found.")
        return
        
    # Calculate total statistics
    total_files = sum(len(group.files) for group in duplicate_groups)
    total_duplicates = sum(len(group.files) - 1 for group in duplicate_groups)
    total_wasted_space = sum(
        sum(v.size for v in group.files[1:]) for group in duplicate_groups
    )
    
    logger.info(f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicates")
    logger.info(f"Total wasted space: {utils.format_size(total_wasted_space)}")
    
    # Create output directory if needed
    output_dir = None
    if args.action in ['move', 'symlink', 'hardlink'] and args.target_dir:
        output_dir = Path(args.target_dir)
        utils.ensure_dir(output_dir)
    
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
                   output_file: Optional[Path] = None,
                   html_report_dir: Optional[Path] = None) -> None:
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
            logger.info(f"Report saved to {output_file}")
        else:
            print(json.dumps(json_data, indent=2))
    
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
            logger.info(f"Report saved to {output_file}")
        else:
            print('\n'.join(csv_lines))
    
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
                          output_file: Optional[Path] = None) -> None:
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
        logger.info(f"Script saved to {output_file}")
        
        # Make script executable on Unix-like systems
        if script_type == 'bash' and os.name == 'posix':
            os.chmod(output_file, 0o755)
    else:
        print(script_content)
