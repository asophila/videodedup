"""
RAM disk management utilities.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import logging
import psutil

logger = logging.getLogger(__name__)

class RAMDiskManager:
    """Manages a RAM disk for temporary file storage."""
    
    def __init__(self, size_mb: int = None, mount_point: Path = None):
        """Initialize RAM disk manager.
        
        Args:
            size_mb: Size of RAM disk in MB. If None, will be calculated based on available memory.
            mount_point: Path to mount RAM disk. If None, a temporary directory will be used.
        """
        self.size_mb = size_mb or self._calculate_safe_size()
        self.mount_point = mount_point or Path(tempfile.mkdtemp(prefix='videodedup_ramdisk_'))
        self.is_mounted = False
        
    def _calculate_safe_size(self) -> int:
        """Calculate a safe RAM disk size based on available memory."""
        mem = psutil.virtual_memory()
        # Use 80% of available memory
        safe_size = int(mem.available * 0.8 / (1024 * 1024))
        # Ensure at least 1GB
        return max(safe_size, 1024)
    
    def mount(self) -> bool:
        """Mount the RAM disk."""
        if self.is_mounted:
            return True
            
        try:
            if os.name == 'posix':  # Linux/macOS
                # Create mount point if it doesn't exist
                self.mount_point.mkdir(parents=True, exist_ok=True)
                
                # Create a memory-mapped temporary directory
                self.mount_point = Path(tempfile.mkdtemp(prefix='videodedup_ramdisk_'))
                self.is_mounted = True
                
            else:  # Windows not supported yet
                raise NotImplementedError("RAM disk support for Windows not implemented")
            
            self.is_mounted = True
            logger.info(f"Mounted {self.size_mb}MB RAM disk at {self.mount_point}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mount RAM disk: {e}")
            return False
    
    def unmount(self) -> bool:
        """Unmount the RAM disk."""
        if not self.is_mounted:
            return True
            
        try:
            if os.name == 'posix':  # Linux/macOS
                # Remove all files and subdirectories
                for item in self.mount_point.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                
                # Remove the directory itself
                self.mount_point.rmdir()
            
            self.is_mounted = False
            logger.info(f"Unmounted RAM disk from {self.mount_point}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unmount RAM disk: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all files from the RAM disk."""
        if not self.is_mounted:
            return True
            
        try:
            # Remove all files and subdirectories
            for item in self.mount_point.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            logger.info("Cleared RAM disk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear RAM disk: {e}")
            return False
    
    def will_file_fit(self, file_size: int) -> bool:
        """Check if a file of given size will fit in the RAM disk.
        
        Args:
            file_size: Size of the file in bytes
            
        Returns:
            bool: True if the file will fit, False otherwise
        """
        if not self.is_mounted:
            return False
            
        try:
            free_space = self.get_free_space()
            # Leave some buffer space (5% of RAM disk size)
            buffer_space = self.size_mb * 1024 * 1024 * 0.05
            return file_size <= (free_space - buffer_space)
        except Exception:
            return False
    
    def copy_to_ramdisk(self, file_path: Path) -> Path:
        """Copy a file to the RAM disk.
        
        Args:
            file_path: Path to the file to copy
            
        Returns:
            Path to the file in the RAM disk
            
        Raises:
            RuntimeError: If RAM disk is not mounted
            ValueError: If file is too large for RAM disk
            OSError: If copy fails
        """
        if not self.is_mounted:
            raise RuntimeError("RAM disk not mounted")
            
        file_size = file_path.stat().st_size
        if not self.will_file_fit(file_size):
            raise ValueError(f"File {file_path.name} ({file_size / (1024*1024):.1f}MB) is too large for RAM disk with {self.get_free_space() / (1024*1024):.1f}MB free")
            
        try:
            dest_path = self.mount_point / file_path.name
            shutil.copy2(file_path, dest_path)
            return dest_path
            
        except Exception as e:
            logger.error(f"Failed to copy {file_path} to RAM disk: {e}")
            raise
    
    def get_free_space(self) -> int:
        """Get free space in RAM disk in bytes."""
        if not self.is_mounted:
            return 0
            
        try:
            usage = shutil.disk_usage(self.mount_point)
            return usage.free
        except Exception:
            return 0
    
    def __enter__(self):
        """Context manager entry."""
        self.mount()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unmount()
