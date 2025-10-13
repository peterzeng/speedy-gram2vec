"""
Utilities for logging training output to files while still showing in terminal.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TeeLogger:
    """
    Write output to both terminal and a log file simultaneously.
    Similar to Unix 'tee' command.
    """
    
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        sys.stdout = self.terminal
        self.log.close()


def generate_log_filename(
    script_name: str,
    log_dir: str = "logs",
    **kwargs
) -> Path:
    """
    Generate a descriptive log filename with timestamp and parameters.
    
    Args:
        script_name: Name of the script (e.g., 'train_top_features', 'full_training')
        log_dir: Directory to store logs
        **kwargs: Additional parameters to include in filename (e.g., top_n=10)
        
    Returns:
        Path to log file
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename parts
    parts = [script_name, timestamp]
    
    # Add any additional parameters
    for key, value in kwargs.items():
        if value is not None:
            # Clean up the value for filename
            clean_value = str(value).replace('/', '_').replace(' ', '_')
            parts.append(f"{key}_{clean_value}")
    
    filename = "_".join(parts) + ".log"
    return log_dir_path / filename


def start_logging(log_file: Path) -> TeeLogger:
    """
    Start logging to file and terminal.
    
    Args:
        log_file: Path to log file
        
    Returns:
        TeeLogger instance (should be saved to close later)
    """
    logger = TeeLogger(log_file)
    sys.stdout = logger
    
    print(f"{'='*80}")
    print(f"Logging to: {log_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return logger


def stop_logging(logger: TeeLogger, log_file: Path):
    """
    Stop logging and restore normal output.
    
    Args:
        logger: TeeLogger instance from start_logging
        log_file: Path to log file
    """
    print(f"\n{'='*80}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*80}")
    
    logger.close()


def get_latest_log(log_dir: str = "logs", pattern: str = "*.log") -> Optional[Path]:
    """
    Get the most recent log file.
    
    Args:
        log_dir: Directory containing logs
        pattern: File pattern to match
        
    Returns:
        Path to most recent log file, or None if no logs found
    """
    log_dir_path = Path(log_dir)
    
    if not log_dir_path.exists():
        return None
    
    log_files = sorted(log_dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if log_files:
        return log_files[0]
    
    return None


def list_logs(log_dir: str = "logs", pattern: str = "*.log", limit: int = 10):
    """
    List recent log files.
    
    Args:
        log_dir: Directory containing logs
        pattern: File pattern to match
        limit: Maximum number of logs to show
    """
    log_dir_path = Path(log_dir)
    
    if not log_dir_path.exists():
        print(f"No log directory found: {log_dir}")
        return
    
    log_files = sorted(log_dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Recent logs in {log_dir}/ (showing up to {limit}):")
    print(f"{'='*80}")
    
    for i, log_file in enumerate(log_files[:limit], 1):
        size_mb = log_file.stat().st_size / 1024 / 1024
        modified = datetime.fromtimestamp(log_file.stat().st_mtime)
        print(f"{i:2d}. {log_file.name}")
        print(f"    Size: {size_mb:.2f} MB | Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(log_files) > limit:
        print(f"\n... and {len(log_files) - limit} more")


def view_log(log_file: Path, tail: Optional[int] = None):
    """
    View contents of a log file.
    
    Args:
        log_file: Path to log file
        tail: If specified, only show last N lines
    """
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"Log: {log_file.name}")
    print(f"{'='*80}\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        if tail:
            lines = f.readlines()
            for line in lines[-tail:]:
                print(line, end='')
        else:
            print(f.read())

