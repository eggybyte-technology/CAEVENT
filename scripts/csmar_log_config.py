import logging
import os
from pathlib import Path
import sys
import atexit
import builtins
import io

# Set up logs directory structure
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)
CSMAR_LOGS_DIR = LOGS_DIR / 'csmar'
CSMAR_LOGS_DIR.mkdir(exist_ok=True)

# Original open function to replace later
original_open = builtins.open

# Configure CSMAR logging
def setup_csmar_logging():
    """Configure CSMAR logging to use the logs directory"""
    # Create a file handler for csmar-log.log
    csmar_handler = logging.FileHandler(CSMAR_LOGS_DIR / 'csmar-api.log')
    csmar_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    csmar_handler.setFormatter(formatter)
    
    # Get the root logger (used by CSMAR)
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to prevent duplicate logging
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and (
            handler.baseFilename.endswith('csmar-log.log') or 
            handler.baseFilename.endswith('csmar-api.log')
        ):
            root_logger.removeHandler(handler)
    
    # Add our handler
    root_logger.addHandler(csmar_handler)
    
    # Handle the csmar-log.log that gets created in the root directory
    root_log_path = Path('csmar-log.log')
    
    # Delete any existing root log file
    if root_log_path.exists():
        try:
            # Move content from root log to logs directory
            with original_open(root_log_path, 'r') as root_file:
                root_content = root_file.read()
                
            with original_open(CSMAR_LOGS_DIR / 'csmar-api.log', 'a') as logs_file:
                logs_file.write(root_content)
                
            # Remove the root log file
            os.remove(root_log_path)
            print(f"Moved content from root csmar-log.log to {CSMAR_LOGS_DIR / 'csmar-api.log'}")
        except Exception as e:
            print(f"Warning: Could not process root csmar-log.log: {str(e)}")
    
    # Override open() to intercept csmar-log.log creation
    def patched_open(file, mode='r', *args, **kwargs):
        # Convert Path objects to strings for comparison
        file_str = str(file)
        
        # Check if trying to open csmar-log.log in the current directory
        if file_str == 'csmar-log.log' or file_str.endswith('/csmar-log.log') or file_str.endswith('\\csmar-log.log'):
            # Redirect to logs directory
            redirected_path = CSMAR_LOGS_DIR / 'csmar-api.log'
            print(f"Redirecting log write from {file_str} to {redirected_path}")
            return original_open(redirected_path, mode, *args, **kwargs)
        
        # Otherwise use the original open function
        return original_open(file, mode, *args, **kwargs)
    
    # Replace the built-in open function
    builtins.open = patched_open
    
    # Register a function to restore the original open function on exit
    def cleanup():
        builtins.open = original_open
        if root_log_path.exists():
            try:
                os.remove(root_log_path)
            except Exception:
                pass
    
    # Register cleanup to run on program exit
    atexit.register(cleanup)
    
    print(f"CSMAR logging configured to use {CSMAR_LOGS_DIR / 'csmar-api.log'}")
    
if __name__ == "__main__":
    setup_csmar_logging() 