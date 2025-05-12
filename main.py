from scripts.auth_manager import CSMARAuthManager
from scripts.analyzer import StockAnalyzer
from scripts.csmar_log_config import setup_csmar_logging
from pathlib import Path
import sys
import logging
import os
import json
import datetime

# Set up directory structure
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path('config')
CONFIG_DIR.mkdir(exist_ok=True)

# Default configuration file path
DEFAULT_CONFIG_PATH = CONFIG_DIR / 'config.json'

# Delete any existing csmar-log.log file in the root directory
root_log_path = Path('csmar-log.log')
if root_log_path.exists():
    try:
        os.remove(root_log_path)
        print(f"Deleted existing root csmar-log.log file")
    except Exception as e:
        print(f"Warning: Could not delete root csmar-log.log: {str(e)}")

# Get or configure logger
logger = logging.getLogger('boe_analysis')
if not logger.handlers:
    # Configure logging only if handlers don't already exist
    handler = logging.FileHandler(LOGS_DIR / 'analysis' / 'boe_analysis.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure CSMAR logging is configured to use logs directory
setup_csmar_logging()

class ConfigManager:
    """Manager for handling configuration from config.json file"""
    
    def __init__(self):
        self.config = {
            'start_date': '2022-01-01',
            'end_date': '2025-06-30',
            'stock_code': '000725',
            'stock_name': None,
            'mode': 'multiple',
            'regenerate_summary': False,
            'run_id': None,
            'save_events': False,
            'events': []
        }
    
    def load_from_json(self, file_path):
        """Load configuration from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_config = json.load(f)
            
            # Update only keys that are present in the JSON file
            for key in json_config:
                if key in self.config:
                    self.config[key] = json_config[key]
            
            logger.info(f"Configuration loaded from JSON file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            return False
    
    def save_to_json(self, file_path=DEFAULT_CONFIG_PATH):
        """Save current configuration to a JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            return False

def main():
    """Main entry point for CAEVENT stock analysis"""
    print("CAEVENT - CSMAR Advanced Event Study Analysis Tool")
    print("Copyright © 2024-2025 EggyByte Technology. All rights reserved.")
    print("\nNote: Command line arguments are disabled. Please edit config/config.json to configure the analysis.")
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Load configuration from file
    if not DEFAULT_CONFIG_PATH.exists():
        print(f"Configuration file not found: {DEFAULT_CONFIG_PATH}")
        print("Please create a configuration file or run the check_config script.")
        return
    
    config_manager.load_from_json(DEFAULT_CONFIG_PATH)
    config = config_manager.config
    
    # Display the loaded configuration
    print("\nLoaded configuration:")
    print(f"  Stock: {config['stock_code']} - {config['stock_name']}")
    print(f"  Date range: {config['start_date']} to {config['end_date']}")
    print(f"  Mode: {config['mode']}")
    
    if 'events' in config and config['events']:
        print(f"  Events loaded: {len(config['events'])}")
        for i, event in enumerate(config['events']):
            print(f"    {i+1}. {event['date']} - {event['name']}")
    else:
        print("  No events configured in config.json")
    
    # Create a stock analyzer instance
    analyzer = StockAnalyzer(stock_code=config['stock_code'], stock_name=config['stock_name'])
    
    # Run the analysis
    try:
        if config['mode'] == 'single' and config.get('event_date'):
            # Run single event analysis
            analyzer.full_analysis_pipeline(
                event_date=config['event_date'],
                start_date=config['start_date'],
                end_date=config['end_date']
            )
        elif config['regenerate_summary'] and config.get('run_id'):
            # Regenerate summary from existing event reports
            analyzer.regenerate_summary(config['events'], run_id=config['run_id'])
        else:
            # Run multiple events analysis (default)
            events_list = [
                {'date': event['date'], 'name': event['name']}
                for event in config['events']
            ]
            analyzer.analyze_multiple_events(
                events_list=events_list,
                start_date=config['start_date'],
                end_date=config['end_date']
            )
        
        print("\nAnalysis completed successfully.")
        print(f"Results available in: {analyzer.results_dir}")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\nError during analysis: {str(e)}")
        return

if __name__ == "__main__":
    main() 