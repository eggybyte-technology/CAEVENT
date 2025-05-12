import json
import os
import logging
from pathlib import Path
import getpass
from scripts.csmar_log_config import setup_csmar_logging
from scripts.csmarapi.CsmarService import CsmarService

# Set up logs directory
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)
AUTH_LOGS_DIR = LOGS_DIR / 'auth'
AUTH_LOGS_DIR.mkdir(exist_ok=True)

# Set up data directory paths
DATA_DIR = Path('data')
DATA_AUTH_DIR = DATA_DIR / 'auth'
DATA_AUTH_DIR.mkdir(exist_ok=True)

# Get or configure logger
logger = logging.getLogger('csmar-auth')
if not logger.handlers:
    # Configure logging only if handlers don't already exist
    handler = logging.FileHandler(AUTH_LOGS_DIR / 'csmar-auth.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure CSMAR logging is configured to use logs directory
setup_csmar_logging()

class CSMARAuthManager:
    """Manager for CSMAR authentication and credentials"""
    
    def __init__(self):
        self.credentials_file = Path('data/auth/csmar_credentials.json')
        self.csmar = CsmarService()
        self.login_status = False
    
    def credentials_exist(self):
        """Check if credentials file exists"""
        return self.credentials_file.exists()
    
    def load_credentials(self):
        """Load credentials from file"""
        try:
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)
                # Validate credentials format
                if 'email' not in credentials or 'password' not in credentials:
                    logger.error("Invalid credentials format: missing email or password")
                    return None
                return credentials
        except Exception as e:
            logger.error(f"Failed to load credentials: {str(e)}")
            return None
    
    def save_credentials(self, email, password):
        """Save credentials to file"""
        try:
            credentials = {
                'email': email,
                'password': password
            }
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f)
            logger.info("Credentials saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save credentials: {str(e)}")
            return False
    
    def verify_credentials(self, email, password):
        """Verify if credentials are valid by attempting login"""
        try:
            self.csmar.login(email, password)
            logger.info("CSMAR authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def prompt_login(self):
        """Interactive login prompt"""
        print("\n=== CSMAR Data Access ===")
        
        if self.credentials_exist():
            credentials = self.load_credentials()
            if credentials and 'email' in credentials and 'password' in credentials:
                print("Attempting login with saved credentials...")
                if self.verify_credentials(credentials['email'], credentials['password']):
                    print("Login successful with saved credentials!")
                    self.login_status = True
                    return True
                else:
                    print("Saved credentials are invalid.")
            else:
                print("Saved credentials file exists but has invalid format.")
        
        while True:
            login_choice = input("\nWould you like to log in to CSMAR? (y/n): ").strip().lower()
            
            if login_choice == 'n':
                print("Continuing without CSMAR authentication.")
                return False
            
            if login_choice == 'y':
                email = input("Email: ").strip()
                password = getpass.getpass("Password: ")
                
                print("Attempting login...")
                if self.verify_credentials(email, password):
                    print("Login successful!")
                    self.login_status = True
                    
                    save_choice = input("Save credentials for future use? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        self.save_credentials(email, password)
                        print("Credentials saved.")
                    
                    return True
                else:
                    print("Login failed. Invalid credentials.")
            else:
                print("Invalid choice. Please enter 'y' or 'n'.")

    def check_data_availability(self, data_paths):
        """Check if required data files exist"""
        all_exist = True
        for path in data_paths:
            if not os.path.exists(path):
                all_exist = False
                logger.info(f"Required data file not found: {path}")
        
        return all_exist

# Example usage
if __name__ == "__main__":
    auth_manager = CSMARAuthManager()
    login_success = auth_manager.prompt_login()
    
    if login_success:
        print("You can now proceed with data analysis.")
    else:
        # Check if required data files exist locally
        data_dir = Path('data')
        required_files = [
            data_dir / 'market' / "market_2022-01-01_2025-06-30.csv",
            data_dir / 'stock' / "stock_000725_2022-01-01_2025-06-30.csv"
        ]
        
        if auth_manager.check_data_availability(required_files):
            print("Required data files found locally. You can proceed with analysis.")
        else:
            print("Required data not available locally and you are not logged in.")
            print("Please log in to download the necessary data or provide local data files.")
            exit(1) 