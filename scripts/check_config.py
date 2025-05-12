import json
import os
from pathlib import Path

# Get the path to the config file relative to the project root
config_dir = Path(__file__).parent.parent / 'config'
config_path = config_dir / 'config.json'

def check_config():
    # Read the configuration file with UTF-8 encoding
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Print the stock name to verify encoding
    print(f"Stock name in configuration: {config['stock_name']}")

    # Update the config and write it back
    config['stock_name'] = "京东方A (测试)"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Configuration updated. Stock name set to: 京东方A (测试)")

if __name__ == "__main__":
    check_config() 