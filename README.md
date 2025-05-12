# CAEVENT - CSMAR Advanced Event Study Analysis Tool

<div align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/license-Proprietary-red.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.8+-yellow.svg" alt="Python 3.8+">
</div>

## 📋 Overview

CAEVENT is a sophisticated tool for conducting financial event studies using the CSMAR database. It evaluates market efficiency by analyzing abnormal returns around significant corporate events, providing comprehensive statistical analysis and visualizations.

## ✨ Features

- **Flexible Event Analysis**: Analyze single events or multiple events in one run
- **CAPM Model Integration**: Utilize the Capital Asset Pricing Model for evaluating abnormal returns
- **Market Efficiency Testing**: Assess market reaction to corporate events
- **Intelligent Data Caching**: Speed up subsequent analyses with local data storage
- **Comprehensive Visualization**: Generate publication-quality charts and graphs
- **JSON Configuration**: Simple configuration through a single JSON file

## 🗂️ Directory Structure

```
caevent/
  ├── config/              # Configuration files
  │   └── config.json      # JSON configuration with events
  ├── data/                # Data storage
  │   ├── auth/            # Authentication data
  │   ├── market/          # Market data cache
  │   └── stock/           # Stock data cache
  ├── logs/                # Application logs
  │   ├── analysis/        # Analysis process logs
  │   ├── auth/            # Authentication logs
  │   └── csmar/           # CSMAR API logs
  ├── results/             # Analysis results
  │   └── run_XXXXXX/      # Individual analysis run 
  │       └── event_XXXXXX/# Event-specific results
  ├── scripts/             # Application modules
  │   ├── analyzer.py      # Stock analysis engine
  │   ├── auth_manager.py  # Authentication manager
  │   ├── check_config.py  # Configuration utility
  │   └── csmar_log_config.py  # Logging configuration
  └── main.py              # Main application entry point
```

## ⚙️ Configuration

CAEVENT has been updated to use a single JSON configuration file approach. All settings are controlled through the `config/config.json` file.

### JSON Configuration Format

```json
{
  "start_date": "2022-01-01",
  "end_date": "2025-06-30",
  "stock_code": "000725",
  "stock_name": "京东方A",
  "mode": "multiple",
  "regenerate_summary": false,
  "run_id": null,
  "save_events": false,
  "events": [
    {
      "date": "2024-06-12",
      "name": "AMOLED Shipment Target Announcement",
      "description": "Event description here..."
    },
    {
      "date": "2024-08-28",
      "name": "Shareholder Structure Disclosure",
      "description": "Event description here..."
    }
  ]
}
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | String | Start date for data collection (YYYY-MM-DD) |
| `end_date` | String | End date for data collection (YYYY-MM-DD) |
| `stock_code` | String | Stock code to analyze (e.g., "000725" for BOE) |
| `stock_name` | String | Stock name for display (optional) |
| `mode` | String | Analysis mode: "single" or "multiple" |
| `regenerate_summary` | Boolean | Whether to regenerate summary from existing data |
| `run_id` | String | Run ID for summary regeneration (format: run_YYYYMMDDHHMMSS_xxxxxxxx) |
| `save_events` | Boolean | Whether to save events to a separate file |
| `events` | Array | List of events to analyze, each with date, name, and description |

### Checking and Updating Configuration

You can use the `scripts/check_config.py` utility to verify your configuration:

```bash
python -m scripts.check_config
```

## 🚀 Running the Analysis

To run the analysis:

```bash
python main.py
```

The tool will:
1. Load settings from `config/config.json`
2. Display loaded configuration details
3. Run the appropriate analysis based on configuration mode
4. Generate results in the `results` directory

## 📊 Output

Analysis results are stored in the `results` directory, with each analysis run having a unique timestamped folder. Results include:

- **CAPM Model Analysis**: Detailed regression statistics and model fit
- **Event Study Reports**: Comprehensive event analysis with statistical measures
- **Abnormal Returns Charts**: Visualizations of abnormal returns around event dates
- **Cumulative AR Charts**: Cumulative abnormal return progression
- **Trading Volume Analysis**: Volume changes around events
- **Summary Reports**: Aggregated findings across all events

## 🔐 Authentication

CAEVENT requires authentication with the CSMAR database. You can:

1. Log in interactively when prompted
2. Save credentials for future use in `data/auth/csmar_credentials.json`

## 📜 License

Copyright © 2024-2025 EggyByte Technology. All rights reserved.

---

<div align="center">
  <p>Developed by EggyByte Technology • 2024-2025</p>
</div>