import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib import rcParams
import os
import logging
from pathlib import Path
import datetime
import uuid
import json
from scipy import stats
import re
from scripts.csmar_log_config import setup_csmar_logging
from scripts.csmarapi.CsmarService import CsmarService

# Configure Chinese display
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# Set up directory structure
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
DATA_MARKET_DIR = DATA_DIR / 'market'
DATA_MARKET_DIR.mkdir(exist_ok=True)
DATA_STOCK_DIR = DATA_DIR / 'stock'
DATA_STOCK_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)
ANALYSIS_LOGS_DIR = LOGS_DIR / 'analysis'
ANALYSIS_LOGS_DIR.mkdir(exist_ok=True)

# Get or configure logger
logger = logging.getLogger('stock_analysis')
if not logger.handlers:
    # Configure logging only if handlers don't already exist
    handler = logging.FileHandler(ANALYSIS_LOGS_DIR / 'stock_analysis.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure CSMAR logging is configured to use logs directory
setup_csmar_logging()

class StockAnalyzer:
    def __init__(self, stock_code='000725', stock_name=None, results_base_dir='results'):
        """Initialize the stock analyzer
        
        Args:
            stock_code (str): The stock code to analyze (default: 000725 for BOE)
            stock_name (str): The stock name for display purposes (defaults to the stock code)
            results_base_dir (str): Directory for storing results
        """
        self.stock_code = stock_code
        self.stock_name = stock_name or f"Stock {stock_code}"
        self.login_status = False
        self.csmar = None
        self.alpha = None
        self.beta = None
        
        # Create a unique run ID with timestamp for results
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.results_dir = Path(results_base_dir) / run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized StockAnalyzer for stock {stock_code} ({self.stock_name}), run_id: {run_id}")
        
    def login(self, email=None, password=None):
        """Authenticate with CSMAR database"""
        if not self.login_status:
            try:
                # If credentials are provided, use them
                if email and password:
                    self.csmar.login(email, password)
                # Otherwise use default credentials from file
                else:
                    with open(Path('data/auth/csmar_credentials.json'), 'r') as f:
                        import json
                        credentials = json.load(f)
                        self.csmar.login(credentials['email'], credentials['password'])
                
                self.login_status = True
                logger.info("CSMAR authentication successful")
                print("CSMAR authentication successful")
            except Exception as e:
                logger.error(f"Authentication failed: {str(e)}")
                print(f"Authentication failed: {str(e)}")
                raise

    def _get_csmar_data(self, table, condition, cache_file, start_date, end_date):
        """Retrieve data with local caching"""
        cache_path = cache_file
        
        if cache_path.exists():
            logger.info(f"Loading cached data: {cache_file}")
            return pd.read_csv(cache_path, parse_dates=['Trddt'], index_col='Trddt')
        
        if not self.login_status:
            self.login()
            
        # Define fields based on the table type
        if table == 'TRD_Dalym':
            fields = [
                'Markettype',   # Market type
                'Trddt',        # Trading date
                'Dnshrtrdtl',   # Daily total market shares traded
                'Dnvaltrdtl',   # Daily total market value traded
                'Dretwdos',     # Market return (float-adjusted with dividend)
                'Dretmdos',     # Market return (float-adjusted without dividend)
                'Dnstkcal'      # Number of valid constituent stocks
            ]
        elif table == 'TRD_Dalyr':
            fields = [
                'Stkcd',        # Stock code
                'Trddt',        # Trading date
                'Dsmvosd',      # Daily float market value
                'Dretwd',       # Daily return with dividend
                'Dnshrtrd',     # Daily trading volume (shares)
                'Dnvaltrd',     # Daily trading value
                'Opnprc',       # Open price
                'Hiprc',        # High price
                'Loprc',        # Low price
                'Clsprc'        # Close price
            ]
        else:
            raise ValueError(f"Unknown table: {table}")
            
        # Use the query_df with startTime and endTime parameters
        df = self.csmar.query_df(fields, condition, table, start_date, end_date)
        
        if df.empty:
            logger.error(f"No data found for table {table}, check parameters")
            raise ValueError(f"No data found for table {table}, check parameters")
        
        # Standardize data format
        df['Trddt'] = pd.to_datetime(df['Trddt'])
        df.set_index('Trddt', inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(cache_path)
        logger.info(f"Data saved to {cache_path}")
        return df

    def get_market_data(self, start_date, end_date):
        """Retrieve market-level data using float-adjusted returns"""
        cache_file = DATA_MARKET_DIR / f"market_{start_date}_{end_date}.csv"
        return self._get_csmar_data(
            table='TRD_Dalym',
            condition="",  # Empty condition as we're using start_date and end_date
            cache_file=cache_file,
            start_date=start_date,
            end_date=end_date
        )
        
    def get_stock_data(self, start_date, end_date):
        """Retrieve stock-level data"""
        cache_file = DATA_STOCK_DIR / f"stock_{self.stock_code}_{start_date}_{end_date}.csv"
        return self._get_csmar_data(
            table='TRD_Dalyr',
            condition=f"Stkcd='{self.stock_code}'",  # Only stock code in condition
            cache_file=cache_file,
            start_date=start_date,
            end_date=end_date
        )

    def estimate_capm(self, stock_data, market_data):
        """Estimate CAPM model for the stock using float-adjusted market returns"""
        # Align dates between market and stock data
        aligned_data = pd.merge(
            stock_data[['Dretwd']],
            market_data[['Dretwdos']],
            left_index=True, 
            right_index=True
        )
        
        # Prepare data for regression
        X = sm.add_constant(aligned_data['Dretwdos'])  # Market return (float-adjusted)
        y = aligned_data['Dretwd']                    # Stock return
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Store results for later use - fix the FutureWarning by using proper indexing
        self.capm_model = model
        # Use .iloc for positional indexing instead of direct list-like indexing
        self.beta = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]
        self.alpha = model.params.iloc[0]
        self.aligned_data = aligned_data
        
        # Generate CAPM summary document
        self.generate_capm_summary(model, aligned_data)
        
        return model
    
    def generate_capm_summary(self, model, data):
        """Generate a comprehensive CAPM model summary document"""
        summary_path = self.results_dir / f'capm_model_summary_{self.stock_code}.md'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# CAPM Model Analysis for {self.stock_name} ({self.stock_code})\n\n")
            f.write(f"## Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model parameters
            f.write("## Model Parameters\n\n")
            f.write(f"- **Alpha (α)**: {self.alpha:.6f}\n")
            f.write(f"- **Beta (β)**: {self.beta:.6f}\n")
            f.write(f"- **R-squared**: {model.rsquared:.6f}\n")
            f.write(f"- **Adjusted R-squared**: {model.rsquared_adj:.6f}\n")
            f.write(f"- **F-statistic**: {model.fvalue:.4f}\n")
            f.write(f"- **Prob (F-statistic)**: {model.f_pvalue:.6f}\n\n")
            
            # Interpretation
            f.write("## Interpretation\n\n")
            f.write("### Beta Interpretation\n")
            if self.beta > 1.1:
                f.write("The stock has a **high beta** (> 1.1), indicating it's more volatile than the market. It tends to amplify market movements.\n\n")
            elif self.beta < 0.9:
                f.write("The stock has a **low beta** (< 0.9), indicating it's less volatile than the market. It tends to be more stable during market fluctuations.\n\n")
            else:
                f.write("The stock has a **beta close to 1**, indicating it tends to move in line with the market.\n\n")
                
            f.write("### Alpha Interpretation\n")
            if self.alpha > 0:
                f.write(f"The positive alpha ({self.alpha:.6f}) suggests the stock has outperformed what would be predicted by the CAPM model, indicating potential undervaluation or strong company-specific performance.\n\n")
            else:
                f.write(f"The non-positive alpha ({self.alpha:.6f}) suggests the stock has not outperformed what would be predicted by the CAPM model.\n\n")
            
            # Statistical significance - fix the FutureWarning by using proper indexing
            f.write("## Statistical Significance\n\n")
            f.write("### Coefficient Analysis\n\n")
            f.write("| Parameter | Coefficient | Std Error | t-value | p-value | Significance |\n")
            f.write("|-----------|-------------|-----------|---------|---------|---------------|\n")
            
            # Alpha significance
            alpha_sig = "Significant" if model.pvalues.iloc[0] < 0.05 else "Not significant"
            f.write(f"| Alpha | {model.params.iloc[0]:.6f} | {model.bse.iloc[0]:.6f} | {model.tvalues.iloc[0]:.4f} | {model.pvalues.iloc[0]:.6f} | {alpha_sig} |\n")
            
            # Beta significance
            beta_sig = "Significant" if model.pvalues.iloc[1] < 0.05 else "Not significant"
            f.write(f"| Beta | {model.params.iloc[1]:.6f} | {model.bse.iloc[1]:.6f} | {model.tvalues.iloc[1]:.4f} | {model.pvalues.iloc[1]:.6f} | {beta_sig} |\n\n")
            
            # Model diagnostics
            f.write("## Model Diagnostics\n\n")
            
            # Data summary
            f.write("### Data Summary\n\n")
            f.write("#### Stock Returns Summary\n\n")
            stock_summary = data['Dretwd'].describe()
            f.write(f"- **Count**: {stock_summary['count']}\n")
            f.write(f"- **Mean**: {stock_summary['mean']:.6f}\n")
            f.write(f"- **Std Dev**: {stock_summary['std']:.6f}\n")
            f.write(f"- **Min**: {stock_summary['min']:.6f}\n")
            f.write(f"- **Max**: {stock_summary['max']:.6f}\n\n")
            
            f.write("#### Market Returns Summary\n\n")
            market_summary = data['Dretwdos'].describe()
            f.write(f"- **Count**: {market_summary['count']}\n")
            f.write(f"- **Mean**: {market_summary['mean']:.6f}\n")
            f.write(f"- **Std Dev**: {market_summary['std']:.6f}\n")
            f.write(f"- **Min**: {market_summary['min']:.6f}\n")
            f.write(f"- **Max**: {market_summary['max']:.6f}\n\n")
            
            # Correlation
            correlation = data[['Dretwd', 'Dretwdos']].corr().iloc[0, 1]
            f.write(f"### Correlation Between Stock and Market Returns\n\n")
            f.write(f"Correlation coefficient: {correlation:.6f}\n\n")
            
            # Model Formula
            f.write("## CAPM Model Formula\n\n")
            f.write(f"Expected Return = {self.alpha:.6f} + {self.beta:.6f} × Market Return\n\n")
            
            # Save regression results as text
            f.write("## Detailed Regression Results\n\n")
            f.write("```\n")
            f.write(model.summary().as_text())
            f.write("\n```\n")
            
        logger.info(f"CAPM model summary document saved to {summary_path}")

    def conduct_event_study(self, stock_data, market_data, event_date, event_name="Unnamed Event", window=(-10, 10)):
        """Perform stock-level event analysis"""
        event_date = pd.to_datetime(event_date)
        
        # Create event directory
        event_dir = self.results_dir / f"event_{event_date.strftime('%Y%m%d')}_{event_name.replace(' ', '_')}"
        event_dir.mkdir(exist_ok=True)
        
        logger.info(f"Conducting event study for {event_name} on {event_date}")
        
        # Calculate estimation window parameters (typically 120-day period before event window)
        est_start = event_date + pd.DateOffset(days=window[0]-120)
        est_end = event_date + pd.DateOffset(days=window[0]-1)
        
        # Create event window data
        event_window = stock_data.loc[
            event_date + pd.DateOffset(days=window[0]):
            event_date + pd.DateOffset(days=window[1])
        ].copy()
        
        # Get corresponding market data
        event_window_dates = event_window.index
        market_event = market_data.loc[market_data.index.isin(event_window_dates)].copy()
        
        # Ensure index uniqueness before merging
        if not event_window.index.is_unique:
            event_window = event_window.loc[~event_window.index.duplicated(keep='first')]
            logger.warning(f"Removed duplicate indices from stock data for event {event_name}")
            
        if not market_event.index.is_unique:
            market_event = market_event.loc[~market_event.index.duplicated(keep='first')]
            logger.warning(f"Removed duplicate indices from market data for event {event_name}")
        
        # Merge data using inner join to ensure only matching dates are included
        merged_data = pd.merge(
            event_window[['Dretwd', 'Dnshrtrd']], 
            market_event[['Dretwdos']], 
            left_index=True, 
            right_index=True,
            how='inner'
        )
        
        # Check if we have enough data for meaningful analysis
        if len(merged_data) < 3:
            error_msg = f"Insufficient data points for event analysis: {len(merged_data)} days available"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get estimation window data for CAPM if not already calculated
        if not hasattr(self, 'capm_model'):
            est_stock_data = stock_data.loc[est_start:est_end]
            est_market_data = market_data.loc[est_start:est_end]
            self.estimate_capm(est_stock_data, est_market_data)
        
        # Calculate expected returns using CAPM
        merged_data['expected_return'] = self.alpha + self.beta * merged_data['Dretwdos']
        
        # Calculate abnormal returns
        merged_data['abnormal_return'] = merged_data['Dretwd'] - merged_data['expected_return']
        
        # Calculate cumulative abnormal returns
        merged_data['cumulative_ar'] = merged_data['abnormal_return'].cumsum()
        
        # Save event study results
        merged_data.to_csv(event_dir / f'event_study_data.csv')
        
        # Generate event analysis report
        self.generate_event_report(merged_data, event_date, event_name, event_dir)
        
        # Generate visualizations
        self.visualize_trading_volume(merged_data, event_date, event_name, event_dir)
        self.visualize_abnormal_returns(merged_data, event_date, event_name, event_dir)
        self.visualize_cumulative_ar(merged_data, event_date, event_name, event_dir)
        
        return merged_data
    
    def generate_event_report(self, event_data, event_date, event_name, event_dir):
        """Generate comprehensive event study report"""
        report_path = event_dir / 'event_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Event Study Analysis: {event_name}\n\n")
            f.write(f"## Event Date: {event_date.strftime('%Y-%m-%d')}\n\n")
            
            # Event summary statistics
            f.write("## Summary Statistics\n\n")
            f.write("### Abnormal Returns\n\n")
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            
            ar_stats = event_data['abnormal_return'].describe()
            f.write(f"| Mean | {ar_stats['mean']*100:.4f}% |\n")
            f.write(f"| Std Dev | {ar_stats['std']*100:.4f}% |\n")
            f.write(f"| Min | {ar_stats['min']*100:.4f}% |\n")
            f.write(f"| Max | {ar_stats['max']*100:.4f}% |\n")
            f.write(f"| Cumulative AR | {event_data['cumulative_ar'].iloc[-1]*100:.4f}% |\n\n")
            
            # Trading volume statistics
            f.write("### Trading Volume\n\n")
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            
            vol_stats = event_data['Dnshrtrd'].describe()
            f.write(f"| Mean | {vol_stats['mean']/10000:.2f}万股 |\n")
            f.write(f"| Std Dev | {vol_stats['std']/10000:.2f}万股 |\n")
            f.write(f"| Min | {vol_stats['min']/10000:.2f}万股 |\n")
            f.write(f"| Max | {vol_stats['max']/10000:.2f}万股 |\n\n")
            
            # Market efficiency analysis
            f.write("## Market Efficiency Analysis\n\n")
            
            # Calculate t-statistic for CAR
            car = event_data['cumulative_ar'].iloc[-1]
            car_std = event_data['abnormal_return'].std() * np.sqrt(len(event_data))
            t_stat = car / (car_std / np.sqrt(len(event_data)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(event_data)-1))
            
            f.write(f"### Statistical Significance of Cumulative Abnormal Returns\n\n")
            f.write(f"- **CAR**: {car*100:.4f}%\n")
            f.write(f"- **t-statistic**: {t_stat:.4f}\n")
            f.write(f"- **p-value**: {p_value:.4f}\n")
            f.write(f"- **Significance**: {'Significant' if p_value < 0.05 else 'Not significant'} at 5% level\n\n")
            
            # Market efficiency conclusion
            f.write("### Market Efficiency Interpretation\n\n")
            
            if p_value < 0.05:
                if car > 0:
                    f.write("The significant positive cumulative abnormal returns suggest that the market did not immediately and fully incorporate the information from this event. This finding is inconsistent with the semi-strong form of market efficiency, as investors could potentially have earned abnormal returns by trading after the event announcement.\n\n")
                else:
                    f.write("The significant negative cumulative abnormal returns suggest that the market did not immediately and fully incorporate the information from this event. This finding is inconsistent with the semi-strong form of market efficiency, as the market may have overreacted to the event.\n\n")
            else:
                f.write("The cumulative abnormal returns are not statistically significant, suggesting that the market efficiently incorporated the information from this event. This finding is consistent with the semi-strong form of market efficiency, as investors could not have earned abnormal returns by trading after the event announcement.\n\n")
            
            # Daily data table
            f.write("## Daily Event Window Data\n\n")
            f.write("| Date | Stock Return | Market Return | Expected Return | Abnormal Return | Cumulative AR | Trading Volume |\n")
            f.write("|------|-------------|---------------|-----------------|----------------|--------------|----------------|\n")
            
            for idx, row in event_data.iterrows():
                f.write(f"| {idx.strftime('%Y-%m-%d')} | {row['Dretwd']*100:.4f}% | {row['Dretwdos']*100:.4f}% | {row['expected_return']*100:.4f}% | {row['abnormal_return']*100:.4f}% | {row['cumulative_ar']*100:.4f}% | {row['Dnshrtrd']/10000:.2f}万股 |\n")
        
        logger.info(f"Event analysis report saved to {report_path}")

    def visualize_trading_volume(self, event_data, event_date, event_name, event_dir):
        """Generate trading volume visualization"""
        plt.figure(figsize=(12, 6))
        
        # Check if event date is in the data
        event_idx = None
        if event_date in event_data.index:
            # Get the integer position of the event date in the index
            event_idx = event_data.index.get_indexer([event_date])[0]
            
            # Create lists of indices to avoid slicing with DatetimeIndex
            pre_event_indices = list(range(len(event_data.index)))[:event_idx]
            post_event_indices = list(range(len(event_data.index)))[event_idx+1:]
            
            # Pre-event bars
            if pre_event_indices:
                plt.bar(event_data.index[pre_event_indices], 
                       event_data['Dnshrtrd'].iloc[pre_event_indices],
                       color='skyblue', label='Pre-event')
            
            # Event day bar
            plt.bar([event_data.index[event_idx]], 
                   [event_data['Dnshrtrd'].iloc[event_idx]], 
                   color='red', label='Event day')
            
            # Post-event bars
            if post_event_indices:
                plt.bar(event_data.index[post_event_indices], 
                       event_data['Dnshrtrd'].iloc[post_event_indices],
                       color='lightgreen', label='Post-event')
        else:
            # If event date not in data, use single color
            plt.bar(event_data.index, event_data['Dnshrtrd'], color='skyblue')
        
        plt.title(f'Trading Volume for {self.stock_name} ({self.stock_code})\nEvent: {event_name} ({event_date.strftime("%Y-%m-%d")})')
        plt.ylabel('Shares Traded')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Add average line
        avg_volume = event_data['Dnshrtrd'].mean()
        plt.axhline(y=avg_volume, color='navy', linestyle='--', alpha=0.7, 
                   label=f'Avg: {avg_volume/10000:.2f}万股')
        plt.legend()
        
        # Save figure
        output_path = event_dir / 'trading_volume.png'
        plt.savefig(output_path)
        logger.info(f"Trading volume chart saved to {output_path}")
        plt.close()

    def visualize_abnormal_returns(self, event_data, event_date, event_name, event_dir):
        """Generate abnormal returns visualization"""
        plt.figure(figsize=(12, 6))
        
        # Check if event date is in the data
        event_idx = None
        if event_date in event_data.index:
            # Get the integer position of the event date in the index
            event_idx = event_data.index.get_indexer([event_date])[0]
            
            # Create lists of indices to avoid slicing with DatetimeIndex
            pre_event_indices = list(range(len(event_data.index)))[:event_idx]
            post_event_indices = list(range(len(event_data.index)))[event_idx+1:]
            
            # Pre-event bars
            if pre_event_indices:
                plt.bar(event_data.index[pre_event_indices], 
                       event_data['abnormal_return'].iloc[pre_event_indices]*100,
                       color='skyblue', label='Pre-event')
            
            # Event day bar
            plt.bar([event_data.index[event_idx]], 
                   [event_data['abnormal_return'].iloc[event_idx]*100], 
                   color='red', label='Event day')
            
            # Post-event bars
            if post_event_indices:
                plt.bar(event_data.index[post_event_indices], 
                       event_data['abnormal_return'].iloc[post_event_indices]*100,
                       color='lightgreen', label='Post-event')
        else:
            # If event date not in data, use single color
            plt.bar(event_data.index, event_data['abnormal_return']*100, color='skyblue')
        
        plt.title(f'Daily Abnormal Returns for {self.stock_name} ({self.stock_code})\nEvent: {event_name} ({event_date.strftime("%Y-%m-%d")})')
        plt.ylabel('Abnormal Return (%)')
        plt.axhline(0, color='grey', linestyle='-')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        output_path = event_dir / 'abnormal_returns.png'
        plt.savefig(output_path)
        logger.info(f"Abnormal returns chart saved to {output_path}")
        plt.close()

    def visualize_cumulative_ar(self, event_data, event_date, event_name, event_dir):
        """Generate cumulative abnormal returns visualization"""
        plt.figure(figsize=(12, 6))
        
        # Check if event date is in the data
        event_idx = None
        if event_date in event_data.index:
            # Get the integer position of the event date in the index
            event_idx = event_data.index.get_indexer([event_date])[0]
            
            # Create numerical x-axis relative to event day
            x_values = list(range(-event_idx, len(event_data) - event_idx))
            
            # Plot all data
            plt.plot(x_values, event_data['cumulative_ar']*100, 
                    marker='o', color='blue', linewidth=2)
            
            # Highlight event day
            plt.plot([0], [event_data['cumulative_ar'].iloc[event_idx]*100], 
                    marker='s', color='red', markersize=10, label='Event day')
            
            # Add vertical line at event day
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Set x-ticks to show relative days to event
            plt.xticks(range(min(x_values), max(x_values)+1, 2))
        else:
            # If event date not in data, use regular indices
            plt.plot(range(len(event_data)), 
                   event_data['cumulative_ar']*100, 
                   marker='o', color='green', linewidth=2)
        
        plt.title(f'Cumulative Abnormal Returns for {self.stock_name} ({self.stock_code})\nEvent: {event_name} ({event_date.strftime("%Y-%m-%d")})')
        plt.xlabel('Days Relative to Event Day')
        plt.ylabel('Cumulative Abnormal Return (%)')
        plt.axhline(0, color='grey', linestyle='-')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        output_path = event_dir / 'cumulative_ar.png'
        plt.savefig(output_path)
        logger.info(f"Cumulative abnormal returns chart saved to {output_path}")
        plt.close()

    def analyze_multiple_events(self, events_list, start_date='2022-01-01', end_date='2025-06-30'):
        """Analyze multiple events"""
        try:
            logger.info(f"Starting multi-event analysis for stock {self.stock_code} ({self.stock_name})")
            print(f"Starting multi-event analysis for {self.stock_name} ({self.stock_code})")
            
            # Data acquisition (acquire once for all events)
            market_data = self.get_market_data(start_date, end_date)
            stock_data = self.get_stock_data(start_date, end_date)
            
            logger.info("Market and stock data retrieved successfully")
            print("Market and stock data retrieved successfully")
            
            # CAPM Model estimation
            capm_model = self.estimate_capm(stock_data, market_data)
            logger.info(f"CAPM Model estimated: Alpha={self.alpha:.6f}, Beta={self.beta:.6f}")
            print(f"\nCAPM Model Results for {self.stock_name} ({self.stock_code}):")
            print(f"Alpha: {self.alpha:.6f}")
            print(f"Beta: {self.beta:.6f}")
            print(f"R-squared: {capm_model.rsquared:.6f}")
            
            # Create a summary document for all events
            events_summary_path = self.results_dir / 'all_events_summary.md'
            with open(events_summary_path, 'w', encoding='utf-8') as f:
                f.write(f"# {self.stock_name} ({self.stock_code}) Event Study Analysis Summary\n\n")
                f.write(f"## Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Events Analyzed\n\n")
                f.write("| Event Date | Event Description | CAR | Significant? |\n")
                f.write("|------------|-------------------|-----|-------------|\n")
            
            # Analyze each event
            for event in events_list:
                event_date = event['date']
                event_name = event['name']
                
                print(f"\nAnalyzing event: {event_name} ({event_date})")
                
                try:
                    # Conduct event study
                    event_data = self.conduct_event_study(
                        stock_data, 
                        market_data, 
                        event_date, 
                        event_name
                    )
                    
                    # Get the final cumulative abnormal return
                    car = event_data['cumulative_ar'].iloc[-1]
                    
                    # Calculate statistical significance for summary
                    car_std = event_data['abnormal_return'].std() * np.sqrt(len(event_data))
                    t_stat = car / (car_std / np.sqrt(len(event_data)))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(event_data)-1))
                    is_significant = "Yes" if p_value < 0.05 else "No"
                    
                    # Update summary document
                    with open(events_summary_path, 'a', encoding='utf-8') as f:
                        f.write(f"| {event_date} | {event_name} | {car*100:.4f}% | {is_significant} |\n")
                
                except Exception as e:
                    error_msg = f"Error analyzing event {event_name} ({event_date}): {str(e)}"
                    logger.error(error_msg)
                    print(error_msg)
                    
                    # Try to read the event report file to extract CAR and significance
                    try:
                        event_dir = self.results_dir / f"event_{pd.to_datetime(event_date).strftime('%Y%m%d')}_{event_name.replace(' ', '_')}"
                        report_path = event_dir / 'event_analysis_report.md'
                        
                        if report_path.exists():
                            # Extract CAR and significance from the report
                            with open(report_path, 'r', encoding='utf-8') as report_file:
                                report_content = report_file.read()
                                
                                # Extract CAR
                                car_match = re.search(r'\*\*CAR\*\*: ([\d\.-]+)%', report_content)
                                if car_match:
                                    car_value = car_match.group(1)
                                    
                                    # Extract significance
                                    sig_match = re.search(r'\*\*Significance\*\*: (\w+)', report_content)
                                    is_significant = "Yes" if sig_match and "Significant" in sig_match.group(1) else "No"
                                    
                                    # Update summary with extracted values
                                    with open(events_summary_path, 'a', encoding='utf-8') as f:
                                        f.write(f"| {event_date} | {event_name} | {car_value}% | {is_significant} |\n")
                                    continue
                        
                        # Fall back to error status if extraction fails
                        with open(events_summary_path, 'a', encoding='utf-8') as f:
                            f.write(f"| {event_date} | {event_name} | Error | N/A |\n")
                            
                    except Exception as inner_e:
                        logger.error(f"Error extracting data from report: {str(inner_e)}")
                        with open(events_summary_path, 'a', encoding='utf-8') as f:
                            f.write(f"| {event_date} | {event_name} | Error | N/A |\n")
            
            # Add overall conclusion to summary
            with open(events_summary_path, 'a', encoding='utf-8') as f:
                f.write("\n## Market Efficiency Conclusion\n\n")
                f.write(f"Based on the analysis of these events, the market for {self.stock_name} ({self.stock_code}) stock appears to exhibit varying degrees of efficiency. Please review the individual event reports for detailed analysis of each event's market efficiency implications.\n")
            
            print(f"\nMulti-event analysis completed successfully")
            print(f"Results saved to {self.results_dir} directory")
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            raise

    def full_analysis_pipeline(self, event_date, 
                             start_date='2022-01-01', 
                             end_date='2025-06-30'):
        """Complete analytical workflow for single stock event study"""
        try:
            logger.info(f"Starting analysis pipeline for stock {self.stock_code} ({self.stock_name}), event date {event_date}")
            print(f"Starting analysis for {self.stock_name} ({self.stock_code}), event date {event_date}")
            
            # Data acquisition
            market_data = self.get_market_data(start_date, end_date)
            stock_data = self.get_stock_data(start_date, end_date)
            
            logger.info("Market and stock data retrieved successfully")
            print("Market and stock data retrieved successfully")
            
            # CAPM Model estimation
            capm_model = self.estimate_capm(stock_data, market_data)
            logger.info(f"CAPM Model estimated: Alpha={self.alpha:.6f}, Beta={self.beta:.6f}")
            print(f"\nCAPM Model Results for {self.stock_name} ({self.stock_code}):")
            print(f"Alpha: {self.alpha:.6f}")
            print(f"Beta: {self.beta:.6f}")
            print(f"R-squared: {capm_model.rsquared:.6f}")
            
            # Efficiency testing
            lb_test = acorr_ljungbox(capm_model.resid, lags=10)
            logger.info(f"Market efficiency test completed")
            print("\nMarket Efficiency Test (Ljung-Box Test):")
            print(lb_test)
            
            # Event study
            event_data = self.conduct_event_study(stock_data, market_data, event_date)
            logger.info(f"Event study completed for {event_date}")
            
            print(f"\nEvent study for {self.stock_name} ({self.stock_code}) completed successfully")
            print(f"Results saved to {self.results_dir} directory")
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            raise

    def regenerate_summary(self, events_list, run_id=None):
        """Regenerate the summary file from existing event reports without running the full analysis"""
        logger.info("Regenerating summary file from existing event reports")
        
        # If run_id is provided, use it directly
        if run_id:
            run_dir = Path('results') / run_id
            if not run_dir.exists() or not run_dir.is_dir():
                error_msg = f"Specified run directory {run_dir} not found"
                logger.error(error_msg)
                print(error_msg)
                return
            self.results_dir = run_dir
            print(f"Using specified run directory: {run_dir}")
        else:
            # Find the most recent run directory
            results_base = Path('results')
            run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('run_')], 
                            key=lambda x: x.name, reverse=True)
            
            if not run_dirs:
                error_msg = "No previous run directories found"
                logger.error(error_msg)
                print(error_msg)
                return
                
            latest_run_dir = run_dirs[0]
            self.results_dir = latest_run_dir
            print(f"Using latest run directory: {latest_run_dir}")
        
        # Create a new summary document
        events_summary_path = self.results_dir / 'all_events_summary.md'
        with open(events_summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.stock_name} ({self.stock_code}) Event Study Analysis Summary\n\n")
            f.write(f"## Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Events Analyzed\n\n")
            f.write("| Event Date | Event Description | CAR | Significant? |\n")
            f.write("|------------|-------------------|-----|-------------|\n")
        
        # Process each event
        for event in events_list:
            event_date = event['date']
            event_name = event['name']
            
            print(f"Processing event: {event_name} ({event_date})")
            
            try:
                # Construct the event directory path
                event_dir = self.results_dir / f"event_{pd.to_datetime(event_date).strftime('%Y%m%d')}_{event_name.replace(' ', '_')}"
                report_path = event_dir / 'event_analysis_report.md'
                
                if not report_path.exists():
                    logger.warning(f"Event report not found for {event_name} ({event_date})")
                    print(f"Event report not found for {event_name} ({event_date})")
                    with open(events_summary_path, 'a', encoding='utf-8') as f:
                        f.write(f"| {event_date} | {event_name} | Not Found | N/A |\n")
                    continue
                
                # Extract CAR and significance from the report
                with open(report_path, 'r', encoding='utf-8') as report_file:
                    report_content = report_file.read()
                    
                    # Extract CAR
                    car_match = re.search(r'\*\*CAR\*\*: ([\d\.-]+)%', report_content)
                    if car_match:
                        car_value = car_match.group(1)
                        
                        # Extract significance
                        sig_match = re.search(r'\*\*Significance\*\*: (\w+)', report_content)
                        is_significant = "Yes" if sig_match and "Significant" in sig_match.group(1) else "No"
                        
                        # Update summary with extracted values
                        with open(events_summary_path, 'a', encoding='utf-8') as f:
                            f.write(f"| {event_date} | {event_name} | {car_value}% | {is_significant} |\n")
                    else:
                        logger.warning(f"Could not extract CAR for {event_name} ({event_date})")
                        print(f"Could not extract CAR for {event_name} ({event_date})")
                        with open(events_summary_path, 'a', encoding='utf-8') as f:
                            f.write(f"| {event_date} | {event_name} | Data Error | N/A |\n")
            
            except Exception as e:
                error_msg = f"Error processing event {event_name} ({event_date}): {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                with open(events_summary_path, 'a', encoding='utf-8') as f:
                    f.write(f"| {event_date} | {event_name} | Error | N/A |\n")
        
        # Add overall conclusion to summary
        with open(events_summary_path, 'a', encoding='utf-8') as f:
            f.write("\n## Market Efficiency Conclusion\n\n")
            f.write(f"Based on the analysis of these events, the market for {self.stock_name} ({self.stock_code}) stock appears to exhibit varying degrees of efficiency. Please review the individual event reports for detailed analysis of each event's market efficiency implications.\n")
        
        print(f"\nSummary regeneration completed successfully")
        print(f"Updated summary saved to {events_summary_path}")

# Execution block
if __name__ == "__main__":
    # Example usage
    analyzer = StockAnalyzer(stock_code='000725')  # BOE (京东方A)
    
    # Define the events
    boe_events = [
        {
            'date': '2024-06-12',
            'name': 'AMOLED Shipment Target Announcement'
        },
        {
            'date': '2024-08-28',
            'name': 'Shareholder Structure Disclosure'
        },
        {
            'date': '2024-10-31',
            'name': '2024 Second Interim Shareholder Meeting'
        },
        {
            'date': '2025-01-21',
            'name': '2024 Annual Performance Preview'
        },
        {
            'date': '2025-04-22',
            'name': '2024 Annual Shareholder Meeting'
        }
    ]
    
    # Analyze all events
    analyzer.analyze_multiple_events(
        events_list=boe_events,
        start_date='2022-01-01',
        end_date='2025-06-30'
    )
