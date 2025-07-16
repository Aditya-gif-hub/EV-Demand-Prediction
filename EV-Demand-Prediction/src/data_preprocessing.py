"""
Data preprocessing utilities for EV demand prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial cleaning of the EV dataset
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Cleaned pandas DataFrame
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert numeric columns with commas
    numeric_columns = [
        'Battery Electric Vehicles (BEVs)',
        'Plug-In Hybrid Electric Vehicles (PHEVs)',
        'Electric Vehicle (EV) Total',
        'Non-Electric Vehicle Total',
        'Total Vehicles'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', ''), 
            errors='coerce'
        )
    
    logger.info(f"Data cleaned. Shape: {df.shape}")
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the date column
    
    Args:
        df: Input DataFrame with Date column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Season
    df['Season'] = df['Month'].apply(lambda x: (x%12 + 3)//3)
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['Season_Name'] = df['Season'].map(season_map)
    
    return df

def create_lag_features(df: pd.DataFrame, 
                       target_col: str, 
                       lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series modeling
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    df = df.sort_values(['State', 'County', 'Date'])
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(['State', 'County'])[target_col].shift(lag)
    
    return df

def create_rolling_features(df: pd.DataFrame, 
                          target_col: str, 
                          windows: List[int]) -> pd.DataFrame:
    """
    Create rolling average features
    
    Args:
        df: Input DataFrame
        target_col: Column to calculate rolling averages for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    df = df.sort_values(['State', 'County', 'Date'])
    
    for window in windows:
        df[f'{target_col}_ma_{window}'] = (
            df.groupby(['State', 'County'])[target_col]
            .rolling(window=window)
            .mean()
            .reset_index(0, drop=True)
        )
    
    return df

if __name__ == "__main__":
    # Example usage
    df = load_and_clean_data('../data/raw/EV_DATASET.csv')
    df = create_time_features(df)
    df = create_lag_features(df, 'Electric Vehicle (EV) Total', [1, 3, 6, 12])
    df = create_rolling_features(df, 'Electric Vehicle (EV) Total', [3, 6, 12])
    
    df.to_csv('../data/processed/cleaned_ev_data.csv', index=False)
    logger.info("Preprocessing complete!")