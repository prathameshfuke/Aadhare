import pandas as pd
import numpy as np
from typing import Tuple

STATE_NAME_MAP = {
    'Andaman And Nicobar Islands': 'Andaman & Nicobar',
    'Andaman and Nicobar Islands': 'Andaman & Nicobar',
    'Andhra Pradesh': 'Andhra Pradesh',
    'Andhra pradesh': 'Andhra Pradesh',
    'Dadra And Nagar Haveli': 'Dadra & Nagar Haveli',
    'Dadra and Nagar Haveli': 'Dadra & Nagar Haveli',
    'Dadra And Nagar Haveli And Daman And Diu': 'Dadra & Nagar Haveli and Daman & Diu',
    'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra & Nagar Haveli and Daman & Diu',
    'Daman And Diu': 'Daman & Diu',
    'Daman and Diu': 'Daman & Diu',
    'Jammu And Kashmir': 'Jammu & Kashmir',
    'Jammu and Kashmir': 'Jammu & Kashmir',
    'Orissa': 'Odisha',
    'ODISHA': 'Odisha',
    'Pondicherry': 'Puducherry',
    'West Bangal': 'West Bengal',
    'Westbengal': 'West Bengal',
    'West bengal': 'West Bengal',
    'WEST BENGAL': 'West Bengal',
    'WESTBENGAL': 'West Bengal',
    'West  Bengal': 'West Bengal',
}

def parse_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Convert date strings to datetime objects."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
    return df

def validate_pincode(df: pd.DataFrame, pincode_col: str = 'pincode') -> pd.DataFrame:
    """Filter to valid 6-digit PIN codes."""
    df = df.copy()
    df[pincode_col] = df[pincode_col].astype(str).str.strip()
    valid_mask = df[pincode_col].str.match(r'^\d{6}$', na=False)
    return df[valid_mask].copy()

def normalize_state_names(df: pd.DataFrame, state_col: str = 'state') -> pd.DataFrame:
    """Standardize state names to consistent format."""
    df = df.copy()
    # Filter out numeric states (bad data)
    df = df[~df[state_col].astype(str).str.match(r'^\d+$', na=False)]
    
    # Capitalize properly and strip
    df[state_col] = df[state_col].astype(str).str.strip()
    # Handle specific case-insensitive matches first
    df[state_col] = df[state_col].replace(STATE_NAME_MAP)
    # Then title case
    df[state_col] = df[state_col].str.title()
    # Apply map again to catch any title-cased variations
    df[state_col] = df[state_col].replace(STATE_NAME_MAP)
    return df

def add_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Extract year, month, quarter, and day features from date column."""
    df = df.copy()
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month_name'] = df[date_col].dt.month_name()
    df['week'] = df[date_col].dt.isocalendar().week.astype(int)
    return df

def add_enrolment_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Add total enrolments column for enrolment dataset."""
    df = df.copy()
    age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    if all(col in df.columns for col in age_cols):
        df['total_enrolments'] = df[age_cols].sum(axis=1)
    return df

def add_update_totals(df: pd.DataFrame, prefix: str = 'demo') -> pd.DataFrame:
    """Add total updates column for demographic/biometric datasets."""
    df = df.copy()
    update_cols = [col for col in df.columns if col.startswith(prefix)]
    if update_cols:
        df['total_updates'] = df[update_cols].sum(axis=1)
    return df

def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with null geography or date fields."""
    df = df.copy()
    required_cols = ['date', 'state', 'district', 'pincode']
    existing_required = [c for c in required_cols if c in df.columns]
    return df.dropna(subset=existing_required)

def preprocess_enrolment(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for enrolment data."""
    df = parse_dates(df)
    df = remove_invalid_records(df)
    df = validate_pincode(df)
    df = normalize_state_names(df)
    df = add_temporal_features(df)
    df = add_enrolment_totals(df)
    return df.reset_index(drop=True)

def preprocess_demographic(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for demographic update data."""
    df = parse_dates(df)
    df = remove_invalid_records(df)
    df = validate_pincode(df)
    df = normalize_state_names(df)
    df = add_temporal_features(df)
    df = add_update_totals(df, prefix='demo')
    return df.reset_index(drop=True)

def preprocess_biometric(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for biometric update data."""
    df = parse_dates(df)
    df = remove_invalid_records(df)
    df = validate_pincode(df)
    df = normalize_state_names(df)
    df = add_temporal_features(df)
    df = add_update_totals(df, prefix='bio')
    return df.reset_index(drop=True)

def get_data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """Generate data quality metrics for a dataset."""
    return {
        'name': name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
    }

def preprocess_all(enrolment: pd.DataFrame, demographic: pd.DataFrame, 
                   biometric: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess all three datasets."""
    return (
        preprocess_enrolment(enrolment),
        preprocess_demographic(demographic),
        preprocess_biometric(biometric)
    )
