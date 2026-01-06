import pandas as pd
import pytest
import sys
import os
from pathlib import Path

# Adjust path to include project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import normalize_state_names, parse_dates, validate_pincode

def test_normalize_state_names():
    # Create dummy data
    data = {
        'state': ['West Bengal', 'WestBENGAL', 'andhra pradesh', 'Orissa', '123']
    }
    df = pd.DataFrame(data)
    
    cleaned_df = normalize_state_names(df, 'state')
    
    # Assertions
    assert "West Bengal" in cleaned_df['state'].values
    assert "Andhra Pradesh" in cleaned_df['state'].values
    assert "Odisha" in cleaned_df['state'].values # Orissa -> Odisha
    assert "123" not in cleaned_df['state'].values # Numeric should be removed
    assert len(cleaned_df) == 4

def test_parse_dates():
    data = {'date': ['01-01-2023', 'invalid', '2023/01/01']}
    df = pd.DataFrame(data)
    
    cleaned_df = parse_dates(df, 'date')
    
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['date'])
    assert not pd.isna(cleaned_df.iloc[0]['date'])
    assert pd.isna(cleaned_df.iloc[1]['date']) # Invalid format should be NaT

def test_validate_pincode():
    data = {'pincode': ['110001', '123', 'abcdef', '1100012']}
    df = pd.DataFrame(data)
    
    cleaned_df = validate_pincode(df, 'pincode')
    
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]['pincode'] == '110001'
