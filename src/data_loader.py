import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def load_enrolment_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate all enrolment CSV files."""
    base_path = data_dir or DATA_DIR / "api_data_aadhar_enrolment"
    files = sorted(base_path.glob("*.csv"))
    
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_demographic_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate all demographic update CSV files."""
    base_path = data_dir or DATA_DIR / "api_data_aadhar_demographic"
    files = sorted(base_path.glob("*.csv"))
    
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_biometric_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate all biometric update CSV files."""
    base_path = data_dir or DATA_DIR / "api_data_aadhar_biometric"
    files = sorted(base_path.glob("*.csv"))
    
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_all_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three datasets and return as tuple."""
    return (
        load_enrolment_data(),
        load_demographic_data(),
        load_biometric_data()
    )

if __name__ == "__main__":
    enrolment, demographic, biometric = load_all_datasets()
    print(f"Enrolment: {len(enrolment):,} records")
    print(f"Demographic: {len(demographic):,} records")
    print(f"Biometric: {len(biometric):,} records")
