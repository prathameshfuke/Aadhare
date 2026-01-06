import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def load_from_files(files: list) -> pd.DataFrame:
    """Load dataframe from list of file paths or buffers."""
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_enrolment_data(data_dir: Optional[Path] = None, files: Optional[list] = None) -> pd.DataFrame:
    """Load enrolment data from directory or provided files."""
    if files:
        return load_from_files(files)
        
    base_path = data_dir or DATA_DIR / "api_data_aadhar_enrolment"
    file_paths = sorted(base_path.glob("*.csv"))
    return load_from_files(file_paths)

def load_demographic_data(data_dir: Optional[Path] = None, files: Optional[list] = None) -> pd.DataFrame:
    """Load demographic data from directory or provided files."""
    if files:
        return load_from_files(files)

    base_path = data_dir or DATA_DIR / "api_data_aadhar_demographic"
    file_paths = sorted(base_path.glob("*.csv"))
    return load_from_files(file_paths)

def load_biometric_data(data_dir: Optional[Path] = None, files: Optional[list] = None) -> pd.DataFrame:
    """Load biometric data from directory or provided files."""
    if files:
        return load_from_files(files)

    base_path = data_dir or DATA_DIR / "api_data_aadhar_biometric"
    file_paths = sorted(base_path.glob("*.csv"))
    return load_from_files(file_paths)

def load_all_datasets(enrol_files=None, demo_files=None, bio_files=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all datasets, optionally from provided file lists."""
    return (
        load_enrolment_data(files=enrol_files),
        load_demographic_data(files=demo_files),
        load_biometric_data(files=bio_files)
    )

if __name__ == "__main__":
    enrolment, demographic, biometric = load_all_datasets()
    print(f"Enrolment: {len(enrolment):,} records")
    print(f"Demographic: {len(demographic):,} records")
    print(f"Biometric: {len(biometric):,} records")
