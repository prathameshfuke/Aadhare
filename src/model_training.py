import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
import sys
import os

# Adjust path to include project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_all, get_data_quality_report
from src.data_loader import load_enrolment_data, load_demographic_data, load_biometric_data

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def prepare_features(df: pd.DataFrame, target_col: str = 'total'):
    """Create time-based features for ML."""
    df = df.copy()
    df = df.sort_values('date')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Rolling features
    df['rolling_7_mean'] = df[target_col].rolling(window=7).mean().shift(1)
    
    # Drop NaN created by rolling/shifting
    df = df.dropna()
    return df

def train_anomaly_detector(df: pd.DataFrame, target_col: str = 'total'):
    """Train Isolation Forest for anomaly detection."""
    print("Training Anomaly Detector (Isolation Forest)...")
    model = IsolationForest(contamination=0.05, random_state=42)
    # Reshape for sklearn
    X = df[[target_col]]
    model.fit(X)
    
    joblib.dump(model, MODELS_DIR / "anomaly_model.joblib")
    print("  Anomaly model saved.")
    return model

def train_forecaster(df: pd.DataFrame, target_col: str = 'total'):
    """Train Random Forest for forecasting."""
    print("Training Forecaster (Random Forest)...")
    
    features = ['day_of_week', 'month', 'year', 'day_of_month', 'is_weekend', 'rolling_7_mean']
    X = df[features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  Forecasting MAE: {mae:.2f}")
    
    # Retrain on full data for production
    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / "forecast_model.joblib")
    print("  Forecast model saved.")
    return model

def main():
    print("Loading data...")
    # Load all raw data
    enrol_raw = load_enrolment_data()
    # We only need enrolment for this specific ML demo, but preprocess_all expects 3
    # We can pass empty for others if needed, or just load them to be safe
    demo_raw = load_demographic_data()
    bio_raw = load_biometric_data()
    
    print("Preprocessing...")
    enrol, _, _ = preprocess_all(enrol_raw, demo_raw, bio_raw)
    
    # Aggregate Enrolment to Daily Level
    # preprocess_all returns cleaned raw data, we need to aggregate
    daily_enrol = enrol.groupby('date').size().reset_index(name='total')
    
    # Feature Engineering
    print("Feature Engineering...")
    ml_df = prepare_features(daily_enrol, 'total')
    
    # Train Models
    train_anomaly_detector(daily_enrol, 'total') # Use raw daily for anomaly (univariate)
    train_forecaster(ml_df, 'total') # Use featurized for forecast
    
    print("Done! Models saved to models/")

if __name__ == "__main__":
    main()
