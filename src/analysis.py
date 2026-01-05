import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

def temporal_trends(df: pd.DataFrame, value_col: str, date_col: str = 'date',
                    freq: str = 'D') -> pd.DataFrame:
    """Aggregate values by time frequency and calculate trends."""
    daily = df.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum().reset_index()
    daily.columns = ['date', 'total']
    
    daily['rolling_7d'] = daily['total'].rolling(window=7, min_periods=1).mean()
    daily['rolling_30d'] = daily['total'].rolling(window=30, min_periods=1).mean()
    daily['pct_change'] = daily['total'].pct_change() * 100
    daily['cumulative'] = daily['total'].cumsum()
    
    return daily

def state_aggregations(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate values by state with rankings."""
    state_totals = df.groupby('state')[value_col].sum().reset_index()
    state_totals.columns = ['state', 'total']
    state_totals = state_totals.sort_values('total', ascending=False)
    state_totals['rank'] = range(1, len(state_totals) + 1)
    state_totals['pct_of_total'] = state_totals['total'] / state_totals['total'].sum() * 100
    state_totals['cumulative_pct'] = state_totals['pct_of_total'].cumsum()
    return state_totals

def district_aggregations(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate values by state and district."""
    district_totals = df.groupby(['state', 'district'])[value_col].sum().reset_index()
    district_totals.columns = ['state', 'district', 'total']
    district_totals = district_totals.sort_values('total', ascending=False)
    return district_totals

def age_group_analysis(enrolment_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze enrolment distribution by age groups."""
    age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    totals = enrolment_df[age_cols].sum()
    
    result = pd.DataFrame({
        'age_group': ['0-5 years', '5-17 years', '18+ years'],
        'total': totals.values,
        'percentage': (totals.values / totals.sum() * 100)
    })
    return result

def monthly_patterns(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Analyze monthly patterns for seasonality."""
    monthly = df.groupby(['year', 'month'])[value_col].sum().reset_index()
    monthly['year_month'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
    
    month_avg = df.groupby('month')[value_col].mean().reset_index()
    month_avg.columns = ['month', 'avg_value']
    month_avg['month_name'] = pd.to_datetime(month_avg['month'], format='%m').dt.month_name()
    
    return monthly, month_avg

def detect_anomalies_iqr(df: pd.DataFrame, value_col: str, 
                         multiplier: float = 1.5) -> pd.DataFrame:
    """Detect anomalies using IQR method."""
    df = df.copy()
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    df['is_anomaly'] = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
    df['anomaly_type'] = np.where(df[value_col] < lower_bound, 'low',
                                   np.where(df[value_col] > upper_bound, 'high', 'normal'))
    return df

def detect_anomalies_zscore(df: pd.DataFrame, value_col: str, 
                            threshold: float = 3.0) -> pd.DataFrame:
    """Detect anomalies using Z-score method."""
    df = df.copy()
    df['zscore'] = np.abs(stats.zscore(df[value_col].fillna(0)))
    df['is_anomaly'] = df['zscore'] > threshold
    return df

def growth_rate_analysis(df: pd.DataFrame, date_col: str = 'date', 
                         value_col: str = 'total') -> Dict:
    """Calculate growth rates over different periods."""
    df = df.sort_values(date_col)
    
    total_growth = (df[value_col].iloc[-1] - df[value_col].iloc[0]) / df[value_col].iloc[0] * 100 if df[value_col].iloc[0] != 0 else 0
    
    if len(df) >= 7:
        weekly_growth = (df[value_col].iloc[-1] - df[value_col].iloc[-7]) / df[value_col].iloc[-7] * 100 if df[value_col].iloc[-7] != 0 else 0
    else:
        weekly_growth = None
    
    if len(df) >= 30:
        monthly_growth = (df[value_col].iloc[-1] - df[value_col].iloc[-30]) / df[value_col].iloc[-30] * 100 if df[value_col].iloc[-30] != 0 else 0
    else:
        monthly_growth = None
    
    return {
        'total_growth_pct': total_growth,
        'weekly_growth_pct': weekly_growth,
        'monthly_growth_pct': monthly_growth,
        'avg_daily': df[value_col].mean(),
        'max_daily': df[value_col].max(),
        'min_daily': df[value_col].min(),
    }

def comparative_state_metrics(enrolment: pd.DataFrame, demographic: pd.DataFrame,
                               biometric: pd.DataFrame) -> pd.DataFrame:
    """Compare enrolment and update rates across states."""
    enrol_state = enrolment.groupby('state')['total_enrolments'].sum().reset_index()
    enrol_state.columns = ['state', 'enrolments']
    
    demo_state = demographic.groupby('state')['total_updates'].sum().reset_index()
    demo_state.columns = ['state', 'demo_updates']
    
    bio_state = biometric.groupby('state')['total_updates'].sum().reset_index()
    bio_state.columns = ['state', 'bio_updates']
    
    merged = enrol_state.merge(demo_state, on='state', how='outer')
    merged = merged.merge(bio_state, on='state', how='outer')
    merged = merged.fillna(0)
    
    merged['demo_to_enrol_ratio'] = merged['demo_updates'] / merged['enrolments'].replace(0, np.nan)
    merged['bio_to_enrol_ratio'] = merged['bio_updates'] / merged['enrolments'].replace(0, np.nan)
    merged['total_activity'] = merged['enrolments'] + merged['demo_updates'] + merged['bio_updates']
    
    return merged.sort_values('total_activity', ascending=False)

def identify_hotspots(df: pd.DataFrame, value_col: str, 
                      percentile: float = 90) -> pd.DataFrame:
    """Identify geographic hotspots based on activity percentile."""
    threshold = df[value_col].quantile(percentile / 100)
    hotspots = df[df[value_col] >= threshold].copy()
    hotspots['hotspot_rank'] = hotspots[value_col].rank(ascending=False)
    return hotspots

def identify_coldspots(df: pd.DataFrame, value_col: str, 
                       percentile: float = 10) -> pd.DataFrame:
    """Identify geographic coldspots based on activity percentile."""
    threshold = df[value_col].quantile(percentile / 100)
    coldspots = df[df[value_col] <= threshold].copy()
    coldspots['coldspot_rank'] = coldspots[value_col].rank(ascending=True)
    return coldspots

def youth_transition_analysis(enrolment: pd.DataFrame, biometric: pd.DataFrame) -> pd.DataFrame:
    """Analyze child-to-adult biometric transition patterns."""
    enrol_youth = enrolment.groupby('state')['age_5_17'].sum().reset_index()
    enrol_youth.columns = ['state', 'youth_enrolments']
    
    bio_youth = biometric.groupby('state')['bio_age_5_17'].sum().reset_index()
    bio_youth.columns = ['state', 'youth_bio_updates']
    
    merged = enrol_youth.merge(bio_youth, on='state', how='outer').fillna(0)
    merged['transition_ratio'] = merged['youth_bio_updates'] / merged['youth_enrolments'].replace(0, np.nan)
    
    return merged.sort_values('transition_ratio', ascending=False)

def weekly_pattern_analysis(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Analyze patterns by day of week."""
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_totals = df.groupby('day_of_week')[value_col].agg(['sum', 'mean', 'count']).reset_index()
    dow_totals['day_name'] = dow_totals['day_of_week'].map(lambda x: dow_names[x])
    dow_totals = dow_totals.sort_values('day_of_week')
    return dow_totals
