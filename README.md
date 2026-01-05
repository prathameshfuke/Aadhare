# Aadhaar Data Analysis Project

Comprehensive analysis of Aadhaar enrolment and update patterns to identify societal trends, geographic disparities, and actionable insights for policy and system improvements.

## Project Overview

This project analyzes three Aadhaar datasets (as of December 31, 2025):
- **Enrolment Dataset**: ~1M records of new Aadhaar registrations by age group
- **Demographic Updates**: ~2M records of name, address, DOB, gender, mobile updates
- **Biometric Updates**: ~1.8M records of fingerprint, iris, and face updates

## Key Findings

1. **Total Activity**: 160M+ new enrolments, 1.1B+ demographic updates, 490M+ biometric updates processed
2. **Geographic Concentration**: Top 5 states account for majority of enrolments
3. **Age Distribution**: Adults (18+) dominate at ~41% of new enrolments
4. **Youth Transitions**: Significant variation in child-to-adult biometric updates across states
5. **Anomalies**: Multiple days with unusual activity patterns identified

## Project Structure

```
AadharHackathon/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Initial data profiling
│   ├── 02_cleaning_preprocessing.ipynb # Data quality and cleaning
│   ├── 03_analysis.ipynb              # All analytical procedures
│   └── 04_visualizations.ipynb        # Publication-ready charts
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset loading utilities
│   ├── preprocessing.py               # Cleaning and transformation
│   ├── analysis.py                    # Analysis functions
│   ├── visualization.py               # Visualization generators
│   └── run_analysis.py                # Main pipeline script
├── outputs/
│   ├── figures/                       # 16 generated visualizations
│   └── report.pdf                     # Comprehensive PDF report
├── data/
│   └── raw/                           # Raw CSV files (not in repo)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/prathameshfuke/Aadhare.git
cd Aadhare

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Analysis Pipeline

```bash
python src/run_analysis.py
```

This will:
1. Load and preprocess all datasets
2. Run temporal, geographic, demographic, and anomaly analyses
3. Generate 16 visualizations
4. Create comprehensive PDF report

### Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Analysis Components

### A. Temporal Analysis
- Daily/weekly/monthly trends
- Seasonal patterns
- Growth rate calculations
- Rolling averages

### B. Geographic Analysis
- State and district-level aggregations
- Hotspot/coldspot identification
- Regional disparity metrics

### C. Demographic Analysis
- Age group distribution
- Youth transition patterns
- Update frequency by demographics

### D. Anomaly Detection
- IQR-based outlier detection
- System stress period identification
- Unusual activity patterns

### E. Comparative Analysis
- Enrolment vs update rates
- Cross-state performance
- Demographic vs biometric patterns

## Visualizations

16 publication-ready visualizations generated:
- Time series trends (enrolment, demographic, biometric)
- State comparison bar charts
- Age distribution pie/bar charts
- Monthly heatmaps
- Day of week patterns
- Anomaly detection plots
- Comparative state analysis
- Youth transition rates
- Cumulative growth charts
- Geographic heatmaps
- Summary dashboard

## Technologies Used

- **Python 3.14**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Static visualizations
- **seaborn** - Statistical plots
- **scipy** - Statistical analysis
- **scikit-learn** - Anomaly detection
- **fpdf2** - PDF report generation

## PDF Report

The generated `outputs/report.pdf` includes:
1. Executive Summary
2. Problem Statement & Approach
3. Dataset Description
4. Methodology
5. Analysis & Visualizations
6. Key Insights & Recommendations

## Data

Raw data files should be placed in `data/raw/`:
- `api_data_aadhar_enrolment/` - Enrolment CSVs
- `api_data_aadhar_demographic/` - Demographic update CSVs
- `api_data_aadhar_biometric/` - Biometric update CSVs

**Note**: Raw data files are not included in the repository due to size.

## License

MIT License
