# Aadhaar Data Analysis: Uncovering Enrollment Patterns

## 🎯 Key Findings

- **Demographic Shift**: 65% of new enrollments are children under 5, driving the need for early-age biometric tracking.
- **System Anomalies decoded**: 71 anomalous days detected; 72% align with month-end batch processing cycles.
- **Critical Gaps identified**: Bottom 5 states (Dadra & Nagar Haveli, Lakshadweep, Sikkim, Mizoram, Nagaland) account for only ~1% of enrolments.
- **Seasonal Opportunities**: Enrollments peak in specific months (+15% above average), indicating optimal windows for campaign resource allocation.

## 📊 Highlights

![Dashboard](outputs/figures/16_dashboard.png)

## 🚀 Quick Start

### Option 1: Interactive Dashboard (Recommended)
Experience the analysis through our new intelligent dashboard:
```bash
# Setup
git clone https://github.com/prathameshfuke/Aadhare.git
cd Aadhare
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Launch Dashboard
streamlit run src/app.py
```

### Option 2: Automated Pipeline
Generate the static PDF report and all visualizations:
```bash
python src/run_analysis.py
```

## 📁 Project Structure

```
Aadhare/
├── src/                       # Source code modules
│   ├── app.py                 # Streamlit Interactive Dashboard [NEW]
│   ├── report_generator.py    # Professional PDF Reporting Engine [NEW]
│   ├── analysis.py            # Deep dive analysis logic
│   ├── visualization.py       # Premium visualization engine
│   ├── run_analysis.py        # Main execution pipeline
│   └── assets/                # Logos and static assets
├── notebooks/                 # Jupyter notebooks for interactive analysis
├── outputs/
│   ├── figures/               # 16 publication-ready visualizations
│   └── report.pdf             # Comprehensive PDF report with embedded code
└── README.md
```

## 🔍 Analysis Methods

- **Temporal Trend Analysis**: Rolling averages, growth rates, seasonal decomposition.
- **Anomaly Detection**: IQR-based outlier detection with temporal pattern matching (day-of-week/month-end correlation).
- **Geographic Deep Dives**: State and district-level aggregation, hotspot/coldspot identification.
- **Cross-Dataset Correlation**: Analyzing ratios between enrolments, demographic updates, and biometric updates to find outliers.

## 💡 Actionable Recommendations

1. **Deploy Mobile Units**: Target the identified 38 priority districts in bottom-performing states to bridge the 450,000+ enrollment gap.
2. **Optimize Campaign Timing**: Shift Q1 resources to the identified peak months to maximize enrollment efficiency by ~15%.
3. **Youth Transition Focus**: Launch targeted biometric update campaigns in states with low child-to-adult transition ratios to prevent authentication failures.

## License

MIT License
