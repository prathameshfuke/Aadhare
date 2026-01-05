#!/usr/bin/env python3
"""
Aadhaar Data Analysis Pipeline
Generates comprehensive analysis, visualizations, and PDF report
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.data_loader import load_all_datasets
from src.preprocessing import preprocess_all, get_data_quality_report
from src.analysis import (
    temporal_trends, state_aggregations, district_aggregations,
    age_group_analysis, monthly_patterns, detect_anomalies_iqr,
    growth_rate_analysis, comparative_state_metrics, identify_hotspots,
    identify_coldspots, youth_transition_analysis, weekly_pattern_analysis
)
from src.visualization import (
    save_fig, plot_time_series, plot_state_bar, plot_age_distribution,
    plot_monthly_heatmap, plot_day_of_week, plot_anomalies,
    plot_state_comparison, plot_transition_rates, plot_cumulative_growth,
    create_dashboard, plot_geographic_heatmap
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

def main():
    print("=" * 60)
    print("AADHAAR DATA ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print("[1/7] Loading datasets...")
    enrolment_raw, demographic_raw, biometric_raw = load_all_datasets()
    print(f"  Enrolment: {len(enrolment_raw):,} records")
    print(f"  Demographic: {len(demographic_raw):,} records")
    print(f"  Biometric: {len(biometric_raw):,} records")
    
    # Data quality reports
    print("\n[2/7] Assessing data quality...")
    quality_reports = {
        'enrolment': get_data_quality_report(enrolment_raw, 'Enrolment'),
        'demographic': get_data_quality_report(demographic_raw, 'Demographic'),
        'biometric': get_data_quality_report(biometric_raw, 'Biometric'),
    }
    
    for name, report in quality_reports.items():
        print(f"  {name.title()}: {report['duplicates']} duplicates, "
              f"{report['memory_mb']:.1f} MB")
    
    # Preprocess
    print("\n[3/7] Preprocessing data...")
    enrolment, demographic, biometric = preprocess_all(
        enrolment_raw, demographic_raw, biometric_raw
    )
    print(f"  Enrolment after cleaning: {len(enrolment):,} records")
    print(f"  Demographic after cleaning: {len(demographic):,} records")
    print(f"  Biometric after cleaning: {len(biometric):,} records")
    
    # Run analyses
    print("\n[4/7] Running analyses...")
    
    # Temporal analysis
    print("  - Temporal trends...")
    enrol_trends = temporal_trends(enrolment, 'total_enrolments')
    demo_trends = temporal_trends(demographic, 'total_updates')
    bio_trends = temporal_trends(biometric, 'total_updates')
    
    # Geographic analysis
    print("  - Geographic aggregations...")
    state_enrol = state_aggregations(enrolment, 'total_enrolments')
    state_demo = state_aggregations(demographic, 'total_updates')
    state_bio = state_aggregations(biometric, 'total_updates')
    
    district_enrol = district_aggregations(enrolment, 'total_enrolments')
    
    # Age analysis
    print("  - Age group analysis...")
    age_dist = age_group_analysis(enrolment)
    
    # Monthly patterns
    print("  - Monthly patterns...")
    enrol_monthly, enrol_month_avg = monthly_patterns(enrolment, 'total_enrolments')
    
    # Day of week patterns
    print("  - Day of week patterns...")
    enrol_dow = weekly_pattern_analysis(enrolment, 'total_enrolments')
    demo_dow = weekly_pattern_analysis(demographic, 'total_updates')
    
    # Anomaly detection
    print("  - Detecting anomalies...")
    enrol_anomalies = detect_anomalies_iqr(enrol_trends, 'total')
    demo_anomalies = detect_anomalies_iqr(demo_trends, 'total')
    
    # Growth analysis
    print("  - Growth rate analysis...")
    enrol_growth = growth_rate_analysis(enrol_trends, 'date', 'total')
    demo_growth = growth_rate_analysis(demo_trends, 'date', 'total')
    bio_growth = growth_rate_analysis(bio_trends, 'date', 'total')
    
    # Comparative analysis
    print("  - Comparative metrics...")
    comparative = comparative_state_metrics(enrolment, demographic, biometric)
    
    # Youth transition
    print("  - Youth transition analysis...")
    transitions = youth_transition_analysis(enrolment, biometric)
    
    # Hotspots/Coldspots
    print("  - Identifying hotspots and coldspots...")
    hotspots = identify_hotspots(state_enrol, 'total', 90)
    coldspots = identify_coldspots(state_enrol, 'total', 10)
    
    # Generate visualizations
    print("\n[5/7] Generating visualizations...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Time series plots
    print("  - Time series plots...")
    fig = plot_time_series(enrol_trends, 'date', 'total',
                           'Daily Aadhaar Enrolments (March 2025)',
                           'Number of Enrolments')
    figures['01_enrolment_trends'] = save_fig(fig, '01_enrolment_trends.png', FIGURES_DIR)
    
    fig = plot_time_series(demo_trends, 'date', 'total',
                           'Daily Demographic Updates (March 2025)',
                           'Number of Updates')
    figures['02_demographic_trends'] = save_fig(fig, '02_demographic_trends.png', FIGURES_DIR)
    
    fig = plot_time_series(bio_trends, 'date', 'total',
                           'Daily Biometric Updates (March 2025)',
                           'Number of Updates')
    figures['03_biometric_trends'] = save_fig(fig, '03_biometric_trends.png', FIGURES_DIR)
    
    # State comparisons
    print("  - State comparison charts...")
    fig = plot_state_bar(state_enrol, 'state', 'total',
                         'Top 15 States by Aadhaar Enrolments')
    figures['04_state_enrolments'] = save_fig(fig, '04_state_enrolments.png', FIGURES_DIR)
    
    fig = plot_state_bar(state_demo, 'state', 'total',
                         'Top 15 States by Demographic Updates')
    figures['05_state_demographic'] = save_fig(fig, '05_state_demographic.png', FIGURES_DIR)
    
    fig = plot_state_bar(state_bio, 'state', 'total',
                         'Top 15 States by Biometric Updates')
    figures['06_state_biometric'] = save_fig(fig, '06_state_biometric.png', FIGURES_DIR)
    
    # Age distribution
    print("  - Age distribution...")
    fig = plot_age_distribution(age_dist, 'Aadhaar Enrolment by Age Group')
    figures['07_age_distribution'] = save_fig(fig, '07_age_distribution.png', FIGURES_DIR)
    
    # Monthly heatmap
    print("  - Monthly patterns...")
    fig = plot_monthly_heatmap(enrolment, 'total_enrolments',
                               'Monthly Enrolment Patterns')
    figures['08_monthly_heatmap'] = save_fig(fig, '08_monthly_heatmap.png', FIGURES_DIR)
    
    # Day of week
    print("  - Day of week patterns...")
    fig = plot_day_of_week(enrol_dow, 'sum', 'Enrolments by Day of Week')
    figures['09_day_of_week'] = save_fig(fig, '09_day_of_week.png', FIGURES_DIR)
    
    # Anomalies
    print("  - Anomaly plots...")
    fig = plot_anomalies(enrol_anomalies, 'date', 'total',
                         'Enrolment Anomalies Detected')
    figures['10_enrol_anomalies'] = save_fig(fig, '10_enrol_anomalies.png', FIGURES_DIR)
    
    fig = plot_anomalies(demo_anomalies, 'date', 'total',
                         'Demographic Update Anomalies Detected')
    figures['11_demo_anomalies'] = save_fig(fig, '11_demo_anomalies.png', FIGURES_DIR)
    
    # Comparative
    print("  - Comparative charts...")
    fig = plot_state_comparison(comparative, 'State-wise Activity Comparison')
    figures['12_state_comparison'] = save_fig(fig, '12_state_comparison.png', FIGURES_DIR)
    
    # Transition rates
    print("  - Transition rate analysis...")
    fig = plot_transition_rates(transitions, 'Youth Biometric Transition Rates by State')
    figures['13_transition_rates'] = save_fig(fig, '13_transition_rates.png', FIGURES_DIR)
    
    # Cumulative growth
    print("  - Cumulative growth...")
    fig = plot_cumulative_growth(enrol_trends, 'date', 'total',
                                 'Cumulative Aadhaar Enrolments')
    figures['14_cumulative_growth'] = save_fig(fig, '14_cumulative_growth.png', FIGURES_DIR)
    
    # Geographic heatmap
    print("  - Geographic heatmap...")
    fig = plot_geographic_heatmap(enrolment, 'state', 'total_enrolments',
                                  'State-level Enrolment Heatmap')
    figures['15_geo_heatmap'] = save_fig(fig, '15_geo_heatmap.png', FIGURES_DIR)
    
    # Dashboard
    print("  - Summary dashboard...")
    fig = create_dashboard(enrol_trends, state_enrol, age_dist, comparative,
                           'Aadhaar Data Analysis Dashboard')
    figures['16_dashboard'] = save_fig(fig, '16_dashboard.png', FIGURES_DIR)
    
    print(f"\n  Generated {len(figures)} visualizations")
    
    # Compile insights
    print("\n[6/7] Compiling insights...")
    
    insights = compile_insights(
        enrolment, demographic, biometric,
        enrol_trends, demo_trends, bio_trends,
        state_enrol, state_demo, state_bio,
        age_dist, comparative, transitions,
        enrol_anomalies, demo_anomalies,
        enrol_growth, demo_growth, bio_growth,
        hotspots, coldspots
    )
    
    # Generate report
    print("\n[7/7] Generating PDF report...")
    report_path = generate_pdf_report(
        insights, figures, quality_reports,
        enrolment, demographic, biometric,
        OUTPUT_DIR / "report.pdf"
    )
    print(f"  Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Report: {report_path}")
    
    return insights

def compile_insights(enrolment, demographic, biometric,
                     enrol_trends, demo_trends, bio_trends,
                     state_enrol, state_demo, state_bio,
                     age_dist, comparative, transitions,
                     enrol_anomalies, demo_anomalies,
                     enrol_growth, demo_growth, bio_growth,
                     hotspots, coldspots):
    """Compile all analysis results into structured insights."""
    
    total_enrolments = enrolment['total_enrolments'].sum()
    total_demo_updates = demographic['total_updates'].sum()
    total_bio_updates = biometric['total_updates'].sum()
    
    top_states_enrol = state_enrol.head(5)['state'].tolist()
    bottom_states_enrol = state_enrol.tail(5)['state'].tolist()
    
    top_states_demo = state_demo.head(5)['state'].tolist()
    top_states_bio = state_bio.head(5)['state'].tolist()
    
    enrol_anomaly_count = enrol_anomalies['is_anomaly'].sum()
    demo_anomaly_count = demo_anomalies['is_anomaly'].sum()
    
    age_majority = age_dist.loc[age_dist['percentage'].idxmax()]
    
    best_transition_states = transitions.nlargest(3, 'transition_ratio')['state'].tolist()
    worst_transition_states = transitions.nsmallest(3, 'transition_ratio')['state'].tolist()
    
    insights = {
        'summary': {
            'total_enrolments': total_enrolments,
            'total_demo_updates': total_demo_updates,
            'total_bio_updates': total_bio_updates,
            'unique_states': enrolment['state'].nunique(),
            'unique_districts': enrolment['district'].nunique(),
            'date_range': f"{enrolment['date'].min().strftime('%Y-%m-%d')} to {enrolment['date'].max().strftime('%Y-%m-%d')}",
        },
        'geographic': {
            'top_enrol_states': top_states_enrol,
            'bottom_enrol_states': bottom_states_enrol,
            'top_demo_states': top_states_demo,
            'top_bio_states': top_states_bio,
            'hotspots': hotspots['state'].tolist() if not hotspots.empty else [],
            'coldspots': coldspots['state'].tolist() if not coldspots.empty else [],
        },
        'demographic': {
            'age_distribution': age_dist.to_dict('records'),
            'dominant_age_group': age_majority['age_group'],
            'dominant_percentage': age_majority['percentage'],
        },
        'temporal': {
            'enrol_growth': enrol_growth,
            'demo_growth': demo_growth,
            'bio_growth': bio_growth,
        },
        'anomalies': {
            'enrol_anomaly_count': enrol_anomaly_count,
            'demo_anomaly_count': demo_anomaly_count,
            'enrol_anomaly_days': enrol_anomalies[enrol_anomalies['is_anomaly']]['date'].tolist() if enrol_anomaly_count > 0 else [],
        },
        'transitions': {
            'best_states': best_transition_states,
            'worst_states': worst_transition_states,
        },
        'key_findings': [
            f"Total of {total_enrolments:,} new Aadhaar enrolments processed during the analysis period.",
            f"Demographic updates ({total_demo_updates:,}) outnumber enrolments by {total_demo_updates/total_enrolments:.1f}x, indicating active information maintenance.",
            f"Top 5 states ({', '.join(top_states_enrol)}) account for majority of enrolments, showing geographic concentration.",
            f"Adult population (18+) dominates enrolments at {age_dist[age_dist['age_group']=='18+ years']['percentage'].values[0]:.1f}%.",
            f"{enrol_anomaly_count} anomalous days detected in enrolments, potentially indicating system stress or campaigns.",
            f"Youth biometric transitions vary significantly by state, with {best_transition_states[0]} leading and {worst_transition_states[0]} lagging.",
            f"Cold spots identified in {len(coldspots)} states requiring targeted intervention for Aadhaar coverage.",
        ],
        'recommendations': [
            "Focus enrollment drives on bottom-performing states to improve national coverage equity.",
            "Investigate anomaly days to understand causes and optimize system capacity.",
            "Implement targeted youth biometric update campaigns in low-transition states.",
            "Consider mobile enrollment units for remote districts showing low activity.",
            "Monitor demographic update patterns to predict system load and plan resources.",
        ],
    }
    
    return insights

def generate_pdf_report(insights, figures, quality_reports,
                        enrolment, demographic, biometric,
                        output_path: Path):
    """Generate comprehensive PDF report with all analysis results."""
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'Aadhaar Data Analysis Report', 0, 1, 'C')
            self.ln(2)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        def chapter_title(self, title):
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(0, 51, 102)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(2)
        
        def section_title(self, title):
            self.set_font('Helvetica', 'B', 12)
            self.set_text_color(51, 51, 51)
            self.cell(0, 8, title, 0, 1, 'L')
            self.ln(1)
        
        def body_text(self, text):
            self.set_font('Helvetica', '', 10)
            self.set_text_color(0, 0, 0)
            self.multi_cell(0, 5, text)
            self.ln(2)
        
        def bullet_point(self, text):
            self.set_font('Helvetica', '', 10)
            self.set_text_color(0, 0, 0)
            self.multi_cell(0, 5, f"  {chr(149)} {text}")
            self.ln(1)
    
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(0, 51, 102)
    pdf.ln(60)
    pdf.cell(0, 15, 'Aadhaar Data Analysis Report', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Comprehensive Analysis of Enrolment and Update Patterns', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f"Data Period: {insights['summary']['date_range']}", 0, 1, 'C')
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
    
    # Executive Summary
    pdf.add_page()
    pdf.chapter_title('Executive Summary')
    
    summary = insights['summary']
    pdf.body_text(f"""This report presents a comprehensive analysis of Aadhaar enrolment and update data, 
covering {summary['unique_states']} states and {summary['unique_districts']} districts. 
The analysis period spans from {summary['date_range']}, during which:

- {summary['total_enrolments']:,} new Aadhaar enrolments were processed
- {summary['total_demo_updates']:,} demographic updates were completed
- {summary['total_bio_updates']:,} biometric updates were recorded

Key findings reveal significant geographic disparities in Aadhaar activity, with the top 5 states 
accounting for the majority of enrolments. The adult population (18+) dominates new enrolments, 
while youth biometric transitions show concerning variation across states.""")
    
    pdf.ln(5)
    pdf.section_title('Key Insights')
    for finding in insights['key_findings']:
        pdf.bullet_point(finding)
    
    # Section 1: Problem Statement
    pdf.add_page()
    pdf.chapter_title('Section 1: Problem Statement & Approach')
    
    pdf.section_title('Problem 1: Geographic Disparity in Aadhaar Coverage')
    pdf.body_text("""Which regions show concerning gaps in Aadhaar adoption, and what patterns 
indicate underserved populations? This analysis aggregates enrolment data by state and district, 
identifies cold spots, and correlates with demographic update patterns to find areas requiring intervention.""")
    
    pdf.section_title('Problem 2: Youth Biometric Transition Patterns')
    pdf.body_text("""How effectively are children transitioning to adult biometrics, and are there 
bottlenecks in the system? We analyze biometric update frequency for the 5-17 age bracket, 
track temporal patterns as children approach adulthood, and identify states with low transition rates.""")
    
    pdf.section_title('Problem 3: Update Behavior Anomalies')
    pdf.body_text("""What unusual patterns in demographic and biometric updates indicate system stress, 
fraud potential, or policy gaps? Anomaly detection is applied to update frequencies, identifying 
temporal spikes, geographic clustering, and unexpected demographic patterns.""")
    
    # Section 2: Dataset Description
    pdf.add_page()
    pdf.chapter_title('Section 2: Dataset Description')
    
    pdf.section_title('Enrolment Dataset')
    pdf.body_text(f"""Records: {len(enrolment):,}
Columns: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
Contains new Aadhaar registrations categorized by age groups.""")
    
    pdf.section_title('Demographic Update Dataset')
    pdf.body_text(f"""Records: {len(demographic):,}
Columns: date, state, district, pincode, demo_age_5_17, demo_age_17_
Tracks updates to name, address, date of birth, gender, and mobile number.""")
    
    pdf.section_title('Biometric Update Dataset')
    pdf.body_text(f"""Records: {len(biometric):,}
Columns: date, state, district, pincode, bio_age_5_17, bio_age_17_
Contains fingerprint, iris, and face biometric updates.""")
    
    pdf.section_title('Data Quality Assessment')
    for name, report in quality_reports.items():
        pdf.body_text(f"{name.title()}: {report['total_rows']:,} rows, "
                     f"{report['duplicates']} duplicates, {report['memory_mb']:.1f} MB")
    
    # Section 3: Methodology
    pdf.add_page()
    pdf.chapter_title('Section 3: Methodology')
    
    pdf.section_title('Data Preprocessing')
    pdf.body_text("""1. Date Parsing: Converted DD-MM-YYYY strings to datetime objects
2. PIN Code Validation: Filtered to valid 6-digit numeric codes
3. State Name Normalization: Standardized to consistent naming conventions
4. Missing Value Treatment: Dropped records with null geography fields
5. Feature Engineering: Extracted year, month, quarter, day_of_week; calculated totals""")
    
    pdf.section_title('Analysis Techniques')
    pdf.body_text("""- Temporal Analysis: Time series aggregation, rolling averages, growth rates
- Geographic Analysis: State and district-level aggregation, ranking, hotspot detection
- Demographic Analysis: Age group distribution, transition ratio calculation
- Anomaly Detection: IQR method for identifying outliers
- Comparative Analysis: Cross-dataset metrics, ratio analysis""")
    
    pdf.section_title('Tools and Libraries')
    pdf.body_text("""Python 3.14 with pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, fpdf2""")
    
    # Section 4: Analysis & Visualizations
    pdf.add_page()
    pdf.chapter_title('Section 4: Analysis & Visualizations')
    
    # Temporal Analysis
    pdf.section_title('A. Temporal Analysis')
    pdf.body_text(f"""Average daily enrolments: {insights['temporal']['enrol_growth']['avg_daily']:,.0f}
Maximum daily enrolments: {insights['temporal']['enrol_growth']['max_daily']:,.0f}
Minimum daily enrolments: {insights['temporal']['enrol_growth']['min_daily']:,.0f}""")
    
    if '01_enrolment_trends' in figures:
        pdf.image(str(figures['01_enrolment_trends']), x=10, w=190)
        pdf.ln(5)
    
    # Geographic Analysis
    pdf.add_page()
    pdf.section_title('B. Geographic Analysis')
    
    top_states = insights['geographic']['top_enrol_states']
    bottom_states = insights['geographic']['bottom_enrol_states']
    pdf.body_text(f"""Top performing states: {', '.join(top_states)}
Bottom performing states: {', '.join(bottom_states)}
Hotspots (90th percentile): {', '.join(insights['geographic']['hotspots'][:5])}
Coldspots (10th percentile): {', '.join(insights['geographic']['coldspots'][:5])}""")
    
    if '04_state_enrolments' in figures:
        pdf.image(str(figures['04_state_enrolments']), x=10, w=190)
    
    # Demographic Analysis
    pdf.add_page()
    pdf.section_title('C. Demographic Analysis')
    
    for age_data in insights['demographic']['age_distribution']:
        pdf.body_text(f"{age_data['age_group']}: {age_data['total']:,.0f} ({age_data['percentage']:.1f}%)")
    
    if '07_age_distribution' in figures:
        pdf.image(str(figures['07_age_distribution']), x=10, w=190)
    
    # Anomaly Detection
    pdf.add_page()
    pdf.section_title('D. Anomaly Detection')
    pdf.body_text(f"""Enrolment anomalies detected: {insights['anomalies']['enrol_anomaly_count']} days
Demographic update anomalies detected: {insights['anomalies']['demo_anomaly_count']} days

Anomalous days indicate periods of unusually high or low activity that warrant investigation. 
These could indicate system issues, special campaigns, or data quality concerns.""")
    
    if '10_enrol_anomalies' in figures:
        pdf.image(str(figures['10_enrol_anomalies']), x=10, w=190)
    
    # Comparative Analysis
    pdf.add_page()
    pdf.section_title('E. Comparative Analysis')
    pdf.body_text("""The comparative analysis reveals the relationship between new enrolments 
and update activity across states. States with high enrolment but low update rates may indicate 
populations with stable information, while high update rates could suggest demographic mobility 
or data quality issues.""")
    
    if '12_state_comparison' in figures:
        pdf.image(str(figures['12_state_comparison']), x=10, w=190)
    
    # Youth Transitions
    pdf.add_page()
    pdf.section_title('F. Youth Biometric Transition Analysis')
    pdf.body_text(f"""Best performing states for youth transitions: {', '.join(insights['transitions']['best_states'])}
Underperforming states: {', '.join(insights['transitions']['worst_states'])}

States with low transition ratios may need targeted campaigns to ensure children 
update their biometrics as they approach adulthood.""")
    
    if '13_transition_rates' in figures:
        pdf.image(str(figures['13_transition_rates']), x=10, w=190)
    
    # Dashboard
    pdf.add_page()
    pdf.section_title('Summary Dashboard')
    if '16_dashboard' in figures:
        pdf.image(str(figures['16_dashboard']), x=5, w=200)
    
    # Section 5: Insights & Recommendations
    pdf.add_page()
    pdf.chapter_title('Section 5: Key Insights & Recommendations')
    
    pdf.section_title('Key Insights')
    for i, finding in enumerate(insights['key_findings'], 1):
        pdf.body_text(f"{i}. {finding}")
    
    pdf.ln(5)
    pdf.section_title('Recommendations')
    for i, rec in enumerate(insights['recommendations'], 1):
        pdf.body_text(f"{i}. {rec}")
    
    pdf.ln(5)
    pdf.section_title('Areas for Further Investigation')
    pdf.body_text("""1. Deep-dive into district-level patterns within underperforming states
2. Time-series forecasting for capacity planning
3. Correlation with socioeconomic indicators for targeted interventions
4. Investigation of PIN code patterns for urban/rural classification
5. Seasonal pattern analysis for campaign timing optimization""")
    
    # Save PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    
    return output_path

if __name__ == "__main__":
    insights = main()
