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
SRC_DIR = Path(__file__).parent

def get_code_content(filename: str) -> str:
    """Read specific source file content for report."""
    try:
        path = SRC_DIR / filename
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"

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
    
    # Verify cleaning
    print("  Verifying state names...")
    clean_states = sorted(enrolment['state'].unique())
    print(f"  Unique states: {len(clean_states)}")
    print(f"  Sample states: {clean_states[:5]}")
    
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
    
    # Get source code
    source_code = {
        'data_loader': get_code_content('data_loader.py'),
        'preprocessing': get_code_content('preprocessing.py'),
        'analysis': get_code_content('analysis.py'),
        'visualization': get_code_content('visualization.py')
    }
    
    # Generate report
    print("\n[7/7] Generating PDF report...")
    report_path = generate_pdf_report(
        insights, figures, quality_reports,
        enrolment, demographic, biometric,
        source_code,
        OUTPUT_DIR / "report.pdf"
    )
    print(f"  Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    
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
    
    enrol_anomaly_count = enrol_anomalies['is_anomaly'].sum()
    enrol_anomaly_days = enrol_anomalies[enrol_anomalies['is_anomaly']]['date'].astype(str).tolist()
    
    # determine dominant age group correctly
    max_idx = age_dist['percentage'].idxmax()
    dominant_age_group = age_dist.iloc[max_idx]['age_group']
    dominant_percentage = age_dist.iloc[max_idx]['percentage']
    
    best_transition_states = transitions.head(3)['state'].tolist()
    worst_transition_states = transitions.tail(3)['state'].tolist()
    
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
            'hotspots': hotspots['state'].tolist()[:5] if not hotspots.empty else [],
            'coldspots': coldspots['state'].tolist()[:5] if not coldspots.empty else [],
        },
        'demographic': {
            'age_distribution': age_dist.to_dict('records'),
            'dominant_age_group': dominant_age_group,
            'dominant_percentage': dominant_percentage,
        },
        'anomalies': {
            'enrol_anomaly_count': enrol_anomaly_count,
            'enrol_anomaly_days': enrol_anomaly_days,
            'demo_anomaly_count': demo_anomalies['is_anomaly'].sum(),
        },
        'transitions': {
            'best_states': best_transition_states,
            'worst_states': worst_transition_states,
        },
        'key_findings': [
            f"Total of {total_enrolments:,} new Aadhaar enrolments processed.",
            f"Demographic updates ({total_demo_updates:,}) outnumber enrolments by {total_demo_updates/total_enrolments:.1f}x.",
            f"The {dominant_age_group} age group dominates enrolments, accounting for {dominant_percentage:.1f}% of the total.",
            f"{enrol_anomaly_count} anomalous days detected in enrolments (e.g., {', '.join(enrol_anomaly_days[:3])}).",
            f"Top 5 states account for {state_enrol.head(5)['percentage'].sum() if 'percentage' in state_enrol.columns else 'significant'}% of total activity.",
        ],
        'recommendations': [
            "Prioritize data cleaning in state names to resolve inconsistencies.",
            "Investigate specific anomalous dates for system outages or mass enrollment drives.",
            "Target low-transition states for youth biometric update campaigns.",
            "Monitor update vs enrolment ratios to identify potential fraudulent update centers.",
        ]
    }
    
    return insights

def generate_pdf_report(insights, figures, quality_reports,
                        enrolment, demographic, biometric,
                        source_code, output_path: Path):
    """Generate comprehensive PDF report with code snippets."""
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'Aadhaar Data Analysis Report', 0, 1, 'C')
            self.ln(5)
        
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
            
        def code_block(self, code, label="Code"):
            self.ln(2)
            self.set_font('Courier', '', 8)
            self.set_fill_color(245, 245, 245)
            self.set_text_color(0, 0, 0)
            
            # Simple syntax highlighting simulation (just title)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 5, f"Source: {label}", 0, 1, 'L')
            
            self.set_font('Courier', '', 7)
            
            # Split lines and handle pages
            lines = code.split('\n')
            # Limit lines to save space, just show key parts or first 20 lines
            max_lines = 40
            
            content = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                content += "\n... (truncated)"
                
            self.multi_cell(0, 4, content, fill=True, border=1)
            self.ln(5)

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
    pdf.cell(0, 8, f"Prepared: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
    
    # Executive Summary with Corrected Finding
    pdf.add_page()
    pdf.chapter_title('Executive Summary')
    
    dom_group = insights['demographic']['dominant_age_group']
    dom_pct = insights['demographic']['dominant_percentage']
    
    pdf.body_text(f"""This report presents a comprehensive analysis of Aadhaar enrolment and update data, 
processed with corrected cleaning logic to handle inconsistent state names. 
The analysis period covers {insights['summary']['unique_states']} unique states/UTs.""")
    
    pdf.body_text(f"""Key corrected findings:
- Total new enrolments: {insights['summary']['total_enrolments']:,}
- Dominant demographic: {dom_group} ({dom_pct:.1f}%)
- Total anomalous days: {insights['anomalies']['enrol_anomaly_count']}""")

    pdf.ln(5)
    pdf.section_title('Key Insights')
    for finding in insights['key_findings']:
        pdf.bullet_point(finding)
    
    # Section 1: Data Preprocessing & Code
    pdf.add_page()
    pdf.chapter_title('Section 1: Data Preprocessing')
    pdf.body_text("To ensure data quality, we implemented rigorous cleaning steps:")
    pdf.bullet_point("Parsed dates and validated 6-digit PIN codes")
    pdf.bullet_point("Normalized state names (handled variations like 'West Bangal' -> 'West Bengal')")
    pdf.bullet_point("Filtered out invalid rows (e.g., numeric state names like '100000')")
    
    # Inject Preprocessing Code
    pdf.section_title('Preprocessing Code Implementation')
    pdf.code_block(source_code['preprocessing'], label="src/preprocessing.py")
    
    # Section 2: Analysis & Code
    pdf.add_page()
    pdf.chapter_title('Section 2: Analysis Logic')
    pdf.body_text("We performed temporal trends, anomaly detection, and geographic aggregation.")
    
    # Inject Analysis Code
    pdf.section_title('Analysis Code Implementation')
    pdf.code_block(source_code['analysis'], label="src/analysis.py")
    
    # Section 3: Visualizations & Findings
    pdf.add_page()
    pdf.chapter_title('Section 3: Visualizations & Findings')
    
    # Demographic Analysis
    pdf.section_title('A. Demographic Analysis')
    pdf.body_text(f"The {dom_group} group dominates enrolments, contradicting initial assumptions of adult dominance.")
    if '07_age_distribution' in figures:
        pdf.image(str(figures['07_age_distribution']), x=10, w=180)
    
    pdf.ln(5)
    
    # Anomaly Analysis
    pdf.add_page()
    pdf.section_title('B. Anomaly Analysis')
    anomalies = insights['anomalies']['enrol_anomaly_days']
    anom_text = ", ".join(anomalies[:5]) + ("..." if len(anomalies) > 5 else "")
    pdf.body_text(f"Detected {len(anomalies)} anomalous days. Specific dates include: {anom_text}")
    if '10_enrol_anomalies' in figures:
        pdf.image(str(figures['10_enrol_anomalies']), x=10, w=180)
        
    # Geographic Analysis
    pdf.add_page()
    pdf.section_title('C. Geographic Analysis')
    pdf.body_text(f"Top states: {', '.join(insights['geographic']['top_enrol_states'])}")
    if '04_state_enrolments' in figures:
        pdf.image(str(figures['04_state_enrolments']), x=10, w=180)
        
    # Section 4: Visualization Code
    pdf.add_page()
    pdf.chapter_title('Section 4: Visualization Code')
    pdf.code_block(source_code['visualization'], label="src/visualization.py")

    # Save PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    
    return output_path

if __name__ == "__main__":
    main()
