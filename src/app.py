import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust path to include project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_datasets
from src.preprocessing import preprocess_all, get_data_quality_report
from src.analysis import (
    temporal_trends, state_aggregations, age_group_analysis, 
    monthly_patterns, detect_anomalies_iqr, comparative_state_metrics,
    analyze_anomaly_patterns, identify_cross_dataset_outliers, district_deep_dive
)
from src.visualization import (
    plot_time_series, plot_state_bar, plot_age_distribution,
    plot_monthly_heatmap, plot_anomalies, plot_state_comparison,
    create_dashboard, COLORS
)
from src.report_generator import generate_pdf_report
from src.run_analysis import compile_insights, get_code_content

st.set_page_config(
    page_title="Aadhaar Analytics Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #F8F9FA;
    }
    .stButton>button {
        background-color: #0F4C81;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #09345c;
    }
    h1, h2, h3 {
        color: #0F4C81;
        font-family: 'Helvetica', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # --- Sidebar ---
    st.sidebar.image("src/assets/logo.png", use_container_width=True)
    st.sidebar.title("Configuration")
    
    st.sidebar.subheader("Data Upload")
    st.sidebar.info("Upload CSV files or use default data.")
    
    enrol_files = st.sidebar.file_uploader("Enrolment Data", accept_multiple_files=True, type=['csv'])
    demo_files = st.sidebar.file_uploader("Demographic Updates", accept_multiple_files=True, type=['csv'])
    bio_files = st.sidebar.file_uploader("Biometric Updates", accept_multiple_files=True, type=['csv'])
    
    run_btn = st.sidebar.button("Run Analysis", type="primary")

    # Only load default data if no files uploaded and button not clicked (initial load)
    # OR if button clicked.
    # We want to persist the state.
    
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
        
    if run_btn:
        st.session_state.analysis_done = True
        
    # --- Main Content ---
    st.title("🧬 Aadhaar Analytics Pro")
    st.markdown("### Intelligent Data Analysis & Reporting System")
    
    if st.session_state.analysis_done:
        with st.spinner("Crunching numbers... Please wait."):
            try:
                # Load Data
                enrolment_raw, demographic_raw, biometric_raw = load_all_datasets(
                    enrol_files=enrol_files if enrol_files else None,
                    demo_files=demo_files if demo_files else None,
                    bio_files=bio_files if bio_files else None
                )
                
                # Preprocess
                enrolment, demographic, biometric = preprocess_all(enrolment_raw, demographic_raw, biometric_raw)
                
                # --- Tab Layout ---
                tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Deep Insights", "Data Explorer", "Report Generation"])
                
                # Run Core Analysis
                enrol_trends = temporal_trends(enrolment, 'total_enrolments')
                state_enrol = state_aggregations(enrolment, 'total_enrolments')
                age_dist = age_group_analysis(enrolment)
                comparative = comparative_state_metrics(enrolment, demographic, biometric)
                enrol_monthly, enrol_month_avg = monthly_patterns(enrolment, 'total_enrolments')
                enrol_anomalies = detect_anomalies_iqr(enrol_trends, 'total')
                anomaly_patterns = analyze_anomaly_patterns(enrol_anomalies)
                
                # Insights
                insights = compile_insights(
                    enrolment, enrol_trends, state_enrol, age_dist, comparative, 
                    pd.DataFrame(), enrol_anomalies, anomaly_patterns, pd.DataFrame(), 
                    {}, [], enrol_month_avg, demographic, biometric
                )
                
                with tab1:
                    # Metrics Row
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Enrolments", f"{insights['summary']['total_enrolments']:,}")
                    col2.metric("Active States", insights['summary']['unique_states'])
                    col3.metric("Anomalies Detectd", insights['anomalies_deep_dive']['count'])
                    col4.metric("Peak Month", insights['seasonality']['peak_month'])
                    
                    st.divider()
                    
                    # Dashboard Plot
                    st.subheader("Executive Dashboard")
                    fig_dash = create_dashboard(enrol_trends, state_enrol, age_dist, comparative, "Live Dashboard")
                    st.pyplot(fig_dash)
                    
                with tab2:
                    st.header("Deep Dive Analysis")
                    
                    col_l, col_r = st.columns(2)
                    
                    with col_l:
                        st.subheader("Monthly Seasonality")
                        st.pyplot(plot_monthly_heatmap(enrolment, 'total_enrolments', 'Monthly Patterns'))
                        st.caption(insights['seasonality']['msg'])
                        
                    with col_r:
                        st.subheader("Anomaly Detection")
                        st.pyplot(plot_anomalies(enrol_anomalies, 'date', 'total', 'Detected Anomalies'))
                        st.caption(insights['anomalies_deep_dive']['explanation'])
                        
                    st.divider()
                    st.subheader("Key Findings")
                    for finding in insights['key_findings']:
                        st.info(f"💡 {finding}")

                with tab3:
                    st.header("Data Explorer")
                    dataset = st.selectbox("Select Dataset", ["Enrolment", "Demographic", "Biometric"])
                    
                    if dataset == "Enrolment":
                        df_show = enrolment
                    elif dataset == "Demographic":
                        df_show = demographic
                    else:
                        df_show = biometric
                        
                    st.dataframe(df_show.head(1000), use_container_width=True)
                    
                    csv = df_show.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Processed Data", csv, f"{dataset.lower()}_processed.csv", "text/csv")

                with tab4:
                    st.header("Generate Official Report")
                    st.write("Generate a comprehensive PDF report based on the current analysis.")
                    
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating PDF..."):
                            # Helpers for report
                            # We need to regenerate all figs to pass to report generator or save them
                            # For simplicity in this demo, we'll suggest checking the output dir or re-running full pipeline logic
                            # But here we can simulate it by running the main logic if needed, or just generating a simple one.
                            # For correct implementation, we should save the figs currently in memory to a temp dir.
                            
                            temp_dir = Path("outputs/interactive_figures")
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save current figures
                            fig_dash.savefig(temp_dir / "16_dashboard.png", dpi=300, bbox_inches='tight')
                            
                            # Generate others
                            save_fig = lambda f, n: f.savefig(temp_dir / n, dpi=300, bbox_inches='tight')
                            save_fig(plot_monthly_heatmap(enrolment, 'total_enrolments', 'Seasonality'), '08_monthly_heatmap.png')
                            save_fig(plot_anomalies(enrol_anomalies, 'date', 'total', 'Enrolment Anomalies'), '10_enrol_anomalies.png')
                            
                            figures_map = {
                                '16_dashboard': temp_dir / '16_dashboard.png',
                                '08_monthly_heatmap': temp_dir / '08_monthly_heatmap.png',
                                '10_enrol_anomalies': temp_dir / '10_enrol_anomalies.png'
                            }
                            
                            qual_reports = {
                                'enrolment': get_data_quality_report(enrolment_raw, 'Enrolment'),
                                'demographic': get_data_quality_report(demographic_raw, 'Demographic'),
                                'biometric': get_data_quality_report(biometric_raw, 'Biometric'),
                            }
                            
                            src_code = {
                                'analysis': get_code_content('analysis.py'),
                                'preprocessing': get_code_content('preprocessing.py')
                            }
                            
                            report_path = generate_pdf_report(insights, figures_map, qual_reports, src_code, Path("outputs/interactive_report.pdf"))
                            
                            with open(report_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=f,
                                    file_name="Aadhaar_Analysis_Report.pdf",
                                    mime="application/pdf",
                                    type="primary"
                                )
                            st.success("Report generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Please upload data (optional) and click 'Run Analysis' to begin.")

if __name__ == "__main__":
    main()
