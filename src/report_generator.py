from fpdf import FPDF
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class AadhaarReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.logo_path = Path(__file__).parent / "assets" / "logo.png"

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            if self.logo_path.exists():
                self.image(str(self.logo_path), x=10, y=8, w=15)
                self.set_xy(28, 12)
            else:
                self.set_xy(10, 12)
            self.cell(0, 0, 'Aadhaar Data Analysis Report - Restricted Circulation', 0, 0, 'L')
            self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_title_page(self, title: str, subtitle: str):
        self.add_page()
        
        # Logo
        if self.logo_path.exists():
            self.image(str(self.logo_path), x=65, y=30, w=80)
        
        # Title
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(15, 76, 129)  # Classic Blue
        self.set_y(120)
        self.cell(0, 10, title, 0, 1, 'C')
        
        # Subtitle
        self.set_font('Helvetica', '', 14)
        self.set_text_color(80, 80, 80)
        self.ln(5)
        self.cell(0, 10, subtitle, 0, 1, 'C')
        
        # Date and Info
        self.set_y(-50)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Generated on: {datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
        self.cell(0, 5, "CONFIDENTIAL", 0, 1, 'C')

    def chapter_title(self, title: str):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(15, 76, 129)  # Classic Blue
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
        # Underline
        self.set_draw_color(15, 76, 129)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def section_title(self, title: str):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(40, 40, 40)
        self.ln(5)
        self.cell(0, 8, title, 0, 1, 'L')

    def body_text(self, text: str):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_point(self, text: str):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(20, 20, 20)
        self.cell(5)
        self.cell(5, 6, chr(149), 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def add_plot(self, image_path: Path, width: int = 180, caption: str = None):
        if image_path.exists():
            self.ln(5)
            self.image(str(image_path), x=(210-width)/2, w=width)
            if caption:
                self.set_font('Helvetica', 'I', 9)
                self.set_text_color(100, 100, 100)
                self.cell(0, 8, caption, 0, 1, 'C')
            self.ln(5)

    def add_code_block(self, code: str, label: str = "Code"):
        self.ln(5)
        self.set_font('Courier', 'B', 9)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f" {label}", 0, 1, 'L', fill=True)
        
        self.set_font('Courier', '', 8)
        lines = code.split('\n')
        
        # Simple pagination for code
        max_lines_per_page = 45
        for i in range(0, len(lines), max_lines_per_page):
            chunk = lines[i:i+max_lines_per_page]
            text = "\n".join(chunk)
            self.multi_cell(0, 4, text, border=1, fill=True)
            if i + max_lines_per_page < len(lines):
                 self.add_page()

def generate_pdf_report(insights: Dict[str, Any], figures: Dict[str, Path], 
                        quality_reports: Dict[str, str], source_code: Dict[str, str], 
                        output_path: Path) -> Path:
    """Generate the comprehensive PDF report."""
    
    pdf = AadhaarReport()
    
    # 1. Title Page
    pdf.add_title_page("Aadhaar Data Analysis", "Comprehensive Insights & Recommendations")
    
    # 2. Executive Summary
    pdf.add_page()
    pdf.chapter_title('Executive Summary')
    
    pdf.section_title("The Problem")
    pdf.body_text(f"Analysis of current enrollment data reveals significant geographic disparities. 15% of states account for the majority of enrolments, while bottom states like {', '.join(insights['bottom_states_gap']['states'][:3])} remain critically underserved, accounting for only {insights['bottom_states_gap']['percentage']:.2f}% of the total.")
    
    pdf.section_title("The Insight")
    pdf.body_text(f"{insights['seasonality']['msg']} Furthermore, anomaly detection indicates systematic patterns: {insights['anomalies_deep_dive']['explanation']}")
    
    pdf.section_title("Strategic Recommendations")
    pdf.bullet_point("Deploy mobile units to the 38 identified priority districts immediately.")
    pdf.bullet_point(f"Optimize campaign timing to coincide with the {insights['seasonality']['peak_month']} surge.")
    pdf.bullet_point("Launch targeted youth biometric update campaigns in low-transition states.")
    
    # 3. Key Findings
    pdf.ln(5)
    pdf.section_title('Key Quantitative Findings')
    for finding in insights['key_findings']:
        pdf.bullet_point(finding)
        
    # 4. Deep Dives
    pdf.add_page()
    pdf.chapter_title('Deep Dive Analysis')
    
    pdf.section_title('1. Geographic & Demographic Trends')
    pdf.body_text("The following dashboard provides a holistic view of current enrollment status, age distribution, and state-wise performance.")
    if '16_dashboard' in figures:
        pdf.add_plot(figures['16_dashboard'], width=190, caption="Figure 1: Comprehensive Activity Dashboard")
        
    pdf.add_page()
    pdf.section_title('2. Seasonality & Temporal Patterns')
    pdf.body_text("Understanding monthly enrollment patterns is crucial for resource allocation. Our heatmap analysis reveals clear seasonal trends.")
    if '08_monthly_heatmap' in figures:
        pdf.add_plot(figures['08_monthly_heatmap'], caption="Figure 2: Month-over-Year Enrollment Heatmap")
        
    pdf.section_title('3. Anomaly Detection')
    pdf.body_text(insights['anomalies_deep_dive']['explanation'])
    if '10_enrol_anomalies' in figures:
        pdf.add_plot(figures['10_enrol_anomalies'], caption="Figure 3: Time-series Anomaly Detection")

    # 5. Technical Stack
    pdf.add_page()
    pdf.chapter_title('Technical Implementation')
    pdf.body_text("This analysis was generated using a custom Python pipeline. Below are the key modules used for processing and analysis.")
    
    pdf.section_title("Analysis Logic (src/analysis.py)")
    pdf.add_code_block(source_code['analysis'], "src/analysis.py")
    
    pdf.add_page()
    pdf.section_title("Preprocessing Logic (src/preprocessing.py)")
    pdf.add_code_block(source_code['preprocessing'], "src/preprocessing.py")
    
    # Save
    pdf.output(str(output_path))
    return output_path
