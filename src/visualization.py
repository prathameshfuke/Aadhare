import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Premium Color Palette
COLORS = {
    'primary': '#0F4C81',      # Classic Blue
    'secondary': '#53B0AE',    # Teal
    'tertiary': '#D74E09',     # Burnt Orange
    'success': '#2E8B57',      # Sea Green
    'warning': '#F4A261',      # Sandy Brown
    'danger': '#E76F51',       # Terra Cotta
    'neutral': '#6C757D',      # Gray
    'accent': '#8E44AD',       # Purple
    'background': '#F8F9FA'    # Off-white
}

AGE_COLORS = ['#457B9D', '#E63946', '#A8DADC'] # Muted Blue, Red, Light Blue
STATE_CMAP = 'viridis'

def save_fig(fig: plt.Figure, filename: str, output_dir: Path) -> Path:
    """Save figure to specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return filepath

def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str,
                     title: str, ylabel: str, 
                     rolling_window: Optional[int] = 7) -> plt.Figure:
    """Create time series plot with optional rolling average."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df[date_col], df[value_col], alpha=0.4, linewidth=0.8, 
            color=COLORS['primary'], label='Daily')
    
    if rolling_window:
        rolling = df[value_col].rolling(window=rolling_window, min_periods=1).mean()
        ax.plot(df[date_col], rolling, linewidth=2, 
                color=COLORS['secondary'], label=f'{rolling_window}-day avg')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def plot_state_bar(df: pd.DataFrame, state_col: str, value_col: str,
                   title: str, top_n: int = 15) -> plt.Figure:
    """Create horizontal bar chart for state comparison."""
    plot_df = df.nlargest(top_n, value_col)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_df)))
    bars = ax.barh(plot_df[state_col], plot_df[value_col], color=colors)
    
    ax.set_xlabel(value_col.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    for bar, val in zip(bars, plot_df[value_col]):
        ax.text(val + plot_df[value_col].max() * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    return fig

def plot_age_distribution(age_data: pd.DataFrame, title: str) -> plt.Figure:
    """Create pie chart for age group distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    wedges, texts, autotexts = ax1.pie(
        age_data['total'], 
        labels=age_data['age_group'],
        autopct='%1.1f%%',
        colors=AGE_COLORS,
        explode=[0.02, 0.02, 0.02],
        shadow=True
    )
    ax1.set_title('Distribution by Age Group', fontweight='bold')
    
    bars = ax2.bar(age_data['age_group'], age_data['total'], color=AGE_COLORS)
    ax2.set_ylabel('Total Enrolments')
    ax2.set_title('Absolute Numbers by Age Group', fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig

def plot_monthly_heatmap(df: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    """Create heatmap of monthly patterns."""
    pivot = df.pivot_table(index='month', columns='year', values=value_col, aggfunc='sum')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': value_col.replace('_', ' ').title()})
    
    ax.set_yticklabels([month_names[int(m)-1] for m in pivot.index], rotation=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_title(title, fontweight='bold', pad=20)
    
    fig.tight_layout()
    return fig

def plot_day_of_week(dow_data: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    """Create day of week pattern chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS['primary'] if i < 5 else COLORS['tertiary'] 
              for i in range(len(dow_data))]
    
    bars = ax.bar(dow_data['day_name'], dow_data[value_col], color=colors)
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel(value_col.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold', pad=20)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig

def plot_anomalies(df: pd.DataFrame, date_col: str, value_col: str,
                   title: str) -> plt.Figure:
    """Plot time series highlighting anomalies."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    normal = df[~df['is_anomaly']]
    anomalies = df[df['is_anomaly']]
    
    ax.plot(normal[date_col], normal[value_col], 
            color=COLORS['primary'], alpha=0.6, linewidth=0.8, label='Normal')
    
    ax.scatter(anomalies[date_col], anomalies[value_col], 
               color=COLORS['danger'], s=50, zorder=5, label='Anomaly')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def plot_state_comparison(df: pd.DataFrame, title: str) -> plt.Figure:
    """Create comparative bar chart for enrolment vs updates by state."""
    df_sorted = df.nlargest(15, 'total_activity')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df_sorted))
    width = 0.25
    
    ax.barh(x - width, df_sorted['enrolments'], width, 
            label='Enrolments', color=COLORS['primary'])
    ax.barh(x, df_sorted['demo_updates'], width,
            label='Demographic Updates', color=COLORS['secondary'])
    ax.barh(x + width, df_sorted['bio_updates'], width,
            label='Biometric Updates', color=COLORS['tertiary'])
    
    ax.set_yticks(x)
    ax.set_yticklabels(df_sorted['state'])
    ax.set_xlabel('Count')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    return fig

def plot_transition_rates(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot youth biometric transition rates by state."""
    df_valid = df[df['transition_ratio'].notna() & (df['transition_ratio'] < 10)]
    df_sorted = df_valid.nlargest(20, 'youth_enrolments')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn(df_sorted['transition_ratio'] / df_sorted['transition_ratio'].max())
    
    bars = ax.barh(df_sorted['state'], df_sorted['transition_ratio'], color=colors)
    
    ax.axvline(x=df_sorted['transition_ratio'].median(), color='red', 
               linestyle='--', label=f"Median: {df_sorted['transition_ratio'].median():.2f}")
    
    ax.set_xlabel('Biometric Update to Enrolment Ratio')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    return fig

def plot_cumulative_growth(df: pd.DataFrame, date_col: str, value_col: str,
                           title: str) -> plt.Figure:
    """Plot cumulative growth over time."""
    df = df.sort_values(date_col)
    cumulative = df[value_col].cumsum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(df[date_col], cumulative, alpha=0.3, color=COLORS['primary'])
    ax.plot(df[date_col], cumulative, color=COLORS['primary'], linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Total')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def create_dashboard(enrolment_trends: pd.DataFrame, 
                    state_enrol: pd.DataFrame,
                    age_data: pd.DataFrame,
                    comparative: pd.DataFrame,
                    title: str) -> plt.Figure:
    """Create dashboard-style summary visualization."""
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    fig.patch.set_facecolor(COLORS['background'])
    
    gs = fig.add_gridspec(3, 3)
    
    # 1. Daily Trends (Top Left - Spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(enrolment_trends['date'], enrolment_trends['total'], 
             alpha=0.3, color=COLORS['primary'], label='Daily Raw')
    ax1.plot(enrolment_trends['date'], 
             enrolment_trends['total'].rolling(7).mean(),
             color=COLORS['primary'], linewidth=2.5, label='7-Day Avg')
    ax1.set_title('Daily Enrolment Trends', pad=15)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('white')
    
    # 2. Age Distribution (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    wedges, texts, autotexts = ax2.pie(age_data['total'], labels=age_data['age_group'],
            autopct='%1.1f%%', colors=AGE_COLORS, explode=[0.05]*3, 
            pctdistance=0.85, shadow=True)
    # Draw circle for donut chart
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title('Age Distribution', pad=15)
    
    # 3. Top States (Middle - Full Width)
    ax3 = fig.add_subplot(gs[1, :])
    top_states = state_enrol.nlargest(10, 'total').sort_values('total', ascending=True)
    bars = ax3.barh(top_states['state'], top_states['total'], color=COLORS['secondary'])
    ax3.set_title('Top 10 States by Enrolment', pad=15)
    ax3.grid(True, alpha=0.2, axis='x')
    ax3.set_facecolor('white')
    
    # Add values to bars
    max_val = top_states['total'].max()
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2, 
                 f'{width:,.0f}', 
                 ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 4. Activity Comparison Grouped Bar (Bottom Left - Spans 2 cols)
    ax4 = fig.add_subplot(gs[2, :2])
    top_comp = comparative.nlargest(8, 'total_activity')
    x = np.arange(len(top_comp))
    width = 0.25
    
    ax4.bar(x - width, top_comp['enrolments'], width, label='Enrolments', color=COLORS['primary'])
    ax4.bar(x, top_comp['demo_updates'], width, label='Demo Updates', color=COLORS['secondary'])
    ax4.bar(x + width, top_comp['bio_updates'], width, label='Bio Updates', color=COLORS['tertiary'])
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_comp['state'], rotation=45, ha='right')
    ax4.set_title('Activity Comparison by State (Top 8)', pad=15)
    ax4.legend(frameon=True)
    ax4.grid(True, alpha=0.2, axis='y')
    ax4.set_facecolor('white')
    
    # 5. Overall KPIs (Bottom Right)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    total_enrol = state_enrol['total'].sum()
    total_demo = comparative['demo_updates'].sum()
    total_bio = comparative['bio_updates'].sum()
    
    # Draw Key Metrics as text
    y_pos = [0.8, 0.5, 0.2]
    labels = ['Total Enrolments', 'Demo Updates', 'Biometric Updates']
    values = [total_enrol, total_demo, total_bio]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    
    for y, label, val, col in zip(y_pos, labels, values, colors):
        ax5.text(0.5, y, label, ha='center', va='center', fontsize=14, color=COLORS['neutral'])
        ax5.text(0.5, y-0.15, f'{val:,.0f}', ha='center', va='center', 
                 fontsize=24, fontweight='bold', color=col)
        
    fig.suptitle(title, fontsize=24, fontweight='bold', y=1.02, color=COLORS['primary'])
    
    return fig

def plot_geographic_heatmap(df: pd.DataFrame, state_col: str, value_col: str,
                            title: str) -> plt.Figure:
    """Create state-level heatmap visualization."""
    state_data = df.groupby(state_col)[value_col].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    n_states = len(state_data)
    cols = 6
    rows = (n_states + cols - 1) // cols
    
    heatmap_data = np.full((rows, cols), np.nan)
    state_labels = np.empty((rows, cols), dtype=object)
    state_labels.fill('')
    
    for idx, (state, value) in enumerate(state_data.items()):
        row = idx // cols
        col = idx % cols
        heatmap_data[row, col] = value
        state_labels[row, col] = f"{state[:12]}\n{value/1e6:.2f}M"
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    for i in range(rows):
        for j in range(cols):
            if state_labels[i, j]:
                ax.text(j, i, state_labels[i, j], ha='center', va='center',
                       fontsize=8, color='black')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Total Count')
    
    fig.tight_layout()
    return fig
