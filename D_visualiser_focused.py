"""
Strategy D Focused Visualizer - ESG Supply Chain Emission Reduction
====================================================================
Clean, focused visualization for Strategy D (Zone IV - Observation Zone)
showing only the most critical insights in 2 main panels.

Focus: Low-emission, low-cooperation suppliers
- 5-10% reduction target with minimal investment
- Automated monitoring and cost-effectiveness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import os

# Chinese font configuration
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# File paths
pathway_file = 'ML_simulation_D_三年减排路径.csv'
summary_file = 'ML_simulation_D_strategy_summary.csv'
budget_file = 'ML_simulation_D_投资预算分配.csv'
supplier_file = 'ML_simulation_D_供应商详细信息.csv'

# Check if files exist
if not all(os.path.exists(f) for f in [pathway_file, summary_file, budget_file, supplier_file]):
    print("\n⚠️  Missing data files. Please run D_strategy_ML_simulation.py first!")
    exit(1)

# Load data
print("\n📊 Loading Strategy D data files...")
df_pathway = pd.read_csv(pathway_file)
df_summary = pd.read_csv(summary_file)
df_budget = pd.read_csv(budget_file)
df_supplier = pd.read_csv(supplier_file)

print(f"✓ Loaded {len(df_supplier)} suppliers")

# ============================================================================
# Create Clean 2-Panel Visualization
# ============================================================================

# Create figure with 1 row, 2 columns
fig = plt.figure(figsize=(24, 10))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25, top=0.82, bottom=0.08)

# Color scheme
zone_color = '#95A5A6'  # Gray for Zone IV
reduction_color = '#3498DB'  # Blue
investment_color = '#E67E22'  # Orange
success_color = '#27AE60'  # Green

# ============================================================================
# LEFT PANEL: Three-Year Emission Reduction Pathway
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Select top 30 suppliers by baseline emission for clearer visualization
top_suppliers = df_summary.nlargest(30, '基线排放')['供应商ID'].tolist()

# Plot individual pathways (lighter, in background)
for supplier_id in top_suppliers:
    supplier_data = df_pathway[df_pathway['供应商ID'] == supplier_id]
    if len(supplier_data) > 0:
        years = [0, 1, 2, 3]
        emissions = supplier_data['碳排放量'].tolist()
        ax1.plot(years, emissions, color=zone_color, alpha=0.2, linewidth=1.5)

# Calculate and plot average pathway (bold, highlighted)
avg_by_year = df_pathway.groupby('年份')['碳排放量'].mean()
years_avg = [0, 1, 2, 3]
avg_emissions = [avg_by_year[f'Y{i}'] for i in range(4)]

# Plot average with fill
ax1.plot(years_avg, avg_emissions, marker='o', linewidth=5, color=reduction_color, 
         label='平均减排路径 (Average Pathway)', linestyle='-', markersize=12, 
         markeredgecolor='white', markeredgewidth=2, zorder=10)

# Add shaded area to show reduction
ax1.fill_between(years_avg, avg_emissions, avg_emissions[0], 
                 alpha=0.3, color=success_color, label='减排区域 (Reduction Area)')

# Annotate key points with adjusted positions to avoid overlap
offsets = [(0, 20), (0, 25), (0, 20), (0, -30)]  # Variable offsets to prevent overlap
for i, (year, emission) in enumerate(zip(years_avg, avg_emissions)):
    reduction_pct = ((avg_emissions[0] - emission) / avg_emissions[0]) * 100
    if i == 0:
        label = f'{emission:.0f}\n基线'
    else:
        label = f'{emission:.0f}\n-{reduction_pct:.1f}%'
    
    ax1.annotate(label, xy=(year, emission), xytext=offsets[i], 
                textcoords='offset points', ha='center', fontsize=10,
                weight='bold', color=reduction_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=reduction_color, linewidth=1.5))

ax1.set_xlabel('年份 (Year)', fontsize=16, weight='bold')
ax1.set_ylabel('碳排放量 (吨 CO₂e)', fontsize=16, weight='bold')
ax1.set_title('IV区供应商三年减排路径\nZone IV Three-Year Emission Reduction Pathway', 
             fontsize=18, weight='bold', pad=20)
ax1.set_ylim(4100, 4600)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['基线\nBaseline', '第1年\nYear 1', '第2年\nYear 2', '第3年\nYear 3'], 
                    fontsize=12)
ax1.legend(fontsize=13, loc='upper right', framealpha=0.95)

# Add summary stats box
total_baseline = df_summary['基线排放'].sum()
total_reduction = df_summary['最终减排量'].sum()
avg_reduction_pct = df_summary['减排率'].mean()
success_rate = (df_budget['目标达成'] == '是').sum() / len(df_budget) * 100

stats_text = f"""减排总览 (Reduction Overview)
━━━━━━━━━━━━━━━━━━
基线总排放: {total_baseline:,.0f} 吨
总减排量: {total_reduction:,.0f} 吨
平均减排率: {avg_reduction_pct:.1f}%
目标达成率: {success_rate:.0f}%"""

ax1.text(0.03, 0.58, stats_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', 
                 edgecolor='black', linewidth=1.5, alpha=0.9), zorder=20)

# ============================================================================
# RIGHT PANEL: Investment vs ROI Analysis (Scatter plot)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Calculate cost per ton (inverse of efficiency)
df_summary['成本每吨'] = df_summary['总投资'] / df_summary['最终减排量']

# Scatter plot - Investment vs Reduction
scatter_size = df_summary['效率(吨/美元)'] * 50000  # Scale for visibility (reduced to 50%)
scatter = ax2.scatter(df_summary['总投资'], df_summary['最终减排量'], 
                     s=scatter_size, alpha=0.7, 
                     c=df_summary['成本每吨'], cmap='RdYlGn_r',
                     edgecolors='black', linewidth=1.5, zorder=5,
                     vmin=df_summary['成本每吨'].min(), 
                     vmax=df_summary['成本每吨'].max())


# Add trend line
z = np.polyfit(df_summary['总投资'], df_summary['最终减排量'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_summary['总投资'].min(), df_summary['总投资'].max(), 100)
ax2.plot(x_trend, p(x_trend), "r--", linewidth=3, alpha=0.8, 
        label='投资回报趋势 (ROI Trend)', zorder=3)

ax2.set_xlabel('总投资 (USD)', fontsize=16, weight='bold')
ax2.set_ylabel('总减排量 (吨 CO₂e)', fontsize=16, weight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=1)

# Title
ax2.set_title('投资效率分析\nInvestment Efficiency Analysis', 
             fontsize=18, weight='bold', pad=20)

# Colorbar for scatter points
cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
cbar.set_label('减排成本 (USD/吨CO₂) | Cost (USD/tCO₂)', fontsize=12, weight='bold')
cbar.ax.tick_params(labelsize=10)

# Legend for scatter
ax2.legend(fontsize=12, loc='upper left', framealpha=0.95)

# Add key insights box
total_investment = df_summary['总投资'].sum()
avg_cost_per_ton = df_summary['成本每吨'].mean()
avg_payback = df_budget['投资回收期'].mean()
avg_automation = df_summary['自动化水平'].mean()

insights_text = f"""关键指标 (Key Metrics)
━━━━━━━━━━━━━━━━━━
总投资: ${total_investment:,.0f}
平均成本: {avg_cost_per_ton:.1f} USD/tCO₂
平均回收期: {avg_payback:.1f} 年
自动化水平: {avg_automation:.2f}

Zone IV特点:
• 最低投资需求
• 依赖自动化监测
• 稳定但适度的减排
• 高成本效益比"""

ax2.text(0.03, 0.55, insights_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                 edgecolor='black', linewidth=1.5, alpha=0.95), zorder=20)

# ============================================================================
# Main Title
# ============================================================================
fig.suptitle('IV区观察区 (低排放×低配合)\nZone IV Observation Zone (Low Emission × Low Cooperation)',
            fontsize=20, weight='bold', y=0.97)

# Add footer with methodology
footer_text = '基于ML模拟的100个供应商样本 | 目标: 5-10%减排 | 方法: 基础管理+自动监测+低成本技术'
fig.text(0.5, 0.005, footer_text, ha='center', fontsize=11, style='italic', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

# ============================================================================
# Save and Display
# ============================================================================
plt.tight_layout(rect=[0, 0.04, 1, 0.86])

output_path = 'D区减排核心分析_ML_data.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Visualization saved: {output_path}")

print("\n" + "="*80)
print("STRATEGY D FOCUSED VISUALIZATION COMPLETE")
print("="*80)
print(f"\n📊 Generated clean 2-panel analysis")
print(f"📁 Output file: {output_path}")
print("\n💡 Key Highlights:")
print(f"  • Average reduction: {avg_reduction_pct:.1f}% over 3 years")
print(f"  • Total investment: ${total_investment:,.0f}")
print(f"  • Success rate: {success_rate:.0f}%")
print(f"  • Average cost: {avg_cost_per_ton:.1f} USD per ton CO₂e")
print(f"  • Strategy: Minimal investment + Automated monitoring")
print("\n" + "="*80)

plt.show()
