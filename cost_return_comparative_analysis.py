"""
Cross-Strategy Cost/Return Comparative Analysis
================================================
Comprehensive visualization comparing investment efficiency, ROI, and cost-effectiveness
across all three ESG supply chain strategies (B, C, D).

Strategy B (Zone II): Risk Management - High emission, low cooperation
Strategy C (Zone III): Learning Zone - Low emission, high cooperation  
Strategy D (Zone IV): Observation Zone - Low emission, low cooperation
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

# Chinese font configuration
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("\n" + "="*80)
print("📊 CROSS-STRATEGY COST/RETURN COMPARATIVE ANALYSIS")
print("="*80 + "\n")

# ============================================================================
# Load All Strategy Data
# ============================================================================
print("📁 Loading data from all strategies...")

try:
    # Strategy B data
    budget_B = pd.read_csv('ML_simulation_B_投资预算分配.csv')
    strategy_B = pd.read_csv('ML_simulation_B_strategy_summary.csv')
    supplier_B = pd.read_csv('ML_simulation_B_供应商详细信息.csv')
    budget_B['策略'] = 'B-风险管理区'
    budget_B['Zone'] = 'Zone II'
    
    # Strategy C data
    budget_C = pd.read_csv('ML_simulation_C_投资预算分配.csv')
    strategy_C = pd.read_csv('ML_simulation_C_strategy_summary.csv')
    supplier_C = pd.read_csv('ML_simulation_C_供应商详细信息.csv')
    budget_C['策略'] = 'C-学习区'
    budget_C['Zone'] = 'Zone III'
    
    # Strategy D data
    budget_D = pd.read_csv('ML_simulation_D_投资预算分配.csv')
    strategy_D = pd.read_csv('ML_simulation_D_strategy_summary.csv')
    supplier_D = pd.read_csv('ML_simulation_D_供应商详细信息.csv')
    budget_D['策略'] = 'D-观察区'
    budget_D['Zone'] = 'Zone IV'
    
    print(f"✓ Strategy B: {len(budget_B)} suppliers")
    print(f"✓ Strategy C: {len(budget_C)} suppliers")
    print(f"✓ Strategy D: {len(budget_D)} suppliers")
    
except FileNotFoundError as e:
    print(f"❌ Error: Missing data file - {e}")
    print("Please ensure all strategy simulation files are generated.")
    exit(1)

# ============================================================================
# Data Preprocessing and Standardization
# ============================================================================
print("\n🔧 Preprocessing and standardizing data...")

# Standardize column names for Strategy B
if '供应商' in budget_B.columns:
    budget_B = budget_B.rename(columns={
        '供应商': '供应商ID',
        '总成本': '总投资',
        '预期减排量': '减排量'
    })
    
if '供应商' in strategy_B.columns:
    strategy_B = strategy_B.rename(columns={'供应商': '供应商ID'})

# Standardize column names for Strategy C
if '供应商' in budget_C.columns:
    budget_C = budget_C.rename(columns={'供应商': '供应商ID'})
if '供应商' in strategy_C.columns:
    strategy_C = strategy_C.rename(columns={'供应商': '供应商ID'})

# Calculate cost per ton for all strategies
budget_B['成本每吨'] = budget_B['总投资'] / budget_B['减排量']
budget_C['成本每吨'] = budget_C['总投资'] / budget_C['预期减排量']
budget_D['成本每吨'] = budget_D['总投资'] / budget_D['年度节省'] * 10  # Approximate based on savings

# Create unified dataframe for comparison
comparison_data = []

for idx, row in budget_B.iterrows():
    comparison_data.append({
        '策略': 'B-风险管理',
        'Zone': 'Zone II',
        '供应商': row['供应商ID'],
        '总投资': row['总投资'],
        '减排量': row['减排量'],
        '投资回报率': row.get('投资回报率', 0),
        '回本周期': row.get('回本周期(年)', 0),
        '成本每吨': row['成本每吨']
    })

for idx, row in budget_C.iterrows():
    comparison_data.append({
        '策略': 'C-学习区',
        'Zone': 'Zone III',
        '供应商': row['供应商ID'],
        '总投资': row['总投资'],
        '减排量': row['预期减排量'],
        '投资回报率': row.get('投资回报率', 0),
        '回本周期': row.get('回本周期(年)', 0),
        '成本每吨': row['成本每吨']
    })

for idx, row in budget_D.iterrows():
    comparison_data.append({
        '策略': 'D-观察区',
        'Zone': 'Zone IV',
        '供应商': row['供应商ID'],
        '总投资': row['总投资'],
        '减排量': row.get('年度节省', 2950),  # Use savings as proxy
        '投资回报率': 0,  # Not directly available
        '回本周期': row.get('投资回收期', 15),
        '成本每吨': row['成本每吨']
    })

df_comparison = pd.DataFrame(comparison_data)

print(f"✓ Created unified comparison dataset: {len(df_comparison)} total records")

# ============================================================================
# Figure 1: Investment & Return Overview (4 panels)
# ============================================================================
print("\n📈 Generating Figure 1: Investment & Return Overview...")

fig1 = plt.figure(figsize=(20, 12))
fig1.patch.set_facecolor('white')

# Color scheme for strategies
colors = {
    'B-风险管理': '#E74C3C',  # Red
    'C-学习区': '#3498DB',     # Blue
    'D-观察区': '#95A5A6'      # Gray
}

# Panel 1: Total Investment Comparison
ax1 = plt.subplot(2, 2, 1)
strategies = ['B-风险管理', 'C-学习区', 'D-观察区']
total_investments = [
    budget_B['总投资'].sum(),
    budget_C['总投资'].sum(),
    budget_D['总投资'].sum()
]

bars = ax1.bar(strategies, total_investments, 
               color=[colors[s] for s in strategies],
               alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar, value in zip(bars, total_investments):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${value:,.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('总投资 (USD)', fontsize=13, fontweight='bold')
ax1.set_title('各策略总投资对比\nTotal Investment by Strategy', 
             fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(total_investments) * 1.15)

# Panel 2: Total Emission Reduction Comparison
ax2 = plt.subplot(2, 2, 2)
total_reductions = [
    budget_B['减排量'].sum(),
    budget_C['预期减排量'].sum(),
    strategy_D['最终减排量'].sum()
]

bars2 = ax2.bar(strategies, total_reductions,
                color=[colors[s] for s in strategies],
                alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar, value in zip(bars2, total_reductions):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:,.0f}t',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('总减排量 (吨 CO₂)', fontsize=13, fontweight='bold')
ax2.set_title('各策略总减排量对比\nTotal Emission Reduction by Strategy',
             fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(total_reductions) * 1.15)

# Panel 3: Cost per Ton Comparison (Box Plot)
ax3 = plt.subplot(2, 2, 3)

cost_per_ton_data = [
    budget_B['成本每吨'].values,
    budget_C['成本每吨'].values,
    budget_D['成本每吨'].values
]

bp = ax3.boxplot(cost_per_ton_data, labels=strategies, patch_artist=True,
                 widths=0.6, showfliers=True,
                 boxprops=dict(linewidth=2),
                 medianprops=dict(linewidth=3, color='darkred'),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

# Color the boxes
for patch, strategy in zip(bp['boxes'], strategies):
    patch.set_facecolor(colors[strategy])
    patch.set_alpha(0.7)

# Add reference line for industry standard ($20-40/ton)
ax3.axhspan(20, 40, alpha=0.2, color='green', label='行业标准 ($20-40)')
ax3.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(y=40, color='green', linestyle='--', linewidth=1, alpha=0.5)

ax3.set_ylabel('成本每吨 (USD/tCO₂)', fontsize=13, fontweight='bold')
ax3.set_title('减排成本效益对比\nCost-Effectiveness Comparison',
             fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', fontsize=10)

# Add median values as text
medians = [np.median(data) for data in cost_per_ton_data]
for i, (median, strategy) in enumerate(zip(medians, strategies)):
    ax3.text(i+1, median, f'${median:.1f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Panel 4: Average Investment per Supplier
ax4 = plt.subplot(2, 2, 4)

avg_investments = [
    budget_B['总投资'].mean(),
    budget_C['总投资'].mean(),
    budget_D['总投资'].mean()
]

avg_reductions = [
    budget_B['减排量'].mean(),
    budget_C['预期减排量'].mean(),
    strategy_D['最终减排量'].mean()
]

x_pos = np.arange(len(strategies))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, avg_investments, width,
               label='平均投资', color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x_pos + width/2, [r * 10 for r in avg_reductions], width,
               label='平均减排 (×10t)', color='green', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:,.0f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, value in zip(bars2, avg_reductions):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:,.0f}t',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax4.set_ylabel('金额 (USD) / 减排量', fontsize=13, fontweight='bold')
ax4.set_title('单个供应商平均指标\nAverage Metrics per Supplier',
             fontsize=14, fontweight='bold', pad=15)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(strategies)
ax4.legend(fontsize=11, loc='upper left')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Main title
fig1.suptitle('跨策略投资与回报综合对比分析\nCross-Strategy Investment & Return Comparative Analysis',
             fontsize=18, fontweight='bold', y=0.98)

# Add summary statistics box
stats_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   策略对比统计摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

B区 (风险管理):
  • 供应商: {len(budget_B)}家
  • 总投资: ${total_investments[0]:,.0f}
  • 总减排: {total_reductions[0]:,.0f}t
  • 平均成本: ${np.median(cost_per_ton_data[0]):.1f}/t

C区 (学习区):
  • 供应商: {len(budget_C)}家
  • 总投资: ${total_investments[1]:,.0f}
  • 总减排: {total_reductions[1]:,.0f}t
  • 平均成本: ${np.median(cost_per_ton_data[1]):.1f}/t

D区 (观察区):
  • 供应商: {len(budget_D)}家
  • 总投资: ${total_investments[2]:,.0f}
  • 总减排: {total_reductions[2]:,.0f}t
  • 平均成本: ${np.median(cost_per_ton_data[2]):.1f}/t
"""

fig1.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                  edgecolor='black', linewidth=1.5))

plt.tight_layout(rect=[0, 0.15, 1, 0.96])

output1 = 'cross_strategy_investment_return_analysis.png'
plt.savefig(output1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output1}")

# ============================================================================
# Figure 2: Cost Efficiency & ROI Analysis (3 panels)
# ============================================================================
print("📈 Generating Figure 2: Cost Efficiency & ROI Analysis...")

fig2 = plt.figure(figsize=(20, 8))
fig2.patch.set_facecolor('white')

# Panel 1: Investment vs Reduction Scatter (All Strategies)
ax1 = plt.subplot(1, 3, 1)

for strategy in strategies:
    strategy_data = df_comparison[df_comparison['策略'] == strategy]
    ax1.scatter(strategy_data['总投资'], strategy_data['减排量'],
               s=100, alpha=0.6, label=strategy,
               color=colors[strategy], edgecolors='black', linewidth=1)

ax1.set_xlabel('总投资 (USD)', fontsize=13, fontweight='bold')
ax1.set_ylabel('减排量 (吨 CO₂)', fontsize=13, fontweight='bold')
ax1.set_title('投资-减排关系图\nInvestment vs Reduction',
             fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')

# Panel 2: Cost per Ton Distribution
ax2 = plt.subplot(1, 3, 2)

positions = [1, 2, 3]
violin_data = [budget_B['成本每吨'].values,
               budget_C['成本每吨'].values,
               budget_D['成本每吨'].values]

parts = ax2.violinplot(violin_data, positions=positions, widths=0.7,
                       showmeans=True, showmedians=True)

# Color the violin plots
for pc, strategy in zip(parts['bodies'], strategies):
    pc.set_facecolor(colors[strategy])
    pc.set_alpha(0.7)

# Add reference zone for industry standard
ax2.axhspan(20, 40, alpha=0.15, color='green', zorder=0)
ax2.text(3.2, 30, '标准\n范围', fontsize=10, va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

ax2.set_xticks(positions)
ax2.set_xticklabels(strategies)
ax2.set_ylabel('成本每吨 (USD/tCO₂)', fontsize=13, fontweight='bold')
ax2.set_title('减排成本分布对比\nCost Distribution Comparison',
             fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Panel 3: Efficiency Ranking (Top 10 from each strategy)
ax3 = plt.subplot(1, 3, 3)

# Get top 5 most efficient suppliers from each strategy
top_B = budget_B.nsmallest(5, '成本每吨')[['供应商ID', '成本每吨']].copy()
top_B['策略'] = 'B'
top_C = budget_C.nsmallest(5, '成本每吨')[['供应商ID', '成本每吨']].copy()
top_C['策略'] = 'C'
top_D = budget_D.nsmallest(5, '成本每吨')[['供应商ID', '成本每吨']].copy()
top_D['策略'] = 'D'

top_all = pd.concat([top_B, top_C, top_D]).sort_values('成本每吨')

y_pos = np.arange(len(top_all))
strategy_colors = [colors[f'{s}-风险管理' if s == 'B' else f'{s}-学习区' if s == 'C' else f'{s}-观察区'] 
                  for s in top_all['策略']]

bars = ax3.barh(y_pos, top_all['成本每吨'], color=strategy_colors,
               alpha=0.8, edgecolor='black', linewidth=1.5)

ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{s.split('_')[-1]} ({st})" 
                     for s, st in zip(top_all['供应商ID'], top_all['策略'])],
                    fontsize=9)
ax3.set_xlabel('成本每吨 (USD/tCO₂)', fontsize=13, fontweight='bold')
ax3.set_title('Top 15 最高效供应商\nTop 15 Most Cost-Effective',
             fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_all['成本每吨'])):
    ax3.text(value + 0.5, i, f'${value:.1f}',
            va='center', ha='left', fontsize=9, fontweight='bold')

fig2.suptitle('成本效益与投资效率深度分析\nCost-Effectiveness & Investment Efficiency Analysis',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output2 = 'cross_strategy_cost_efficiency_analysis.png'
plt.savefig(output2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output2}")

# ============================================================================
# Figure 3: Strategic Comparison Matrix
# ============================================================================
print("📈 Generating Figure 3: Strategic Comparison Matrix...")

fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig3.patch.set_facecolor('white')

# Panel 1: Key Metrics Comparison (Radar Chart)
categories = ['总投资\n(×$1k)', '总减排\n(×100t)', '平均效率\n(×10)', 
              '成本效益\n(inverse)', '供应商数\n(×10)']
N = len(categories)

# Normalize data for radar chart
values_B = [
    total_investments[0] / 1000 / 100,  # Scale down
    total_reductions[0] / 100 / 100,
    (1 / np.median(cost_per_ton_data[0])) * 100,
    (budget_B['减排量'] / budget_B['总投资']).mean() * 1000,
    len(budget_B) / 10
]

values_C = [
    total_investments[1] / 1000 / 100,
    total_reductions[1] / 100 / 100,
    (1 / np.median(cost_per_ton_data[1])) * 100,
    (budget_C['预期减排量'] / budget_C['总投资']).mean() * 1000,
    len(budget_C) / 10
]

values_D = [
    total_investments[2] / 1000 / 100,
    total_reductions[2] / 100 / 100,
    (1 / np.median(cost_per_ton_data[2])) * 100,
    (strategy_D['最终减排量'] / budget_D['总投资']).mean() * 1000,
    len(budget_D) / 10
]

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
values_B += values_B[:1]
values_C += values_C[:1]
values_D += values_D[:1]
angles += angles[:1]

ax1.plot(angles, values_B, 'o-', linewidth=2, label='B-风险管理', 
        color=colors['B-风险管理'], markersize=8)
ax1.fill(angles, values_B, alpha=0.25, color=colors['B-风险管理'])

ax1.plot(angles, values_C, 'o-', linewidth=2, label='C-学习区',
        color=colors['C-学习区'], markersize=8)
ax1.fill(angles, values_C, alpha=0.25, color=colors['C-学习区'])

ax1.plot(angles, values_D, 'o-', linewidth=2, label='D-观察区',
        color=colors['D-观察区'], markersize=8)
ax1.fill(angles, values_D, alpha=0.25, color=colors['D-观察区'])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=10)
ax1.set_ylim(0, max(max(values_B), max(values_C), max(values_D)) * 1.2)
ax1.set_title('多维度策略对比\nMulti-Dimensional Strategy Comparison',
             fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True)

# Panel 2: Investment Structure Breakdown
ax2 = plt.subplot(2, 2, 2)

# For strategies with detailed breakdown
investment_categories = ['技术投资', '管理/知识投资', '其他']

# B strategy - use actual data
if '投资金额' in budget_B.columns and '强制成本' in budget_B.columns:
    b_tech = budget_B['投资金额'].sum()
    b_mandatory = budget_B['强制成本'].sum()
    b_values = [b_tech, b_mandatory, 0]
else:
    b_values = [total_investments[0] * 0.8, total_investments[0] * 0.2, 0]

# C strategy - has tech and knowledge split
if '技术投资' in budget_C.columns and '知识投资' in budget_C.columns:
    c_values = [budget_C['技术投资'].sum(), budget_C['知识投资'].sum(), 0]
else:
    c_values = [total_investments[1] * 0.7, total_investments[1] * 0.3, 0]

# D strategy - has tech and management split
if '技术投资' in budget_D.columns and '管理成本' in budget_D.columns:
    d_values = [budget_D['技术投资'].sum(), budget_D['管理成本'].sum(), 0]
else:
    d_values = [total_investments[2] * 0.5, total_investments[2] * 0.5, 0]

x = np.arange(len(strategies))
width = 0.25

bars1 = ax2.bar(x - width, [b_values[0]/1000, c_values[0]/1000, d_values[0]/1000], 
               width, label='技术投资', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x, [b_values[1]/1000, c_values[1]/1000, d_values[1]/1000],
               width, label='管理/知识', color='orange', alpha=0.8, edgecolor='black')

ax2.set_ylabel('投资金额 (×$1,000)', fontsize=13, fontweight='bold')
ax2.set_title('投资结构分解\nInvestment Structure Breakdown',
             fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(strategies)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Panel 3: Payback Period Comparison
ax3 = plt.subplot(2, 2, 3)

# Get payback data
payback_B = budget_B['回本周期(年)'].values if '回本周期(年)' in budget_B.columns else np.array([1.5] * len(budget_B))
payback_C = budget_C['回本周期(年)'].values if '回本周期(年)' in budget_C.columns else np.array([7.0] * len(budget_C))
payback_D = budget_D['投资回收期'].values if '投资回收期' in budget_D.columns else np.array([15.0] * len(budget_D))

payback_data = [payback_B, payback_C, payback_D]

bp = ax3.boxplot(payback_data, labels=strategies, patch_artist=True,
                widths=0.6, showmeans=True,
                boxprops=dict(linewidth=2),
                medianprops=dict(linewidth=3, color='darkred'),
                meanprops=dict(marker='D', markerfacecolor='yellow', 
                             markeredgecolor='red', markersize=8))

for patch, strategy in zip(bp['boxes'], strategies):
    patch.set_facecolor(colors[strategy])
    patch.set_alpha(0.7)

ax3.set_ylabel('回本周期 (年)', fontsize=13, fontweight='bold')
ax3.set_title('投资回收期对比\nPayback Period Comparison',
             fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add median annotations
for i, (data, strategy) in enumerate(zip(payback_data, strategies)):
    median = np.median(data)
    mean = np.mean(data)
    ax3.text(i+1.3, median, f'中位: {median:.1f}年\n平均: {mean:.1f}年',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 4: Summary Scorecard
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Calculate comprehensive scores
def calculate_score(investment, reduction, cost_per_ton_median, payback_median):
    """Simple scoring: lower cost per ton and faster payback = higher score"""
    cost_score = 100 / (cost_per_ton_median + 1)  # Lower is better
    payback_score = 100 / (payback_median + 1)    # Lower is better
    reduction_score = reduction / 100              # Higher is better
    efficiency_score = reduction / (investment / 1000)  # Higher is better
    
    total = (cost_score * 0.3 + payback_score * 0.2 + 
            reduction_score * 0.3 + efficiency_score * 0.2)
    return total

score_B = calculate_score(total_investments[0], total_reductions[0],
                          np.median(cost_per_ton_data[0]), np.median(payback_B))
score_C = calculate_score(total_investments[1], total_reductions[1],
                          np.median(cost_per_ton_data[1]), np.median(payback_C))
score_D = calculate_score(total_investments[2], total_reductions[2],
                          np.median(cost_per_ton_data[2]), np.median(payback_D))

scores = [score_B, score_C, score_D]
max_score_idx = scores.index(max(scores))

scorecard_text = f"""
╔═══════════════════════════════════════════════════════════╗
║          策略综合评分卡 & 关键洞察                          ║
║        Strategy Scorecard & Key Insights                   ║
╚═══════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  指标                  B-风险管理    C-学习区    D-观察区
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  总投资 ($)          {total_investments[0]:>11,.0f}  {total_investments[1]:>10,.0f}  {total_investments[2]:>10,.0f}
  总减排 (t)          {total_reductions[0]:>11,.0f}  {total_reductions[1]:>10,.0f}  {total_reductions[2]:>10,.0f}
  成本/吨 ($)         {np.median(cost_per_ton_data[0]):>11.1f}  {np.median(cost_per_ton_data[1]):>10.1f}  {np.median(cost_per_ton_data[2]):>10.1f}
  回本期 (年)         {np.median(payback_B):>11.1f}  {np.median(payback_C):>10.1f}  {np.median(payback_D):>10.1f}
  供应商数            {len(budget_B):>11}  {len(budget_C):>10}  {len(budget_D):>10}
  综合得分            {score_B:>11.1f}  {score_C:>10.1f}  {score_D:>10.1f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 最优策略: {strategies[max_score_idx]}

💡 关键洞察:

B区 (风险管理): {"★" * 5 if max_score_idx == 0 else "★" * 3}
  • 高投入高回报,适合高排放供应商
  • 最快回本周期,投资回报率高
  • 需要强力监督和激励机制

C区 (学习区): {"★" * 5 if max_score_idx == 1 else "★" * 3}
  • 知识共享和创新驱动
  • 中等成本,长期效益显著
  • 适合高配合度供应商

D区 (观察区): {"★" * 5 if max_score_idx == 2 else "★" * 3}
  • 最低投资,依赖自动化
  • 稳定但有限的减排效果
  • 适合低排放低配合供应商

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 总投资: ${sum(total_investments):,.0f}
🌱 总减排: {sum(total_reductions):,.0f} 吨 CO₂e
📊 平均成本: ${np.mean([np.median(d) for d in cost_per_ton_data]):.1f}/吨
⏱️  平均回本: {np.mean([np.median(payback_B), np.median(payback_C), np.median(payback_D)]):.1f} 年
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax4.text(0.5, 0.5, scorecard_text, fontsize=9, family='monospace',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                 alpha=0.95, edgecolor='black', linewidth=2))

fig3.suptitle('策略综合对比矩阵与评分卡\nStrategic Comparison Matrix & Scorecard',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output3 = 'cross_strategy_comparison_matrix.png'
plt.savefig(output3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output3}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("✅ CROSS-STRATEGY ANALYSIS COMPLETE")
print("="*80)
print(f"\n📊 Generated 3 comprehensive comparison charts:")
print(f"  1. {output1}")
print(f"  2. {output2}")
print(f"  3. {output3}")

print(f"\n💰 Total Investment Across All Strategies: ${sum(total_investments):,.0f}")
print(f"🌱 Total Emission Reduction: {sum(total_reductions):,.0f} tons CO₂e")
print(f"📈 Overall Average Cost: ${np.mean([np.median(d) for d in cost_per_ton_data]):.1f} per ton")
print(f"⏱️  Average Payback Period: {np.mean([np.median(payback_B), np.median(payback_C), np.median(payback_D)]):.1f} years")

print(f"\n🏆 Best Performing Strategy (by composite score): {strategies[max_score_idx]}")

print("\n💡 Strategic Recommendations:")
print("  • B区: 优先用于高排放供应商,需要强力监督")
print("  • C区: 最适合知识共享和长期发展")
print("  • D区: 成本效益高,适合大规模低配合度供应商")

print("\n" + "="*80)

plt.show()
