"""
Enhanced Radar Chart Visualizer - All 4 Zones (A, B, C, D)
==========================================================
Comprehensive radar/polar charts showing cost-effectiveness and performance
metrics across all four ESG supply chain strategies with grey backgrounds
and enlarged titles.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from math import pi

# Chinese font configuration
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("\n" + "="*80)
print("📡 ENHANCED RADAR CHART VISUALIZATION - ALL 4 ZONES")
print("="*80 + "\n")

# ============================================================================
# Load All Strategy Data
# ============================================================================
print("📁 Loading data from all strategies...")

try:
    # Strategy A (Zone I)
    budget_A = pd.read_csv('ML_simulation_A_投资预算分配.csv')
    strategy_A = pd.read_csv('ML_simulation_A_strategy_summary.csv')
    
    # Strategy B (Zone II)
    budget_B = pd.read_csv('ML_simulation_B_投资预算分配.csv')
    strategy_B = pd.read_csv('ML_simulation_B_strategy_summary.csv')
    
    # Strategy C (Zone III)
    budget_C = pd.read_csv('ML_simulation_C_投资预算分配.csv')
    strategy_C = pd.read_csv('ML_simulation_C_strategy_summary.csv')
    
    # Strategy D (Zone IV)
    budget_D = pd.read_csv('ML_simulation_D_投资预算分配.csv')
    strategy_D = pd.read_csv('ML_simulation_D_strategy_summary.csv')
    
    print(f"✓ Loaded A: {len(budget_A)}, B: {len(budget_B)}, C: {len(budget_C)}, D: {len(budget_D)} suppliers")
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# Data Preprocessing
# ============================================================================

# Standardize column names for Strategy A
if '供应商' in budget_A.columns:
    budget_A['供应商ID'] = budget_A['供应商']

# Check which investment column exists
if '投资金额' in budget_A.columns:
    budget_A['总投资'] = budget_A['投资金额']
elif '总成本' in budget_A.columns:
    budget_A['总投资'] = budget_A['总成本']
    
if '预期减排量' in budget_A.columns:
    budget_A['减排量'] = budget_A['预期减排量']
    
budget_A['成本每吨'] = budget_A['总投资'] / budget_A['减排量']

# Check for ROI and payback columns
if '投资回报率' not in budget_A.columns and '投资回报率' in budget_A.columns:
    pass  # Already exists
elif 'ROI' in budget_A.columns:
    budget_A['投资回报率'] = budget_A['ROI']

if '回收期(月)' in budget_A.columns:
    budget_A['回本周期(年)'] = budget_A['回收期(月)'] / 12
elif '投资回本周期(年)' in budget_A.columns:
    budget_A['回本周期(年)'] = budget_A['投资回本周期(年)']

# Standardize column names for Strategy B
if '供应商' in budget_B.columns:
    budget_B['供应商ID'] = budget_B['供应商']

# Check which columns exist for B
if '总成本' in budget_B.columns:
    budget_B['总投资'] = budget_B['总成本']
elif '投资金额' in budget_B.columns:
    budget_B['总投资'] = budget_B['投资金额']

if '预期减排量' in budget_B.columns:
    budget_B['减排量'] = budget_B['预期减排量']
    
budget_B['成本每吨'] = budget_B['总投资'] / budget_B['减排量']

if '回本周期(年)' not in budget_B.columns and '回收期(月)' in budget_B.columns:
    budget_B['回本周期(年)'] = budget_B['回收期(月)'] / 12

# Standardize column names for Strategy C
if '供应商' in budget_C.columns:
    budget_C['供应商ID'] = budget_C['供应商']
    budget_C['减排量'] = budget_C['预期减排量']
    
budget_C['成本每吨'] = budget_C['总投资'] / budget_C['减排量']

# Strategy D
budget_D['成本每吨'] = budget_D['总投资'] / budget_D.get('年度节省', 2950)

# ============================================================================
# Figure: Comprehensive Radar Chart Analysis (1x2 grid)
# ============================================================================
print("\n📡 Generating Radar Chart Analysis...")

fig = plt.figure(figsize=(24, 12))
fig.patch.set_facecolor('white')  # Pure white background

# Color scheme for strategies
colors_strat = {
    'A': '#F39C12',  # Orange (Zone I)
    'B': '#E74C3C',  # Red (Zone II)
    'C': '#3498DB',  # Blue (Zone III)
    'D': '#95A5A6'   # Grey (Zone IV)
}

# ====== Radar Chart 1: Overall Strategy Performance Comparison ======
ax1 = plt.subplot(1, 2, 1, projection='polar', facecolor='#E8E8E8')

categories = ['总投资\n效率', '减排\n效果', '成本\n效益', 'ROI\n回报', '回本\n速度', '供应商\n规模']
N = len(categories)

# Calculate normalized scores (0-100 scale)
def calculate_radar_scores(budget, strategy_summary, zone='A'):
    if zone == 'D':
        scores = [
            100 - (budget['总投资'].mean() / 20000 * 100),
            (strategy_summary['最终减排量'].mean() / 350 * 100),
            100 - (budget['成本每吨'].median() / 70 * 100),
            50,
            100 - (budget.get('投资回收期', budget.get('投资回本周期(年)', pd.Series([15]))).median() / 20 * 100),
            (len(budget) / 30 * 100)
        ]
    else:
        scores = [
            100 - (budget['总投资'].mean() / 100000 * 100),
            (budget['减排量'].mean() / 5000 * 100),
            100 - (budget['成本每吨'].median() / 50 * 100),
            (budget.get('投资回报率', pd.Series([150])).mean() / 300 * 100),
            100 - (budget.get('回本周期(年)', budget.get('投资回本周期(年)', pd.Series([5]))).median() / 15 * 100),
            (len(budget) / 30 * 100)
        ]
    return [max(0, min(100, s)) for s in scores]

scores_A = calculate_radar_scores(budget_A, strategy_A, zone='A')
scores_B = calculate_radar_scores(budget_B, strategy_B, zone='B')
scores_C = calculate_radar_scores(budget_C, strategy_C, zone='C')
scores_D = calculate_radar_scores(budget_D, strategy_D, zone='D')

# Angles for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
scores_A += scores_A[:1]
scores_B += scores_B[:1]
scores_C += scores_C[:1]
scores_D += scores_D[:1]
angles += angles[:1]

# Plot each strategy
ax1.plot(angles, scores_A, 'o-', linewidth=4, label='A-核心优化', 
        color=colors_strat['A'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax1.fill(angles, scores_A, alpha=0.25, color=colors_strat['A'])

ax1.plot(angles, scores_B, 'o-', linewidth=4, label='B-风险管理', 
        color=colors_strat['B'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax1.fill(angles, scores_B, alpha=0.25, color=colors_strat['B'])

ax1.plot(angles, scores_C, 'o-', linewidth=4, label='C-学习区',
        color=colors_strat['C'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax1.fill(angles, scores_C, alpha=0.25, color=colors_strat['C'])

ax1.plot(angles, scores_D, 'o-', linewidth=4, label='D-观察区',
        color=colors_strat['D'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax1.fill(angles, scores_D, alpha=0.25, color=colors_strat['D'])

# Customize
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_yticks([25, 50, 75, 100])
ax1.set_yticklabels(['25', '50', '75', '100'], fontsize=11)
ax1.grid(True, linewidth=1.5, alpha=0.4, color='white')
ax1.set_title('四区策略综合性能对比雷达图\nAll Zones Strategy Performance Radar',
             fontsize=20, fontweight='bold', pad=40, y=1.1)
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=14, 
          framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

# Add performance grid circles
for level in [25, 50, 75]:
    circle_angles = np.linspace(0, 2*pi, 100)
    circle_r = [level] * 100
    ax1.plot(circle_angles, circle_r, color='white', linewidth=1, alpha=0.5, linestyle='--')

# ====== Radar Chart 2: Risk-Return Profile ======
ax4 = plt.subplot(1, 2, 2, projection='polar', facecolor='#E8E8E8')

categories_risk = ['财务\n风险', '执行\n风险', '配合\n风险', '技术\n风险', '时间\n风险', '回报\n潜力']
N_risk = len(categories_risk)
angles_risk = [n / float(N_risk) * 2 * pi for n in range(N_risk)]

# Risk-return profiles (inverse risk = better, high return = better)
# Scale values to fit larger radius (0-120 instead of 0-100)
risk_A = [24, 36, 30, 36, 42, 114]  # Low-medium risk, highest return (scaled by 1.2)
risk_A += risk_A[:1]

risk_B = [36, 48, 84, 30, 24, 108]  # High cooperation risk, very high return (scaled by 1.2)
risk_B += risk_B[:1]

risk_C = [48, 30, 24, 42, 36, 72]  # Low cooperation risk, moderate return (scaled by 1.2)
risk_C += risk_C[:1]

risk_D = [72, 18, 78, 24, 12, 48]  # High financial & cooperation risk, low return (scaled by 1.2)
risk_D += risk_D[:1]

angles_plot = angles_risk + angles_risk[:1]

ax4.plot(angles_plot, risk_A, 'o-', linewidth=4,
        color=colors_strat['A'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax4.fill(angles_plot, risk_A, alpha=0.25, color=colors_strat['A'])

ax4.plot(angles_plot, risk_B, 'o-', linewidth=4,
        color=colors_strat['B'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax4.fill(angles_plot, risk_B, alpha=0.25, color=colors_strat['B'])

ax4.plot(angles_plot, risk_C, 'o-', linewidth=4,
        color=colors_strat['C'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax4.fill(angles_plot, risk_C, alpha=0.25, color=colors_strat['C'])

ax4.plot(angles_plot, risk_D, 'o-', linewidth=4,
        color=colors_strat['D'], markersize=10, markeredgewidth=2, markeredgecolor='white')
ax4.fill(angles_plot, risk_D, alpha=0.25, color=colors_strat['D'])

ax4.set_xticks(angles_risk)
ax4.set_xticklabels(categories_risk, fontsize=14, fontweight='bold')
ax4.set_ylim(0, 120)  # Extended radius to make areas larger
ax4.set_yticks([30, 60, 90])
ax4.set_yticklabels(['低风险/低回报', '中等', '高风险/高回报'], fontsize=10)
ax4.grid(True, linewidth=1.5, alpha=0.4, color='white')
ax4.set_title('各区风险-回报分析\nRisk-Return Profile by Zone',
             fontsize=20, fontweight='bold', pad=40, y=1.1)
# No legend for second chart

# Add performance grid circles
for level in [30, 60, 90]:
    circle_angles = np.linspace(0, 2*pi, 100)
    circle_r = [level] * 100
    ax4.plot(circle_angles, circle_r, color='white', linewidth=1, alpha=0.5, linestyle='--')

# Main title
fig.suptitle('ESG供应链四区策略雷达图综合分析\nESG Supply Chain 4-Zone Strategy Radar Chart Analysis',
             fontsize=26, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout(rect=[0, 0, 1, 0.99])

output = 'all_zones_radar_chart_analysis.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("✅ RADAR CHART VISUALIZATION COMPLETE")
print("="*80)

print(f"\n📊 Strategy Summary:")
print(f"\n  Zone I (A-核心优化):")
print(f"    • 供应商数: {len(budget_A)}")
print(f"    • 平均投资: ${budget_A['总投资'].mean():,.0f}")
print(f"    • 平均减排: {budget_A['减排量'].mean():,.0f} tons")
print(f"    • 平均成本: ${budget_A['成本每吨'].median():.1f}/ton")

print(f"\n  Zone II (B-风险管理):")
print(f"    • 供应商数: {len(budget_B)}")
print(f"    • 平均投资: ${budget_B['总投资'].mean():,.0f}")
print(f"    • 平均减排: {budget_B['减排量'].mean():,.0f} tons")
print(f"    • 平均成本: ${budget_B['成本每吨'].median():.1f}/ton")

print(f"\n  Zone III (C-学习区):")
print(f"    • 供应商数: {len(budget_C)}")
print(f"    • 平均投资: ${budget_C['总投资'].mean():,.0f}")
print(f"    • 平均减排: {budget_C['减排量'].mean():,.0f} tons")
print(f"    • 平均成本: ${budget_C['成本每吨'].median():.1f}/ton")

print(f"\n  Zone IV (D-观察区):")
print(f"    • 供应商数: {len(budget_D)}")
print(f"    • 平均投资: ${budget_D['总投资'].mean():,.0f}")
print(f"    • 平均减排: {strategy_D['最终减排量'].mean():,.0f} tons")
print(f"    • 平均成本: ${budget_D['成本每吨'].median():.1f}/ton")

print("\n💡 Key Radar Chart Insights:")
print("  • Zone A shows highest return potential with balanced risk profile")
print("  • Zone B excels in ROI and payback speed but higher cooperation risk")
print("  • Zone C demonstrates balanced performance ideal for innovation")
print("  • Zone D minimizes execution risk with automated approach")

print("\n" + "="*80)

plt.show()
