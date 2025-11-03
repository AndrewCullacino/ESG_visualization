import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import numpy as np
import os
from scipy.interpolate import griddata

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取ML模拟生成的数据
df = pd.read_excel(os.path.join(script_dir, 'ML_simulation_ESG供应商数据.xlsx'), sheet_name='三年减排路径')
budget_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_投资预算分配.csv'))
supplier_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_供应商详细信息.csv'))
classification_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_四象限分类.csv'))

# 只保留I区供应商数据
df_i = df[df['象限'] == 'I区'].copy()

# 使用所有24个供应商的数据
suppliers = df_i['供应商'].unique()
years = ['基线年', '第1年', '第2年', '第3年']

print(f"✓ 正在分析 {len(suppliers)} 个I区供应商的数据...")

# ============================================================================
# 图表1: 排放量分析 (Annual Emissions + Cumulative Reduction)
# ============================================================================
print("\n" + "="*80)
print("📊 生成图表1: 排放量分析...")
print("="*80)

fig1 = plt.figure(figsize=(18, 8))
fig1.patch.set_facecolor('white')
fig1.suptitle('I区供应商三年减排路径分析\nZone I Suppliers: Three-Year Emission Reduction Analysis', 
             fontsize=16, fontweight='bold', color='black', y=0.98)
ax1 = fig1.add_subplot(1, 2, 1)
ax2 = fig1.add_subplot(1, 2, 2)

# ========== 左图：按年份汇总的总排放量（柱状图）==========
# 计算每年的总排放量
year_totals = []
for year in years:
    year_data = df_i[df_i['年份'] == year]
    total_emission = year_data['年排放量'].sum()
    year_totals.append(total_emission)

# 创建横向柱状图
y_positions = np.arange(len(years))
colors_gradient = ['#E53935', '#FF7043', '#FFB74D', '#66BB6A']  # 红->橙->黄->绿

bars = ax1.barh(y_positions, year_totals, 
               color=colors_gradient, 
               alpha=0.85, 
               edgecolor='black', 
               linewidth=1.5)

# 添加数值标签和减排百分比
for i, (bar, emission, year) in enumerate(zip(bars, year_totals, years)):
    # 添加排放量标签
    ax1.text(emission + max(year_totals) * 0.02, bar.get_y() + bar.get_height()/2, 
            f'{emission:,.0f}',
            ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 添加减排百分比标签（相对于基线年）
    if i > 0:
        reduction_pct = (year_totals[0] - emission) / year_totals[0] * 100
        ax1.text(emission * 0.5, bar.get_y() + bar.get_height()/2, 
                f'↓ {reduction_pct:.1f}%',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))

ax1.set_ylabel('时间节点', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_xlabel('总碳排放量 (吨CO₂e)', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_title('I区所有供应商年度总排放量汇总\n' + 
              'Total Annual Emissions by Year (All Zone I Suppliers)', 
              fontsize=14, fontweight='bold', pad=20, color='black')
ax1.set_yticks(y_positions)
ax1.set_yticklabels(years, fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
ax1.set_xlim(0, max(year_totals) * 1.15)
ax1.invert_yaxis()

# 添加基线参考线
ax1.axvline(x=year_totals[0], color='red', linestyle='--', alpha=0.5, linewidth=2, label='基线年参考')
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)

# ========== 右图：汇总累计减排量（柱状图）==========
# 计算每年的总累计减排量
cumulative_totals = []
for year in years:
    year_data = df_i[df_i['年份'] == year]
    total_reduction = year_data['累计减排量'].sum()
    cumulative_totals.append(total_reduction)

# 创建柱状图
x_positions = np.arange(len(years))
width = 0.6

bars = ax2.bar(x_positions, cumulative_totals, width,
              color=['#BDBDBD', '#81C784', '#66BB6A', '#43A047'],  # 灰->浅绿->绿->深绿
              alpha=0.85,
              edgecolor='black',
              linewidth=1.5)

# 添加数值标签和年度新增减排量
for i, (bar, cum_reduction) in enumerate(zip(bars, cumulative_totals)):
    # 累计减排量标签
    ax2.text(bar.get_x() + bar.get_width()/2, cum_reduction + max(cumulative_totals) * 0.02,
            f'{cum_reduction:,.0f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 年度新增减排量（标注在柱子中间）
    if i > 0:
        annual_new_reduction = cumulative_totals[i] - cumulative_totals[i-1]
        mid_height = cumulative_totals[i-1] + annual_new_reduction / 2
        ax2.text(bar.get_x() + bar.get_width()/2, mid_height,
                f'+{annual_new_reduction:,.0f}\n(年增量)',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

ax2.set_xlabel('时间节点', fontsize=13, fontweight='bold', labelpad=10)
ax2.set_ylabel('累计减排量 (吨CO₂e)', fontsize=13, fontweight='bold', labelpad=10)
ax2.set_title('I区所有供应商累计减排量汇总\n' + 
              'Total Cumulative Emission Reduction (All Zone I Suppliers)', 
              fontsize=14, fontweight='bold', pad=20, color='black')
ax2.set_xticks(x_positions)
ax2.set_xticklabels(years, fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax2.set_ylim(0, max(cumulative_totals) * 1.15)

# 添加减排目标参考线和注释
target_reductions = [0, year_totals[0] * 0.15, year_totals[0] * 0.30, year_totals[0] * 0.40]
target_labels = ['基线', '15%目标', '30%目标', '40%目标']
target_colors = ['gray', '#4CAF50', '#FF9800', '#F44336']

for i, (target, label, color) in enumerate(zip(target_reductions, target_labels, target_colors)):
    if target > 0:
        ax2.axhline(y=target, color=color, linestyle='--', linewidth=2, alpha=0.6)
        ax2.text(len(years) - 0.5, target, f'  {label}\n  ({target:,.0f}吨)',
                fontsize=9, va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor=color))

# 添加实际完成率标注
actual_vs_target = (cumulative_totals[-1] / target_reductions[-1]) * 100 if target_reductions[-1] > 0 else 0
ax2.text(0.5, max(cumulative_totals) * 1.08,
        f'目标完成率: {actual_vs_target:.1f}%\n总减排: {cumulative_totals[-1]:,.0f} 吨CO₂e',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2),
        transform=ax2.transData)

# 添加底部说明文字
fig1.text(0.5, 0.02, f'N = {len(suppliers)} | Simulated by Machine Learning Optimization Model | Zone I Priority Suppliers', 
         ha='center', fontsize=10, style='italic', color='dimgray',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7, edgecolor='gray', linewidth=1))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

# 保存图表1
output_path1 = os.path.join(script_dir, '图表1_排放量分析_ML_data.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 图表1已生成: {output_path1}")
print(f"  - 基线年总排放: {year_totals[0]:,.0f} 吨CO₂e")
print(f"  - 第3年总排放: {year_totals[-1]:,.0f} 吨CO₂e")
print(f"  - 三年累计减排: {cumulative_totals[-1]:,.0f} 吨CO₂e ({(cumulative_totals[-1]/year_totals[0]*100):.1f}%)")
plt.show()


# ============================================================================
# 图表2: 投资效率与供应商对比 (Investment Efficiency + Radar)
# ============================================================================
print("\n" + "="*80)
print("📊 生成图表2: 投资效率与供应商对比...")
print("="*80)

fig2 = plt.figure(figsize=(18, 8))
fig2.patch.set_facecolor('white')
fig2.suptitle('投资效率与供应商综合对比分析\nInvestment Efficiency & Supplier Comparison Analysis', 
             fontsize=16, fontweight='bold', color='black', y=0.98)
ax3 = fig2.add_subplot(1, 2, 1)
ax4 = fig2.add_subplot(1, 2, 2, projection='polar')

# --- 左侧：投资效率分析散点图 ---
# 计算效率指标
investments = budget_df['投资金额'].values
reductions = budget_df['预期减排量'].values
efficiency = reductions / investments  # tons CO2e per USD

# 创建散点图，颜色根据效率值设置
scatter = ax3.scatter(investments, reductions, 
                    c=efficiency, 
                    cmap='RdYlGn',  # 红->黄->绿
                    s=250, 
                    alpha=0.75, 
                    edgecolors='black', 
                    linewidth=1.5,
                    zorder=3)

# 添加供应商标签（只显示top 6）
top_6_idx = np.argsort(efficiency)[-6:]
for i in top_6_idx:
    inv, red, supplier = investments[i], reductions[i], budget_df['供应商'].iloc[i]
    ax3.annotate(supplier, 
                xy=(inv, red), 
                xytext=(6, 6), 
                textcoords='offset points',
                fontsize=7, 
                fontweight='bold',
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1))

# 添加趋势线
z = np.polyfit(investments, reductions, 1)
p = np.poly1d(z)
x_trend = np.linspace(investments.min(), investments.max(), 100)
ax3.plot(x_trend, p(x_trend), 'b--', alpha=0.5, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}', zorder=2)

ax3.set_xlabel('Investment (USD)', fontsize=12, fontweight='bold', labelpad=8)
ax3.set_ylabel('Emission Reduction (tons CO₂e)', fontsize=12, fontweight='bold', labelpad=8)
ax3.set_title('Investment Efficiency Analysis\n投资效率分析', 
            fontsize=13, fontweight='bold', pad=15, color='black')

# 颜色条
cbar = plt.colorbar(scatter, ax=ax3, label='Efficiency (tons/$)', pad=0.02)
cbar.ax.tick_params(labelsize=9)

# 添加统计信息框
stats_text = f"Avg Eff: {efficiency.mean():.4f}\nBest: {efficiency.max():.4f}\nTotal: ${investments.sum()/1e6:.2f}M"
ax3.text(0.02, 0.98, stats_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=1.5),
        fontfamily='monospace')

ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, zorder=0)
ax3.legend(loc='lower right', fontsize=8, framealpha=0.95)

# --- 右侧：Top 6供应商雷达图对比 ---
# 合并数据获取top 6供应商
full_df = budget_df.copy()
for idx, row in full_df.iterrows():
    supplier_id = row['供应商']
    supplier_info = supplier_df[supplier_df['supplier_id'] == supplier_id]
    if not supplier_info.empty:
        full_df.loc[idx, 'tech_adoption_level'] = supplier_info.iloc[0]['tech_adoption_level']
        full_df.loc[idx, 'cooperation_score'] = supplier_info.iloc[0]['cooperation_score']
        full_df.loc[idx, 'financial_capacity'] = supplier_info.iloc[0]['financial_capacity']

classification_info = classification_df[['供应商', '综合得分(S)']]
full_df = full_df.merge(classification_info, on='供应商', how='left')
full_df['效率'] = full_df['预期减排量'] / full_df['投资金额']

# 选择top 6供应商
top_6_suppliers = full_df.nlargest(6, '效率')

# 雷达图指标（标准化到0-100）
categories = ['ROI\n投资回报', 'Reduction\n减排率', 'Tech\n技术采纳', 
              'Coop.\n配合度', 'Finance\n财务', 'Cost Eff.\n成本效率']

# 为每个供应商绘制雷达图
colors = plt.cm.Greens(np.linspace(0.4, 0.9, 6))
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for idx, (_, supplier) in enumerate(top_6_suppliers.iterrows()):
    # 提取并标准化数据
    values = [
        min(supplier['投资回报率'] / 10, 100),  # ROI (scale down)
        supplier['减排率'] * 2,  # Reduction rate (0-100)
        supplier['tech_adoption_level'] * 100,  # Tech adoption (0-100)
        supplier['cooperation_score'] * 10,  # Cooperation (0-100)
        supplier['financial_capacity'] * 50,  # Financial capacity (0-100)
        supplier['效率'] * 1000,  # Cost efficiency (scaled)
    ]
    
    values += values[:1]  # 闭合图形
    
    # 绘制
    ax4.plot(angles, values, 'o-', linewidth=2, color=colors[idx], 
            label=f"{supplier['供应商']} ({supplier['效率']:.4f})", markersize=5, alpha=0.8)
    ax4.fill(angles, values, alpha=0.15, color=colors[idx])

# 设置雷达图
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax4.set_ylim(0, 130)  # 扩大到130%
ax4.set_yticks([20, 40, 60, 80, 100, 120, 130])
ax4.set_yticklabels(['20', '40', '60', '80', '100', '120', '130'], fontsize=8, color='gray')
ax4.grid(True, linestyle='--', alpha=0.4, linewidth=1)
ax4.set_title('Top 6 Suppliers Comparison (Cost/Return)\nTop 6供应商对比分析', 
            fontsize=13, fontweight='bold', pad=20, color='black')

# 图例
ax4.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=8, 
          framealpha=0.95, edgecolor='darkgreen', title='Efficiency Ranking', title_fontsize=9)

# 添加背景色
ax4.patch.set_facecolor('honeydew')
ax4.patch.set_alpha(0.3)

# 添加底部说明文字
fig2.text(0.5, 0.02, f'N = {len(suppliers)} | Simulated by Machine Learning Optimization Model | Avg. Efficiency: {efficiency.mean():.4f} tons CO₂e/$', 
         ha='center', fontsize=10, style='italic', color='dimgray',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7, edgecolor='gray', linewidth=1))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

# 保存图表2
output_path2 = os.path.join(script_dir, '图表2_投资效率与供应商对比_ML_data.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 图表2已生成: {output_path2}")
print(f"  - 平均投资效率: {efficiency.mean():.4f} 吨CO₂e/美元")
print(f"  - 最佳投资效率: {efficiency.max():.4f} 吨CO₂e/美元")
print(f"  - Top 6 供应商: {', '.join(top_6_suppliers['供应商'].tolist())}")
plt.show()

