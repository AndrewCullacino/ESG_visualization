import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取ML模拟生成的数据
try:
    df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_B_三年减排路径.csv'))
    budget_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_B_投资预算分配.csv'))
    supplier_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_B_供应商详细信息.csv'))
    classification_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_B_四象限分类.csv'))
    strategy_df = pd.read_csv(os.path.join(script_dir, 'ML_simulation_B_strategy_summary.csv'))
    
    print("✓ 成功读取所有数据文件")
except FileNotFoundError as e:
    print(f"❌ 错误: 找不到数据文件 - {e}")
    print("请先运行 B_strategy_ML_simulation.py 生成数据文件")
    exit(1)

# 只保留II区供应商数据
df_ii = df[df['象限'] == 'II区'].copy()

# 使用所有供应商的数据
suppliers = df_ii['供应商'].unique()
years = ['基线年', '第1年', '第2年', '第3年']

print(f"✓ 正在分析 {len(suppliers)} 个II区供应商的数据...")

# ============================================================================
# 图表: 配合度改善综合分析（堆叠柱状图 + 散点趋势图）
# ============================================================================
print("\n" + "="*80)
print("📊 生成配合度改善综合分析图...")
print("="*80)

fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor('white')
fig.suptitle('II区供应商配合度改善分析\nZone II Suppliers: Cooperation Score Improvement Analysis', 
             fontsize=18, fontweight='bold', color='black', y=0.96)

# 创建子图
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 获取配合度数据
initial_coop = strategy_df['初始配合度'].values
final_coop = strategy_df['最终配合度'].values
coop_improvement = strategy_df['配合度提升'].values
suppliers_list = strategy_df['供应商'].values

# ========== 左图：堆叠柱状图（Top 12 提升最大供应商）==========
# 按提升幅度排序
sorted_indices = np.argsort(coop_improvement)
top_12_indices = sorted_indices[-12:]  # 显示提升最大的12个供应商

suppliers_top = strategy_df.iloc[top_12_indices]['供应商'].values
initial_top = initial_coop[top_12_indices]
improvement_top = coop_improvement[top_12_indices]

# 创建堆叠柱状图
x_pos = np.arange(len(suppliers_top))
width = 0.7

# 初始配合度（底部）
bars1 = ax1.bar(x_pos, initial_top, width, label='初始配合度', 
               color='#E57373', alpha=0.85, edgecolor='black', linewidth=1.2)

# 提升部分（顶部）
bars2 = ax1.bar(x_pos, improvement_top, width, bottom=initial_top,
               label='配合度提升', color='#81C784', alpha=0.9, 
               edgecolor='black', linewidth=1.2)

# 添加数值标签
for i, (init, imp, bar1, bar2) in enumerate(zip(initial_top, improvement_top, bars1, bars2)):
    # 初始值标签
    ax1.text(bar1.get_x() + bar1.get_width()/2, init/2,
            f'{init:.1f}',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 提升值标签
    if imp > 0.3:  # 只在提升较大时显示
        ax1.text(bar2.get_x() + bar2.get_width()/2, init + imp/2,
                f'+{imp:.1f}',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 最终值标签（顶部）
    final_val = init + imp
    ax1.text(bar2.get_x() + bar2.get_width()/2, final_val + 0.15,
            f'{final_val:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')

ax1.set_xlabel('供应商 (Top 12 按提升排序)\nSuppliers (Top 12 by Improvement)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('配合度评分 (0-10)\nCooperation Score', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('配合度改善分析 (Top 12 供应商)\nCooperation Score Improvement Analysis', 
              fontsize=13, fontweight='bold', pad=15, color='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([s.replace('SUP_B_', 'S') for s in suppliers_top], 
                    fontsize=9, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax1.set_ylim(0, 10)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)

# 添加Zone I门槛线
ax1.axhline(y=5.5, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.text(len(suppliers_top) - 0.5, 5.7, 'Zone I 门槛', fontsize=10, 
        bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.7))

# ========== 右图：散点图 + 趋势线（所有供应商）==========
# 为每个供应商创建数据点
x_positions = np.arange(len(suppliers_list))

# 绘制初始配合度散点
scatter1 = ax2.scatter(x_positions, initial_coop, 
                     s=120, color='#E57373', alpha=0.7, 
                     edgecolors='darkred', linewidth=1.5,
                     label='初始配合度 / Initial Score',
                     zorder=3)

# 绘制最终配合度散点
scatter2 = ax2.scatter(x_positions, final_coop, 
                     s=120, color='#81C784', alpha=0.7, 
                     edgecolors='darkgreen', linewidth=1.5,
                     label='最终配合度 / Final Score',
                     zorder=3)

# 绘制连接线显示提升
for i, (init, final) in enumerate(zip(initial_coop, final_coop)):
    ax2.plot([i, i], [init, final], 
           color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
           zorder=2)
    
    # 添加箭头显示提升方向
    ax2.annotate('', xy=(i, final), xytext=(i, init),
               arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.6),
               zorder=2)

# 绘制初始配合度趋势线
z_init = np.polyfit(x_positions, initial_coop, 2)
p_init = np.poly1d(z_init)
x_smooth = np.linspace(x_positions.min(), x_positions.max(), 200)
ax2.plot(x_smooth, p_init(x_smooth), 
       color='red', linestyle='--', linewidth=2.5, alpha=0.6,
       label='初始趋势线 / Initial Trend',
       zorder=1)

# 绘制最终配合度趋势线
z_final = np.polyfit(x_positions, final_coop, 2)
p_final = np.poly1d(z_final)
ax2.plot(x_smooth, p_final(x_smooth), 
       color='green', linestyle='--', linewidth=2.5, alpha=0.6,
       label='最终趋势线 / Final Trend',
       zorder=1)

# 添加Zone I门槛线
ax2.axhline(y=5.5, color='orange', linestyle='--', linewidth=2.5, alpha=0.7, 
          label='Zone I 门槛 / Threshold (5.5)')
ax2.fill_between(x_positions, 5.5, 10, color='lightgreen', alpha=0.1, label='Zone I 范围')

# 设置图表样式
ax2.set_xlabel('供应商 / Suppliers', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_ylabel('配合度评分 / Cooperation Score (0-10)', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title('配合度改善轨迹与趋势分析\nCooperation Score Improvement Trajectory & Trends', 
            fontsize=13, fontweight='bold', pad=15, color='black')

ax2.set_xticks(x_positions[::2])  # 每隔一个显示
ax2.set_xticklabels([s.replace('SUP_B_', 'S') for s in suppliers_list[::2]], 
                   fontsize=8, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, zorder=0)
ax2.set_ylim(2, 10)
ax2.set_xlim(-0.5, len(suppliers_list) - 0.5)

# 图例
ax2.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black', ncol=2)

# 添加统计信息框
avg_initial = initial_coop.mean()
avg_final = final_coop.mean()
avg_improvement = coop_improvement.mean()
max_improvement = coop_improvement.max()
min_improvement = coop_improvement.min()

stats_text = (f"平均初始配合度: {avg_initial:.2f}\n"
             f"平均最终配合度: {avg_final:.2f}\n"
             f"平均提升: +{avg_improvement:.2f}\n"
             f"最大提升: +{max_improvement:.2f}\n"
             f"最小提升: +{min_improvement:.2f}\n"
             f"达到Zone I门槛: {sum(final_coop >= 5.5)}家")

ax2.text(0.98, 0.05, stats_text,
        transform=ax2.transAxes,
        fontsize=9, fontweight='bold',
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', 
                 alpha=0.95, edgecolor='darkblue', linewidth=2))

# 添加底部说明文字
fig.text(0.5, 0.02, 
         f'N = {len(suppliers_list)} | 平均配合度提升: +{avg_improvement:.2f} | {sum(final_coop >= 5.5)}家供应商达到Zone I门槛 | Zone II Risk Management Strategy', 
         ha='center', fontsize=11, style='italic', color='dimgray',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.75, 
                  edgecolor='gray', linewidth=1))

plt.tight_layout(rect=[0, 0.05, 1, 0.94])

# 保存图表
output_path = os.path.join(script_dir, 'B区配合度改善综合分析_ML_data.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 配合度改善综合分析图已生成: {output_path}")
print(f"  - 平均初始配合度: {avg_initial:.2f}")
print(f"  - 平均最终配合度: {avg_final:.2f}")
print(f"  - 平均配合度提升: +{avg_improvement:.2f}")
print(f"  - 达到Zone I门槛供应商: {sum(final_coop >= 5.5)}家")
plt.show()


# ============================================================================
# 图表: 投资回报与成本分析（Top 6 供应商）
# ============================================================================
print("\n" + "="*80)
print("📊 生成投资回报与成本分析图...")
print("="*80)

fig2 = plt.figure(figsize=(14, 8))
fig2.patch.set_facecolor('white')
fig2.suptitle('II区供应商投资回报与成本分析 (Top 6)\nZone II Suppliers: Investment Return & Cost Analysis (Top 6)', 
             fontsize=16, fontweight='bold', color='black', y=0.96)

ax = fig2.add_subplot(1, 1, 1)

# 选择投资回报率最高的6个供应商
top_6_roi = budget_df.nlargest(6, '投资回报率')
suppliers_roi = top_6_roi['供应商'].values
total_costs = top_6_roi['总成本'].values
roi_values = top_6_roi['投资回报率'].values
reductions = top_6_roi['预期减排量'].values

# 计算回报（基于减排量和ROI的关系）
# ROI = (减排量 / 总成本) * 100，所以回报 = 减排量
returns = reductions

# 转换成本单位为$10
total_costs_unit = total_costs / 10

# 计算每吨CO2的成本
cost_per_ton = total_costs / reductions

# 创建横向柱状图的位置
y_positions = np.arange(len(suppliers_roi))
height = 0.35

# 绘制成本（红色，向左）
bars_cost = ax.barh(y_positions - height/2, -total_costs_unit, height,
                    label='总成本 / Total Cost (×$10)', 
                    color='#E57373', alpha=0.85, 
                    edgecolor='darkred', linewidth=1.5)

# 绘制回报（绿色，向右）
bars_return = ax.barh(y_positions + height/2, returns, height,
                      label='减排量 (回报) / Emission Reduction (Return)', 
                      color='#81C784', alpha=0.85, 
                      edgecolor='darkgreen', linewidth=1.5)

# 添加成本标签
for i, (bar, cost_unit, cost_per_t) in enumerate(zip(bars_cost, total_costs_unit, cost_per_ton)):
    ax.text(bar.get_width() - abs(cost_unit) * 0.05, bar.get_y() + bar.get_height()/2,
            f'{cost_unit:.0f}×$10',
            ha='right', va='center', fontsize=10, fontweight='bold', color='darkred')

# 添加回报标签和$/tCO2成本
for i, (bar, ret, roi, supplier, cost_per_t) in enumerate(zip(bars_return, returns, roi_values, suppliers_roi, cost_per_ton)):
    ax.text(bar.get_width() + max(returns) * 0.02, bar.get_y() + bar.get_height()/2,
            f'{ret:,.0f}t CO₂',
            ha='left', va='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    # 添加$/tCO2成本和ROI标签在中间
    # 判断成本是否在标准范围内
    if cost_per_t < 20:
        cost_color = 'green'
        cost_status = '✓'
    elif cost_per_t <= 40:
        cost_color = 'orange'
        cost_status = '~'
    else:
        cost_color = 'red'
        cost_status = '✗'
    
    ax.text(0, y_positions[i],
            f'{cost_status} ${cost_per_t:.1f}/t\nROI: {roi:.0f}%',
            ha='center', va='center', fontsize=8, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=cost_color, alpha=0.8))

# 设置图表样式
ax.set_ylabel('供应商 (按ROI排序) / Suppliers (Ranked by ROI)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_xlabel('金额 (×$10) ← 成本  |  回报 (吨 CO₂e) →\nCost (×$10) ←  |  Return (tons CO₂e) →', 
              fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('投资成本与减排回报对比分析 (标准成本: $20-$40/tCO₂)\nInvestment Cost vs. Emission Reduction Return (Standard: $20-$40/tCO₂)', 
            fontsize=13, fontweight='bold', pad=20, color='black')

ax.set_yticks(y_positions)
ax.set_yticklabels([s.replace('SUP_B_', 'S') for s in suppliers_roi], 
                   fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# 设置x轴范围，让成本和回报都能完整显示
max_cost_unit = max(total_costs_unit)
max_return = max(returns)
ax.set_xlim(-max_cost_unit * 1.2, max_return * 1.2)

# 添加中轴线
ax.axvline(x=0, color='black', linewidth=2, alpha=0.5)

# 图例
ax.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')

# 添加统计信息框
avg_cost = total_costs.mean()
avg_cost_unit = total_costs_unit.mean()
avg_return = returns.mean()
avg_roi = roi_values.mean()
avg_cost_per_ton = cost_per_ton.mean()

# 统计成本范围内的供应商数量
below_standard = sum(cost_per_ton < 20)
in_standard = sum((cost_per_ton >= 20) & (cost_per_ton <= 40))
above_standard = sum(cost_per_ton > 40)

stats_text = (f"平均成本: {avg_cost_unit:.0f}×$10\n"
             f"平均减排: {avg_return:,.0f}t\n"
             f"平均$/tCO₂: ${avg_cost_per_ton:.1f}\n"
             f"平均ROI: {avg_roi:.0f}%\n"
             f"━━━━━━━━━━\n"
             f"成本对比标准($20-$40/t):\n"
             f"✓ 低于标准: {below_standard}家\n"
             f"~ 标准范围: {in_standard}家\n"
             f"✗ 高于标准: {above_standard}家")

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', 
                 alpha=0.95, edgecolor='orange', linewidth=2))

# 添加底部说明文字
fig2.text(0.5, 0.02, 
         f'Top 6 Suppliers by ROI | 平均成本: {avg_cost_unit:.0f}×$10 (${avg_cost:,.0f}) | 平均$/tCO₂: ${avg_cost_per_ton:.1f} | 标准: $20-$40/tCO₂', 
         ha='center', fontsize=11, style='italic', color='dimgray',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.75, 
                  edgecolor='darkblue', linewidth=1))

plt.tight_layout(rect=[0, 0.05, 1, 0.94])

# 保存图表
output_path_roi = os.path.join(script_dir, 'B区投资回报与成本分析_Top6_ML_data.png')
plt.savefig(output_path_roi, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 投资回报与成本分析图已生成: {output_path_roi}")
print(f"  - 平均成本: {avg_cost_unit:.0f}×$10 (${avg_cost:,.0f})")
print(f"  - 平均减排回报: {avg_return:,.0f} 吨CO₂e")
print(f"  - 平均$/tCO₂: ${avg_cost_per_ton:.1f} (标准: $20-$40)")
print(f"  - 平均ROI: {avg_roi:.0f}%")
print(f"  - 成本表现: ✓{below_standard}家 ~{in_standard}家 ✗{above_standard}家")
plt.show()


# ============================================================================
# 最终汇总统计
# ============================================================================
print("\n" + "="*80)
print("📊 B策略 (Zone II) 配合度改善分析汇总")
print("="*80)
print(f"\n供应商概况:")
print(f"  - 总供应商数: {len(suppliers)}家")
print(f"  - 初始平均配合度: {avg_initial:.2f}/10")
print(f"  - 最终平均配合度: {avg_final:.2f}/10")
print(f"  - 平均配合度提升: +{avg_improvement:.2f} 分")
print(f"  - 达到Zone I门槛: {sum(final_coop >= 5.5)}家")

print(f"\n配合度分布:")
print(f"  - 最大提升: +{max_improvement:.2f} 分")
print(f"  - 最小提升: +{min_improvement:.2f} 分")
print(f"  - 初始配合度范围: {initial_coop.min():.1f} - {initial_coop.max():.1f}")
print(f"  - 最终配合度范围: {final_coop.min():.1f} - {final_coop.max():.1f}")

print("\n" + "="*80)
print("✅ B策略可视化分析完成!")
print("="*80)
print(f"\n生成的图表:")
print(f"  1. {output_path}")
print(f"  2. {output_path_roi}")
print("\n💡 关键发现:")
print(f"  • Zone II供应商配合度显著提升，{sum(final_coop >= 5.5)}家供应商达到Zone I门槛")
print(f"  • 平均配合度从{avg_initial:.2f}提升至{avg_final:.2f}，提升幅度达{avg_improvement:.2f}分")
print(f"  • 趋势线显示整体改善态势良好，多数供应商响应积极")
print(f"  • Top 6供应商平均ROI达{avg_roi:.1f}%，投资回报优秀")
print("  • 需持续监督和激励以确保配合度持续提升")


