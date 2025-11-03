import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 创建供应商数据
# 基于纺织行业基准数据估算
suppliers_data = {
    'HANGZHOU FUEN Ltd': {'type': '染整', 'annual_output_m': 8000000, 'engagement': 7},
    'SHAOXING KEQIAO': {'type': '织造', 'annual_output_m': 12000000, 'engagement': 8},
    'NINGBO YINZHOU': {'type': '染整', 'annual_output_m': 6000000, 'engagement': 5},
    'SUZHOU WUJIANG': {'type': '织造', 'annual_output_m': 10000000, 'engagement': 6},
    'CHANGZHOU TEXTILE': {'type': '染整', 'annual_output_m': 9000000, 'engagement': 4},
    'JIAXING TONGXIANG': {'type': '织造', 'annual_output_m': 7000000, 'engagement': 7},
    'HUZHOU NANXUN': {'type': '染整', 'annual_output_m': 5000000, 'engagement': 3},
    'WENZHOU OUHAI': {'type': '整理', 'annual_output_m': 4000000, 'engagement': 8},
    'TAIZHOU LUQIAO': {'type': '织造', 'annual_output_m': 11000000, 'engagement': 5},
    'JINHUA YIWU': {'type': '整理', 'annual_output_m': 3500000, 'engagement': 6},
    'QUZHOU KECHENG': {'type': '染整', 'annual_output_m': 7500000, 'engagement': 4},
    'ZHOUSHAN DINGHAI': {'type': '织造', 'annual_output_m': 6500000, 'engagement': 7},
    'LISHUI LIANDU': {'type': '整理', 'annual_output_m': 3000000, 'engagement': 9},
    'HUAI\'AN TEXTILE': {'type': '染整', 'annual_output_m': 8500000, 'engagement': 3},
    'YANGZHOU JIANGDU': {'type': '织造', 'annual_output_m': 9500000, 'engagement': 6},
    'NANTONG HAIMEN': {'type': '染整', 'annual_output_m': 10000000, 'engagement': 2},
    'WUXI YIXING': {'type': '织造', 'annual_output_m': 8000000, 'engagement': 8},
    'CHANGZHOU WUJIN': {'type': '整理', 'annual_output_m': 4500000, 'engagement': 7},
    'XUZHOU PEIXIAN': {'type': '染整', 'annual_output_m': 7000000, 'engagement': 5},
    'LIANYUNGANG TEXTILE': {'type': '织造', 'annual_output_m': 6000000, 'engagement': 4},
    # New I-zone suppliers from A_reduction_3years
    'HANGZHOU FUEN': {'type': '染整', 'annual_output_m': 9200000, 'engagement': 8},
    'JIANGSU TEXTILE': {'type': '织造', 'annual_output_m': 8800000, 'engagement': 7},
    'ZHEJIANG DYEING': {'type': '染整', 'annual_output_m': 10500000, 'engagement': 8},
    'SUZHOU FABRIC': {'type': '织造', 'annual_output_m': 7800000, 'engagement': 7},
}

# 2. 计算碳排放（基于行业基准）
# 能耗标准（kWh/米）：织造 1.0, 染整 3.5, 整理 0.5
# 中国电网平均碳排放因子: 0.58 kgCO2/kWh
energy_intensity = {
    '织造': 1.0,
    '染整': 3.5,
    '整理': 0.5
}

carbon_factor = 0.58  # kgCO2/kWh

def calculate_emissions(supplier_name, data):
    output = data['annual_output_m']
    process_type = data['type']
    energy = output * energy_intensity[process_type]  # kWh
    emissions = energy * carbon_factor / 1000  # 转换为吨CO2
    return emissions

# 创建DataFrame
df_list = []
for name, data in suppliers_data.items():
    emissions = calculate_emissions(name, data)
    df_list.append({
        '供应商': name,
        '工艺类型': data['type'],
        '年产量(百万米)': data['annual_output_m'] / 1_000_000,
        '年碳排放(吨CO2)': round(emissions, 2),
        '配合程度': data['engagement']
    })

df = pd.DataFrame(df_list)

# 3. 标准化评分 (0-10分)
# 排放影响 C_i: 按碳排放量占比线性映射
total_emissions = df['年碳排放(吨CO2)'].sum()
df['排放占比(%)'] = (df['年碳排放(吨CO2)'] / total_emissions * 100).round(2)
df['排放影响得分(C)'] = (df['年碳排放(吨CO2)'] / df['年碳排放(吨CO2)'].max() * 10).round(2)

# 配合程度 E_i: 已有评分(1-10)
df['配合程度得分(E)'] = df['配合程度']

# 4. 计算综合等级（可选权重）
w1 = 0.4  # 配合程度权重
w2 = 0.6  # 排放影响权重
df['综合得分(S)'] = (w1 * df['配合程度得分(E)'] + w2 * df['排放影响得分(C)']).round(2)

# 5. 象限分类
def classify_quadrant(row):
    e = row['配合程度得分(E)']
    c = row['排放影响得分(C)']
    
    if e >= 5.5 and c >= 5.5:
        return 'I-核心合作区'
    elif e < 5.5 and c >= 5.5:
        return 'II-风险区'
    elif e >= 5.5 and c < 5.5:
        return 'III-学习区'
    else:
        return 'IV-观察区'

df['象限分类'] = df.apply(classify_quadrant, axis=1)

# 保存数据
df_sorted = df.sort_values('年碳排放(吨CO2)', ascending=False)
df_sorted.to_csv(os.path.join(script_dir, 'supplier_classification.csv'), index=False, encoding='utf-8-sig')

print("=" * 80)
print("供应商ESG分层分析结果")
print("=" * 80)
print(f"\n总供应商数量: {len(df)}")
print(f"总碳排放量: {df['年碳排放(吨CO2)'].sum():.2f} 吨CO2/年")
print(f"\n各象限分布:")
print(df['象限分类'].value_counts().sort_index())

print("\n" + "=" * 80)
print("TOP 10 高排放供应商:")
print("=" * 80)
print(df_sorted[['供应商', '工艺类型', '年碳排放(吨CO2)', '配合程度', '象限分类']].head(10).to_string(index=False))

# 7. 创建可视化
fig, ax = plt.subplots(figsize=(14, 10))

# 定义象限颜色
colors = {
    'I-核心合作区': '#2ECC71',  # 绿色
    'II-风险区': '#E74C3C',      # 红色
    'III-学习区': '#3498DB',    # 蓝色
    'IV-观察区': '#95A5A6'      # 灰色
}

# 绘制背景象限
ax.axhline(y=5.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=5.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# 添加象限背景色
quadrant_alpha = 0.1
rect1 = patches.Rectangle((5.5, 5.5), 5.3, 5.3, linewidth=0, 
                          edgecolor='none', facecolor=colors['I-核心合作区'], alpha=quadrant_alpha)
rect2 = patches.Rectangle((-0.8, 5.5), 6.3, 5.3, linewidth=0, 
                          edgecolor='none', facecolor=colors['II-风险区'], alpha=quadrant_alpha)
rect3 = patches.Rectangle((5.5, -0.8), 5.3, 6.3, linewidth=0, 
                          edgecolor='none', facecolor=colors['III-学习区'], alpha=quadrant_alpha)
rect4 = patches.Rectangle((-0.8, -0.8), 6.3, 6.3, linewidth=0, 
                          edgecolor='none', facecolor=colors['IV-观察区'], alpha=quadrant_alpha)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)

# 绘制散点图
for quadrant in df['象限分类'].unique():
    mask = df['象限分类'] == quadrant
    ax.scatter(df[mask]['配合程度得分(E)'], 
              df[mask]['排放影响得分(C)'],
              c=colors[quadrant],
              s=df[mask]['年碳排放(吨CO2)'] * 2,  # 减小气泡大小以适应图表
              alpha=0.7,
              edgecolors='black',
              linewidth=1.5,
              label=quadrant)

# 添加标签 - 在气泡中心显示
for idx, row in df.iterrows():
    # 简化供应商名称（取前15个字符）
    label = row['供应商'][:15]
    
    x_pos = row['配合程度得分(E)']
    y_pos = row['排放影响得分(C)']
    
    # 在气泡中心显示文字，黄色背景
    ax.text(x_pos, y_pos, label,
           fontsize=7, ha='center', va='center',
           weight='bold', color='black',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                    edgecolor='none', alpha=0.7))

# 添加象限标签 - 使用黑色文字
ax.text(8, 8.5, 'I 核心合作区\n(高配合×高排放)', 
       fontsize=12, ha='center', weight='bold', color='black')
ax.text(2.5, 8.5, 'II 风险区\n(低配合×高排放)', 
       fontsize=12, ha='center', weight='bold', color='black')
ax.text(8, 2.5, 'III 学习区\n(高配合×低排放)', 
       fontsize=12, ha='center', weight='bold', color='black')
ax.text(2.5, 2.5, 'IV 观察区\n(低配合×低排放)', 
       fontsize=12, ha='center', weight='bold', color='black')

# 设置坐标轴
ax.set_xlabel('配合程度得分 (Engagement Level)', fontsize=14, weight='bold')
ax.set_ylabel('排放影响得分 (Emission Impact)', fontsize=14, weight='bold')
ax.set_title('ABC时尚供应商ESG四象限分层模型\nSupplier Engagement & Emission Matrix', 
            fontsize=16, weight='bold', pad=20)

# 根据数据自动调整坐标轴范围，留出一些边距
x_margin = 0.8
y_margin = 0.8
ax.set_xlim(-x_margin, 10 + x_margin)
ax.set_ylim(-y_margin, 10 + y_margin)
ax.grid(True, alpha=0.3, linestyle=':')

# 添加说明
info_text = f"""
数据说明:
• 总供应商: {len(df)}家
• 总排放: {df['年碳排放(吨CO2)'].sum():.0f} 吨CO2/年
• 气泡大小 = 年碳排放量
• 评分范围: 0-10分
• 分界线: 5.5分
"""
ax.text(1.02, 0.15, info_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='bottom',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))



plt.tight_layout()
output_path = os.path.join(script_dir, 'supplier_quadrant_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图表已保存为 {output_path}")

# 8. 创建策略建议表
print("\n" + "=" * 80)
print("各象限策略建议:")
print("=" * 80)

strategies = {
    'I-核心合作区': {
        '特征': '最大减碳潜力、最容易共创成功',
        '策略': '🌿 重点伙伴关系-共建低碳项目-联合投资创新技术-优先签订长期合同',
        '供应商': df[df['象限分类'] == 'I-核心合作区']['供应商'].tolist()
    },
    'II-风险区': {
        '特征': '排放高但不配合，潜在威胁最大',
        '策略': '⚠️ 重点干预对象-加强沟通与合规要求-提供培训或资金激励-若持续不配合→逐步替换',
        '供应商': df[df['象限分类'] == 'II-风险区']['供应商'].tolist()
    },
    'III-学习区': {
        '特征': '主动配合但排放较低',
        '策略': '💬 示范与传播者-作为"绿色先锋"案例-参与经验分享-可带动同行改进',
        '供应商': df[df['象限分类'] == 'III-学习区']['供应商'].tolist()
    },
    'IV-观察区': {
        '特征': '影响有限、资源投入回报低',
        '策略': '💤 基础管理-保持沟通-简化要求，不重点投入',
        '供应商': df[df['象限分类'] == 'IV-观察区']['供应商'].tolist()
    }
}

for quadrant, info in strategies.items():
    print(f"\n【{quadrant}】")
    print(f"特征: {info['特征']}")
    print(f"策略: {info['策略']}")
    print(f"供应商数量: {len(info['供应商'])}家")
    if info['供应商']:
        print(f"供应商列表: {', '.join(info['供应商'][:3])}" + 
              (f" 等{len(info['供应商'])}家" if len(info['供应商']) > 3 else ""))

# 9. 创建详细策略表
strategy_df = pd.DataFrame([
    {
        '象限': 'I-核心合作区',
        '供应商数量': len(strategies['I-核心合作区']['供应商']),
        '平均排放(吨)': df[df['象限分类'] == 'I-核心合作区']['年碳排放(吨CO2)'].mean(),
        '平均配合度': df[df['象限分类'] == 'I-核心合作区']['配合程度'].mean(),
        '优先级': '★★★★★',
        '投入资源': '高',
        '预期减排潜力': '40-50%'
    },
    {
        '象限': 'II-风险区',
        '供应商数量': len(strategies['II-风险区']['供应商']),
        '平均排放(吨)': df[df['象限分类'] == 'II-风险区']['年碳排放(吨CO2)'].mean(),
        '平均配合度': df[df['象限分类'] == 'II-风险区']['配合程度'].mean(),
        '优先级': '★★★★☆',
        '投入资源': '中-高',
        '预期减排潜力': '20-30%'
    },
    {
        '象限': 'III-学习区',
        '供应商数量': len(strategies['III-学习区']['供应商']),
        '平均排放(吨)': df[df['象限分类'] == 'III-学习区']['年碳排放(吨CO2)'].mean(),
        '平均配合度': df[df['象限分类'] == 'III-学习区']['配合程度'].mean(),
        '优先级': '★★★☆☆',
        '投入资源': '低-中',
        '预期减排潜力': '10-15%'
    },
    {
        '象限': 'IV-观察区',
        '供应商数量': len(strategies['IV-观察区']['供应商']),
        '平均排放(吨)': df[df['象限分类'] == 'IV-观察区']['年碳排放(吨CO2)'].mean(),
        '平均配合度': df[df['象限分类'] == 'IV-观察区']['配合程度'].mean(),
        '优先级': '★★☆☆☆',
        '投入资源': '低',
        '预期减排潜力': '5-10%'
    }
])

strategy_df.to_csv(os.path.join(script_dir, 'strategy_summary.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("象限策略汇总表:")
print("=" * 80)
print(strategy_df.to_string(index=False))

print("\n" + "=" * 80)
print("✅ 分析完成!")
print("=" * 80)
print("\n生成文件:")
print("1. supplier_classification.csv - 完整供应商分类数据")
print("2. strategy_summary.csv - 策略汇总表")
print("3. supplier_quadrant_analysis.png - 四象限可视化图")