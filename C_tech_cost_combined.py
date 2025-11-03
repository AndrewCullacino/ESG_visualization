"""
Strategy C - Technology Effectiveness & Cost Analysis Combined
================================================================
Creates a single combined chart with:
1. Technology Effectiveness (left) - horizontal bar chart
2. Technology Cost vs Knowledge Value (right) - scatter plot
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Chinese font configuration
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# Load Data
# ============================================================================

def load_data():
    """Load required CSV files"""
    try:
        tech_db = pd.read_csv('ML_simulation_C_技术数据库.csv', encoding='utf-8-sig')
        budget_df = pd.read_csv('ML_simulation_C_投资预算分配.csv', encoding='utf-8-sig')
        summary_df = pd.read_csv('ML_simulation_C_strategy_summary.csv', encoding='utf-8-sig')
        print("✓ Data loaded successfully")
        return tech_db, budget_df, summary_df
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None, None


# ============================================================================
# Create Combined Chart
# ============================================================================

def create_combined_chart(tech_db, budget_df, summary_df):
    """Create a single figure with both charts side by side"""
    
    # Create figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # ========================================================================
    # Left Chart: Technology Effectiveness
    # ========================================================================
    
    # Sort technologies by reduction rate
    tech_db_sorted = tech_db.sort_values('减排率', ascending=True)
    
    # Create color gradient from light to dark green
    colors_tech = plt.cm.YlGn(np.linspace(0.4, 0.9, len(tech_db_sorted)))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(tech_db_sorted))
    bars1 = ax1.barh(y_pos, tech_db_sorted['减排率'] * 100,
                     color=colors_tech, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # Set labels for left chart
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tech_db_sorted['技术名称'], fontsize=11)
    ax1.set_xlabel('减排率 (%)', fontsize=13, weight='bold')
    ax1.set_title('技术减排效果\nTechnology Effectiveness', fontsize=15, weight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, tech_db_sorted['减排率'] * 100)):
        ax1.text(value + 0.15, i, f'{value:.1f}%',
                va='center', ha='left', fontsize=10, weight='bold')
    
    # ========================================================================
    # Right Chart: Technology Investment vs Knowledge Transfer (One point per technology)
    # ========================================================================
    
    # Create scatter plot with one point per technology
    scatter = ax2.scatter(tech_db['单位成本'] / 1000,
                         tech_db['知识转移价值'],
                         s=tech_db['减排率'] * 1500,  # Bubble size based on reduction rate
                         c=tech_db['年度运营节省'],
                         cmap='RdYlBu_r',
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=1.5)
    
    # Add labels for each technology
    for idx, tech in tech_db.iterrows():
        ax2.annotate(tech['技术名称'],
                    (tech['单位成本'] / 1000, tech['知识转移价值']),
                    xytext=(8, 5),  # Offset the text slightly
                    textcoords='offset points',
                    fontsize=9,
                    weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                    ha='left')
    
    # Set labels for right chart
    ax2.set_xlabel('单位成本 (千USD)', fontsize=13, weight='bold')
    ax2.set_ylabel('知识转移价值', fontsize=13, weight='bold')
    ax2.set_title('技术成本 vs 知识价值\nCost vs Knowledge Value', fontsize=15, weight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('年度节省 (Annual Savings, USD)', fontsize=11, weight='bold', rotation=270, labelpad=20)
    
    # Add trend line using tech_db data
    costs = tech_db['单位成本'] / 1000
    knowledge_values = tech_db['知识转移价值']
    z = np.polyfit(costs, knowledge_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(costs.min(), costs.max(), 100)
    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2.5, label='趋势线')
    
    ax2.legend(fontsize=10, loc='lower right')
    
    # Overall title
    plt.suptitle('III区供应商技术效果与成本价值分析 (Zone III: Learning Zone)\nTechnology & Knowledge Initiative Analysis',
                 fontsize=17, weight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Add simple explanation text at the bottom, positioned lower to avoid overlap
    explanation_text = "注: 知识转移价值衡量技术在供应商间传播和共享最佳实践的能力，数值越高表示该技术越容易推广复制。年度节省指技术实施后每年可节约的运营成本。"
    fig.text(0.5, -0.01, explanation_text, ha='center', va='top', fontsize=9, 
             style='italic', color='dimgray')
    
    # Save figure
    output_path = 'C区_技术效果与成本分析_组合图.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Combined chart saved: {output_path}")
    print(f"✓ Scatter plot shows {len(tech_db)} technology data points with labels")
    
    return fig


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STRATEGY C - Technology Effectiveness & Cost Analysis Combined")
    print("="*80 + "\n")
    
    # Load data
    print("📁 Loading data...")
    tech_db, budget_df, summary_df = load_data()
    
    if tech_db is None:
        print("\n❌ Failed to load data. Please run C_strategy_ML_simulation.py first.")
        return
    
    print(f"  - {len(tech_db)} technologies")
    print(f"  - {len(budget_df)} supplier records")
    
    # Create combined chart
    print("\n📊 Creating combined visualization...")
    fig = create_combined_chart(tech_db, budget_df, summary_df)
    
    print("\n" + "="*80)
    print("✅ Combined visualization generated successfully!")
    print("="*80)
    
    # Show insights
    print("\n📊 ANALYSIS INSIGHTS:")
    
    print("\n🔧 Technology Effectiveness:")
    top_tech = tech_db.nlargest(3, '减排率')
    for idx, tech in top_tech.iterrows():
        print(f"  • {tech['技术名称']}: {tech['减排率']*100:.1f}% reduction")
    
    print(f"\n💰 Cost Efficiency:")
    tech_db['cost_efficiency'] = tech_db['减排率'] / (tech_db['单位成本'] / 1000)
    top_efficient = tech_db.nlargest(3, 'cost_efficiency')
    for idx, tech in top_efficient.iterrows():
        print(f"  • {tech['技术名称']}: {tech['cost_efficiency']:.4f} reduction/千USD")
    
    print(f"\n🧠 Knowledge Transfer Value:")
    top_knowledge = tech_db.nlargest(3, '知识转移价值')
    for idx, tech in top_knowledge.iterrows():
        print(f"  • {tech['技术名称']}: {tech['知识转移价值']:.2f}")
    
    plt.show()
    
    print("\n✨ Chart saved: C区_技术效果与成本分析_组合图.png")


if __name__ == "__main__":
    main()
