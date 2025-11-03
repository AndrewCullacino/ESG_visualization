"""
Strategy C - Technology & Knowledge Combined Visualization
===========================================================
Creates a single combined chart with:
1. Technology Effectiveness (left)
2. Knowledge Initiative Impact (right)
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
        knowledge_db = pd.read_csv('ML_simulation_C_知识共享数据库.csv', encoding='utf-8-sig')
        print("✓ Data loaded successfully")
        return tech_db, knowledge_db
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None


# ============================================================================
# Create Combined Chart
# ============================================================================

def create_combined_chart(tech_db, knowledge_db):
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
    # Right Chart: Knowledge Initiative Impact
    # ========================================================================
    
    # Prepare data
    x_pos = np.arange(len(knowledge_db))
    width = 0.35
    
    # Create grouped bars
    bars2a = ax2.bar(x_pos - width/2, knowledge_db['知识倍增器'],
                     width, label='知识倍增器', color='steelblue', 
                     alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2b = ax2.bar(x_pos + width/2, knowledge_db['网络效应'],
                     width, label='网络效应', color='orange', 
                     alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # Set labels for right chart
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(knowledge_db['倡议名称'], rotation=30, ha='right', fontsize=10)
    ax2.set_ylabel('效果值', fontsize=13, weight='bold')
    ax2.set_title('知识共享倡议效果\nKnowledge Initiative Impact', fontsize=15, weight='bold', pad=20)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Overall title
    plt.suptitle('C区技术与知识共享效果分析 (Zone III: Learning Zone)\nTechnology & Knowledge Sharing Impact Analysis',
                 fontsize=17, weight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'C区_技术与知识共享_组合图.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Combined chart saved: {output_path}")
    
    return fig


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STRATEGY C - Technology & Knowledge Combined Visualization")
    print("="*80 + "\n")
    
    # Load data
    print("📁 Loading data...")
    tech_db, knowledge_db = load_data()
    
    if tech_db is None or knowledge_db is None:
        print("\n❌ Failed to load data. Please run C_strategy_ML_simulation.py first.")
        return
    
    print(f"  - {len(tech_db)} technologies")
    print(f"  - {len(knowledge_db)} knowledge initiatives")
    
    # Create combined chart
    print("\n📊 Creating combined visualization...")
    fig = create_combined_chart(tech_db, knowledge_db)
    
    print("\n" + "="*80)
    print("✅ Combined visualization generated successfully!")
    print("="*80)
    
    # Show insights
    print("\n🔧 TECHNOLOGY INSIGHTS:")
    top_tech = tech_db.nlargest(3, '减排率')
    print(f"  Top 3 Most Effective Technologies:")
    for idx, tech in top_tech.iterrows():
        print(f"    • {tech['技术名称']}: {tech['减排率']*100:.1f}% reduction")
    
    print(f"\n🧠 KNOWLEDGE INITIATIVE INSIGHTS:")
    top_initiative = knowledge_db.nlargest(1, '知识倍增器')
    print(f"  Highest Knowledge Multiplier:")
    print(f"    • {top_initiative.iloc[0]['倡议名称']}: {top_initiative.iloc[0]['知识倍增器']:.2f}x")
    
    top_network = knowledge_db.nlargest(1, '网络效应')
    print(f"  Highest Network Effect:")
    print(f"    • {top_network.iloc[0]['倡议名称']}: {top_network.iloc[0]['网络效应']:.2f}")
    
    plt.show()
    
    print("\n✨ Chart saved: C区_技术与知识共享_组合图.png")


if __name__ == "__main__":
    main()
