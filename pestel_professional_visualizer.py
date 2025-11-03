"""
PESTEL Analysis Professional Visualization
专业PESTEL分析可视化工具
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Circle
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PESTELVisualizer:
    """专业PESTEL分析可视化类"""
    
    def __init__(self):
        """初始化PESTEL分析数据"""
        self.sectors = {
            'Political': {
                'label_cn': '政治',
                'label_en': 'Political',
                'content': '欧盟《碳边境调节机制》\n《企业可持续报告指令》\n实施',
                'color': '#2E8B57',
                'position': 1
            },
            'Economic': {
                'label_cn': '经济',
                'label_en': 'Economic',
                'content': '绿色金融与碳\n税制度强化强\n化ESG门槛',
                'color': '#7CB342',
                'position': 2
            },
            'Social': {
                'label_cn': '社会',
                'label_en': 'Social',
                'content': '消费者偏好具有\n可持续理念的\n品牌',
                'color': '#42A5F5',
                'position': 3
            },
            'Technological': {
                'label_cn': '技术',
                'label_en': 'Technological',
                'content': '碳监测、AI与区块\n链追溯技术成熟',
                'color': '#1976D2',
                'position': 4
            },
            'Environmental': {
                'label_cn': '环境',
                'label_en': 'Environmental',
                'content': '气候极端事件频\n发，全球净零承\n诺逐步强化',
                'color': '#8D6E63',
                'position': 5
            },
            'Legal': {
                'label_cn': '法律',
                'label_en': 'Legal',
                'content': '香港交易所\nESG披露规则\n升级',
                'color': '#F57C00',
                'position': 6
            }
        }
        
    def create_professional_pestel(self, output_file='pestel_professional_analysis.png'):
        """创建专业的PESTEL分析图"""
        
        # 创建图形（增大画布以提供更多边距）
        fig, ax = plt.subplots(figsize=(18, 16), facecolor='white')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 设置中心点和半径（缩小圆圈以使文字更清晰）
        center_x, center_y = 0.5, 0.5
        outer_radius = 0.35
        inner_radius = 0.13
        mid_radius = 0.09
        
        # 计算每个扇形的角度
        n_sectors = 6
        angle_per_sector = 360 / n_sectors
        
        # 绘制外圈扇形（主要内容区域）
        for key, sector in self.sectors.items():
            position = sector['position']
            start_angle = 90 - (position - 1) * angle_per_sector
            end_angle = start_angle - angle_per_sector
            
            # 绘制外圈扇形
            wedge_outer = Wedge(
                (center_x, center_y), 
                outer_radius,
                end_angle,
                start_angle,
                facecolor='white',
                edgecolor=sector['color'],
                linewidth=3,
                alpha=1.0
            )
            ax.add_patch(wedge_outer)
            
            # 绘制中圈扇形（编号区域）
            wedge_mid = Wedge(
                (center_x, center_y), 
                inner_radius,
                end_angle,
                start_angle,
                facecolor=sector['color'],
                edgecolor='white',
                linewidth=2,
                alpha=0.85
            )
            ax.add_patch(wedge_mid)
            
            # 计算标题位置（弧形外侧）
            title_angle = np.radians(start_angle - angle_per_sector / 2)
            title_radius = outer_radius + 0.08
            title_x = center_x + title_radius * np.cos(title_angle)
            title_y = center_y + title_radius * np.sin(title_angle)
            
            # 添加英文标题
            ax.text(
                title_x, title_y,
                sector['label_en'],
                ha='center', va='center',
                fontsize=26,
                fontweight='bold',
                color=sector['color'],
                family='Arial'
            )
            
            # 计算内容位置（扇形中间）
            content_angle = np.radians(start_angle - angle_per_sector / 2)
            content_radius = (outer_radius + inner_radius) / 2
            content_x = center_x + content_radius * np.cos(content_angle)
            content_y = center_y + content_radius * np.sin(content_angle)
            
            # 添加中文内容（大幅放大字体，加粗关键词）
            ax.text(
                content_x, content_y,
                sector['content'],
                ha='center', va='center',
                fontsize=18,
                color='#2C3E50',
                linespacing=1.6,
                weight='bold'
            )
            
            # 在中圈添加编号（更大字体）
            number_x = center_x + mid_radius * np.cos(content_angle)
            number_y = center_y + mid_radius * np.sin(content_angle)
            
            ax.text(
                number_x, number_y,
                str(position),
                ha='center', va='center',
                fontsize=24,
                fontweight='bold',
                color='white'
            )
        
        # 绘制中心圆
        center_circle = Circle(
            (center_x, center_y),
            inner_radius,
            facecolor='white',
            edgecolor='#34495E',
            linewidth=3,
            zorder=10
        )
        ax.add_patch(center_circle)
        
        # 添加中心文字（更大字体）
        ax.text(
            center_x, center_y,
            'PESTEL',
            ha='center', va='center',
            fontsize=32,
            fontweight='bold',
            color='#34495E',
            family='Arial',
            zorder=11
        )
        
        # 设置图形范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 添加标题（更靠近图表）
        plt.suptitle(
            'PESTEL 宏观环境分析框架',
            fontsize=26,
            fontweight='bold',
            y=0.94,
            color='#2C3E50'
        )
        
        plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)
        
        # 保存图形
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
        print(f"✓ PESTEL专业分析图已保存至: {output_file}")
        
        plt.close()
        
    def create_enhanced_pestel(self, output_file='pestel_enhanced_analysis.png'):
        """创建增强版PESTEL分析图（带渐变效果）"""
        
        # 增大画布以提供更多边距
        fig, ax = plt.subplots(figsize=(20, 18), facecolor='white')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 缩小圆圈以使文字更清晰
        center_x, center_y = 0.5, 0.5
        outer_radius = 0.35
        inner_radius = 0.13
        mid_radius = 0.09
        
        n_sectors = 6
        angle_per_sector = 360 / n_sectors
        
        # 绘制外圈扇形
        for key, sector in self.sectors.items():
            position = sector['position']
            start_angle = 90 - (position - 1) * angle_per_sector
            end_angle = start_angle - angle_per_sector
            
            # 绘制带阴影的外圈扇形
            wedge_outer = Wedge(
                (center_x, center_y), 
                outer_radius,
                end_angle,
                start_angle,
                facecolor='white',
                edgecolor=sector['color'],
                linewidth=4,
                alpha=1.0
            )
            ax.add_patch(wedge_outer)
            
            # 添加浅色填充区域
            wedge_fill = Wedge(
                (center_x, center_y), 
                outer_radius * 0.95,
                end_angle + 1,
                start_angle - 1,
                facecolor=sector['color'],
                edgecolor='none',
                linewidth=0,
                alpha=0.08
            )
            ax.add_patch(wedge_fill)
            
            # 绘制中圈扇形（编号区域）
            wedge_mid = Wedge(
                (center_x, center_y), 
                inner_radius,
                end_angle,
                start_angle,
                facecolor=sector['color'],
                edgecolor='white',
                linewidth=3,
                alpha=0.90
            )
            ax.add_patch(wedge_mid)
            
            # 计算标题位置
            title_angle = np.radians(start_angle - angle_per_sector / 2)
            title_radius = outer_radius + 0.09
            title_x = center_x + title_radius * np.cos(title_angle)
            title_y = center_y + title_radius * np.sin(title_angle)
            
            # 添加英文标题（带背景框）
            bbox_props = dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor=sector['color'],
                linewidth=2,
                alpha=0.95
            )
            
            ax.text(
                title_x, title_y,
                sector['label_en'],
                ha='center', va='center',
                fontsize=28,
                fontweight='bold',
                color=sector['color'],
                family='Arial',
                bbox=bbox_props
            )
            
            # 计算内容位置
            content_angle = np.radians(start_angle - angle_per_sector / 2)
            content_radius = (outer_radius + inner_radius) / 2
            content_x = center_x + content_radius * np.cos(content_angle)
            content_y = center_y + content_radius * np.sin(content_angle)
            
            # 添加中文内容（大幅放大字体，加粗）
            ax.text(
                content_x, content_y,
                sector['content'],
                ha='center', va='center',
                fontsize=19,
                color='#1A1A1A',
                linespacing=1.7,
                weight='bold'
            )
            
            # 在中圈添加编号（更大字体）
            number_x = center_x + mid_radius * np.cos(content_angle)
            number_y = center_y + mid_radius * np.sin(content_angle)
            
            ax.text(
                number_x, number_y,
                str(position),
                ha='center', va='center',
                fontsize=26,
                fontweight='bold',
                color='white'
            )
        
        # 绘制中心圆（多层效果）
        center_circle_outer = Circle(
            (center_x, center_y),
            inner_radius + 0.01,
            facecolor='#34495E',
            edgecolor='none',
            linewidth=0,
            zorder=10,
            alpha=0.2
        )
        ax.add_patch(center_circle_outer)
        
        center_circle = Circle(
            (center_x, center_y),
            inner_radius,
            facecolor='white',
            edgecolor='#34495E',
            linewidth=4,
            zorder=11
        )
        ax.add_patch(center_circle)
        
        # 添加中心文字（更大字体）
        ax.text(
            center_x, center_y,
            'PESTEL',
            ha='center', va='center',
            fontsize=36,
            fontweight='bold',
            color='#34495E',
            family='Arial',
            zorder=12
        )
        
        # 设置图形范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 添加标题和副标题（更靠近图表）
        fig.text(
            0.5, 0.93,
            'PESTEL 宏观环境分析框架',
            ha='center',
            fontsize=28,
            fontweight='bold',
            color='#2C3E50'
        )
        
        fig.text(
            0.5, 0.90,
            'Strategic Macro-Environmental Analysis Framework',
            ha='center',
            fontsize=16,
            color='#7F8C8D',
            style='italic',
            family='Arial'
        )
        
        plt.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.08)
        
        # 保存图形
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
        print(f"✓ PESTEL增强版分析图已保存至: {output_file}")
        
        plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("PESTEL 专业分析可视化工具")
    print("Professional PESTEL Analysis Visualizer")
    print("=" * 60)
    
    # 创建可视化对象
    visualizer = PESTELVisualizer()
    
    # 生成专业版PESTEL图
    print("\n正在生成专业版PESTEL分析图...")
    visualizer.create_professional_pestel()
    
    # 生成增强版PESTEL图
    print("\n正在生成增强版PESTEL分析图...")
    visualizer.create_enhanced_pestel()
    
    print("\n" + "=" * 60)
    print("✓ 所有PESTEL分析图生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
