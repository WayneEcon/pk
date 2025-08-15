#!/usr/bin/env python3
"""
专业网络可视化样式系统
====================

为骨干网络分析提供学术发表级别的可视化样式标准，
确保图表满足顶级期刊的美学和信息传达要求。

核心设计原则：
1. 信息密度最大化 - 在有限空间内展示最多有用信息
2. 视觉层次清晰 - 重要信息突出，次要信息适当弱化  
3. 色彩科学应用 - 遵循色彩心理学和无障碍设计原则
4. 学术规范遵循 - 符合国际顶级期刊可视化标准

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkTheme:
    """网络可视化主题配置"""
    # 基础配色方案 - 确保色盲友好
    primary_color: str = '#2E86AB'      # 主色调：专业蓝
    secondary_color: str = '#A23B72'    # 次色调：深玫红
    accent_color: str = '#F18F01'       # 强调色：橙黄
    background_color: str = '#FFFFFF'   # 背景色：纯白
    text_color: str = '#2C3E50'         # 文字色：深灰蓝
    
    # 节点配色方案 - 地理区域分类
    region_colors: Dict[str, str] = None
    
    # 字体设置 - 学术标准
    title_font_size: int = 16
    label_font_size: int = 10
    legend_font_size: int = 9
    annotation_font_size: int = 8
    
    # 图形参数
    node_size_range: Tuple[int, int] = (50, 800)
    edge_width_range: Tuple[float, float] = (0.5, 6.0)
    edge_alpha: float = 0.6
    node_alpha: float = 0.8
    
    def __post_init__(self):
        if self.region_colors is None:
            self.region_colors = {
                'North America': '#1f77b4',    # 蓝色 - 美国、加拿大、墨西哥
                'Europe': '#ff7f0e',           # 橙色 - 欧盟国家
                'Asia': '#2ca02c',             # 绿色 - 中国、日本、韩国等
                'Middle East': '#d62728',      # 红色 - 沙特、阿联酋等
                'Latin America': '#9467bd',    # 紫色 - 巴西、委内瑞拉等
                'Africa': '#8c564b',           # 棕色 - 尼日利亚、安哥拉等
                'Oceania': '#e377c2',          # 粉色 - 澳大利亚等
                'Other': '#7f7f7f'             # 灰色 - 其他/未分类
            }

class ProfessionalNetworkStyling:
    """专业网络可视化样式系统"""
    
    # 扩展的国家-地理区域映射
    COUNTRY_TO_REGION = {
        # 北美
        'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
        
        # 欧洲主要国家
        'GBR': 'Europe', 'DEU': 'Europe', 'FRA': 'Europe', 'ITA': 'Europe', 
        'ESP': 'Europe', 'NLD': 'Europe', 'BEL': 'Europe', 'NOR': 'Europe', 
        'SWE': 'Europe', 'DNK': 'Europe', 'FIN': 'Europe', 'POL': 'Europe',
        'CZE': 'Europe', 'AUT': 'Europe', 'CHE': 'Europe', 'IRL': 'Europe',
        'PRT': 'Europe', 'GRC': 'Europe', 'HUN': 'Europe', 'SVK': 'Europe',
        'SVN': 'Europe', 'EST': 'Europe', 'LVA': 'Europe', 'LTU': 'Europe',
        'RUS': 'Europe',  # 俄罗斯归类为欧洲
        
        # 亚洲主要国家
        'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia', 'IND': 'Asia', 
        'SGP': 'Asia', 'THA': 'Asia', 'MYS': 'Asia', 'IDN': 'Asia',
        'PHL': 'Asia', 'VNM': 'Asia', 'PAK': 'Asia', 'BGD': 'Asia',
        'LKA': 'Asia', 'MMR': 'Asia', 'KHM': 'Asia', 'LAO': 'Asia',
        'MNG': 'Asia', 'NPL': 'Asia', 'BTN': 'Asia', 'KAZ': 'Asia',
        'UZB': 'Asia', 'TKM': 'Asia', 'KGZ': 'Asia', 'TJK': 'Asia',
        
        # 中东
        'SAU': 'Middle East', 'ARE': 'Middle East', 'QAT': 'Middle East', 
        'KWT': 'Middle East', 'BHR': 'Middle East', 'OMN': 'Middle East',
        'IRN': 'Middle East', 'IRQ': 'Middle East', 'ISR': 'Middle East',
        'JOR': 'Middle East', 'LBN': 'Middle East', 'SYR': 'Middle East',
        'TUR': 'Middle East', 'YEM': 'Middle East', 'AFG': 'Middle East',
        
        # 拉丁美洲
        'BRA': 'Latin America', 'VEN': 'Latin America', 'COL': 'Latin America', 
        'ARG': 'Latin America', 'CHL': 'Latin America', 'PER': 'Latin America',
        'ECU': 'Latin America', 'BOL': 'Latin America', 'PRY': 'Latin America',
        'URY': 'Latin America', 'GUY': 'Latin America', 'SUR': 'Latin America',
        'GTM': 'Latin America', 'BLZ': 'Latin America', 'SLV': 'Latin America',
        'HND': 'Latin America', 'NIC': 'Latin America', 'CRI': 'Latin America',
        'PAN': 'Latin America', 'CUB': 'Latin America', 'DOM': 'Latin America',
        'HTI': 'Latin America', 'JAM': 'Latin America', 'TTO': 'Latin America',
        
        # 非洲
        'NGA': 'Africa', 'AGO': 'Africa', 'LBY': 'Africa', 'DZA': 'Africa',
        'EGY': 'Africa', 'ZAF': 'Africa', 'MAR': 'Africa', 'TUN': 'Africa',
        'GHA': 'Africa', 'CIV': 'Africa', 'KEN': 'Africa', 'ETH': 'Africa',
        'TZA': 'Africa', 'UGA': 'Africa', 'MOZ': 'Africa', 'MDG': 'Africa',
        'CMR': 'Africa', 'SEN': 'Africa', 'MLI': 'Africa', 'BFA': 'Africa',
        'NER': 'Africa', 'TCD': 'Africa', 'SUD': 'Africa', 'SSD': 'Africa',
        
        # 大洋洲
        'AUS': 'Oceania', 'NZL': 'Oceania', 'PNG': 'Oceania', 'FJI': 'Oceania',
        'NCL': 'Oceania', 'VUT': 'Oceania', 'SLB': 'Oceania', 'TON': 'Oceania'
    }
    
    def __init__(self, theme: NetworkTheme = None):
        """
        初始化专业网络样式系统
        
        Args:
            theme: 网络主题配置
        """
        self.theme = theme or NetworkTheme()
        self.setup_matplotlib_style()
        
        logger.info("🎨 专业网络样式系统初始化完成")
    
    def setup_matplotlib_style(self):
        """设置matplotlib的专业样式"""
        # 设置高质量输出参数
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': self.theme.background_color,
            'figure.facecolor': self.theme.background_color,
            
            # 字体设置
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'font.size': 10,
            'axes.titlesize': self.theme.title_font_size,
            'axes.labelsize': self.theme.label_font_size,
            'legend.fontsize': self.theme.legend_font_size,
            
            # 颜色和样式
            'text.color': self.theme.text_color,
            'axes.edgecolor': self.theme.text_color,
            'axes.labelcolor': self.theme.text_color,
            'xtick.color': self.theme.text_color,
            'ytick.color': self.theme.text_color,
            
            # 网格和背景
            'axes.facecolor': self.theme.background_color,
            'axes.grid': False,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def assign_node_colors(self, 
                          G: nx.Graph, 
                          color_scheme: str = 'geographic',
                          node_attributes: Dict = None) -> Dict[str, str]:
        """
        为节点分配专业级颜色方案
        
        Args:
            G: 网络图
            color_scheme: 着色方案 ('geographic', 'centrality', 'trade_volume')
            node_attributes: 节点属性字典
            
        Returns:
            节点颜色映射字典
        """
        
        logger.info(f"🎨 分配节点颜色 (方案: {color_scheme})...")
        
        node_colors = {}
        
        if color_scheme == 'geographic':
            # 基于地理区域的专业配色
            for node in G.nodes():
                region = self.COUNTRY_TO_REGION.get(node, 'Other')
                node_colors[node] = self.theme.region_colors[region]
            
            # 美国使用特殊突出色
            if 'USA' in node_colors:
                node_colors['USA'] = '#FF4136'  # 鲜红色突出美国
                
        elif color_scheme == 'trade_volume' and node_attributes:
            # 基于贸易量的渐变配色
            if 'total_strength' in node_attributes:
                strengths = {node: node_attributes['total_strength'].get(node, 0) 
                           for node in G.nodes()}
                
                # 标准化到[0,1]
                min_strength = min(strengths.values())
                max_strength = max(strengths.values())
                
                # 使用专业的蓝色渐变
                cmap = plt.cm.Blues
                
                for node in G.nodes():
                    if max_strength > min_strength:
                        norm_val = (strengths[node] - min_strength) / (max_strength - min_strength)
                    else:
                        norm_val = 0.5
                    # 避免太浅的颜色
                    norm_val = 0.3 + 0.7 * norm_val
                    node_colors[node] = cmap(norm_val)
                    
        elif color_scheme == 'centrality' and node_attributes:
            # 基于中心性的配色
            if 'pagerank' in node_attributes:
                centralities = {node: node_attributes['pagerank'].get(node, 0) 
                              for node in G.nodes()}
                
                min_cent = min(centralities.values())
                max_cent = max(centralities.values())
                
                # 使用橙色渐变表示中心性
                cmap = plt.cm.Oranges
                
                for node in G.nodes():
                    if max_cent > min_cent:
                        norm_val = (centralities[node] - min_cent) / (max_cent - min_cent)
                    else:
                        norm_val = 0.5
                    norm_val = 0.3 + 0.7 * norm_val
                    node_colors[node] = cmap(norm_val)
        
        else:
            # 默认专业单色方案
            for node in G.nodes():
                node_colors[node] = self.theme.primary_color
        
        logger.info(f"✅ 节点着色完成 ({len(set(node_colors.values()))} 种颜色)")
        return node_colors
    
    def calculate_node_sizes(self, 
                           G: nx.Graph,
                           full_network_G: nx.Graph = None,
                           node_attributes: Dict = None,
                           size_attribute: str = 'total_strength') -> Dict[str, float]:
        """
        计算专业级节点大小
        
        Args:
            G: 骨干网络图
            full_network_G: 完整网络图（用于获取真实属性）
            node_attributes: 节点属性字典
            size_attribute: 大小依据属性
            
        Returns:
            节点大小映射字典
        """
        
        logger.info(f"📏 计算节点大小 (属性: {size_attribute})...")
        
        node_sizes = {}
        min_size, max_size = self.theme.node_size_range
        
        # 优先使用完整网络的数据确保信息保真
        if full_network_G is not None:
            if size_attribute == 'total_strength':
                # 使用完整网络的节点强度
                strengths = {node: full_network_G.degree(node, weight='weight') 
                           for node in G.nodes() if node in full_network_G.nodes()}
            else:
                # 使用完整网络的度数
                strengths = {node: full_network_G.degree(node) 
                           for node in G.nodes() if node in full_network_G.nodes()}
        
        elif node_attributes and size_attribute in node_attributes:
            # 使用提供的节点属性
            strengths = {node: node_attributes[size_attribute].get(node, 0) 
                        for node in G.nodes()}
        
        else:
            # 回退到骨干网络自身的属性
            if size_attribute == 'total_strength':
                strengths = dict(G.degree(weight='weight'))
            else:
                strengths = dict(G.degree())
        
        # 标准化节点大小
        if strengths:
            values = list(strengths.values())
            min_strength = min(values)
            max_strength = max(values)
            
            for node in G.nodes():
                strength = strengths.get(node, min_strength)
                if max_strength > min_strength:
                    norm_val = (strength - min_strength) / (max_strength - min_strength)
                else:
                    norm_val = 0.5
                
                # 应用平方根变换，避免极端大小差异
                norm_val = np.sqrt(norm_val)
                node_sizes[node] = min_size + norm_val * (max_size - min_size)
        
        else:
            # 默认统一大小
            for node in G.nodes():
                node_sizes[node] = (min_size + max_size) / 2
        
        # 美国节点特殊处理
        if 'USA' in node_sizes:
            node_sizes['USA'] = max(node_sizes['USA'] * 1.3, max_size * 1.1)
        
        logger.info(f"✅ 节点大小计算完成")
        return node_sizes
    
    def calculate_edge_widths(self, G: nx.Graph) -> List[float]:
        """
        计算专业级边宽度
        
        Args:
            G: 网络图
            
        Returns:
            边宽度列表
        """
        
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        
        if not edge_weights:
            return []
        
        min_width, max_width = self.theme.edge_width_range
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        if max_weight > min_weight:
            # 使用对数缩放处理极端权重差异
            log_weights = np.log1p(np.array(edge_weights) - min_weight)
            max_log = np.log1p(max_weight - min_weight)
            
            normalized_weights = log_weights / max_log
            edge_widths = min_width + normalized_weights * (max_width - min_width)
        else:
            edge_widths = [min_width] * len(edge_weights)
        
        return edge_widths.tolist()
    
    def create_intelligent_labels(self, 
                                G: nx.Graph, 
                                pos: Dict,
                                node_sizes: Dict,
                                centrality_data: Dict = None,
                                max_labels: int = 20) -> Dict[str, str]:
        """
        创建智能标签布局，避免重叠
        
        Args:
            G: 网络图
            pos: 节点位置
            node_sizes: 节点大小
            centrality_data: 中心性数据
            max_labels: 最大标签数量
            
        Returns:
            标签字典
        """
        
        logger.info(f"🏷️ 创建智能标签布局 (最多{max_labels}个)...")
        
        # 计算节点重要性排序
        importance_scores = {}
        
        for node in G.nodes():
            score = 0
            
            # 基于度数的重要性
            score += G.degree(node, weight='weight') / 1e9  # 标准化到十亿
            
            # 基于节点大小的重要性
            score += node_sizes.get(node, 0) / 1000
            
            # 基于中心性数据的重要性（如果可用）
            if centrality_data and 'pagerank' in centrality_data:
                score += centrality_data['pagerank'].get(node, 0) * 10000
            
            # 美国特殊加权
            if node == 'USA':
                score *= 3
            
            importance_scores[node] = score
        
        # 选择最重要的节点进行标签显示
        top_nodes = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = top_nodes[:max_labels]
        
        # 检查标签位置冲突并调整
        labels = {}
        label_positions = []
        
        for node, importance in top_nodes:
            if node in pos:
                node_pos = pos[node]
                node_size = node_sizes.get(node, 50)
                
                # 检查与已有标签的距离
                min_distance = (node_size / 100) * 0.5  # 基于节点大小的最小距离
                
                conflict = False
                for existing_pos in label_positions:
                    distance = np.sqrt((node_pos[0] - existing_pos[0])**2 + 
                                     (node_pos[1] - existing_pos[1])**2)
                    if distance < min_distance:
                        conflict = True
                        break
                
                if not conflict or node == 'USA':  # 美国标签总是显示
                    labels[node] = node
                    label_positions.append(node_pos)
        
        logger.info(f"✅ 标签布局完成 ({len(labels)}个标签)")
        return labels
    
    def create_legend(self, 
                     fig: plt.Figure, 
                     color_scheme: str,
                     additional_info: Dict = None) -> None:
        """
        创建专业级图例
        
        Args:
            fig: matplotlib图形对象
            color_scheme: 使用的配色方案
            additional_info: 额外信息
        """
        
        if color_scheme == 'geographic':
            # 地理区域图例
            legend_elements = []
            for region, color in self.theme.region_colors.items():
                if region != 'Other':  # 跳过"其他"类别
                    patch = patches.Patch(color=color, label=region)
                    legend_elements.append(patch)
            
            # 添加美国特殊标记
            usa_patch = patches.Patch(color='#FF4136', label='USA (highlighted)')
            legend_elements.append(usa_patch)
            
            legend = fig.legend(legend_elements, [elem.get_label() for elem in legend_elements],
                              loc='upper left', bbox_to_anchor=(0.02, 0.98),
                              frameon=True, fancybox=True, shadow=True,
                              fontsize=self.theme.legend_font_size)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
    
    def add_professional_annotations(self, 
                                   ax: plt.Axes,
                                   G: nx.Graph,
                                   full_network_G: nx.Graph = None,
                                   algorithm_name: str = "Backbone") -> None:
        """
        添加专业级注释和统计信息
        
        Args:
            ax: matplotlib轴对象
            G: 骨干网络图
            full_network_G: 完整网络图
            algorithm_name: 算法名称
        """
        
        # 计算网络统计信息
        stats_text = []
        stats_text.append(f"Algorithm: {algorithm_name}")
        stats_text.append(f"Nodes: {G.number_of_nodes():,}")
        stats_text.append(f"Edges: {G.number_of_edges():,}")
        
        if full_network_G is not None:
            retention_rate = G.number_of_edges() / full_network_G.number_of_edges()
            stats_text.append(f"Retention: {retention_rate:.1%}")
        
        # 添加美国特殊统计
        if 'USA' in G.nodes():
            usa_degree = G.degree('USA')
            usa_strength = G.degree('USA', weight='weight')
            stats_text.append(f"USA Connections: {usa_degree}")
            stats_text.append(f"USA Trade Volume: ${usa_strength/1e9:.1f}B")
        
        # 创建专业样式的信息框
        info_text = '\n'.join(stats_text)
        
        # 使用专业样式的文本框
        bbox_props = dict(boxstyle="round,pad=0.5", 
                         facecolor='white', 
                         edgecolor=self.theme.text_color,
                         alpha=0.9)
        
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes,
               fontsize=self.theme.annotation_font_size,
               verticalalignment='top',
               bbox=bbox_props,
               family='monospace')  # 使用等宽字体确保对齐

    def apply_professional_layout_algorithm(self, 
                                          G: nx.Graph, 
                                          algorithm: str = 'force_atlas2',
                                          seed: int = 42) -> Dict[str, Tuple[float, float]]:
        """
        应用专业级布局算法
        
        Args:
            G: 网络图
            algorithm: 布局算法名称
            seed: 随机种子
            
        Returns:
            节点位置字典
        """
        
        logger.info(f"📐 应用{algorithm}布局算法...")
        
        if algorithm == 'force_atlas2' or algorithm == 'spring':
            # 高质量弹簧布局（Force Atlas 2近似）
            pos = nx.spring_layout(
                G,
                k=3.0,  # 增加节点间距
                iterations=100,  # 增加迭代次数提高质量
                weight='weight',
                seed=seed
            )
            
        elif algorithm == 'fruchterman_reingold':
            # Fruchterman-Reingold布局
            pos = nx.fruchterman_reingold_layout(
                G,
                k=2.0,
                iterations=100,
                weight='weight',
                seed=seed
            )
            
        elif algorithm == 'kamada_kawai':
            # Kamada-Kawai布局（适合较小网络）
            if G.number_of_nodes() <= 100:
                pos = nx.kamada_kawai_layout(G, weight='weight')
            else:
                # 回退到spring布局
                pos = nx.spring_layout(G, k=3.0, iterations=100, weight='weight', seed=seed)
                
        else:
            # 默认使用优化的spring布局
            pos = nx.spring_layout(G, k=3.0, iterations=100, weight='weight', seed=seed)
        
        # 后处理：优化节点位置，减少重叠
        pos = self._optimize_node_positions(G, pos)
        
        logger.info("✅ 布局计算完成")
        return pos
    
    def _optimize_node_positions(self, 
                               G: nx.Graph, 
                               pos: Dict,
                               min_distance: float = 0.1) -> Dict:
        """
        优化节点位置，减少重叠
        
        Args:
            G: 网络图
            pos: 原始位置
            min_distance: 最小距离
            
        Returns:
            优化后的位置
        """
        
        nodes = list(G.nodes())
        positions = np.array([pos[node] for node in nodes])
        
        # 计算距离矩阵
        distances = squareform(pdist(positions))
        
        # 对距离过近的节点进行调整
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if distances[i, j] < min_distance:
                    # 计算调整向量
                    diff = positions[i] - positions[j]
                    norm = np.linalg.norm(diff)
                    
                    if norm > 0:
                        # 标准化并调整距离
                        unit_vector = diff / norm
                        adjustment = unit_vector * (min_distance - norm) / 2
                        
                        positions[i] += adjustment
                        positions[j] -= adjustment
        
        # 更新位置字典
        optimized_pos = {}
        for i, node in enumerate(nodes):
            optimized_pos[node] = tuple(positions[i])
        
        return optimized_pos

if __name__ == "__main__":
    # 测试专业样式系统
    logger.info("🧪 测试专业网络样式系统...")
    
    # 创建测试网络
    G_test = nx.Graph()
    countries = ['USA', 'CAN', 'MEX', 'GBR', 'DEU', 'CHN', 'JPN', 'SAU', 'BRA']
    
    # 添加边
    edges = [
        ('USA', 'CAN', 1000), ('USA', 'MEX', 800), ('USA', 'GBR', 600),
        ('USA', 'SAU', 500), ('GBR', 'DEU', 400), ('CHN', 'JPN', 300),
        ('CHN', 'KOR', 250), ('BRA', 'ARG', 200)
    ]
    
    for source, target, weight in edges:
        G_test.add_edge(source, target, weight=weight)
    
    # 初始化样式系统
    styling = ProfessionalNetworkStyling()
    
    # 测试各种功能
    colors = styling.assign_node_colors(G_test, 'geographic')
    sizes = styling.calculate_node_sizes(G_test)
    pos = styling.apply_professional_layout_algorithm(G_test)
    labels = styling.create_intelligent_labels(G_test, pos, sizes)
    
    print("🎉 专业样式系统测试完成!")
    print(f"颜色方案: {len(set(colors.values()))} 种颜色")
    print(f"节点大小范围: {min(sizes.values()):.1f} - {max(sizes.values()):.1f}")
    print(f"标签数量: {len(labels)}")
    print(f"重要节点: {list(labels.keys())}")