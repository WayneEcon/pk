#!/usr/bin/env python3
"""
骨干网络可视化模块
================

专门用于骨干网络的高质量可视化，解决完整网络"毛球"图的问题，
提供清晰直观的网络结构展示。

核心功能：
1. Force Atlas 2布局优化
2. 多种节点着色方案（地理、中心性、贸易量）
3. 美国中心地位的突出显示
4. 时间序列对比可视化

设计原则：
- 学术出版级别的图表质量
- 政策制定者友好的视觉传达
- 跨算法结果的一致性展示

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# 设置可视化环境
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style("whitegrid")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 地理区域配色方案
GEOGRAPHIC_COLORS = {
    'North America': '#1f77b4',      # 蓝色 - 美国、加拿大、墨西哥
    'Europe': '#ff7f0e',             # 橙色 - 欧盟等
    'Asia': '#2ca02c',               # 绿色 - 中国、日本、韩国等
    'Middle East': '#d62728',        # 红色 - 沙特、阿联酋等
    'Latin America': '#9467bd',      # 紫色 - 巴西、委内瑞拉等
    'Africa': '#8c564b',             # 棕色 - 尼日利亚、安哥拉等
    'Oceania': '#e377c2',            # 粉色 - 澳大利亚等
    'Other': '#7f7f7f'               # 灰色 - 其他/未分类
}

# 国家到地理区域的映射（简化版，可扩展）
COUNTRY_TO_REGION = {
    'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
    'GBR': 'Europe', 'DEU': 'Europe', 'FRA': 'Europe', 'ITA': 'Europe', 'ESP': 'Europe',
    'NLD': 'Europe', 'BEL': 'Europe', 'NOR': 'Europe', 'SWE': 'Europe', 'DNK': 'Europe',
    'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia', 'IND': 'Asia', 'SGP': 'Asia',
    'SAU': 'Middle East', 'ARE': 'Middle East', 'QAT': 'Middle East', 'KWT': 'Middle East',
    'BRA': 'Latin America', 'VEN': 'Latin America', 'COL': 'Latin America', 'ARG': 'Latin America',
    'NGA': 'Africa', 'AGO': 'Africa', 'LBY': 'Africa', 'DZA': 'Africa',
    'AUS': 'Oceania', 'RUS': 'Europe'  # 俄罗斯归类为欧洲
}

def get_force_atlas_layout(G: nx.Graph, 
                          pos_seed: Dict = None,
                          iterations: int = 50,
                          k: float = None,
                          fixed_positions: Dict = None) -> Dict[str, Tuple[float, float]]:
    """
    使用Force Atlas 2风格的布局算法
    
    Args:
        G: 网络图
        pos_seed: 初始位置（用于一致性）
        iterations: 迭代次数
        k: 弹簧常数
        fixed_positions: 固定位置的节点
        
    Returns:
        节点位置字典
    """
    
    logger.info(f"🎨 计算Force Atlas布局 ({G.number_of_nodes()}节点, {iterations}次迭代)...")
    
    # 使用spring layout作为Force Atlas的近似
    # 可以根据需要替换为更精确的实现
    try:
        pos = nx.spring_layout(
            G, 
            pos=pos_seed,
            k=k,
            iterations=iterations,
            weight='weight',
            seed=42  # 固定随机种子保证可重现性
        )
        
        # 应用固定位置
        if fixed_positions:
            pos.update(fixed_positions)
        
        logger.info("✅ 布局计算完成")
        return pos
        
    except Exception as e:
        logger.error(f"❌ 布局计算失败: {e}")
        # 回退到简单布局
        return nx.circular_layout(G)

def assign_node_colors(G: nx.Graph, 
                      color_scheme: str = 'geographic',
                      centrality_data: pd.DataFrame = None,
                      country_metadata: pd.DataFrame = None,
                      original_network: nx.Graph = None) -> Dict[str, str]:
    """
    为节点分配颜色
    
    Args:
        G: 网络图
        color_scheme: 着色方案 ('geographic', 'centrality', 'trade_volume', 'community')
        centrality_data: 中心性数据
        country_metadata: 国家元数据
        
    Returns:
        节点颜色字典
    """
    
    logger.info(f"🎨 分配节点颜色 (方案: {color_scheme})...")
    
    node_colors = {}
    
    if color_scheme == 'geographic':
        # 基于地理区域着色
        for node in G.nodes():
            region = COUNTRY_TO_REGION.get(node, 'Other')
            node_colors[node] = GEOGRAPHIC_COLORS[region]
        
        # 特殊突出显示美国
        if 'USA' in node_colors:
            node_colors['USA'] = '#ff0000'  # 红色突出美国
    
    elif color_scheme == 'centrality':
        # 基于中心性着色（需要中心性数据）
        if centrality_data is not None:
            # 使用PageRank或Betweenness中心性
            if 'pagerank' in centrality_data.columns:
                centrality_col = 'pagerank'
            elif 'betweenness' in centrality_data.columns:
                centrality_col = 'betweenness'
            else:
                centrality_col = centrality_data.columns[0]  # 使用第一列
            
            # 标准化中心性值到[0,1]
            centrality_values = centrality_data[centrality_col]
            min_val, max_val = centrality_values.min(), centrality_values.max()
            
            # 使用colormap
            cmap = plt.cm.viridis
            
            for node in G.nodes():
                if node in centrality_data.index:
                    norm_value = (centrality_data.loc[node, centrality_col] - min_val) / (max_val - min_val)
                    node_colors[node] = cmap(norm_value)
                else:
                    node_colors[node] = '#cccccc'  # 灰色表示缺失数据
        else:
            logger.warning("⚠️ 缺少中心性数据，回退到地理着色")
            return assign_node_colors(G, 'geographic', centrality_data, country_metadata)
    
    elif color_scheme == 'trade_volume':
        # 基于贸易总量着色 - 优先使用原始网络的数据保证信息保真
        if original_network is not None:
            # 使用原始网络的贸易强度确保信息保真
            node_strengths = {node: original_network.degree(node, weight='weight') 
                            for node in G.nodes() if node in original_network.nodes()}
            # 补充骨干网络中不在原始网络的节点（理论上不应该出现）
            for node in G.nodes():
                if node not in node_strengths:
                    node_strengths[node] = G.degree(node, weight='weight')
        else:
            # 回退到骨干网络数据
            node_strengths = dict(G.degree(weight='weight'))
        
        if node_strengths:
            min_strength = min(node_strengths.values())
            max_strength = max(node_strengths.values())
            
            cmap = plt.cm.Blues
            
            for node in G.nodes():
                if max_strength > min_strength:
                    norm_value = (node_strengths[node] - min_strength) / (max_strength - min_strength)
                else:
                    norm_value = 0.5
                node_colors[node] = cmap(0.3 + 0.7 * norm_value)  # 避免太浅的颜色
        else:
            # 全部使用默认颜色
            for node in G.nodes():
                node_colors[node] = '#1f77b4'
    
    else:
        # 默认单色
        for node in G.nodes():
            node_colors[node] = '#1f77b4'
    
    logger.info(f"✅ 节点着色完成 ({len(set(node_colors.values()))} 种颜色)")
    return node_colors

def calculate_node_sizes(G: nx.Graph, 
                        size_attribute: str = 'strength',
                        size_range: Tuple[int, int] = (20, 200),
                        highlight_nodes: List[str] = ['USA'],
                        original_network: nx.Graph = None) -> Dict[str, float]:
    """
    计算节点大小
    
    Args:
        G: 网络图
        size_attribute: 大小依据 ('strength', 'degree', 'uniform')
        size_range: 大小范围 (最小值, 最大值)
        highlight_nodes: 需要突出显示的节点
        
    Returns:
        节点大小字典
    """
    
    logger.info(f"📏 计算节点大小 (属性: {size_attribute})...")
    
    node_sizes = {}
    min_size, max_size = size_range
    
    if size_attribute == 'strength':
        # 基于节点强度（加权度）- 优先使用原始网络数据保证信息保真
        if original_network is not None:
            # 使用原始网络的强度数据确保信息保真
            node_strengths = {node: original_network.degree(node, weight='weight') 
                            for node in G.nodes() if node in original_network.nodes()}
            # 补充骨干网络中不在原始网络的节点
            for node in G.nodes():
                if node not in node_strengths:
                    node_strengths[node] = G.degree(node, weight='weight')
        else:
            # 回退到骨干网络数据
            node_strengths = dict(G.degree(weight='weight'))
        
        min_strength = min(node_strengths.values()) if node_strengths else 0
        max_strength = max(node_strengths.values()) if node_strengths else 1
        
        for node in G.nodes():
            if max_strength > min_strength:
                norm_value = (node_strengths[node] - min_strength) / (max_strength - min_strength)
            else:
                norm_value = 0.5
            node_sizes[node] = min_size + norm_value * (max_size - min_size)
    
    elif size_attribute == 'degree':
        # 基于节点度数
        node_degrees = dict(G.degree())
        min_degree = min(node_degrees.values()) if node_degrees else 0
        max_degree = max(node_degrees.values()) if node_degrees else 1
        
        for node in G.nodes():
            if max_degree > min_degree:
                norm_value = (node_degrees[node] - min_degree) / (max_degree - min_degree)
            else:
                norm_value = 0.5
            node_sizes[node] = min_size + norm_value * (max_size - min_size)
    
    else:  # uniform
        for node in G.nodes():
            node_sizes[node] = (min_size + max_size) / 2
    
    # 突出显示特殊节点
    for node in highlight_nodes:
        if node in node_sizes:
            node_sizes[node] = max(node_sizes[node] * 1.5, max_size * 1.2)
    
    logger.info(f"✅ 节点大小计算完成")
    return node_sizes

def draw_backbone_network(G: nx.Graph,
                         pos: Dict = None,
                         node_colors: Dict = None,
                         node_sizes: Dict = None,
                         title: str = "Backbone Network",
                         save_path: Path = None,
                         figsize: Tuple[int, int] = (12, 10),
                         show_labels: bool = True,
                         label_threshold: int = 5,
                         highlight_usa: bool = True,
                         original_network: nx.Graph = None,
                         color_scheme: str = 'geographic',
                         size_attribute: str = 'strength') -> plt.Figure:
    """
    绘制骨干网络图
    
    Args:
        G: 骨干网络图
        pos: 节点位置
        node_colors: 节点颜色
        node_sizes: 节点大小
        title: 图标题
        save_path: 保存路径
        figsize: 图片大小
        show_labels: 是否显示节点标签
        label_threshold: 标签显示的最小度数阈值
        highlight_usa: 是否突出显示美国
        
    Returns:
        matplotlib Figure对象
    """
    
    logger.info(f"🎨 绘制骨干网络图: {title}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # 计算布局
    if pos is None:
        pos = get_force_atlas_layout(G)
    
    # 设置默认颜色和大小 - 使用原始网络数据确保信息保真
    if node_colors is None:
        node_colors = assign_node_colors(G, color_scheme, original_network=original_network)
    
    if node_sizes is None:
        node_sizes = calculate_node_sizes(G, size_attribute, original_network=original_network)
    
    # 准备绘图数据
    colors = [node_colors.get(node, '#1f77b4') for node in G.nodes()]
    sizes = [node_sizes.get(node, 50) for node in G.nodes()]
    
    # 绘制边
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    if edge_weights:
        # 标准化边宽度
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            edge_widths = [0.5 + 2.0 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights]
        else:
            edge_widths = [1.0] * len(edge_weights)
    else:
        edge_widths = [1.0] * G.number_of_edges()
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.6,
        edge_color='#666666'
    )
    
    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors,
        node_size=sizes,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    # 添加节点标签
    if show_labels:
        # 只为重要节点添加标签
        labels = {}
        for node in G.nodes():
            node_degree = G.degree(node)
            if node_degree >= label_threshold or node == 'USA':
                labels[node] = node
        
        if labels:
            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=8,
                font_weight='bold',
                font_color='black'
            )
    
    # 特殊处理美国节点
    if highlight_usa and 'USA' in G.nodes():
        usa_pos = pos.get('USA')
        if usa_pos is not None:
            # 添加美国标签和特殊标记
            ax.annotate('USA', usa_pos, 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 设置标题和样式
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')  # 隐藏坐标轴
    
    # 添加网络统计信息
    stats_text = f"Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}"
    if hasattr(G, 'graph') and 'retention_rate' in G.graph:
        stats_text += f" | Retention: {G.graph['retention_rate']:.1%}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
    
    # 保存图形
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"💾 图形已保存: {save_path}")
    
    return fig

def create_backbone_comparison_plot(backbones: Dict[str, nx.Graph],
                                  year: int,
                                  save_path: Path = None,
                                  figsize: Tuple[int, int] = (18, 6)) -> plt.Figure:
    """
    创建多个骨干网络的对比图
    
    Args:
        backbones: 骨干网络字典 {algorithm_name: network}
        year: 年份
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    
    logger.info(f"📊 创建{year}年骨干网络对比图...")
    
    n_algorithms = len(backbones)
    fig, axes = plt.subplots(1, n_algorithms, figsize=figsize, facecolor='white')
    
    if n_algorithms == 1:
        axes = [axes]
    
    # 为保持一致性，使用相同的布局种子
    base_network = next(iter(backbones.values()))
    base_pos = get_force_atlas_layout(base_network)
    
    for i, (algorithm_name, G) in enumerate(backbones.items()):
        ax = axes[i]
        
        # 调整布局以适应当前网络
        if set(G.nodes()) != set(base_network.nodes()):
            current_pos = get_force_atlas_layout(G)
        else:
            current_pos = {node: base_pos[node] for node in G.nodes() if node in base_pos}
        
        # 获取颜色和大小
        colors = assign_node_colors(G, 'geographic')
        sizes = calculate_node_sizes(G, 'strength')
        
        # 绘制网络
        node_colors_list = [colors.get(node, '#1f77b4') for node in G.nodes()]
        node_sizes_list = [sizes.get(node, 50) for node in G.nodes()]
        
        # 边
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [2.0 * w / max_weight for w in edge_weights]
        else:
            edge_widths = [1.0] * G.number_of_edges()
        
        nx.draw_networkx_edges(G, current_pos, ax=ax,
                             width=edge_widths, alpha=0.5, edge_color='gray')
        
        nx.draw_networkx_nodes(G, current_pos, ax=ax,
                             node_color=node_colors_list,
                             node_size=node_sizes_list,
                             alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # 添加重要节点标签
        important_nodes = {}
        for node in G.nodes():
            if G.degree(node) >= 3 or node == 'USA':
                important_nodes[node] = node
        
        nx.draw_networkx_labels(G, current_pos, important_nodes, ax=ax,
                              font_size=6, font_weight='bold')
        
        # 设置子图标题
        retention_rate = G.graph.get('retention_rate', 0)
        ax.set_title(f"{algorithm_name}\n{G.number_of_edges()} edges ({retention_rate:.1%})",
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 设置总标题
    fig.suptitle(f'Backbone Network Comparison - {year}', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"💾 对比图已保存: {save_path}")
    
    return fig

def draw_professional_backbone_network(backbone_G: nx.Graph,
                                     full_network_G: nx.Graph = None,
                                     node_centrality_data: Dict = None,
                                     title: str = "Professional Backbone Network",
                                     save_path: Path = None,
                                     figsize: Tuple[int, int] = (15, 12),
                                     layout_algorithm: str = 'force_atlas2',
                                     color_scheme: str = 'geographic',
                                     show_legend: bool = True,
                                     max_labels: int = 20) -> plt.Figure:
    """
    绘制专业级骨干网络图
    
    关键改进：
    1. 节点大小：基于完整网络的total_strength
    2. 节点颜色：按地理区域或贸易规模分类
    3. 标签处理：智能避免重叠，关键节点突出显示
    4. 布局优化：使用professional layout算法
    5. 美学设计：配色方案、边透明度、图例设计
    
    Args:
        backbone_G: 骨干网络
        full_network_G: 完整网络（用于提取节点属性）
        node_centrality_data: 来自03模块的中心性数据
        title: 图标题
        save_path: 保存路径
        figsize: 图片大小
        layout_algorithm: 布局算法
        color_scheme: 颜色方案
        show_legend: 是否显示图例
        max_labels: 最大标签数量
        
    Returns:
        matplotlib Figure对象
    """
    
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir))
    sys.path.append(str(current_dir.parent))
    
    try:
        from visualization.styling import ProfessionalNetworkStyling, NetworkTheme
        from data_io.attribute_loader import NetworkAttributeLoader
    except ImportError:
        try:
            from styling import ProfessionalNetworkStyling, NetworkTheme
            sys.path.append(str(current_dir.parent / "data_io"))
            from attribute_loader import NetworkAttributeLoader
        except ImportError:
            # Create a minimal fallback if imports fail
            logger.warning("⚠️ 无法导入专业样式系统，使用基础功能")
            ProfessionalNetworkStyling = None
            NetworkTheme = None
            NetworkAttributeLoader = None
    
    logger.info(f"🎨 创建专业级骨干网络可视化: {title}...")
    
    # 初始化专业样式系统（如果可用）
    if ProfessionalNetworkStyling and NetworkTheme:
        theme = NetworkTheme()
        styling = ProfessionalNetworkStyling(theme)
        use_professional_styling = True
    else:
        # 使用基础样式
        logger.warning("⚠️ 使用基础样式系统")
        theme = None
        styling = None
        use_professional_styling = False
    
    # 加载或准备节点属性数据
    node_attributes = {}
    if node_centrality_data:
        node_attributes.update(node_centrality_data)
    
    if full_network_G:
        # 从完整网络提取属性
        for node in backbone_G.nodes():
            if node in full_network_G.nodes():
                if 'total_strength' not in node_attributes:
                    node_attributes['total_strength'] = {}
                node_attributes['total_strength'][node] = full_network_G.degree(node, weight='weight')
    
    # 创建主图形
    if use_professional_styling:
        fig, ax = plt.subplots(figsize=figsize, facecolor=theme.background_color)
        ax.set_facecolor(theme.background_color)
        
        # 计算专业级布局
        pos = styling.apply_professional_layout_algorithm(backbone_G, layout_algorithm)
        
        # 分配专业级颜色方案
        node_colors = styling.assign_node_colors(backbone_G, color_scheme, node_attributes)
        
        # 计算节点大小（基于完整网络数据）
        node_sizes = styling.calculate_node_sizes(backbone_G, full_network_G, node_attributes)
        
        # 计算边宽度
        edge_widths = styling.calculate_edge_widths(backbone_G)
        
        # 创建智能标签
        labels = styling.create_intelligent_labels(backbone_G, pos, node_sizes, node_centrality_data, max_labels)
        
    else:
        # 基础样式回退
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # 基础布局
        pos = nx.spring_layout(backbone_G, k=3.0, iterations=50, weight='weight', seed=42)
        
        # 基础颜色（地理区域）
        node_colors = assign_node_colors(backbone_G, 'geographic', original_network=full_network_G)
        
        # 基础节点大小
        node_sizes = calculate_node_sizes(backbone_G, 'strength', original_network=full_network_G)
        
        # 基础边宽度
        edge_weights = [backbone_G[u][v].get('weight', 1.0) for u, v in backbone_G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [2.0 * w / max_weight for w in edge_weights]
        else:
            edge_widths = [1.0] * backbone_G.number_of_edges()
        
        # 基础标签（重要节点）
        labels = {}
        for node in backbone_G.nodes():
            if backbone_G.degree(node) >= 3 or node == 'USA':
                labels[node] = node
        
    # 绘制边
    if edge_widths:
        nx.draw_networkx_edges(
            backbone_G, pos, ax=ax,
            width=edge_widths,
            alpha=theme.edge_alpha,
            edge_color='#666666'
        )
    
    # 绘制节点
    colors_list = [node_colors.get(node, theme.primary_color) for node in backbone_G.nodes()]
    sizes_list = [node_sizes.get(node, 100) for node in backbone_G.nodes()]
    
    nx.draw_networkx_nodes(
        backbone_G, pos, ax=ax,
        node_color=colors_list,
        node_size=sizes_list,
        alpha=theme.node_alpha,
        edgecolors='black',
        linewidths=1.0
    )
        
    # 绘制智能标签
    if labels:
        nx.draw_networkx_labels(
            backbone_G, pos, labels, ax=ax,
            font_size=theme.label_font_size,
            font_weight='bold',
            font_color=theme.text_color
        )
    
    # 特殊处理美国节点
    if 'USA' in backbone_G.nodes() and 'USA' in pos:
        usa_pos = pos['USA']
        ax.annotate('USA', usa_pos, 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=theme.label_font_size + 2, 
                   fontweight='bold', 
                   color='#FF4136',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='white', 
                           edgecolor='#FF4136',
                           alpha=0.9))
        
    # 设置标题
    ax.set_title(title, fontsize=theme.title_font_size, fontweight='bold', 
                color=theme.text_color, pad=20)
    
    # 隐藏坐标轴
    ax.axis('off')
    ax.set_aspect('equal')
    
    # 添加专业级注释
    algorithm_name = backbone_G.graph.get('backbone_method', 'Unknown')
    styling.add_professional_annotations(ax, backbone_G, full_network_G, algorithm_name)
    
    # 创建专业级图例
    if show_legend:
        styling.create_legend(fig, color_scheme)
    
    plt.tight_layout()
        
    # 保存图形
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=theme.background_color, edgecolor='none')
        logger.info(f"💾 专业级可视化已保存: {save_path}")
    
    return fig

def create_information_preserving_visualization(backbone_network: nx.Graph,
                                              original_network: nx.Graph,
                                              year: int,
                                              algorithm_name: str,
                                              save_path: Path = None,
                                              show_comparison: bool = True) -> plt.Figure:
    """
    兼容性函数：调用新的专业级绘制功能
    
    这个函数保持向后兼容性，同时利用新的专业级可视化系统
    """
    
    title = f"{algorithm_name} Backbone Network - {year}"
    
    return draw_professional_backbone_network(
        backbone_G=backbone_network,
        full_network_G=original_network,
        title=title,
        save_path=save_path,
        figsize=(15, 12),
        layout_algorithm='force_atlas2',
        color_scheme='geographic',
        show_legend=True,
        max_labels=20
    )
    
    # 保存图形
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"💾 信息保真可视化已保存: {save_path}")
    
    return fig

if __name__ == "__main__":
    # 测试可视化功能
    logger.info("🧪 测试骨干网络可视化...")
    
    # 创建测试网络
    G_test = nx.Graph()
    
    # 添加一些测试节点和边
    countries = ['USA', 'CAN', 'MEX', 'GBR', 'DEU', 'CHN', 'JPN', 'SAU']
    edges = [
        ('USA', 'CAN', 100), ('USA', 'MEX', 80), ('USA', 'GBR', 60),
        ('USA', 'SAU', 50), ('GBR', 'DEU', 40), ('CHN', 'JPN', 30),
        ('CAN', 'GBR', 20), ('MEX', 'SAU', 15)
    ]
    
    for country in countries:
        G_test.add_node(country)
    
    for source, target, weight in edges:
        G_test.add_edge(source, target, weight=weight)
    
    # 测试绘制
    fig = draw_backbone_network(
        G_test, 
        title="Test Backbone Network",
        highlight_usa=True,
        show_labels=True
    )
    
    print("🎉 可视化测试完成!")
    plt.show()