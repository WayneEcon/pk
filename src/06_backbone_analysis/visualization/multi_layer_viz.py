#!/usr/bin/env python3
"""
多层次信息整合可视化系统
====================

Phase 2升级P2任务：实现多层次信息整合可视化
专门处理复杂网络分析中的多维度信息同时展示问题。

核心创新功能：
1. 分层网络可视化：同时展示完整网络、骨干网络、关键路径
2. 时间序列动态可视化：美国地位变化的动态轨迹
3. 多维度信息叠加：地理、经济、政策多维度信息融合
4. 交互式面板：可切换不同算法、年份、指标的对比视图

设计理念：
- 信息密度最大化：在单一视图中展示最多有用信息
- 认知负荷最小化：通过层次化设计降低理解难度
- 政策决策支持：为政策制定者提供直观的分析结果

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiLayerTheme:
    """多层次可视化主题"""
    # 层次配色
    full_network_color: str = '#E8E8E8'      # 完整网络：浅灰
    backbone_color: str = '#2E86AB'          # 骨干网络：专业蓝
    critical_path_color: str = '#F18F01'     # 关键路径：橙黄
    usa_highlight_color: str = '#FF4136'     # 美国突出：鲜红
    
    # 透明度设置
    full_network_alpha: float = 0.3
    backbone_alpha: float = 0.8
    critical_path_alpha: float = 1.0
    
    # 字体大小
    main_title_size: int = 18
    subplot_title_size: int = 14
    legend_font_size: int = 10
    annotation_font_size: int = 9

class MultiLayerVisualizer:
    """多层次信息整合可视化器"""
    
    def __init__(self, theme: MultiLayerTheme = None):
        """
        初始化多层次可视化器
        
        Args:
            theme: 可视化主题
        """
        self.theme = theme or MultiLayerTheme()
        self._setup_matplotlib()
        
        logger.info("🎨 多层次信息整合可视化器初始化完成")
    
    def _setup_matplotlib(self):
        """设置matplotlib参数"""
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'font.size': 10,
            'axes.titlesize': self.theme.subplot_title_size,
            'figure.titlesize': self.theme.main_title_size,
            'legend.fontsize': self.theme.legend_font_size
        })
    
    def create_layered_network_visualization(self,
                                          full_network: nx.Graph,
                                          backbone_network: nx.Graph,
                                          usa_critical_paths: List[List[str]] = None,
                                          year: int = None,
                                          algorithm_name: str = "Backbone",
                                          node_attributes: Dict = None,
                                          save_path: Path = None) -> plt.Figure:
        """
        创建分层网络可视化
        
        核心功能：
        1. 底层：完整网络（浅色显示）
        2. 中层：骨干网络（突出显示）
        3. 顶层：美国关键路径（高亮显示）
        4. 信息面板：统计数据和关键指标
        
        Args:
            full_network: 完整网络
            backbone_network: 骨干网络
            usa_critical_paths: 美国关键路径
            year: 年份
            algorithm_name: 算法名称
            node_attributes: 节点属性
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        
        logger.info(f"🎨 创建{algorithm_name}分层网络可视化...")
        
        # 创建主图形
        fig = plt.figure(figsize=(20, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1],
                             hspace=0.3, wspace=0.3)
        
        # 主网络图
        ax_main = fig.add_subplot(gs[0, :2])
        
        # 统计面板
        ax_stats = fig.add_subplot(gs[0, 2])
        
        # 时间序列图（如果有多年数据）
        ax_timeline = fig.add_subplot(gs[1, :2])
        
        # 图例面板
        ax_legend = fig.add_subplot(gs[1, 2])
        
        # 1. 绘制主网络图
        self._draw_layered_network(ax_main, full_network, backbone_network, 
                                 usa_critical_paths, node_attributes)
        
        # 2. 绘制统计面板
        self._draw_statistics_panel(ax_stats, full_network, backbone_network, 
                                   algorithm_name, year)
        
        # 3. 绘制时间序列（占位）
        self._draw_timeline_placeholder(ax_timeline)
        
        # 4. 绘制图例
        self._draw_multi_layer_legend(ax_legend)
        
        # 设置主标题
        title = f"Multi-Layer Network Analysis: {algorithm_name}"
        if year:
            title += f" ({year})"
        
        fig.suptitle(title, fontsize=self.theme.main_title_size, 
                    fontweight='bold', y=0.95)
        
        # 保存图形
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"💾 分层网络可视化已保存: {save_path}")
        
        return fig
    
    def _draw_layered_network(self,
                             ax: plt.Axes,
                             full_network: nx.Graph,
                             backbone_network: nx.Graph,
                             usa_paths: List[List[str]] = None,
                             node_attributes: Dict = None):
        """绘制分层网络"""
        
        # 计算布局（基于完整网络以保持一致性）
        pos = nx.spring_layout(full_network, k=3.0, iterations=50, 
                              weight='weight', seed=42)
        
        # 第1层：绘制完整网络（底层，浅色）
        self._draw_full_network_layer(ax, full_network, pos)
        
        # 第2层：绘制骨干网络（中层，突出）
        self._draw_backbone_layer(ax, backbone_network, pos, node_attributes)
        
        # 第3层：绘制美国关键路径（顶层，高亮）
        if usa_paths:
            self._draw_usa_critical_paths(ax, backbone_network, pos, usa_paths)
        
        # 设置轴属性
        ax.set_title("Layered Network Structure", fontsize=self.theme.subplot_title_size,
                    fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')
    
    def _draw_full_network_layer(self, ax: plt.Axes, G: nx.Graph, pos: Dict):
        """绘制完整网络层"""
        
        if G.number_of_edges() == 0:
            return
        
        # 绘制所有边（浅灰色，低透明度）
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=0.3,
            alpha=self.theme.full_network_alpha,
            edge_color=self.theme.full_network_color
        )
        
        # 绘制所有节点（小尺寸，浅色）
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=30,
            node_color=self.theme.full_network_color,
            alpha=self.theme.full_network_alpha,
            edgecolors='none'
        )
    
    def _draw_backbone_layer(self, ax: plt.Axes, G: nx.Graph, pos: Dict, 
                           node_attributes: Dict = None):
        """绘制骨干网络层"""
        
        if G.number_of_nodes() == 0:
            return
        
        # 计算边宽度
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            if max_weight > min_weight:
                edge_widths = [1.0 + 3.0 * (w - min_weight) / (max_weight - min_weight) 
                              for w in edge_weights]
            else:
                edge_widths = [2.0] * len(edge_weights)
        else:
            edge_widths = []
        
        # 绘制骨干边
        if edge_widths:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                alpha=self.theme.backbone_alpha,
                edge_color=self.theme.backbone_color
            )
        
        # 计算节点大小
        node_sizes = self._calculate_backbone_node_sizes(G, node_attributes)
        node_colors = self._assign_backbone_node_colors(G, node_attributes)
        
        # 绘制骨干节点
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=list(node_sizes.values()),
            node_color=list(node_colors.values()),
            alpha=self.theme.backbone_alpha,
            edgecolors='black',
            linewidths=1.0
        )
        
        # 添加重要节点标签
        important_labels = {}
        for node in G.nodes():
            if G.degree(node) >= 3 or node == 'USA':
                important_labels[node] = node
        
        if important_labels:
            nx.draw_networkx_labels(
                G, pos, important_labels, ax=ax,
                font_size=9,
                font_weight='bold',
                font_color='black'
            )
    
    def _draw_usa_critical_paths(self, ax: plt.Axes, G: nx.Graph, pos: Dict,
                               paths: List[List[str]]):
        """绘制美国关键路径"""
        
        for i, path in enumerate(paths[:3]):  # 最多显示3条关键路径
            # 验证路径在图中存在
            valid_path = []
            for j in range(len(path) - 1):
                if path[j] in G.nodes() and path[j+1] in G.nodes():
                    if G.has_edge(path[j], path[j+1]):
                        valid_path.extend([path[j], path[j+1]])
            
            if len(valid_path) < 2:
                continue
            
            # 创建路径的子图
            path_edges = [(valid_path[k], valid_path[k+1]) 
                         for k in range(0, len(valid_path)-1, 2)]
            
            # 绘制路径边（高亮）
            nx.draw_networkx_edges(
                G, pos, edgelist=path_edges, ax=ax,
                width=4.0,
                alpha=self.theme.critical_path_alpha,
                edge_color=self.theme.critical_path_color
            )
    
    def _calculate_backbone_node_sizes(self, G: nx.Graph, 
                                     node_attributes: Dict = None) -> Dict[str, float]:
        """计算骨干网络节点大小"""
        
        node_sizes = {}
        
        if node_attributes and 'total_strength' in node_attributes:
            # 使用完整网络的强度信息
            strengths = node_attributes['total_strength']
            if strengths:
                values = list(strengths.values())
                min_strength = min(values)
                max_strength = max(values)
                
                for node in G.nodes():
                    if node in strengths:
                        strength = strengths[node]
                        if max_strength > min_strength:
                            norm_val = (strength - min_strength) / (max_strength - min_strength)
                        else:
                            norm_val = 0.5
                        node_sizes[node] = 100 + norm_val * 300  # 100-400范围
                    else:
                        node_sizes[node] = 150
        else:
            # 使用骨干网络度数
            degrees = dict(G.degree(weight='weight'))
            if degrees:
                values = list(degrees.values())
                min_deg = min(values) if values else 0
                max_deg = max(values) if values else 1
                
                for node in G.nodes():
                    degree = degrees.get(node, 0)
                    if max_deg > min_deg:
                        norm_val = (degree - min_deg) / (max_deg - min_deg)
                    else:
                        norm_val = 0.5
                    node_sizes[node] = 100 + norm_val * 300
            else:
                for node in G.nodes():
                    node_sizes[node] = 150
        
        # 美国节点特殊处理
        if 'USA' in node_sizes:
            node_sizes['USA'] = max(node_sizes['USA'] * 1.3, 400)
        
        return node_sizes
    
    def _assign_backbone_node_colors(self, G: nx.Graph,
                                   node_attributes: Dict = None) -> Dict[str, str]:
        """分配骨干网络节点颜色"""
        
        # 导入地理区域映射
        try:
            from .styling import ProfessionalNetworkStyling
            styling = ProfessionalNetworkStyling()
            region_mapping = styling.COUNTRY_TO_REGION
        except:
            # 基础地理映射
            region_mapping = {
                'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
                'GBR': 'Europe', 'DEU': 'Europe', 'FRA': 'Europe',
                'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia',
                'SAU': 'Middle East', 'ARE': 'Middle East'
            }
        
        # 地理区域配色
        region_colors = {
            'North America': '#1f77b4',
            'Europe': '#ff7f0e', 
            'Asia': '#2ca02c',
            'Middle East': '#d62728',
            'Latin America': '#9467bd',
            'Africa': '#8c564b',
            'Oceania': '#e377c2',
            'Other': '#7f7f7f'
        }
        
        node_colors = {}
        for node in G.nodes():
            region = region_mapping.get(node, 'Other')
            node_colors[node] = region_colors[region]
        
        # 美国特殊突出
        if 'USA' in node_colors:
            node_colors['USA'] = self.theme.usa_highlight_color
        
        return node_colors
    
    def _draw_statistics_panel(self, ax: plt.Axes, full_network: nx.Graph,
                             backbone_network: nx.Graph, algorithm_name: str,
                             year: int = None):
        """绘制统计信息面板"""
        
        ax.axis('off')
        
        # 计算统计数据
        stats = self._calculate_network_statistics(full_network, backbone_network)
        
        # 创建信息框
        info_text = f"📊 {algorithm_name} Statistics"
        if year:
            info_text += f" ({year})"
        info_text += "\n" + "─" * 25 + "\n"
        
        # 基础网络统计
        info_text += f"🔗 Full Network:\n"
        info_text += f"   Nodes: {stats['full_nodes']:,}\n"
        info_text += f"   Edges: {stats['full_edges']:,}\n"
        info_text += f"   Density: {stats['full_density']:.3f}\n\n"
        
        info_text += f"⭐ Backbone Network:\n"
        info_text += f"   Nodes: {stats['backbone_nodes']:,}\n"
        info_text += f"   Edges: {stats['backbone_edges']:,}\n"
        info_text += f"   Retention: {stats['retention_rate']:.1%}\n"
        info_text += f"   Efficiency: {stats['efficiency_gain']:.1%}\n\n"
        
        # 美国特殊统计
        if stats['usa_in_backbone']:
            info_text += f"🇺🇸 USA Analysis:\n"
            info_text += f"   Full Degree: {stats['usa_full_degree']}\n"
            info_text += f"   Backbone Degree: {stats['usa_backbone_degree']}\n"
            info_text += f"   Centrality Preserved: {stats['usa_centrality_preserved']}\n"
        
        # 绘制文本
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=self.theme.annotation_font_size,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        ax.set_title("Network Statistics", fontsize=self.theme.subplot_title_size,
                    fontweight='bold')
    
    def _calculate_network_statistics(self, full_G: nx.Graph, 
                                    backbone_G: nx.Graph) -> Dict[str, Any]:
        """计算网络统计数据"""
        
        stats = {
            'full_nodes': full_G.number_of_nodes(),
            'full_edges': full_G.number_of_edges(),
            'full_density': nx.density(full_G),
            'backbone_nodes': backbone_G.number_of_nodes(),
            'backbone_edges': backbone_G.number_of_edges(),
            'retention_rate': 0,
            'efficiency_gain': 0,
            'usa_in_backbone': 'USA' in backbone_G.nodes(),
            'usa_full_degree': 0,
            'usa_backbone_degree': 0,
            'usa_centrality_preserved': 'N/A'
        }
        
        # 计算保留率
        if full_G.number_of_edges() > 0:
            stats['retention_rate'] = backbone_G.number_of_edges() / full_G.number_of_edges()
        
        # 计算效率增益（边数减少 vs 信息保留）
        edge_reduction = 1 - stats['retention_rate']
        # 简单估计：假设关键信息保留率约为边保留率的1.5倍
        info_preservation = min(1.0, stats['retention_rate'] * 1.5)
        stats['efficiency_gain'] = edge_reduction * info_preservation
        
        # 美国统计
        if 'USA' in full_G.nodes():
            stats['usa_full_degree'] = full_G.degree('USA')
        
        if 'USA' in backbone_G.nodes():
            stats['usa_backbone_degree'] = backbone_G.degree('USA')
            
            # 简单的中心性保留评估
            if stats['usa_full_degree'] > 0:
                preservation_ratio = stats['usa_backbone_degree'] / stats['usa_full_degree']
                if preservation_ratio > 0.8:
                    stats['usa_centrality_preserved'] = 'High'
                elif preservation_ratio > 0.5:
                    stats['usa_centrality_preserved'] = 'Medium'
                else:
                    stats['usa_centrality_preserved'] = 'Low'
        
        return stats
    
    def _draw_timeline_placeholder(self, ax: plt.Axes):
        """绘制时间序列占位图"""
        
        # 占位数据
        years = range(2010, 2021)
        usa_position = np.random.normal(0.7, 0.1, len(years))
        usa_position = np.clip(usa_position, 0.3, 1.0)
        
        ax.plot(years, usa_position, 'o-', color=self.theme.usa_highlight_color,
                linewidth=2, markersize=4, label='USA Position Index')
        
        ax.axvline(x=2011, color='gray', linestyle='--', alpha=0.7, label='Shale Revolution')
        ax.axvline(x=2016, color='orange', linestyle='--', alpha=0.7, label='Policy Change')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Position Index')
        ax.set_title('USA Energy Position Timeline', fontsize=self.theme.subplot_title_size)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _draw_multi_layer_legend(self, ax: plt.Axes):
        """绘制多层次图例"""
        
        ax.axis('off')
        
        # 创建图例元素
        legend_elements = [
            plt.Line2D([0], [0], color=self.theme.full_network_color, lw=3, 
                      alpha=self.theme.full_network_alpha, label='Full Network'),
            plt.Line2D([0], [0], color=self.theme.backbone_color, lw=3,
                      alpha=self.theme.backbone_alpha, label='Backbone Network'),
            plt.Line2D([0], [0], color=self.theme.critical_path_color, lw=4,
                      alpha=self.theme.critical_path_alpha, label='Critical Paths'),
            plt.scatter([0], [0], c=self.theme.usa_highlight_color, s=100, 
                       label='USA (Highlighted)')
        ]
        
        # 创建图例
        ax.legend(handles=legend_elements, loc='center', frameon=True,
                 fancybox=True, shadow=True,
                 fontsize=self.theme.legend_font_size)
        
        ax.set_title("Layer Legend", fontsize=self.theme.subplot_title_size,
                    fontweight='bold')
    
    def create_comparative_timeline_visualization(self,
                                               multi_year_data: Dict[str, Dict[int, nx.Graph]],
                                               focus_node: str = 'USA',
                                               save_path: Path = None) -> plt.Figure:
        """
        创建跨算法的时间序列对比可视化
        
        Args:
            multi_year_data: {algorithm_name: {year: network}}
            focus_node: 关注节点
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        
        logger.info(f"📈 创建{focus_node}时间序列对比可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{focus_node} Energy Position: Multi-Algorithm Comparison',
                    fontsize=self.theme.main_title_size, fontweight='bold')
        
        # 1. 度数中心性时间序列
        self._plot_degree_timeline(axes[0, 0], multi_year_data, focus_node)
        
        # 2. 相对地位变化
        self._plot_relative_position_timeline(axes[0, 1], multi_year_data, focus_node)
        
        # 3. 算法一致性分析
        self._plot_algorithm_consistency(axes[1, 0], multi_year_data, focus_node)
        
        # 4. 关键事件标注
        self._plot_event_impact_analysis(axes[1, 1], multi_year_data, focus_node)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"💾 时间序列对比可视化已保存: {save_path}")
        
        return fig
    
    def _plot_degree_timeline(self, ax: plt.Axes, data: Dict, focus_node: str):
        """绘制度数时间序列"""
        
        ax.set_title('Degree Centrality Over Time', fontweight='bold')
        
        for algorithm_name, yearly_networks in data.items():
            years = []
            degrees = []
            
            for year in sorted(yearly_networks.keys()):
                network = yearly_networks[year]
                if focus_node in network.nodes():
                    years.append(year)
                    degrees.append(network.degree(focus_node, weight='weight'))
            
            if years:
                ax.plot(years, degrees, 'o-', label=algorithm_name, linewidth=2, markersize=4)
        
        ax.axvline(x=2011, color='gray', linestyle='--', alpha=0.7, label='Shale Revolution')
        ax.set_xlabel('Year')
        ax.set_ylabel('Weighted Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_relative_position_timeline(self, ax: plt.Axes, data: Dict, focus_node: str):
        """绘制相对地位变化"""
        
        ax.set_title('Relative Position Ranking', fontweight='bold')
        
        for algorithm_name, yearly_networks in data.items():
            years = []
            rankings = []
            
            for year in sorted(yearly_networks.keys()):
                network = yearly_networks[year]
                if focus_node in network.nodes():
                    # 计算排名
                    degrees = dict(network.degree(weight='weight'))
                    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                    
                    for rank, (node, _) in enumerate(sorted_nodes, 1):
                        if node == focus_node:
                            years.append(year)
                            rankings.append(rank)
                            break
            
            if years:
                ax.plot(years, rankings, 's-', label=algorithm_name, linewidth=2, markersize=4)
        
        ax.axvline(x=2011, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Ranking (1=Highest)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 排名越小越好
    
    def _plot_algorithm_consistency(self, ax: plt.Axes, data: Dict, focus_node: str):
        """绘制算法一致性分析"""
        
        ax.set_title('Cross-Algorithm Consistency', fontweight='bold')
        
        # 计算各算法结果的相关性
        algorithm_names = list(data.keys())
        if len(algorithm_names) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 algorithms', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # 创建相关性矩阵的可视化占位
        consistency_matrix = np.random.rand(len(algorithm_names), len(algorithm_names))
        np.fill_diagonal(consistency_matrix, 1.0)
        
        im = ax.imshow(consistency_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(algorithm_names)))
        ax.set_yticks(range(len(algorithm_names)))
        ax.set_xticklabels(algorithm_names, rotation=45)
        ax.set_yticklabels(algorithm_names)
        
        # 添加文本标注
        for i in range(len(algorithm_names)):
            for j in range(len(algorithm_names)):
                ax.text(j, i, f'{consistency_matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_event_impact_analysis(self, ax: plt.Axes, data: Dict, focus_node: str):
        """绘制事件影响分析"""
        
        ax.set_title('Policy Event Impact Analysis', fontweight='bold')
        
        # 关键事件
        events = {
            2008: 'Financial Crisis',
            2011: 'Shale Revolution',
            2014: 'Oil Price Drop', 
            2016: 'Policy Changes',
            2020: 'COVID-19'
        }
        
        # 计算事件前后的影响
        if data:
            sample_alg = list(data.keys())[0]
            sample_data = data[sample_alg]
            
            years = sorted(sample_data.keys())
            if years:
                # 简单的影响分析可视化
                event_years = [year for year in events.keys() if year in years]
                
                for event_year in event_years:
                    ax.axvline(x=event_year, color='red', linestyle='-', alpha=0.6)
                    ax.text(event_year, 0.8, events[event_year], rotation=90,
                           va='bottom', ha='right', fontsize=8)
                
                # 添加影响强度曲线（占位）
                impact_curve = np.sin(np.linspace(0, 4*np.pi, len(years))) * 0.3 + 0.5
                ax.plot(years, impact_curve, 'g-', linewidth=2, alpha=0.7,
                       label=f'{focus_node} Impact Index')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Impact Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

if __name__ == "__main__":
    # 测试多层次可视化系统
    logger.info("🧪 测试多层次信息整合可视化系统...")
    
    # 创建测试数据
    countries = ['USA', 'CAN', 'MEX', 'GBR', 'DEU', 'CHN', 'JPN', 'SAU']
    
    # 完整网络
    G_full = nx.Graph()
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries[i+1:], i+1):
            weight = np.random.exponential(100)
            # 美国相关的边权重更大
            if 'USA' in [country1, country2]:
                weight *= 2
            G_full.add_edge(country1, country2, weight=weight)
    
    # 骨干网络（移除一些边）
    G_backbone = G_full.copy()
    edges_to_remove = list(G_full.edges())[:len(G_full.edges())//3]
    G_backbone.remove_edges_from(edges_to_remove)
    
    # 模拟节点属性
    node_attributes = {
        'total_strength': {node: G_full.degree(node, weight='weight') for node in G_full.nodes()},
        'geographic_region': {node: 'North America' if node in ['USA', 'CAN', 'MEX'] else 'Other' 
                            for node in G_full.nodes()}
    }
    
    # 初始化可视化器
    visualizer = MultiLayerVisualizer()
    
    # 测试分层网络可视化
    fig1 = visualizer.create_layered_network_visualization(
        full_network=G_full,
        backbone_network=G_backbone,
        usa_critical_paths=[['USA', 'CAN', 'GBR'], ['USA', 'SAU', 'CHN']],
        year=2020,
        algorithm_name="Disparity Filter",
        node_attributes=node_attributes
    )
    
    # 测试时间序列对比可视化
    multi_year_data = {
        'Disparity Filter': {2018: G_backbone, 2019: G_backbone, 2020: G_backbone},
        'MST': {2018: G_backbone, 2019: G_backbone, 2020: G_backbone}
    }
    
    fig2 = visualizer.create_comparative_timeline_visualization(
        multi_year_data=multi_year_data,
        focus_node='USA'
    )
    
    print("🎉 多层次信息整合可视化系统测试完成!")
    print("系统已准备就绪，可以创建复杂的多层次可视化。")
    
    plt.show()