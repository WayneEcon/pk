#!/usr/bin/env python3
"""
骨干网络提取模块 (04_backbone_extraction)
===========================================

实现三种主要的网络骨干提取方法：
1. Disparity Filter (DF) - 差异性过滤器
2. Polarity Filter (PF) - 极性过滤器  
3. Minimum Spanning Tree (MST) - 最小生成树

这些方法用于从完整的贸易网络中提取最重要的连接，
简化网络结构同时保留关键的贸易关系。

参考文献：
- Serrano et al. (2009) Extracting the multiscale backbone of complex weighted networks
- Tumminello et al. (2011) Statistically Validated Networks in Bipartite Complex Systems
- Kruskal (1956) On the shortest spanning subtree of a graph
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import logging
from pathlib import Path
import sys

# 添加上级模块路径以导入工具函数
sys.path.append(str(Path(__file__).parent / "03_metrics"))
from utils import setup_logger, validate_graph, timer_decorator, handle_computation_error

logger = setup_logger(__name__)

class BackboneExtractor:
    """
    网络骨干提取器类
    
    提供三种主要的骨干提取方法，每种方法都有不同的理论基础和适用场景
    """
    
    def __init__(self, G: nx.DiGraph):
        """
        初始化骨干提取器
        
        Args:
            G: 输入的加权有向图
        """
        validate_graph(G, "BackboneExtractor")
        self.G = G.copy()
        self.logger = setup_logger(f"{__name__}.BackboneExtractor")
        
        # 预计算一些常用统计量
        self._precompute_stats()
    
    def _precompute_stats(self):
        """预计算网络统计量"""
        self.n_nodes = self.G.number_of_nodes()
        self.n_edges = self.G.number_of_edges()
        
        # 计算节点的总强度
        self.node_strengths = {}
        for node in self.G.nodes():
            out_strength = sum(self.G[node][neighbor]['weight'] 
                             for neighbor in self.G.neighbors(node))
            in_strength = sum(self.G[pred][node]['weight'] 
                            for pred in self.G.predecessors(node))
            self.node_strengths[node] = {
                'out': out_strength,
                'in': in_strength,
                'total': out_strength + in_strength
            }
    
    @timer_decorator
    def extract_disparity_filter_backbone(self, alpha: float = 0.05, 
                                         direction: str = 'both') -> nx.DiGraph:
        """
        使用差异性过滤器提取骨干网络
        
        差异性过滤器基于假设：如果一个节点的连接权重分布是随机的，
        那么权重应该服从均匀分布。显著偏离这种分布的边被认为是重要的。
        
        Args:
            alpha: 显著性水平 (默认 0.05)
            direction: 过滤方向 ('out', 'in', 'both')
            
        Returns:
            提取的骨干网络
            
        References:
            Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). 
            Extracting the multiscale backbone of complex weighted networks. 
            PNAS, 106(16), 6483-6488.
        """
        self.logger.info(f"使用差异性过滤器提取骨干网络 (α={alpha}, direction={direction})")
        
        backbone = nx.DiGraph()
        backbone.add_nodes_from(self.G.nodes(data=True))
        
        significant_edges = 0
        
        for u, v, data in self.G.edges(data=True):
            weight = data['weight']
            
            # 计算p值
            p_values = []
            
            if direction in ['out', 'both']:
                # 出度方向的p值
                k_out = self.G.out_degree(u)
                if k_out > 1:
                    s_out = self.node_strengths[u]['out']
                    p_out = self._calculate_disparity_p_value(weight, s_out, k_out)
                    p_values.append(p_out)
            
            if direction in ['in', 'both']:
                # 入度方向的p值  
                k_in = self.G.in_degree(v)
                if k_in > 1:
                    s_in = self.node_strengths[v]['in']
                    p_in = self._calculate_disparity_p_value(weight, s_in, k_in)
                    p_values.append(p_in)
            
            # 判断边是否显著
            if p_values:
                # 使用最小p值（最保守的估计）
                min_p_value = min(p_values)
                if min_p_value < alpha:
                    backbone.add_edge(u, v, **data, p_value=min_p_value)
                    significant_edges += 1
        
        self.logger.info(f"差异性过滤器完成: {significant_edges}/{self.n_edges} 条边被保留 "
                        f"({significant_edges/self.n_edges*100:.1f}%)")
        
        return backbone
    
    def _calculate_disparity_p_value(self, weight: float, total_strength: float, 
                                   degree: int) -> float:
        """
        计算差异性过滤器的p值
        
        基于零假设：权重比例 p = weight/total_strength 来自均匀分布
        """
        if total_strength == 0 or degree <= 1:
            return 1.0
        
        p = weight / total_strength
        
        # 使用积分公式计算p值
        # P(X >= p) = (1-p)^(k-1) where k is degree
        p_value = (1 - p) ** (degree - 1)
        
        return p_value
    
    @timer_decorator  
    def extract_polarity_filter_backbone(self, alpha: float = 0.05,
                                        method: str = 'proximity') -> nx.DiGraph:
        """
        使用极性过滤器提取骨干网络
        
        极性过滤器基于网络中三元组结构的统计显著性，
        识别在局部拓扑中起重要作用的边。
        
        Args:
            alpha: 显著性水平
            method: 计算方法 ('proximity', 'similarity')
            
        Returns:
            提取的骨干网络
            
        References:
            Tumminello, M., Miccichè, S., Lillo, F., Piilo, J., & Mantegna, R. N. (2011).
            Statistically validated networks in bipartite complex systems.
            PloS one, 6(3), e17994.
        """
        self.logger.info(f"使用极性过滤器提取骨干网络 (α={alpha}, method={method})")
        
        backbone = nx.DiGraph()
        backbone.add_nodes_from(self.G.nodes(data=True))
        
        # 计算每条边的统计显著性
        significant_edges = 0
        
        for u, v, data in self.G.edges(data=True):
            weight = data['weight']
            
            # 计算基于三元组的统计量
            if method == 'proximity':
                p_value = self._calculate_proximity_p_value(u, v, weight)
            elif method == 'similarity':
                p_value = self._calculate_similarity_p_value(u, v, weight)
            else:
                raise ValueError(f"未知的极性过滤器方法: {method}")
            
            if p_value < alpha:
                backbone.add_edge(u, v, **data, p_value=p_value)
                significant_edges += 1
        
        self.logger.info(f"极性过滤器完成: {significant_edges}/{self.n_edges} 条边被保留 "
                        f"({significant_edges/self.n_edges*100:.1f}%)")
        
        return backbone
    
    def _calculate_proximity_p_value(self, u: str, v: str, weight: float) -> float:
        """
        计算基于邻近性的p值
        
        评估节点u和v之间的连接相对于它们的共同邻居是否显著
        """
        # 获取共同邻居
        u_neighbors = set(self.G.neighbors(u)) | set(self.G.predecessors(u))
        v_neighbors = set(self.G.neighbors(v)) | set(self.G.predecessors(v))
        common_neighbors = u_neighbors & v_neighbors
        
        if len(common_neighbors) == 0:
            return 1.0
        
        # 计算基于共同邻居的期望权重
        u_total_weight = self.node_strengths[u]['total']
        v_total_weight = self.node_strengths[v]['total']
        
        # 使用hypergeometric分布近似
        # 这是一个简化的实现，实际的极性过滤器可能更复杂
        expected_weight = (u_total_weight * v_total_weight) / (2 * self.n_edges)
        
        if expected_weight == 0:
            return 1.0
        
        # 使用泊松分布近似计算p值
        lambda_param = expected_weight
        p_value = 1 - stats.poisson.cdf(weight - 1, lambda_param)
        
        return min(p_value, 1.0)
    
    def _calculate_similarity_p_value(self, u: str, v: str, weight: float) -> float:
        """
        计算基于相似性的p值
        
        评估节点u和v的相似性是否足以解释它们之间的连接强度
        """
        # 计算Jaccard相似性
        u_neighbors = set(self.G.neighbors(u)) | set(self.G.predecessors(u))
        v_neighbors = set(self.G.neighbors(v)) | set(self.G.predecessors(v))
        
        intersection = len(u_neighbors & v_neighbors)
        union = len(u_neighbors | v_neighbors)
        
        if union == 0:
            jaccard_similarity = 0
        else:
            jaccard_similarity = intersection / union
        
        # 基于相似性计算期望权重
        max_possible_weight = min(self.node_strengths[u]['total'], 
                                self.node_strengths[v]['total'])
        expected_weight = jaccard_similarity * max_possible_weight
        
        if expected_weight == 0:
            return 1.0
        
        # 使用正态分布近似
        # 这是简化实现，实际可能需要更精确的统计模型
        std_dev = np.sqrt(expected_weight)
        if std_dev == 0:
            return 1.0
        
        z_score = (weight - expected_weight) / std_dev
        p_value = 1 - stats.norm.cdf(z_score)
        
        return min(p_value, 1.0)
    
    @timer_decorator
    def extract_mst_backbone(self, algorithm: str = 'kruskal') -> nx.DiGraph:
        """
        使用最小生成树提取骨干网络
        
        MST方法保留连接所有节点所需的最小权重边集合。
        对于有向图，我们首先转换为无向图，然后重新分配方向。
        
        Args:
            algorithm: MST算法 ('kruskal', 'prim')
            
        Returns:
            MST骨干网络
            
        References:
            Kruskal, J. B. (1956). On the shortest spanning subtree of a graph 
            and the traveling salesman problem. Proceedings of the American 
            Mathematical society, 7(1), 48-50.
        """
        self.logger.info(f"使用最小生成树提取骨干网络 (algorithm={algorithm})")
        
        # 转换为无向图用于MST计算
        # 对于有向图，我们需要聚合双向边的权重
        undirected_G = nx.Graph()
        
        # 添加节点
        undirected_G.add_nodes_from(self.G.nodes(data=True))
        
        # 聚合边权重
        edge_weights = {}
        for u, v, data in self.G.edges(data=True):
            edge_key = tuple(sorted([u, v]))
            weight = data['weight']
            
            if edge_key in edge_weights:
                edge_weights[edge_key] += weight
            else:
                edge_weights[edge_key] = weight
        
        # 添加聚合后的边
        for (u, v), weight in edge_weights.items():
            undirected_G.add_edge(u, v, weight=weight)
        
        # 计算MST
        if algorithm == 'kruskal':
            mst_edges = nx.minimum_spanning_edges(undirected_G, algorithm='kruskal', 
                                                data=True, weight='weight')
        elif algorithm == 'prim':
            mst_edges = nx.minimum_spanning_edges(undirected_G, algorithm='prim',
                                                data=True, weight='weight')
        else:
            raise ValueError(f"未知的MST算法: {algorithm}")
        
        # 创建骨干网络
        backbone = nx.DiGraph()
        backbone.add_nodes_from(self.G.nodes(data=True))
        
        # 将MST边重新添加为有向边
        mst_edge_set = set()
        for u, v, data in mst_edges:
            mst_edge_set.add((u, v))
            mst_edge_set.add((v, u))  # 两个方向都考虑
        
        # 从原图中添加MST包含的边
        added_edges = 0
        for u, v, data in self.G.edges(data=True):
            if (u, v) in mst_edge_set or (v, u) in mst_edge_set:
                backbone.add_edge(u, v, **data)
                added_edges += 1
        
        self.logger.info(f"最小生成树完成: {added_edges}/{self.n_edges} 条边被保留 "
                        f"({added_edges/self.n_edges*100:.1f}%)")
        
        return backbone
    
    def extract_all_backbones(self, df_alpha: float = 0.05, 
                            pf_alpha: float = 0.05,
                            mst_algorithm: str = 'kruskal') -> Dict[str, nx.DiGraph]:
        """
        一次性提取所有三种骨干网络
        
        Args:
            df_alpha: 差异性过滤器的显著性水平
            pf_alpha: 极性过滤器的显著性水平
            mst_algorithm: MST算法
            
        Returns:
            包含三种骨干网络的字典
        """
        self.logger.info("开始提取所有骨干网络...")
        
        backbones = {}
        
        try:
            # 差异性过滤器
            backbones['disparity_filter'] = self.extract_disparity_filter_backbone(df_alpha)
            
            # 极性过滤器
            backbones['polarity_filter'] = self.extract_polarity_filter_backbone(pf_alpha)
            
            # 最小生成树
            backbones['minimum_spanning_tree'] = self.extract_mst_backbone(mst_algorithm)
            
            self.logger.info("所有骨干网络提取完成")
            
        except Exception as e:
            self.logger.error(f"骨干网络提取过程中出错: {e}")
            raise
        
        return backbones
    
    def compare_backbones(self, backbones: Dict[str, nx.DiGraph]) -> pd.DataFrame:
        """
        比较不同骨干网络的统计特征
        
        Args:
            backbones: 骨干网络字典
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        # 原始网络统计
        original_stats = self._calculate_network_stats(self.G, "Original")
        comparison_data.append(original_stats)
        
        # 各骨干网络统计
        for method_name, backbone in backbones.items():
            stats = self._calculate_network_stats(backbone, method_name)
            comparison_data.append(stats)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        self.logger.info("骨干网络比较分析完成")
        
        return comparison_df
    
    def _calculate_network_stats(self, G: nx.DiGraph, method_name: str) -> Dict[str, Any]:
        """计算网络统计指标"""
        stats = {
            'method': method_name,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'edge_retention_rate': G.number_of_edges() / self.n_edges if self.n_edges > 0 else 0
        }
        
        # 计算连通性
        try:
            if G.number_of_edges() > 0:
                stats['avg_clustering'] = nx.average_clustering(G.to_undirected())
                
                # 转为无向图计算连通组件
                undirected = G.to_undirected()
                stats['connected_components'] = nx.number_connected_components(undirected)
                stats['largest_component_size'] = len(max(nx.connected_components(undirected), key=len))
            else:
                stats['avg_clustering'] = 0
                stats['connected_components'] = stats['nodes']
                stats['largest_component_size'] = 1 if stats['nodes'] > 0 else 0
                
        except Exception as e:
            self.logger.warning(f"计算{method_name}网络统计时出错: {e}")
            stats['avg_clustering'] = 0
            stats['connected_components'] = stats['nodes']
            stats['largest_component_size'] = 1 if stats['nodes'] > 0 else 0
        
        return stats


def extract_backbones_for_year(G: nx.DiGraph, year: int, 
                              methods: List[str] = ['df', 'pf', 'mst'],
                              **kwargs) -> Dict[str, nx.DiGraph]:
    """
    为单个年份提取骨干网络的便捷函数
    
    Args:
        G: 网络图
        year: 年份
        methods: 要使用的方法列表 ('df', 'pf', 'mst')
        **kwargs: 传递给各方法的参数
        
    Returns:
        骨干网络字典
    """
    logger.info(f"🔍 {year}: 开始提取骨干网络...")
    
    extractor = BackboneExtractor(G)
    backbones = {}
    
    if 'df' in methods:
        df_alpha = kwargs.get('df_alpha', 0.05)
        backbones['disparity_filter'] = extractor.extract_disparity_filter_backbone(df_alpha)
    
    if 'pf' in methods:
        pf_alpha = kwargs.get('pf_alpha', 0.05)
        pf_method = kwargs.get('pf_method', 'proximity')
        backbones['polarity_filter'] = extractor.extract_polarity_filter_backbone(pf_alpha, pf_method)
    
    if 'mst' in methods:
        mst_algorithm = kwargs.get('mst_algorithm', 'kruskal')
        backbones['minimum_spanning_tree'] = extractor.extract_mst_backbone(mst_algorithm)
    
    logger.info(f"✅ {year}: 骨干网络提取完成")
    
    return backbones


def analyze_backbone_evolution(annual_networks: Dict[int, nx.DiGraph],
                             methods: List[str] = ['df', 'pf', 'mst'],
                             **kwargs) -> pd.DataFrame:
    """
    分析骨干网络随时间的演化
    
    Args:
        annual_networks: 年度网络字典
        methods: 要分析的方法
        **kwargs: 传递给提取方法的参数
        
    Returns:
        演化分析结果DataFrame
    """
    logger.info(f"🌟 开始分析骨干网络演化 - {len(annual_networks)} 个年份")
    
    evolution_data = []
    
    for year in sorted(annual_networks.keys()):
        G = annual_networks[year]
        
        try:
            # 提取骨干网络
            backbones = extract_backbones_for_year(G, year, methods, **kwargs)
            
            # 分析每种方法
            extractor = BackboneExtractor(G)
            year_comparison = extractor.compare_backbones(backbones)
            year_comparison['year'] = year
            
            evolution_data.append(year_comparison)
            
        except Exception as e:
            logger.error(f"❌ {year}年骨干网络分析失败: {e}")
            continue
    
    if evolution_data:
        # 合并所有年份的数据
        all_data = pd.concat(evolution_data, ignore_index=True)
        logger.info(f"✅ 骨干网络演化分析完成")
        return all_data
    else:
        logger.error("所有年份分析都失败了")
        return pd.DataFrame()


# 导出的主要函数
__all__ = [
    'BackboneExtractor',
    'extract_backbones_for_year', 
    'analyze_backbone_evolution'
]