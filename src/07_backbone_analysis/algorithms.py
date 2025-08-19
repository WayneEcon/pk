#!/usr/bin/env python3
"""
骨干网络提取算法集成模块
==========================

整合 Disparity Filter, Maximum Spanning Tree, 和 Pólya Urn Filter 三种核心算法。
保持算法的科学严谨性，特别是对入度/出度的分别检验和FDR多重检验校正。

核心算法：
1. disparity_filter() - 基于Serrano et al. (2009)的统计显著性检验
2. maximum_spanning_tree() - 基于图论的最优连通性算法
3. polya_urn_filter() - 基于Pólya Urn模型的补充验证算法

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_disparity_pvalue(weight: float, node_strength: float, node_degree: int) -> float:
    """
    计算单条边的Disparity Filter p值
    
    基于Serrano et al. (2009)的正确公式：
    在null model下，权重按节点强度随机分配，每条边权重比例遵循Beta分布
    
    Args:
        weight: 边的权重
        node_strength: 节点的总强度（所有出边权重之和）
        node_degree: 节点的度数（出边数量）
        
    Returns:
        该边在null model下的p值
    """
    
    if node_degree <= 1 or node_strength <= 0 or weight <= 0:
        return 1.0  # 度为1、强度为0或权重为0时，无法计算显著性
    
    # 计算归一化权重比例
    p_ij = weight / node_strength
    
    # Serrano et al. (2009)的正确公式
    # P(X >= p_ij) = (1 - p_ij)^(k-1)
    # 这基于在k个边中随机分配权重的null model
    k = node_degree
    
    # 避免数值计算问题
    if p_ij >= 1.0:
        return 0.0
    if p_ij <= 0.0:
        return 1.0
    
    # 计算p值：P(proportion >= p_ij | null model)
    p_value = (1.0 - p_ij) ** (k - 1)
    
    return min(p_value, 1.0)


def benjamini_hochberg_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR多重比较校正
    
    Args:
        p_values: 原始p值数组
        alpha: FDR水平
        
    Returns:
        FDR校正后的显著性判断（布尔数组）
    """
    
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    
    # 按p值排序
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # BH临界值：p_i <= (i/n) * alpha
    critical_values = (np.arange(1, n + 1) / n) * alpha
    
    # 找到最大的满足条件的索引
    significant_sorted = sorted_p_values <= critical_values
    
    # 如果有显著的p值，找到最大的索引
    if np.any(significant_sorted):
        max_significant_idx = np.max(np.where(significant_sorted)[0])
        # 前max_significant_idx+1个都是显著的
        significant_sorted[:max_significant_idx + 1] = True
        significant_sorted[max_significant_idx + 1:] = False
    
    # 恢复原始顺序
    significant = np.empty(n, dtype=bool)
    significant[sorted_indices] = significant_sorted
    
    return significant


def disparity_filter(G: nx.Graph, 
                    alpha: float = 0.05,
                    fdr_correction: bool = True,
                    weight_attr: str = 'weight',
                    directed: bool = None) -> nx.Graph:
    """
    对网络应用Disparity Filter算法
    
    **关键要求**: 对有向图必须分别检验入度和出度，并应用FDR多重检验校正
    
    Args:
        G: 输入网络（NetworkX图对象）
        alpha: 显著性水平
        fdr_correction: 是否应用FDR多重比较校正
        weight_attr: 边权重属性名
        directed: 是否作为有向图处理，None时自动检测
        
    Returns:
        过滤后的骨干网络
    """
    
    logger.info(f"🔍 开始应用Disparity Filter (α={alpha}, FDR={fdr_correction})...")
    
    if directed is None:
        directed = G.is_directed()
    
    # 复制图以避免修改原图
    G_filtered = G.copy()
    
    # 收集所有边的p值
    edge_pvalues = []
    edge_list = []
    
    # 处理每个节点的出边和入边
    for node in G.nodes():
        if directed:
            # 有向图：分别处理出边和入边
            out_edges = list(G.out_edges(node, data=True))
            in_edges = list(G.in_edges(node, data=True))
            
            # 处理出边
            if len(out_edges) > 1:
                out_strength = sum([data.get(weight_attr, 1.0) for _, _, data in out_edges])
                out_degree = len(out_edges)
                
                for source, target, data in out_edges:
                    weight = data.get(weight_attr, 1.0)
                    p_value = calculate_disparity_pvalue(weight, out_strength, out_degree)
                    edge_pvalues.append(p_value)
                    edge_list.append((source, target, 'out'))
            
            # 处理入边
            if len(in_edges) > 1:
                in_strength = sum([data.get(weight_attr, 1.0) for _, _, data in in_edges])
                in_degree = len(in_edges)
                
                for source, target, data in in_edges:
                    weight = data.get(weight_attr, 1.0)
                    p_value = calculate_disparity_pvalue(weight, in_strength, in_degree)
                    edge_pvalues.append(p_value)
                    edge_list.append((source, target, 'in'))
        
        else:
            # 无向图：只处理度大于1的节点
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                node_strength = sum([G[node][neighbor].get(weight_attr, 1.0) 
                                   for neighbor in neighbors])
                node_degree = len(neighbors)
                
                for neighbor in neighbors:
                    # 避免重复计算同一条边
                    if node < neighbor or directed:
                        weight = G[node][neighbor].get(weight_attr, 1.0)
                        p_value = calculate_disparity_pvalue(weight, node_strength, node_degree)
                        edge_pvalues.append(p_value)
                        edge_list.append((node, neighbor, 'undirected'))
    
    if not edge_pvalues:
        logger.warning("⚠️ 没有足够的边进行显著性检验")
        return G_filtered
    
    edge_pvalues = np.array(edge_pvalues)
    
    # 应用FDR校正或直接比较alpha
    if fdr_correction:
        significant_edges = benjamini_hochberg_fdr(edge_pvalues, alpha)
        logger.info(f"📊 FDR校正后保留边数: {np.sum(significant_edges)}/{len(edge_pvalues)}")
    else:
        significant_edges = edge_pvalues <= alpha
        logger.info(f"📊 直接比较后保留边数: {np.sum(significant_edges)}/{len(edge_pvalues)}")
    
    # 构建要删除的边集合
    edges_to_remove = set()
    
    for i, (source, target, direction) in enumerate(edge_list):
        if not significant_edges[i]:
            if direction == 'undirected':
                edges_to_remove.add((source, target))
                if not directed:  # 无向图的对称性
                    edges_to_remove.add((target, source))
            else:
                edges_to_remove.add((source, target))
    
    # 删除不显著的边
    G_filtered.remove_edges_from(edges_to_remove)
    
    # 添加过滤信息到图属性
    G_filtered.graph['backbone_method'] = 'disparity_filter'
    G_filtered.graph['alpha'] = alpha
    G_filtered.graph['fdr_correction'] = fdr_correction
    G_filtered.graph['original_edges'] = G.number_of_edges()
    G_filtered.graph['filtered_edges'] = G_filtered.number_of_edges()
    G_filtered.graph['retention_rate'] = G_filtered.number_of_edges() / G.number_of_edges()
    
    logger.info(f"✅ Disparity Filter完成:")
    logger.info(f"   原始边数: {G.number_of_edges():,}")
    logger.info(f"   保留边数: {G_filtered.number_of_edges():,}")
    logger.info(f"   保留率: {G_filtered.graph['retention_rate']:.1%}")
    
    return G_filtered


def symmetrize_graph(G: nx.DiGraph, weight_attr: str = 'weight', 
                    method: str = 'max') -> nx.Graph:
    """
    将有向图对称化为无向图
    
    Args:
        G: 有向图
        weight_attr: 边权重属性名
        method: 对称化方法 ('max', 'sum', 'mean')
            - 'max': 取双向边权重的最大值
            - 'sum': 取双向边权重的和
            - 'mean': 取双向边权重的平均值
            
    Returns:
        对称化后的无向图
    """
    
    logger.info(f"🔄 对称化有向图 (方法: {method})...")
    
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(G.nodes(data=True))
    
    # 收集所有边的权重
    edge_weights = {}
    
    for source, target, data in G.edges(data=True):
        weight = data.get(weight_attr, 1.0)
        
        # 标准化边的表示 (较小节点在前)
        edge_key = tuple(sorted([source, target]))
        
        if edge_key not in edge_weights:
            edge_weights[edge_key] = []
        
        edge_weights[edge_key].append(weight)
    
    # 根据方法计算最终权重
    for (node1, node2), weights in edge_weights.items():
        if method == 'max':
            final_weight = max(weights)
        elif method == 'sum':
            final_weight = sum(weights)
        elif method == 'mean':
            final_weight = sum(weights) / len(weights)
        else:
            raise ValueError(f"不支持的对称化方法: {method}")
        
        G_undirected.add_edge(node1, node2, **{weight_attr: final_weight})
    
    logger.info(f"   有向边: {G.number_of_edges():,} -> 无向边: {G_undirected.number_of_edges():,}")
    
    return G_undirected


def maximum_spanning_tree(G: nx.Graph, 
                         weight_attr: str = 'weight',
                         algorithm: str = 'kruskal') -> nx.Graph:
    """
    计算最大生成树
    
    Args:
        G: 输入图（无向图）
        weight_attr: 边权重属性名
        algorithm: 算法选择 ('kruskal' 或 'prim')
        
    Returns:
        最大生成树
    """
    
    logger.info(f"🌳 计算最大生成树 (算法: {algorithm})...")
    
    if G.number_of_nodes() == 0:
        logger.warning("⚠️ 输入图为空")
        return nx.Graph()
    
    # NetworkX的MST算法默认计算最小生成树
    # 要计算最大生成树，需要将权重取负数
    G_negative = G.copy()
    
    for source, target, data in G_negative.edges(data=True):
        original_weight = data.get(weight_attr, 1.0)
        G_negative[source][target][weight_attr] = -original_weight
    
    # 计算最小生成树（负权重）
    if algorithm == 'kruskal':
        mst_negative = nx.minimum_spanning_tree(G_negative, weight=weight_attr, algorithm='kruskal')
    elif algorithm == 'prim':
        mst_negative = nx.minimum_spanning_tree(G_negative, weight=weight_attr, algorithm='prim')
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 恢复正权重
    mst = nx.Graph()
    mst.add_nodes_from(G.nodes(data=True))
    
    total_weight = 0
    for source, target, data in mst_negative.edges(data=True):
        original_weight = -data[weight_attr]  # 恢复正权重
        mst.add_edge(source, target, **{weight_attr: original_weight})
        total_weight += original_weight
    
    # 添加生成树信息
    mst.graph['backbone_method'] = 'maximum_spanning_tree'
    mst.graph['algorithm'] = algorithm
    mst.graph['original_edges'] = G.number_of_edges()
    mst.graph['mst_edges'] = mst.number_of_edges()
    mst.graph['total_mst_weight'] = total_weight
    mst.graph['retention_rate'] = mst.number_of_edges() / G.number_of_edges()
    
    logger.info(f"✅ 最大生成树完成:")
    logger.info(f"   原始边数: {G.number_of_edges():,}")
    logger.info(f"   MST边数: {mst.number_of_edges():,}")
    logger.info(f"   总权重: {total_weight:.2f}")
    logger.info(f"   保留率: {mst.graph['retention_rate']:.1%}")
    
    return mst


def polya_urn_filter(G: nx.Graph, 
                    beta: float = 0.05,
                    weight_attr: str = 'weight') -> nx.Graph:
    """
    基于Pólya Urn模型的骨干网络提取算法
    
    该算法作为Disparity Filter的补充验证方法，基于不同的零假设：
    - 零假设：边权重遵循Pólya分布
    - 用于交叉验证DF结果的稳健性
    
    Args:
        G: 输入网络
        beta: 显著性阈值
        weight_attr: 边权重属性名
        
    Returns:
        过滤后的骨干网络
    """
    
    logger.info(f"🎲 开始应用Pólya Urn Filter (β={beta})...")
    
    G_filtered = G.copy()
    edges_to_remove = []
    
    # 为每个节点应用Pólya Urn检验
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) <= 1:
            continue
            
        # 获取节点的所有边权重
        weights = [G[node][neighbor].get(weight_attr, 1.0) for neighbor in neighbors]
        total_weight = sum(weights)
        
        # Pólya Urn模型下的p值计算
        for i, neighbor in enumerate(neighbors):
            weight = weights[i]
            
            # 简化的Pólya Urn p值计算
            # 实际实现中应该使用更精确的统计检验
            p_ij = weight / total_weight
            k = len(neighbors)
            
            # Beta分布近似（简化版本）
            expected_prob = 1.0 / k
            variance = expected_prob * (1 - expected_prob) / (k + 1)
            
            if variance > 0:
                z_score = (p_ij - expected_prob) / np.sqrt(variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 双尾检验
            else:
                p_value = 1.0
            
            # 如果边不显著，标记为删除
            if p_value > beta:
                if node < neighbor:  # 避免重复删除
                    edges_to_remove.append((node, neighbor))
    
    # 删除不显著的边
    G_filtered.remove_edges_from(edges_to_remove)
    
    # 添加过滤信息
    G_filtered.graph['backbone_method'] = 'polya_urn_filter'
    G_filtered.graph['beta'] = beta
    G_filtered.graph['original_edges'] = G.number_of_edges()
    G_filtered.graph['filtered_edges'] = G_filtered.number_of_edges()
    G_filtered.graph['retention_rate'] = G_filtered.number_of_edges() / G.number_of_edges()
    
    logger.info(f"✅ Pólya Urn Filter完成:")
    logger.info(f"   原始边数: {G.number_of_edges():,}")
    logger.info(f"   保留边数: {G_filtered.number_of_edges():,}")
    logger.info(f"   保留率: {G_filtered.graph['retention_rate']:.1%}")
    
    return G_filtered


def apply_all_algorithms(G: nx.Graph, 
                        alpha_values: List[float] = [0.01, 0.05, 0.1],
                        beta: float = 0.05,
                        weight_attr: str = 'weight') -> Dict[str, nx.Graph]:
    """
    对单个网络应用所有骨干提取算法
    
    Args:
        G: 输入网络
        alpha_values: DF算法的alpha值列表
        beta: PF算法的beta值
        weight_attr: 边权重属性名
        
    Returns:
        包含所有算法结果的字典
    """
    
    results = {}
    
    # 1. Disparity Filter (多个alpha值)
    for alpha in alpha_values:
        key = f'disparity_filter_{alpha}'
        results[key] = disparity_filter(G, alpha=alpha, fdr_correction=True, weight_attr=weight_attr)
    
    # 2. Maximum Spanning Tree
    if G.is_directed():
        G_sym = symmetrize_graph(G, weight_attr=weight_attr, method='max')
        results['mst'] = maximum_spanning_tree(G_sym, weight_attr=weight_attr)
    else:
        results['mst'] = maximum_spanning_tree(G, weight_attr=weight_attr)
    
    # 3. Pólya Urn Filter
    if G.is_directed():
        G_sym = symmetrize_graph(G, weight_attr=weight_attr, method='max')
        results['polya_urn'] = polya_urn_filter(G_sym, beta=beta, weight_attr=weight_attr)
    else:
        results['polya_urn'] = polya_urn_filter(G, beta=beta, weight_attr=weight_attr)
    
    return results


def batch_backbone_extraction(networks: Dict[int, nx.Graph],
                             alpha_values: List[float] = [0.01, 0.05, 0.1],
                             beta: float = 0.05,
                             weight_attr: str = 'weight') -> Dict[str, Dict[int, nx.Graph]]:
    """
    批量对多年网络数据应用骨干提取算法
    
    Args:
        networks: 年份到网络的映射字典
        alpha_values: DF算法的alpha值列表
        beta: PF算法的beta值
        weight_attr: 边权重属性名
        
    Returns:
        嵌套字典: {algorithm: {year: backbone_network}}
    """
    
    logger.info(f"🚀 开始批量骨干提取分析...")
    logger.info(f"   年份范围: {min(networks.keys())}-{max(networks.keys())}")
    logger.info(f"   算法: DF (α={alpha_values}), MST, PF (β={beta})")
    
    # 初始化结果字典
    batch_results = {}
    algorithm_keys = [f'disparity_filter_{alpha}' for alpha in alpha_values] + ['mst', 'polya_urn']
    
    for key in algorithm_keys:
        batch_results[key] = {}
    
    # 对每年网络应用所有算法
    for year, network in networks.items():
        logger.info(f"⚡ 处理{year}年网络 ({network.number_of_nodes()}节点, {network.number_of_edges()}边)...")
        
        try:
            year_results = apply_all_algorithms(
                network, 
                alpha_values=alpha_values, 
                beta=beta, 
                weight_attr=weight_attr
            )
            
            # 将结果分配到对应算法
            for alg_key, backbone_net in year_results.items():
                batch_results[alg_key][year] = backbone_net
                
        except Exception as e:
            logger.error(f"❌ {year}年数据处理失败: {e}")
            continue
    
    # 输出批量处理统计
    logger.info("📊 批量处理统计摘要:")
    for alg_key, year_networks in batch_results.items():
        if year_networks:
            retention_rates = [G.graph.get('retention_rate', 0) for G in year_networks.values()]
            avg_retention = np.mean(retention_rates)
            logger.info(f"   {alg_key}: 平均保留率 = {avg_retention:.1%}")
    
    return batch_results