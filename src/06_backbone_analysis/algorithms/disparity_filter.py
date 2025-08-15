#!/usr/bin/env python3
"""
Disparity Filter算法实现
=============================

基于Serrano et al. (2009)的统计显著性检验方法，
识别网络中每个节点的"异常强"连接，过滤掉统计上不显著的噪声连接。

理论基础：
- 零假设H0: 权重按节点强度随机分配
- 检验统计量: 基于多项分布的p值计算
- 多重比较校正: Benjamini-Hochberg FDR控制

参考文献：
Serrano, M. Á., Boguná, M., & Vespignani, A. (2009). 
Extracting the multiscale backbone of complex weighted networks. 
Proceedings of the national academy of sciences, 106(16), 6483-6488.

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
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
    
    # 处理每个节点的出边
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

def apply_disparity_filter_batch(networks: Dict[int, nx.Graph],
                                alpha_values: List[float] = [0.01, 0.05, 0.1],
                                fdr_correction: bool = True,
                                weight_attr: str = 'weight') -> Dict[str, Dict[int, nx.Graph]]:
    """
    批量应用Disparity Filter到多年网络数据
    
    Args:
        networks: 年份到网络的映射字典
        alpha_values: 要测试的显著性水平列表
        fdr_correction: 是否应用FDR校正
        weight_attr: 边权重属性名
        
    Returns:
        嵌套字典: {f'alpha_{alpha}': {year: backbone_network}}
    """
    
    logger.info(f"🚀 开始批量Disparity Filter分析...")
    logger.info(f"   年份范围: {min(networks.keys())}-{max(networks.keys())}")
    logger.info(f"   Alpha值: {alpha_values}")
    logger.info(f"   FDR校正: {fdr_correction}")
    
    results = {}
    
    for alpha in alpha_values:
        alpha_key = f'alpha_{alpha}'
        results[alpha_key] = {}
        
        logger.info(f"⚡ 处理α={alpha}...")
        
        for year in sorted(networks.keys()):
            G = networks[year]
            logger.info(f"   处理{year}年网络 ({G.number_of_nodes()}节点, {G.number_of_edges()}边)...")
            
            try:
                G_backbone = disparity_filter(
                    G, 
                    alpha=alpha,
                    fdr_correction=fdr_correction,
                    weight_attr=weight_attr
                )
                
                results[alpha_key][year] = G_backbone
                
            except Exception as e:
                logger.error(f"❌ {year}年数据处理失败: {e}")
                continue
    
    # 统计摘要
    logger.info("📊 批量处理统计摘要:")
    for alpha_key, year_networks in results.items():
        retention_rates = [G.graph['retention_rate'] for G in year_networks.values()]
        if retention_rates:
            avg_retention = np.mean(retention_rates)
            logger.info(f"   {alpha_key}: 平均保留率 = {avg_retention:.1%}")
    
    return results

if __name__ == "__main__":
    # 测试代码
    logger.info("🧪 测试Disparity Filter算法...")
    
    # 创建测试网络
    G_test = nx.DiGraph()
    
    # 添加测试边（权重差异很大）
    edges = [
        ('A', 'B', 10.0), ('A', 'C', 1.0), ('A', 'D', 0.1),
        ('B', 'C', 5.0), ('B', 'D', 2.0),
        ('C', 'D', 8.0)
    ]
    
    for source, target, weight in edges:
        G_test.add_edge(source, target, weight=weight)
    
    # 应用Disparity Filter
    G_backbone = disparity_filter(G_test, alpha=0.05, fdr_correction=True)
    
    print("🎉 测试完成!")
    print(f"原始图: {G_test.number_of_edges()} 条边")
    print(f"骨干图: {G_backbone.number_of_edges()} 条边")
    print(f"保留的边: {list(G_backbone.edges())}")