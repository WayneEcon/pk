#!/usr/bin/env python3
"""
Maximum Spanning Tree算法实现
=============================

基于图论的最优连通性保证算法，保持网络连通性的同时，
选择权重最大的边构成生成树，识别全球贸易的"主干道"。

核心思想：
- 保持所有节点连通的前提下，选择权重最大的边
- 对于有向图，先进行对称化处理
- 适用于结构可视化和关键路径识别

算法特点：
- 确定性结果：给定网络的MST是唯一的（权重不重复时）
- 连通性保证：所有节点都在同一个连通分量中
- 权重最优：边权重之和最大

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def maximum_spanning_forest(G: nx.Graph, 
                          weight_attr: str = 'weight',
                          algorithm: str = 'kruskal') -> nx.Graph:
    """
    计算最大生成森林（处理非连通图）
    
    Args:
        G: 输入图（可能非连通）
        weight_attr: 边权重属性名  
        algorithm: 算法选择
        
    Returns:
        最大生成森林
    """
    
    logger.info("🌲 计算最大生成森林...")
    
    # 找到所有连通分量
    connected_components = list(nx.connected_components(G))
    logger.info(f"   发现 {len(connected_components)} 个连通分量")
    
    # 初始化结果图
    msf = nx.Graph()
    msf.add_nodes_from(G.nodes(data=True))
    
    total_weight = 0
    total_edges = 0
    
    # 对每个连通分量计算MST
    for i, component in enumerate(connected_components):
        if len(component) < 2:
            logger.info(f"   分量 {i+1}: 孤立节点，跳过")
            continue
            
        # 提取子图
        subgraph = G.subgraph(component).copy()
        
        logger.info(f"   分量 {i+1}: {len(component)} 节点, {subgraph.number_of_edges()} 边")
        
        # 计算该分量的MST
        component_mst = maximum_spanning_tree(subgraph, weight_attr, algorithm)
        
        # 合并到总结果中
        msf.add_edges_from(component_mst.edges(data=True))
        
        total_weight += component_mst.graph.get('total_mst_weight', 0)
        total_edges += component_mst.number_of_edges()
    
    # 添加森林信息
    msf.graph['backbone_method'] = 'maximum_spanning_forest'
    msf.graph['algorithm'] = algorithm
    msf.graph['original_edges'] = G.number_of_edges()
    msf.graph['msf_edges'] = total_edges
    msf.graph['total_msf_weight'] = total_weight
    msf.graph['retention_rate'] = total_edges / G.number_of_edges() if G.number_of_edges() > 0 else 0
    msf.graph['connected_components'] = len(connected_components)
    
    logger.info(f"✅ 最大生成森林完成:")
    logger.info(f"   连通分量: {len(connected_components)}")
    logger.info(f"   总边数: {total_edges:,}")
    logger.info(f"   总权重: {total_weight:.2f}")
    
    return msf

def apply_mst_to_directed_graph(G: nx.DiGraph,
                              weight_attr: str = 'weight',
                              symmetrize_method: str = 'max',
                              algorithm: str = 'kruskal') -> nx.Graph:
    """
    对有向图应用最大生成树算法
    
    Args:
        G: 有向输入图
        weight_attr: 边权重属性名
        symmetrize_method: 对称化方法
        algorithm: MST算法
        
    Returns:
        最大生成树（无向图）
    """
    
    logger.info("🎯 对有向图应用MST算法...")
    
    # 1. 对称化
    G_sym = symmetrize_graph(G, weight_attr, symmetrize_method)
    
    # 2. 计算MST/MSF
    if nx.is_connected(G_sym):
        mst = maximum_spanning_tree(G_sym, weight_attr, algorithm)
    else:
        mst = maximum_spanning_forest(G_sym, weight_attr, algorithm)
    
    # 3. 添加有向图处理信息
    mst.graph['original_directed'] = True
    mst.graph['symmetrize_method'] = symmetrize_method
    
    return mst

def apply_mst_batch(networks: Dict[int, nx.Graph],
                   weight_attr: str = 'weight',
                   symmetrize_method: str = 'max',
                   algorithm: str = 'kruskal') -> Dict[int, nx.Graph]:
    """
    批量应用MST算法到多年网络数据
    
    Args:
        networks: 年份到网络的映射字典
        weight_attr: 边权重属性名
        symmetrize_method: 对称化方法（仅对有向图）
        algorithm: MST算法
        
    Returns:
        年份到MST的映射字典
    """
    
    logger.info(f"🚀 开始批量MST分析...")
    logger.info(f"   年份范围: {min(networks.keys())}-{max(networks.keys())}")
    logger.info(f"   对称化方法: {symmetrize_method}")
    logger.info(f"   MST算法: {algorithm}")
    
    results = {}
    
    for year in sorted(networks.keys()):
        G = networks[year]
        logger.info(f"⚡ 处理{year}年网络 ({G.number_of_nodes()}节点, {G.number_of_edges()}边)...")
        
        try:
            if isinstance(G, nx.DiGraph):
                mst = apply_mst_to_directed_graph(
                    G, weight_attr, symmetrize_method, algorithm
                )
            else:
                if nx.is_connected(G):
                    mst = maximum_spanning_tree(G, weight_attr, algorithm)
                else:
                    mst = maximum_spanning_forest(G, weight_attr, algorithm)
            
            results[year] = mst
            
        except Exception as e:
            logger.error(f"❌ {year}年数据处理失败: {e}")
            continue
    
    # 统计摘要
    logger.info("📊 批量MST处理统计摘要:")
    if results:
        retention_rates = [G.graph['retention_rate'] for G in results.values()]
        total_weights = [G.graph.get('total_mst_weight', 0) for G in results.values()]
        
        logger.info(f"   平均保留率: {np.mean(retention_rates):.1%}")
        logger.info(f"   权重变化: {min(total_weights):.0f} - {max(total_weights):.0f}")
    
    return results

if __name__ == "__main__":
    # 测试代码
    logger.info("🧪 测试Maximum Spanning Tree算法...")
    
    # 创建测试网络（加权无向图）
    G_test = nx.Graph()
    
    # 添加测试边
    edges = [
        ('A', 'B', 10), ('A', 'C', 5), ('A', 'D', 8),
        ('B', 'C', 3), ('B', 'D', 2), ('B', 'E', 7),
        ('C', 'D', 1), ('C', 'E', 6),
        ('D', 'E', 9)
    ]
    
    for source, target, weight in edges:
        G_test.add_edge(source, target, weight=weight)
    
    print("原始图的边权重:")
    for source, target, data in G_test.edges(data=True):
        print(f"  {source}-{target}: {data['weight']}")
    
    # 计算MST
    mst = maximum_spanning_tree(G_test)
    
    print("\nMST的边权重:")
    for source, target, data in mst.edges(data=True):
        print(f"  {source}-{target}: {data['weight']}")
    
    print(f"\n总权重: {mst.graph['total_mst_weight']}")
    
    # 测试有向图
    logger.info("\n🧪 测试有向图对称化...")
    G_directed = nx.DiGraph()
    G_directed.add_edge('X', 'Y', weight=5)
    G_directed.add_edge('Y', 'X', weight=3)
    G_directed.add_edge('X', 'Z', weight=7)
    
    mst_directed = apply_mst_to_directed_graph(G_directed, symmetrize_method='max')
    
    print("🎉 测试完成!")