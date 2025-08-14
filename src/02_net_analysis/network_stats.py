#!/usr/bin/env python3
"""
网络统计模块
负责计算网络的各种统计指标
"""

import networkx as nx
import numpy as np
import logging
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple
from .utils import setup_path, safe_divide, FOCUS_COUNTRIES

# 确保路径设置
setup_path()

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def _cached_edge_weights_stats(edges_tuple: Tuple[Tuple[str, str, float], ...]) -> dict:
    """
    缓存的边权重统计计算（优化版）
    
    Args:
        edges_tuple: 边的元组，格式为((source, target, weight), ...)
        
    Returns:
        包含权重统计信息的字典
    """
    if not edges_tuple:
        return {
            'total_trade_value': 0,
            'avg_trade_value': 0,
            'max_trade_value': 0,
            'median_trade_value': 0,
            'std_trade_value': 0
        }
    
    weights = [edge[2] for edge in edges_tuple]
    return {
        'total_trade_value': sum(weights),
        'avg_trade_value': np.mean(weights),
        'max_trade_value': max(weights),
        'median_trade_value': np.median(weights),
        'std_trade_value': np.std(weights)
    }

def calculate_basic_network_stats(G: nx.DiGraph, year: int) -> dict:
    """
    计算网络的基础统计信息（优化缓存版）
    
    Args:
        G: NetworkX有向图对象，必须包含权重信息
        year: 统计年份，用于记录和日志
        
    Returns:
        包含网络统计指标的字典
        
    Raises:
        ValueError: 当图对象无效或缺少权重信息时
        
    Example:
        >>> G = nx.DiGraph()
        >>> G.add_edge('A', 'B', weight=100)
        >>> stats = calculate_basic_network_stats(G, 2020)
        >>> print(stats['nodes'])  # 输出: 2
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("输入必须是NetworkX有向图对象")
    
    logger.debug(f"     {year}: 开始计算网络统计...")
    
    # 基础拓扑统计
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    stats = {
        'year': year,
        'nodes': num_nodes,
        'edges': num_edges,
        'density': nx.density(G) if num_nodes > 1 else 0,
        'is_connected': nx.is_weakly_connected(G) if num_nodes > 0 else False,
        'num_weakly_connected_components': nx.number_weakly_connected_components(G) if num_nodes > 0 else 0
    }
    
    # 权重统计（使用缓存优化）
    if num_edges > 0:
        # 创建可缓存的边权重元组
        edges_with_weights = tuple(
            (u, v, data.get('weight', 0)) 
            for u, v, data in G.edges(data=True)
        )
        
        # 使用缓存的统计函数
        weight_stats = _cached_edge_weights_stats(edges_with_weights)
        stats.update(weight_stats)
    else:
        stats.update({
            'total_trade_value': 0,
            'avg_trade_value': 0,
            'max_trade_value': 0,
            'median_trade_value': 0,
            'std_trade_value': 0
        })
    
    # 计算核心国家统计（参数化，预计算strength）
    if hasattr(G, '_node_strengths'):
        # 使用预计算的strength（如果存在）
        node_strengths = G._node_strengths
    else:
        # 计算并缓存所有节点的strength
        node_strengths = {}
        for node in G.nodes():
            out_strength = G.out_degree(node, weight='weight')
            in_strength = G.in_degree(node, weight='weight')
            node_strengths[node] = {
                'out_strength': out_strength,
                'in_strength': in_strength,
                'total_strength': out_strength + in_strength
            }
        G._node_strengths = node_strengths  # 缓存计算结果
    
    # 为关注的国家添加统计
    for country_code, country_name in FOCUS_COUNTRIES.items():
        if country_code in node_strengths:
            strengths = node_strengths[country_code]
            stats.update({
                f'{country_code.lower()}_out_strength': strengths['out_strength'],
                f'{country_code.lower()}_in_strength': strengths['in_strength'], 
                f'{country_code.lower()}_total_strength': strengths['total_strength'],
                f'{country_code.lower()}_trade_share': safe_divide(
                    strengths['total_strength'], stats['total_trade_value']
                )
            })
        else:
            stats.update({
                f'{country_code.lower()}_out_strength': 0,
                f'{country_code.lower()}_in_strength': 0,
                f'{country_code.lower()}_total_strength': 0,
                f'{country_code.lower()}_trade_share': 0
            })
    
    logger.debug(f"     {year}: 统计计算完成")
    return stats