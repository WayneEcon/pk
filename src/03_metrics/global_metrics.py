#!/usr/bin/env python3
"""
全局网络指标计算模块

负责计算整个网络的全局拓扑特征：
- 网络密度
- 连通性指标  
- 路径长度指标（修正加权路径计算）
- 聚类系数
- 网络效率
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any
from utils import (
    setup_logger, validate_graph, safe_divide, timer_decorator, 
    handle_computation_error, add_distance_weights, get_node_sample
)

logger = setup_logger(__name__)

@timer_decorator
def calculate_density_metrics(G: nx.DiGraph, year: int) -> Dict[str, Any]:
    """
    计算网络密度相关指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        密度指标字典
    """
    validate_graph(G, "calculate_density_metrics")
    logger.info(f"     {year}: 计算网络密度...")
    
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # 基本密度
        density = nx.density(G)
        
        # 最大可能边数 (有向图)
        max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 0
        
        # 权重密度 (基于边权重)
        edge_weights = [data.get('weight', 0) for _, _, data in G.edges(data=True)]
        total_weight = sum(edge_weights)
        avg_edge_weight = safe_divide(total_weight, n_edges)
        
        return {
            'year': year,
            'density': density,
            'nodes': n_nodes,
            'edges': n_edges,
            'max_possible_edges': max_edges,
            'edge_coverage_ratio': safe_divide(n_edges, max_edges),
            'total_weight': total_weight,
            'avg_edge_weight': avg_edge_weight,
            'weight_density': total_weight / (max_edges * avg_edge_weight) if max_edges > 0 and avg_edge_weight > 0 else 0
        }
        
    except Exception as e:
        return handle_computation_error("calculate_density_metrics", year, e, 
                                      {'year': year, 'density': 0, 'nodes': 0, 'edges': 0})

@timer_decorator
def calculate_connectivity_metrics(G: nx.DiGraph, year: int) -> Dict[str, Any]:
    """
    计算连通性指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        连通性指标字典
    """
    validate_graph(G, "calculate_connectivity_metrics")
    logger.info(f"     {year}: 计算连通性指标...")
    
    try:
        # 强连通性
        is_strongly_connected = nx.is_strongly_connected(G)
        num_strongly_connected_components = nx.number_strongly_connected_components(G)
        
        # 弱连通性
        is_weakly_connected = nx.is_weakly_connected(G)
        num_weakly_connected_components = nx.number_weakly_connected_components(G)
        
        # 最大强连通分量大小
        if num_strongly_connected_components > 0:
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            largest_scc_size = len(largest_scc)
            largest_scc_ratio = safe_divide(largest_scc_size, G.number_of_nodes())
        else:
            largest_scc_size = 0
            largest_scc_ratio = 0
        
        # 最大弱连通分量大小
        if num_weakly_connected_components > 0:
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            largest_wcc_size = len(largest_wcc)
            largest_wcc_ratio = safe_divide(largest_wcc_size, G.number_of_nodes())
        else:
            largest_wcc_size = 0
            largest_wcc_ratio = 0
        
        return {
            'year': year,
            'is_strongly_connected': is_strongly_connected,
            'num_strongly_connected_components': num_strongly_connected_components,
            'largest_scc_size': largest_scc_size,
            'largest_scc_ratio': largest_scc_ratio,
            'is_weakly_connected': is_weakly_connected,
            'num_weakly_connected_components': num_weakly_connected_components,
            'largest_wcc_size': largest_wcc_size,
            'largest_wcc_ratio': largest_wcc_ratio
        }
        
    except Exception as e:
        return handle_computation_error("calculate_connectivity_metrics", year, e,
                                      {'year': year, 'is_strongly_connected': False, 'is_weakly_connected': False})

@timer_decorator
def calculate_path_metrics(G: nx.DiGraph, year: int, sample_size: int = 1000) -> Dict[str, Any]:
    """
    计算路径长度相关指标（修正版：使用distance权重）
    
    Args:
        G: NetworkX有向图
        year: 年份
        sample_size: 采样大小（对于大网络进行采样以加速计算）
    
    Returns:
        路径指标字典
    """
    validate_graph(G, "calculate_path_metrics")
    logger.info(f"     {year}: 计算路径长度指标（修正版）...")
    
    try:
        # 修正：添加距离权重（distance = 1/weight）
        G_with_distance = add_distance_weights(G)
        
        # 获取采样节点
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        if n_nodes <= sample_size:
            sample_nodes = nodes
        else:
            # 使用缓存的采样函数
            sample_nodes = list(get_node_sample(tuple(nodes), sample_size))
        
        path_lengths = []
        weighted_path_lengths = []  # 基于贸易强度的路径长度
        reachable_pairs = 0
        total_pairs = 0
        
        for source in sample_nodes:
            try:
                # 计算未加权最短路径
                unweighted_lengths = nx.single_source_shortest_path_length(G, source)
                
                # 计算加权最短路径（使用distance权重，修正后）
                weighted_lengths = nx.single_source_shortest_path_length(G_with_distance, source, weight='distance')
                
                for target in sample_nodes:
                    if source != target:
                        total_pairs += 1
                        if target in unweighted_lengths:
                            path_lengths.append(unweighted_lengths[target])
                            reachable_pairs += 1
                            
                        if target in weighted_lengths:
                            # 检查路径长度是否有效（不是无穷大）
                            if weighted_lengths[target] != float('inf'):
                                weighted_path_lengths.append(weighted_lengths[target])
                                
            except Exception as e:
                logger.debug(f"     {year}: 节点{source}路径计算部分失败: {e}")
                continue
        
        # 计算统计量
        if path_lengths:
            avg_path_length = np.mean(path_lengths)
            median_path_length = np.median(path_lengths)
            max_path_length = np.max(path_lengths)
            min_path_length = np.min(path_lengths)
        else:
            avg_path_length = median_path_length = max_path_length = min_path_length = 0
        
        # 计算加权路径统计量
        if weighted_path_lengths:
            avg_weighted_path_length = np.mean(weighted_path_lengths)
            median_weighted_path_length = np.median(weighted_path_lengths)
        else:
            avg_weighted_path_length = median_weighted_path_length = 0
        
        reachability_ratio = safe_divide(reachable_pairs, total_pairs)
        
        return {
            'year': year,
            'avg_path_length': avg_path_length,
            'median_path_length': median_path_length,
            'max_path_length': max_path_length,
            'min_path_length': min_path_length,
            'avg_weighted_path_length': avg_weighted_path_length,
            'median_weighted_path_length': median_weighted_path_length,
            'reachability_ratio': reachability_ratio,
            'sampled_pairs': total_pairs,
            'reachable_pairs': reachable_pairs,
            'weighted_reachable_pairs': len(weighted_path_lengths)
        }
        
    except Exception as e:
        return handle_computation_error("calculate_path_metrics", year, e,
                                      {'year': year, 'avg_path_length': 0, 'reachability_ratio': 0})

@timer_decorator
def calculate_efficiency_metrics(G: nx.DiGraph, year: int, sample_size: int = 500) -> Dict[str, Any]:
    """
    计算网络效率指标（修正版：使用distance权重）
    
    Args:
        G: NetworkX有向图
        year: 年份
        sample_size: 采样大小
    
    Returns:
        效率指标字典
    """
    validate_graph(G, "calculate_efficiency_metrics")
    logger.info(f"     {year}: 计算网络效率（修正版）...")
    
    try:
        # 修正：添加距离权重
        G_with_distance = add_distance_weights(G)
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        if n_nodes <= sample_size:
            sample_nodes = nodes
        else:
            sample_nodes = list(get_node_sample(tuple(nodes), sample_size))
        
        efficiency_sum = 0
        weighted_efficiency_sum = 0
        total_pairs = 0
        
        for source in sample_nodes:
            try:
                # 未加权路径长度
                unweighted_lengths = nx.single_source_shortest_path_length(G, source)
                
                # 加权路径长度（使用distance权重）
                weighted_lengths = nx.single_source_shortest_path_length(G_with_distance, source, weight='distance')
                
                for target in sample_nodes:
                    if source != target:
                        total_pairs += 1
                        
                        # 未加权效率
                        if target in unweighted_lengths and unweighted_lengths[target] > 0:
                            efficiency_sum += 1.0 / unweighted_lengths[target]
                        
                        # 加权效率
                        if target in weighted_lengths and weighted_lengths[target] > 0 and weighted_lengths[target] != float('inf'):
                            weighted_efficiency_sum += 1.0 / weighted_lengths[target]
                            
            except Exception as e:
                logger.debug(f"     {year}: 节点{source}效率计算部分失败: {e}")
                continue
        
        global_efficiency = safe_divide(efficiency_sum, total_pairs)
        weighted_global_efficiency = safe_divide(weighted_efficiency_sum, total_pairs)
        
        return {
            'year': year,
            'global_efficiency': global_efficiency,
            'weighted_global_efficiency': weighted_global_efficiency,
            'efficiency_sample_size': len(sample_nodes),
            'efficiency_sample_pairs': total_pairs
        }
        
    except Exception as e:
        return handle_computation_error("calculate_efficiency_metrics", year, e,
                                      {'year': year, 'global_efficiency': 0, 'weighted_global_efficiency': 0})

@timer_decorator
def calculate_clustering_metrics(G: nx.DiGraph, year: int) -> Dict[str, Any]:
    """
    计算聚类系数相关指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        聚类指标字典
    """
    validate_graph(G, "calculate_clustering_metrics")
    logger.info(f"     {year}: 计算聚类系数...")
    
    try:
        # 转换为无向图计算聚类系数
        G_undirected = G.to_undirected()
        
        # 全局聚类系数（传递性）
        global_clustering = nx.transitivity(G_undirected)
        
        # 平均聚类系数
        avg_clustering = nx.average_clustering(G_undirected)
        
        # 加权平均聚类系数
        try:
            weighted_avg_clustering = nx.average_clustering(G_undirected, weight='weight')
        except:
            weighted_avg_clustering = avg_clustering
        
        return {
            'year': year,
            'global_clustering': global_clustering,
            'avg_clustering': avg_clustering,
            'weighted_avg_clustering': weighted_avg_clustering
        }
        
    except Exception as e:
        return handle_computation_error("calculate_clustering_metrics", year, e,
                                      {'year': year, 'global_clustering': 0, 'avg_clustering': 0})

def calculate_all_global_metrics(G: nx.DiGraph, year: int) -> Dict[str, Any]:
    """
    计算所有全局网络指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        包含所有全局网络指标的字典
    """
    logger.info(f"🌐 {year}: 开始计算全局网络指标...")
    
    # 计算各类指标
    density_metrics = calculate_density_metrics(G, year)
    connectivity_metrics = calculate_connectivity_metrics(G, year)
    path_metrics = calculate_path_metrics(G, year)
    efficiency_metrics = calculate_efficiency_metrics(G, year)
    clustering_metrics = calculate_clustering_metrics(G, year)
    
    # 合并所有指标
    all_metrics = {}
    all_metrics.update(density_metrics)
    all_metrics.update(connectivity_metrics)
    all_metrics.update(path_metrics)
    all_metrics.update(efficiency_metrics)
    all_metrics.update(clustering_metrics)
    
    logger.info(f"✅ {year}: 全局网络指标计算完成")
    
    return all_metrics

def get_global_metrics_summary(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成全局指标的摘要信息
    
    Args:
        metrics_dict: 全局指标字典
        
    Returns:
        摘要信息字典
    """
    year = metrics_dict.get('year', 'Unknown')
    
    summary = {
        'year': year,
        'network_scale': f"{metrics_dict.get('nodes', 0)} nodes, {metrics_dict.get('edges', 0)} edges",
        'connectivity_status': 'Strongly Connected' if metrics_dict.get('is_strongly_connected', False) else 
                              'Weakly Connected' if metrics_dict.get('is_weakly_connected', False) else 'Disconnected',
        'density_level': 'High' if metrics_dict.get('density', 0) > 0.1 else 
                        'Medium' if metrics_dict.get('density', 0) > 0.01 else 'Low',
        'avg_path_length': round(metrics_dict.get('avg_path_length', 0), 2),
        'global_efficiency': round(metrics_dict.get('global_efficiency', 0), 4),
        'clustering_coefficient': round(metrics_dict.get('global_clustering', 0), 4)
    }
    
    return summary