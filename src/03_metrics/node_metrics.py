#!/usr/bin/env python3
"""
节点级别指标计算模块

负责计算网络中各个节点的中心性指标：
- 度中心性 (Degree Centrality)
- 强度中心性 (Strength Centrality) 
- 中介中心性 (Betweenness Centrality)
- PageRank 中心性
- 特征向量中心性 (Eigenvector Centrality)
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from utils import (
    setup_logger, validate_graph, safe_divide, timer_decorator,
    handle_computation_error, validate_metrics_result, add_distance_weights
)

logger = setup_logger(__name__)

@timer_decorator
def calculate_degree_centrality(G: nx.DiGraph, year: int) -> pd.DataFrame:
    """
    计算度中心性指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        包含度中心性指标的DataFrame
        
    Raises:
        ValueError: 当输入图无效时
    """
    validate_graph(G, "calculate_degree_centrality")
    logger.info(f"     {year}: 计算度中心性...")
    
    try:
        results = []
        n_nodes = G.number_of_nodes()
        
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            total_degree = in_degree + out_degree
            
            # 归一化度中心性 (除以可能的最大连接数)
            norm_in_degree = safe_divide(in_degree, n_nodes - 1)
            norm_out_degree = safe_divide(out_degree, n_nodes - 1)
            norm_total_degree = safe_divide(total_degree, 2 * (n_nodes - 1))
            
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': total_degree,
                'norm_in_degree': norm_in_degree,
                'norm_out_degree': norm_out_degree,
                'norm_total_degree': norm_total_degree
            })
        
        df = pd.DataFrame(results)
        
        # 验证结果
        expected_cols = ['year', 'country_code', 'in_degree', 'out_degree', 'total_degree']
        validate_metrics_result(df, expected_cols, year, "度中心性")
        
        return df
        
    except Exception as e:
        return handle_computation_error("calculate_degree_centrality", year, e, 
                                      pd.DataFrame(columns=['year', 'country_code', 'in_degree', 'out_degree']))

@timer_decorator  
def calculate_strength_centrality(G: nx.DiGraph, year: int) -> pd.DataFrame:
    """
    计算强度中心性指标 (加权度)
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        包含强度中心性指标的DataFrame
    """
    validate_graph(G, "calculate_strength_centrality")
    logger.info(f"     {year}: 计算强度中心性...")
    
    try:
        results = []
        
        # 计算总贸易额用于归一化
        total_trade = sum(data.get('weight', 0) for _, _, data in G.edges(data=True))
        
        for node in G.nodes():
            in_strength = G.in_degree(node, weight='weight')
            out_strength = G.out_degree(node, weight='weight')
            total_strength = in_strength + out_strength
            
            # 归一化强度中心性
            norm_in_strength = safe_divide(in_strength, total_trade)
            norm_out_strength = safe_divide(out_strength, total_trade)
            norm_total_strength = safe_divide(total_strength, total_trade)
            
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'in_strength': in_strength,
                'out_strength': out_strength,
                'total_strength': total_strength,
                'norm_in_strength': norm_in_strength,
                'norm_out_strength': norm_out_strength,
                'norm_total_strength': norm_total_strength
            })
        
        df = pd.DataFrame(results)
        
        # 验证结果
        expected_cols = ['year', 'country_code', 'in_strength', 'out_strength', 'total_strength']
        validate_metrics_result(df, expected_cols, year, "强度中心性")
        
        return df
        
    except Exception as e:
        return handle_computation_error("calculate_strength_centrality", year, e,
                                      pd.DataFrame(columns=['year', 'country_code', 'in_strength', 'out_strength']))

@timer_decorator
def calculate_betweenness_centrality(G: nx.DiGraph, year: int) -> pd.DataFrame:
    """
    计算中介中心性指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        包含中介中心性指标的DataFrame
    """
    validate_graph(G, "calculate_betweenness_centrality")
    logger.info(f"     {year}: 计算中介中心性...")
    
    try:
        # 修正：为正确计算加权中介中心性，需使用距离作为权重 (distance = 1/weight)
        G_with_distance = add_distance_weights(G)
        
        # 使用修正后的距离权重进行计算
        betweenness = nx.betweenness_centrality(G_with_distance, weight='distance', normalized=True)
        
        results = []
        for node in G.nodes():
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'betweenness_centrality': betweenness.get(node, 0)
            })
        
        df = pd.DataFrame(results)
        
        # 验证结果
        expected_cols = ['year', 'country_code', 'betweenness_centrality']
        validate_metrics_result(df, expected_cols, year, "中介中心性")
        
        return df
        
    except Exception as e:
        # 对于计算失败的情况，返回零值结果
        results = []
        for node in G.nodes():
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'betweenness_centrality': 0.0
            })
        return pd.DataFrame(results)

@timer_decorator
def calculate_pagerank_centrality(G: nx.DiGraph, year: int, 
                                alpha: float = 0.85, max_iter: int = 100) -> pd.DataFrame:
    """
    计算PageRank中心性指标
    
    Args:
        G: NetworkX有向图
        year: 年份
        alpha: 阻尼参数
        max_iter: 最大迭代次数
    
    Returns:
        包含PageRank中心性指标的DataFrame
    """
    validate_graph(G, "calculate_pagerank_centrality")
    logger.info(f"     {year}: 计算PageRank中心性...")
    
    try:
        # 使用权重计算PageRank
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, weight='weight')
        
        results = []
        for node in G.nodes():
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'pagerank_centrality': pagerank.get(node, 0)
            })
        
        df = pd.DataFrame(results)
        
        # 验证结果
        expected_cols = ['year', 'country_code', 'pagerank_centrality']
        validate_metrics_result(df, expected_cols, year, "PageRank中心性")
        
        return df
        
    except Exception as e:
        # 返回均匀分布的结果
        n_nodes = G.number_of_nodes()
        uniform_value = safe_divide(1.0, n_nodes)
        
        results = []
        for node in G.nodes():
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'pagerank_centrality': uniform_value
            })
        return pd.DataFrame(results)

@timer_decorator
def calculate_eigenvector_centrality(G: nx.DiGraph, year: int, 
                                   max_iter: int = 100, tolerance: float = 1e-6) -> pd.DataFrame:
    """
    计算特征向量中心性指标
    
    Args:
        G: NetworkX有向图
        year: 年份
        max_iter: 最大迭代次数
        tolerance: 收敛容差
    
    Returns:
        包含特征向量中心性指标的DataFrame
    """
    validate_graph(G, "calculate_eigenvector_centrality")
    logger.info(f"     {year}: 计算特征向量中心性...")
    
    try:
        # 使用权重计算特征向量中心性
        eigenvector = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tolerance, weight='weight')
        
        results = []
        for node in G.nodes():
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'eigenvector_centrality': eigenvector.get(node, 0)
            })
        
        df = pd.DataFrame(results)
        
        # 验证结果
        expected_cols = ['year', 'country_code', 'eigenvector_centrality']
        validate_metrics_result(df, expected_cols, year, "特征向量中心性")
        
        return df
        
    except Exception as e:
        # 对于不收敛或其他错误，返回零值结果
        logger.warning(f"     {year}: 特征向量中心性计算失败，返回零值: {e}")
        results = []
        for node in G.nodes():
            results.append({
                'year': year,
                'country_code': node,
                'country_name': G.nodes[node].get('name', node),
                'eigenvector_centrality': 0.0
            })
        return pd.DataFrame(results)

def calculate_all_node_centralities(G: nx.DiGraph, year: int) -> pd.DataFrame:
    """
    计算所有节点中心性指标
    
    Args:
        G: NetworkX有向图
        year: 年份
    
    Returns:
        包含所有节点中心性指标的完整DataFrame
    """
    logger.info(f"📊 {year}: 开始计算节点中心性指标...")
    
    # 计算各种中心性指标
    degree_df = calculate_degree_centrality(G, year)
    strength_df = calculate_strength_centrality(G, year)
    betweenness_df = calculate_betweenness_centrality(G, year)
    pagerank_df = calculate_pagerank_centrality(G, year)
    eigenvector_df = calculate_eigenvector_centrality(G, year)
    
    # 合并所有指标
    result_df = degree_df.copy()
    
    # 添加强度指标
    strength_cols = ['in_strength', 'out_strength', 'total_strength', 
                     'norm_in_strength', 'norm_out_strength', 'norm_total_strength']
    for col in strength_cols:
        if col in strength_df.columns:
            result_df[col] = strength_df[col]
    
    # 添加中介中心性
    if 'betweenness_centrality' in betweenness_df.columns:
        result_df['betweenness_centrality'] = betweenness_df['betweenness_centrality']
    
    # 添加PageRank
    if 'pagerank_centrality' in pagerank_df.columns:
        result_df['pagerank_centrality'] = pagerank_df['pagerank_centrality']
        
    # 添加特征向量中心性
    if 'eigenvector_centrality' in eigenvector_df.columns:
        result_df['eigenvector_centrality'] = eigenvector_df['eigenvector_centrality']
    
    logger.info(f"✅ {year}: 节点中心性指标计算完成 - {len(result_df)} 个节点")
    
    return result_df

def get_node_centrality_rankings(df: pd.DataFrame, year: int, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    获取各种中心性指标的排名
    
    Args:
        df: 包含中心性指标的DataFrame
        year: 年份
        top_k: 返回前k个节点
    
    Returns:
        包含各种排名的字典
    """
    rankings = {}
    
    centrality_metrics = [
        'total_degree', 'total_strength', 'betweenness_centrality', 
        'pagerank_centrality', 'eigenvector_centrality'
    ]
    
    for metric in centrality_metrics:
        if metric in df.columns:
            top_nodes = df.nlargest(top_k, metric)[['country_code', 'country_name', metric]].to_dict('records')
            rankings[f'top_{metric}'] = top_nodes
    
    return rankings

def get_node_centrality_summary(df: pd.DataFrame, year: int) -> Dict[str, Any]:
    """
    生成节点中心性指标的统计摘要
    
    Args:
        df: 包含中心性指标的DataFrame  
        year: 年份
    
    Returns:
        统计摘要字典
    """
    summary = {
        'year': year,
        'total_nodes': len(df)
    }
    
    # 计算各指标的基本统计量
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != 'year':  # 排除年份列
            summary.update({
                f'{col}_mean': df[col].mean(),
                f'{col}_std': df[col].std(),
                f'{col}_max': df[col].max(),
                f'{col}_min': df[col].min(),
                f'{col}_median': df[col].median()
            })
            
            # 找到最大值对应的国家
            max_idx = df[col].idxmax()
            if not pd.isna(max_idx):
                summary[f'{col}_max_country'] = df.loc[max_idx, 'country_code']
    
    return summary