#!/usr/bin/env python3
"""
网络构建模块
负责从数据构建NetworkX图对象
"""

import networkx as nx
import pandas as pd
import logging
from pathlib import Path
from .utils import setup_path, validate_dataframe_columns, log_dataframe_info, get_country_region_safe, DATA_CONSISTENCY_STRATEGY, ENERGY_PRODUCT_CODES

# 确保路径设置
setup_path()

logger = logging.getLogger(__name__)

def build_network_from_data(df: pd.DataFrame, year: int) -> nx.DiGraph:
    """
    从聚合的贸易数据构建加权有向图
    
    Args:
        df: 聚合后的贸易数据DataFrame
        year: 网络年份
        
    Returns:
        构建好的NetworkX有向图
        
    Raises:
        ValueError: 当输入数据不符合要求时
        
    Example:
        >>> aggregated_df = aggregate_trade_flows(consistent_df, 2020)
        >>> G = build_network_from_data(aggregated_df, 2020)
        >>> print(G.number_of_nodes())
    """
    logger.info(f"     {year}: 构建网络图...")
    
    # 数据验证
    if df.empty:
        logger.warning(f"     {year}: 输入数据为空，创建空网络")
        return nx.DiGraph()
    
    required_cols = ['source', 'target', 'trade_value_raw_usd', 'source_name', 'target_name', 'primary_data_source']
    try:
        validate_dataframe_columns(df, required_cols, "网络构建数据")
    except ValueError as e:
        logger.error(f"     {year}: {e}")
        raise
    
    # 创建空的加权有向图
    G = nx.DiGraph()
    
    # 设置图的基本属性
    G.graph.update({
        'year': year,
        'description': f'Global Energy Trade Network {year}',
        'data_consistency_strategy': DATA_CONSISTENCY_STRATEGY,
        'energy_products': list(ENERGY_PRODUCT_CODES.keys()),
        'created_at': pd.Timestamp.now().isoformat(),
        'input_records': len(df)
    })
    
    # 预处理数据以提高效率
    duplicate_edges = 0
    
    # 添加节点和边（批量操作优化）
    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        weight = row['trade_value_raw_usd']
        
        # 添加节点（带属性）
        if not G.has_node(source):
            region = get_country_region_safe(source)
            G.add_node(source, 
                      name=row['source_name'],
                      country_code=source,
                      region=region)
        
        if not G.has_node(target):
            region = get_country_region_safe(target)
            G.add_node(target,
                      name=row['target_name'], 
                      country_code=target,
                      region=region)
        
        # 添加边（累加权重，以防重复）
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
            duplicate_edges += 1
            logger.debug(f"     {year}: 发现重复边 {source}->{target}，累加权重")
        else:
            G.add_edge(source, target, 
                      weight=weight,
                      data_source=row['primary_data_source'])
    
    # 记录构建统计
    if duplicate_edges > 0:
        logger.warning(f"     {year}: 发现 {duplicate_edges} 条重复边，已累加权重")
    
    logger.info(f"     {year}: 网络构建完成 - {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    return G