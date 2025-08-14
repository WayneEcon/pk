#!/usr/bin/env python3
"""
03_metrics 模块统一工具函数
消除代码重复，提供共享功能
"""

import logging
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from functools import lru_cache
import time

def setup_logger(name: str) -> logging.Logger:
    """
    统一的日志设置函数
    
    Args:
        name: 日志记录器名称
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def validate_graph(G: nx.DiGraph, func_name: str) -> bool:
    """
    验证NetworkX图的有效性
    
    Args:
        G: NetworkX有向图
        func_name: 调用函数名称
        
    Returns:
        验证结果
        
    Raises:
        ValueError: 当图无效时
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError(f"{func_name}: 输入必须是NetworkX有向图")
    
    if G.number_of_nodes() == 0:
        raise ValueError(f"{func_name}: 图不能为空")
    
    return True

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法运算
    
    Args:
        numerator: 分子
        denominator: 分母  
        default: 分母为0时的默认返回值
        
    Returns:
        除法结果或默认值
    """
    return numerator / denominator if denominator != 0 else default

def add_distance_weights(G: nx.DiGraph) -> nx.DiGraph:
    """
    为图添加距离权重属性 (distance = 1/weight)
    修正加权最短路径计算逻辑
    
    Args:
        G: 原始图
        
    Returns:
        添加了distance属性的图副本
    """
    G_with_distance = G.copy()
    
    for u, v, data in G_with_distance.edges(data=True):
        weight = data.get('weight', 1.0)
        # 避免除零错误，权重为0时设置最大距离
        if weight > 0:
            G_with_distance[u][v]['distance'] = 1.0 / weight
        else:
            G_with_distance[u][v]['distance'] = float('inf')
    
    return G_with_distance

@lru_cache(maxsize=128)
def get_node_sample(node_tuple: tuple, sample_size: int, seed: int = 42) -> tuple:
    """
    获取节点采样（带缓存）
    
    Args:
        node_tuple: 节点元组
        sample_size: 采样大小
        seed: 随机种子
        
    Returns:
        采样节点元组
    """
    nodes = list(node_tuple)
    if len(nodes) <= sample_size:
        return node_tuple
    
    np.random.seed(seed)
    sampled = np.random.choice(nodes, size=sample_size, replace=False)
    return tuple(sorted(sampled))

def timer_decorator(func):
    """
    计时装饰器
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # 从参数中提取year用于日志
        year = None
        if 'year' in kwargs:
            year = kwargs['year']
        elif len(args) > 1:
            year = args[1]  # 通常year是第二个参数
            
        func_name = func.__name__
        if year:
            logging.getLogger(func.__module__).info(
                f"     {year}: {func_name}计算完成 ({elapsed:.2f}秒)"
            )
        else:
            logging.getLogger(func.__module__).info(
                f"     {func_name}计算完成 ({elapsed:.2f}秒)"
            )
        
        return result
    return wrapper

def merge_metric_dataframes(dataframes: List[pd.DataFrame], on_columns: List[str]) -> pd.DataFrame:
    """
    合并多个指标DataFrame
    
    Args:
        dataframes: DataFrame列表
        on_columns: 合并依据的列
        
    Returns:
        合并后的DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    result = dataframes[0]
    for df in dataframes[1:]:
        result = result.merge(df, on=on_columns, how='outer')
    
    return result

def create_metrics_summary(df: pd.DataFrame, metric_columns: List[str], year: int) -> Dict[str, Any]:
    """
    创建指标统计摘要
    
    Args:
        df: 包含指标的DataFrame
        metric_columns: 指标列名列表
        year: 年份
        
    Returns:
        统计摘要字典
    """
    summary = {'year': year, 'total_nodes': len(df)}
    
    for col in metric_columns:
        if col in df.columns:
            summary[f'{col}_mean'] = df[col].mean()
            summary[f'{col}_std'] = df[col].std()
            summary[f'{col}_max'] = df[col].max()
            summary[f'{col}_min'] = df[col].min()
            
            # 找到最大值对应的国家
            max_idx = df[col].idxmax()
            if not pd.isna(max_idx) and 'country_code' in df.columns:
                summary[f'{col}_max_country'] = df.loc[max_idx, 'country_code']
    
    return summary

def validate_metrics_result(df: pd.DataFrame, expected_columns: List[str], 
                          year: int, metric_type: str) -> bool:
    """
    验证指标计算结果
    
    Args:
        df: 结果DataFrame
        expected_columns: 期望的列
        year: 年份
        metric_type: 指标类型
        
    Returns:
        验证结果
        
    Raises:
        ValueError: 当结果无效时
    """
    logger = logging.getLogger(__name__)
    
    if df.empty:
        raise ValueError(f"{year}: {metric_type}计算结果为空")
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{year}: {metric_type}缺少必要列: {missing_cols}")
    
    # 检查是否有无效值
    for col in expected_columns:
        if col in df.columns:
            if df[col].isna().any():
                logger.warning(f"{year}: {metric_type}中{col}列存在NaN值")
            
            # 只对数值列检查无穷大值
            if pd.api.types.is_numeric_dtype(df[col]) and np.isinf(df[col]).any():
                logger.warning(f"{year}: {metric_type}中{col}列存在无穷大值")
    
    return True

def handle_computation_error(func_name: str, year: int, error: Exception, 
                           default_result: Any) -> Any:
    """
    统一的计算错误处理
    
    Args:
        func_name: 函数名
        year: 年份
        error: 异常对象
        default_result: 默认返回结果
        
    Returns:
        默认结果或重新抛出异常
    """
    logger = logging.getLogger(__name__)
    logger.error(f"     {year}: {func_name}计算失败: {error}")
    
    # 对于严重错误，重新抛出异常
    if isinstance(error, (MemoryError, KeyboardInterrupt)):
        raise error
    
    # 其他错误返回默认结果
    return default_result