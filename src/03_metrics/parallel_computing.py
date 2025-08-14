#!/usr/bin/env python3
"""
并行计算模块
为网络指标计算提供多进程并行支持
"""

import multiprocessing as mp
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Callable
import logging
from functools import partial
import time

from utils import setup_logger

logger = setup_logger(__name__)

def _calculate_metrics_worker(args: Tuple[nx.DiGraph, int, str]) -> Tuple[int, pd.DataFrame]:
    """
    并行计算工作函数
    
    Args:
        args: (图对象, 年份, 计算类型)
        
    Returns:
        (年份, 结果DataFrame)
    """
    G, year, metric_type = args
    
    try:
        if metric_type == 'all':
            from . import calculate_all_metrics_for_year
            result = calculate_all_metrics_for_year(G, year)
        elif metric_type == 'node':
            from .node_metrics import calculate_all_node_centralities
            result = calculate_all_node_centralities(G, year)
        elif metric_type == 'global':
            from .global_metrics import calculate_all_global_metrics
            global_dict = calculate_all_global_metrics(G, year)
            result = pd.DataFrame([global_dict])
        else:
            raise ValueError(f"未知的计算类型: {metric_type}")
            
        return year, result
        
    except Exception as e:
        logger.error(f"工作进程计算 {year} 年指标失败: {e}")
        return year, pd.DataFrame()

def calculate_metrics_parallel(annual_networks: Dict[int, nx.DiGraph], 
                             metric_type: str = 'all',
                             n_processes: int = None) -> pd.DataFrame:
    """
    并行计算多年份网络指标
    
    Args:
        annual_networks: 年度网络字典
        metric_type: 计算类型 ('all', 'node', 'global')
        n_processes: 进程数，默认为CPU核心数
        
    Returns:
        包含所有年份指标的DataFrame
        
    Example:
        >>> networks = {2020: G2020, 2021: G2021, 2022: G2022}
        >>> result_df = calculate_metrics_parallel(networks, 'all', 4)
    """
    if not annual_networks:
        logger.warning("没有网络数据，返回空DataFrame")
        return pd.DataFrame()
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(annual_networks))
    
    logger.info(f"🚀 启动并行计算 - {len(annual_networks)} 个年份，{n_processes} 个进程")
    
    start_time = time.time()
    
    # 准备参数
    args_list = [(G, year, metric_type) for year, G in annual_networks.items()]
    
    try:
        # 启动多进程计算
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(_calculate_metrics_worker, args_list)
        
        # 合并结果
        all_dataframes = []
        successful_years = []
        
        for year, df in results:
            if not df.empty:
                all_dataframes.append(df)
                successful_years.append(year)
            else:
                logger.warning(f"❌ {year} 年计算失败或返回空结果")
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 并行计算完成 - {len(successful_years)}/{len(annual_networks)} 年份成功，"
                       f"用时 {elapsed:.2f} 秒，速度提升约 {len(successful_years)/elapsed:.1f}x")
            
            return combined_df
        else:
            logger.error("❌ 所有年份计算都失败了")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ 并行计算失败: {e}")
        # 回退到串行计算
        logger.info("🔄 回退到串行计算...")
        return _fallback_serial_calculation(annual_networks, metric_type)

def _fallback_serial_calculation(annual_networks: Dict[int, nx.DiGraph], 
                                metric_type: str) -> pd.DataFrame:
    """
    并行计算失败时的串行回退方案
    
    Args:
        annual_networks: 年度网络字典
        metric_type: 计算类型
        
    Returns:
        结果DataFrame
    """
    logger.info("使用串行计算模式...")
    
    all_results = []
    
    for year in sorted(annual_networks.keys()):
        G = annual_networks[year]
        try:
            year, result = _calculate_metrics_worker((G, year, metric_type))
            if not result.empty:
                all_results.append(result)
        except Exception as e:
            logger.error(f"串行计算 {year} 年失败: {e}")
            continue
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def calculate_centralities_batch(graphs_and_years: List[Tuple[nx.DiGraph, int]], 
                               centrality_functions: List[Callable],
                               n_processes: int = None) -> Dict[str, pd.DataFrame]:
    """
    批量并行计算特定中心性指标
    
    Args:
        graphs_and_years: (图, 年份) 元组列表
        centrality_functions: 中心性计算函数列表
        n_processes: 进程数
        
    Returns:
        {函数名: 结果DataFrame} 字典
        
    Example:
        >>> from .node_metrics import calculate_degree_centrality, calculate_pagerank_centrality
        >>> functions = [calculate_degree_centrality, calculate_pagerank_centrality]
        >>> results = calculate_centralities_batch(graphs_years, functions)
    """
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(graphs_and_years))
    
    logger.info(f"批量并行计算 {len(centrality_functions)} 种中心性指标，"
               f"{len(graphs_and_years)} 个网络，{n_processes} 个进程")
    
    results = {}
    
    for func in centrality_functions:
        func_name = func.__name__
        logger.info(f"  计算 {func_name}...")
        
        start_time = time.time()
        
        try:
            # 创建部分函数
            worker_func = partial(_centrality_worker, func=func)
            
            with mp.Pool(processes=n_processes) as pool:
                func_results = pool.map(worker_func, graphs_and_years)
            
            # 合并结果
            valid_results = [df for df in func_results if not df.empty]
            if valid_results:
                results[func_name] = pd.concat(valid_results, ignore_index=True)
            else:
                results[func_name] = pd.DataFrame()
            
            elapsed = time.time() - start_time
            logger.info(f"  {func_name} 完成，用时 {elapsed:.2f} 秒")
            
        except Exception as e:
            logger.error(f"  {func_name} 计算失败: {e}")
            results[func_name] = pd.DataFrame()
    
    return results

def _centrality_worker(graph_year: Tuple[nx.DiGraph, int], func: Callable) -> pd.DataFrame:
    """
    中心性计算工作函数
    
    Args:
        graph_year: (图, 年份) 元组
        func: 中心性计算函数
        
    Returns:
        结果DataFrame
    """
    G, year = graph_year
    try:
        return func(G, year)
    except Exception as e:
        logger.error(f"计算 {year} 年 {func.__name__} 失败: {e}")
        return pd.DataFrame()

def estimate_computation_time(annual_networks: Dict[int, nx.DiGraph], 
                            metric_type: str = 'all',
                            sample_size: int = 3) -> Dict[str, float]:
    """
    估算计算时间
    
    Args:
        annual_networks: 年度网络字典
        metric_type: 计算类型
        sample_size: 用于估算的样本数量
        
    Returns:
        时间估算字典
    """
    if len(annual_networks) == 0:
        return {'serial_time': 0, 'parallel_time': 0, 'speedup': 1}
    
    # 选择样本年份
    years = sorted(annual_networks.keys())
    sample_years = years[:min(sample_size, len(years))]
    
    logger.info(f"使用 {len(sample_years)} 个年份样本估算计算时间...")
    
    total_sample_time = 0
    
    for year in sample_years:
        G = annual_networks[year]
        
        start_time = time.time()
        try:
            _calculate_metrics_worker((G, year, metric_type))
        except:
            pass
        elapsed = time.time() - start_time
        
        total_sample_time += elapsed
        logger.debug(f"  {year} 年样本用时: {elapsed:.2f} 秒")
    
    # 估算总时间
    avg_time_per_year = total_sample_time / len(sample_years)
    total_years = len(annual_networks)
    
    estimated_serial_time = avg_time_per_year * total_years
    
    # 估算并行时间（考虑并行开销）
    n_processes = min(mp.cpu_count(), total_years)
    parallel_efficiency = 0.8  # 假设80%的并行效率
    estimated_parallel_time = (estimated_serial_time / n_processes) / parallel_efficiency
    
    speedup = estimated_serial_time / estimated_parallel_time if estimated_parallel_time > 0 else 1
    
    estimates = {
        'serial_time': estimated_serial_time,
        'parallel_time': estimated_parallel_time,
        'speedup': speedup,
        'recommended_parallel': speedup > 1.5 and total_years >= 4
    }
    
    logger.info(f"时间估算 - 串行: {estimated_serial_time:.1f}s, "
               f"并行: {estimated_parallel_time:.1f}s, "
               f"加速比: {speedup:.1f}x")
    
    return estimates

def get_optimal_process_count(annual_networks: Dict[int, nx.DiGraph]) -> int:
    """
    获取最优进程数
    
    Args:
        annual_networks: 年度网络字典
        
    Returns:
        推荐的进程数
    """
    n_years = len(annual_networks)
    n_cpus = mp.cpu_count()
    
    # 基于网络规模调整
    avg_nodes = sum(G.number_of_nodes() for G in annual_networks.values()) / n_years if n_years > 0 else 0
    avg_edges = sum(G.number_of_edges() for G in annual_networks.values()) / n_years if n_years > 0 else 0
    
    # 大网络需要更少的并行进程（因为每个进程占用更多内存）
    if avg_nodes > 200 or avg_edges > 2000:
        optimal_processes = min(n_cpus // 2, n_years, 4)
    elif avg_nodes > 100 or avg_edges > 1000:
        optimal_processes = min(n_cpus - 1, n_years, 6)
    else:
        optimal_processes = min(n_cpus, n_years, 8)
    
    logger.info(f"推荐进程数: {optimal_processes} "
               f"(基于 {n_years} 个网络，平均 {avg_nodes:.0f} 节点 {avg_edges:.0f} 边)")
    
    return max(1, optimal_processes)