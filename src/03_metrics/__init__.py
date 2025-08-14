#!/usr/bin/env python3
"""
网络指标计算模块 (03_metrics)
====================================

这是一个全面重构的网络指标计算模块，提供节点级别和全局级别的网络分析功能。

主要功能：
- 节点级别指标 (node_metrics): 度中心性、强度中心性、中介中心性、PageRank、特征向量中心性
- 全局级别指标 (global_metrics): 网络密度、连通性、路径长度、聚类系数、网络效率
- 统一工具函数 (utils): 数据验证、缓存、错误处理

核心修正：
- 修正了加权最短路径计算逻辑（使用distance = 1/weight）
- 消除了代码重复，提供统一的工具函数
- 完善的错误处理和数据验证机制
- 性能优化：缓存机制、采样策略、计时装饰器

使用示例：
    from 03_metrics import calculate_all_metrics_for_year
    
    # 计算单个年份的所有指标
    all_metrics_df = calculate_all_metrics_for_year(G, 2020)
    
    # 或分别计算
    from 03_metrics import calculate_all_node_centralities, calculate_all_global_metrics
    
    node_metrics_df = calculate_all_node_centralities(G, 2020)
    global_metrics_dict = calculate_all_global_metrics(G, 2020)
"""

import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple

# 导入核心功能
from node_metrics import (
    calculate_all_node_centralities,
    get_node_centrality_rankings,
    get_node_centrality_summary
)

from global_metrics import (
    calculate_all_global_metrics,
    get_global_metrics_summary
)

from utils import (
    setup_logger, validate_graph, safe_divide, timer_decorator,
    merge_metric_dataframes, create_metrics_summary, validate_metrics_result
)

# 版本信息
__version__ = '3.0.0'
__author__ = 'Energy Network Analysis Team'

# 设置日志
logger = setup_logger(__name__)

def calculate_all_metrics_for_year(G: nx.DiGraph, year: int) -> pd.DataFrame:
    """
    计算单个年份网络的所有指标（统一接口）
    
    Args:
        G: NetworkX有向图
        year: 年份
        
    Returns:
        包含所有指标的DataFrame，每行代表一个节点及其指标
        
    Raises:
        ValueError: 当输入图无效时
        
    Example:
        >>> G = nx.DiGraph()
        >>> G.add_edge('A', 'B', weight=100)
        >>> metrics_df = calculate_all_metrics_for_year(G, 2020)
        >>> print(metrics_df.columns)
    """
    logger.info(f"🚀 {year}: 开始计算所有网络指标...")
    
    # 验证输入
    validate_graph(G, "calculate_all_metrics_for_year")
    
    try:
        # 1. 计算节点级别指标
        node_metrics_df = calculate_all_node_centralities(G, year)
        
        # 2. 计算全局级别指标
        global_metrics_dict = calculate_all_global_metrics(G, year)
        
        # 3. 将全局指标添加到每个节点行中（广播）
        for key, value in global_metrics_dict.items():
            if key != 'year':  # 避免重复年份列
                node_metrics_df[f'global_{key}'] = value
        
        logger.info(f"🎯 {year}: 所有网络指标计算完成 - {len(node_metrics_df)} 个节点")
        
        return node_metrics_df
        
    except Exception as e:
        logger.error(f"❌ {year}: 指标计算失败: {e}")
        raise

def calculate_metrics_for_multiple_years(annual_networks: Dict[int, nx.DiGraph]) -> pd.DataFrame:
    """
    并行计算多个年份的网络指标
    
    Args:
        annual_networks: 年度网络字典 {year: Graph}
        
    Returns:
        包含所有年份所有指标的完整DataFrame
        
    Example:
        >>> networks = {2020: G2020, 2021: G2021}
        >>> all_metrics_df = calculate_metrics_for_multiple_years(networks)
    """
    logger.info(f"🌟 开始计算多年份网络指标 - {len(annual_networks)} 个年份")
    
    if not annual_networks:
        logger.warning("没有网络数据，返回空DataFrame")
        return pd.DataFrame()
    
    all_metrics_list = []
    
    for year in sorted(annual_networks.keys()):
        G = annual_networks[year]
        try:
            year_metrics = calculate_all_metrics_for_year(G, year)
            all_metrics_list.append(year_metrics)
        except Exception as e:
            logger.error(f"❌ {year}年指标计算失败: {e}")
            continue
    
    if all_metrics_list:
        # 合并所有年份的数据
        combined_df = pd.concat(all_metrics_list, ignore_index=True)
        logger.info(f"✅ 多年份指标计算完成 - 总计 {len(combined_df)} 条记录")
        return combined_df
    else:
        logger.error("所有年份计算都失败了")
        return pd.DataFrame()

def get_metrics_summary_report(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成指标计算的详细摘要报告
    
    Args:
        metrics_df: 包含指标的DataFrame
        
    Returns:
        摘要报告字典
        
    Example:
        >>> summary = get_metrics_summary_report(metrics_df)
        >>> print(summary['total_records'])
    """
    if metrics_df.empty:
        return {'error': '没有数据可用于生成摘要'}
    
    years = sorted(metrics_df['year'].unique())
    
    summary = {
        'report_generated': pd.Timestamp.now().isoformat(),
        'total_records': len(metrics_df),
        'years_covered': len(years),
        'year_range': f"{min(years)} - {max(years)}" if years else "无数据",
        'total_countries': metrics_df['country_code'].nunique(),
    }
    
    # 按年份统计
    yearly_stats = []
    for year in years:
        year_data = metrics_df[metrics_df['year'] == year]
        
        # 节点中心性排名
        top_by_strength = year_data.nlargest(5, 'total_strength')[['country_code', 'total_strength']].to_dict('records')
        top_by_pagerank = year_data.nlargest(5, 'pagerank_centrality')[['country_code', 'pagerank_centrality']].to_dict('records')
        
        yearly_stats.append({
            'year': year,
            'nodes': len(year_data),
            'avg_total_strength': year_data['total_strength'].mean(),
            'network_density': year_data['global_density'].iloc[0] if 'global_density' in year_data.columns else 0,
            'avg_path_length': year_data['global_avg_path_length'].iloc[0] if 'global_avg_path_length' in year_data.columns else 0,
            'top_countries_by_strength': top_by_strength,
            'top_countries_by_pagerank': top_by_pagerank
        })
    
    summary['yearly_statistics'] = yearly_stats
    
    # 整体趋势分析
    if len(years) > 1:
        # 网络规模趋势
        network_sizes = [len(metrics_df[metrics_df['year'] == y]) for y in years]
        summary['network_growth'] = {
            'initial_size': network_sizes[0],
            'final_size': network_sizes[-1],
            'growth_rate': (network_sizes[-1] - network_sizes[0]) / network_sizes[0] if network_sizes[0] > 0 else 0
        }
        
        # 密度趋势（如果有全局指标）
        if 'global_density' in metrics_df.columns:
            densities = [metrics_df[metrics_df['year'] == y]['global_density'].iloc[0] for y in years]
            summary['density_trend'] = {
                'initial_density': densities[0],
                'final_density': densities[-1],
                'density_change': densities[-1] - densities[0]
            }
    
    return summary

def export_metrics_to_files(metrics_df: pd.DataFrame, output_dir: str = "./outputs/metrics") -> Dict[str, str]:
    """
    将指标结果导出到文件
    
    Args:
        metrics_df: 包含指标的DataFrame
        output_dir: 输出目录
        
    Returns:
        导出文件路径字典
        
    Example:
        >>> file_paths = export_metrics_to_files(metrics_df, "./results")
    """
    from pathlib import Path
    import os
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    try:
        # 1. 完整数据CSV
        full_csv_path = output_path / "all_metrics.csv"
        metrics_df.to_csv(full_csv_path, index=False)
        exported_files['full_data'] = str(full_csv_path)
        
        # 2. 节点中心性汇总
        node_centrality_cols = [col for col in metrics_df.columns 
                               if any(c in col for c in ['degree', 'strength', 'centrality'])]
        centrality_cols = ['year', 'country_code', 'country_name'] + node_centrality_cols
        centrality_df = metrics_df[centrality_cols]
        
        centrality_csv_path = output_path / "node_centrality_metrics.csv"
        centrality_df.to_csv(centrality_csv_path, index=False)
        exported_files['node_centrality'] = str(centrality_csv_path)
        
        # 3. 全局指标汇总
        global_cols = [col for col in metrics_df.columns if col.startswith('global_')]
        if global_cols:
            global_cols = ['year'] + global_cols
            global_df = metrics_df[global_cols].drop_duplicates()
            
            global_csv_path = output_path / "global_network_metrics.csv"
            global_df.to_csv(global_csv_path, index=False)
            exported_files['global_metrics'] = str(global_csv_path)
        
        # 4. 生成摘要报告
        summary_report = get_metrics_summary_report(metrics_df)
        summary_json_path = output_path / "metrics_summary_report.json"
        
        import json
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2, default=str)
        exported_files['summary_report'] = str(summary_json_path)
        
        logger.info(f"📊 指标数据已导出到 {output_dir}")
        
        return exported_files
        
    except Exception as e:
        logger.error(f"❌ 导出文件时出错: {e}")
        return {}

# 导出的主要函数
__all__ = [
    # 主要计算函数
    'calculate_all_metrics_for_year',
    'calculate_metrics_for_multiple_years',
    
    # 节点指标
    'calculate_all_node_centralities',
    'get_node_centrality_rankings', 
    'get_node_centrality_summary',
    
    # 全局指标
    'calculate_all_global_metrics',
    'get_global_metrics_summary',
    
    # 辅助功能
    'get_metrics_summary_report',
    'export_metrics_to_files',
    
    # 工具函数
    'setup_logger',
    'validate_graph',
    'safe_divide'
]