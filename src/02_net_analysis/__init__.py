"""
网络分析模块 (02_net_analysis)
====================================

这是一个优化重构的能源贸易网络构建模块，提供完整的数据处理、网络构建、
统计计算和结果输出功能。

主要功能：
- 数据加载和验证 (data_loader)
- 数据一致性处理和聚合 (data_processor) 
- 网络图构建 (network_builder)
- 网络统计计算 (network_stats)
- 结果输出管理 (output_manager)
- 工具函数和验证 (utils)

使用示例：
    from 02_net_analysis import (
        load_yearly_data, 
        resolve_trade_data_consistency,
        aggregate_trade_flows,
        build_network_from_data,
        calculate_basic_network_stats,
        save_networks_comprehensive
    )
    
    # 完整的网络构建流程
    raw_data = load_yearly_data(2020)
    consistent_data = resolve_trade_data_consistency(raw_data, 2020)
    aggregated_data = aggregate_trade_flows(consistent_data, 2020) 
    G = build_network_from_data(aggregated_data, 2020)
    stats = calculate_basic_network_stats(G, 2020)
"""

from .data_loader import load_yearly_data
from .data_processor import resolve_trade_data_consistency, aggregate_trade_flows
from .network_builder import build_network_from_data
from .network_stats import calculate_basic_network_stats
from .output_manager import save_networks_comprehensive, generate_annual_nodes_edges, generate_summary_report
from .utils import (
    validate_dataframe_columns, 
    safe_divide, 
    log_dataframe_info,
    validate_network_graph,
    validate_trade_data_schema,
    validate_statistics_data,
    DataQualityReporter
)

# 版本信息
__version__ = '2.0.0'
__author__ = 'Energy Network Analysis Team'

# 导出的主要函数
__all__ = [
    # 数据处理
    'load_yearly_data',
    'resolve_trade_data_consistency', 
    'aggregate_trade_flows',
    
    # 网络构建
    'build_network_from_data',
    'calculate_basic_network_stats',
    
    # 输出管理
    'save_networks_comprehensive',
    'generate_annual_nodes_edges',
    'generate_summary_report',
    
    # 工具函数
    'validate_dataframe_columns',
    'safe_divide',
    'log_dataframe_info',
    'validate_network_graph', 
    'validate_trade_data_schema',
    'validate_statistics_data',
    'DataQualityReporter'
]