#!/usr/bin/env python3
"""
04_policy_analysis 模块
美国能源独立政策影响分析

基于"事前-事后"对比分析方法，量化美国能源独立政策对全球能源贸易网络结构的影响
"""

# 从子模块导入主要功能
from .analysis import load_and_prepare_data, run_pre_post_analysis, calculate_policy_impact_statistics
from .plotting import plot_metric_timeseries, plot_period_comparison, create_policy_impact_dashboard  
from .main import run_full_policy_analysis

# 版本信息
__version__ = '1.0.0'
__author__ = 'Energy Network Analysis Team'

# 政策冲击时间窗口定义
POLICY_PERIODS = {
    'pre': (2001, 2008),      # 事前期：基准期
    'transition': (2009, 2015), # 转型期：页岩油革命加速期  
    'post': (2016, 2024)      # 事后期：美国成为能源出口国
}

# 重点关注的国家
KEY_COUNTRIES = ['USA', 'CHN', 'RUS', 'SAU', 'CAN', 'MEX', 'ARE', 'IND', 'JPN', 'KOR']

# 重点关注的指标
KEY_METRICS = [
    'in_strength', 'out_strength', 'total_strength',
    'betweenness_centrality', 'pagerank_centrality', 'eigenvector_centrality',
    'in_degree', 'out_degree', 'total_degree'
]

# 全局指标
GLOBAL_METRICS = [
    'global_density', 'global_avg_clustering', 'global_avg_path_length',
    'global_global_efficiency', 'global_total_weight'
]

# 导出的主要函数
__all__ = [
    # 核心分析函数
    'load_and_prepare_data',
    'run_pre_post_analysis', 
    'calculate_policy_impact_statistics',
    
    # 可视化函数
    'plot_metric_timeseries',
    'plot_period_comparison',
    'create_policy_impact_dashboard',
    
    # 主执行函数
    'run_full_policy_analysis',
    
    # 配置常量
    'POLICY_PERIODS',
    'KEY_COUNTRIES', 
    'KEY_METRICS',
    'GLOBAL_METRICS'
]