#!/usr/bin/env python3
"""
04_dli_analysis 模块
动态锁定指数(DLI)构建与政策冲击效应验证

本模块是项目的核心创新阶段，旨在从"关系粘性"维度深度揭示美国能源独立政策的国际影响。
通过构建创新的复合指标——"动态锁定指数(DLI)"，量化国家间能源贸易关系的路径依赖和转换成本。
"""

from .data_preparation import load_us_trade_data, add_distance_data
from .dli_calculator import (
    calculate_continuity, 
    calculate_infrastructure, 
    calculate_stability, 
    calculate_market_locking_power,
    calculate_dli_composite,
    generate_dli_panel_data
)
from .statistical_verification import run_did_analysis, generate_verification_report
from .main import run_full_dli_analysis

# 版本信息
__version__ = '1.0.0'
__author__ = 'Energy Network Analysis Team'

# DLI分析的核心常量
DLI_DIMENSIONS = ['continuity', 'infrastructure', 'stability', 'market_locking_power']
POLICY_SHOCK_YEAR = 2011  # 页岩革命显著产出效应年份
TREATMENT_COUNTRIES = ['CAN', 'MEX']  # 处理组：管道贸易国家
CONTROL_COUNTRIES = ['SAU', 'QAT', 'VEN', 'NOR', 'GBR']  # 控制组：海运贸易国家

# 导出的主要函数
__all__ = [
    # 数据准备
    'load_us_trade_data',
    'add_distance_data',
    
    # DLI计算
    'calculate_continuity',
    'calculate_infrastructure', 
    'calculate_stability',
    'calculate_market_locking_power',
    'calculate_dli_composite',
    'generate_dli_panel_data',
    
    # 统计验证
    'run_did_analysis',
    'generate_verification_report',
    
    # 主执行
    'run_full_dli_analysis',
    
    # 常量
    'DLI_DIMENSIONS',
    'POLICY_SHOCK_YEAR',
    'TREATMENT_COUNTRIES',
    'CONTROL_COUNTRIES'
]