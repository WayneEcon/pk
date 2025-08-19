#!/usr/bin/env python3
"""
09_econometric_analysis - 计量经济学分析模块
===========================================

完整的计量经济学分析框架，专门处理空数据和缺失数据的情况。

主要功能:
- 健壮的数据加载与验证
- 三个核心计量模型框架
- 自动化报告生成
- 空数据兼容的可视化
- 端到端分析流水线

作者: Energy Network Analysis Team
版本: v1.0 - 计量分析框架
"""

__version__ = "1.0.0"
__author__ = "Energy Network Analysis Team"

from .main import EconometricAnalysisPipeline, main
from .data_loader import DataLoader, load_data, get_data_status  
from .models import EconometricModels, run_single_model
from .reporting import ReportGenerator, generate_reports
from .visualization import VisualizationEngine, generate_visualizations
from .config import config, get_config

__all__ = [
    # 主要类
    'EconometricAnalysisPipeline',
    'DataLoader', 
    'EconometricModels',
    'ReportGenerator',
    'VisualizationEngine',
    
    # 便捷函数
    'main',
    'load_data',
    'get_data_status', 
    'run_single_model',
    'generate_reports',
    'generate_visualizations',
    
    # 配置
    'config',
    'get_config'
]