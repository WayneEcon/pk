#!/usr/bin/env python3
"""
因果验证模块 (Causal Validation Module)
====================================

本模块实现从描述性分析向因果推断的关键跃进，通过严谨的计量经济学方法，
检验DLI（动态锁定指数）与网络韧性之间的因果关系。

核心功能：
1. 网络韧性量化 - 拓扑抗毁性和供应缺口吸收率双轨测量
2. 因果推断 - 双向固定效应面板模型和工具变量法
3. 可视化分析 - 韧性指标图表和回归诊断图
4. 增强报告 - 学术级因果验证报告生成

作者：Energy Network Analysis Team
版本：v2.0 (Enhanced Econometric Edition)
"""

__version__ = "2.0.0"
__author__ = "Energy Network Analysis Team"

# 核心模块导入
from .resilience_calculator import (
    NetworkResilienceCalculator,
    calculate_topological_resilience,
    calculate_supply_absorption,
    generate_resilience_database
)

from .causal_model import (
    CausalAnalyzer,
    TwoWayFixedEffectsModel,
    InstrumentalVariablesModel,
    run_causal_validation
)

from .visualization import (
    CausalVisualization,
    create_visualizations
)

from .main import (
    CausalValidationPipeline
)

__all__ = [
    # 韧性计算
    'NetworkResilienceCalculator',
    'calculate_topological_resilience', 
    'calculate_supply_absorption',
    'generate_resilience_database',
    
    # 因果分析
    'CausalAnalyzer',
    'TwoWayFixedEffectsModel',
    'InstrumentalVariablesModel', 
    'run_causal_validation',
    
    # 可视化分析
    'CausalVisualization',
    'create_visualizations',
    
    # 完整管道
    'CausalValidationPipeline'
]