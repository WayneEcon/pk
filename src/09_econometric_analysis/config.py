#!/usr/bin/env python3
"""
配置文件 (Configuration File)
=============================

09_econometric_analysis 模块的配置参数

作者：Energy Network Analysis Team
版本：v1.0 - 计量分析框架
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 基础路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# 模块特定路径
MODULE_DIR = Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / "outputs"
FIGURES_DIR = MODULE_DIR / "figures"

# 确保目录存在
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# 计量模型配置
class ModelConfig:
    """计量模型相关配置"""
    
    # 核心研究变量定义
    DEPENDENT_VARIABLES = {
        'vul_us': '美国锚定脆弱性指数',
        'node_dli_us': 'Node-DLI美国指数',
        'delta_vul_us': '脆弱性指数变化率',
        'delta_node_dli_us': 'Node-DLI变化率'
    }
    
    # 关键解释变量
    KEY_EXPLANATORY_VARIABLES = {
        'node_dli_us': 'Node-DLI美国指数',
        'ovi': '物理冗余指数OVI',
        'us_prod_shock': '美国产量冲击',
        'ovi_lag1': 'OVI滞后1期',
        'us_prod_shock_lag1': '美国产量冲击滞后1期'
    }
    
    # 控制变量组
    CONTROL_VARIABLES = {
        'macro_controls': [
            'log_gdp',                    # GDP对数
            'log_population',             # 人口对数  
            'trade_openness_gdp_pct'      # 贸易开放度
        ],
        'network_controls': [
            'betweenness_centrality',     # 介数中心性
            'eigenvector_centrality',     # 特征向量中心性
            'in_degree',                  # 入度
            'out_degree'                  # 出度
        ],
        'energy_controls': [
            'lng_capacity',               # LNG容量
            'pipeline_capacity',          # 管道容量
            'energy_demand'               # 能源需求
        ]
    }
    
    # 面板数据设定
    PANEL_SETTINGS = {
        'time_var': 'year',              # 时间变量
        'entity_var': 'country',         # 个体变量
        'min_time_periods': 5,           # 最少时间期数
        'min_entities': 10,              # 最少个体数
        'balanced_panel': False          # 是否要求平衡面板
    }
    
    # 固定效应设定
    FIXED_EFFECTS = {
        'time_effects': True,            # 时间固定效应
        'entity_effects': True,          # 个体固定效应
        'two_way_effects': True          # 双向固定效应
    }
    
    # 模型估计设定
    ESTIMATION_SETTINGS = {
        'robust': True,                  # 稳健标准误
        'cluster_var': 'country',        # 聚类变量
        'bootstrap_reps': 1000,          # Bootstrap重复次数
        'confidence_level': 0.95         # 置信水平
    }

class AnalysisConfig:
    """分析设定配置"""
    
    # 研究模型定义
    RESEARCH_MODELS = {
        'model_1_dli_vul_association': {
            'name': '模型1: DLI-脆弱性关联检验',
            'description': '双向固定效应面板模型检验Node-DLI与脆弱性的关联',
            'formula': 'vul_us ~ node_dli_us + macro_controls + C(year) + C(country)',
            'method': 'fixed_effects',
            'chapter': '第3章',
            'priority': 1
        },
        'model_2_ovi_dli_causality': {
            'name': '模型2: OVI对DLI的因果效应',
            'description': '双向固定效应面板模型检验OVI对Node-DLI的因果效应',
            'formula': 'node_dli_us ~ ovi_lag1 + macro_controls + C(year) + C(country)',
            'method': 'fixed_effects',
            'chapter': '补充分析',
            'priority': 2
        },
        'model_3_local_projection_validation': {
            'name': '模型3: 局部投影因果验证', 
            'description': '局部投影模型验证美国产量冲击的动态效应',
            'formula': 'delta_y_h ~ us_prod_shock * ovi_lag1 + macro_controls + C(year)',
            'method': 'local_projections',
            'chapter': '第4章',
            'priority': 3,
            'horizons': [0, 1, 2, 3, 4, 5]  # 预测期数
        }
    }
    
    # 稳健性检验设定
    ROBUSTNESS_CHECKS = {
        'alternative_specifications': [
            'exclude_outliers',           # 排除异常值
            'winsorize_variables',        # 变量缩尾处理
            'alternative_controls',       # 替代控制变量
            'subperiod_analysis'          # 分时期分析
        ],
        'sensitivity_analysis': [
            'bootstrap_inference',        # Bootstrap推断
            'jackknife_validation',       # Jackknife验证
            'alternative_clustering'      # 替代聚类方法
        ]
    }

class OutputConfig:
    """输出配置"""
    
    # 输出文件路径
    OUTPUT_PATHS = {
        'regression_results': OUTPUT_DIR / 'regression_results.csv',
        'analysis_report': OUTPUT_DIR / 'analysis_report.md', 
        'model_diagnostics': OUTPUT_DIR / 'model_diagnostics.json',
        'robustness_results': OUTPUT_DIR / 'robustness_results.csv'
    }
    
    # 图表输出路径  
    FIGURE_PATHS = {
        'coefficient_comparison': FIGURES_DIR / 'coefficient_comparison.png',
        'diagnostic_plots': FIGURES_DIR / 'diagnostic_plots.png',
        'impulse_response': FIGURES_DIR / 'impulse_response.png',
        'robustness_charts': FIGURES_DIR / 'robustness_charts.png'
    }
    
    # 报告格式设定
    REPORT_SETTINGS = {
        'include_diagnostics': True,     # 包含诊断统计
        'include_robustness': True,      # 包含稳健性检验
        'significance_levels': [0.01, 0.05, 0.10],  # 显著性水平
        'decimal_places': 4,             # 小数位数
        'table_format': 'markdown',      # 表格格式
        'export_latex': True             # 导出LaTeX格式
    }

class ValidationConfig:
    """数据验证配置"""
    
    # 数据质量要求
    DATA_QUALITY_THRESHOLDS = {
        'min_observations': 50,          # 最少观测数
        'max_missing_rate': 0.5,         # 最大缺失率
        'min_variation': 0.01,           # 最小变异系数
        'outlier_threshold': 3.0         # 异常值阈值(Z-score)
    }
    
    # 模型诊断要求
    MODEL_DIAGNOSTICS = {
        'check_multicollinearity': True,  # 检查多重共线性
        'vif_threshold': 10.0,            # VIF阈值
        'check_autocorrelation': True,    # 检查自相关
        'check_heteroskedasticity': True, # 检查异方差
        'normality_test': False           # 正态性检验(可选)
    }

class ComputationConfig:
    """计算配置"""
    
    # 并行计算设定
    PARALLEL_SETTINGS = {
        'use_multiprocessing': False,    # 暂时关闭并行
        'n_workers': 4,                  # 工作进程数
        'memory_limit': '4GB'            # 内存限制
    }
    
    # 数值计算精度
    NUMERICAL_SETTINGS = {
        'float_precision': 'float64',    # 浮点精度
        'convergence_tolerance': 1e-8,   # 收敛容差
        'max_iterations': 1000           # 最大迭代次数
    }

class LoggingConfig:
    """日志配置"""
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = MODULE_DIR / 'econometric_analysis.log'
    
    # 控制台输出
    CONSOLE_OUTPUT = True
    CONSOLE_LEVEL = 'INFO'
    
    # 文件输出  
    FILE_OUTPUT = True
    FILE_LEVEL = 'DEBUG'

# 集成配置类
class Config:
    """主配置类，整合所有配置"""
    
    models = ModelConfig()
    analysis = AnalysisConfig() 
    output = OutputConfig()
    validation = ValidationConfig()
    computation = ComputationConfig()
    logging = LoggingConfig()
    
    @classmethod
    def get_model_formula(cls, model_name: str) -> str:
        """获取模型公式"""
        return cls.analysis.RESEARCH_MODELS.get(model_name, {}).get('formula', '')
    
    @classmethod
    def get_control_variables(cls, control_type: str = 'all') -> List[str]:
        """获取控制变量列表"""
        if control_type == 'all':
            all_controls = []
            for controls in cls.models.CONTROL_VARIABLES.values():
                all_controls.extend(controls)
            return all_controls
        else:
            return cls.models.CONTROL_VARIABLES.get(control_type, [])
    
    @classmethod
    def validate_data_requirements(cls, df) -> Dict[str, bool]:
        """验证数据是否满足分析要求"""
        if df is None or len(df) == 0:
            return {'sufficient_data': False, 'reason': 'empty_dataframe'}
        
        # 检查最少观测数
        if len(df) < cls.validation.DATA_QUALITY_THRESHOLDS['min_observations']:
            return {'sufficient_data': False, 'reason': 'insufficient_observations'}
        
        # 检查面板结构
        if cls.models.PANEL_SETTINGS['time_var'] not in df.columns:
            return {'sufficient_data': False, 'reason': 'missing_time_variable'}
        
        if cls.models.PANEL_SETTINGS['entity_var'] not in df.columns:
            return {'sufficient_data': False, 'reason': 'missing_entity_variable'}
        
        # 检查关键变量
        key_vars = list(cls.models.DEPENDENT_VARIABLES.keys()) + list(cls.models.KEY_EXPLANATORY_VARIABLES.keys())
        missing_vars = [var for var in key_vars if var not in df.columns or df[var].isna().all()]
        
        if missing_vars:
            return {'sufficient_data': False, 'reason': f'missing_key_variables: {missing_vars}'}
        
        return {'sufficient_data': True, 'reason': 'data_ready'}
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'models': cls.models.__dict__,
            'analysis': cls.analysis.__dict__,
            'output': cls.output.__dict__,
            'validation': cls.validation.__dict__,
            'computation': cls.computation.__dict__,
            'logging': cls.logging.__dict__
        }

# 导出主要配置
config = Config()

# 便捷函数
def get_config() -> Config:
    """获取配置实例"""
    return config

def print_config_summary():
    """打印配置摘要"""
    print("📊 09_econometric_analysis 配置摘要")
    print("=" * 50)
    print(f"研究模型数量: {len(config.analysis.RESEARCH_MODELS)}")
    print(f"因变量数量: {len(config.models.DEPENDENT_VARIABLES)}")
    print(f"解释变量数量: {len(config.models.KEY_EXPLANATORY_VARIABLES)}")
    print(f"控制变量数量: {len(config.get_control_variables())}")
    print(f"输出目录: {config.output.OUTPUT_PATHS['regression_results'].parent}")
    print(f"图表目录: {config.output.FIGURE_PATHS['coefficient_comparison'].parent}")
    
    print("\n📈 研究模型:")
    for model_id, model_info in config.analysis.RESEARCH_MODELS.items():
        print(f"  • {model_info['name']} ({model_info['chapter']})")
    
    print(f"\n⚙️ 分析设定:")
    print(f"  最少观测数: {config.validation.DATA_QUALITY_THRESHOLDS['min_observations']}")
    print(f"  置信水平: {config.models.ESTIMATION_SETTINGS['confidence_level']}")
    print(f"  稳健标准误: {config.models.ESTIMATION_SETTINGS['robust']}")
    print(f"  聚类变量: {config.models.ESTIMATION_SETTINGS['cluster_var']}")

if __name__ == "__main__":
    print_config_summary()