#!/usr/bin/env python3
"""
配置文件 (Configuration File)
============================

08_variable_construction模块的配置参数

作者：Energy Network Analysis Team
版本：v1.0
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# 基础路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# 模块特定路径
MODULE_DIR = Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / "outputs"
TEMP_DATA_DIR = MODULE_DIR / "08data"  # 使用08data作为数据存储目录

# 确保目录存在
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DATA_DIR.mkdir(exist_ok=True)

# API配置
class APIConfig:
    """API相关配置"""
    
    # World Bank API
    WB_INDICATORS = {
        'NY.GDP.MKTP.CD': 'gdp_current_usd',
        'SP.POP.TOTL': 'population_total', 
        'NE.TRD.GNFS.ZS': 'trade_openness_gdp_pct'
    }
    
    # EIA API
    EIA_API_KEY = "kCKMXECZ7EZxHpYPXekyOhSdccpNc85aeOpDGIwm"  # 用户提供的密钥
    EIA_BASE_URL = "https://api.eia.gov/v2"
    
    # API超时设置
    REQUEST_TIMEOUT = 30  # 秒
    MAX_RETRIES = 3

# 数据处理配置
class DataConfig:
    """数据处理相关配置"""
    
    # 年份范围
    YEAR_START = 2000
    YEAR_END = 2024
    
    # 默认国家列表 (如果无法从现有数据推断)
    DEFAULT_COUNTRIES = [
        'USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'ITA', 'BRA', 'CAN',
        'RUS', 'AUS', 'KOR', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'CHE',
        'ARG', 'NOR', 'POL', 'BEL', 'IRN', 'THA', 'IRQ', 'ARE', 'EGY', 'ISR',
        'MYS', 'SGP', 'PHL', 'VNM', 'BGD', 'UKR', 'DZA', 'QAT', 'KWT', 'NGA'
    ]
    
    # 数据质量阈值
    MAX_MISSING_RATE = 0.8  # 最大缺失率 80%
    MIN_OBSERVATIONS = 10   # 每个国家最少观测值
    
    # 异常值处理
    OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', 'none'
    OUTLIER_THRESHOLD = 3.0
    
    # 数值处理
    LOG_TRANSFORM_VARS = ['gdp_current_usd', 'population_total']
    CLIP_BOUNDS = {
        'ovi_gas': (0, 10),  # OVI_gas合理范围
        'ovi_oil': (0, 10),  # OVI_oil合理范围
        'vul_us': (0, 1),    # 脆弱性指数范围
        'node_dli_us': (0, 1)  # DLI范围
    }

# 变量构建配置
class VariableConfig:
    """变量构建相关配置"""
    
    # Node-DLI_US构建参数
    NODE_DLI_CONFIG = {
        'weight_method': 'trade_share',  # 'trade_share', 'equal', 'value_weighted'
        'min_trade_threshold': 1e6,      # 最小贸易额阈值 (美元)
        'aggregation_method': 'weighted_mean'  # 'weighted_mean', 'max', 'sum'
    }
    
    # Vul_US构建参数
    VUL_US_CONFIG = {
        'hhi_method': 'standard',        # 'standard', 'normalized'
        'include_domestic': False,       # 是否包含国内生产
        'min_suppliers': 2               # 最少供应商数量
    }
    
    # OVI构建参数
    OVI_CONFIG = {
        'capacity_units': 'bcm_per_year',  # 容量单位
        'demand_proxy': 'total_imports',   # 需求代理变量
        'smooth_method': 'rolling_3y',     # 平滑方法
        'interpolate_missing': True        # 是否插值缺失值
    }
    
    # 单位换算系数
    UNIT_CONVERSIONS = {
        'MTPA_TO_BCM': 1.36,  # MTPA (百万吨/年) 到 BCM (十亿立方米) 的换算系数
        'BCM_TO_MTOE': 0.9,   # BCM (十亿立方米) 到 MTOE (百万吨石油当量) 的换算系数
        'BPD_TO_MTPA': 50,    # 百万桶/天 到 百万吨/年 的换算系数
        'BCF_TO_BCM': 0.0283, # BCF (十亿立方英尺) 到 BCM (十亿立方米) 的换算系数
        'TCF_TO_BCM': 28.32,  # TCF (万亿立方英尺) 到 BCM (十亿立方米) 的换算系数
    }
    
    # US_ProdShock构建参数
    PROD_SHOCK_CONFIG = {
        'filter_method': 'hp',           # 'hp', 'bandpass', 'linear_trend'
        'hp_lambda': 100,                # HP滤波参数
        'shock_definition': 'cycle',     # 'cycle', 'growth_rate', 'deviation'
        'normalize': True                # 是否标准化冲击
    }

# 文件路径配置
class PathConfig:
    """文件路径相关配置"""
    
    # 输入文件路径映射
    INPUT_PATHS = {
        'trade_flow': [
            SRC_DIR / "01_data_processing" / "cleaned_trade_flow.csv",
            SRC_DIR / "01_data_processing" / "trade_data.csv",
            SRC_DIR / "01_data_processing" / "processed_trade_data.csv"
        ],
        'node_metrics': [
            SRC_DIR / "03_metrics" / "node_centrality_metrics.csv",
            SRC_DIR / "03_metrics" / "all_metrics.csv"
        ],
        'global_metrics': [
            SRC_DIR / "03_metrics" / "global_network_metrics.csv"
        ],
        'dli_panel': [
            SRC_DIR / "04_dli_analysis" / "dli_panel_data.csv"
        ]
    }
    
    # 输出文件路径
    OUTPUT_PATHS = {
        'analytical_panel': [
            DATA_DIR / "processed_data" / "analytical_panel.csv",
            OUTPUT_DIR / "analytical_panel.csv"
        ],
        'data_dictionary': OUTPUT_DIR / "data_dictionary.md",
        'construction_log': MODULE_DIR / "variable_construction.log"
    }
    
    # 中间文件路径
    TEMP_PATHS = {
        'macro_controls': TEMP_DATA_DIR / "macro_controls.csv",
        'node_dli_us': TEMP_DATA_DIR / "node_dli_us.csv",
        'vul_us': TEMP_DATA_DIR / "vul_us.csv",
        'ovi_gas': TEMP_DATA_DIR / "ovi_gas.csv",
        'ovi_oil': TEMP_DATA_DIR / "ovi_oil.csv",
        'us_prod_shock': TEMP_DATA_DIR / "us_prod_shock.csv",
        'ovi_gas_components': TEMP_DATA_DIR / "ovi_gas_components.csv",
        'ovi_oil_components': TEMP_DATA_DIR / "ovi_oil_components.csv"
    }

# 模拟数据配置
class MockDataConfig:
    """模拟数据生成配置"""
    
    # 随机种子 (保证可重复性)
    RANDOM_SEED = 42
    
    # 模拟数据规模
    N_COUNTRIES = 30
    N_YEARS = 25  # 2000-2024
    TRADE_EDGE_PROBABILITY = 0.3  # 30%的国家对有贸易关系
    DLI_EDGE_PROBABILITY = 0.2    # 20%的边有DLI数据
    
    # 模拟参数分布
    MOCK_DISTRIBUTIONS = {
        'gdp': {'loc': 25, 'scale': 1.5, 'dist': 'lognormal'},
        'population': {'loc': 15, 'scale': 1.0, 'dist': 'lognormal'},
        'trade_openness': {'loc': 50, 'scale': 15, 'dist': 'normal'},
        'dli_score': {'loc': 0.4, 'scale': 0.2, 'dist': 'beta'},
        'centrality': {'scale': 0.1, 'dist': 'exponential'},
        'trade_value': {'loc': 15, 'scale': 2, 'dist': 'lognormal'}
    }

# 日志配置
class LogConfig:
    """日志相关配置"""
    
    LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = MODULE_DIR / "variable_construction.log"
    
    # 控制台输出
    CONSOLE_OUTPUT = True
    CONSOLE_LEVEL = 'INFO'
    
    # 文件输出
    FILE_OUTPUT = True
    FILE_LEVEL = 'DEBUG'

# 性能配置
class PerformanceConfig:
    """性能优化配置"""
    
    # 并行处理
    USE_MULTIPROCESSING = False  # 暂时关闭以避免复杂性
    N_WORKERS = 4
    
    # 内存优化
    CHUNK_SIZE = 10000           # 大数据集分块处理
    LOW_MEMORY_MODE = False      # 低内存模式
    
    # 缓存设置
    ENABLE_CACHE = True
    CACHE_DIR = TEMP_DATA_DIR / "cache"

# 验证配置
class ValidationConfig:
    """数据验证配置"""
    
    # 必需变量 (如果缺失则报错)
    REQUIRED_VARS = ['year', 'country']
    
    # 期望变量 (如果缺失则警告)
    EXPECTED_VARS = [
        'gdp_current_usd', 'population_total', 'trade_openness_gdp_pct',
        'node_dli_us', 'vul_us', 'ovi_gas', 'ovi_oil'
    ]
    
    # 数据类型验证
    VAR_TYPES = {
        'year': 'int',
        'country': 'str',
        'gdp_current_usd': 'float',
        'population_total': 'float',
        'node_dli_us': 'float',
        'vul_us': 'float',
        'ovi_gas': 'float',
        'ovi_oil': 'float'
    }
    
    # 数值范围验证
    VAR_RANGES = {
        'year': (1990, 2030),
        'gdp_current_usd': (0, 1e15),
        'population_total': (0, 2e9),
        'trade_openness_gdp_pct': (0, 500),
        'node_dli_us': (0, 1),
        'vul_us': (0, 1),
        'ovi_gas': (0, 20),
        'ovi_oil': (0, 20)
    }

# 集成配置类
class Config:
    """主配置类，整合所有配置"""
    
    api = APIConfig()
    data = DataConfig()
    variables = VariableConfig()
    paths = PathConfig()
    mock = MockDataConfig()
    log = LogConfig()
    performance = PerformanceConfig()
    validation = ValidationConfig()
    
    @classmethod
    def get_country_list(cls) -> List[str]:
        """获取国家列表"""
        return cls.data.DEFAULT_COUNTRIES
    
    @classmethod
    def get_year_range(cls) -> tuple:
        """获取年份范围"""
        return (cls.data.YEAR_START, cls.data.YEAR_END)
    
    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """验证文件路径是否存在"""
        status = {}
        
        for category, paths in cls.paths.INPUT_PATHS.items():
            if isinstance(paths, list):
                status[category] = any(p.exists() for p in paths)
            else:
                status[category] = paths.exists()
        
        return status
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'api': cls.api.__dict__,
            'data': cls.data.__dict__,
            'variables': cls.variables.__dict__,
            'mock': cls.mock.__dict__,
            'log': cls.log.__dict__,
            'performance': cls.performance.__dict__,
            'validation': cls.validation.__dict__
        }

# 导出主要配置
config = Config()

# 便捷函数
def get_config() -> Config:
    """获取配置实例"""
    return config

def print_config_summary():
    """打印配置摘要"""
    print("📋 08_variable_construction 配置摘要")
    print("=" * 50)
    print(f"年份范围: {config.data.YEAR_START}-{config.data.YEAR_END}")
    print(f"默认国家数: {len(config.data.DEFAULT_COUNTRIES)}")
    print(f"输出目录: {config.paths.OUTPUT_PATHS['analytical_panel'][0].parent}")
    print(f"日志级别: {config.log.LOG_LEVEL}")
    print(f"API配置: WB={bool(config.api.WB_INDICATORS)}, EIA={bool(config.api.EIA_API_KEY)}")
    
    # 检查输入文件状态
    path_status = config.validate_paths()
    print("\n📁 输入文件状态:")
    for category, exists in path_status.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {category}")

if __name__ == "__main__":
    print_config_summary()