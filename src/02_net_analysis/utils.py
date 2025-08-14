#!/usr/bin/env python3
"""
通用工具模块
提供统一的导入、验证和辅助功能
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd

# 统一的路径设置
def setup_path() -> None:
    """设置统一的模块搜索路径"""
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))

# 确保在模块加载时设置路径
setup_path()

# 统一导入配置
try:
    from config import *
except ImportError as e:
    logging.error(f"无法导入配置文件: {e}")
    raise

def validate_dataframe_columns(df: pd.DataFrame, required_columns: Union[List[str], Set[str]], 
                             name: str = "数据框") -> None:
    """
    验证DataFrame是否包含必要的列
    
    Args:
        df: 要验证的DataFrame
        required_columns: 必要的列名列表或集合
        name: 数据框名称，用于错误信息
        
    Raises:
        ValueError: 当缺少必要列时
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> validate_dataframe_columns(df, ['A', 'B'], "测试数据")
    """
    if df.empty:
        raise ValueError(f"{name}为空")
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name}缺少必要字段: {missing_cols}")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 当分母为0时返回的默认值
        
    Returns:
        除法结果或默认值
        
    Example:
        >>> safe_divide(10, 2)  # 返回 5.0
        >>> safe_divide(10, 0)  # 返回 0.0
    """
    return numerator / denominator if denominator != 0 else default

def log_dataframe_info(df: pd.DataFrame, name: str, year: Optional[int] = None, 
                      logger: Optional[logging.Logger] = None) -> None:
    """
    记录DataFrame的基本信息
    
    Args:
        df: 要记录的DataFrame
        name: 数据描述名称
        year: 可选的年份信息
        logger: 可选的日志记录器，默认使用当前模块的logger
        
    Example:
        >>> log_dataframe_info(df, "贸易数据", 2020)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    year_str = f"{year}: " if year is not None else ""
    
    if df.empty:
        logger.warning(f"     {year_str}{name}: 数据为空")
    else:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        logger.info(f"     {year_str}{name}: {len(df):,} 行 x {len(df.columns)} 列 "
                   f"(内存: {memory_usage:.1f}MB)")

def create_trade_flow_id(source: str, target: str) -> str:
    """
    创建标准化的贸易流标识符
    
    Args:
        source: 源国家代码
        target: 目标国家代码
        
    Returns:
        格式为"source_to_target"的贸易流ID
        
    Example:
        >>> create_trade_flow_id("USA", "CHN")  # 返回 "USA_to_CHN"
    """
    return f"{source}_to_{target}"

def get_country_region_safe(country_code: str, default: str = 'Other') -> str:
    """
    安全地获取国家所属区域
    
    Args:
        country_code: 国家代码
        default: 当无法找到区域时返回的默认值
        
    Returns:
        国家所属区域名称
        
    Example:
        >>> get_country_region_safe("USA")  # 返回对应区域或'Other'
    """
    try:
        for region, countries in REGIONAL_GROUPS.items():
            if country_code in countries:
                return region
        return default
    except (NameError, AttributeError):
        # 如果REGIONAL_GROUPS未定义，返回默认值
        return default

def validate_network_graph(G: Any, name: str = "网络图") -> None:
    """
    验证NetworkX图对象的有效性
    
    Args:
        G: 要验证的图对象
        name: 图对象名称，用于错误信息
        
    Raises:
        TypeError: 当对象不是NetworkX图时
        ValueError: 当图为空或缺少必要属性时
        
    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> validate_network_graph(G, "测试网络")
    """
    import networkx as nx
    
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"{name}必须是NetworkX图对象")
    
    if G.number_of_nodes() == 0:
        logging.getLogger(__name__).warning(f"{name}没有节点")

def validate_trade_data_schema(df: pd.DataFrame, year: Optional[int] = None) -> List[str]:
    """
    验证贸易数据的模式和数据质量
    
    Args:
        df: 要验证的贸易数据DataFrame
        year: 可选的年份信息
        
    Returns:
        验证警告列表，空列表表示数据完全合格
        
    Example:
        >>> warnings = validate_trade_data_schema(df, 2020)
        >>> if warnings:
        ...     print("数据质量问题:", warnings)
    """
    warnings = []
    year_str = f"{year}年" if year is not None else ""
    
    if df.empty:
        warnings.append(f"{year_str}数据为空")
        return warnings
    
    # 检查必要列
    required_cols = ['source', 'target', 'trade_value_raw_usd']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        warnings.append(f"{year_str}缺少必要列: {missing_cols}")
    
    # 检查数据类型
    if 'trade_value_raw_usd' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['trade_value_raw_usd']):
            warnings.append(f"{year_str}trade_value_raw_usd列应为数值类型")
    
    # 检查缺失值
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_info = missing_data[missing_data > 0]
        warnings.append(f"{year_str}存在缺失值: {dict(missing_info)}")
    
    # 检查负值贸易额
    if 'trade_value_raw_usd' in df.columns:
        negative_count = (df['trade_value_raw_usd'] < 0).sum()
        if negative_count > 0:
            warnings.append(f"{year_str}存在 {negative_count} 条负值贸易记录")
    
    # 检查重复记录
    if 'source' in df.columns and 'target' in df.columns:
        duplicate_count = df.duplicated(subset=['source', 'target']).sum()
        if duplicate_count > 0:
            warnings.append(f"{year_str}存在 {duplicate_count} 条重复的贸易对")
    
    # 检查自环（国家与自己的贸易）
    if 'source' in df.columns and 'target' in df.columns:
        self_trade = (df['source'] == df['target']).sum()
        if self_trade > 0:
            warnings.append(f"{year_str}存在 {self_trade} 条自环贸易记录")
    
    return warnings

def validate_statistics_data(stats_list: List[Dict], expected_years: Optional[List[int]] = None) -> List[str]:
    """
    验证统计数据的完整性和一致性
    
    Args:
        stats_list: 统计数据字典列表
        expected_years: 期望包含的年份列表
        
    Returns:
        验证警告列表
        
    Example:
        >>> warnings = validate_statistics_data(stats, [2020, 2021, 2022])
        >>> print(f"发现 {len(warnings)} 个统计数据问题")
    """
    warnings = []
    
    if not stats_list:
        warnings.append("统计数据为空")
        return warnings
    
    # 检查年份完整性
    if expected_years:
        actual_years = [stat.get('year') for stat in stats_list if 'year' in stat]
        missing_years = set(expected_years) - set(actual_years)
        if missing_years:
            warnings.append(f"缺少年份的统计数据: {sorted(missing_years)}")
    
    # 检查必要字段
    required_fields = ['year', 'nodes', 'edges', 'total_trade_value']
    for i, stat in enumerate(stats_list):
        missing_fields = set(required_fields) - set(stat.keys())
        if missing_fields:
            year = stat.get('year', f'第{i+1}条')
            warnings.append(f"{year}统计数据缺少字段: {missing_fields}")
    
    # 检查数据一致性
    for stat in stats_list:
        year = stat.get('year', '未知年份')
        
        # 检查节点边数合理性
        nodes = stat.get('nodes', 0)
        edges = stat.get('edges', 0)
        if edges > nodes * (nodes - 1):  # 有向图最大边数
            warnings.append(f"{year}边数({edges})超过理论最大值({nodes * (nodes - 1)})")
        
        # 检查密度合理性
        density = stat.get('density', 0)
        if density < 0 or density > 1:
            warnings.append(f"{year}网络密度({density})超出合理范围[0,1]")
        
        # 检查贸易额非负性
        total_trade = stat.get('total_trade_value', 0)
        if total_trade < 0:
            warnings.append(f"{year}总贸易额为负值({total_trade})")
    
    return warnings

class DataQualityReporter:
    """
    数据质量报告工具类
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.reports: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
    
    def add_report(self, step: str, year: int, before_count: int, after_count: int, 
                  description: str = "") -> None:
        """
        添加数据处理步骤的报告
        
        Args:
            step: 处理步骤名称
            year: 数据年份
            before_count: 处理前记录数
            after_count: 处理后记录数
            description: 可选的额外描述
        """
        change_rate = safe_divide(after_count - before_count, before_count, 0)
        
        report = {
            'step': step,
            'year': year,
            'before_count': before_count,
            'after_count': after_count,
            'change_count': after_count - before_count,
            'change_rate': change_rate,
            'description': description
        }
        
        self.reports.append(report)
        
        # 记录日志
        change_str = f"({change_rate:+.1%})" if change_rate != 0 else ""
        self.logger.info(f"     {year}: {step} - {before_count:,} -> {after_count:,} {change_str}")
        if description:
            self.logger.debug(f"     {year}: {description}")
    
    def add_warning(self, warning: str) -> None:
        """
        添加数据质量警告
        
        Args:
            warning: 警告信息
        """
        self.warnings.append(warning)
        self.logger.warning(warning)
    
    def validate_and_report(self, df: pd.DataFrame, name: str, year: Optional[int] = None) -> None:
        """
        验证数据并添加到报告中
        
        Args:
            df: 要验证的DataFrame
            name: 数据名称
            year: 可选的年份
        """
        warnings = validate_trade_data_schema(df, year)
        for warning in warnings:
            self.add_warning(f"{name}: {warning}")
    
    def get_summary(self) -> pd.DataFrame:
        """
        获取所有报告的汇总DataFrame
        
        Returns:
            包含所有处理步骤报告的DataFrame
        """
        return pd.DataFrame(self.reports) if self.reports else pd.DataFrame()
    
    def get_warnings_summary(self) -> List[str]:
        """
        获取所有警告的汇总
        
        Returns:
            警告列表
        """
        return self.warnings.copy()
    
    def clear(self) -> None:
        """清空所有报告和警告"""
        self.reports.clear()
        self.warnings.clear()