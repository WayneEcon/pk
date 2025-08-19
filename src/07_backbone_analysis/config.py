#!/usr/bin/env python3
"""
骨干网络分析配置模块
====================

包含所有配置参数的集中管理。
简化的配置系统，避免过度复杂化。

配置类别：
1. 数据路径配置
2. 算法参数配置
3. 验证标准配置
4. 输出格式配置

作者：Energy Network Analysis Team
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class AnalysisConfig:
    """骨干网络分析配置类"""
    
    # === 数据路径配置 ===
    data_path: str = "../../data/processed_data"
    output_path: str = "./outputs"
    figures_path: str = "./figures"
    
    # === 分析范围配置 ===
    analysis_years: Optional[List[int]] = None  # 全面分析的年份
    visualization_years: Optional[List[int]] = None  # 重点可视化的年份
    
    # === 算法参数配置 ===
    algorithms: List[str] = field(default_factory=lambda: ['disparity_filter', 'mst', 'polya_urn'])
    alpha_values: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    beta_value: float = 0.05  # Pólya Urn Filter参数
    weight_attr: str = 'weight'
    
    # === 验证标准配置 ===
    validation_standards: Dict[str, float] = field(default_factory=lambda: {
        'spearman_threshold': 0.7,      # Spearman相关系数阈值
        'stability_threshold': 0.8,     # 稳定性阈值
        'significance_level': 0.05,     # 统计显著性水平
        'consistency_threshold': 0.75   # 跨算法一致性阈值
    })
    
    # === 输出控制配置 ===
    generate_reports: bool = True
    create_visualizations: bool = True
    run_validation: bool = True
    save_networks: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['markdown', 'json'])
    
    # === 可视化配置 ===
    visualization_config: Dict[str, Any] = field(default_factory=lambda: {
        'figure_dpi': 300,
        'figure_format': 'png',
        'node_size_range': (100, 1000),
        'edge_width_range': (0.5, 3.0),
        'layout_algorithm': 'spring',
        'layout_seed': 42
    })
    
    # === 日志配置 ===
    log_level: str = 'INFO'
    log_to_file: bool = True
    
    def __post_init__(self):
        """后处理：设置默认值和验证配置"""
        
        # 设置默认分析年份
        if self.analysis_years is None:
            self.analysis_years = list(range(2008, 2021))  # 2008-2020
        
        # 设置默认可视化年份（最近3年）
        if self.visualization_years is None:
            if self.analysis_years:
                self.visualization_years = sorted(self.analysis_years)[-3:]
            else:
                self.visualization_years = [2018, 2019, 2020]
        
        # 确保路径为Path对象
        self.data_path = Path(self.data_path)
        self.output_path = Path(self.output_path)
        self.figures_path = Path(self.figures_path)
        
        # 验证配置参数
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        
        # 验证算法列表
        valid_algorithms = ['disparity_filter', 'mst', 'polya_urn']
        for alg in self.algorithms:
            if alg not in valid_algorithms:
                raise ValueError(f"不支持的算法: {alg}。支持的算法: {valid_algorithms}")
        
        # 验证alpha值范围
        for alpha in self.alpha_values:
            if not (0 < alpha < 1):
                raise ValueError(f"Alpha值必须在(0,1)范围内: {alpha}")
        
        # 验证beta值范围
        if not (0 < self.beta_value < 1):
            raise ValueError(f"Beta值必须在(0,1)范围内: {self.beta_value}")
        
        # 验证验证标准
        for threshold_name, threshold_value in self.validation_standards.items():
            if not (0 <= threshold_value <= 1):
                raise ValueError(f"验证标准{threshold_name}必须在[0,1]范围内: {threshold_value}")
    
    def save_config(self, filepath: Path):
        """保存配置到文件"""
        
        config_dict = {
            'data_path': str(self.data_path),
            'output_path': str(self.output_path),
            'figures_path': str(self.figures_path),
            'analysis_years': self.analysis_years,
            'visualization_years': self.visualization_years,
            'algorithms': self.algorithms,
            'alpha_values': self.alpha_values,
            'beta_value': self.beta_value,
            'weight_attr': self.weight_attr,
            'validation_standards': self.validation_standards,
            'generate_reports': self.generate_reports,
            'create_visualizations': self.create_visualizations,
            'run_validation': self.run_validation,
            'save_networks': self.save_networks,
            'export_formats': self.export_formats,
            'visualization_config': self.visualization_config,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: Path) -> 'AnalysisConfig':
        """从文件加载配置"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def create_output_directories(self):
        """创建输出目录"""
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_path / 'networks').mkdir(exist_ok=True)
        (self.output_path / 'reports').mkdir(exist_ok=True)
        (self.output_path / 'validation').mkdir(exist_ok=True)
    
    def get_algorithm_params(self, algorithm: str) -> Dict[str, Any]:
        """获取特定算法的参数"""
        
        if algorithm == 'disparity_filter':
            return {
                'alpha_values': self.alpha_values,
                'fdr_correction': True,
                'weight_attr': self.weight_attr
            }
        elif algorithm == 'mst':
            return {
                'weight_attr': self.weight_attr,
                'algorithm': 'kruskal'
            }
        elif algorithm == 'polya_urn':
            return {
                'beta': self.beta_value,
                'weight_attr': self.weight_attr
            }
        else:
            raise ValueError(f"未知算法: {algorithm}")
    
    def get_validation_params(self) -> Dict[str, Any]:
        """获取验证参数"""
        
        return {
            'spearman_threshold': self.validation_standards['spearman_threshold'],
            'stability_threshold': self.validation_standards['stability_threshold'],
            'significance_level': self.validation_standards['significance_level'],
            'consistency_threshold': self.validation_standards['consistency_threshold']
        }
    
    def get_visualization_params(self) -> Dict[str, Any]:
        """获取可视化参数"""
        
        return {
            'figure_dpi': self.visualization_config['figure_dpi'],
            'figure_format': self.visualization_config['figure_format'],
            'node_size_range': self.visualization_config['node_size_range'],
            'edge_width_range': self.visualization_config['edge_width_range'],
            'layout_algorithm': self.visualization_config['layout_algorithm'],
            'layout_seed': self.visualization_config['layout_seed']
        }


# === 预定义配置模板 ===

def get_quick_demo_config() -> AnalysisConfig:
    """快速演示配置"""
    
    return AnalysisConfig(
        analysis_years=[2018, 2020],
        visualization_years=[2018, 2020],
        algorithms=['disparity_filter', 'mst'],
        alpha_values=[0.05],
        generate_reports=True,
        create_visualizations=True,
        run_validation=False  # 快速模式跳过验证
    )


def get_full_analysis_config() -> AnalysisConfig:
    """完整分析配置"""
    
    return AnalysisConfig(
        analysis_years=list(range(2008, 2021)),
        visualization_years=[2011, 2018, 2020],  # 关键年份
        algorithms=['disparity_filter', 'mst', 'polya_urn'],
        alpha_values=[0.01, 0.05, 0.1],
        generate_reports=True,
        create_visualizations=True,
        run_validation=True
    )


def get_validation_focused_config() -> AnalysisConfig:
    """验证重点配置"""
    
    return AnalysisConfig(
        analysis_years=list(range(2010, 2021)),
        visualization_years=[2018, 2020],
        algorithms=['disparity_filter', 'mst'],
        alpha_values=[0.01, 0.05, 0.1, 0.2],  # 更多alpha值用于敏感性分析
        generate_reports=True,
        create_visualizations=False,  # 专注于验证，减少可视化
        run_validation=True,
        validation_standards={
            'spearman_threshold': 0.75,     # 更严格的标准
            'stability_threshold': 0.85,
            'significance_level': 0.01,
            'consistency_threshold': 0.8
        }
    )


# === 常用配置常量 ===

# 关键年份
KEY_YEARS = {
    'shale_revolution': 2011,
    'policy_change': 2016,
    'pre_covid': 2019,
    'covid_impact': 2020
}

# 重要国家列表
IMPORTANT_COUNTRIES = [
    'USA', 'CHN', 'RUS', 'SAU', 'CAN', 'GBR', 'DEU', 'JPN', 'IND', 'BRA'
]

# 算法默认参数
DEFAULT_ALGORITHM_PARAMS = {
    'disparity_filter': {
        'alpha_values': [0.01, 0.05, 0.1],
        'fdr_correction': True
    },
    'mst': {
        'algorithm': 'kruskal',
        'symmetrize_method': 'max'
    },
    'polya_urn': {
        'beta': 0.05
    }
}

# 验证标准
ACADEMIC_STANDARDS = {
    'spearman_threshold': 0.7,      # 学术标准：Spearman > 0.7
    'stability_threshold': 0.8,     # 学术标准：稳定性 > 80%
    'significance_level': 0.05,     # 学术标准：p < 0.05
    'consistency_threshold': 0.75   # 学术标准：一致性 > 75%
}