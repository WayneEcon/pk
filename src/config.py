#!/usr/bin/env python3
"""
美国能源独立政策研究 - 配置文件
Configuration file for US Energy Independence Policy Research

作者：研究团队
创建日期：2025-08-13
"""

from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# 基础路径配置 (Base Path Configuration)
# =============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
NETWORKS_DIR = PROCESSED_DATA_DIR / "networks"

# 输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
NETWORK_STATS_DIR = OUTPUTS_DIR / "network_stats"

# 日志目录
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# 数据参数配置 (Data Parameters Configuration)
# =============================================================================

# 研究时间范围
START_YEAR = 2001
END_YEAR = 2024
YEARS_RANGE = range(START_YEAR, END_YEAR + 1)

# 能源产品HS代码 (Energy Product HS Codes)
ENERGY_PRODUCT_CODES = {
    2701: "煤炭 (Coal)",
    2709: "原油 (Crude Oil)", 
    2710: "成品油 (Refined Oil)",
    2711: "天然气 (Natural Gas)"
}

# 贸易流向代码 (Trade Flow Codes)
TRADE_FLOWS = {
    'M': 'Import (进口)',
    'X': 'Export (出口)'
}

# =============================================================================
# 数据处理配置 (Data Processing Configuration)
# =============================================================================

# 数据一致性处理策略
DATA_CONSISTENCY_STRATEGY = "prioritize_imports"  # 优先使用进口数据

# 核心数据字段映射
REQUIRED_COLUMNS = {
    'reporter': 'reporter',           # 报告国代码
    'partner': 'partner',             # 伙伴国代码  
    'flow': 'flow',                   # 贸易流向
    'trade_value_raw_usd': 'trade_value_raw_usd',  # 贸易额
    'reporter_name': 'reporter_name', # 报告国名称
    'partner_name': 'partner_name',   # 伙伴国名称
    'year': 'year'                    # 年份
}

# 数据验证阈值
DATA_VALIDATION = {
    'min_trade_value': 1,                 # 最小有效贸易额（美元）- 不过滤任何数据
    'max_missing_ratio': 0.1,             # 最大缺失值比例
    'min_countries_per_year': 50          # 每年最少国家数
}

# =============================================================================
# 网络分析配置 (Network Analysis Configuration)
# =============================================================================

# 核心分析国家（30个重点国家）
FOCUS_COUNTRIES = {
    'USA': '美国',
    'CHN': '中国', 
    'RUS': '俄罗斯',
    'SAU': '沙特阿拉伯',
    'ARE': '阿联酋',
    'NOR': '挪威',
    'QAT': '卡塔尔',
    'AUS': '澳大利亚',
    'DEU': '德国',
    'JPN': '日本',
    'KOR': '韩国',
    'IND': '印度',
    'GBR': '英国',
    'FRA': '法国',
    'ITA': '意大利',
    'ESP': '西班牙',
    'NLD': '荷兰',
    'BEL': '比利时',
    'CAN': '加拿大',
    'MEX': '墨西哥',
    'BRA': '巴西',
    'ARG': '阿根廷',
    'VEN': '委内瑞拉',
    'COL': '哥伦比亚',
    'IRN': '伊朗',
    'IRQ': '伊拉克',
    'KWT': '科威特',
    'OMN': '阿曼',
    'IDN': '印度尼西亚',
    'THA': '泰国'
}

# 网络构建参数
NETWORK_PARAMS = {
    'directed': True,                 # 有向图
    'self_loops': False,             # 不允许自环
    'multigraph': False,             # 不允许重边
    'min_edge_weight': 1             # 最小边权重（美元）- 不过滤任何数据
}

# 中心性计算参数
CENTRALITY_PARAMS = {
    'calculate_degree': True,
    'calculate_betweenness': True,
    'calculate_eigenvector': True,
    'calculate_pagerank': True,
    'weight_attribute': 'weight'
}

# =============================================================================
# 骨干网络提取配置 (Backbone Network Configuration)
# =============================================================================

# Disparity Filter 参数
DISPARITY_FILTER_PARAMS = {
    'alpha': 0.05,                   # 显著性水平
    'correction_method': 'fdr_bh',   # FDR多重检验校正方法
    'direction': 'undirected'        # 处理方向：'undirected', 'in', 'out'
}

# Polya Urn Filter 参数  
POLYA_URN_PARAMS = {
    'alpha': 0.05,
    'correction_method': 'fdr_bh'
}

# Maximum Spanning Tree 参数
MST_PARAMS = {
    'algorithm': 'kruskal',          # 算法选择
    'weight_attribute': 'weight'
}

# =============================================================================
# 输出文件配置 (Output Files Configuration)
# =============================================================================

# 文件名模板
FILE_TEMPLATES = {
    # 清洗数据文件
    'cleaned_data': 'cleaned_energy_trade_{year}.csv',
    
    # 网络文件
    'network_pickle': 'annual_networks_{start_year}_{end_year}.pkl',
    'network_graphml': 'annual_networks_{start_year}_{end_year}.graphml',
    'nodes_file': 'nodes_{year}.csv',
    'edges_file': 'edges_{year}.csv',
    
    # 统计文件
    'basic_stats': 'network_basic_stats.csv',
    'centrality_stats': 'centrality_stats_{year}.csv',
    'summary_report': 'network_summary_report.md',
    
    # 骨干网络文件
    'backbone_df': 'backbone_disparity_filter_{year}.graphml',
    'backbone_pf': 'backbone_polya_urn_{year}.graphml', 
    'backbone_mst': 'backbone_mst_{year}.graphml'
}

# =============================================================================
# 可视化配置 (Visualization Configuration)
# =============================================================================

# 图形参数
PLOT_PARAMS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'Set2',
    'font_size': 12
}

# 网络布局参数
LAYOUT_PARAMS = {
    'default_layout': 'spring',      # 默认布局算法
    'node_size_range': (100, 1000),  # 节点大小范围
    'edge_width_range': (0.5, 5),    # 边宽度范围
    'alpha': 0.7                     # 透明度
}

# =============================================================================
# 日志配置 (Logging Configuration)
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_name': 'energy_network.log'
}

# =============================================================================
# 区域分组配置 (Regional Groups Configuration)
# =============================================================================

REGIONAL_GROUPS = {
    'North_America': ['USA', 'CAN', 'MEX'],
    'Europe': ['DEU', 'GBR', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'NOR', 'DNK', 'SWE'],
    'Middle_East': ['SAU', 'ARE', 'IRN', 'IRQ', 'KWT', 'QAT', 'OMN', 'BHR'],
    'Asia_Pacific': ['CHN', 'JPN', 'KOR', 'IND', 'AUS', 'IDN', 'THA', 'SGP', 'MYS'],
    'Former_USSR': ['RUS', 'KAZ', 'UZB', 'TKM', 'AZE'],
    'Africa': ['DZA', 'LBY', 'EGY', 'NGA', 'AGO', 'ZAF'],
    'Latin_America': ['BRA', 'ARG', 'VEN', 'COL', 'ECU', 'PER', 'CHL']
}

# =============================================================================
# 验证函数 (Validation Functions)
# =============================================================================

def validate_config():
    """验证配置文件的有效性"""
    
    # 检查年份范围
    assert START_YEAR <= END_YEAR, f"开始年份 {START_YEAR} 不能大于结束年份 {END_YEAR}"
    assert START_YEAR >= 1990, f"开始年份 {START_YEAR} 过早，数据可能不可靠"
    assert END_YEAR <= 2030, f"结束年份 {END_YEAR} 超出合理范围"
    
    # 检查能源产品代码
    assert all(isinstance(code, int) for code in ENERGY_PRODUCT_CODES.keys()), "能源产品代码必须是整数"
    assert all(2700 <= code <= 2799 for code in ENERGY_PRODUCT_CODES.keys()), "能源产品代码应在27章范围内"
    
    # 检查路径
    assert PROJECT_ROOT.exists(), f"项目根目录不存在: {PROJECT_ROOT}"
    
    print("✅ 配置文件验证通过")

if __name__ == "__main__":
    validate_config()