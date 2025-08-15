#!/usr/bin/env python3
"""
算法参数配置文件
Configuration file for backbone extraction algorithms
"""

# Disparity Filter 参数设置
DISPARITY_FILTER_PARAMS = {
    'alpha_values': [0.01, 0.05, 0.1],  # 显著性水平
    'fdr_correction': True,              # 是否使用Benjamini-Hochberg FDR校正
    'directed': True,                    # 是否考虑有向图
    'multiedges': False                  # 是否允许多重边
}

# Maximum Spanning Tree 参数设置
MST_PARAMS = {
    'symmetric_weights': True,           # 是否对称化权重
    'algorithm': 'kruskal'               # 'kruskal' 或 'prim'
}

# Pólya Urn Filter 参数设置（高级选项）
POLYA_URN_PARAMS = {
    'iterations': 1000,                  # 迭代次数
    'confidence_level': 0.95             # 置信水平
}

# 通用设置
GENERAL_PARAMS = {
    'min_edge_weight': 0.0,              # 最小边权重阈值
    'preserve_isolates': False,          # 是否保留孤立节点
    'output_format': 'graphml',          # 输出格式
    'random_seed': 42                    # 随机种子
}

# 验证年份设置（关键时间点）
KEY_YEARS = {
    'baseline': [2001, 2005, 2010],      # 基准期
    'transition': [2011, 2012, 2013],    # 页岩革命过渡期  
    'mature': [2018, 2020, 2024]         # 成熟期
}

# 可视化参数
VISUALIZATION_PARAMS = {
    'layout_algorithm': 'force_atlas_2',  # 布局算法
    'node_size_factor': 100,              # 节点大小因子
    'edge_width_factor': 2,               # 边宽度因子
    'color_scheme': 'geographic',         # 着色方案: 'geographic', 'centrality', 'trade_volume'
    'figure_size': (12, 10),              # 图片大小
    'dpi': 300                            # 图片分辨率
}