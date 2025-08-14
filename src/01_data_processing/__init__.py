"""
数据处理模块 (01_data_processing)
====================================

负责原始能源贸易数据的清洗、验证和预处理工作。

主要功能：
- 原始数据加载和验证
- 数据清洗和标准化
- 缺失值处理
- 数据质量检查
- 清洗后数据导出

使用说明：
    这个模块包含一个主要脚本 01_data_processing.py，
    用于处理从UN Comtrade等来源获取的原始能源贸易数据。
    
    运行方式：
    cd src/01_data_processing
    python 01_data_processing.py
"""

__version__ = '1.0.0'
__author__ = 'Energy Network Analysis Team'