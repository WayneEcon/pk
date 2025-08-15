#!/usr/bin/env python3
"""
骨干网络分析模块 (Backbone Network Analysis Module)
===================================================

本模块实现多种骨干网络提取算法，用于验证"轨道一"发现的美国能源地位变化：
1. Disparity Filter: 基于统计显著性的骨干提取
2. Maximum Spanning Tree: 保持连通性的最小骨架
3. Pólya Urn Filter: 高级统计建模方法

核心验证问题：
- 美国中心性的变化在骨干网络中是否依然成立？
- 关键贸易关系的重组是否在不同算法下一致？ 
- DLI指标识别的锁定关系如何在骨干网络中体现？

作者：Energy Network Analysis Team
版本：1.0
"""

__version__ = "1.0.0"
__author__ = "Energy Network Analysis Team"

# 导入核心模块
from .algorithms import disparity_filter, maximum_spanning_tree
from .data_io import network_loader
from .visualization import network_layout