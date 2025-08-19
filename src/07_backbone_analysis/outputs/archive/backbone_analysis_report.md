# 骨干网络分析报告

**生成时间**: 2025-08-15 18:44:33

## 分析概述

- **数据范围**: 2001-2024
- **网络数量**: 24 个年度网络
- **算法类型**: 4 种骨干提取算法

## 算法摘要

- **alpha_0.01**: 24年数据，平均保留率1.0%
- **alpha_0.05**: 24年数据，平均保留率1.5%
- **alpha_0.1**: 24年数据，平均保留率1.8%
- **mst**: 24年数据，平均保留率3.5%

## 输出文件

### 数据文件
- `backbone_statistics.csv`: 骨干网络统计汇总
- `network_consistency_check.json`: 数据一致性检查结果
- `{algorithm}/{year}.graphml`: 各算法的年度骨干网络

### 可视化文件
- `backbone_comparison_{year}.png`: 算法对比图
- `backbone_{algorithm}_{year}.png`: 单算法网络图
- `*_timeseries.png`: 时间序列统计图

---
*本报告由07_backbone_analysis模块自动生成*
