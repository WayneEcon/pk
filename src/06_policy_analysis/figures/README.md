# 政策影响分析可视化图表

本文件夹包含美国能源独立政策影响分析的所有可视化图表。

## 图表类型

### 1. 国家仪表盘 (Country Dashboards)
展示单个国家在多个网络指标上的时间序列变化，包含政策前、过渡期、政策后三个阶段的背景色块。

**文件命名规则**: `{国家代码}_dashboard.png`

**包含国家**:
- `ARE_dashboard.png` - 阿联酋
- `AUS_dashboard.png` - 澳大利亚  
- `CHN_dashboard.png` - 中国
- `DEU_dashboard.png` - 德国
- `FRA_dashboard.png` - 法国
- `IND_dashboard.png` - 印度
- `JPN_dashboard.png` - 日本
- `KOR_dashboard.png` - 韩国
- `KWT_dashboard.png` - 科威特
- `NGA_dashboard.png` - 尼日利亚
- `NLD_dashboard.png` - 荷兰
- `NOR_dashboard.png` - 挪威
- `QAT_dashboard.png` - 卡塔尔
- `RUS_dashboard.png` - 俄罗斯
- `SAU_dashboard.png` - 沙特阿拉伯
- `SGP_dashboard.png` - 新加坡
- `USA_dashboard.png` - 美国

### 2. 期间对比图 (Period Comparison Charts)
展示各指标在政策前后期间的变化对比，显示变化最大的前10个国家。

**文件命名规则**: `{指标名}_period_comparison.png`

**包含指标**:
- `betweenness_centrality_period_comparison.png` - 中介中心性
- `in_degree_period_comparison.png` - 入度中心性
- `in_strength_period_comparison.png` - 入强度
- `out_degree_period_comparison.png` - 出度中心性  
- `out_strength_period_comparison.png` - 出强度
- `pagerank_centrality_period_comparison.png` - PageRank中心性
- `total_degree_period_comparison.png` - 总度数
- `total_strength_period_comparison.png` - 总强度

### 3. 综合分析图表

- `metrics_change_correlation.png` - 各指标变化量的相关性热力图
- `policy_impact_overview.png` - 政策影响统计概览，包含平均变化、显著性检验、国家数量变化等四个子图

## 政策期间划分

- **政策前期 (Pre-Policy)**: 2001-2008年，蓝色背景
- **过渡期 (Transition)**: 2009-2015年，紫色背景  
- **政策后期 (Post-Policy)**: 2016-2024年，橙色背景

## 生成脚本

所有图表由以下脚本生成：
- `../plotting.py` - 包含所有绘图函数
- `../main.py` - 主执行脚本
- `../analysis.py` - 数据分析处理

## 数据来源

图表基于以下数据：
- 原始数据：`../../../03_metrics/all_metrics.csv`
- 分析结果：`../policy_impact_summary.csv`
- 统计结果：`../policy_impact_statistics.json`

---
*生成时间：2025-08-14*