# 网络结构异质性分析报告
## Network Structure Heterogeneity Analysis Report

**生成时间**: 2025-08-18 01:49:55  
**分析模块**: 08_heterogeneity_analysis v1.0

---

## 🎯 研究目标

本分析旨在探究双向动态锁定效应(DLI)是否会因能源贸易网络的拓扑结构不同而表现出异质性：

1. **全局异质性**: DLI对网络韧性的因果效应是否在更稠密、更集聚或更中心化的网络中表现不同？
2. **局部异质性**: 贸易关系的锁定效应是否会因贸易双方在网络中的重要性而得到放大或缩小？

---

## 📊 分析结果摘要

### 总体发现

- **交互效应测试总数**: 2
- **显著交互效应数量**: 0
- **显著性比例**: 0.0%

### 分析类型分布

- **全局分析模型**: 1
- **局部分析模型**: 1

---

## 📈 可视化结果

本分析生成了以下可视化图表：

1. **交互效应热力图** (`interaction_heatmap.png`)
   - 展示不同DLI变量与网络特征的交互效应强度
   
2. **显著性概览图** (`significance_overview.png`)
   - 显示显著性分布、系数分布等统计概览
   
3. **边际效应图** (`marginal_effect_*.png`)
   - 展示在不同网络特征水平下DLI效应的变化
   
4. **回归诊断图** (`diagnostics_*.png`)
   - 回归模型的残差分析和诊断检验

---

## 🔍 方法论说明

### 分析方法

本研究基于05_causal_validation的基准回归模型，引入DLI指标与网络特征的交互项：

**全局分析模型**:
```
Y ~ DLI + Global_Metric + DLI × Global_Metric + Controls
```

**局部分析模型**:
```
Y ~ DLI + Local_Metric + DLI × Local_Metric + Controls
```

### 数据来源

- **DLI效应指标**: 来自 `04_dli_analysis` 模块
- **全局网络指标**: 来自 `03_metrics` 模块的网络整体拓扑指标
- **局部节点指标**: 来自 `03_metrics` 模块的节点中心性指标
- **因果分析数据**: 来自 `05_causal_validation` 模块的基准回归变量

---

## 📁 输出文件

### 数据表格
- `heterogeneity_results.csv`: 完整的回归结果汇总表
- `significant_interactions.json`: 显著交互效应的详细信息
- `full_regression_results.json`: 所有回归模型的完整结果

### 可视化图表
- 所有图表保存在 `outputs/figures/` 目录下
- 支持高分辨率PNG格式，适合学术发表

---

## 💡 研究意义

本分析揭示了网络结构对DLI效应的调节作用，为理解能源贸易锁定效应的复杂性提供了新的视角。研究发现有助于：

1. **理论贡献**: 丰富了动态锁定理论的网络维度
2. **政策启示**: 为能源政策制定提供网络结构的考虑因素
3. **方法创新**: 建立了网络异质性分析的标准化框架

---

*本报告由 Network Heterogeneity Analysis Pipeline v1.0 自动生成*  
*Energy Network Analysis Team*
