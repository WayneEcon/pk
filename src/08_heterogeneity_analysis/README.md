# 08_heterogeneity_analysis: 网络结构异质性分析模块

## 🎯 模块概述

本模块是能源网络分析项目的重要补充章节，专门研究**双向动态锁定效应(DLI)的网络结构异质性**。通过引入交互项分析，探究DLI效应如何因网络拓扑结构的不同而表现出异质性特征。

### 核心研究问题

**Q1 (全局异质性)**: DLI对网络韧性的因果效应，是否在一个更稠密、更集聚、或更中心化的网络中表现得更强或更弱？

**Q2 (局部异质性)**: 一个贸易关系（国家对）的锁定效应，是否会因其贸易双方在网络中的重要性（即节点中心性）而得到放大或缩小？

---

## 🏗️ 模块结构

```
08_heterogeneity_analysis/
├── main.py                 # 主执行脚本，完整分析流程
├── data_loader.py          # 数据加载与预处理
├── analysis.py             # 核心回归分析功能
├── visualizer.py           # 可视化图表生成
├── outputs/                # 分析结果输出目录
│   ├── tables/             # 回归结果表格
│   │   ├── heterogeneity_results.csv
│   │   ├── significant_interactions.json
│   │   └── full_regression_results.json
│   └── figures/            # 可视化图表
│       ├── interaction_heatmap.png
│       ├── significance_overview.png
│       ├── marginal_effect_*.png
│       └── diagnostics_*.png
└── README.md               # 本文档
```

---

## 🚀 快速开始

### 基本用法

```bash
# 快速演示模式（推荐首次使用）
python main.py --mode demo

# 完整分析模式
python main.py --mode full

# 指定输出目录
python main.py --mode demo --output-dir my_analysis
```

### 高级用法

```bash
# 使用配置文件
python main.py --mode full --config config.json --output-dir results
```

配置文件示例 (`config.json`):
```json
{
    "dli_vars": ["dli_composite", "dli_import", "dli_export"],
    "global_vars": ["global_density", "global_transitivity", "global_efficiency"],
    "local_vars": ["betweenness_centrality", "degree_centrality", "pagerank_centrality"],
    "control_vars": ["control_var1", "control_var2"]
}
```

---

## 📊 数据输入

本模块整合来自项目其他模块的数据：

### 必需数据源

1. **DLI效应指标** (来自 `04_dli_analysis`)
   - DLI复合指数
   - 进口锁定指数
   - 出口锁定指数

2. **全局网络指标** (来自 `03_metrics`)
   - 网络密度 (density)
   - 传递性 (transitivity)
   - 平均聚类系数 (clustering)
   - 网络效率 (efficiency)

3. **局部节点指标** (来自 `03_metrics`)
   - 度中心性 (degree centrality)
   - 介数中心性 (betweenness centrality)
   - 特征向量中心性 (eigenvector centrality)
   - PageRank中心性

4. **因果分析基准数据** (来自 `05_causal_validation`)
   - 网络韧性指标
   - 控制变量
   - 面板数据结构

### 数据自动检测

如果找不到真实数据文件，模块会自动生成示例数据以确保程序正常运行。

---

## 🔬 分析方法

### 回归模型设计

本分析基于 `05_causal_validation` 的基准回归模型，引入交互项进行异质性检验：

#### 全局异质性模型
```
Y ~ DLI + Global_Metric + DLI × Global_Metric + Controls
```

#### 局部异质性模型  
```
Y ~ DLI + Local_Metric + DLI × Local_Metric + Controls
```

### 统计方法

- **主要方法**: OLS回归分析 (使用statsmodels)
- **交互效应**: 计算边际效应和条件效应
- **模型诊断**: 残差分析、异方差检验、多重共线性检验
- **稳健性**: VIF检验、Breusch-Pagan检验

---

## 📈 输出结果

### 统计表格

1. **`heterogeneity_results.csv`**: 完整回归结果汇总
   - 交互项系数、标准误、p值
   - 模型拟合度指标
   - 显著性标记

2. **`significant_interactions.json`**: 显著交互效应详情
   - 显著性统计摘要
   - 最强效应识别
   - 边际效应计算

3. **`full_regression_results.json`**: 所有模型完整结果
   - 系数估计值
   - 置信区间
   - 模型诊断指标

### 可视化图表

1. **交互效应热力图**: 系数强度的直观展示
2. **显著性概览图**: 四象限统计概览
3. **边际效应图**: 条件效应变化趋势
4. **回归诊断图**: 模型适用性检验
5. **摘要报告图**: 关键发现总结

---

## 💡 研究意义

### 理论贡献

1. **丰富DLI理论**: 揭示锁定效应的网络结构依赖性
2. **拓展异质性分析**: 建立网络拓扑调节效应的实证框架
3. **深化因果机制**: 从"是否存在"到"何时更强"的递进认识

### 政策启示

1. **网络意识**: 政策制定需考虑网络结构差异
2. **精准施策**: 基于网络特征的差异化策略
3. **风险评估**: 网络脆弱性的动态监测

### 方法创新

1. **标准化流程**: 建立可复制的异质性分析框架
2. **模块化设计**: 支持灵活的变量组合和扩展
3. **可视化系统**: 复杂交互效应的直观表达

---

## 🔧 技术要求

### Python依赖

**必需包**:
```
pandas >= 1.5.0
numpy >= 1.21.0
```

**统计分析** (推荐):
```
statsmodels >= 0.13.0
scipy >= 1.9.0
scikit-learn >= 1.1.0
```

**可视化** (推荐):
```
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

### 安装依赖

```bash
# 基础依赖
pip install pandas numpy

# 完整功能
pip install statsmodels scipy scikit-learn matplotlib seaborn
```

---

## 📋 使用示例

### 程序化调用

```python
from data_loader import HeterogeneityDataLoader
from analysis import HeterogeneityAnalyzer
from visualizer import HeterogeneityVisualizer

# 创建分析管道
loader = HeterogeneityDataLoader()
analyzer = HeterogeneityAnalyzer()
visualizer = HeterogeneityVisualizer()

# 加载数据
global_data, local_data = loader.create_analysis_dataset()

# 运行分析
global_results = analyzer.run_global_analysis(global_data)
local_results = analyzer.run_local_analysis(local_data)

# 生成报告
results_table = analyzer.create_results_table()
significant = analyzer.get_significant_interactions()

# 创建可视化
visualizer.plot_interaction_heatmap(results_table)
visualizer.plot_significance_overview(results_table)
```

### 自定义分析

```python
# 指定特定变量
custom_config = {
    'dli_vars': ['dli_composite'],
    'global_vars': ['global_density', 'global_clustering'],
    'control_vars': ['gdp_log', 'trade_openness']
}

analyzer.run_global_analysis(global_data, **custom_config)
```

---

## 🐛 故障排除

### 常见问题

1. **数据文件未找到**
   - 检查其他模块是否已运行并生成输出
   - 使用演示模式测试功能

2. **统计包缺失**
   - 安装statsmodels: `pip install statsmodels`
   - 使用简化回归模式

3. **可视化错误**
   - 安装matplotlib/seaborn: `pip install matplotlib seaborn`
   - 检查中文字体设置

4. **内存不足**
   - 减少分析变量数量
   - 使用数据子集

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行分析查看详细日志
```

---

## 📞 技术支持

### 模块维护

- **负责团队**: Energy Network Analysis Team
- **版本**: v1.0
- **更新日期**: 2025-08-17

### 扩展说明

本模块采用模块化设计，支持：
- 自定义变量组合
- 新增网络指标
- 扩展可视化功能
- 集成其他统计方法

### 引用格式

如在学术研究中使用本模块，建议引用：
```
Energy Network Analysis Team (2025). Network Structure Heterogeneity Analysis Module. 
Version 1.0. https://github.com/your-repo/energy_network
```

---

## 🔮 未来发展

### 计划功能

1. **机器学习集成**: 支持非线性交互效应检测
2. **时间异质性**: 加入时间维度的调节效应分析
3. **空间异质性**: 考虑地理距离的影响
4. **多层网络**: 扩展到多层网络结构

### 方法改进

1. **因果识别**: 引入更严格的因果推断方法
2. **稳健性检验**: 增加更多稳健性测试
3. **敏感性分析**: 参数敏感性的系统评估

---

*本模块是能源网络分析项目的重要组成部分，为理解DLI效应的复杂性和条件性提供了强有力的分析工具。*