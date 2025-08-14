# 04_政策影响分析模块

本模块专门分析美国能源独立政策对全球能源贸易网络的影响，使用事前-事后对比和统计检验方法评估政策效果。

## 📁 文件结构

```
04_policy_analysis/
├── main.py                    # 主执行脚本，完整分析流程
├── analysis.py                # 数据分析和统计检验功能
├── plotting.py                # 可视化绘图功能
├── run_analysis.py            # 简化运行接口
├── view_figures.py            # 图表查看和整理工具
├── policy_impact_summary.csv  # 对比分析结果
├── policy_impact_statistics.json # 统计检验结果
└── figures/                   # 图表文件夹
    ├── README.md              # 图表说明文档
    ├── gallery.html           # HTML图表画廊
    ├── *_dashboard.png        # 国家仪表盘图表 (18个)
    ├── *_period_comparison.png # 期间对比图表 (8个)
    └── *.png                  # 综合分析图表 (2个)
```

## 🚀 快速开始

### 方式1: 使用简化接口 (推荐)
```bash
python3 run_analysis.py
```
提供菜单式操作界面，包含：
- 运行完整分析
- 生成可视化图表
- 查看和整理图表
- 查看结果摘要

### 方式2: 直接运行分析
```bash
python3 main.py
```

### 方式3: 仅生成图表
```bash
python3 -c "from main import run_visualization_only; run_visualization_only()"
```

### 方式4: 查看图表
```bash
python3 view_figures.py
```

## 📊 分析内容

### 政策期间划分
- **政策前期 (Pre-Policy)**: 2001-2008年
- **过渡期 (Transition)**: 2009-2015年  
- **政策后期 (Post-Policy)**: 2016-2024年

### 分析指标
- **网络中心性**: 度中心性、中介中心性、PageRank中心性、特征向量中心性
- **贸易强度**: 进口强度、出口强度、总贸易强度
- **网络位置**: 入度、出度、总度数

### 统计方法
- **事前-事后对比**: 比较政策前后期的指标变化
- **配对t检验**: 检验变化的统计显著性
- **变化相关性**: 分析不同指标变化之间的关系

## 🎨 可视化图表

### 图表类型
1. **国家仪表盘** (18个): 展示单个国家多指标时间序列
2. **期间对比图** (8个): 展示各指标政策前后对比
3. **综合分析图** (2个): 相关性热力图和统计概览

### 查看方式
1. **HTML画廊**: 在浏览器中查看所有图表
2. **文件夹浏览**: 直接打开 `figures/` 文件夹
3. **命令行工具**: 使用 `view_figures.py`

## 📈 输出文件

### 数据文件
- `policy_impact_summary.csv`: 各国各指标的政策前后对比结果
- `policy_impact_statistics.json`: 统计检验和汇总结果

### 图表文件
- `figures/`: 包含所有PNG格式的可视化图表
- `figures/gallery.html`: HTML图表画廊页面
- `figures/README.md`: 图表详细说明

## 🔧 技术实现

### 核心功能模块
- **main.py**: 流程控制和参数配置
- **analysis.py**: 
  - `run_pre_post_analysis()`: 事前-事后对比分析
  - `calculate_policy_impact_statistics()`: 统计检验计算
  - `export_analysis_results()`: 结果导出
- **plotting.py**:
  - `plot_country_dashboard()`: 国家仪表盘
  - `plot_period_comparison()`: 期间对比图
  - `plot_correlation_heatmap()`: 相关性热力图
  - `create_policy_impact_dashboard()`: 完整仪表盘

### 依赖关系
- **数据来源**: `../03_metrics/all_metrics.csv`
- **Python包**: pandas, numpy, matplotlib, seaborn, scipy, pathlib

## 📝 使用示例

### 运行完整分析
```python
from main import run_full_policy_analysis

# 使用默认参数
success = run_full_policy_analysis()

# 自定义参数
success = run_full_policy_analysis(
    data_filepath="custom_data.csv",
    countries_list=["USA", "CHN", "RUS"],
    output_figures_dir="custom_figures/"
)
```

### 生成特定图表
```python
from plotting import plot_country_dashboard
import pandas as pd

df = pd.read_csv("../03_metrics/all_metrics.csv")
df['period'] = df['year'].apply(lambda x: 'pre' if x <= 2008 else 'post')

# 生成美国仪表盘
dashboard_file = plot_country_dashboard(
    df, "USA", 
    ["in_strength", "out_strength", "betweenness_centrality"],
    output_dir="figures/"
)
```

## 🎯 主要发现

政策影响分析揭示了美国能源独立政策的多方面效应：

1. **网络中心性变化**: 部分国家在全球能源网络中的地位发生显著改变
2. **贸易模式调整**: 贸易强度和方向出现结构性变化
3. **区域差异**: 不同地区国家受政策影响程度存在差异
4. **时间演化**: 政策效应在不同时期表现出动态特征

详细结果请查看生成的图表和数据文件。

---
*模块版本: v2.0*  
*最后更新: 2025-08-14*