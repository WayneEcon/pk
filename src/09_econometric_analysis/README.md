# 09_econometric_analysis - 计量经济学分析模块

## 🎯 模块概述

本模块是 `energy_network` 项目的计量经济学分析组件，专门设计用于处理能源网络的经济学建模与统计推断。核心特色是**空数据兼容设计** - 即使在08模块数据尚未完全准备的情况下，也能完整运行并生成分析框架。

## 🏗️ 设计理念

> **"只搭框架，不跑实数"** - 优雅处理数据缺失，构建可复现的分析流程

### 核心原则
- ✅ **健壮性**: 所有函数都能处理空数据或缺失变量
- ✅ **可复现性**: 完整的配置驱动和日志记录
- ✅ **模块化**: 五大组件独立工作，便于测试和维护  
- ✅ **学术标准**: 严格的计量经济学规范和报告格式

## 📊 核心功能

### 三大研究模型
1. **模型1: DLI-脆弱性关联检验** (第3章)
   - 方法: 双向固定效应面板模型 
   - 公式: `vul_us ~ node_dli_us + Controls + FE(country,year)`

2. **模型2: OVI对DLI的因果效应** (补充分析)
   - 方法: 滞后变量面板模型
   - 公式: `node_dli_us ~ ovi(t-1) + Controls + FE(country,year)`

3. **模型3: 局部投影因果验证** (第4章)  
   - 方法: Jordà局部投影法
   - 公式: `Δvul_us(t+h) ~ us_prod_shock(t) * ovi(t-1) + Controls`

## 🔧 模块架构

```
09_econometric_analysis/
├── main.py              # 🚀 主流水线入口
├── config.py            # ⚙️ 配置管理
├── data_loader.py       # 📊 健壮数据加载
├── models.py            # 🔬 计量模型框架
├── reporting.py         # 📝 报告生成系统  
├── visualization.py     # 🎨 可视化引擎
├── outputs/             # 📁 分析输出
│   ├── regression_results.csv
│   ├── analysis_report.md
│   └── model_diagnostics.json
└── figures/             # 📈 图表输出
    ├── coefficient_comparison.png
    ├── diagnostic_plots.png
    ├── impulse_response.png
    └── robustness_charts.png
```

## 🚀 快速开始

### 基本用法
```python
from src.econometric_analysis import EconometricAnalysisPipeline

# 创建分析流水线
pipeline = EconometricAnalysisPipeline()

# 运行完整分析
results = pipeline.run_full_pipeline()

# 查看结果
print(f"流水线状态: {results['status']}")
print(f"成功模型数: {results['model_summary']['successful_models']}")
```

### 命令行运行
```bash
# 进入模块目录
cd src/09_econometric_analysis

# 运行完整流水线
python main.py

# 运行诊断检查
python main.py --diagnostic
```

### 独立组件使用
```python
# 单独使用数据加载器
from src.econometric_analysis import DataLoader
loader = DataLoader()
data = loader.load_analytical_panel()

# 单独运行某个模型
from src.econometric_analysis import run_single_model
result = run_single_model('model_1_dli_vul_association', data)

# 单独生成报告
from src.econometric_analysis import generate_reports
files = generate_reports(model_results, data_summary)
```

## 📋 依赖要求

### 核心依赖 (必需)
```bash
pip install pandas numpy pathlib
```

### 计量分析依赖 (推荐)
```bash  
pip install statsmodels linearmodels
```

### 可视化依赖 (推荐)
```bash
pip install matplotlib seaborn
```

**注意**: 即使没有安装可选依赖，模块仍能运行并生成占位符输出。

## 🔍 空数据处理机制

本模块的独特之处在于**空数据兼容设计**:

### 数据加载层
- ✅ 文件不存在 → 返回空DataFrame但保持正确列结构
- ✅ 文件为空 → 创建标准化的空框架  
- ✅ 关键变量缺失 → 智能填充NaN值

### 模型估计层  
- ✅ 数据不足 → 返回标准化的失败结果字典
- ✅ 变量全为空 → 跳过估计，记录详细原因
- ✅ 依赖库缺失 → 优雅降级，生成占位符结果

### 输出生成层
- ✅ 无结果数据 → 生成"等待数据"的占位符报告
- ✅ 部分失败 → 混合展示成功和失败的模型
- ✅ 图表生成失败 → 创建带说明的占位符图像

## 📈 输出文件说明

### 分析报告
- `analysis_report.md`: 完整的Markdown格式分析报告
- `regression_results.csv`: 机器可读的回归结果表
- `model_diagnostics.json`: 详细的模型诊断信息

### 可视化图表
- `coefficient_comparison.png`: 跨模型系数对比图
- `diagnostic_plots.png`: 模型诊断图集合
- `impulse_response.png`: 局部投影脉冲响应图  
- `robustness_charts.png`: 稳健性检验图表

## ⚙️ 配置选项

主要配置在 `config.py` 中管理:

```python
from src.econometric_analysis import config

# 查看模型配置
print(config.analysis.RESEARCH_MODELS)

# 修改估计设置
config.models.ESTIMATION_SETTINGS['robust'] = True
config.models.ESTIMATION_SETTINGS['cluster_var'] = 'country'

# 自定义输出路径
config.output.OUTPUT_PATHS['regression_results'] = Path('/custom/path/results.csv')
```

## 🔧 故障排除

### 常见问题

**Q: 为什么所有模型都显示"数据不可用"？**
A: 这是正常现象。在08模块数据构建完成前，本模块会运行"空转"模式，展示完整的分析框架。

**Q: 如何确认模块正常工作？**  
A: 运行 `python main.py --diagnostic` 检查所有组件状态。

**Q: 可视化图表为什么是占位符？**
A: 安装 `matplotlib` 和 `seaborn` 库即可生成真实图表：
```bash
pip install matplotlib seaborn
```

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行流水线，查看详细日志
pipeline = EconometricAnalysisPipeline()
results = pipeline.run_full_pipeline()
```

## 🎓 学术使用指南

### 引用格式
```
Energy Network Analysis Team. (2025). 
09_econometric_analysis: Econometric Analysis Framework for Energy Networks. 
Version 1.0. GitHub Repository.
```

### 方法论说明
本模块实现的计量方法基于以下学术标准:
- 面板数据分析: Baltagi (2013) 
- 局部投影法: Jordà (2005)
- 稳健标准误: Cameron & Miller (2015)

## 🤝 贡献指南

欢迎贡献！请遵循以下原则:
1. 保持空数据兼容性 - 所有新函数都必须处理空输入
2. 添加详细日志 - 用户需要了解每个步骤的执行状态
3. 编写测试 - 特别是边界条件和异常情况
4. 更新文档 - 确保README和docstring同步

## 📞 支持与反馈

- 📧 技术问题: Energy Network Analysis Team
- 🐛 Bug报告: 请提供详细的日志输出
- 💡 功能建议: 欢迎提出改进意见

---

*本模块是 energy_network 项目的核心组件，专注于提供可靠、可复现的计量经济学分析能力。*