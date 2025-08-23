# 091_geographic_heterogeneity_analysis - 地理异质性分析模块

本模块专注于地理距离在能源网络分析中的作用，通过局部投影脉冲响应(LP-IRF)方法检验OVI在不同地理位置国家中的差异化效应。

## 🎯 核心分析

### 局部投影脉冲响应分析 (LP-IRF) - 地理控制版本
**文件**: `run_lp_irf.py`  
**功能**: 检验OVI在缓冲美国供给冲击时的因果作用，控制地理距离效应  
**特色**: 加入`us_prod_shock × distance_to_us`交互项，剥离纯粹地理噪音  
**模型**: 
- 价格通道: `P_it(t+h) ~ us_prod_shock × ovi_gas + us_prod_shock × distance_to_us + Controls + α_i`
- 数量通道: `g_it(t+h) ~ us_prod_shock × ovi_gas + us_prod_shock × distance_to_us + Controls + α_i`

**输出**: 
- `figures/lp_irf_results.png` - 地理控制的脉冲响应图
- `outputs/lp_irf_analysis_report.md` - 完整分析报告  
- `outputs/lp_irf_results.csv` - 详细结果数据

### 锚定多样化假说检验 (传统面板回归)
**文件**: `models.py`  
**功能**: 检验与美国能源网络连接对进口多样化的影响，控制地理因素  
**模型**: `HHI_imports ~ NodeDLI_US + distance_to_us + Controls + α_i + λ_t`  
**输出**: `outputs/regression_table.md` - 回归结果表

## 🔧 支持模块

- `data_loader.py`: 数据加载和地理距离整合工具
- `country_standardizer.py`: 国家名称标准化工具

## 📊 数据流

```
08_variable_construction/outputs/analytical_panel.csv
        +
08_variable_construction/outputs/price_quantity_variables.csv  
        +
04_dli_analysis/complete_us_distances_cepii.json (地理距离数据)
                        ↓
            run_lp_irf.py (地理控制LP-IRF分析)
                        ↓
        outputs/lp_irf_analysis_report.md
        figures/lp_irf_results.png
```

## 🚀 快速开始

```bash
# 运行地理控制LP-IRF分析
python3 run_lp_irf.py

# 运行地理控制面板回归
python3 -c "from models import *; run_analysis()"
```

## 📈 核心特色

**地理异质性控制**: 本模块的核心创新在于系统性地控制地理距离效应，通过`us_prod_shock × distance_to_us`交互项剥离纯粹的地理噪音，从而更准确地识别OVI网络结构的独立因果效应。

## 📁 文件结构

```
091_geographic_heterogeneity_analysis/
├── README.md                    # 本文档
├── run_lp_irf.py               # 地理控制LP-IRF分析
├── models.py                   # 地理控制面板回归
├── data_loader.py              # 数据加载和地理整合
├── country_standardizer.py     # 国家标准化工具
├── geographic_heterogeneity_diagnosis.md  # 地理异质性诊断报告
├── figures/
│   └── lp_irf_results.png     # 地理控制脉冲响应图
└── outputs/
    ├── lp_irf_analysis_report.md  # 完整分析报告
    ├── lp_irf_results.csv         # LP-IRF详细结果
    └── regression_table.md        # 地理控制回归结果
```

## 🎯 核心贡献

**地理控制的必要性**: 091模块证明了在能源网络分析中控制地理距离的重要性。通过对比有无地理控制的结果，我们能够区分：
1. **纯粹地理效应**: 距离美国越远，冲击传导越弱
2. **网络结构效应**: OVI的真实因果作用，独立于地理位置

这种方法论创新为能源网络的因果推断提供了更加严谨的识别策略。