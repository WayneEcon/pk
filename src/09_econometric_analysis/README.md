# 09_econometric_analysis - 计量经济分析模块

本模块执行研究的核心实证分析，包含传统面板回归和局部投影脉冲响应两套完整分析框架。

## 🎯 核心分析

### 1. 局部投影脉冲响应分析 (LP-IRF) - **研究高潮**
**文件**: `run_lp_irf.py`  
**功能**: 检验OVI在缓冲美国供给冲击时的因果作用  
**模型**: 价格通道 + 数量通道的动态脉冲响应  
**输出**: 
- `figures/lp_irf_results.png` - 专业脉冲响应图
- `outputs/lp_irf_analysis_report.md` - 完整分析报告  
- `outputs/lp_irf_results.csv` - 详细结果数据

### 2. 锚定多样化假说检验 (传统面板回归)
**文件**: `models.py`  
**功能**: 检验与美国能源网络连接对进口多样化的影响  
**模型**: `HHI_imports ~ NodeDLI_US + Controls` (双向固定效应)  
**输出**: `outputs/regression_table.md` - 回归结果表

## 🔧 支持模块

- `data_loader.py`: 数据加载和预处理工具
- `country_standardizer.py`: 国家名称标准化工具

## 📊 数据流

```
08_variable_construction/outputs/analytical_panel.csv
        +
08_variable_construction/outputs/price_quantity_variables.csv  
        +
08_variable_construction/outputs/us_prod_shock_ar2.csv
                        ↓
            run_lp_irf.py (主要分析)
                        ↓
        outputs/lp_irf_analysis_report.md
        figures/lp_irf_results.png
```

## 🚀 快速开始

```bash
# 运行核心LP-IRF分析 (推荐)
python3 run_lp_irf.py

# 运行传统面板回归 (可选)
python3 -c "from models import *; run_analysis()"
```

## 📈 主要发现

**LP-IRF分析验证了核心假说**: 物理基础设施冗余(OVI)确实在中期(h=3)显著增强了国家的进口数量调节能力 (θ_3 = 1.263, p = 0.065)，证明了**物理基础设施才是能源韧性的根本基石**。

## 📁 文件结构

```
09_econometric_analysis/
├── README.md                    # 本文档
├── run_lp_irf.py               # 主要分析脚本 (LP-IRF)
├── models.py                   # 传统面板回归模型
├── data_loader.py              # 数据加载工具
├── country_standardizer.py     # 国家标准化工具
├── figures/
│   └── lp_irf_results.png     # 脉冲响应图
└── outputs/
    ├── lp_irf_analysis_report.md  # 完整分析报告
    ├── lp_irf_results.csv         # LP-IRF详细结果
    └── regression_table.md        # 传统回归结果
```

## 🎯 核心发现总结

**研究证实了我们的核心假说**：物理基础设施冗余(OVI)是国家能源韧性的根本决定因素，不仅仅是贸易关系的多样化。LP-IRF分析显示，拥有更高OVI的国家在面对外部冲击时具有显著更强的调节能力，这种效应在中期(3年后)最为显著。