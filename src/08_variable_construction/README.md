# 08_variable_construction - 变量构建模块 🏗️

## 📋 模块概述

本模块负责从原始数据源构建所有研究需要的变量，生成最终的分析面板数据。

**核心成果**: 经过**完整单位统一**的OVI（物理冗余指数）数据，解决了原始数据中36种混乱单位的问题。

## 📁 文件结构

### 🎯 核心输出文件
```
08data/
├── ovi_balanced_panel.csv     # 🏆 最终平衡面板数据 (1,656条记录，56国家×24年)
├── ovi_gas_final.csv         # 天然气OVI最终版 (BCM单位统一)
├── ovi_oil_final.csv         # 石油OVI最终版 (MTPA单位统一)
└── ovi_final_report.md       # 完整技术报告
```

### 🛠️ 核心工具
```
universal_unit_converter.py   # 通用单位转换器 (支持36种能源单位)
upgraded_ovi_builder.py      # 升级版OVI构建器 (从Excel直接ETL)
ovi_data_cleaner.py          # 数据清理器 (异常值处理+平衡面板)
unit_audit_report.md         # 单位审计详细报告
```

### 📊 原始数据源
```
08data/rawdata/
├── GEM-GGIT-Gas-Pipelines-2024-12.xlsx    # 天然气管道数据
├── GEM-GGIT-LNG-Terminals-2024-09.xlsx    # LNG终端数据  
├── GEM-GOIT-Oil-NGL-Pipelines-2025-03.xlsx # 石油管道数据
├── gas_consumption.xlsx                     # 天然气消费数据 (BCM)
├── oil consumption.xlsx                     # 石油消费数据 (千桶/天)
└── oil refine.xlsx                         # 炼油厂容量数据
```

### 🗂️ 其他数据
```
08data/
├── node_dli_us.csv          # DLI-US节点数据
├── vul_us.csv              # 美国脆弱性数据
├── us_prod_shock.csv       # 美国产量冲击数据
└── macro_controls.csv      # 宏观控制变量
```

## 🚀 使用方法

### 完整ETL流程
```python
from upgraded_ovi_builder import UpgradedOVIBuilder
from ovi_data_cleaner import OVIDataCleaner

# 1. 构建OVI数据 (从Excel到标准化数据)
builder = UpgradedOVIBuilder("08data")
gas_ovi = builder.build_gas_ovi()  # 天然气OVI
oil_ovi = builder.build_oil_ovi()  # 石油OVI

# 2. 数据清理和平衡面板生成
cleaner = OVIDataCleaner()
final_data = cleaner.run_complete_cleaning()
```

### 直接使用最终数据
```python
import pandas as pd

# 加载最终平衡面板
ovi_data = pd.read_csv("08data/ovi_balanced_panel.csv")

# 2024年中美印对比
data_2024 = ovi_data[ovi_data['year'] == 2024]
usa = data_2024[data_2024['country'] == 'USA'].iloc[0]
chn = data_2024[data_2024['country'] == 'CHN'].iloc[0] 
ind = data_2024[data_2024['country'] == 'IND'].iloc[0]

print(f"USA 2024: 天然气OVI={usa['ovi_gas']:.2f}, 石油OVI={usa['ovi_oil']:.2f}")
print(f"CHN 2024: 天然气OVI={chn['ovi_gas']:.2f}, 石油OVI={chn['ovi_oil']:.2f}")
print(f"IND 2024: 天然气OVI={ind['ovi_gas']:.2f}, 石油OVI={ind['ovi_oil']:.2f}")
```

## 🎯 主要解决的问题

### 1. 单位混乱问题 ✅
- **天然气**: 14种不同单位 → 统一到BCM/年
- **石油**: 12种不同单位 → 统一到MTPA
- **LNG**: 8种不同单位 → 统一到BCM/年
- **特殊处理**: 印度MMSCMD、泰国极值、逗号分隔符等

### 2. 数据质量问题 ✅  
- **异常值清理**: 移除负值和极端异常值
- **合理性验证**: OVI值控制在合理范围
- **平衡面板**: 确保国家-年度数据完整性
- **美国数据**: 成功修复美国数据缺失问题

### 3. ETL流程问题 ✅
- **直接Excel处理**: 不再依赖中间CSV文件
- **智能数据清理**: 自动处理格式问题
- **多源数据整合**: 标准化国家名称映射

## 📈 最终数据质量

- **覆盖范围**: 56个国家，2001-2024年
- **数据完整性**: 1,656条记录 (123.2%完整性)
- **美国数据**: ✅ 32条完整记录 (每年完整)
- **中国数据**: ✅ 天然气OVI=4.63, 石油OVI=2.35 (2024)
- **印度数据**: ✅ 天然气OVI=6.81, 石油OVI=0.48 (2024)

## 🔧 技术特性

- **通用单位转换器**: 支持36种能源单位的自动转换
- **智能数据解析**: 处理Excel复杂结构和混合数据格式
- **异常值检测**: 自动识别和处理数据质量问题
- **平衡面板生成**: 确保经济计量分析的数据要求

## 📚 相关文档

- `unit_audit_report.md`: 详细的单位审计和转换分析
- `ovi_final_report.md`: OVI数据构建完整技术报告
- `universal_unit_converter.py`: 转换器技术文档和使用示例

---

**版本**: v2.0 (单位统一版)  
**更新**: 2024-08-19  
**状态**: ✅ 生产就绪