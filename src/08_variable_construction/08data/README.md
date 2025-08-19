# 08_variable_construction 数据文件说明

## 📁 文件结构

### 🎯 **最终输出文件 (用于分析)**
- `ovi_gas.csv` - 天然气OVI指数 **(39国家, 2001-2024年)**
- `ovi_oil.csv` - 石油OVI指数 **(39国家, 2001-2024年)**

### 🔧 **中间处理文件**
- `gas_consumption_processed.csv` - 天然气消费数据 (2000-2024, BCM)
- `oil_consumption_processed.csv` - 石油消费数据 (2000-2024, MTPA)
- `oil_refinery_processed.csv` - 炼油厂容量数据 (2000-2024, MTPA)
- `lng_capacity.csv` - LNG接收站容量 (BCM/年)
- `gas_pipeline_capacity.csv` - 天然气管道容量 (BCM/年)
- `oil_pipeline_capacity.csv` - 石油管道容量 (原始单位)

### 📋 **现有变量文件**
- `macro_controls.csv` - 宏观控制变量
- `node_dli_us.csv` - 节点DLI指数
- `vul_us.csv` - 脆弱性指数
- `us_prod_shock.csv` - 美国生产冲击

### 📂 **原始数据文件夹: rawdata/**
- `EI-Stats-Review-ALL-data.xlsx` - BP能源统计年鉴 (14MB)
- `GEM-GGIT-LNG-Terminals-2024-09.xlsx` - LNG终端数据
- `GEM-GGIT-Gas-Pipelines-2024-12.xlsx` - 天然气管道数据
- `GEM-GOIT-Oil-NGL-Pipelines-2025-03.xlsx` - 石油管道数据
- `gas_consumption.xlsx` - 天然气消费数据 (从大文件提取)
- `oil consumption.xlsx` - 石油消费数据 (从大文件提取)
- `oil refine.xlsx` - 炼油厂容量数据 (从大文件提取)

## 🔄 **数据处理流程**

1. **原始数据提取**: 从大Excel文件提取所需sheet
2. **基础设施容量计算**: 处理LNG、管道等设施数据
3. **消费数据处理**: 清理时间序列消费数据
4. **OVI计算**: 容量/消费量比值计算
5. **数据平衡**: 创建balanced面板数据集

## 📈 **最终数据规格**

- **国家数量**: 39个 (所有数据集交集)
- **时间范围**: 2001-2024年 (24年)
- **总观测值**: 936行/指标 (39×24)
- **数据完整性**: 100% (无缺失值)