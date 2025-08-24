# 个性化PageRank模块说明

## 概述

`personalized_pagerank.py` 是一个独立的网络分析模块，专门计算美国在全球能源贸易网络中的**方向性PageRank影响力**。该模块为 `04_dli_analysis` 模块提供新的网络中心性维度，用于增强DLI（Dynamic Locking Index）综合指标的计算。

## 核心功能

### 1. 出口锁定影响力 (US Export Locking Influence)
- **计算逻辑**: 以美国为唯一种子节点的个性化PageRank
- **业务含义**: 量化美国通过网络路径对其他国家的出口影响力
- **输出字段**: `ppr_us_export_influence`

### 2. 进口锁定影响力 (Import Locking Influence on US)  
- **计算逻辑**: 以每个国家为种子节点，计算其对美国的影响力
- **业务含义**: 量化各国通过网络路径对美国进口市场的潜在锁定能力
- **输出字段**: `ppr_influence_on_us`

## 主要输出

### 数据文件
- **主数据**: `outputs/personalized_pagerank_panel.csv`
  - 5,751条记录 (248个国家 × 24年)
  - 覆盖2001-2024年完整面板数据
- **摘要统计**: `outputs/personalized_pagerank_summary.json`
  - 包含描述性统计和排名信息

### 数据结构
```csv
year,country_name,ppr_us_export_influence,ppr_influence_on_us
2024,CHN,0.051974,0.025790
2024,CAN,0.016126,0.097452
...
```

## 关键发现

### 美国网络影响力演变 (2001-2024)
- **出口锁定影响力**: 从0.315 (2001) 下降至0.193 (2024)，**下降38.9%**
- **趋势解读**: 反映了全球能源格局多极化，美国相对影响力有所分散

### 2024年重要排名
**美国出口影响力最高的国家**:
1. 中国 (0.052) - 美国最大的能源出口影响对象
2. 荷兰 (0.037) - 欧洲能源贸易枢纽
3. 韩国 (0.034) - 亚太重要伙伴

**对美进口影响力最高的国家**:
1. 加拿大 (0.097) - 北美能源一体化的体现
2. 沙特 (0.046) - 传统石油供应大国
3. 荷兰 (0.039) - 欧洲转口贸易角色

## 在04模块中的整合方式

### 数据加载
```python
from pathlib import Path
import pandas as pd

def load_personalized_pagerank():
    metrics_dir = Path('../03_metrics')
    ppr_file = metrics_dir / 'outputs' / 'personalized_pagerank_panel.csv'
    return pd.read_csv(ppr_file)
```

### DLI整合
```python
# 加载个性化PageRank数据
ppr_data = load_personalized_pagerank()

# 与现有DLI数据合并
enhanced_dli = existing_dli.merge(
    ppr_data[['year', 'country_name', 'ppr_us_export_influence', 'ppr_influence_on_us']],
    on=['year', 'country_name'], 
    how='left'
)

# 在PCA或加权合成中使用新维度
dli_components = [
    'continuity', 'infrastructure', 'stability', 
    'market_locking_power',  # 原有维度
    'ppr_us_export_influence', 'ppr_influence_on_us'  # 新增维度
]
```

## 使用方法

### 基础运行
```bash
cd src/03_metrics
python3 personalized_pagerank.py
```

### 自定义参数
```bash
python3 personalized_pagerank.py \
  --networks-dir ../02_net_analysis/outputs/networks \
  --output-dir ./outputs \
  --verbose
```

### 演示和测试
```bash
python3 example_usage.py  # 运行完整演示
```

## 技术特点

1. **独立性**: 不依赖现有03_metrics模块，避免代码侵入
2. **可扩展性**: 支持命令行参数，便于自动化流程调用  
3. **标准化**: 输出格式与项目其他模块保持一致
4. **稳健性**: 完整的错误处理和日志记录机制

## 计算性能

- **处理能力**: 24年 × ~240国家 × 双向计算 = ~11,520次PageRank计算
- **执行时间**: 约50秒 (取决于网络规模)
- **内存消耗**: 适中，支持标准工作站运行

---

**版本**: v1.0  
**作者**: Energy Network Analysis Team  
**创建日期**: 2025-08-24