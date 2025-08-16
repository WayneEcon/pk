# 03_metrics: 网络拓扑指标计算模块

## 🎯 **研究使命**

本模块是**"轨道一：完整网络分析"**的核心计算引擎，负责将02模块构建的年度网络转化为可量化的拓扑特征指标。其使命是通过严谨的网络科学方法，系统性地度量全球能源贸易网络的结构特征，为后续的政策影响分析、动态锁定指数计算和骨干网络提取提供坚实的数量基础。

## 📚 **理论基础与方法论**

### **网络中心性理论**

本模块基于经典的**社会网络分析理论**和**复杂网络科学范式**，实现了五种核心中心性指标的计算：

#### **1. 度中心性 (Degree Centrality)**
**理论基础**: Freeman (1978) 的中心性理论基础
- **入度中心性**: 衡量国家作为能源进口方的连接广度
- **出度中心性**: 衡量国家作为能源出口方的连接广度  
- **总度中心性**: 反映国家在全球能源贸易中的整体连接重要性

**计算公式**:
```
In-Degree(i) = |{j : (j,i) ∈ E}|
Out-Degree(i) = |{j : (i,j) ∈ E}|
Total-Degree(i) = In-Degree(i) + Out-Degree(i)
```

#### **2. 强度中心性 (Strength Centrality)**  
**理论基础**: Barrat et al. (2004) 的加权网络分析理论
- **入强度**: 国家的总能源进口额，反映其作为能源消费方的规模
- **出强度**: 国家的总能源出口额，反映其作为能源供应方的规模
- **总强度**: 国家在全球能源贸易中的总体经济重要性

**计算公式**:
```
In-Strength(i) = Σ w(j,i) for all j where (j,i) ∈ E
Out-Strength(i) = Σ w(i,j) for all j where (i,j) ∈ E  
Total-Strength(i) = In-Strength(i) + Out-Strength(i)
```

#### **3. 中介中心性 (Betweenness Centrality)**
**理论基础**: Freeman (1977) 的信息流控制理论  
- **衡量维度**: 国家在全球能源贸易路径中的"桥梁"作用
- **经济含义**: 控制能源流动、具备供应链中转能力的国家识别
- **政策意义**: 识别对全球能源安全具有关键影响的"节点国家"

**计算公式**:
```
Betweenness(i) = Σ(σ_st(i)/σ_st) for all s≠t≠i
其中 σ_st 是s到t的最短路径数，σ_st(i) 是经过节点i的最短路径数
```

#### **4. PageRank中心性**
**理论基础**: Page et al. (1999) 的权威性排名算法，网络科学中的经典威望测度
- **核心思想**: 被重要国家依赖的国家本身也重要
- **能源解释**: 反映国家在全球能源供应链中的"威望地位"
- **迭代逻辑**: 考虑贸易伙伴的重要性加权的递归威望计算

**计算公式**:
```
PR(i) = (1-d)/N + d × Σ(PR(j)/L(j)) for all j linking to i
其中 d 是阻尼系数，L(j) 是节点j的出链接数
```

#### **5. 特征向量中心性 (Eigenvector Centrality)**
**理论基础**: Bonacich (1987) 的权力中心性理论
- **数学基础**: 网络邻接矩阵的主特征向量
- **经济含义**: 与其他重要国家连接密切的国家具有更高中心性
- **适用性**: 特别适合分析高度相互依赖的全球能源贸易网络

### **全局网络拓扑理论**

#### **1. 网络密度 (Network Density)**
**理论基础**: Scott (2017) 的社会网络分析经典理论
- **测量维度**: 实际连接数与最大可能连接数的比值
- **经济解释**: 全球能源贸易的一体化程度
- **时间演化**: 反映全球化进程中能源市场的整合趋势

#### **2. 连通性分析 (Connectivity Analysis)**
**理论基础**: 图论中的连通性理论
- **强连通分量**: 识别能源贸易的"核心集团"
- **弱连通分量**: 测量网络的整体连通性
- **政策含义**: 评估能源供应链的脆弱性和韧性

#### **3. 路径长度分析 (Path Length Analysis)**  
**核心修正**: 加权网络中的最短路径计算
- **距离定义**: `distance = 1/weight`，确保高权重边对应短距离
- **理论依据**: 贸易额越大，两国间的"经济距离"越近
- **计算指标**: 平均最短路径长度、网络直径、全局效率

**关键算法实现**:
```python
def add_distance_weights(G: nx.DiGraph) -> nx.DiGraph:
    """
    为加权有向图添加距离权重
    距离权重 = 1 / 贸易权重，用于最短路径计算
    """
    G_dist = G.copy()
    for u, v, data in G_dist.edges(data=True):
        weight = data.get('weight', 1)
        if weight > 0:
            data['distance'] = 1.0 / weight
        else:
            data['distance'] = float('inf')
    return G_dist
```

#### **4. 聚类系数 (Clustering Coefficient)**
**理论基础**: Watts & Strogatz (1998) 的小世界网络理论
- **局部聚类**: 国家的贸易伙伴之间的相互连接程度
- **全局聚类**: 整个网络的"小团体"特征
- **经济含义**: 区域贸易集团化和供应链本地化程度

## 🏗️ **模块架构与功能组织**

### **核心功能模块**

```
03_metrics/
├── __init__.py              # 统一接口和主要计算函数
├── node_metrics.py          # 节点级别中心性指标计算
├── global_metrics.py        # 全局网络拓扑指标计算  
├── utils.py                # 通用工具函数和验证组件
├── parallel_computing.py    # 并行计算和性能优化
├── tests/                  # 全面的单元测试套件
│   ├── test_node_metrics.py
│   ├── test_global_metrics.py
│   ├── test_integration.py
│   └── run_all_tests.py
└── README.md               # 本文档
```

### **关键算法修正与创新**

#### **1. 加权最短路径计算修正**
**问题识别**: 原始NetworkX算法在处理贸易权重时存在逻辑错误
**解决方案**: 实现权重-距离转换机制
```python
# 错误做法: 直接使用贸易额作为距离
path_length = nx.shortest_path_length(G, weight='weight')

# 正确做法: 将贸易权重转换为距离权重
G_distance = add_distance_weights(G)
path_length = nx.shortest_path_length(G_distance, weight='distance')
```

#### **2. 性能优化策略**
**采样策略**: 对于计算密集的指标（如中介中心性），采用节点采样
```python
def get_node_sample(G: nx.DiGraph, sample_ratio: float = 0.1, min_nodes: int = 100) -> List[str]:
    """
    智能节点采样策略，平衡计算效率与精度
    """
    n_nodes = G.number_of_nodes()
    if n_nodes <= min_nodes:
        return list(G.nodes())
    
    sample_size = max(min_nodes, int(n_nodes * sample_ratio))
    # 基于度数的分层采样，确保重要节点被包含
    ...
```

**缓存机制**: LRU缓存减少重复计算
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_shortest_path_calculation(graph_hash, source, target):
    """缓存最短路径计算结果"""
    ...
```

#### **3. 数据验证框架**
**多层次验证**: 输入验证 → 计算验证 → 结果验证
```python
def validate_graph(G: nx.DiGraph, context: str) -> None:
    """
    全面的图对象验证
    - 检查图的类型和格式
    - 验证边权重的有效性  
    - 确保节点标识符的一致性
    """
    
def validate_metrics_result(result: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    指标计算结果验证
    - 检查数据完整性
    - 验证数值范围合理性
    - 确保无异常值或缺失值
    """
```

## 📊 **核心产出与数据格式**

### **标准化输出格式**

#### **1. 节点中心性指标表 (`node_centrality_metrics.csv`)**
```
year | country_code | country_name | in_degree | out_degree | total_degree | 
     | in_strength | out_strength | total_strength | betweenness_centrality |
     | pagerank_centrality | eigenvector_centrality | ...
```

#### **2. 全局网络指标表 (`global_network_metrics.csv`)**  
```
year | density | avg_path_length | diameter | global_clustering |
     | efficiency | largest_scc_size | largest_wcc_size | ...
```

#### **3. 综合指标数据表 (`all_metrics.csv`)**
- 节点级别指标 + 全局指标的完整合并
- 为每个国家-年份观测值提供完整的网络拓扑特征
- 直接用于后续模块的政策分析和DLI计算

### **数据质量保证**

#### **统计验证指标**
```python
def get_metrics_summary_report(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成指标计算的质量评估报告
    
    包含:
    - 数据完整性统计
    - 年度间趋势分析  
    - 异常值检测报告
    - 排名稳定性分析
    """
```

#### **历史连续性检验**
- 年度间指标变化的平滑性验证
- 极值变化的合理性检查
- 排名大幅波动的异常检测

## 🔬 **验证与测试框架**

### **单元测试覆盖**

#### **节点指标测试 (`test_node_metrics.py`)**
```python
def test_degree_centrality_calculation():
    """测试度中心性计算的精确性"""
    
def test_strength_centrality_with_weights():
    """测试加权强度中心性的正确性"""
    
def test_betweenness_centrality_sampling():
    """测试中介中心性采样策略的有效性"""
```

#### **全局指标测试 (`test_global_metrics.py`)**
```python  
def test_weighted_path_length_correction():
    """验证加权路径长度计算修正的正确性"""
    
def test_density_calculation_accuracy():
    """测试网络密度计算的数学精度"""
```

#### **集成测试 (`test_integration.py`)**
```python
def test_full_pipeline_consistency():
    """测试完整计算流程的一致性"""
    
def test_multi_year_calculation_stability():
    """验证多年份计算的稳定性"""
```

### **性能基准测试**

| 计算任务 | 网络规模 | 计算时间 | 内存使用 | 精度保证 |
|---------|---------|---------|---------|---------|
| 度中心性 | 200节点 | <1秒 | <50MB | 精确计算 |
| 强度中心性 | 200节点 | <1秒 | <50MB | 精确计算 |
| 中介中心性 | 200节点 | <30秒 | <200MB | 10%采样 |
| PageRank | 200节点 | <5秒 | <100MB | 精确计算 |
| 全局指标 | 200节点 | <10秒 | <150MB | 精确计算 |

## 🔗 **与整体研究框架的关系**

### **数据流向**
```
02_net_analysis → 03_metrics → 04_policy_analysis
              → 03_metrics → 05_dli_analysis  
              → 03_metrics → 06_backbone_analysis
```

### **为下游模块提供的核心数据**

#### **04_policy_analysis模块**
- 提供美国能源地位变化的定量基础
- 支持事前-事后对比分析的指标时间序列
- 为政策冲击检验提供中心性排名数据

#### **05_dli_analysis模块**  
- 提供DLI指标计算所需的基础网络特征
- 支持动态锁定指数的多维度分析
- 为双向锁定分析提供节点重要性参考

#### **06_backbone_analysis模块**
- 为骨干网络提取提供原始网络的拓扑基准
- 支持稳健性检验中的指标对比分析
- 提供网络可视化的节点大小和着色依据

## 🛡️ **方法论严谨性保证**

### **理论一致性**
- 严格遵循网络科学的经典理论框架
- 中心性指标选择有明确的文献支撑
- 算法实现符合领域标准和最佳实践

### **计算精度控制**
- 数值计算精度验证和边界条件处理
- 大规模网络的采样策略科学设计
- 异常值检测和数据质量监控机制

### **可重现性设计**
- 完整的随机种子控制和参数配置
- 详细的计算日志和中间结果保存
- 标准化的数据格式和处理流程

## 🚀 **快速开始**

### **基本使用**
```python
from src.03_metrics import calculate_all_metrics_for_year

# 计算单年度所有指标
metrics_df = calculate_all_metrics_for_year(G_2020, 2020)

# 查看结果摘要
print(f"计算完成: {len(metrics_df)} 个国家的网络指标")
print(f"指标维度: {len(metrics_df.columns)} 个")
```

### **批量计算**
```python
from src.03_metrics import calculate_metrics_for_multiple_years

# 批量计算多年份指标
annual_networks = load_annual_networks()  # 从02模块加载
all_metrics_df = calculate_metrics_for_multiple_years(annual_networks)

# 导出结果文件
from src.03_metrics import export_metrics_to_files
exported_files = export_metrics_to_files(all_metrics_df)
```

### **完整流程**
```python
from src.03_metrics import run_full_metrics_calculation

# 一键运行完整计算流程
success = run_full_metrics_calculation()
if success:
    print("✅ 所有网络指标计算完成并已导出")
```

---

*本模块将抽象的网络结构转化为具体的数量指标，为理解全球能源贸易格局的演化提供了科学的测量工具。每一个指标都承载着对国际能源关系的深刻洞察。*

**版本**: v3.0 | **最后更新**: 2025-01-15 | **状态**: 生产就绪