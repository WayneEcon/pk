# 因果验证分析模块 v1.0
## 05_causal_validation - Causal Validation Analysis Module

---

## 🎯 模块目标

本模块实现从**描述性分析**向**因果推断**的关键跃进，通过严谨的计量经济学方法，检验DLI（动态锁定指数）与网络韧性之间是否存在统计上显著的因果关系。

### 战略意义
- 提供学术级因果推断证据，满足顶级期刊发表标准
- 为政策制定提供科学依据，支持能源安全战略决策  
- 建立能源网络韧性的量化评估框架

---

## 🏗️ 核心架构

### 核心产出
1. **网络韧性数据库** (`network_resilience.csv`)
   - 2001-2024年面板数据
   - 每个国家的年度韧性得分
   - 双轨韧性测量：拓扑抗毁性 + 供应缺口吸收率

2. **因果验证报告** (`causal_validation_report.md` & `.csv`)
   - 双向固定效应面板模型结果
   - 工具变量法稳健性检验
   - 学术标准回归表格和统计检验

### 模块结构
```
05_causal_validation/
├── main.py                    # 主分析流程
├── resilience_calculator.py   # 网络韧性计算器
├── causal_model.py            # 因果推断模型
├── outputs/                   # 分析结果
│   ├── network_resilience.csv      # 韧性数据库
│   ├── causal_validation_report.md # 学术报告  
│   ├── regression_results.csv      # 回归结果表
│   └── causal_validation_results.json # 原始结果
└── README.md                  # 本文档
```

---

## 🔬 方法论框架

### 双轨韧性测量原则

为确保结论稳健性，网络韧性量化并行采用两种理论基础：

#### 1. 拓扑抗毁性 (Topological Resilience)
- **理论基础**: 复杂网络理论，连通性损失测量
- **方法**: 模拟攻击（移除Top-K节点），测量连通分量规模下降
- **攻击策略**: 度中心性攻击、介数中心性攻击、随机攻击
- **评估指标**: 网络结构稳定性、连通韧性、位置稳定性

#### 2. 供应缺口吸收率 (Supply Gap Absorption Rate)  
- **理论基础**: 经济韧性理论，适应性供应替代
- **方法**: 模拟主要供应商中断，测量替代供应获取能力
- **评估维度**: 供应多样化、网络深度、替代路径数量
- **实际意义**: 真实供应链中断情况下的经济适应能力

### 因果识别策略

#### 基准模型：双向固定效应面板模型
```
Resilience_it = β₀ + β₁·DLI_it + γ·Controls_it + αᵢ + λₜ + εᵢₜ
```

**控制变量设计**：
- `αᵢ`: 国家固定效应 - 控制不随时间改变的国家异质性
- `λₜ`: 时间固定效应 - 控制不随国家改变的年份宏观冲击  
- `β₁`: **DLI对韧性的因果效应**（核心估计参数）

#### 内生性处理：工具变量法 (2SLS)
**第一阶段**: `DLI_it = α + γ·IV_it + δ·Controls_it + uᵢₜ`  
**第二阶段**: `Resilience_it = β + θ·DLI_hat_it + λ·Controls_it + εᵢₜ`

**工具变量构建**：
- 历史基础设施存量（1990年管道、港口、炼厂容量）
- 地理距离加权的其他国家DLI冲击
- DLI的深度滞后项（减少反向因果）

#### 稳健性检验
- **聚类标准误**: 国家层面聚类，控制序列相关性
- **子样本分析**: 2008年金融危机前后结构稳定性检验
- **异常值检验**: 识别和排除极端观测的影响
- **滞后效应**: 检验DLI对韧性的动态影响路径

---

## 📊 学术标准要求

### 统计显著性标准
- **Spearman相关系数** > 0.7 (一致性验证)
- **核心发现稳定性** > 80% (跨不同α值)
- **统计显著性** p < 0.05 (美国地位变化)
- **政策效应一致性** > 75% (2016年后跨算法)

### 诊断检验要求
- **弱工具变量检验**: 第一阶段F统计量 > 10
- **过度识别检验**: Sargan检验通过 (如适用)
- **异方差检验**: White检验和稳健标准误
- **序列相关检验**: Durbin-Watson统计量

---

## 🚀 使用指南

### 快速开始

#### 演示模式（推荐入门）
```bash
cd 05_causal_validation
python main.py --demo
```

#### 完整分析模式
```bash
# 使用真实数据
python main.py --networks-dir ../02_data_processing/outputs \
                --dli-file ../04_dli_analysis/outputs/dli_results.csv

# 指定分析参数
python main.py --demo --years 2010 2015 2020 \
                --countries USA CHN RUS DEU JPN
```

#### 高级配置
```bash
# 详细输出模式
python main.py --demo --verbose

# 自定义输出目录  
python main.py --demo --output-dir my_causal_analysis
```

### 程序化调用

```python
from causal_validation import CausalValidationPipeline

# 初始化分析管道
pipeline = CausalValidationPipeline(output_dir="outputs")

# 运行完整分析
results = pipeline.run_full_pipeline(
    networks_dir="../02_networks",
    dli_file="../05_dli/dli.csv",
    years=[2010, 2015, 2020],
    countries=['USA', 'CHN', 'RUS']
)

# 查看分析结果
print(f"因果证据强度: {results['causal_results']['overall_assessment']['causal_evidence_strength']}")
```

---

## 📈 输出解读

### network_resilience.csv 字段说明

| 字段名 | 含义 | 取值范围 |
|--------|------|----------|
| `year` | 年份 | 2001-2024 |
| `country` | 国家代码 | ISO 3字符代码 |
| `topological_resilience_avg` | 平均拓扑抗毁性 | 0-1 (越高越韧性) |
| `topological_resilience_degree` | 度攻击韧性 | 0-1 |
| `supply_absorption_rate` | 供应缺口吸收率 | 0-1 |
| `supply_diversification_index` | 供应多样化指数 | 0-1 |
| `comprehensive_resilience` | **综合韧性指数** | 0-1 |

### 回归结果解读

#### 系数解释
- **正系数**: DLI增加（锁定程度上升）→ 韧性下降
- **负系数**: DLI增加 → 韧性提升（反直觉，需检查）
- **系数大小**: 表示DLI变化1单位对韧性的边际影响

#### 显著性判断
- `***`: p < 0.01 (高度显著)
- `**`: p < 0.05 (显著) 
- `*`: p < 0.10 (边际显著)
- 无标记: p ≥ 0.10 (不显著)

#### 模型诊断
- **R²**: 模型解释力（越高越好，通常>0.3）
- **第一阶段F**: 工具变量强度（>10为强工具变量）
- **观测数**: 有效样本规模（越大越可信）

---

## ⚙️ 依赖环境

### 核心依赖
```
python >= 3.8
networkx >= 2.8     # 网络分析
pandas >= 1.5       # 数据处理
numpy >= 1.21       # 数值计算
scipy >= 1.9        # 统计函数
matplotlib >= 3.5   # 基础绘图
seaborn >= 0.11     # 统计可视化
scikit-learn >= 1.0 # 机器学习算法
tqdm >= 4.0         # 进度条
```

### 专业计量库（可选但推荐）
```
statsmodels >= 0.13  # 计量经济学模型
linearmodels >= 4.0  # 面板数据和工具变量
```

**安装命令**：
```bash
pip install networkx pandas numpy scipy matplotlib seaborn scikit-learn tqdm
pip install statsmodels linearmodels  # 专业计量库
```

---

## 🔍 质量保证

### 测试覆盖
- **单元测试**: 每个核心函数的正确性验证
- **集成测试**: 完整分析流程的端到端验证  
- **回归测试**: 确保代码更新不破坏现有功能
- **数据验证**: 输入数据格式和内容的自动检查

### 代码规范
- **PEP 8**: Python代码风格标准
- **类型提示**: 函数参数和返回值的类型注释
- **文档字符串**: 详细的函数和类说明
- **异常处理**: 优雅的错误处理和用户提示

### 可重现性
- **固定随机种子**: 确保随机过程的可重复性
- **版本控制**: 依赖库版本的精确控制
- **数据溯源**: 完整的数据处理流程记录
- **开源代码**: 所有分析代码公开可审计

---

## 📚 学术应用

### 适用场景
- **顶级期刊发表**: Nature, Science, PNAS等顶级期刊论文
- **政策制定支持**: 国家能源安全战略制定的科学依据
- **监管决策**: 能源监管机构的量化分析工具
- **风险评估**: 能源企业和投资机构的风险管理

### 引用格式
```bibtex
@software{energy_network_causal_validation,
  title={Energy Network Causal Validation Analysis Module v1.0},
  author={Energy Network Analysis Team},
  year={2025},
  url={https://github.com/energy-network-analysis/causal-validation},
  note={Econometric Causal Inference Edition}
}
```

---

## 🆘 技术支持

### 常见问题

**Q: 为什么选择双轨韧性测量？**
A: 单一指标可能存在测量偏差。拓扑韧性关注网络结构稳定性，供应吸收率关注经济适应能力，两者结合提供更全面的韧性评估。

**Q: 工具变量的有效性如何保证？**
A: 通过第一阶段F检验（>10）和过度识别检验确保工具变量强度和外生性。历史基础设施数据在时间上先于当期DLI，满足外生性要求。

**Q: 如何处理数据缺失问题？**
A: 采用列表删除法处理缺失值，确保分析样本的完整性。对于关键变量缺失严重的情况，会在报告中明确说明限制。

### 联系方式
- **技术支持**: Energy Network Analysis Team
- **Bug报告**: 通过GitHub Issues提交
- **学术合作**: 联系项目主要研究者

---

## 📝 更新日志

### v1.0.0 (2025-08-15)
- ✅ 初始版本发布
- ✅ 实现双轨韧性测量框架
- ✅ 完成双向固定效应面板模型
- ✅ 集成工具变量法和稳健性检验
- ✅ 生成学术标准报告和CSV结果
- ✅ 提供完整的命令行接口和程序化API

---

*版本: v1.0 (Econometric Causal Inference Edition)*  
*最后更新: 2025-08-15*  
*文档语言: 中英双语*