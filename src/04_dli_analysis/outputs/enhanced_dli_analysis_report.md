# 增强版动态锁定指数(Enhanced DLI)分析报告
============================================================
生成时间: 2025-08-24 16:25:22

## 1. 数据概览
- 总记录数: 41,349
- 年份范围: 2001-2024
- 覆盖国家: 221
- 能源产品: 4

## 2. 方法论创新
- 五维度分析框架: 传统4维 + 个性化PageRank网络维度
- 严格方向性区分: 美国出口锁定 vs 美国进口被锁定
- PCA处理多重共线性: 数据驱动的权重确定
- 学术规范诊断: 相关性分析和可视化

## 3. PCA权重分析
### 美国出口锁定权重:
- continuity: 0.386
- infrastructure: 0.667
- stability: 0.131
- market_locking_power: 0.194
- pagerank_dimension: 0.593

### 美国进口被锁定权重:
- continuity: 0.420
- infrastructure: 0.624
- stability: 0.338
- market_locking_power: 0.297
- pagerank_dimension: 0.481

- 出口锁定第一主成分解释方差: 0.317
- 进口锁定第一主成分解释方差: 0.343

## 4. 核心发现
- 美国出口锁定平均水平: 0.000
- 美国进口被锁定平均水平: 0.000

## 5. 文件输出
- dli_pagerank.csv: 增强版DLI面板数据
- dli_pagerank_weights.json: PCA权重和分析参数
- dli_dimensions_correlation.png: 维度相关性热力图