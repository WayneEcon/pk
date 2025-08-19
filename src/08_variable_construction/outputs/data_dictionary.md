# 分析面板数据字典
## Analytical Panel Data Dictionary

**生成时间**: 2025-08-19 20:17:36  
**模块**: 08_variable_construction v1.0  
**总行数**: 500  
**总列数**: 34  
**年份范围**: 2000-2024  
**国家数量**: 20

---

## 变量详细说明

### 基础标识变量
- **year**: 年份 (整数)
- **country**: 国家ISO3代码 (字符串)

### 宏观经济控制变量 (来源：World Bank WDI API)
- **gdp_current_usd**: GDP，现价美元 (NY.GDP.MKTP.CD)
- **population_total**: 总人口数 (SP.POP.TOTL)
- **trade_openness_gdp_pct**: 贸易开放度，占GDP百分比 (NE.TRD.GNFS.ZS)
- **log_gdp**: GDP的自然对数
- **log_population**: 人口的自然对数

### 核心研究变量 (本模块构建)
- **node_dli_us**: Node-DLI_US: 美国锚定动态锁定指数，基于04_dli_analysis的边级DLI聚合
- **vul_us**: Vul_US: 美国锚定脆弱性指数，基于进口份额×HHI指数
- **ovi_gas**: OVI_gas: 天然气物理冗余指数 (主指标)，基于LNG接收站和管道容量
- **us_prod_shock**: US_ProdShock: 美国页岩油气产量冲击，HP滤波后的周期成分

### 网络拓扑指标 (来源：03_metrics)
- **betweenness_centrality**: 介数中心性，衡量节点在网络中的桥梁作用
- **eigenvector_centrality**: 特征向量中心性，考虑邻居重要性的中心性

---

## 数据质量摘要

### 缺失值统计

| 变量名 | 缺失值数量 | 缺失率(%) |
|--------|------------|----------|
| gdp_current_usd | 1 | 0.2% |
| trade_openness_gdp_pct | 2 | 0.4% |
| log_gdp | 1 | 0.2% |
| node_dli_us | 500 | 100.0% |
| import_share_from_us | 500 | 100.0% |
| vul_us | 500 | 100.0% |
| us_import_share | 500 | 100.0% |
| hhi_imports | 500 | 100.0% |
| ovi_gas | 478 | 95.6% |
| us_production_oil | 40 | 8.0% |
| us_production_gas | 40 | 8.0% |
| us_prod_shock | 40 | 8.0% |
| country_code | 500 | 100.0% |
| country_name_y | 500 | 100.0% |
| in_degree | 500 | 100.0% |
| out_degree | 500 | 100.0% |
| total_degree | 500 | 100.0% |
| norm_in_degree | 500 | 100.0% |
| norm_out_degree | 500 | 100.0% |
| norm_total_degree | 500 | 100.0% |
| in_strength | 500 | 100.0% |
| out_strength | 500 | 100.0% |
| total_strength | 500 | 100.0% |
| norm_in_strength | 500 | 100.0% |
| norm_out_strength | 500 | 100.0% |
| norm_total_strength | 500 | 100.0% |
| betweenness_centrality | 500 | 100.0% |
| pagerank_centrality | 500 | 100.0% |
| eigenvector_centrality | 500 | 100.0% |

### 数值变量基础统计

|       |      year |   gdp_current_usd |   population_total |   trade_openness_gdp_pct |   log_gdp |   log_population |   node_dli_us |   import_share_from_us |   vul_us |   us_import_share |   hhi_imports |   ovi_gas |   us_production_oil |   us_production_gas |   us_prod_shock |   in_degree |   out_degree |   total_degree |   norm_in_degree |   norm_out_degree |   norm_total_degree |   in_strength |   out_strength |   total_strength |   norm_in_strength |   norm_out_strength |   norm_total_strength |   betweenness_centrality |   pagerank_centrality |   eigenvector_centrality |
|:------|----------:|------------------:|-------------------:|-------------------------:|----------:|-----------------:|--------------:|-----------------------:|---------:|------------------:|--------------:|----------:|--------------------:|--------------------:|----------------:|------------:|-------------:|---------------:|-----------------:|------------------:|--------------------:|--------------:|---------------:|-----------------:|-------------------:|--------------------:|----------------------:|-------------------------:|----------------------:|-------------------------:|
| count |  500      |     499           |      500           |                 498      |  499      |         500      |             0 |                      0 |        0 |                 0 |             0 |   22      |       460           |       460           |        460      |           0 |            0 |              0 |                0 |                 0 |                   0 |             0 |              0 |                0 |                  0 |                   0 |                     0 |                        0 |                     0 |                        0 |
| mean  | 2012      |       2.83452e+12 |        2.15952e+08 |                  60.9756 |   28.1124 |          18.2721 |           nan |                    nan |      nan |               nan |           nan |    0.7251 |    748366           |         1.34858e+07 |         -0      |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |
| std   |    7.2183 |       4.25309e+12 |        3.74998e+08 |                  28.8287 |    0.9689 |           1.273  |           nan |                    nan |      nan |               nan |           nan |    0.2878 |    361393           |         1.23529e+07 |          0.7889 |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |
| min   | 2000      |       1.60447e+11 |        7.18425e+06 |                  19.5596 |   25.8012 |          15.7874 |           nan |                    nan |      nan |               nan |           nan |    0.1999 |    188365           |         1.29101e+06 |         -1.786  |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |
| 25%   | 2006      |       8.62584e+11 |        4.07798e+07 |                  43.0968 |   27.4832 |          17.5237 |           nan |                    nan |      nan |               nan |           nan |    0.5212 |    458474           |         4.94591e+06 |         -0.3686 |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |
| 50%   | 2012      |       1.48757e+12 |        6.81763e+07 |                  55.8223 |   28.0282 |          18.0376 |           nan |                    nan |      nan |               nan |           nan |    0.8128 |    653059           |         9.55428e+06 |         -0.1384 |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |
| 75%   | 2018      |       2.66933e+12 |        1.53452e+08 |                  68.9957 |   28.6128 |          18.8461 |           nan |                    nan |      nan |               nan |           nan |    0.8878 |         1.11747e+06 |         1.71512e+07 |          0.4155 |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |
| max   | 2024      |       2.91849e+13 |        1.45094e+09 |                 184.107  |   31.0047 |          21.0955 |           nan |                    nan |      nan |               nan |           nan |    1.2402 |         1.40409e+06 |         4.49613e+07 |          1.9659 |         nan |          nan |            nan |              nan |               nan |                 nan |           nan |            nan |              nan |                nan |                 nan |                   nan |                      nan |                   nan |                      nan |

---

## 数据来源与构建方法


1. **宏观经济数据**: 通过wbdata包从世界银行WDI数据库获取
2. **贸易网络数据**: 基于01_data_processing模块的清洗贸易流数据
3. **网络拓扑指标**: 基于03_metrics模块计算的中心性指标
4. **DLI指标**: 基于04_dli_analysis模块的边级动态锁定指数
5. **物理基础设施数据**: 手动收集的LNG接收站和管道容量数据
6. **美国产量数据**: 通过EIA API获取的美国石油天然气产量数据

## 使用建议

1. **因变量选择**: 建议使用vul_us作为主要的脆弱性指标
2. **解释变量**: node_dli_us和ovi是核心解释变量
3. **控制变量**: 建议控制log_gdp, log_population, trade_openness_gdp_pct
4. **工具变量**: us_prod_shock可作为外生冲击的工具变量
5. **网络控制**: 可加入网络中心性指标作为额外控制

---

*本数据字典由08_variable_construction模块自动生成*  
*Energy Network Analysis Team*
