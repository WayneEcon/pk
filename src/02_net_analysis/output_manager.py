#!/usr/bin/env python3
"""
输出管理模块
负责保存网络数据和生成报告
"""

import pandas as pd
import networkx as nx
import pickle
import logging
from pathlib import Path
from typing import Dict, List
from .utils import (setup_path, log_dataframe_info, safe_divide, 
                   NETWORKS_DIR, NETWORK_STATS_DIR, FILE_TEMPLATES, 
                   DATA_CONSISTENCY_STRATEGY, ENERGY_PRODUCT_CODES, 
                   REGIONAL_GROUPS, FOCUS_COUNTRIES, DATA_VALIDATION)

# 确保路径设置  
setup_path()

logger = logging.getLogger(__name__)

def save_networks_comprehensive(annual_networks: Dict[int, nx.DiGraph], network_stats: List[Dict]) -> None:
    """
    全面保存网络数据（多种格式）
    
    Args:
        annual_networks: 年度网络字典，键为年份，值为NetworkX图对象
        network_stats: 网络统计列表，每个元素为包含统计指标的字典
        
    Raises:
        Exception: 当保存过程中出现错误时
        
    Example:
        >>> networks = {2020: G2020, 2021: G2021}
        >>> stats = [{"year": 2020, "nodes": 100}, {"year": 2021, "nodes": 105}]
        >>> save_networks_comprehensive(networks, stats)
    """
    logger.info(f"\n💾 开始保存网络数据...")
    
    if not annual_networks:
        logger.warning("没有网络数据需要保存")
        return
    
    years = sorted(annual_networks.keys())
    start_year, end_year = min(years), max(years)
    
    try:
        # 1. 保存网络对象（Pickle格式 - 快速加载）
        networks_file = NETWORKS_DIR / FILE_TEMPLATES['network_pickle'].format(
            start_year=start_year, end_year=end_year)
        with open(networks_file, 'wb') as f:
            pickle.dump(annual_networks, f)
        logger.info(f"     ✅ Pickle网络文件: {networks_file}")
        
        # 2. 保存网络集合（GraphML格式 - 通用兼容，优化版）
        graphml_count = 0
        
        def clean_graph_for_graphml(G: nx.DiGraph) -> nx.DiGraph:
            """清理图以适配GraphML格式"""
            G_clean = G.copy()
            
            # 清理图级别属性（保留基本类型）
            allowed_types = (str, int, float, bool)
            G_clean.graph = {
                k: v for k, v in G_clean.graph.items() 
                if isinstance(v, allowed_types)
            }
            
            # 批量清理节点属性
            for node in G_clean.nodes():
                G_clean.nodes[node] = {
                    k: v for k, v in G_clean.nodes[node].items()
                    if isinstance(v, allowed_types)
                }
            
            # 批量清理边属性
            for source, target in G_clean.edges():
                G_clean.edges[source, target] = {
                    k: v for k, v in G_clean.edges[source, target].items()
                    if isinstance(v, allowed_types)
                }
            
            return G_clean
        
        for year, G in annual_networks.items():
            try:
                # 使用优化的清理函数
                G_clean = clean_graph_for_graphml(G)
                
                graphml_file = NETWORKS_DIR / f"network_{year}.graphml"
                nx.write_graphml(G_clean, graphml_file)
                graphml_count += 1
                
            except Exception as e:
                logger.warning(f"保存 {year} 年GraphML文件失败: {e}")
        
        logger.info(f"     ✅ GraphML网络文件: {graphml_count} 个年度文件")
        
        # 3. 保存网络统计信息
        stats_df = pd.DataFrame(network_stats)
        stats_file = NETWORK_STATS_DIR / FILE_TEMPLATES['basic_stats']
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"     ✅ 统计信息: {stats_file}")
        
        # 4. 生成年度节点和边文件
        logger.info(f"     🔄 生成年度节点和边文件...")
        nodes_edges_generated = generate_annual_nodes_edges(annual_networks)
        logger.info(f"     ✅ 节点边文件: {nodes_edges_generated} 对")
        
        # 5. 生成详细摘要报告（Markdown格式）
        generate_summary_report(annual_networks, network_stats)
        logger.info(f"     ✅ 摘要报告: {NETWORK_STATS_DIR / FILE_TEMPLATES['summary_report']}")
        
        logger.info(f"\n🎯 保存完成! 所有文件已输出到相应目录")
        
    except Exception as e:
        logger.error(f"保存网络数据时出错: {e}")
        raise

def generate_annual_nodes_edges(annual_networks: Dict[int, nx.DiGraph]) -> int:
    """
    为每年的网络生成节点和边CSV文件（优化版）
    
    Args:
        annual_networks: 年度网络字典
        
    Returns:
        成功生成的年度文件对数量
        
    Example:
        >>> count = generate_annual_nodes_edges(annual_networks)
        >>> print(f"生成了 {count} 对年度文件")
    """
    generated_count = 0
    
    for year, G in annual_networks.items():
        try:
            # 优化：预计算所有节点的度和强度（批量计算）
            if G.number_of_nodes() > 0:
                out_degrees = dict(G.out_degree())
                in_degrees = dict(G.in_degree())
                out_strengths = dict(G.out_degree(weight='weight'))
                in_strengths = dict(G.in_degree(weight='weight'))
                
                # 生成节点文件（vectorized操作）
                nodes_data = {
                    'country_code': list(G.nodes()),
                    'country_name': [G.nodes[node].get('name', node) for node in G.nodes()],
                    'region': [G.nodes[node].get('region', 'Other') for node in G.nodes()],
                    'out_degree': [out_degrees[node] for node in G.nodes()],
                    'in_degree': [in_degrees[node] for node in G.nodes()],
                    'out_strength': [out_strengths[node] for node in G.nodes()],
                    'in_strength': [in_strengths[node] for node in G.nodes()],
                    'total_strength': [out_strengths[node] + in_strengths[node] for node in G.nodes()]
                }
                
                nodes_df = pd.DataFrame(nodes_data)
                nodes_file = NETWORKS_DIR / FILE_TEMPLATES['nodes_file'].format(year=year)
                
                # 优化：使用高效的CSV写入
                nodes_df.to_csv(nodes_file, index=False, float_format='%.2f')
            else:
                # 空网络的处理
                empty_nodes_df = pd.DataFrame(columns=[
                    'country_code', 'country_name', 'region', 'out_degree', 
                    'in_degree', 'out_strength', 'in_strength', 'total_strength'
                ])
                nodes_file = NETWORKS_DIR / FILE_TEMPLATES['nodes_file'].format(year=year)
                empty_nodes_df.to_csv(nodes_file, index=False)
            
            # 生成边文件（优化：使用列表推导式）
            if G.number_of_edges() > 0:
                edges_data = {
                    'source': [edge[0] for edge in G.edges()],
                    'target': [edge[1] for edge in G.edges()],
                    'weight': [G.edges[edge]['weight'] for edge in G.edges()],
                    'data_source': [G.edges[edge].get('data_source', 'unknown') for edge in G.edges()]
                }
                
                edges_df = pd.DataFrame(edges_data)
                edges_file = NETWORKS_DIR / FILE_TEMPLATES['edges_file'].format(year=year)
                
                # 优化：使用高效的CSV写入
                edges_df.to_csv(edges_file, index=False, float_format='%.2f')
            else:
                # 空网络的边文件
                empty_edges_df = pd.DataFrame(columns=['source', 'target', 'weight', 'data_source'])
                edges_file = NETWORKS_DIR / FILE_TEMPLATES['edges_file'].format(year=year)
                empty_edges_df.to_csv(edges_file, index=False)
            
            generated_count += 1
            
        except Exception as e:
            logger.error(f"生成 {year} 年节点边文件时出错: {e}")
    
    return generated_count

def generate_summary_report(annual_networks: Dict[int, nx.DiGraph], network_stats: List[Dict]) -> None:
    """
    生成Markdown格式的详细摘要报告
    
    Args:
        annual_networks: 年度网络字典
        network_stats: 网络统计数据列表
        
    Side Effects:
        在NETWORK_STATS_DIR目录下生成Markdown报告文件
        
    Example:
        >>> generate_summary_report(annual_networks, network_stats)
        # 生成报告文件到指定目录
    """
    years = sorted(annual_networks.keys())
    start_year, end_year = min(years), max(years)
    
    report_content = f"""# 美国能源独立政策研究 - 网络构建摘要报告

## 📊 构建概览

- **构建时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **网络数量**: {len(annual_networks)} 个年度网络
- **时间跨度**: {start_year} - {end_year}
- **数据一致性策略**: {DATA_CONSISTENCY_STRATEGY}
- **能源产品**: {', '.join([f"{code}({name})" for code, name in ENERGY_PRODUCT_CODES.items()])}

## 🔍 整体统计

"""
    
    # 计算整体统计
    if network_stats:
        stats_df = pd.DataFrame(network_stats)
        
        report_content += f"""
### 网络规模演化
- **平均节点数**: {stats_df['nodes'].mean():.0f}
- **平均边数**: {stats_df['edges'].mean():.0f}
- **平均密度**: {stats_df['density'].mean():.4f}
- **总贸易额**: ${stats_df['total_trade_value'].sum()/1e12:.1f}万亿美元

### 美国贸易地位演化
- **{start_year}年美国贸易份额**: {stats_df.iloc[0]['usa_trade_share']:.1%}
- **{end_year}年美国贸易份额**: {stats_df.iloc[-1]['usa_trade_share']:.1%}
- **变化**: {(stats_df.iloc[-1]['usa_trade_share'] - stats_df.iloc[0]['usa_trade_share']):.1%}

## 📈 年度网络详情

| 年份 | 节点数 | 边数 | 密度 | 总贸易额(万亿$) | 美国份额 |
|------|--------|------|------|----------------|----------|
"""
        
        for _, row in stats_df.iterrows():
            report_content += f"| {row['year']} | {row['nodes']} | {row['edges']} | {row['density']:.4f} | {row['total_trade_value']/1e12:.2f} | {row['usa_trade_share']:.1%} |\n"
    
    report_content += f"""

## 🔧 技术说明

### 数据处理策略
1. **优先进口数据**: 对于双边贸易，优先使用进口方报告的数据
2. **镜像数据补充**: 当进口数据缺失时，使用出口方数据作为镜像补充
3. **贸易流聚合**: 将同一国家对的多种能源产品贸易额合并

### 数据质量控制
- **最小贸易额阈值**: ${DATA_VALIDATION['min_trade_value']:,} 美元
- **国家区域分组**: 已实现 {len(REGIONAL_GROUPS)} 个区域分组
- **核心关注国家**: {len(FOCUS_COUNTRIES)} 个重点分析国家

### 输出文件说明
- **Pickle格式**: 快速加载的Python网络对象
- **GraphML格式**: 通用的网络交换格式，支持Gephi、Cytoscape等工具
- **CSV格式**: 年度节点和边文件，便于进一步分析
- **统计文件**: 包含所有年度的网络拓扑统计指标

## 🚀 后续分析建议

1. **中心性分析**: 计算度中心性、中介中心性、特征向量中心性
2. **社群检测**: 使用Leiden算法识别贸易集团
3. **骨干网络提取**: 实施DF、PF、MST三种骨干提取方法
4. **动态分析**: 追踪关键指标的时间序列变化
5. **政策效应评估**: 将网络变化与政策时间点对应分析

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    report_file = NETWORK_STATS_DIR / FILE_TEMPLATES['summary_report']
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)