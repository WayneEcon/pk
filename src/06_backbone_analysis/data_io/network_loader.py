#!/usr/bin/env python3
"""
网络数据加载模块
================

从项目的02_net_analysis模块加载年度网络数据，
并提供统一的数据接口供骨干网络分析使用。

主要功能：
1. 加载02模块生成的年度网络文件
2. 加载03模块的节点中心性数据
3. 数据格式标准化和验证
4. 提供批量数据加载功能

作者：Energy Network Analysis Team
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
NET_ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed_data" / "networks"
CENTRALITY_DIR = PROJECT_ROOT / "src" / "03_centrality_analysis" 

def load_annual_network(year: int, 
                       networks_dir: Path = None,
                       file_format: str = 'graphml') -> Optional[nx.DiGraph]:
    """
    加载指定年份的网络数据
    
    Args:
        year: 年份
        networks_dir: 网络文件目录，默认使用02模块目录
        file_format: 文件格式 ('graphml', 'gexf', 'pickle')
        
    Returns:
        NetworkX有向图对象，失败时返回None
    """
    
    if networks_dir is None:
        networks_dir = NET_ANALYSIS_DIR
    
    # 尝试不同的文件名模式
    possible_filenames = [
        f"energy_network_{year}.{file_format}",
        f"network_{year}.{file_format}",
        f"{year}.{file_format}",
        f"trade_network_{year}.{file_format}"
    ]
    
    for filename in possible_filenames:
        filepath = networks_dir / filename
        
        if filepath.exists():
            logger.info(f"📂 加载{year}年网络: {filepath}")
            
            try:
                if file_format == 'graphml':
                    G = nx.read_graphml(filepath)
                elif file_format == 'gexf':
                    G = nx.read_gexf(filepath)
                elif file_format == 'pickle':
                    with open(filepath, 'rb') as f:
                        G = pickle.load(f)
                else:
                    raise ValueError(f"不支持的文件格式: {file_format}")
                
                # 确保是有向图
                if not isinstance(G, nx.DiGraph):
                    G = G.to_directed()
                
                # 验证网络基本信息
                logger.info(f"   节点数: {G.number_of_nodes():,}")
                logger.info(f"   边数: {G.number_of_edges():,}")
                
                # 验证权重属性
                weight_attrs = []
                for _, _, data in G.edges(data=True):
                    weight_attrs.extend(data.keys())
                    break
                
                if weight_attrs:
                    logger.info(f"   边属性: {weight_attrs}")
                
                return G
                
            except Exception as e:
                logger.error(f"❌ 加载{filepath}失败: {e}")
                continue
    
    logger.warning(f"⚠️ 未找到{year}年的网络文件")
    return None

def load_annual_networks(year_range: Tuple[int, int] = (2001, 2024),
                        networks_dir: Path = None,
                        file_format: str = 'graphml') -> Dict[int, nx.DiGraph]:
    """
    批量加载年度网络数据
    
    Args:
        year_range: 年份范围 (start, end) 包含边界
        networks_dir: 网络文件目录
        file_format: 文件格式
        
    Returns:
        年份到网络的映射字典
    """
    
    logger.info(f"🚀 批量加载网络数据 ({year_range[0]}-{year_range[1]})...")
    
    networks = {}
    start_year, end_year = year_range
    
    for year in range(start_year, end_year + 1):
        G = load_annual_network(year, networks_dir, file_format)
        if G is not None:
            networks[year] = G
    
    logger.info(f"✅ 成功加载 {len(networks)} 个年份的网络数据")
    logger.info(f"   覆盖年份: {sorted(networks.keys())}")
    
    if len(networks) == 0:
        logger.error("❌ 没有成功加载任何网络数据")
        
        # 尝试列出目录中的文件帮助调试
        if networks_dir and networks_dir.exists():
            logger.info(f"📁 目录 {networks_dir} 中的文件:")
            for file in sorted(networks_dir.glob("*")):
                if file.is_file():
                    logger.info(f"   {file.name}")
    
    return networks

def load_centrality_data(year: int,
                        centrality_dir: Path = None) -> Optional[pd.DataFrame]:
    """
    加载指定年份的节点中心性数据
    
    Args:
        year: 年份
        centrality_dir: 中心性数据目录
        
    Returns:
        包含中心性指标的DataFrame，失败时返回None
    """
    
    if centrality_dir is None:
        centrality_dir = CENTRALITY_DIR
    
    # 尝试不同的文件名模式
    possible_filenames = [
        f"centrality_metrics_{year}.csv",
        f"node_metrics_{year}.csv",
        f"{year}_centrality.csv"
    ]
    
    for filename in possible_filenames:
        filepath = centrality_dir / filename
        
        if filepath.exists():
            logger.info(f"📊 加载{year}年中心性数据: {filepath}")
            
            try:
                df = pd.read_csv(filepath)
                logger.info(f"   节点数: {len(df)}")
                logger.info(f"   指标列: {list(df.columns)}")
                return df
                
            except Exception as e:
                logger.error(f"❌ 加载{filepath}失败: {e}")
                continue
    
    logger.warning(f"⚠️ 未找到{year}年的中心性数据")
    return None

def load_country_metadata() -> Optional[pd.DataFrame]:
    """
    加载国家元数据（地理区域、经济分类等）
    
    Returns:
        国家元数据DataFrame，失败时返回None
    """
    
    # 尝试从不同位置加载元数据
    possible_paths = [
        PROJECT_ROOT / "data" / "country_metadata.csv",
        PROJECT_ROOT / "src" / "country_metadata.csv",
        NET_ANALYSIS_DIR / "country_metadata.csv",
        CENTRALITY_DIR / "country_metadata.csv"
    ]
    
    for filepath in possible_paths:
        if filepath.exists():
            logger.info(f"🌍 加载国家元数据: {filepath}")
            
            try:
                df = pd.read_csv(filepath)
                logger.info(f"   国家数: {len(df)}")
                return df
                
            except Exception as e:
                logger.error(f"❌ 加载{filepath}失败: {e}")
                continue
    
    logger.warning("⚠️ 未找到国家元数据文件")
    return None

def validate_network_consistency(networks: Dict[int, nx.DiGraph]) -> Dict[str, any]:
    """
    验证多年网络数据的一致性
    
    Args:
        networks: 年份到网络的映射字典
        
    Returns:
        一致性检查结果字典
    """
    
    logger.info("🔍 验证网络数据一致性...")
    
    if not networks:
        return {'status': 'empty', 'message': '没有网络数据'}
    
    years = sorted(networks.keys())
    results = {
        'years': years,
        'node_consistency': {},
        'edge_attributes': {},
        'graph_attributes': {},
        'issues': []
    }
    
    # 1. 检查节点一致性
    all_nodes = set()
    for year, G in networks.items():
        year_nodes = set(G.nodes())
        all_nodes.update(year_nodes)
        results['node_consistency'][year] = len(year_nodes)
    
    results['total_unique_nodes'] = len(all_nodes)
    
    # 检查节点数量变化
    node_counts = list(results['node_consistency'].values())
    if max(node_counts) - min(node_counts) > len(all_nodes) * 0.1:
        results['issues'].append('节点数量变化较大')
    
    # 2. 检查边属性一致性
    edge_attrs = {}
    for year, G in networks.items():
        year_attrs = set()
        for _, _, data in G.edges(data=True):
            year_attrs.update(data.keys())
            break  # 只检查第一条边
        edge_attrs[year] = year_attrs
    
    results['edge_attributes'] = edge_attrs
    
    # 检查属性一致性
    if len(edge_attrs) > 1:
        common_attrs = set.intersection(*edge_attrs.values())
        if len(common_attrs) == 0:
            results['issues'].append('没有共同的边属性')
    
    # 3. 检查图属性
    graph_attrs = {}
    for year, G in networks.items():
        graph_attrs[year] = dict(G.graph)
    
    results['graph_attributes'] = graph_attrs
    
    # 4. 统计摘要
    edge_counts = [G.number_of_edges() for G in networks.values()]
    results['edge_count_range'] = (min(edge_counts), max(edge_counts))
    results['avg_edge_count'] = sum(edge_counts) / len(edge_counts)
    
    # 状态判断
    if results['issues']:
        results['status'] = 'issues_found'
        logger.warning(f"⚠️ 发现 {len(results['issues'])} 个问题:")
        for issue in results['issues']:
            logger.warning(f"   {issue}")
    else:
        results['status'] = 'consistent'
        logger.info("✅ 网络数据一致性检查通过")
    
    logger.info("📊 网络数据统计:")
    logger.info(f"   年份范围: {min(years)} - {max(years)}")
    logger.info(f"   节点数范围: {min(node_counts)} - {max(node_counts)}")
    logger.info(f"   边数范围: {results['edge_count_range'][0]:,} - {results['edge_count_range'][1]:,}")
    
    return results

def save_backbone_network(G: nx.Graph, 
                         filepath: Path,
                         file_format: str = 'graphml',
                         include_metadata: bool = True) -> bool:
    """
    保存骨干网络到文件
    
    Args:
        G: 骨干网络
        filepath: 保存路径
        file_format: 文件格式
        include_metadata: 是否包含元数据
        
    Returns:
        是否保存成功
    """
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format == 'graphml':
            nx.write_graphml(G, filepath)
        elif file_format == 'gexf':
            nx.write_gexf(G, filepath)
        elif file_format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(G, f)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        logger.info(f"💾 骨干网络已保存: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 保存骨干网络失败: {e}")
        return False

if __name__ == "__main__":
    # 测试数据加载功能
    logger.info("🧪 测试网络数据加载...")
    
    # 测试加载单年网络
    test_year = 2020
    G = load_annual_network(test_year)
    
    if G:
        print(f"✅ 成功加载{test_year}年网络")
        print(f"   节点数: {G.number_of_nodes()}")
        print(f"   边数: {G.number_of_edges()}")
    
    # 测试批量加载
    networks = load_annual_networks((2018, 2020))
    
    if networks:
        print(f"✅ 成功批量加载: {list(networks.keys())}")
        
        # 一致性检查
        consistency_results = validate_network_consistency(networks)
        print(f"一致性状态: {consistency_results['status']}")
    
    print("🎉 测试完成!")