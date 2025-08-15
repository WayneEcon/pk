#!/usr/bin/env python3
"""
02模块标准数据接口 (Network Data Interface)
=========================================

为其他模块提供标准化的网络数据访问接口
"""

import networkx as nx
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

def get_networks_by_years(years: List[int] = None) -> Dict[int, nx.Graph]:
    """
    标准网络数据接口
    
    Args:
        years: 指定年份列表，None则返回所有年份
        
    Returns:
        Dict[year, networkx.Graph]: 年份到网络图的字典
    """
    module_dir = Path(__file__).parent
    outputs_dir = module_dir / "outputs"
    
    networks = {}
    
    if not outputs_dir.exists():
        logger.warning(f"02模块输出目录不存在: {outputs_dir}")
        return {}
    
    # 搜索网络文件
    network_files = list(outputs_dir.glob("*.pickle")) + list(outputs_dir.glob("**/*.pickle"))
    
    for file in network_files:
        try:
            # 从文件名提取年份
            parts = file.stem.split('_')
            year = None
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    break
            
            if year is None:
                continue
                
            if years is None or year in years:
                G = nx.read_gpickle(file)
                networks[year] = G
                logger.info(f"✅ 加载网络: {year}")
                
        except Exception as e:
            logger.warning(f"⚠️ 无法加载网络文件 {file}: {e}")
    
    return networks

def get_available_years() -> List[int]:
    """获取可用的年份列表"""
    networks = get_networks_by_years()
    return sorted(networks.keys())

def get_network_summary() -> pd.DataFrame:
    """获取网络数据概览"""
    networks = get_networks_by_years()
    
    summary_data = []
    for year, G in networks.items():
        summary_data.append({
            'year': year,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'is_directed': G.is_directed(),
            'total_weight': sum(d.get('weight', 0) for u, v, d in G.edges(data=True))
        })
    
    return pd.DataFrame(summary_data)

def validate_network_data() -> Dict[str, bool]:
    """验证网络数据的完整性和一致性"""
    networks = get_networks_by_years()
    
    validation = {
        'has_data': len(networks) > 0,
        'has_weights': False,
        'has_country_codes': False,
        'time_series_complete': False
    }
    
    if networks:
        sample_graph = next(iter(networks.values()))
        
        # 检查权重
        if sample_graph.edges():
            validation['has_weights'] = any('weight' in d for u, v, d in sample_graph.edges(data=True))
        
        # 检查国家代码
        if sample_graph.nodes():
            validation['has_country_codes'] = any('country' in d for n, d in sample_graph.nodes(data=True))
        
        # 检查时间序列完整性
        years = sorted(networks.keys())
        if years:
            expected_years = list(range(min(years), max(years) + 1))
            validation['time_series_complete'] = len(years) == len(expected_years)
    
    return validation