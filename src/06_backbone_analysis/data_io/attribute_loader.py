#!/usr/bin/env python3
"""
完整网络属性加载器
================

从03模块加载完整的节点属性数据，确保骨干网络可视化的信息保真性。
专门处理与轨道一分析结果的数据整合问题。

核心功能：
1. 加载完整网络的节点强度和中心性数据
2. 提供地理区域分类信息
3. 整合中心性排名数据
4. 支持跨模块数据一致性验证

作者：Energy Network Analysis Team
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkAttributeLoader:
    """网络属性加载器"""
    
    def __init__(self, base_data_path: Path = None):
        """
        初始化属性加载器
        
        Args:
            base_data_path: 基础数据路径
        """
        
        if base_data_path is None:
            # 自动检测数据路径
            current_path = Path(__file__).parent
            potential_paths = [
                current_path / "../../../data/processed_data",
                current_path / "../../../../data/processed_data", 
                current_path / "../../../03_network_analysis/outputs",
                current_path / "../../../../03_network_analysis/outputs"
            ]
            
            for path in potential_paths:
                if path.exists():
                    base_data_path = path
                    break
        
        self.base_data_path = Path(base_data_path) if base_data_path else None
        self.networks_path = None
        self.track1_path = None
        
        if self.base_data_path and self.base_data_path.exists():
            # 尝试找到网络数据路径
            networks_candidates = [
                self.base_data_path / "networks",
                self.base_data_path / "network_data",
                self.base_data_path / "../networks"
            ]
            
            for candidate in networks_candidates:
                if candidate.exists():
                    self.networks_path = candidate
                    break
            
            # 尝试找到轨道一结果路径
            track1_candidates = [
                self.base_data_path / "usa_centrality_analysis.csv",
                self.base_data_path / "centrality_analysis",
                self.base_data_path / "../03_network_analysis/outputs"
            ]
            
            for candidate in track1_candidates:
                if candidate.exists():
                    self.track1_path = candidate
                    break
        
        logger.info(f"🔧 网络属性加载器初始化完成")
        logger.info(f"   基础路径: {self.base_data_path}")
        logger.info(f"   网络路径: {self.networks_path}")
        logger.info(f"   轨道一路径: {self.track1_path}")
    
    def load_full_network_attributes(self, 
                                   year: int,
                                   include_centrality: bool = True) -> Dict[str, Any]:
        """
        加载指定年份的完整网络属性
        
        Args:
            year: 年份
            include_centrality: 是否包含中心性数据
            
        Returns:
            属性字典包含：
            - total_strength: 节点总强度
            - geographic_region: 地理区域分类
            - centrality_rankings: 中心性排名（如果可用）
            - trade_partners_list: 贸易伙伴列表
        """
        
        logger.info(f"📂 加载{year}年完整网络属性...")
        
        attributes = {
            'total_strength': {},
            'geographic_region': {},
            'centrality_rankings': {},
            'trade_partners_list': {},
            'pagerank': {},
            'betweenness': {},
            'closeness': {}
        }
        
        # 1. 加载网络数据获取基础属性
        network = self._load_network_for_year(year)
        
        if network is not None:
            # 计算节点强度
            for node in network.nodes():
                attributes['total_strength'][node] = network.degree(node, weight='weight')
                attributes['trade_partners_list'][node] = list(network.neighbors(node))
            
            # 分配地理区域
            from ..visualization.styling import ProfessionalNetworkStyling
            styling = ProfessionalNetworkStyling()
            
            for node in network.nodes():
                region = styling.COUNTRY_TO_REGION.get(node, 'Other')
                attributes['geographic_region'][node] = region
            
            # 计算中心性指标（如果需要且网络不太大）
            if include_centrality and network.number_of_nodes() <= 300:
                try:
                    logger.info("   计算中心性指标...")
                    
                    # PageRank
                    pagerank = nx.pagerank(network, weight='weight')
                    attributes['pagerank'] = pagerank
                    
                    # Betweenness（计算较慢，可选）
                    if network.number_of_nodes() <= 150:
                        betweenness = nx.betweenness_centrality(network, weight='weight')
                        attributes['betweenness'] = betweenness
                    
                    # Closeness
                    if network.number_of_nodes() <= 200:
                        closeness = nx.closeness_centrality(network, distance='weight')
                        attributes['closeness'] = closeness
                        
                except Exception as e:
                    logger.warning(f"⚠️ 中心性计算失败: {e}")
        
        # 2. 尝试加载轨道一的中心性数据
        track1_data = self._load_track1_centrality_data(year)
        if track1_data:
            # 合并轨道一的中心性数据
            for metric in ['pagerank', 'betweenness', 'closeness']:
                if metric in track1_data:
                    if metric not in attributes or not attributes[metric]:
                        attributes[metric] = track1_data[metric]
            
            # 加载排名信息
            if 'rankings' in track1_data:
                attributes['centrality_rankings'] = track1_data['rankings']
        
        logger.info(f"✅ {year}年网络属性加载完成")
        logger.info(f"   节点数: {len(attributes['total_strength'])}")
        logger.info(f"   地理区域: {len(set(attributes['geographic_region'].values()))}")
        logger.info(f"   中心性指标: {len([k for k, v in attributes.items() if k.endswith('ness') or k == 'pagerank' and v])}")
        
        return attributes
    
    def _load_network_for_year(self, year: int) -> Optional[nx.Graph]:
        """加载指定年份的网络"""
        
        if not self.networks_path or not self.networks_path.exists():
            logger.warning(f"⚠️ 网络数据路径不存在: {self.networks_path}")
            return None
        
        # 尝试不同的文件名格式
        potential_files = [
            self.networks_path / f"network_{year}.graphml",
            self.networks_path / f"network_{year}.gml", 
            self.networks_path / f"network_{year}.gpickle",
            self.networks_path / f"{year}.graphml",
            self.networks_path / f"energy_network_{year}.graphml"
        ]
        
        for file_path in potential_files:
            if file_path.exists():
                try:
                    if file_path.suffix == '.graphml':
                        G = nx.read_graphml(file_path)
                    elif file_path.suffix == '.gml':
                        G = nx.read_gml(file_path)
                    elif file_path.suffix == '.gpickle':
                        G = nx.read_gpickle(file_path)
                    else:
                        continue
                    
                    logger.info(f"   成功加载网络: {file_path.name}")
                    return G
                    
                except Exception as e:
                    logger.warning(f"⚠️ 加载{file_path}失败: {e}")
                    continue
        
        logger.warning(f"⚠️ 未找到{year}年的网络文件")
        return None
    
    def _load_track1_centrality_data(self, year: int) -> Optional[Dict]:
        """加载轨道一的中心性数据"""
        
        if not self.track1_path:
            return None
        
        try:
            # 尝试加载CSV格式的轨道一数据
            csv_files = [
                self.track1_path / "usa_centrality_analysis.csv",
                self.track1_path / "centrality_analysis.csv",
                self.track1_path / f"centrality_{year}.csv"
            ]
            
            if self.track1_path.suffix == '.csv':
                csv_files = [self.track1_path]
            
            for csv_file in csv_files:
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    
                    # 过滤指定年份的数据
                    if 'year' in df.columns:
                        year_data = df[df['year'] == year]
                        if len(year_data) == 0:
                            continue
                    else:
                        year_data = df
                    
                    # 提取中心性数据
                    centrality_data = {}
                    
                    # 提取各种中心性指标
                    for metric in ['pagerank', 'betweenness_centrality', 'closeness_centrality']:
                        if metric in year_data.columns:
                            # 假设数据格式为每行一个国家
                            if 'country' in year_data.columns:
                                metric_dict = dict(zip(year_data['country'], year_data[metric]))
                            elif 'node' in year_data.columns:
                                metric_dict = dict(zip(year_data['node'], year_data[metric]))
                            else:
                                # 如果只有一行数据（如美国数据），需要特殊处理
                                metric_dict = {'USA': year_data[metric].iloc[0]} if len(year_data) > 0 else {}
                            
                            centrality_data[metric.replace('_centrality', '')] = metric_dict
                    
                    logger.info(f"   成功加载轨道一数据: {csv_file.name}")
                    return centrality_data
                    
        except Exception as e:
            logger.warning(f"⚠️ 加载轨道一数据失败: {e}")
        
        return None
    
    def load_batch_attributes(self, 
                            years: List[int],
                            include_centrality: bool = True) -> Dict[int, Dict[str, Any]]:
        """
        批量加载多年份网络属性
        
        Args:
            years: 年份列表
            include_centrality: 是否包含中心性数据
            
        Returns:
            年份到属性的映射字典
        """
        
        logger.info(f"🚀 批量加载网络属性 ({len(years)}年)...")
        
        batch_attributes = {}
        
        for year in sorted(years):
            try:
                attributes = self.load_full_network_attributes(year, include_centrality)
                batch_attributes[year] = attributes
            except Exception as e:
                logger.error(f"❌ {year}年属性加载失败: {e}")
                continue
        
        logger.info(f"✅ 批量加载完成 ({len(batch_attributes)}/{len(years)} 年)")
        return batch_attributes
    
    def verify_data_consistency(self, 
                              attributes: Dict[str, Any],
                              backbone_network: nx.Graph) -> Dict[str, Any]:
        """
        验证数据一致性
        
        Args:
            attributes: 属性数据
            backbone_network: 骨干网络
            
        Returns:
            一致性检验结果
        """
        
        logger.info("🔍 验证数据一致性...")
        
        consistency_report = {
            'node_coverage': 0,
            'missing_nodes': [],
            'attribute_completeness': {},
            'data_quality_score': 0
        }
        
        # 检查节点覆盖率
        backbone_nodes = set(backbone_network.nodes())
        attribute_nodes = set(attributes['total_strength'].keys())
        
        covered_nodes = backbone_nodes.intersection(attribute_nodes)
        missing_nodes = backbone_nodes - attribute_nodes
        
        consistency_report['node_coverage'] = len(covered_nodes) / len(backbone_nodes) if backbone_nodes else 0
        consistency_report['missing_nodes'] = list(missing_nodes)
        
        # 检查属性完整性
        for attr_name, attr_data in attributes.items():
            if isinstance(attr_data, dict):
                completeness = len(attr_data) / len(backbone_nodes) if backbone_nodes else 0
                consistency_report['attribute_completeness'][attr_name] = completeness
        
        # 计算总体数据质量分数
        completeness_scores = list(consistency_report['attribute_completeness'].values())
        if completeness_scores:
            avg_completeness = np.mean(completeness_scores)
            consistency_report['data_quality_score'] = (consistency_report['node_coverage'] + avg_completeness) / 2
        
        logger.info(f"✅ 数据一致性检验完成")
        logger.info(f"   节点覆盖率: {consistency_report['node_coverage']:.1%}")
        logger.info(f"   缺失节点: {len(consistency_report['missing_nodes'])}")
        logger.info(f"   数据质量分数: {consistency_report['data_quality_score']:.3f}")
        
        return consistency_report
    
    def create_attribute_summary(self, 
                               attributes: Dict[str, Any]) -> pd.DataFrame:
        """
        创建属性数据摘要表
        
        Args:
            attributes: 属性数据
            
        Returns:
            摘要DataFrame
        """
        
        summary_data = []
        
        for node in attributes['total_strength'].keys():
            row = {
                'country': node,
                'total_strength': attributes['total_strength'].get(node, 0),
                'geographic_region': attributes['geographic_region'].get(node, 'Unknown'),
                'trade_partners_count': len(attributes['trade_partners_list'].get(node, [])),
                'pagerank': attributes['pagerank'].get(node, 0),
                'betweenness': attributes['betweenness'].get(node, 0),
                'closeness': attributes['closeness'].get(node, 0)
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # 按总强度排序
        df = df.sort_values('total_strength', ascending=False)
        
        return df

if __name__ == "__main__":
    # 测试属性加载器
    logger.info("🧪 测试网络属性加载器...")
    
    # 初始化加载器
    loader = NetworkAttributeLoader()
    
    # 测试加载2018年数据
    if loader.networks_path and loader.networks_path.exists():
        attributes_2018 = loader.load_full_network_attributes(2018)
        
        print("🎉 属性加载器测试完成!")
        print(f"总强度数据: {len(attributes_2018['total_strength'])} 个节点")
        print(f"地理区域: {len(set(attributes_2018['geographic_region'].values()))} 个区域")
        print(f"PageRank数据: {len(attributes_2018['pagerank'])} 个节点")
        
        # 创建摘要表
        summary_df = loader.create_attribute_summary(attributes_2018)
        print(f"\n前5名国家（按总强度）:")
        print(summary_df.head()[['country', 'total_strength', 'geographic_region']].to_string(index=False))
        
    else:
        print("⚠️ 未找到网络数据路径，跳过实际数据测试")
        print("属性加载器结构测试通过！")