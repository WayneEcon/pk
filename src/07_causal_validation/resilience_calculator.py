#!/usr/bin/env python3
"""
网络韧性计算器 (Network Resilience Calculator)
=============================================

实现双轨韧性测量原则，确保因果推断结论的稳健性：

1. 拓扑抗毁性 (Topological Resilience)
   - 通过模拟攻击测量网络连通性损失速度
   - 基于网络科学理论的结构稳定性分析

2. 供应缺口吸收率 (Supply Gap Absorption Rate)  
   - 模拟主要供应商中断后的补充供应能力
   - 基于经济韧性理论的实际适应能力

作者：Energy Network Analysis Team
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkResilienceCalculator:
    """
    网络韧性计算器
    
    实现两种互补的韧性测量方法：
    1. 拓扑抗毁性 - 基于网络连通性的结构韧性
    2. 供应缺口吸收率 - 基于供需匹配的功能韧性
    """
    
    def __init__(self, 
                 attack_strategies: List[str] = None,
                 attack_proportions: List[float] = None):
        """
        初始化韧性计算器
        
        Args:
            attack_strategies: 攻击策略列表 ['degree', 'betweenness', 'random']
            attack_proportions: 攻击比例列表，默认[0.05, 0.10, 0.15, 0.20, 0.25]
        """
        self.attack_strategies = attack_strategies or ['degree', 'betweenness', 'random']
        self.attack_proportions = attack_proportions or [0.05, 0.10, 0.15, 0.20, 0.25]
        
        logger.info(f"🛡️ 初始化网络韧性计算器")
        logger.info(f"   攻击策略: {self.attack_strategies}")
        logger.info(f"   攻击比例: {self.attack_proportions}")

    def calculate_topological_resilience(self, 
                                       G: nx.Graph,
                                       node_id: str,
                                       year: int = None) -> Dict[str, float]:
        """
        计算节点的拓扑抗毁性
        
        方法：模拟移除Top-K节点，测量该节点所在连通分量规模的下降速度
        
        Args:
            G: 网络图
            node_id: 目标节点ID
            year: 年份（用于日志）
            
        Returns:
            包含各种韧性指标的字典
        """
        
        if node_id not in G.nodes():
            return {
                'topological_resilience_avg': 0.0,
                'topological_resilience_degree': 0.0,
                'topological_resilience_betweenness': 0.0,
                'topological_resilience_random': 0.0,
                'network_position_stability': 0.0
            }
        
        logger.debug(f"🎯 计算{node_id}拓扑抗毁性 ({year}年)" if year else f"🎯 计算{node_id}拓扑抗毁性")
        
        # 记录初始状态
        original_nodes = set(G.nodes())
        
        # 处理有向图和无向图的连通分量
        if isinstance(G, nx.DiGraph):
            # 对有向图，使用弱连通分量
            original_component_size = len(nx.node_connected_component(G.to_undirected(), node_id))
        else:
            original_component_size = len(nx.node_connected_component(G, node_id))
            
        original_network_size = G.number_of_nodes()
        
        resilience_scores = {}
        
        # 对每种攻击策略计算韧性
        for strategy in self.attack_strategies:
            strategy_scores = []
            
            # 确定攻击优先级
            if strategy == 'degree':
                # 按度中心性排序（降序）
                centrality = dict(G.degree(weight='weight'))
            elif strategy == 'betweenness':
                # 按介数中心性排序（降序）
                centrality = nx.betweenness_centrality(G, weight='weight')
            elif strategy == 'random':
                # 随机攻击 - 进行多次模拟取平均
                random_scores = []
                for _ in range(10):  # 10次随机模拟
                    nodes_list = list(original_nodes)
                    np.random.shuffle(nodes_list)
                    random_centrality = {node: np.random.random() for node in nodes_list}
                    random_score = self._simulate_attack(G, node_id, random_centrality, 
                                                       original_component_size)
                    random_scores.append(random_score)
                resilience_scores[f'topological_resilience_{strategy}'] = np.mean(random_scores)
                continue
            else:
                logger.warning(f"未知攻击策略: {strategy}")
                continue
                
            # 模拟攻击过程
            attack_score = self._simulate_attack(G, node_id, centrality, original_component_size)
            resilience_scores[f'topological_resilience_{strategy}'] = attack_score
        
        # 计算综合韧性得分（三种策略的平均值）
        strategy_values = [resilience_scores.get(f'topological_resilience_{s}', 0) 
                          for s in self.attack_strategies]
        resilience_scores['topological_resilience_avg'] = np.mean(strategy_values)
        
        # 计算网络位置稳定性（基于度中心性的相对位置）
        degree_centrality = dict(G.degree(weight='weight'))
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        node_rank = next((i+1 for i, (node, _) in enumerate(sorted_nodes) if node == node_id), 
                        len(sorted_nodes))
        position_stability = 1 - (node_rank - 1) / max(1, len(sorted_nodes) - 1)
        resilience_scores['network_position_stability'] = position_stability
        
        return resilience_scores
    
    def _simulate_attack(self, 
                        G: nx.Graph,
                        target_node: str, 
                        centrality: Dict[str, float],
                        original_component_size: int) -> float:
        """
        模拟攻击过程，计算目标节点连通分量的韧性得分
        
        Args:
            G: 网络图
            target_node: 目标节点
            centrality: 节点中心性字典
            original_component_size: 原始连通分量大小
            
        Returns:
            韧性得分（0-1之间，越高越韧性）
        """
        
        # 排除目标节点本身（不能攻击自己）
        attack_centrality = {node: score for node, score in centrality.items() 
                           if node != target_node}
        
        if not attack_centrality:
            return 1.0  # 如果没有其他节点可攻击，韧性最大
            
        # 按中心性排序确定攻击顺序
        sorted_attackable = sorted(attack_centrality.items(), key=lambda x: x[1], reverse=True)
        
        resilience_over_attacks = []
        G_attack = G.copy()
        
        # 模拟逐步攻击
        for attack_ratio in self.attack_proportions:
            num_attacks = max(1, int(len(sorted_attackable) * attack_ratio))
            
            # 移除前num_attacks个节点
            nodes_to_remove = [node for node, _ in sorted_attackable[:num_attacks]]
            G_attack.remove_nodes_from(nodes_to_remove)
            
            # 检查目标节点是否仍在图中
            if target_node not in G_attack.nodes():
                resilience_over_attacks.append(0.0)
                continue
                
            # 计算目标节点所在连通分量的当前大小
            try:
                if isinstance(G_attack, nx.DiGraph):
                    current_component_size = len(nx.node_connected_component(G_attack.to_undirected(), target_node))
                else:
                    current_component_size = len(nx.node_connected_component(G_attack, target_node))
                # 韧性 = 当前连通分量大小 / 原始连通分量大小
                resilience = current_component_size / max(1, original_component_size)
            except:
                resilience = 0.0
                
            resilience_over_attacks.append(resilience)
        
        # 返回攻击过程中韧性的平均值（代表整体抗攻击能力）
        return np.mean(resilience_over_attacks) if resilience_over_attacks else 0.0

    def calculate_supply_absorption(self, 
                                  G: nx.Graph,
                                  node_id: str,
                                  year: int = None,
                                  top_suppliers: int = 3) -> Dict[str, float]:
        """
        计算供应缺口吸收率
        
        方法：模拟主要供应商中断后，从网络其他节点获得补充供应的能力
        
        Args:
            G: 网络图（有向图，边权重代表供应流量）
            node_id: 目标节点ID  
            year: 年份（用于日志）
            top_suppliers: 考虑的主要供应商数量
            
        Returns:
            供应吸收能力相关指标
        """
        
        if node_id not in G.nodes():
            return {
                'supply_absorption_rate': 0.0,
                'supply_diversification_index': 0.0, 
                'supply_network_depth': 0.0,
                'alternative_suppliers_count': 0.0
            }
        
        logger.debug(f"📦 计算{node_id}供应缺口吸收率 ({year}年)" if year else f"📦 计算{node_id}供应缺口吸收率")
        
        # 1. 识别当前主要供应商（入边权重最大的节点）
        if isinstance(G, nx.DiGraph):
            suppliers = [(supplier, data.get('weight', 0)) 
                        for supplier, _, data in G.in_edges(node_id, data=True)]
        else:
            # 对于无向图，考虑所有邻居
            suppliers = [(neighbor, data.get('weight', 0)) 
                        for neighbor, data in G[node_id].items()]
        
        if not suppliers:
            return {
                'supply_absorption_rate': 0.0,
                'supply_diversification_index': 0.0,
                'supply_network_depth': 0.0, 
                'alternative_suppliers_count': 0.0
            }
        
        # 按供应量排序，取前top_suppliers个
        suppliers.sort(key=lambda x: x[1], reverse=True)
        major_suppliers = suppliers[:top_suppliers]
        total_major_supply = sum(weight for _, weight in major_suppliers)
        
        # 2. 计算供应多样化指数 (HHI的倒数)
        if total_major_supply > 0:
            supply_shares = [weight/total_major_supply for _, weight in major_suppliers]
            hhi = sum(share**2 for share in supply_shares)
            diversification_index = 1 / hhi if hhi > 0 else 1.0
        else:
            diversification_index = 1.0
            
        # 3. 模拟主要供应商中断，计算替代供应能力
        absorption_scores = []
        
        for disrupted_supplier, disrupted_supply in major_suppliers:
            # 创建中断情况下的网络副本
            G_disrupted = G.copy()
            if isinstance(G, nx.DiGraph):
                if G_disrupted.has_edge(disrupted_supplier, node_id):
                    G_disrupted.remove_edge(disrupted_supplier, node_id)
            else:
                if G_disrupted.has_edge(disrupted_supplier, node_id):
                    G_disrupted.remove_edge(disrupted_supplier, node_id)
            
            # 寻找替代供应来源
            alternative_supply = self._find_alternative_supply(
                G_disrupted, node_id, disrupted_supply
            )
            
            # 计算吸收率 = 可获得的替代供应 / 中断的供应
            absorption_rate = alternative_supply / disrupted_supply if disrupted_supply > 0 else 1.0
            absorption_scores.append(min(1.0, absorption_rate))  # 限制在[0,1]
        
        # 4. 计算供应网络深度（二度供应商的数量和强度）
        second_tier_suppliers = set()
        second_tier_supply = 0
        
        for supplier, _ in suppliers:
            if isinstance(G, nx.DiGraph):
                for second_supplier, _, data in G.in_edges(supplier, data=True):
                    if second_supplier != node_id:  # 避免循环
                        second_tier_suppliers.add(second_supplier)
                        second_tier_supply += data.get('weight', 0)
            else:
                for second_supplier, data in G[supplier].items():
                    if second_supplier != node_id:  # 避免循环
                        second_tier_suppliers.add(second_supplier) 
                        second_tier_supply += data.get('weight', 0)
        
        supply_network_depth = len(second_tier_suppliers) / max(1, G.number_of_nodes() - 1)
        
        # 5. 计算可替代供应商数量
        all_potential_suppliers = set()
        if isinstance(G, nx.DiGraph):
            # 对有向图，找所有可能的供应路径（最短路径长度<=2）
            try:
                shortest_paths = nx.single_source_shortest_path_length(
                    G.reverse(), node_id, cutoff=2
                )
                all_potential_suppliers = set(shortest_paths.keys()) - {node_id}
            except:
                all_potential_suppliers = set(G.nodes()) - {node_id}
        else:
            # 对无向图，考虑所有距离<=2的节点
            try:
                shortest_paths = nx.single_source_shortest_path_length(
                    G, node_id, cutoff=2
                )
                all_potential_suppliers = set(shortest_paths.keys()) - {node_id}
            except:
                all_potential_suppliers = set(G.nodes()) - {node_id}
        
        alternative_suppliers_count = len(all_potential_suppliers) / max(1, G.number_of_nodes() - 1)
        
        return {
            'supply_absorption_rate': np.mean(absorption_scores) if absorption_scores else 0.0,
            'supply_diversification_index': min(1.0, diversification_index / top_suppliers),  # 标准化
            'supply_network_depth': supply_network_depth,
            'alternative_suppliers_count': alternative_suppliers_count
        }
    
    def _find_alternative_supply(self, 
                               G_disrupted: nx.Graph, 
                               target_node: str,
                               needed_supply: float) -> float:
        """
        寻找替代供应来源
        
        Args:
            G_disrupted: 供应中断后的网络
            target_node: 目标节点
            needed_supply: 需要的供应量
            
        Returns:
            可获得的替代供应量
        """
        
        alternative_supply = 0
        
        if isinstance(G_disrupted, nx.DiGraph):
            # 计算所有潜在供应商的可用供应能力
            potential_suppliers = []
            for supplier in G_disrupted.predecessors(target_node):
                supply_capacity = G_disrupted[supplier][target_node].get('weight', 0)
                potential_suppliers.append((supplier, supply_capacity))
                
            # 尝试通过短路径增加供应
            try:
                for node in G_disrupted.nodes():
                    if node != target_node and not G_disrupted.has_edge(node, target_node):
                        try:
                            path_length = nx.shortest_path_length(G_disrupted, node, target_node)
                            if path_length <= 2:  # 最多通过1个中介
                                # 估算通过该路径的潜在供应能力
                                path_supply = self._estimate_path_capacity(G_disrupted, node, target_node)
                                potential_suppliers.append((node, path_supply))
                        except nx.NetworkXNoPath:
                            continue
            except:
                pass
                
        else:
            # 无向图情况
            for neighbor, data in G_disrupted[target_node].items():
                supply_capacity = data.get('weight', 0)
                potential_suppliers.append((neighbor, supply_capacity))
        
        # 按供应能力排序，优先使用供应能力强的替代者
        if 'potential_suppliers' in locals():
            potential_suppliers.sort(key=lambda x: x[1], reverse=True)
            alternative_supply = sum(capacity for _, capacity in potential_suppliers)
        
        return alternative_supply
    
    def _estimate_path_capacity(self, G: nx.DiGraph, source: str, target: str) -> float:
        """
        估算通过最短路径的供应能力（瓶颈容量）
        
        Args:
            G: 网络图
            source: 源节点
            target: 目标节点
            
        Returns:
            路径容量估算值
        """
        try:
            path = nx.shortest_path(G, source, target, weight='weight')
            if len(path) < 2:
                return 0
                
            # 找到路径上的最小权重（瓶颈）
            path_capacities = []
            for i in range(len(path) - 1):
                edge_weight = G[path[i]][path[i+1]].get('weight', 0)
                path_capacities.append(edge_weight)
                
            return min(path_capacities) if path_capacities else 0
            
        except:
            return 0

def calculate_topological_resilience(networks: Dict[int, nx.Graph],
                                   countries: List[str] = None,
                                   attack_strategies: List[str] = None) -> pd.DataFrame:
    """
    批量计算多年网络的拓扑抗毁性
    
    Args:
        networks: 年份到网络图的映射
        countries: 要分析的国家列表，None则分析所有国家
        attack_strategies: 攻击策略列表
        
    Returns:
        包含所有年份、国家的拓扑韧性数据框
    """
    
    logger.info("🛡️ 开始批量计算拓扑抗毁性...")
    
    calculator = NetworkResilienceCalculator(attack_strategies=attack_strategies)
    results = []
    
    # 确定要分析的国家
    if countries is None:
        all_countries = set()
        for G in networks.values():
            all_countries.update(G.nodes())
        countries = sorted(list(all_countries))
    
    # 逐年逐国分析
    for year in tqdm(sorted(networks.keys()), desc="年份进度"):
        G = networks[year]
        logger.info(f"📅 处理{year}年网络 ({G.number_of_nodes()}节点, {G.number_of_edges()}边)")
        
        for country in countries:
            if country in G.nodes():
                resilience_scores = calculator.calculate_topological_resilience(G, country, year)
                
                result_row = {
                    'year': year,
                    'country': country,
                    **resilience_scores
                }
                results.append(result_row)
            else:
                logger.debug(f"⚠️ {country}不在{year}年网络中")
    
    df = pd.DataFrame(results)
    logger.info(f"✅ 拓扑抗毁性计算完成: {len(df)}条记录")
    
    return df

def calculate_supply_absorption(networks: Dict[int, nx.Graph],
                              countries: List[str] = None,
                              top_suppliers: int = 3) -> pd.DataFrame:
    """
    批量计算供应缺口吸收率
    
    Args:
        networks: 年份到网络图的映射
        countries: 要分析的国家列表
        top_suppliers: 考虑的主要供应商数量
        
    Returns:
        包含供应吸收能力数据的DataFrame
    """
    
    logger.info("📦 开始批量计算供应缺口吸收率...")
    
    calculator = NetworkResilienceCalculator()
    results = []
    
    # 确定要分析的国家
    if countries is None:
        all_countries = set()
        for G in networks.values():
            all_countries.update(G.nodes())
        countries = sorted(list(all_countries))
    
    # 逐年逐国分析
    for year in tqdm(sorted(networks.keys()), desc="年份进度"):
        G = networks[year]
        logger.info(f"📅 处理{year}年网络 ({G.number_of_nodes()}节点, {G.number_of_edges()}边)")
        
        for country in countries:
            if country in G.nodes():
                absorption_scores = calculator.calculate_supply_absorption(
                    G, country, year, top_suppliers
                )
                
                result_row = {
                    'year': year,
                    'country': country,
                    **absorption_scores
                }
                results.append(result_row)
            else:
                logger.debug(f"⚠️ {country}不在{year}年网络中")
    
    df = pd.DataFrame(results)
    logger.info(f"✅ 供应缺口吸收率计算完成: {len(df)}条记录")
    
    return df

def generate_resilience_database(networks: Dict[int, nx.Graph],
                               output_path: str = "network_resilience.csv",
                               countries: List[str] = None) -> pd.DataFrame:
    """
    生成完整的网络韧性数据库
    
    Args:
        networks: 年份到网络图的映射
        output_path: 输出文件路径
        countries: 要分析的国家列表
        
    Returns:
        完整的韧性数据库DataFrame
    """
    
    logger.info("🗃️ 生成网络韧性数据库...")
    
    # 计算拓扑抗毁性
    topo_resilience = calculate_topological_resilience(networks, countries)
    
    # 计算供应缺口吸收率
    supply_absorption = calculate_supply_absorption(networks, countries)
    
    # 合并两类韧性指标
    resilience_db = pd.merge(
        topo_resilience, 
        supply_absorption, 
        on=['year', 'country'], 
        how='outer'
    )
    
    # 计算综合韧性指数（两个维度的加权平均）
    resilience_db['comprehensive_resilience'] = (
        0.6 * resilience_db['topological_resilience_avg'] + 
        0.4 * resilience_db['supply_absorption_rate']
    )
    
    # 排序并保存
    resilience_db = resilience_db.sort_values(['year', 'country'])
    
    # 保存到文件
    output_file = Path(output_path)
    resilience_db.to_csv(output_file, index=False)
    
    logger.info(f"✅ 网络韧性数据库已保存: {output_file}")
    logger.info(f"   数据维度: {resilience_db.shape}")
    logger.info(f"   年份范围: {resilience_db['year'].min()}-{resilience_db['year'].max()}")
    logger.info(f"   国家数量: {resilience_db['country'].nunique()}")
    
    return resilience_db

# 为main.py提供的简化接口
class SimpleResilienceCalculator:
    """简化的韧性计算器，专注于批量计算"""
    
    def __init__(self):
        self.calculator = NetworkResilienceCalculator()
        
    def calculate_resilience_for_all(self, networks: Dict[int, nx.Graph]) -> pd.DataFrame:
        """
        为所有网络计算韧性指标
        
        Args:
            networks: 年份到网络图的映射
            
        Returns:
            包含所有韧性指标的DataFrame
        """
        return generate_resilience_database(networks)

if __name__ == "__main__":
    # 测试代码
    logger.info("🧪 测试网络韧性计算器...")
    
    # 创建测试网络
    G_test = nx.DiGraph()
    
    # 添加测试边（模拟能源贸易网络）
    edges = [
        ('USA', 'CHN', {'weight': 100}), 
        ('USA', 'JPN', {'weight': 80}),
        ('USA', 'DEU', {'weight': 60}),
        ('RUS', 'CHN', {'weight': 90}),
        ('RUS', 'DEU', {'weight': 70}),
        ('SAU', 'CHN', {'weight': 85}),
        ('SAU', 'JPN', {'weight': 75}),
        ('CAN', 'USA', {'weight': 95}),
        ('MEX', 'USA', {'weight': 50})
    ]
    
    G_test.add_edges_from([(s, t, d) for s, t, d in edges])
    
    # 测试韧性计算
    calculator = NetworkResilienceCalculator()
    
    # 测试中国的韧性
    topo_resilience = calculator.calculate_topological_resilience(G_test, 'CHN')
    supply_absorption = calculator.calculate_supply_absorption(G_test, 'CHN')
    
    print("\n🇨🇳 中国韧性测试结果:")
    print("拓扑抗毁性:")
    for key, value in topo_resilience.items():
        print(f"  {key}: {value:.3f}")
        
    print("\n供应缺口吸收率:")  
    for key, value in supply_absorption.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n🎉 测试完成!")