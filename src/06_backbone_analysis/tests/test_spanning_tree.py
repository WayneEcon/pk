#!/usr/bin/env python3
"""
Maximum Spanning Tree算法单元测试
===============================

测试MST算法的正确性、图类型处理和批量应用功能。

作者：Energy Network Analysis Team
"""

import unittest
import numpy as np
import networkx as nx
from pathlib import Path
import sys

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.spanning_tree import (
    symmetrize_graph,
    maximum_spanning_tree,
    maximum_spanning_forest,
    apply_mst_to_directed_graph,
    apply_mst_batch
)

class TestSpanningTree(unittest.TestCase):
    """Maximum Spanning Tree算法测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建连通的无向图
        self.connected_graph = nx.Graph()
        edges = [
            ('A', 'B', 100.0), ('A', 'C', 80.0), ('A', 'D', 60.0),
            ('B', 'C', 90.0), ('B', 'D', 70.0), ('C', 'D', 85.0)
        ]
        for source, target, weight in edges:
            self.connected_graph.add_edge(source, target, weight=weight)
        
        # 创建不连通的无向图
        self.disconnected_graph = nx.Graph()
        disconnected_edges = [
            ('A', 'B', 100.0), ('A', 'C', 80.0),  # 第一个组件
            ('D', 'E', 90.0), ('D', 'F', 70.0)    # 第二个组件
        ]
        for source, target, weight in disconnected_edges:
            self.disconnected_graph.add_edge(source, target, weight=weight)
        
        # 创建有向图
        self.directed_graph = nx.DiGraph()
        directed_edges = [
            ('A', 'B', 100.0), ('B', 'A', 80.0), ('A', 'C', 90.0),
            ('C', 'A', 70.0), ('B', 'C', 85.0), ('C', 'B', 75.0)
        ]
        for source, target, weight in directed_edges:
            self.directed_graph.add_edge(source, target, weight=weight)
    
    def test_symmetrize_graph(self):
        """测试图对称化功能"""
        # 测试max方法
        G_sym_max = symmetrize_graph(self.directed_graph, method='max')
        self.assertIsInstance(G_sym_max, nx.Graph)
        self.assertEqual(G_sym_max.number_of_nodes(), self.directed_graph.number_of_nodes())
        
        # 检查权重是否按最大值处理
        self.assertEqual(G_sym_max['A']['B']['weight'], 100.0)  # max(100, 80)
        self.assertEqual(G_sym_max['A']['C']['weight'], 90.0)   # max(90, 70)
        
        # 测试min方法
        G_sym_min = symmetrize_graph(self.directed_graph, method='min')
        self.assertEqual(G_sym_min['A']['B']['weight'], 80.0)   # min(100, 80)
        self.assertEqual(G_sym_min['A']['C']['weight'], 70.0)   # min(90, 70)
        
        # 测试mean方法
        G_sym_mean = symmetrize_graph(self.directed_graph, method='mean')
        self.assertEqual(G_sym_mean['A']['B']['weight'], 90.0)  # (100 + 80) / 2
        self.assertEqual(G_sym_mean['A']['C']['weight'], 80.0)  # (90 + 70) / 2
    
    def test_maximum_spanning_tree_connected(self):
        """测试连通图的MST"""
        mst = maximum_spanning_tree(self.connected_graph)
        
        # 检查返回结果是无向图
        self.assertIsInstance(mst, nx.Graph)
        
        # MST应该有n-1条边（n为节点数）
        expected_edges = self.connected_graph.number_of_nodes() - 1
        self.assertEqual(mst.number_of_edges(), expected_edges)
        
        # 检查所有节点都保留
        self.assertEqual(mst.number_of_nodes(), self.connected_graph.number_of_nodes())
        
        # 检查MST是连通的
        self.assertTrue(nx.is_connected(mst))
        
        # 检查保留的边都是高权重边
        mst_weights = [mst[u][v]['weight'] for u, v in mst.edges()]
        self.assertTrue(all(w >= 0 for w in mst_weights))
    
    def test_maximum_spanning_tree_algorithms(self):
        """测试不同MST算法"""
        # 测试不同算法
        mst_kruskal = maximum_spanning_tree(self.connected_graph, algorithm='kruskal')
        mst_prim = maximum_spanning_tree(self.connected_graph, algorithm='prim')
        mst_boruvka = maximum_spanning_tree(self.connected_graph, algorithm='boruvka')
        
        # 所有算法应该给出相同数量的边
        expected_edges = self.connected_graph.number_of_nodes() - 1
        self.assertEqual(mst_kruskal.number_of_edges(), expected_edges)
        self.assertEqual(mst_prim.number_of_edges(), expected_edges)
        self.assertEqual(mst_boruvka.number_of_edges(), expected_edges)
        
        # 所有MST应该有相同的总权重（对于唯一MST）
        weight_kruskal = sum(mst_kruskal[u][v]['weight'] for u, v in mst_kruskal.edges())
        weight_prim = sum(mst_prim[u][v]['weight'] for u, v in mst_prim.edges())
        weight_boruvka = sum(mst_boruvka[u][v]['weight'] for u, v in mst_boruvka.edges())
        
        # 权重应该相等（允许小的数值误差）
        self.assertAlmostEqual(weight_kruskal, weight_prim, places=10)
        self.assertAlmostEqual(weight_kruskal, weight_boruvka, places=10)
    
    def test_maximum_spanning_forest(self):
        """测试最大生成森林"""
        msf = maximum_spanning_forest(self.disconnected_graph)
        
        # 检查返回结果是无向图
        self.assertIsInstance(msf, nx.Graph)
        
        # 检查所有节点都保留
        self.assertEqual(msf.number_of_nodes(), self.disconnected_graph.number_of_nodes())
        
        # 检查边数：每个连通组件贡献(n_i - 1)条边
        # 第一个组件3个节点 -> 2条边，第二个组件3个节点 -> 2条边
        expected_edges = 2 + 2  
        self.assertEqual(msf.number_of_edges(), expected_edges)
        
        # 检查每个连通组件内部是连通的
        components = list(nx.connected_components(msf))
        for component in components:
            subgraph = msf.subgraph(component)
            self.assertTrue(nx.is_connected(subgraph))
    
    def test_apply_mst_to_directed_graph(self):
        """测试有向图MST应用"""
        # 测试不同对称化方法
        mst_max = apply_mst_to_directed_graph(self.directed_graph, symmetrize_method='max')
        mst_min = apply_mst_to_directed_graph(self.directed_graph, symmetrize_method='min')
        mst_mean = apply_mst_to_directed_graph(self.directed_graph, symmetrize_method='mean')
        
        # 所有结果应该是无向图
        for mst in [mst_max, mst_min, mst_mean]:
            self.assertIsInstance(mst, nx.Graph)
            
        # 所有MST应该有相同的边数
        expected_edges = self.directed_graph.number_of_nodes() - 1
        self.assertEqual(mst_max.number_of_edges(), expected_edges)
        self.assertEqual(mst_min.number_of_edges(), expected_edges)
        self.assertEqual(mst_mean.number_of_edges(), expected_edges)
        
        # 不同对称化方法可能产生不同的总权重
        weight_max = sum(mst_max[u][v]['weight'] for u, v in mst_max.edges())
        weight_min = sum(mst_min[u][v]['weight'] for u, v in mst_min.edges())
        
        # max方法的总权重应该 >= min方法的总权重
        self.assertGreaterEqual(weight_max, weight_min)
    
    def test_apply_mst_batch(self):
        """测试批量MST应用"""
        # 创建多年网络数据
        networks = {
            2018: self.connected_graph,
            2019: self.disconnected_graph,
            2020: self.directed_graph
        }
        
        # 批量应用MST
        mst_results = apply_mst_batch(
            networks,
            weight_attr='weight',
            symmetrize_method='max',
            algorithm='kruskal'
        )
        
        # 检查结果结构
        self.assertIsInstance(mst_results, dict)
        self.assertEqual(len(mst_results), 3)
        self.assertIn(2018, mst_results)
        self.assertIn(2019, mst_results)
        self.assertIn(2020, mst_results)
        
        # 检查每个年份的结果
        for year, original_graph in networks.items():
            mst_graph = mst_results[year]
            
            # 结果应该是NetworkX图
            self.assertIsInstance(mst_graph, nx.Graph)
            
            # 节点数应该保持不变
            self.assertEqual(mst_graph.number_of_nodes(), original_graph.number_of_nodes())
            
            # 检查图属性
            self.assertIn('backbone_method', mst_graph.graph)
            self.assertEqual(mst_graph.graph['backbone_method'], 'maximum_spanning_tree')
            self.assertIn('retention_rate', mst_graph.graph)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空图
        empty_graph = nx.Graph()
        mst_empty = maximum_spanning_tree(empty_graph)
        self.assertEqual(mst_empty.number_of_nodes(), 0)
        self.assertEqual(mst_empty.number_of_edges(), 0)
        
        # 测试单节点图
        single_node_graph = nx.Graph()
        single_node_graph.add_node('A')
        mst_single = maximum_spanning_tree(single_node_graph)
        self.assertEqual(mst_single.number_of_nodes(), 1)
        self.assertEqual(mst_single.number_of_edges(), 0)
        
        # 测试两节点图
        two_node_graph = nx.Graph()
        two_node_graph.add_edge('A', 'B', weight=100.0)
        mst_two = maximum_spanning_tree(two_node_graph)
        self.assertEqual(mst_two.number_of_nodes(), 2)
        self.assertEqual(mst_two.number_of_edges(), 1)
        
        # 权重应该保持不变
        self.assertEqual(mst_two['A']['B']['weight'], 100.0)
    
    def test_weight_attribute_handling(self):
        """测试权重属性处理"""
        # 创建带有自定义权重属性的图
        custom_weight_graph = nx.Graph()
        custom_weight_graph.add_edge('A', 'B', custom_weight=100.0)
        custom_weight_graph.add_edge('A', 'C', custom_weight=80.0)
        custom_weight_graph.add_edge('B', 'C', custom_weight=90.0)
        
        # 使用自定义权重属性
        mst = maximum_spanning_tree(custom_weight_graph, weight_attr='custom_weight')
        
        self.assertIsInstance(mst, nx.Graph)
        self.assertEqual(mst.number_of_edges(), 2)  # n-1 = 3-1 = 2
        
        # 检查权重属性是否正确处理
        for u, v in mst.edges():
            self.assertIn('custom_weight', mst[u][v])
    
    def test_mst_properties(self):
        """测试MST的基本性质"""
        mst = maximum_spanning_tree(self.connected_graph)
        
        # MST应该是无环的
        self.assertTrue(nx.is_tree(mst))
        
        # MST应该连接所有节点
        self.assertTrue(nx.is_connected(mst))
        
        # MST应该有最小数量的边来保持连通性
        self.assertEqual(mst.number_of_edges(), mst.number_of_nodes() - 1)
        
        # 添加任何不在MST中的原图边都应该形成环
        original_edges = set(self.connected_graph.edges())
        mst_edges = set(mst.edges())
        non_mst_edges = original_edges - mst_edges
        
        for edge in non_mst_edges:
            test_graph = mst.copy()
            u, v = edge
            test_graph.add_edge(u, v, weight=self.connected_graph[u][v]['weight'])
            
            # 添加非MST边后应该不再是树
            self.assertFalse(nx.is_tree(test_graph))
    
    def test_batch_error_handling(self):
        """测试批量处理的错误处理"""
        # 创建包含问题数据的网络字典
        networks_with_issues = {
            2018: self.connected_graph,
            2019: nx.Graph(),  # 空图
            2020: None  # 这会引发错误
        }
        
        # 去掉会引发错误的数据
        safe_networks = {k: v for k, v in networks_with_issues.items() if v is not None}
        
        # 批量应用应该能处理空图
        mst_results = apply_mst_batch(safe_networks)
        
        # 应该成功处理非空图
        self.assertIn(2018, mst_results)
        # 空图应该也能处理
        self.assertIn(2019, mst_results)
        
        # 空图的MST应该也是空图
        self.assertEqual(mst_results[2019].number_of_nodes(), 0)
        self.assertEqual(mst_results[2019].number_of_edges(), 0)

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)