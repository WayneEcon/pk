#!/usr/bin/env python3
"""
算法模块单元测试
===============

测试 algorithms.py 中核心算法的正确性。
确保 DF, MST, PF 算法按预期工作。

测试覆盖：
1. Disparity Filter算法
2. Maximum Spanning Tree算法  
3. Pólya Urn Filter算法
4. 批量处理功能

作者：Energy Network Analysis Team
"""

import unittest
import networkx as nx
import numpy as np
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from algorithms import (
    disparity_filter,
    maximum_spanning_tree,
    polya_urn_filter,
    apply_all_algorithms,
    batch_backbone_extraction,
    calculate_disparity_pvalue,
    benjamini_hochberg_fdr
)


class TestDisparityFilter(unittest.TestCase):
    """测试Disparity Filter算法"""
    
    def setUp(self):
        """设置测试网络"""
        
        # 创建测试网络
        self.G = nx.Graph()
        
        # 添加节点和边
        edges = [
            ('A', 'B', 10.0),   # 强连接
            ('A', 'C', 1.0),    # 弱连接
            ('A', 'D', 0.1),    # 极弱连接
            ('B', 'C', 5.0),    # 中等连接
            ('B', 'D', 2.0),    # 中等偏弱连接
            ('C', 'D', 8.0)     # 强连接
        ]
        
        for u, v, weight in edges:
            self.G.add_edge(u, v, weight=weight)
    
    def test_disparity_pvalue_calculation(self):
        """测试p值计算函数"""
        
        # 测试正常情况
        p_value = calculate_disparity_pvalue(10.0, 50.0, 3)
        self.assertIsInstance(p_value, float)
        self.assertTrue(0 <= p_value <= 1)
        
        # 测试边界情况
        p_value_edge = calculate_disparity_pvalue(1.0, 1.0, 1)
        self.assertEqual(p_value_edge, 1.0)  # 度为1时应返回1.0
        
        # 测试权重为0的情况
        p_value_zero = calculate_disparity_pvalue(0.0, 10.0, 3)
        self.assertEqual(p_value_zero, 1.0)
    
    def test_benjamini_hochberg_fdr(self):
        """测试FDR校正函数"""
        
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20])
        significant = benjamini_hochberg_fdr(p_values, alpha=0.05)
        
        self.assertIsInstance(significant, np.ndarray)
        self.assertEqual(len(significant), len(p_values))
        self.assertTrue(all(isinstance(x, (bool, np.bool_)) for x in significant))
        
        # 应该有一些显著的结果
        self.assertTrue(any(significant))
    
    def test_disparity_filter_basic(self):
        """测试基本DF算法功能"""
        
        # 应用DF算法
        backbone = disparity_filter(self.G, alpha=0.05, fdr_correction=True)
        
        # 检查结果类型
        self.assertIsInstance(backbone, nx.Graph)
        
        # 检查节点保持不变
        self.assertEqual(set(backbone.nodes()), set(self.G.nodes()))
        
        # 检查边数减少（通常情况下）
        self.assertLessEqual(backbone.number_of_edges(), self.G.number_of_edges())
        
        # 检查图属性
        self.assertEqual(backbone.graph['backbone_method'], 'disparity_filter')
        self.assertEqual(backbone.graph['alpha'], 0.05)
        self.assertTrue(backbone.graph['fdr_correction'])
    
    def test_disparity_filter_different_alphas(self):
        """测试不同alpha值的效果"""
        
        backbone_strict = disparity_filter(self.G, alpha=0.01)
        backbone_loose = disparity_filter(self.G, alpha=0.1)
        
        # 更严格的alpha应该保留更少的边
        self.assertLessEqual(
            backbone_strict.number_of_edges(),
            backbone_loose.number_of_edges()
        )
    
    def test_disparity_filter_directed_graph(self):
        """测试有向图的DF算法"""
        
        # 创建有向图
        G_directed = nx.DiGraph()
        G_directed.add_edge('A', 'B', weight=10.0)
        G_directed.add_edge('B', 'A', weight=5.0)
        G_directed.add_edge('A', 'C', weight=2.0)
        
        backbone = disparity_filter(G_directed, alpha=0.05)
        
        self.assertIsInstance(backbone, nx.DiGraph)
        self.assertEqual(set(backbone.nodes()), set(G_directed.nodes()))


class TestMaximumSpanningTree(unittest.TestCase):
    """测试Maximum Spanning Tree算法"""
    
    def setUp(self):
        """设置测试网络"""
        
        self.G = nx.Graph()
        
        # 创建连通图
        edges = [
            ('A', 'B', 10), ('A', 'C', 5), ('A', 'D', 8),
            ('B', 'C', 3), ('B', 'D', 2), ('B', 'E', 7),
            ('C', 'D', 1), ('C', 'E', 6),
            ('D', 'E', 9)
        ]
        
        for u, v, weight in edges:
            self.G.add_edge(u, v, weight=weight)
    
    def test_mst_basic(self):
        """测试基本MST功能"""
        
        mst = maximum_spanning_tree(self.G)
        
        # 检查结果类型
        self.assertIsInstance(mst, nx.Graph)
        
        # MST应该有n-1条边（n为节点数）
        expected_edges = self.G.number_of_nodes() - 1
        self.assertEqual(mst.number_of_edges(), expected_edges)
        
        # 检查连通性
        self.assertTrue(nx.is_connected(mst))
        
        # 检查图属性
        self.assertEqual(mst.graph['backbone_method'], 'maximum_spanning_tree')
        self.assertIn('total_mst_weight', mst.graph)
    
    def test_mst_empty_graph(self):
        """测试空图的MST"""
        
        G_empty = nx.Graph()
        mst = maximum_spanning_tree(G_empty)
        
        self.assertEqual(mst.number_of_nodes(), 0)
        self.assertEqual(mst.number_of_edges(), 0)
    
    def test_mst_single_node(self):
        """测试单节点图的MST"""
        
        G_single = nx.Graph()
        G_single.add_node('A')
        
        mst = maximum_spanning_tree(G_single)
        
        self.assertEqual(mst.number_of_nodes(), 1)
        self.assertEqual(mst.number_of_edges(), 0)
    
    def test_mst_algorithms(self):
        """测试不同MST算法"""
        
        mst_kruskal = maximum_spanning_tree(self.G, algorithm='kruskal')
        mst_prim = maximum_spanning_tree(self.G, algorithm='prim')
        
        # 两种算法应该产生相同的总权重（可能边不同但权重相同）
        self.assertAlmostEqual(
            mst_kruskal.graph['total_mst_weight'],
            mst_prim.graph['total_mst_weight'],
            places=5
        )


class TestPolyaUrnFilter(unittest.TestCase):
    """测试Pólya Urn Filter算法"""
    
    def setUp(self):
        """设置测试网络"""
        
        self.G = nx.Graph()
        
        # 创建测试图
        edges = [
            ('A', 'B', 10), ('A', 'C', 5), ('A', 'D', 2),
            ('B', 'C', 8), ('B', 'D', 3),
            ('C', 'D', 6)
        ]
        
        for u, v, weight in edges:
            self.G.add_edge(u, v, weight=weight)
    
    def test_polya_urn_basic(self):
        """测试基本PF功能"""
        
        backbone = polya_urn_filter(self.G, beta=0.05)
        
        # 检查结果类型
        self.assertIsInstance(backbone, nx.Graph)
        
        # 检查节点保持不变
        self.assertEqual(set(backbone.nodes()), set(self.G.nodes()))
        
        # 检查边数
        self.assertLessEqual(backbone.number_of_edges(), self.G.number_of_edges())
        
        # 检查图属性
        self.assertEqual(backbone.graph['backbone_method'], 'polya_urn_filter')
        self.assertEqual(backbone.graph['beta'], 0.05)


class TestBatchProcessing(unittest.TestCase):
    """测试批量处理功能"""
    
    def setUp(self):
        """设置多年网络数据"""
        
        self.networks = {}
        
        # 创建多年的测试网络
        for year in [2018, 2019, 2020]:
            G = nx.Graph()
            
            # 添加边（每年略有不同）
            np.random.seed(42 + year)
            edges = [
                ('USA', 'CAN', np.random.exponential(50)),
                ('USA', 'CHN', np.random.exponential(40)),
                ('USA', 'GBR', np.random.exponential(30)),
                ('CAN', 'CHN', np.random.exponential(20)),
                ('CAN', 'GBR', np.random.exponential(25)),
                ('CHN', 'GBR', np.random.exponential(35))
            ]
            
            for u, v, weight in edges:
                G.add_edge(u, v, weight=weight)
            
            self.networks[year] = G
    
    def test_apply_all_algorithms(self):
        """测试对单个网络应用所有算法"""
        
        G = self.networks[2018]
        results = apply_all_algorithms(G)
        
        # 检查返回的算法
        expected_algorithms = ['disparity_filter_0.01', 'disparity_filter_0.05', 'disparity_filter_0.1', 'mst', 'polya_urn']
        
        for alg in expected_algorithms:
            self.assertIn(alg, results)
            self.assertIsInstance(results[alg], nx.Graph)
    
    def test_batch_backbone_extraction(self):
        """测试批量骨干提取"""
        
        results = batch_backbone_extraction(self.networks)
        
        # 检查结果结构
        self.assertIsInstance(results, dict)
        
        # 检查每个算法都有结果
        for algorithm_key in results.keys():
            self.assertIsInstance(results[algorithm_key], dict)
            
            # 检查每年都有结果
            for year in self.networks.keys():
                if year in results[algorithm_key]:
                    self.assertIsInstance(results[algorithm_key][year], nx.Graph)
    
    def test_batch_backbone_custom_params(self):
        """测试自定义参数的批量处理"""
        
        results = batch_backbone_extraction(
            self.networks,
            alpha_values=[0.05, 0.1],
            beta=0.1,
            weight_attr='weight'
        )
        
        # 检查DF算法只有指定的alpha值
        df_algorithms = [k for k in results.keys() if k.startswith('disparity_filter_')]
        expected_df_algorithms = ['disparity_filter_0.05', 'disparity_filter_0.1']
        
        for expected_alg in expected_df_algorithms:
            self.assertIn(expected_alg, df_algorithms)


class TestEdgeCases(unittest.TestCase):
    """测试边缘情况"""
    
    def test_single_edge_graph(self):
        """测试只有一条边的图"""
        
        G = nx.Graph()
        G.add_edge('A', 'B', weight=10)
        
        # DF算法应该保留这条边（没有其他边可比较）
        df_result = disparity_filter(G, alpha=0.05)
        self.assertEqual(df_result.number_of_edges(), 1)
        
        # MST算法应该保留这条边
        mst_result = maximum_spanning_tree(G)
        self.assertEqual(mst_result.number_of_edges(), 1)
    
    def test_star_graph(self):
        """测试星形图（一个中心节点连接所有其他节点）"""
        
        G = nx.star_graph(5)  # 创建星形图，中心节点0连接节点1-5
        
        # 添加权重
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.exponential(10)
        
        # 测试DF算法
        df_result = disparity_filter(G, alpha=0.05)
        self.assertIsInstance(df_result, nx.Graph)
        
        # 测试MST算法
        mst_result = maximum_spanning_tree(G)
        self.assertEqual(mst_result.number_of_edges(), 5)  # 星形图的MST就是自己
    
    def test_complete_graph(self):
        """测试完全图"""
        
        G = nx.complete_graph(4)
        
        # 添加随机权重
        np.random.seed(42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.exponential(10)
        
        # 测试所有算法
        df_result = disparity_filter(G, alpha=0.05)
        mst_result = maximum_spanning_tree(G)
        pf_result = polya_urn_filter(G, beta=0.05)
        
        self.assertIsInstance(df_result, nx.Graph)
        self.assertIsInstance(mst_result, nx.Graph)
        self.assertIsInstance(pf_result, nx.Graph)
        
        # MST应该有n-1条边
        self.assertEqual(mst_result.number_of_edges(), 3)


def run_all_tests():
    """运行所有测试"""
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestDisparityFilter,
        TestMaximumSpanningTree,
        TestPolyaUrnFilter,
        TestBatchProcessing,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    if success:
        print("\n✅ 所有算法测试通过!")
    else:
        print("\n❌ 部分测试失败")
        exit(1)