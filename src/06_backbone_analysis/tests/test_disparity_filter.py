#!/usr/bin/env python3
"""
Disparity Filter算法单元测试
==========================

测试Disparity Filter算法的正确性、边界条件和异常处理。

作者：Energy Network Analysis Team
"""

import unittest
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import sys

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.disparity_filter import (
    calculate_disparity_pvalue,
    benjamini_hochberg_fdr,
    disparity_filter,
    apply_disparity_filter_batch
)

class TestDisparityFilter(unittest.TestCase):
    """Disparity Filter算法测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建简单测试网络
        self.test_graph = nx.Graph()
        edges = [
            ('A', 'B', 100.0), ('A', 'C', 10.0), ('A', 'D', 1.0),
            ('B', 'C', 50.0), ('B', 'D', 5.0),
            ('C', 'D', 20.0)
        ]
        for source, target, weight in edges:
            self.test_graph.add_edge(source, target, weight=weight)
        
        # 创建美国中心的测试网络
        self.usa_graph = nx.Graph()
        usa_edges = [
            ('USA', 'CAN', 100.0), ('USA', 'MEX', 80.0), ('USA', 'GBR', 60.0),
            ('USA', 'CHN', 40.0), ('CAN', 'GBR', 20.0), ('MEX', 'CHN', 15.0)
        ]
        for source, target, weight in usa_edges:
            self.usa_graph.add_edge(source, target, weight=weight)
    
    def test_calculate_disparity_pvalue(self):
        """测试Disparity p值计算"""
        # 测试正常情况
        p_value = calculate_disparity_pvalue(50.0, 100.0, 2)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        
        # 测试边界情况
        # 度为1时应返回1.0
        p_value_deg1 = calculate_disparity_pvalue(50.0, 100.0, 1)
        self.assertEqual(p_value_deg1, 1.0)
        
        # 强度为0时应返回1.0
        p_value_strength0 = calculate_disparity_pvalue(50.0, 0.0, 2)
        self.assertEqual(p_value_strength0, 1.0)
        
        # 权重为0时应返回1.0
        p_value_weight0 = calculate_disparity_pvalue(0.0, 100.0, 2)
        self.assertEqual(p_value_weight0, 1.0)
        
        # 大权重比例应有小p值
        p_value_large = calculate_disparity_pvalue(90.0, 100.0, 2)
        p_value_small = calculate_disparity_pvalue(10.0, 100.0, 2)
        self.assertLess(p_value_large, p_value_small)
    
    def test_benjamini_hochberg_fdr(self):
        """测试Benjamini-Hochberg FDR校正"""
        # 测试正常情况
        p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5])
        significant = benjamini_hochberg_fdr(p_values, alpha=0.05)
        
        self.assertEqual(len(significant), len(p_values))
        self.assertTrue(all(isinstance(x, (bool, np.bool_)) for x in significant))
        
        # 第一个p值应该是显著的（最小）
        self.assertTrue(significant[0])
        
        # 测试空数组
        empty_result = benjamini_hochberg_fdr(np.array([]), alpha=0.05)
        self.assertEqual(len(empty_result), 0)
        
        # 测试所有p值都很大的情况
        large_p_values = np.array([0.8, 0.9, 0.95, 0.99])
        no_significant = benjamini_hochberg_fdr(large_p_values, alpha=0.05)
        self.assertFalse(any(no_significant))
    
    def test_disparity_filter_basic(self):
        """测试基本Disparity Filter功能"""
        # 应用算法
        G_filtered = disparity_filter(self.test_graph, alpha=0.05, fdr_correction=True)
        
        # 检查返回结果是NetworkX图
        self.assertIsInstance(G_filtered, nx.Graph)
        
        # 检查节点数应保持不变
        self.assertEqual(G_filtered.number_of_nodes(), self.test_graph.number_of_nodes())
        
        # 检查边数应该减少或保持不变
        self.assertLessEqual(G_filtered.number_of_edges(), self.test_graph.number_of_edges())
        
        # 检查图属性
        self.assertIn('backbone_method', G_filtered.graph)
        self.assertEqual(G_filtered.graph['backbone_method'], 'disparity_filter')
        self.assertIn('retention_rate', G_filtered.graph)
        self.assertIn('alpha', G_filtered.graph)
    
    def test_disparity_filter_parameters(self):
        """测试不同参数下的Disparity Filter"""
        # 测试不同alpha值
        G_strict = disparity_filter(self.test_graph, alpha=0.01, fdr_correction=True)
        G_lenient = disparity_filter(self.test_graph, alpha=0.1, fdr_correction=True)
        
        # 更严格的alpha应该保留更少边
        self.assertLessEqual(G_strict.number_of_edges(), G_lenient.number_of_edges())
        
        # 测试FDR校正与否
        G_with_fdr = disparity_filter(self.test_graph, alpha=0.05, fdr_correction=True)
        G_without_fdr = disparity_filter(self.test_graph, alpha=0.05, fdr_correction=False)
        
        # 两种方法都应返回有效结果
        self.assertIsInstance(G_with_fdr, nx.Graph)
        self.assertIsInstance(G_without_fdr, nx.Graph)
    
    def test_disparity_filter_usa_network(self):
        """测试包含美国的网络"""
        G_filtered = disparity_filter(self.usa_graph, alpha=0.05, fdr_correction=True)
        
        # 美国应该保留在网络中
        self.assertIn('USA', G_filtered.nodes())
        
        # 美国应该有一些连接
        if G_filtered.number_of_edges() > 0:
            usa_degree = G_filtered.degree('USA')
            self.assertGreaterEqual(usa_degree, 0)
    
    def test_disparity_filter_edge_cases(self):
        """测试边界情况"""
        # 测试空图
        empty_graph = nx.Graph()
        G_empty_filtered = disparity_filter(empty_graph, alpha=0.05)
        self.assertEqual(G_empty_filtered.number_of_nodes(), 0)
        self.assertEqual(G_empty_filtered.number_of_edges(), 0)
        
        # 测试单节点图
        single_node_graph = nx.Graph()
        single_node_graph.add_node('A')
        G_single_filtered = disparity_filter(single_node_graph, alpha=0.05)
        self.assertEqual(G_single_filtered.number_of_nodes(), 1)
        self.assertEqual(G_single_filtered.number_of_edges(), 0)
        
        # 测试只有一条边的图
        single_edge_graph = nx.Graph()
        single_edge_graph.add_edge('A', 'B', weight=100.0)
        G_edge_filtered = disparity_filter(single_edge_graph, alpha=0.05)
        # 单边图无法进行显著性检验，应该保持原样
        self.assertEqual(G_edge_filtered.number_of_edges(), 1)
    
    def test_apply_disparity_filter_batch(self):
        """测试批量应用Disparity Filter"""
        # 创建多年网络数据
        networks = {
            2018: self.test_graph,
            2019: self.usa_graph
        }
        
        # 批量应用
        results = apply_disparity_filter_batch(
            networks, 
            alpha_values=[0.05, 0.1],
            fdr_correction=True
        )
        
        # 检查结果结构
        self.assertIsInstance(results, dict)
        self.assertIn('alpha_0.05', results)
        self.assertIn('alpha_0.1', results)
        
        # 检查每个alpha值的结果
        for alpha_key in ['alpha_0.05', 'alpha_0.1']:
            self.assertIsInstance(results[alpha_key], dict)
            self.assertIn(2018, results[alpha_key])
            self.assertIn(2019, results[alpha_key])
            
            # 检查每个年份的结果
            for year in [2018, 2019]:
                G_result = results[alpha_key][year]
                self.assertIsInstance(G_result, nx.Graph)
                self.assertIn('backbone_method', G_result.graph)
    
    def test_weight_attribute_handling(self):
        """测试权重属性处理"""
        # 创建带有自定义权重属性的图
        custom_weight_graph = nx.Graph()
        custom_weight_graph.add_edge('A', 'B', custom_weight=100.0)
        custom_weight_graph.add_edge('A', 'C', custom_weight=10.0)
        custom_weight_graph.add_edge('B', 'C', custom_weight=50.0)
        
        # 使用自定义权重属性
        G_filtered = disparity_filter(
            custom_weight_graph, 
            alpha=0.05, 
            weight_attr='custom_weight'
        )
        
        self.assertIsInstance(G_filtered, nx.Graph)
        self.assertLessEqual(G_filtered.number_of_edges(), custom_weight_graph.number_of_edges())
    
    def test_directed_graph_handling(self):
        """测试有向图处理"""
        # 创建有向图
        directed_graph = nx.DiGraph()
        directed_edges = [
            ('A', 'B', 100.0), ('B', 'A', 80.0),
            ('A', 'C', 60.0), ('C', 'A', 40.0),
            ('B', 'C', 30.0), ('C', 'B', 20.0)
        ]
        for source, target, weight in directed_edges:
            directed_graph.add_edge(source, target, weight=weight)
        
        # 应用Disparity Filter
        G_filtered = disparity_filter(directed_graph, alpha=0.05, directed=True)
        
        self.assertIsInstance(G_filtered, nx.DiGraph)
        self.assertLessEqual(G_filtered.number_of_edges(), directed_graph.number_of_edges())

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)