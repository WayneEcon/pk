#!/usr/bin/env python3
"""
node_metrics模块单元测试
"""

import unittest
import networkx as nx
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from node_metrics import (
    calculate_degree_centrality, calculate_strength_centrality,
    calculate_betweenness_centrality, calculate_pagerank_centrality,
    calculate_eigenvector_centrality, calculate_all_node_centralities,
    get_node_centrality_rankings, get_node_centrality_summary
)


class TestNodeMetricsBasicFunctions(unittest.TestCase):
    """测试基本节点指标计算函数"""
    
    def setUp(self):
        """设置测试用的图"""
        self.simple_graph = nx.DiGraph()
        self.simple_graph.add_edge('A', 'B', weight=100.0)
        self.simple_graph.add_edge('B', 'C', weight=200.0)
        self.simple_graph.add_edge('C', 'A', weight=50.0)
        self.year = 2020
        
        # 更复杂的图用于测试
        self.complex_graph = nx.DiGraph()
        nodes = ['USA', 'CHN', 'DEU', 'JPN', 'GBR']
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i != j:
                    weight = (i + 1) * (j + 1) * 10
                    self.complex_graph.add_edge(source, target, weight=weight)
    
    def test_calculate_degree_centrality_basic(self):
        """测试度中心性基本计算"""
        result = calculate_degree_centrality(self.simple_graph, self.year)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # 3个节点
        self.assertTrue(all(col in result.columns for col in 
                           ['year', 'country_code', 'in_degree', 'out_degree', 'total_degree']))
        
        # 检查具体值
        a_row = result[result['country_code'] == 'A'].iloc[0]
        self.assertEqual(a_row['out_degree'], 1)  # A -> B
        self.assertEqual(a_row['in_degree'], 1)   # C -> A
        self.assertEqual(a_row['total_degree'], 2)
    
    def test_calculate_strength_centrality_basic(self):
        """测试强度中心性基本计算"""
        result = calculate_strength_centrality(self.simple_graph, self.year)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        
        # 检查权重计算
        a_row = result[result['country_code'] == 'A'].iloc[0]
        self.assertEqual(a_row['out_strength'], 100.0)  # A -> B (100)
        self.assertEqual(a_row['in_strength'], 50.0)    # C -> A (50)
        self.assertEqual(a_row['total_strength'], 150.0)
    
    def test_calculate_betweenness_centrality_basic(self):
        """测试中介中心性基本计算"""
        result = calculate_betweenness_centrality(self.simple_graph, self.year)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn('betweenness_centrality', result.columns)
        
        # 在环形图中，所有节点的中介中心性应该相等
        betweenness_values = result['betweenness_centrality'].values
        self.assertTrue(all(v >= 0 for v in betweenness_values))
    
    def test_calculate_pagerank_centrality_basic(self):
        """测试PageRank中心性基本计算"""
        result = calculate_pagerank_centrality(self.simple_graph, self.year)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn('pagerank_centrality', result.columns)
        
        # PageRank值之和应该接近1
        pagerank_sum = result['pagerank_centrality'].sum()
        self.assertAlmostEqual(pagerank_sum, 1.0, places=5)
        
        # 所有值应该为正
        self.assertTrue(all(result['pagerank_centrality'] > 0))
    
    def test_calculate_eigenvector_centrality_basic(self):
        """测试特征向量中心性基本计算"""
        result = calculate_eigenvector_centrality(self.complex_graph, self.year)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertIn('eigenvector_centrality', result.columns)
        
        # 所有值应该非负
        self.assertTrue(all(result['eigenvector_centrality'] >= 0))


class TestNodeMetricsEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_single_node_graph(self):
        """测试单节点图"""
        single_node_graph = nx.DiGraph()
        single_node_graph.add_node('A')
        
        result = calculate_degree_centrality(single_node_graph, 2020)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['in_degree'], 0)
        self.assertEqual(result.iloc[0]['out_degree'], 0)
    
    def test_disconnected_graph(self):
        """测试不连通图"""
        disconnected_graph = nx.DiGraph()
        disconnected_graph.add_edge('A', 'B', weight=100)
        disconnected_graph.add_edge('C', 'D', weight=200)  # 独立的组件
        
        result = calculate_betweenness_centrality(disconnected_graph, 2020)
        
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_self_loop_graph(self):
        """测试自循环图"""
        self_loop_graph = nx.DiGraph()
        self_loop_graph.add_edge('A', 'A', weight=100)  # 自循环
        self_loop_graph.add_edge('A', 'B', weight=200)
        
        result = calculate_strength_centrality(self_loop_graph, 2020)
        
        # 自循环应该被正确处理
        a_row = result[result['country_code'] == 'A'].iloc[0]
        self.assertEqual(a_row['out_strength'], 300.0)  # 100 + 200
    
    def test_zero_weight_edges(self):
        """测试零权重边"""
        zero_weight_graph = nx.DiGraph()
        zero_weight_graph.add_edge('A', 'B', weight=0.0)
        zero_weight_graph.add_edge('B', 'C', weight=100.0)
        
        # 应该会处理零权重（转换为最小权重）
        result = calculate_strength_centrality(zero_weight_graph, 2020)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)


class TestNodeMetricsErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def test_invalid_graph_type(self):
        """测试无效图类型"""
        with self.assertRaises(TypeError):
            calculate_degree_centrality("not a graph", 2020)
    
    def test_empty_graph(self):
        """测试空图"""
        empty_graph = nx.DiGraph()
        
        with self.assertRaises(ValueError):
            calculate_degree_centrality(empty_graph, 2020)
    
    def test_missing_weights(self):
        """测试缺少权重的图"""
        no_weight_graph = nx.DiGraph()
        no_weight_graph.add_edge('A', 'B')  # 没有weight属性
        
        with self.assertRaises(ValueError):
            calculate_strength_centrality(no_weight_graph, 2020)


class TestCalculateAllNodeCentralities(unittest.TestCase):
    """测试综合节点中心性计算"""
    
    def setUp(self):
        self.test_graph = nx.DiGraph()
        self.test_graph.add_edge('USA', 'CHN', weight=1000)
        self.test_graph.add_edge('CHN', 'DEU', weight=800)
        self.test_graph.add_edge('DEU', 'USA', weight=600)
        self.test_graph.add_edge('JPN', 'USA', weight=400)
        
    def test_calculate_all_node_centralities_success(self):
        """测试成功计算所有节点中心性"""
        result = calculate_all_node_centralities(self.test_graph, 2020)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 4个节点
        
        # 检查必要列存在
        expected_columns = [
            'year', 'country_code', 'in_degree', 'out_degree', 'total_degree',
            'in_strength', 'out_strength', 'total_strength',
            'betweenness_centrality', 'pagerank_centrality', 'eigenvector_centrality'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")
        
        # 检查年份正确
        self.assertTrue(all(result['year'] == 2020))
        
        # 检查数值有效性
        self.assertTrue(all(result['total_degree'] >= 0))
        self.assertTrue(all(result['total_strength'] >= 0))
        self.assertTrue(all(result['betweenness_centrality'] >= 0))
        self.assertTrue(all(result['pagerank_centrality'] > 0))
        self.assertTrue(all(result['eigenvector_centrality'] >= 0))
    
    def test_calculate_all_with_computation_error(self):
        """测试部分计算失败的情况"""
        # 创建一个可能导致特征向量中心性计算失败的图
        problematic_graph = nx.DiGraph()
        problematic_graph.add_edge('A', 'B', weight=1)
        
        with patch('node_metrics.calculate_eigenvector_centrality', 
                  side_effect=Exception("Mock error")):
            
            result = calculate_all_node_centralities(problematic_graph, 2020)
            
            # 应该仍然返回DataFrame，但eigenvector_centrality为0
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)


class TestNodeMetricsRankingsAndSummary(unittest.TestCase):
    """测试节点指标排名和摘要功能"""
    
    def setUp(self):
        # 创建测试DataFrame
        data = {
            'year': [2020] * 5,
            'country_code': ['USA', 'CHN', 'DEU', 'JPN', 'GBR'],
            'country_name': ['United States', 'China', 'Germany', 'Japan', 'United Kingdom'],
            'total_degree': [10, 8, 6, 4, 2],
            'total_strength': [1000, 800, 600, 400, 200],
            'betweenness_centrality': [0.5, 0.4, 0.3, 0.2, 0.1],
            'pagerank_centrality': [0.3, 0.25, 0.2, 0.15, 0.1],
            'eigenvector_centrality': [0.4, 0.3, 0.2, 0.1, 0.0]
        }
        self.test_df = pd.DataFrame(data)
    
    def test_get_node_centrality_rankings(self):
        """测试节点中心性排名"""
        rankings = get_node_centrality_rankings(self.test_df, top_k=3)
        
        self.assertIsInstance(rankings, dict)
        
        # 检查包含的指标
        expected_metrics = [
            'total_degree', 'total_strength', 'betweenness_centrality',
            'pagerank_centrality', 'eigenvector_centrality'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, rankings)
            self.assertEqual(len(rankings[metric]), 3)  # top_k=3
            
            # 检查排名是按降序排列
            values = [item[metric] for item in rankings[metric]]
            self.assertEqual(values, sorted(values, reverse=True))
    
    def test_get_node_centrality_summary(self):
        """测试节点中心性摘要"""
        summary = get_node_centrality_summary(self.test_df)
        
        self.assertIsInstance(summary, dict)
        
        # 检查基本统计信息
        self.assertEqual(summary['total_nodes'], 5)
        self.assertEqual(summary['year'], 2020)
        
        # 检查顶级节点识别
        self.assertIn('top_by_total_strength', summary)
        self.assertEqual(summary['top_by_total_strength']['country_code'], 'USA')
        
        # 检查统计量
        self.assertIn('statistics', summary)
        stats = summary['statistics']
        self.assertIn('total_strength', stats)
        self.assertEqual(stats['total_strength']['mean'], 600.0)  # (1000+800+600+400+200)/5


class TestNodeMetricsPerformance(unittest.TestCase):
    """测试性能相关功能"""
    
    def test_large_graph_handling(self):
        """测试大图处理能力"""
        # 创建较大的图进行性能测试
        large_graph = nx.DiGraph()
        
        # 生成100个节点的完全图
        nodes = [f"NODE_{i:03d}" for i in range(100)]
        for i, source in enumerate(nodes[:10]):  # 限制为前10个节点以避免测试太慢
            for j, target in enumerate(nodes[:10]):
                if i != j:
                    large_graph.add_edge(source, target, weight=float(i * j + 1))
        
        result = calculate_all_node_centralities(large_graph, 2020)
        
        self.assertEqual(len(result), 10)
        self.assertIsInstance(result, pd.DataFrame)
    
    @patch('node_metrics.logger')
    def test_logging_functionality(self, mock_logger):
        """测试日志功能"""
        test_graph = nx.DiGraph()
        test_graph.add_edge('A', 'B', weight=100)
        
        calculate_degree_centrality(test_graph, 2020)
        
        # 验证日志被调用
        self.assertTrue(mock_logger.info.called)


if __name__ == '__main__':
    # 配置测试环境，减少日志输出
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestNodeMetricsBasicFunctions,
        TestNodeMetricsEdgeCases, 
        TestNodeMetricsErrorHandling,
        TestCalculateAllNodeCentralities,
        TestNodeMetricsRankingsAndSummary,
        TestNodeMetricsPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*50}")
    print(f"测试摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")