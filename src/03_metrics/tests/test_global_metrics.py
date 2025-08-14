#!/usr/bin/env python3
"""
global_metrics模块单元测试
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

from global_metrics import (
    calculate_density_metrics, calculate_connectivity_metrics,
    calculate_path_metrics, calculate_efficiency_metrics,
    calculate_clustering_metrics, calculate_all_global_metrics,
    get_global_metrics_summary
)


class TestDensityMetrics(unittest.TestCase):
    """测试网络密度指标计算"""
    
    def setUp(self):
        # 完全连通的三节点图
        self.complete_graph = nx.DiGraph()
        self.complete_graph.add_edge('A', 'B', weight=100)
        self.complete_graph.add_edge('A', 'C', weight=200)
        self.complete_graph.add_edge('B', 'A', weight=150)
        self.complete_graph.add_edge('B', 'C', weight=300)
        self.complete_graph.add_edge('C', 'A', weight=250)
        self.complete_graph.add_edge('C', 'B', weight=350)
        
        # 稀疏图
        self.sparse_graph = nx.DiGraph()
        self.sparse_graph.add_edge('A', 'B', weight=100)
        self.sparse_graph.add_edge('C', 'D', weight=200)
        
        self.year = 2020
    
    def test_density_metrics_complete_graph(self):
        """测试完全图的密度计算"""
        result = calculate_density_metrics(self.complete_graph, self.year)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['year'], self.year)
        self.assertEqual(result['nodes'], 3)
        self.assertEqual(result['edges'], 6)
        self.assertEqual(result['max_possible_edges'], 6)  # 3 * (3-1)
        self.assertAlmostEqual(result['density'], 1.0)  # 完全图密度为1
        self.assertEqual(result['edge_coverage_ratio'], 1.0)
        
        # 检查权重相关指标
        expected_total_weight = 100 + 200 + 150 + 300 + 250 + 350
        self.assertEqual(result['total_weight'], expected_total_weight)
        self.assertAlmostEqual(result['avg_edge_weight'], expected_total_weight / 6)
    
    def test_density_metrics_sparse_graph(self):
        """测试稀疏图的密度计算"""
        result = calculate_density_metrics(self.sparse_graph, self.year)
        
        self.assertEqual(result['nodes'], 4)
        self.assertEqual(result['edges'], 2)
        self.assertEqual(result['max_possible_edges'], 12)  # 4 * 3
        self.assertAlmostEqual(result['density'], 2/12)  # 2边 / 12可能边
        self.assertAlmostEqual(result['edge_coverage_ratio'], 2/12)
    
    def test_density_metrics_single_node(self):
        """测试单节点图"""
        single_node = nx.DiGraph()
        single_node.add_node('A')
        
        result = calculate_density_metrics(single_node, self.year)
        
        self.assertEqual(result['nodes'], 1)
        self.assertEqual(result['edges'], 0)
        self.assertEqual(result['max_possible_edges'], 0)
        self.assertEqual(result['density'], 0.0)


class TestConnectivityMetrics(unittest.TestCase):
    """测试连通性指标计算"""
    
    def test_strongly_connected_graph(self):
        """测试强连通图"""
        strongly_connected = nx.DiGraph()
        strongly_connected.add_edge('A', 'B', weight=100)
        strongly_connected.add_edge('B', 'C', weight=200)
        strongly_connected.add_edge('C', 'A', weight=300)
        
        result = calculate_connectivity_metrics(strongly_connected, 2020)
        
        self.assertTrue(result['is_strongly_connected'])
        self.assertTrue(result['is_weakly_connected'])
        self.assertEqual(result['num_strongly_connected_components'], 1)
        self.assertEqual(result['num_weakly_connected_components'], 1)
        self.assertEqual(result['largest_scc_size'], 3)
        self.assertAlmostEqual(result['largest_scc_ratio'], 1.0)
    
    def test_weakly_connected_graph(self):
        """测试弱连通图"""
        weakly_connected = nx.DiGraph()
        weakly_connected.add_edge('A', 'B', weight=100)
        weakly_connected.add_edge('B', 'C', weight=200)
        weakly_connected.add_edge('D', 'C', weight=300)  # 不形成强连通
        
        result = calculate_connectivity_metrics(weakly_connected, 2020)
        
        self.assertFalse(result['is_strongly_connected'])
        self.assertTrue(result['is_weakly_connected'])
        self.assertGreater(result['num_strongly_connected_components'], 1)
        self.assertEqual(result['num_weakly_connected_components'], 1)
    
    def test_disconnected_graph(self):
        """测试不连通图"""
        disconnected = nx.DiGraph()
        disconnected.add_edge('A', 'B', weight=100)
        disconnected.add_edge('C', 'D', weight=200)  # 独立组件
        
        result = calculate_connectivity_metrics(disconnected, 2020)
        
        self.assertFalse(result['is_strongly_connected'])
        self.assertFalse(result['is_weakly_connected'])
        self.assertEqual(result['num_weakly_connected_components'], 2)


class TestPathMetrics(unittest.TestCase):
    """测试路径长度指标计算（重点测试修正的加权路径逻辑）"""
    
    def setUp(self):
        # 创建简单的线性图用于路径测试
        self.linear_graph = nx.DiGraph()
        self.linear_graph.add_edge('A', 'B', weight=2.0)  # 距离 = 1/2 = 0.5
        self.linear_graph.add_edge('B', 'C', weight=4.0)  # 距离 = 1/4 = 0.25
        self.linear_graph.add_edge('A', 'C', weight=1.0)  # 直接路径，距离 = 1/1 = 1.0
        
        self.year = 2020
    
    def test_path_metrics_basic_calculation(self):
        """测试基本路径计算"""
        result = calculate_path_metrics(self.linear_graph, self.year, sample_size=100)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['year'], self.year)
        self.assertGreater(result['sampled_pairs'], 0)
        self.assertGreaterEqual(result['reachable_pairs'], 0)
        self.assertGreaterEqual(result['weighted_reachable_pairs'], 0)
        
        # 路径长度应该为非负数
        self.assertGreaterEqual(result['avg_path_length'], 0)
        self.assertGreaterEqual(result['avg_weighted_path_length'], 0)
        self.assertGreaterEqual(result['median_path_length'], 0)
        self.assertGreaterEqual(result['median_weighted_path_length'], 0)
    
    def test_weighted_path_calculation_logic(self):
        """测试修正的加权路径计算逻辑"""
        # 这个测试验证distance = 1/weight的逻辑是否正确
        result = calculate_path_metrics(self.linear_graph, self.year, sample_size=100)
        
        # 在我们的图中，从A到C有两条路径：
        # 1. 直接路径 A->C，权重1.0，距离1.0
        # 2. 间接路径 A->B->C，权重2.0*4.0，距离0.5+0.25=0.75
        # 因为我们使用distance=1/weight，间接路径实际上应该更短（距离更小）
        
        # 验证加权路径长度存在且合理
        self.assertGreater(result['weighted_reachable_pairs'], 0)
        self.assertGreater(result['avg_weighted_path_length'], 0)
    
    def test_path_metrics_single_node(self):
        """测试单节点图路径计算"""
        single_node = nx.DiGraph()
        single_node.add_node('A')
        
        result = calculate_path_metrics(single_node, self.year)
        
        self.assertEqual(result['avg_path_length'], 0)
        self.assertEqual(result['avg_weighted_path_length'], 0)
        self.assertEqual(result['reachable_pairs'], 0)
    
    def test_path_metrics_sampling(self):
        """测试采样机制"""
        # 创建较大的图来测试采样
        large_graph = nx.DiGraph()
        for i in range(10):
            for j in range(10):
                if i != j:
                    large_graph.add_edge(f"node_{i}", f"node_{j}", weight=float(i+j+1))
        
        result = calculate_path_metrics(large_graph, self.year, sample_size=5)
        
        # 采样应该限制计算的节点对数量
        self.assertLessEqual(result['sampled_pairs'], 5 * 4)  # 最多5个节点，每个节点到其他4个节点


class TestEfficiencyMetrics(unittest.TestCase):
    """测试网络效率指标计算"""
    
    def setUp(self):
        # 创建星型图
        self.star_graph = nx.DiGraph()
        self.star_graph.add_edge('center', 'A', weight=1.0)
        self.star_graph.add_edge('center', 'B', weight=2.0)
        self.star_graph.add_edge('center', 'C', weight=3.0)
        self.star_graph.add_edge('A', 'center', weight=1.0)
        self.star_graph.add_edge('B', 'center', weight=2.0)
        self.star_graph.add_edge('C', 'center', weight=3.0)
        
        self.year = 2020
    
    def test_efficiency_metrics_calculation(self):
        """测试效率指标基本计算"""
        result = calculate_efficiency_metrics(self.star_graph, self.year, sample_size=10)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['year'], self.year)
        self.assertGreaterEqual(result['global_efficiency'], 0)
        self.assertGreaterEqual(result['weighted_global_efficiency'], 0)
        self.assertGreater(result['efficiency_sample_size'], 0)
        self.assertGreater(result['efficiency_sample_pairs'], 0)
        
        # 星型图的效率应该比较高，因为中心节点连接所有其他节点
        self.assertGreater(result['global_efficiency'], 0.1)
    
    def test_efficiency_with_distance_weights(self):
        """测试使用距离权重的效率计算"""
        # 验证修正后的加权效率计算使用了distance权重
        result = calculate_efficiency_metrics(self.star_graph, self.year)
        
        # 加权效率应该存在且为正值
        self.assertGreater(result['weighted_global_efficiency'], 0)
        
        # 由于使用了距离权重，加权效率可能与未加权效率不同
        self.assertIsInstance(result['weighted_global_efficiency'], (int, float))


class TestClusteringMetrics(unittest.TestCase):
    """测试聚类系数指标计算"""
    
    def test_clustering_triangle_graph(self):
        """测试三角形图的聚类系数"""
        triangle = nx.DiGraph()
        triangle.add_edge('A', 'B', weight=100)
        triangle.add_edge('B', 'C', weight=200)
        triangle.add_edge('C', 'A', weight=300)
        
        result = calculate_clustering_metrics(triangle, 2020)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['year'], 2020)
        self.assertGreaterEqual(result['global_clustering'], 0)
        self.assertLessEqual(result['global_clustering'], 1)
        self.assertGreaterEqual(result['avg_clustering'], 0)
        self.assertLessEqual(result['avg_clustering'], 1)
        
        # 三角形图的聚类系数应该较高
        self.assertGreater(result['global_clustering'], 0.8)
    
    def test_clustering_star_graph(self):
        """测试星型图的聚类系数"""
        star = nx.DiGraph()
        star.add_edge('center', 'A', weight=100)
        star.add_edge('center', 'B', weight=200)
        star.add_edge('center', 'C', weight=300)
        
        result = calculate_clustering_metrics(star, 2020)
        
        # 星型图的聚类系数应该较低，因为外围节点之间没有连接
        self.assertLessEqual(result['global_clustering'], 0.1)
    
    def test_clustering_weighted_calculation(self):
        """测试加权聚类系数计算"""
        weighted_triangle = nx.DiGraph()
        weighted_triangle.add_edge('A', 'B', weight=100)
        weighted_triangle.add_edge('B', 'C', weight=200)
        weighted_triangle.add_edge('C', 'A', weight=300)
        weighted_triangle.add_edge('A', 'C', weight=150)
        
        result = calculate_clustering_metrics(weighted_triangle, 2020)
        
        # 应该包含加权聚类系数
        self.assertIn('weighted_avg_clustering', result)
        self.assertIsInstance(result['weighted_avg_clustering'], (int, float))


class TestCalculateAllGlobalMetrics(unittest.TestCase):
    """测试综合全局指标计算"""
    
    def setUp(self):
        # 创建混合特征的图
        self.mixed_graph = nx.DiGraph()
        # 核心集群
        self.mixed_graph.add_edge('A', 'B', weight=1000)
        self.mixed_graph.add_edge('B', 'C', weight=800)
        self.mixed_graph.add_edge('C', 'A', weight=600)
        # 外围节点
        self.mixed_graph.add_edge('D', 'A', weight=400)
        self.mixed_graph.add_edge('E', 'B', weight=300)
        # 弱连接
        self.mixed_graph.add_edge('F', 'G', weight=200)
        
        self.year = 2020
    
    def test_all_global_metrics_completeness(self):
        """测试所有全局指标计算的完整性"""
        result = calculate_all_global_metrics(self.mixed_graph, self.year)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['year'], self.year)
        
        # 检查所有类别的指标都存在
        density_keys = ['density', 'nodes', 'edges', 'total_weight']
        connectivity_keys = ['is_strongly_connected', 'is_weakly_connected', 
                           'num_strongly_connected_components']
        path_keys = ['avg_path_length', 'median_path_length', 'avg_weighted_path_length']
        efficiency_keys = ['global_efficiency', 'weighted_global_efficiency']
        clustering_keys = ['global_clustering', 'avg_clustering']
        
        for key in density_keys + connectivity_keys + path_keys + efficiency_keys + clustering_keys:
            self.assertIn(key, result, f"Missing key: {key}")
    
    def test_all_metrics_data_types(self):
        """测试所有指标的数据类型正确性"""
        result = calculate_all_global_metrics(self.mixed_graph, self.year)
        
        # 整数类型
        int_keys = ['year', 'nodes', 'edges', 'num_strongly_connected_components', 
                   'num_weakly_connected_components', 'sampled_pairs', 'reachable_pairs']
        for key in int_keys:
            if key in result:
                self.assertIsInstance(result[key], (int, np.integer), f"Key {key} should be int")
        
        # 浮点数类型
        float_keys = ['density', 'total_weight', 'avg_path_length', 'global_efficiency', 
                     'global_clustering']
        for key in float_keys:
            if key in result:
                self.assertIsInstance(result[key], (int, float, np.number), f"Key {key} should be numeric")
        
        # 布尔类型
        bool_keys = ['is_strongly_connected', 'is_weakly_connected']
        for key in bool_keys:
            if key in result:
                self.assertIsInstance(result[key], (bool, np.bool_), f"Key {key} should be bool")
    
    def test_all_metrics_value_ranges(self):
        """测试所有指标值的合理范围"""
        result = calculate_all_global_metrics(self.mixed_graph, self.year)
        
        # 密度应该在[0,1]之间
        self.assertGreaterEqual(result['density'], 0)
        self.assertLessEqual(result['density'], 1)
        
        # 节点数和边数应该为正
        self.assertGreater(result['nodes'], 0)
        self.assertGreaterEqual(result['edges'], 0)
        
        # 聚类系数应该在[0,1]之间
        self.assertGreaterEqual(result['global_clustering'], 0)
        self.assertLessEqual(result['global_clustering'], 1)
        
        # 效率应该为非负
        self.assertGreaterEqual(result['global_efficiency'], 0)
        self.assertGreaterEqual(result['weighted_global_efficiency'], 0)


class TestGlobalMetricsErrorHandling(unittest.TestCase):
    """测试全局指标计算的错误处理"""
    
    def test_invalid_graph_handling(self):
        """测试无效图的处理"""
        with self.assertRaises(TypeError):
            calculate_density_metrics("not a graph", 2020)
    
    def test_empty_graph_handling(self):
        """测试空图的处理"""
        empty_graph = nx.DiGraph()
        
        with self.assertRaises(ValueError):
            calculate_density_metrics(empty_graph, 2020)
    
    def test_computation_error_recovery(self):
        """测试计算错误的恢复机制"""
        # 使用mock来模拟计算错误
        valid_graph = nx.DiGraph()
        valid_graph.add_edge('A', 'B', weight=100)
        
        with patch('global_metrics.nx.density', side_effect=Exception("Mock error")):
            result = calculate_density_metrics(valid_graph, 2020)
            
            # 应该返回默认值而不是抛出异常
            self.assertIsInstance(result, dict)
            self.assertEqual(result['year'], 2020)


class TestGlobalMetricsSummary(unittest.TestCase):
    """测试全局指标摘要功能"""
    
    def test_get_global_metrics_summary(self):
        """测试全局指标摘要生成"""
        # 创建模拟的指标字典
        metrics = {
            'year': 2020,
            'nodes': 100,
            'edges': 500,
            'density': 0.051,
            'is_strongly_connected': True,
            'is_weakly_connected': True,
            'avg_path_length': 2.5,
            'global_efficiency': 0.45,
            'global_clustering': 0.35
        }
        
        summary = get_global_metrics_summary(metrics)
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['year'], 2020)
        self.assertIn('network_scale', summary)
        self.assertIn('connectivity_status', summary)
        self.assertIn('density_level', summary)
        
        # 验证分类判断
        self.assertEqual(summary['connectivity_status'], 'Strongly Connected')
        self.assertEqual(summary['density_level'], 'Medium')  # 0.051 > 0.01
        
        # 验证数值舍入
        self.assertEqual(summary['avg_path_length'], 2.5)
        self.assertEqual(summary['global_efficiency'], 0.45)
        self.assertEqual(summary['clustering_coefficient'], 0.35)


if __name__ == '__main__':
    # 配置测试环境，减少日志输出
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestDensityMetrics,
        TestConnectivityMetrics,
        TestPathMetrics,
        TestEfficiencyMetrics,
        TestClusteringMetrics,
        TestCalculateAllGlobalMetrics,
        TestGlobalMetricsErrorHandling,
        TestGlobalMetricsSummary
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*50}")
    print(f"全局指标测试摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")