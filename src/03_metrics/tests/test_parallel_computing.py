#!/usr/bin/env python3
"""
parallel_computing模块单元测试
"""

import unittest
import networkx as nx
import pandas as pd
import multiprocessing as mp
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from parallel_computing import (
    _calculate_metrics_worker, calculate_metrics_parallel,
    _fallback_serial_calculation, calculate_centralities_batch,
    _centrality_worker, estimate_computation_time,
    get_optimal_process_count
)


class TestCalculateMetricsWorker(unittest.TestCase):
    """测试并行计算工作函数"""
    
    def setUp(self):
        # 创建测试图
        self.test_graph = nx.DiGraph()
        self.test_graph.add_edge('A', 'B', weight=100)
        self.test_graph.add_edge('B', 'C', weight=200)
        self.test_graph.add_edge('C', 'A', weight=300)
        self.year = 2020
    
    def test_worker_all_metrics(self):
        """测试计算所有指标"""
        args = (self.test_graph, self.year, 'all')
        
        result_year, result_df = _calculate_metrics_worker(args)
        
        self.assertEqual(result_year, self.year)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        
        # 检查包含节点指标和全局指标
        expected_cols = ['year', 'country_code', 'total_degree', 'total_strength']
        for col in expected_cols:
            self.assertIn(col, result_df.columns, f"Missing column: {col}")
    
    def test_worker_node_metrics_only(self):
        """测试只计算节点指标"""
        args = (self.test_graph, self.year, 'node')
        
        result_year, result_df = _calculate_metrics_worker(args)
        
        self.assertEqual(result_year, self.year)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        
        # 应该包含节点中心性指标
        node_metrics_cols = ['in_degree', 'out_degree', 'betweenness_centrality', 'pagerank_centrality']
        for col in node_metrics_cols:
            self.assertIn(col, result_df.columns, f"Missing node metric: {col}")
    
    def test_worker_global_metrics_only(self):
        """测试只计算全局指标"""
        args = (self.test_graph, self.year, 'global')
        
        result_year, result_df = _calculate_metrics_worker(args)
        
        self.assertEqual(result_year, self.year)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 1)  # 全局指标只有一行
        
        # 应该包含全局网络指标
        global_metrics_cols = ['year', 'density', 'nodes', 'edges']
        for col in global_metrics_cols:
            self.assertIn(col, result_df.columns, f"Missing global metric: {col}")
    
    def test_worker_invalid_metric_type(self):
        """测试无效的指标类型"""
        args = (self.test_graph, self.year, 'invalid_type')
        
        result_year, result_df = _calculate_metrics_worker(args)
        
        self.assertEqual(result_year, self.year)
        self.assertTrue(result_df.empty)  # 应该返回空DataFrame
    
    def test_worker_computation_error(self):
        """测试计算错误处理"""
        # 使用无效图触发错误
        invalid_graph = "not a graph"
        args = (invalid_graph, self.year, 'all')
        
        result_year, result_df = _calculate_metrics_worker(args)
        
        self.assertEqual(result_year, self.year)
        self.assertTrue(result_df.empty)


class TestCalculateMetricsParallel(unittest.TestCase):
    """测试并行计算主函数"""
    
    def setUp(self):
        # 创建多个年份的网络
        self.annual_networks = {}
        
        for year in [2018, 2019, 2020]:
            G = nx.DiGraph()
            # 创建不同规模的网络
            n_nodes = 3 + year - 2018  # 2018年3个节点，2019年4个，2020年5个
            nodes = [f"NODE_{i}" for i in range(n_nodes)]
            
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if i != j:
                        weight = (i + 1) * (j + 1) * year
                        G.add_edge(source, target, weight=weight)
            
            self.annual_networks[year] = G
    
    def test_parallel_all_metrics(self):
        """测试并行计算所有指标"""
        result_df = calculate_metrics_parallel(self.annual_networks, 'all', n_processes=2)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        
        # 检查包含所有年份
        years_in_result = set(result_df['year'].unique())
        expected_years = set(self.annual_networks.keys())
        self.assertEqual(years_in_result, expected_years)
        
        # 检查数据完整性
        for year in self.annual_networks.keys():
            year_data = result_df[result_df['year'] == year]
            expected_nodes = self.annual_networks[year].number_of_nodes()
            self.assertEqual(len(year_data), expected_nodes)
    
    def test_parallel_node_metrics(self):
        """测试并行计算节点指标"""
        result_df = calculate_metrics_parallel(self.annual_networks, 'node')
        
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # 检查只包含节点指标
        node_cols = ['in_degree', 'out_degree', 'total_strength', 'pagerank_centrality']
        for col in node_cols:
            self.assertIn(col, result_df.columns)
        
        # 不应该包含全局指标列
        global_cols = ['density', 'global_efficiency']
        for col in global_cols:
            self.assertNotIn(col, result_df.columns)
    
    def test_parallel_global_metrics(self):
        """测试并行计算全局指标"""
        result_df = calculate_metrics_parallel(self.annual_networks, 'global')
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.annual_networks))  # 每年一行
        
        # 检查包含全局指标
        global_cols = ['density', 'nodes', 'edges', 'global_efficiency']
        for col in global_cols:
            self.assertIn(col, result_df.columns)
    
    def test_parallel_empty_networks(self):
        """测试空网络字典"""
        result_df = calculate_metrics_parallel({}, 'all')
        
        self.assertTrue(result_df.empty)
    
    def test_parallel_process_count_auto(self):
        """测试自动进程数选择"""
        # 不指定进程数，应该自动选择
        result_df = calculate_metrics_parallel(self.annual_networks, 'all')
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
    
    @patch('parallel_computing.mp.Pool')
    def test_parallel_fallback_on_error(self, mock_pool):
        """测试并行计算失败时的回退机制"""
        # 模拟Pool初始化失败
        mock_pool.side_effect = Exception("Pool creation failed")
        
        result_df = calculate_metrics_parallel(self.annual_networks, 'all')
        
        # 应该成功回退到串行计算
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)


class TestFallbackSerialCalculation(unittest.TestCase):
    """测试串行回退计算"""
    
    def setUp(self):
        self.networks = {}
        for year in [2020, 2021]:
            G = nx.DiGraph()
            G.add_edge('A', 'B', weight=100)
            G.add_edge('B', 'C', weight=200)
            self.networks[year] = G
    
    def test_fallback_serial_success(self):
        """测试串行计算成功"""
        result_df = _fallback_serial_calculation(self.networks, 'all')
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        
        # 检查包含所有年份
        years = result_df['year'].unique()
        self.assertEqual(set(years), set(self.networks.keys()))
    
    def test_fallback_serial_partial_failure(self):
        """测试部分年份计算失败"""
        # 添加一个无效的网络
        invalid_networks = self.networks.copy()
        invalid_networks[2022] = "invalid_graph"
        
        result_df = _fallback_serial_calculation(invalid_networks, 'all')
        
        # 应该返回有效年份的结果
        self.assertIsInstance(result_df, pd.DataFrame)
        years = result_df['year'].unique() if not result_df.empty else []
        self.assertIn(2020, years)
        self.assertIn(2021, years)
        self.assertNotIn(2022, years)


class TestCalculateCentralitiesBatch(unittest.TestCase):
    """测试批量中心性计算"""
    
    def setUp(self):
        # 创建测试图和年份列表
        self.graphs_and_years = []
        for year in [2020, 2021]:
            G = nx.DiGraph()
            G.add_edge('A', 'B', weight=100)
            G.add_edge('B', 'C', weight=200)
            G.add_edge('C', 'A', weight=150)
            self.graphs_and_years.append((G, year))
    
    def test_centralities_batch_multiple_functions(self):
        """测试批量计算多种中心性"""
        # 模拟中心性计算函数
        def mock_degree_centrality(G, year):
            nodes = list(G.nodes())
            data = [{
                'year': year,
                'country_code': node,
                'degree_centrality': len(list(G.neighbors(node)))
            } for node in nodes]
            return pd.DataFrame(data)
        
        def mock_strength_centrality(G, year):
            nodes = list(G.nodes())
            data = [{
                'year': year,
                'country_code': node,
                'strength_centrality': sum(G[node][neighbor]['weight'] 
                                         for neighbor in G.neighbors(node))
            } for node in nodes]
            return pd.DataFrame(data)
        
        centrality_functions = [mock_degree_centrality, mock_strength_centrality]
        
        results = calculate_centralities_batch(
            self.graphs_and_years, 
            centrality_functions, 
            n_processes=2
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('mock_degree_centrality', results)
        self.assertIn('mock_strength_centrality', results)
        
        # 检查每个结果都是DataFrame
        for func_name, df in results.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
    
    def test_centralities_batch_empty_functions(self):
        """测试空函数列表"""
        results = calculate_centralities_batch(self.graphs_and_years, [])
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0)
    
    @patch('parallel_computing.mp.Pool')
    def test_centralities_batch_with_errors(self, mock_pool):
        """测试批量计算中的错误处理"""
        def failing_function(G, year):
            raise Exception("Computation failed")
        
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.side_effect = Exception("Pool error")
        mock_pool.return_value = mock_pool_instance
        
        results = calculate_centralities_batch(
            self.graphs_and_years, 
            [failing_function]
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('failing_function', results)
        self.assertTrue(results['failing_function'].empty)


class TestCentralityWorker(unittest.TestCase):
    """测试中心性工作函数"""
    
    def test_centrality_worker_success(self):
        """测试中心性工作函数成功执行"""
        G = nx.DiGraph()
        G.add_edge('A', 'B', weight=100)
        G.add_edge('B', 'C', weight=200)
        
        def mock_centrality_func(graph, year):
            return pd.DataFrame({
                'year': [year, year],
                'country_code': ['A', 'B'],
                'centrality': [0.5, 0.3]
            })
        
        result_df = _centrality_worker((G, 2020), mock_centrality_func)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 2)
        self.assertIn('centrality', result_df.columns)
    
    def test_centrality_worker_error(self):
        """测试中心性工作函数错误处理"""
        G = nx.DiGraph()
        G.add_edge('A', 'B', weight=100)
        
        def failing_func(graph, year):
            raise Exception("Centrality calculation failed")
        
        result_df = _centrality_worker((G, 2020), failing_func)
        
        self.assertTrue(result_df.empty)


class TestComputationTimeEstimation(unittest.TestCase):
    """测试计算时间估算"""
    
    def setUp(self):
        # 创建不同规模的网络用于时间估算
        self.networks = {}
        for i, year in enumerate([2018, 2019, 2020, 2021, 2022], 1):
            G = nx.DiGraph()
            # 创建递增规模的网络
            nodes = [f"N{j}" for j in range(i * 2)]  # 2, 4, 6, 8, 10 nodes
            for source in nodes:
                for target in nodes:
                    if source != target:
                        G.add_edge(source, target, weight=float(hash(f"{source}{target}") % 1000))
            self.networks[year] = G
    
    def test_estimate_computation_time_basic(self):
        """测试基本时间估算"""
        estimates = estimate_computation_time(self.networks, 'all', sample_size=2)
        
        self.assertIsInstance(estimates, dict)
        
        required_keys = ['serial_time', 'parallel_time', 'speedup', 'recommended_parallel']
        for key in required_keys:
            self.assertIn(key, estimates)
        
        # 检查数值合理性
        self.assertGreaterEqual(estimates['serial_time'], 0)
        self.assertGreaterEqual(estimates['parallel_time'], 0)
        self.assertGreaterEqual(estimates['speedup'], 0)
        self.assertIsInstance(estimates['recommended_parallel'], bool)
    
    def test_estimate_empty_networks(self):
        """测试空网络的时间估算"""
        estimates = estimate_computation_time({}, 'all')
        
        self.assertEqual(estimates['serial_time'], 0)
        self.assertEqual(estimates['parallel_time'], 0)
        self.assertEqual(estimates['speedup'], 1)
    
    def test_estimate_different_metric_types(self):
        """测试不同指标类型的时间估算"""
        metric_types = ['all', 'node', 'global']
        
        for metric_type in metric_types:
            estimates = estimate_computation_time(
                self.networks, 
                metric_type, 
                sample_size=1
            )
            
            self.assertIsInstance(estimates, dict)
            self.assertGreaterEqual(estimates['serial_time'], 0)
            self.assertGreaterEqual(estimates['parallel_time'], 0)


class TestOptimalProcessCount(unittest.TestCase):
    """测试最优进程数确定"""
    
    def test_optimal_process_count_small_networks(self):
        """测试小网络的最优进程数"""
        small_networks = {}
        for year in [2020, 2021]:
            G = nx.DiGraph()
            G.add_edge('A', 'B', weight=100)
            G.add_edge('B', 'C', weight=200)
            small_networks[year] = G
        
        optimal = get_optimal_process_count(small_networks)
        
        self.assertIsInstance(optimal, int)
        self.assertGreaterEqual(optimal, 1)
        self.assertLessEqual(optimal, mp.cpu_count())
    
    def test_optimal_process_count_large_networks(self):
        """测试大网络的最优进程数"""
        large_networks = {}
        for year in [2018, 2019, 2020]:
            G = nx.DiGraph()
            # 创建较大的网络 (>200 nodes)
            nodes = [f"NODE_{i}" for i in range(250)]
            # 只连接部分节点以避免测试时间过长
            for i in range(0, 50, 5):
                for j in range(i+1, min(i+6, 50)):
                    G.add_edge(nodes[i], nodes[j], weight=100)
            large_networks[year] = G
        
        optimal = get_optimal_process_count(large_networks)
        
        self.assertIsInstance(optimal, int)
        self.assertGreaterEqual(optimal, 1)
        # 大网络应该使用较少的进程数
        self.assertLessEqual(optimal, mp.cpu_count() // 2 + 1)
    
    def test_optimal_process_count_empty_networks(self):
        """测试空网络字典"""
        optimal = get_optimal_process_count({})
        
        self.assertEqual(optimal, 1)  # 至少返回1个进程
    
    def test_optimal_process_count_medium_networks(self):
        """测试中等规模网络"""
        medium_networks = {}
        for year in [2019, 2020, 2021, 2022]:
            G = nx.DiGraph()
            # 创建中等规模的网络 (100-200 nodes)
            nodes = [f"NODE_{i}" for i in range(150)]
            # 部分连接
            for i in range(0, 30):
                for j in range(i+1, min(i+6, 30)):
                    G.add_edge(nodes[i], nodes[j], weight=100)
            medium_networks[year] = G
        
        optimal = get_optimal_process_count(medium_networks)
        
        self.assertIsInstance(optimal, int)
        self.assertGreaterEqual(optimal, 1)
        self.assertLessEqual(optimal, mp.cpu_count())


if __name__ == '__main__':
    # 配置测试环境
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCalculateMetricsWorker,
        TestCalculateMetricsParallel,
        TestFallbackSerialCalculation,
        TestCalculateCentralitiesBatch,
        TestCentralityWorker,
        TestComputationTimeEstimation,
        TestOptimalProcessCount
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*50}")
    print(f"并行计算模块测试摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")