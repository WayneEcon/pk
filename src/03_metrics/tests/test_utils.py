#!/usr/bin/env python3
"""
utils模块单元测试
"""

import unittest
import networkx as nx
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    setup_logger, validate_graph, add_distance_weights, safe_divide,
    timer_decorator, handle_computation_error, get_node_sample,
    merge_metric_dataframes, create_metrics_summary, validate_metrics_result
)


class TestSetupLogger(unittest.TestCase):
    """测试日志设置"""
    
    def test_setup_logger_basic(self):
        """测试基本日志设置"""
        logger = setup_logger('test_logger')
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_logger')
        self.assertEqual(logger.level, logging.INFO)
        
    def test_setup_logger_no_duplicate_handlers(self):
        """测试不会重复添加处理器"""
        logger1 = setup_logger('test_logger_dup')
        logger2 = setup_logger('test_logger_dup')
        
        self.assertEqual(len(logger1.handlers), 1)
        self.assertEqual(len(logger2.handlers), 1)


class TestValidateGraph(unittest.TestCase):
    """测试图验证"""
    
    def setUp(self):
        self.valid_graph = nx.DiGraph()
        self.valid_graph.add_edge('A', 'B', weight=1.0)
        self.valid_graph.add_edge('B', 'C', weight=2.0)
        
    def test_validate_graph_success(self):
        """测试有效图验证通过"""
        result = validate_graph(self.valid_graph, "test_function")
        self.assertTrue(result)
        
    def test_validate_graph_wrong_type(self):
        """测试错误类型图"""
        with self.assertRaises(ValueError):
            validate_graph("not a graph", "test_function")
            
    def test_validate_graph_empty(self):
        """测试空图"""
        empty_graph = nx.DiGraph()
        with self.assertRaises(ValueError):
            validate_graph(empty_graph, "test_function")


class TestAddDistanceWeights(unittest.TestCase):
    """测试距离权重添加"""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge('A', 'B', weight=2.0)
        self.graph.add_edge('B', 'C', weight=4.0)
        
    def test_add_distance_weights_basic(self):
        """测试基本距离权重添加"""
        G_with_distance = add_distance_weights(self.graph)
        
        # 检查原图没有被修改
        self.assertNotIn('distance', self.graph.edges['A', 'B'])
        
        # 检查新图有距离属性
        self.assertIn('distance', G_with_distance.edges['A', 'B'])
        self.assertAlmostEqual(G_with_distance.edges['A', 'B']['distance'], 0.5)  # 1/2
        self.assertAlmostEqual(G_with_distance.edges['B', 'C']['distance'], 0.25)  # 1/4
        
    def test_add_distance_weights_zero_weight(self):
        """测试零权重处理"""
        zero_weight_graph = nx.DiGraph()
        zero_weight_graph.add_edge('A', 'B', weight=0.0)
        
        G_with_distance = add_distance_weights(zero_weight_graph)
        
        # 应该设置为无穷大
        self.assertEqual(G_with_distance.edges['A', 'B']['distance'], float('inf'))
        
    def test_add_distance_weights_preserves_other_attrs(self):
        """测试保留其他边属性"""
        graph_with_attrs = nx.DiGraph()
        graph_with_attrs.add_edge('A', 'B', weight=2.0, other_attr='value')
        
        G_with_distance = add_distance_weights(graph_with_attrs)
        
        self.assertEqual(G_with_distance.edges['A', 'B']['other_attr'], 'value')
        self.assertEqual(G_with_distance.edges['A', 'B']['weight'], 2.0)


class TestSafeDivide(unittest.TestCase):
    """测试安全除法"""
    
    def test_safe_divide_normal(self):
        """测试正常除法"""
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(15, 3), 5.0)
        
    def test_safe_divide_by_zero(self):
        """测试除零情况"""
        self.assertEqual(safe_divide(10, 0), 0.0)  # 默认值
        self.assertEqual(safe_divide(10, 0, -1), -1)  # 自定义默认值
        
    def test_safe_divide_negative(self):
        """测试负数除法"""
        self.assertEqual(safe_divide(-10, 2), -5.0)
        self.assertEqual(safe_divide(10, -2), -5.0)


class TestGetNodeSample(unittest.TestCase):
    """测试节点采样函数"""
    
    def test_get_node_sample_normal(self):
        """测试正常采样"""
        nodes = ('A', 'B', 'C', 'D', 'E')
        sample = get_node_sample(nodes, 3, seed=42)
        
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(node in nodes for node in sample))
        
    def test_get_node_sample_smaller_than_input(self):
        """测试采样大小大于输入"""
        nodes = ('A', 'B')
        sample = get_node_sample(nodes, 5, seed=42)
        
        self.assertEqual(sample, nodes)
        
    def test_get_node_sample_reproducible(self):
        """测试采样可重现性"""
        nodes = ('A', 'B', 'C', 'D', 'E')
        sample1 = get_node_sample(nodes, 3, seed=42)
        sample2 = get_node_sample(nodes, 3, seed=42)
        
        self.assertEqual(sample1, sample2)


class TestMergeMetricDataframes(unittest.TestCase):
    """测试DataFrame合并"""
    
    def test_merge_metric_dataframes_basic(self):
        """测试基本合并"""
        df1 = pd.DataFrame({'country': ['A', 'B'], 'metric1': [1, 2]})
        df2 = pd.DataFrame({'country': ['A', 'B'], 'metric2': [3, 4]})
        
        result = merge_metric_dataframes([df1, df2], ['country'])
        
        self.assertEqual(len(result), 2)
        self.assertIn('metric1', result.columns)
        self.assertIn('metric2', result.columns)
        
    def test_merge_metric_dataframes_empty_list(self):
        """测试空列表"""
        result = merge_metric_dataframes([], ['country'])
        self.assertTrue(result.empty)
        
    def test_merge_metric_dataframes_single_df(self):
        """测试单个DataFrame"""
        df = pd.DataFrame({'country': ['A', 'B'], 'metric1': [1, 2]})
        result = merge_metric_dataframes([df], ['country'])
        
        pd.testing.assert_frame_equal(result, df)


class TestCreateMetricsSummary(unittest.TestCase):
    """测试指标摘要创建"""
    
    def test_create_metrics_summary_basic(self):
        """测试基本摘要创建"""
        df = pd.DataFrame({
            'country_code': ['USA', 'CHN', 'DEU'],
            'metric1': [10, 20, 15],
            'metric2': [5, 8, 6]
        })
        
        summary = create_metrics_summary(df, ['metric1', 'metric2'], 2020)
        
        self.assertEqual(summary['year'], 2020)
        self.assertEqual(summary['total_nodes'], 3)
        self.assertAlmostEqual(summary['metric1_mean'], 15.0)
        self.assertEqual(summary['metric1_max_country'], 'CHN')
        
    def test_create_metrics_summary_missing_columns(self):
        """测试缺失列处理"""
        df = pd.DataFrame({'country_code': ['USA'], 'metric1': [10]})
        
        summary = create_metrics_summary(df, ['metric1', 'missing_metric'], 2020)
        
        self.assertIn('metric1_mean', summary)
        self.assertNotIn('missing_metric_mean', summary)


class TestValidateMetricsResult(unittest.TestCase):
    """测试指标结果验证"""
    
    def test_validate_metrics_result_success(self):
        """测试成功验证"""
        df = pd.DataFrame({
            'country_code': ['USA', 'CHN'],
            'metric1': [10, 20],
            'metric2': [5, 8]
        })
        
        result = validate_metrics_result(
            df, ['country_code', 'metric1'], 2020, "test_metrics"
        )
        self.assertTrue(result)
        
    def test_validate_metrics_result_empty_df(self):
        """测试空DataFrame"""
        df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            validate_metrics_result(df, ['metric1'], 2020, "test_metrics")
            
    def test_validate_metrics_result_missing_columns(self):
        """测试缺失列"""
        df = pd.DataFrame({'metric1': [10, 20]})
        
        with self.assertRaises(ValueError):
            validate_metrics_result(
                df, ['metric1', 'missing_metric'], 2020, "test_metrics"
            )


class TestHandleComputationError(unittest.TestCase):
    """测试错误处理"""
    
    def test_handle_computation_error_normal(self):
        """测试一般错误处理"""
        error = ValueError("test error")
        default_result = {"error": True}
        
        result = handle_computation_error(
            "test_function", 2020, error, default_result
        )
        
        self.assertEqual(result, default_result)
        
    def test_handle_computation_error_memory_error(self):
        """测试内存错误重新抛出"""
        error = MemoryError("out of memory")
        default_result = {"error": True}
        
        with self.assertRaises(MemoryError):
            handle_computation_error(
                "test_function", 2020, error, default_result
            )


class TestTimerDecorator(unittest.TestCase):
    """测试计时装饰器"""
    
    def test_timer_decorator_basic(self):
        """测试基本计时功能"""
        @timer_decorator
        def dummy_function(year=2020):
            import time
            time.sleep(0.01)
            return "result"
        
        # 捕获日志输出
        with self.assertLogs(level='INFO') as log:
            result = dummy_function()
            
        self.assertEqual(result, "result")
        self.assertTrue(any("dummy_function计算完成" in message for message in log.output))
        
    def test_timer_decorator_with_args(self):
        """测试带参数的计时"""
        @timer_decorator  
        def dummy_function(graph, year):
            return f"processed {year}"
        
        with self.assertLogs(level='INFO') as log:
            result = dummy_function("fake_graph", 2020)
            
        self.assertEqual(result, "processed 2020")
        self.assertTrue(any("2020:" in message for message in log.output))


if __name__ == '__main__':
    # 配置日志以避免测试时的输出干扰
    logging.getLogger().setLevel(logging.CRITICAL)
    
    unittest.main()