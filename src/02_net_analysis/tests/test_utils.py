#!/usr/bin/env python3
"""
工具函数单元测试
"""

import unittest
import pandas as pd
import networkx as nx
import logging
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    validate_dataframe_columns, safe_divide, log_dataframe_info,
    create_trade_flow_id, get_country_region_safe, validate_network_graph,
    validate_trade_data_schema, validate_statistics_data, DataQualityReporter
)


class TestUtilsFunctions(unittest.TestCase):
    """工具函数测试类"""
    
    def setUp(self):
        """测试设置"""
        self.sample_df = pd.DataFrame({
            'source': ['USA', 'CHN', 'GBR'],
            'target': ['CHN', 'USA', 'DEU'],
            'trade_value_raw_usd': [1000, 2000, 3000]
        })
        
        self.invalid_df = pd.DataFrame({
            'source': ['USA', 'CHN', 'USA'],
            'target': ['CHN', 'USA', 'CHN'],  # 重复记录
            'trade_value_raw_usd': [1000, -500, 0]  # 包含负值
        })
        
    def test_validate_dataframe_columns_success(self):
        """测试DataFrame列验证成功情况"""
        # 应该不抛出异常
        validate_dataframe_columns(self.sample_df, ['source', 'target'], "测试数据")
        
    def test_validate_dataframe_columns_missing(self):
        """测试DataFrame缺少必要列"""
        with self.assertRaises(ValueError) as context:
            validate_dataframe_columns(self.sample_df, ['source', 'missing_col'], "测试数据")
        
        self.assertIn('missing_col', str(context.exception))
        
    def test_validate_dataframe_columns_empty(self):
        """测试空DataFrame"""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            validate_dataframe_columns(empty_df, ['source'], "空数据")
        
        self.assertIn('为空', str(context.exception))
        
    def test_safe_divide(self):
        """测试安全除法"""
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 0), 0.0)  # 默认值
        self.assertEqual(safe_divide(10, 0, -1), -1)  # 自定义默认值
        
    def test_create_trade_flow_id(self):
        """测试贸易流ID创建"""
        flow_id = create_trade_flow_id("USA", "CHN")
        self.assertEqual(flow_id, "USA_to_CHN")
        
    def test_get_country_region_safe(self):
        """测试安全获取国家区域"""
        # 测试默认值
        region = get_country_region_safe("UNKNOWN")
        self.assertEqual(region, 'Other')
        
        # 测试自定义默认值
        region = get_country_region_safe("UNKNOWN", "Custom")
        self.assertEqual(region, "Custom")
        
    def test_validate_network_graph(self):
        """测试网络图验证"""
        # 测试有效图
        G = nx.DiGraph()
        G.add_edge('A', 'B')
        validate_network_graph(G, "测试图")  # 应该不抛出异常
        
        # 测试无效对象
        with self.assertRaises(TypeError):
            validate_network_graph("not a graph", "无效图")
            
    def test_validate_trade_data_schema(self):
        """测试贸易数据模式验证"""
        # 测试有效数据
        warnings = validate_trade_data_schema(self.sample_df, 2020)
        self.assertEqual(len(warnings), 0)
        
        # 测试有问题的数据
        warnings = validate_trade_data_schema(self.invalid_df, 2020)
        self.assertGreater(len(warnings), 0)
        self.assertTrue(any('负值' in w for w in warnings))
        self.assertTrue(any('重复' in w for w in warnings))
        
    def test_validate_trade_data_schema_empty(self):
        """测试空数据验证"""
        empty_df = pd.DataFrame()
        warnings = validate_trade_data_schema(empty_df, 2020)
        self.assertEqual(len(warnings), 1)
        self.assertIn('数据为空', warnings[0])
        
    def test_validate_statistics_data(self):
        """测试统计数据验证"""
        valid_stats = [
            {'year': 2020, 'nodes': 10, 'edges': 15, 'total_trade_value': 1000, 'density': 0.2},
            {'year': 2021, 'nodes': 12, 'edges': 18, 'total_trade_value': 1200, 'density': 0.15}
        ]
        
        warnings = validate_statistics_data(valid_stats, [2020, 2021])
        self.assertEqual(len(warnings), 0)
        
        # 测试缺少年份
        warnings = validate_statistics_data(valid_stats, [2020, 2021, 2022])
        self.assertGreater(len(warnings), 0)
        self.assertTrue(any('缺少年份' in w for w in warnings))
        
    def test_validate_statistics_data_invalid(self):
        """测试无效统计数据"""
        invalid_stats = [
            {'year': 2020, 'nodes': 10, 'edges': 150, 'total_trade_value': -1000, 'density': 1.5}
        ]
        
        warnings = validate_statistics_data(invalid_stats)
        self.assertGreater(len(warnings), 0)
        self.assertTrue(any('边数' in w and '超过' in w for w in warnings))
        self.assertTrue(any('密度' in w and '超出' in w for w in warnings))
        self.assertTrue(any('负值' in w for w in warnings))


class TestDataQualityReporter(unittest.TestCase):
    """数据质量报告器测试类"""
    
    def setUp(self):
        """测试设置"""
        self.reporter = DataQualityReporter()
        
    def test_add_report(self):
        """测试添加报告"""
        self.reporter.add_report("测试步骤", 2020, 100, 90, "测试描述")
        
        summary = self.reporter.get_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['step'], "测试步骤")
        self.assertEqual(summary.iloc[0]['year'], 2020)
        self.assertEqual(summary.iloc[0]['change_count'], -10)
        
    def test_add_warning(self):
        """测试添加警告"""
        self.reporter.add_warning("测试警告")
        warnings = self.reporter.get_warnings_summary()
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0], "测试警告")
        
    def test_validate_and_report(self):
        """测试验证并报告"""
        invalid_df = pd.DataFrame({
            'source': ['USA'],
            'target': ['USA'],  # 自环
            'trade_value_raw_usd': [-100]  # 负值
        })
        
        self.reporter.validate_and_report(invalid_df, "测试数据", 2020)
        warnings = self.reporter.get_warnings_summary()
        self.assertGreater(len(warnings), 0)
        
    def test_clear(self):
        """测试清空功能"""
        self.reporter.add_report("测试", 2020, 100, 90)
        self.reporter.add_warning("测试警告")
        
        self.reporter.clear()
        
        summary = self.reporter.get_summary()
        warnings = self.reporter.get_warnings_summary()
        
        self.assertEqual(len(summary), 0)
        self.assertEqual(len(warnings), 0)


class TestLogDataFrameInfo(unittest.TestCase):
    """数据框信息记录测试类"""
    
    @patch('utils.logging.getLogger')
    def test_log_dataframe_info(self, mock_get_logger):
        """测试数据框信息记录"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        log_dataframe_info(df, "测试数据", 2020, mock_logger)
        
        # 验证日志被调用
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("2020:", call_args)
        self.assertIn("测试数据", call_args)
        self.assertIn("3 行", call_args)
        
    @patch('utils.logging.getLogger')
    def test_log_empty_dataframe_info(self, mock_get_logger):
        """测试空数据框信息记录"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        empty_df = pd.DataFrame()
        log_dataframe_info(empty_df, "空数据", 2020, mock_logger)
        
        # 验证警告被调用
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("数据为空", call_args)


if __name__ == '__main__':
    # 配置日志以避免测试时的输出干扰
    logging.getLogger().setLevel(logging.CRITICAL)
    
    unittest.main()