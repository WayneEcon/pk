#!/usr/bin/env python3
"""
数据处理模块单元测试
"""

import unittest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor import resolve_trade_data_consistency, aggregate_trade_flows


class TestDataProcessor(unittest.TestCase):
    """数据处理器测试类"""
    
    def setUp(self):
        """测试设置"""
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'flow': ['M', 'X', 'M', 'X'],
            'reporter': ['USA', 'CHN', 'DEU', 'GBR'],
            'partner': ['CHN', 'USA', 'FRA', 'DEU'],
            'reporter_name': ['United States', 'China', 'Germany', 'United Kingdom'],
            'partner_name': ['China', 'United States', 'France', 'Germany'],
            'trade_value_raw_usd': [1000, 800, 500, 300],
            'year': [2020, 2020, 2020, 2020]
        })
        
        # 空数据测试
        self.empty_data = pd.DataFrame()
        
        # 缺少列的数据
        self.invalid_data = pd.DataFrame({
            'flow': ['M', 'X'],
            'reporter': ['USA', 'CHN']
            # 缺少其他必要列
        })
        
    def test_resolve_trade_data_consistency_success(self):
        """测试数据一致性处理成功情况"""
        result = resolve_trade_data_consistency(self.test_data, 2020)
        
        # 验证结果不为空
        self.assertFalse(result.empty)
        
        # 验证必要列存在
        expected_cols = ['source', 'target', 'trade_value_raw_usd', 'source_name', 
                        'target_name', 'year', 'data_source', 'trade_flow_id']
        for col in expected_cols:
            self.assertIn(col, result.columns)
            
        # 验证数据源标记
        data_sources = set(result['data_source'].unique())
        self.assertTrue(data_sources.issubset({'import_reported', 'export_mirrored'}))
        
    def test_resolve_trade_data_consistency_missing_columns(self):
        """测试缺少必要列的情况"""
        with self.assertRaises(ValueError) as context:
            resolve_trade_data_consistency(self.invalid_data, 2020)
            
        self.assertIn('缺少必要字段', str(context.exception))
        
    def test_resolve_trade_data_consistency_empty_data(self):
        """测试空数据处理"""
        result = resolve_trade_data_consistency(self.empty_data, 2020)
        self.assertTrue(result.empty)
        
    def test_resolve_trade_data_consistency_only_imports(self):
        """测试只有进口数据的情况"""
        import_only = self.test_data[self.test_data['flow'] == 'M'].copy()
        result = resolve_trade_data_consistency(import_only, 2020)
        
        self.assertFalse(result.empty)
        # 所有记录都应该基于进口数据
        self.assertTrue(all(result['data_source'] == 'import_reported'))
        
    def test_resolve_trade_data_consistency_only_exports(self):
        """测试只有出口数据的情况"""
        export_only = self.test_data[self.test_data['flow'] == 'X'].copy()
        result = resolve_trade_data_consistency(export_only, 2020)
        
        self.assertFalse(result.empty)
        # 所有记录都应该基于出口镜像
        self.assertTrue(all(result['data_source'] == 'export_mirrored'))
        
    def test_aggregate_trade_flows_success(self):
        """测试贸易流聚合成功情况"""
        # 首先处理数据一致性
        consistent_data = resolve_trade_data_consistency(self.test_data, 2020)
        
        # 添加重复的贸易对以测试聚合
        duplicate_row = consistent_data.iloc[0:1].copy()
        duplicate_row['trade_value_raw_usd'] = 100
        test_data_with_duplicates = pd.concat([consistent_data, duplicate_row], ignore_index=True)
        
        result = aggregate_trade_flows(test_data_with_duplicates, 2020)
        
        # 验证聚合结果
        self.assertFalse(result.empty)
        
        # 验证必要列存在
        expected_cols = ['source', 'target', 'trade_value_raw_usd', 'source_name',
                        'target_name', 'year', 'primary_data_source']
        for col in expected_cols:
            self.assertIn(col, result.columns)
            
        # 验证重复记录被合并（聚合后的记录数应该小于等于原始记录数）
        self.assertLessEqual(len(result), len(test_data_with_duplicates))
        
    def test_aggregate_trade_flows_empty_data(self):
        """测试聚合空数据"""
        empty_df = pd.DataFrame()
        result = aggregate_trade_flows(empty_df, 2020)
        
        self.assertTrue(result.empty)
        
    def test_aggregate_trade_flows_missing_columns(self):
        """测试聚合缺少必要列的数据"""
        invalid_data = pd.DataFrame({
            'source': ['USA'],
            'target': ['CHN']
            # 缺少其他必要列
        })
        
        with self.assertRaises(ValueError) as context:
            aggregate_trade_flows(invalid_data, 2020)
            
        self.assertIn('缺少必要字段', str(context.exception))
        
    def test_aggregate_trade_flows_values_summed(self):
        """测试贸易流值正确求和"""
        # 创建包含重复贸易对的数据
        test_data = pd.DataFrame({
            'source': ['USA', 'USA', 'CHN'],
            'target': ['CHN', 'CHN', 'USA'],
            'trade_value_raw_usd': [1000, 500, 2000],
            'source_name': ['United States', 'United States', 'China'],
            'target_name': ['China', 'China', 'United States'],
            'year': [2020, 2020, 2020],
            'data_source': ['import_reported', 'import_reported', 'export_mirrored']
        })
        
        result = aggregate_trade_flows(test_data, 2020)
        
        # 验证USA->CHN的贸易额被正确求和
        usa_chn_row = result[(result['source'] == 'USA') & (result['target'] == 'CHN')]
        self.assertEqual(len(usa_chn_row), 1)
        self.assertEqual(usa_chn_row.iloc[0]['trade_value_raw_usd'], 1500)  # 1000 + 500
        
    @patch('data_processor.logger')
    def test_logging_messages(self, mock_logger):
        """测试日志消息"""
        resolve_trade_data_consistency(self.test_data, 2020)
        
        # 验证日志被调用
        mock_logger.info.assert_called()
        
        # 验证日志消息包含年份
        call_args_list = [call.args[0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any('2020' in msg for msg in call_args_list))


class TestDataProcessorEdgeCases(unittest.TestCase):
    """数据处理器边界情况测试"""
    
    def test_resolve_consistency_no_flow_data(self):
        """测试没有进出口标识的数据"""
        no_flow_data = pd.DataFrame({
            'flow': [],
            'reporter': [],
            'partner': [],
            'reporter_name': [],
            'partner_name': [],
            'trade_value_raw_usd': [],
            'year': []
        })
        
        result = resolve_trade_data_consistency(no_flow_data, 2020)
        self.assertTrue(result.empty)
        
    def test_aggregate_flows_single_record(self):
        """测试单条记录聚合"""
        single_record = pd.DataFrame({
            'source': ['USA'],
            'target': ['CHN'],
            'trade_value_raw_usd': [1000],
            'source_name': ['United States'],
            'target_name': ['China'],
            'year': [2020],
            'data_source': ['import_reported']
        })
        
        result = aggregate_trade_flows(single_record, 2020)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['trade_value_raw_usd'], 1000)
        self.assertEqual(result.iloc[0]['primary_data_source'], 'import_reported')


if __name__ == '__main__':
    # 配置日志以避免测试时的输出干扰
    logging.getLogger().setLevel(logging.CRITICAL)
    
    unittest.main()