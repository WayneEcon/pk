#!/usr/bin/env python3
"""
网络构建模块单元测试
"""

import unittest
import pandas as pd
import networkx as nx
import logging
from unittest.mock import patch
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from network_builder import build_network_from_data


class TestNetworkBuilder(unittest.TestCase):
    """网络构建器测试类"""
    
    def setUp(self):
        """测试设置"""
        self.valid_data = pd.DataFrame({
            'source': ['USA', 'CHN', 'DEU'],
            'target': ['CHN', 'USA', 'FRA'],
            'trade_value_raw_usd': [1000, 2000, 1500],
            'source_name': ['United States', 'China', 'Germany'],
            'target_name': ['China', 'United States', 'France'],
            'primary_data_source': ['import_reported', 'export_mirrored', 'import_reported']
        })
        
        self.empty_data = pd.DataFrame()
        
        self.invalid_data = pd.DataFrame({
            'source': ['USA'],
            'target': ['CHN']
            # 缺少必要列
        })
        
        self.duplicate_edge_data = pd.DataFrame({
            'source': ['USA', 'USA'],
            'target': ['CHN', 'CHN'],  # 重复边
            'trade_value_raw_usd': [1000, 500],
            'source_name': ['United States', 'United States'],
            'target_name': ['China', 'China'],
            'primary_data_source': ['import_reported', 'import_reported']
        })
        
    def test_build_network_success(self):
        """测试网络构建成功情况"""
        G = build_network_from_data(self.valid_data, 2020)
        
        # 验证图类型
        self.assertIsInstance(G, nx.DiGraph)
        
        # 验证节点数和边数
        self.assertEqual(G.number_of_nodes(), 4)  # USA, CHN, DEU, FRA
        self.assertEqual(G.number_of_edges(), 3)
        
        # 验证图属性
        self.assertEqual(G.graph['year'], 2020)
        self.assertIn('description', G.graph)
        self.assertIn('created_at', G.graph)
        
        # 验证节点属性
        for node in G.nodes():
            self.assertIn('name', G.nodes[node])
            self.assertIn('country_code', G.nodes[node])
            self.assertIn('region', G.nodes[node])
            
        # 验证边属性
        for source, target in G.edges():
            self.assertIn('weight', G.edges[source, target])
            self.assertIn('data_source', G.edges[source, target])
            
    def test_build_network_empty_data(self):
        """测试空数据构建网络"""
        G = build_network_from_data(self.empty_data, 2020)
        
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(G.number_of_nodes(), 0)
        self.assertEqual(G.number_of_edges(), 0)
        
    def test_build_network_invalid_data(self):
        """测试无效数据构建网络"""
        with self.assertRaises(ValueError) as context:
            build_network_from_data(self.invalid_data, 2020)
            
        self.assertIn('缺少必要字段', str(context.exception))
        
    def test_build_network_duplicate_edges(self):
        """测试重复边处理"""
        with patch('network_builder.logger') as mock_logger:
            G = build_network_from_data(self.duplicate_edge_data, 2020)
            
            # 验证重复边被合并
            self.assertEqual(G.number_of_edges(), 1)
            
            # 验证权重被累加
            edge_weight = G.edges['USA', 'CHN']['weight']
            self.assertEqual(edge_weight, 1500)  # 1000 + 500
            
            # 验证警告日志被调用
            mock_logger.warning.assert_called()
            
    def test_network_node_attributes(self):
        """测试节点属性设置"""
        G = build_network_from_data(self.valid_data, 2020)
        
        # 验证USA节点
        usa_attrs = G.nodes['USA']
        self.assertEqual(usa_attrs['name'], 'United States')
        self.assertEqual(usa_attrs['country_code'], 'USA')
        self.assertIn('region', usa_attrs)
        
        # 验证CHN节点
        chn_attrs = G.nodes['CHN']
        self.assertEqual(chn_attrs['name'], 'China')
        self.assertEqual(chn_attrs['country_code'], 'CHN')
        
    def test_network_edge_attributes(self):
        """测试边属性设置"""
        G = build_network_from_data(self.valid_data, 2020)
        
        # 验证USA->CHN边
        usa_chn_attrs = G.edges['USA', 'CHN']
        self.assertEqual(usa_chn_attrs['weight'], 1000)
        self.assertEqual(usa_chn_attrs['data_source'], 'import_reported')
        
        # 验证CHN->USA边
        chn_usa_attrs = G.edges['CHN', 'USA']
        self.assertEqual(chn_usa_attrs['weight'], 2000)
        self.assertEqual(chn_usa_attrs['data_source'], 'export_mirrored')
        
    def test_network_graph_metadata(self):
        """测试图元数据"""
        G = build_network_from_data(self.valid_data, 2020)
        
        # 验证基本元数据
        self.assertEqual(G.graph['year'], 2020)
        self.assertIn('Global Energy Trade Network 2020', G.graph['description'])
        self.assertEqual(G.graph['input_records'], len(self.valid_data))
        
        # 验证时间戳存在且格式正确
        self.assertIn('created_at', G.graph)
        self.assertIsInstance(G.graph['created_at'], str)
        
    def test_build_network_large_dataset(self):
        """测试较大数据集的网络构建"""
        # 创建较大的测试数据集
        countries = [f'C{i:03d}' for i in range(50)]
        large_data = []
        
        for i, source in enumerate(countries[:25]):
            for j, target in enumerate(countries[25:]):
                large_data.append({
                    'source': source,
                    'target': target,
                    'trade_value_raw_usd': (i + 1) * (j + 1) * 1000,
                    'source_name': f'Country {source}',
                    'target_name': f'Country {target}',
                    'primary_data_source': 'import_reported'
                })
                
        large_df = pd.DataFrame(large_data)
        
        G = build_network_from_data(large_df, 2020)
        
        # 验证网络规模
        self.assertEqual(G.number_of_nodes(), 50)
        self.assertEqual(G.number_of_edges(), len(large_data))
        
        # 验证所有边都有权重
        for source, target in G.edges():
            self.assertGreater(G.edges[source, target]['weight'], 0)
            
    @patch('network_builder.get_country_region_safe')
    def test_region_assignment(self, mock_get_region):
        """测试区域分配"""
        mock_get_region.return_value = 'Test Region'
        
        G = build_network_from_data(self.valid_data, 2020)
        
        # 验证区域被正确分配
        for node in G.nodes():
            self.assertEqual(G.nodes[node]['region'], 'Test Region')
            
        # 验证函数被调用
        self.assertEqual(mock_get_region.call_count, G.number_of_nodes())


class TestNetworkBuilderEdgeCases(unittest.TestCase):
    """网络构建器边界情况测试"""
    
    def test_build_network_zero_weights(self):
        """测试零权重边"""
        zero_weight_data = pd.DataFrame({
            'source': ['USA'],
            'target': ['CHN'],
            'trade_value_raw_usd': [0],
            'source_name': ['United States'],
            'target_name': ['China'],
            'primary_data_source': ['import_reported']
        })
        
        G = build_network_from_data(zero_weight_data, 2020)
        
        self.assertEqual(G.number_of_edges(), 1)
        self.assertEqual(G.edges['USA', 'CHN']['weight'], 0)
        
    def test_build_network_self_loops(self):
        """测试自环处理"""
        self_loop_data = pd.DataFrame({
            'source': ['USA', 'CHN'],
            'target': ['USA', 'CHN'],  # 自环
            'trade_value_raw_usd': [1000, 2000],
            'source_name': ['United States', 'China'],
            'target_name': ['United States', 'China'],
            'primary_data_source': ['import_reported', 'import_reported']
        })
        
        G = build_network_from_data(self_loop_data, 2020)
        
        # 验证自环被正确处理
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertEqual(G.number_of_edges(), 2)
        self.assertTrue(G.has_edge('USA', 'USA'))
        self.assertTrue(G.has_edge('CHN', 'CHN'))
        
    def test_build_network_single_node(self):
        """测试单节点网络（只有自环）"""
        single_node_data = pd.DataFrame({
            'source': ['USA'],
            'target': ['USA'],
            'trade_value_raw_usd': [1000],
            'source_name': ['United States'],
            'target_name': ['United States'],
            'primary_data_source': ['import_reported']
        })
        
        G = build_network_from_data(single_node_data, 2020)
        
        self.assertEqual(G.number_of_nodes(), 1)
        self.assertEqual(G.number_of_edges(), 1)
        self.assertTrue(G.has_edge('USA', 'USA'))


if __name__ == '__main__':
    # 配置日志以避免测试时的输出干扰
    logging.getLogger().setLevel(logging.CRITICAL)
    
    unittest.main()