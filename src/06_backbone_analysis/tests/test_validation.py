#!/usr/bin/env python3
"""
验证模块单元测试
===============

测试 validation.py 中稳健性检验功能的正确性。
确保验证算法按预期工作。

测试覆盖：
1. 中心性一致性验证
2. 参数敏感性分析
3. 跨算法验证
4. 统计显著性检验

作者：Energy Network Analysis Team
"""

import unittest
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from validation import (
    run_robustness_checks,
    validate_centrality_consistency,
    analyze_parameter_sensitivity,
    validate_cross_algorithm_consistency,
    test_statistical_significance,
    calculate_node_centralities,
    compare_centrality_rankings,
    get_node_rank
)


class TestCentralityFunctions(unittest.TestCase):
    """测试中心性计算功能"""
    
    def setUp(self):
        """设置测试网络"""
        
        self.G = nx.Graph()
        
        # 创建测试网络
        edges = [
            ('USA', 'CAN', 100), ('USA', 'CHN', 80), ('USA', 'GBR', 60),
            ('CAN', 'CHN', 40), ('CAN', 'GBR', 30),
            ('CHN', 'GBR', 50), ('CHN', 'RUS', 70),
            ('GBR', 'DEU', 35), ('DEU', 'FRA', 25)
        ]
        
        for u, v, weight in edges:
            self.G.add_edge(u, v, weight=weight)
    
    def test_calculate_node_centralities(self):
        """测试节点中心性计算"""
        
        centralities = calculate_node_centralities(self.G)
        
        # 检查返回的中心性类型
        expected_metrics = ['degree', 'strength', 'betweenness', 'pagerank', 'closeness']
        for metric in expected_metrics:
            self.assertIn(metric, centralities)
            self.assertIsInstance(centralities[metric], dict)
        
        # 检查所有节点都有中心性值
        for metric in expected_metrics:
            for node in self.G.nodes():
                self.assertIn(node, centralities[metric])
                self.assertIsInstance(centralities[metric][node], (int, float))
        
        # 检查度中心性计算正确
        self.assertEqual(centralities['degree']['USA'], self.G.degree('USA'))
        
        # 检查强度中心性计算正确
        expected_strength = self.G.degree('USA', weight='weight')
        self.assertEqual(centralities['strength']['USA'], expected_strength)
    
    def test_compare_centrality_rankings(self):
        """测试中心性排名比较"""
        
        # 创建两个相似的中心性字典
        centrality1 = {
            'strength': {'USA': 240, 'CHN': 200, 'GBR': 145, 'CAN': 170, 'RUS': 70}
        }
        
        centrality2 = {
            'strength': {'USA': 250, 'CHN': 190, 'GBR': 140, 'CAN': 160, 'RUS': 65}
        }
        
        result = compare_centrality_rankings(centrality1, centrality2, 'strength')
        
        # 检查返回结果包含必要字段
        expected_fields = ['spearman_rho', 'spearman_pvalue', 'kendall_tau', 'kendall_pvalue', 'common_nodes']
        for field in expected_fields:
            self.assertIn(field, result)
        
        # 检查相关系数在合理范围内
        self.assertTrue(-1 <= result['spearman_rho'] <= 1)
        self.assertTrue(0 <= result['spearman_pvalue'] <= 1)
        self.assertTrue(-1 <= result['kendall_tau'] <= 1)
        
        # 相似的排名应该有高相关性
        self.assertGreater(result['spearman_rho'], 0.5)
    
    def test_get_node_rank(self):
        """测试节点排名获取"""
        
        centrality_dict = {'A': 10, 'B': 20, 'C': 15, 'D': 5}
        
        # B应该排名第1（最高值）
        self.assertEqual(get_node_rank(centrality_dict, 'B'), 1)
        
        # C应该排名第2
        self.assertEqual(get_node_rank(centrality_dict, 'C'), 2)
        
        # A应该排名第3
        self.assertEqual(get_node_rank(centrality_dict, 'A'), 3)
        
        # D应该排名第4
        self.assertEqual(get_node_rank(centrality_dict, 'D'), 4)
        
        # 不存在的节点应该返回最后排名+1
        self.assertEqual(get_node_rank(centrality_dict, 'E'), 5)


class TestValidationFunctions(unittest.TestCase):
    """测试验证功能"""
    
    def setUp(self):
        """设置测试数据"""
        
        # 创建多年网络数据
        self.full_networks = {}
        self.backbone_networks = {
            'disparity_filter_0.05': {},
            'mst': {}
        }
        
        years = [2018, 2019, 2020]
        
        for year in years:
            # 创建完整网络
            G_full = nx.Graph()
            
            countries = ['USA', 'CHN', 'CAN', 'GBR', 'DEU', 'RUS']
            
            # 添加边（模拟贸易网络）
            np.random.seed(42 + year)
            for i, c1 in enumerate(countries):
                for c2 in countries[i+1:]:
                    if np.random.random() < 0.6:  # 60%连接概率
                        weight = np.random.exponential(50)
                        # 美国在后期权重更大
                        if 'USA' in [c1, c2] and year >= 2019:
                            weight *= 1.5
                        G_full.add_edge(c1, c2, weight=weight)
            
            self.full_networks[year] = G_full
            
            # 创建骨干网络（移除一些边）
            G_backbone_df = G_full.copy()
            edges_to_remove = list(G_full.edges())[:len(G_full.edges())//3]  # 移除1/3边
            G_backbone_df.remove_edges_from(edges_to_remove)
            self.backbone_networks['disparity_filter_0.05'][year] = G_backbone_df
            
            # MST骨干网络
            G_backbone_mst = nx.Graph()
            G_backbone_mst.add_nodes_from(G_full.nodes())
            # 添加部分边来模拟MST
            mst_edges = list(G_full.edges())[:len(countries)-1]
            for u, v in mst_edges:
                G_backbone_mst.add_edge(u, v, weight=G_full[u][v]['weight'])
            self.backbone_networks['mst'][year] = G_backbone_mst
    
    def test_validate_centrality_consistency(self):
        """测试中心性一致性验证"""
        
        common_years = [2018, 2019, 2020]
        
        result = validate_centrality_consistency(
            self.full_networks, 
            self.backbone_networks, 
            common_years
        )
        
        # 检查返回结果结构
        expected_fields = ['algorithm_correlations', 'usa_rank_analysis', 'overall_consistency_score']
        for field in expected_fields:
            self.assertIn(field, result)
        
        # 检查算法相关性结果
        for algorithm in self.backbone_networks.keys():
            if algorithm in result['algorithm_correlations']:
                alg_result = result['algorithm_correlations'][algorithm]
                self.assertIn('mean_spearman_rho', alg_result)
                self.assertIn('meets_threshold', alg_result)
                self.assertIsInstance(alg_result['mean_spearman_rho'], (int, float))
                self.assertIsInstance(alg_result['meets_threshold'], bool)
        
        # 检查总体一致性分数
        self.assertIsInstance(result['overall_consistency_score'], (int, float))
        self.assertTrue(0 <= result['overall_consistency_score'] <= 1)
    
    def test_analyze_parameter_sensitivity(self):
        """测试参数敏感性分析"""
        
        # 添加多个DF算法结果
        backbone_networks_extended = self.backbone_networks.copy()
        backbone_networks_extended['disparity_filter_0.01'] = {}
        backbone_networks_extended['disparity_filter_0.1'] = {}
        
        for year in [2018, 2019, 2020]:
            # 创建不同alpha值的结果（模拟更严格和更宽松的过滤）
            G_strict = self.backbone_networks['disparity_filter_0.05'][year].copy()
            edges_to_remove = list(G_strict.edges())[:len(G_strict.edges())//2]
            G_strict.remove_edges_from(edges_to_remove)
            backbone_networks_extended['disparity_filter_0.01'][year] = G_strict
            
            G_loose = self.full_networks[year].copy()
            edges_to_remove = list(G_loose.edges())[:len(G_loose.edges())//4]
            G_loose.remove_edges_from(edges_to_remove)
            backbone_networks_extended['disparity_filter_0.1'][year] = G_loose
        
        result = analyze_parameter_sensitivity(backbone_networks_extended, [2018, 2019, 2020])
        
        # 检查返回结果结构
        expected_fields = ['alpha_stability', 'core_findings_stability', 'usa_degree_stability']
        for field in expected_fields:
            self.assertIn(field, result)
    
    def test_validate_cross_algorithm_consistency(self):
        """测试跨算法一致性验证"""
        
        result = validate_cross_algorithm_consistency(
            self.backbone_networks, 
            [2018, 2019, 2020]
        )
        
        # 检查返回结果结构
        expected_fields = ['algorithm_pairs', 'usa_position_consistency', 'algorithm_consistency_score']
        for field in expected_fields:
            self.assertIn(field, result)
        
        # 检查算法配对比较
        self.assertIsInstance(result['algorithm_pairs'], dict)
        
        # 检查一致性分数
        if 'algorithm_consistency_score' in result and result['algorithm_consistency_score']:
            self.assertTrue(0 <= result['algorithm_consistency_score'] <= 1)
    
    def test_test_statistical_significance(self):
        """测试统计显著性检验"""
        
        result = test_statistical_significance(
            self.full_networks,
            self.backbone_networks,
            [2018, 2019, 2020]
        )
        
        # 检查返回结果结构
        expected_fields = ['usa_position_change', 'temporal_trends', 'overall_significance']
        for field in expected_fields:
            self.assertIn(field, result)
        
        # 检查总体显著性
        self.assertIsInstance(result['overall_significance'], bool)


class TestRobustnessChecks(unittest.TestCase):
    """测试完整的稳健性检验"""
    
    def setUp(self):
        """设置测试数据"""
        
        # 创建简单的测试网络
        self.full_networks = {}
        self.backbone_networks = {
            'disparity_filter_0.05': {},
            'mst': {}
        }
        
        for year in [2018, 2019, 2020]:
            # 完整网络
            G_full = nx.Graph()
            G_full.add_edge('USA', 'CHN', weight=100)
            G_full.add_edge('USA', 'CAN', weight=80)
            G_full.add_edge('CHN', 'CAN', weight=60)
            G_full.add_edge('USA', 'GBR', weight=40 + year - 2018)  # 时间增长
            
            self.full_networks[year] = G_full
            
            # 骨干网络
            G_backbone = G_full.copy()
            if year == 2018:  # 2018年移除一条边
                G_backbone.remove_edge('CHN', 'CAN')
            
            self.backbone_networks['disparity_filter_0.05'][year] = G_backbone
            
            # MST
            G_mst = nx.Graph()
            G_mst.add_nodes_from(G_full.nodes())
            G_mst.add_edge('USA', 'CHN', weight=100)
            G_mst.add_edge('USA', 'CAN', weight=80)
            G_mst.add_edge('USA', 'GBR', weight=40 + year - 2018)
            
            self.backbone_networks['mst'][year] = G_mst
    
    def test_run_robustness_checks(self):
        """测试完整稳健性检验流程"""
        
        result = run_robustness_checks(
            self.full_networks,
            self.backbone_networks,
            track1_results=None
        )
        
        # 检查主要结果结构
        expected_main_fields = [
            'centrality_consistency',
            'parameter_sensitivity', 
            'cross_algorithm_validation',
            'statistical_significance',
            'track1_comparison',
            'overall_assessment'
        ]
        
        for field in expected_main_fields:
            self.assertIn(field, result)
        
        # 检查总体评估
        overall = result['overall_assessment']
        self.assertIn('total_score', overall)
        self.assertIn('rating', overall)
        self.assertIn('meets_academic_standards', overall)
        
        # 检查分数范围
        self.assertTrue(0 <= overall['total_score'] <= 1)
        self.assertIn(overall['rating'], ['low', 'moderate', 'high', 'excellent'])
        self.assertIsInstance(overall['meets_academic_standards'], bool)
    
    def test_run_robustness_checks_with_track1(self):
        """测试带有轨道一数据的稳健性检验"""
        
        # 模拟轨道一数据
        track1_data = {
            'centrality_data': [
                {'year': 2018, 'country': 'USA', 'centrality': 0.8},
                {'year': 2018, 'country': 'CHN', 'centrality': 0.6},
                {'year': 2019, 'country': 'USA', 'centrality': 0.85},
                {'year': 2019, 'country': 'CHN', 'centrality': 0.65}
            ]
        }
        
        result = run_robustness_checks(
            self.full_networks,
            self.backbone_networks,
            track1_results=track1_data
        )
        
        # 应该包含轨道一对比结果
        self.assertIn('track1_comparison', result)
        track1_comparison = result['track1_comparison']
        self.assertIsInstance(track1_comparison, dict)


class TestEdgeCases(unittest.TestCase):
    """测试边缘情况"""
    
    def test_empty_networks(self):
        """测试空网络的验证"""
        
        empty_full = {}
        empty_backbone = {}
        
        result = run_robustness_checks(empty_full, empty_backbone)
        
        # 应该能处理空网络而不报错
        self.assertIsInstance(result, dict)
    
    def test_single_year_networks(self):
        """测试单年网络的验证"""
        
        G = nx.Graph()
        G.add_edge('A', 'B', weight=10)
        
        full_networks = {2020: G}
        backbone_networks = {'test_alg': {2020: G.copy()}}
        
        result = run_robustness_checks(full_networks, backbone_networks)
        
        # 应该能处理单年数据
        self.assertIsInstance(result, dict)
    
    def test_disconnected_networks(self):
        """测试非连通网络的验证"""
        
        G = nx.Graph()
        G.add_edge('A', 'B', weight=10)
        G.add_edge('C', 'D', weight=20)  # 独立的连通分量
        
        full_networks = {2020: G}
        backbone_networks = {'test_alg': {2020: G.copy()}}
        
        result = run_robustness_checks(full_networks, backbone_networks)
        
        # 应该能处理非连通网络
        self.assertIsInstance(result, dict)


def run_all_tests():
    """运行所有测试"""
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestCentralityFunctions,
        TestValidationFunctions,
        TestRobustnessChecks,
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
        print("\n✅ 所有验证测试通过!")
    else:
        print("\n❌ 部分测试失败")
        exit(1)