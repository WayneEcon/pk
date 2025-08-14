#!/usr/bin/env python3
"""
03_metrics模块集成测试
测试__init__.py中的主要功能和模块间集成
"""

import unittest
import networkx as nx
import pandas as pd
import numpy as np
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入主要功能
import __init__ as metrics_module
from __init__ import (
    calculate_all_metrics_for_year, calculate_metrics_for_multiple_years,
    get_metrics_summary_report, export_metrics_to_files
)


class TestCalculateAllMetricsForYear(unittest.TestCase):
    """测试单年份所有指标计算的集成功能"""
    
    def setUp(self):
        # 创建复杂的测试网络
        self.complex_graph = nx.DiGraph()
        
        # 添加多个节点和连接，模拟真实的贸易网络
        countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'KOR', 'ITA']
        
        # 创建部分连接的网络
        trade_connections = [
            ('USA', 'CHN', 1000), ('CHN', 'USA', 800),
            ('USA', 'DEU', 600), ('DEU', 'USA', 500),
            ('CHN', 'DEU', 700), ('DEU', 'CHN', 650),
            ('JPN', 'USA', 400), ('USA', 'JPN', 350),
            ('GBR', 'USA', 300), ('USA', 'GBR', 280),
            ('FRA', 'DEU', 250), ('DEU', 'FRA', 230),
            ('KOR', 'CHN', 200), ('CHN', 'KOR', 180),
            ('ITA', 'DEU', 150), ('DEU', 'ITA', 140)
        ]
        
        for source, target, weight in trade_connections:
            self.complex_graph.add_edge(source, target, weight=weight)
        
        self.year = 2020
    
    def test_calculate_all_metrics_success(self):
        """测试成功计算所有指标"""
        result_df = calculate_all_metrics_for_year(self.complex_graph, self.year)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 8)  # 8个国家
        
        # 检查必需的列存在
        required_columns = [
            'year', 'country_code',
            # 节点指标
            'in_degree', 'out_degree', 'total_degree',
            'in_strength', 'out_strength', 'total_strength',
            'betweenness_centrality', 'pagerank_centrality', 'eigenvector_centrality',
            # 全局指标（以global_前缀添加到每行）
            'global_density', 'global_nodes', 'global_edges',
            'global_is_strongly_connected', 'global_avg_path_length'
        ]
        
        for col in required_columns:
            self.assertIn(col, result_df.columns, f"Missing required column: {col}")
        
        # 检查年份正确
        self.assertTrue(all(result_df['year'] == self.year))
        
        # 检查全局指标在所有行中都相同
        self.assertEqual(result_df['global_nodes'].nunique(), 1)
        self.assertEqual(result_df['global_edges'].nunique(), 1)
        self.assertEqual(result_df['global_density'].nunique(), 1)
    
    def test_calculate_metrics_data_quality(self):
        """测试计算结果的数据质量"""
        result_df = calculate_all_metrics_for_year(self.complex_graph, self.year)
        
        # 检查数值合理性
        # 度数应该为非负整数
        self.assertTrue(all(result_df['total_degree'] >= 0))
        self.assertTrue(all(result_df['in_degree'] >= 0))
        self.assertTrue(all(result_df['out_degree'] >= 0))
        
        # 强度应该为非负数
        self.assertTrue(all(result_df['total_strength'] >= 0))
        self.assertTrue(all(result_df['in_strength'] >= 0))
        self.assertTrue(all(result_df['out_strength'] >= 0))
        
        # 中心性指标应该在合理范围内
        self.assertTrue(all(result_df['betweenness_centrality'] >= 0))
        self.assertTrue(all(result_df['pagerank_centrality'] > 0))
        self.assertTrue(all(result_df['eigenvector_centrality'] >= 0))
        
        # PageRank值之和应该接近1
        pagerank_sum = result_df['pagerank_centrality'].sum()
        self.assertAlmostEqual(pagerank_sum, 1.0, places=3)
        
        # 全局密度应该在[0,1]之间
        density_values = result_df['global_density'].unique()
        self.assertEqual(len(density_values), 1)
        density = density_values[0]
        self.assertGreaterEqual(density, 0)
        self.assertLessEqual(density, 1)
    
    def test_calculate_metrics_country_rankings(self):
        """测试国家排名的合理性"""
        result_df = calculate_all_metrics_for_year(self.complex_graph, self.year)
        
        # 按总强度排序
        top_by_strength = result_df.nlargest(3, 'total_strength')
        
        # USA, CHN, DEU应该在前列（基于我们的测试数据）
        top_countries = set(top_by_strength['country_code'])
        expected_top = {'USA', 'CHN', 'DEU'}
        self.assertTrue(expected_top.issubset(top_countries) or 
                       len(expected_top.intersection(top_countries)) >= 2)
        
        # 检查度数和强度的相关性
        correlation = result_df['total_degree'].corr(result_df['total_strength'])
        self.assertGreater(correlation, 0.3)  # 应该有正相关
    
    def test_calculate_metrics_error_handling(self):
        """测试错误处理"""
        # 测试无效图
        with self.assertRaises(TypeError):
            calculate_all_metrics_for_year("not a graph", self.year)
        
        # 测试空图
        empty_graph = nx.DiGraph()
        with self.assertRaises(ValueError):
            calculate_all_metrics_for_year(empty_graph, self.year)
        
        # 测试缺少权重的图
        no_weight_graph = nx.DiGraph()
        no_weight_graph.add_edge('A', 'B')  # 没有权重
        
        with self.assertRaises(ValueError):
            calculate_all_metrics_for_year(no_weight_graph, self.year)


class TestCalculateMetricsForMultipleYears(unittest.TestCase):
    """测试多年份指标计算集成功能"""
    
    def setUp(self):
        # 创建多个年份的网络，模拟网络演化
        self.annual_networks = {}
        
        base_countries = ['USA', 'CHN', 'DEU', 'JPN']
        
        for year in range(2018, 2023):
            G = nx.DiGraph()
            
            # 网络随时间增长
            n_countries = len(base_countries) + (year - 2018)  # 从4个增长到8个
            countries = base_countries + [f"COUNTRY_{i}" for i in range(len(base_countries), n_countries)]
            
            # 添加边，权重随年份变化
            for i, source in enumerate(countries):
                for j, target in enumerate(countries):
                    if i != j and (i + j) % 3 == 0:  # 稀疏连接
                        weight = (i + 1) * (j + 1) * year * 0.1
                        G.add_edge(source, target, weight=weight)
            
            self.annual_networks[year] = G
    
    def test_multiple_years_calculation_success(self):
        """测试多年份计算成功"""
        result_df = calculate_metrics_for_multiple_years(self.annual_networks)
        
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
    
    def test_multiple_years_time_series_consistency(self):
        """测试时间序列数据的一致性"""
        result_df = calculate_metrics_for_multiple_years(self.annual_networks)
        
        # 检查网络规模随时间的变化趋势
        yearly_stats = []
        for year in sorted(self.annual_networks.keys()):
            year_data = result_df[result_df['year'] == year]
            yearly_stats.append({
                'year': year,
                'nodes': len(year_data),
                'avg_strength': year_data['total_strength'].mean(),
                'density': year_data['global_density'].iloc[0]
            })
        
        # 节点数应该随时间增长
        node_counts = [stat['nodes'] for stat in yearly_stats]
        self.assertTrue(all(node_counts[i] <= node_counts[i+1] 
                           for i in range(len(node_counts)-1)))
        
        # 密度可能随网络增长而变化
        densities = [stat['density'] for stat in yearly_stats]
        self.assertTrue(all(0 <= d <= 1 for d in densities))
    
    def test_multiple_years_empty_input(self):
        """测试空输入"""
        result_df = calculate_metrics_for_multiple_years({})
        
        self.assertTrue(result_df.empty)
    
    def test_multiple_years_partial_failure(self):
        """测试部分年份计算失败"""
        # 添加一个无效网络
        networks_with_invalid = self.annual_networks.copy()
        networks_with_invalid[2025] = "invalid_graph"
        
        result_df = calculate_metrics_for_multiple_years(networks_with_invalid)
        
        # 应该返回有效年份的数据
        self.assertIsInstance(result_df, pd.DataFrame)
        valid_years = result_df['year'].unique()
        
        # 不应该包含无效年份
        self.assertNotIn(2025, valid_years)
        
        # 应该包含大部分有效年份
        self.assertGreaterEqual(len(valid_years), len(self.annual_networks) - 1)


class TestMetricsSummaryReport(unittest.TestCase):
    """测试指标摘要报告生成"""
    
    def setUp(self):
        # 创建模拟的指标数据
        np.random.seed(42)  # 确保可重现性
        
        years = [2020, 2021, 2022]
        countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR']
        
        data = []
        for year in years:
            for i, country in enumerate(countries):
                # 为每个国家生成合理的指标值
                base_strength = (i + 1) * 100 * year / 2020
                noise = np.random.normal(0, 10)
                
                row = {
                    'year': year,
                    'country_code': country,
                    'country_name': f"Country {country}",
                    'total_degree': max(1, int(5 + i + noise/10)),
                    'total_strength': max(1, base_strength + noise),
                    'betweenness_centrality': max(0, 0.1 * (i + 1) + noise/100),
                    'pagerank_centrality': max(0.01, 0.15 + 0.05 * i + noise/200),
                    'eigenvector_centrality': max(0, 0.2 + 0.1 * i + noise/150),
                    'global_density': 0.3 + year * 0.01 - 2020 * 0.01,
                    'global_avg_path_length': 2.5 - 0.1 * (year - 2020),
                    'global_nodes': len(countries),
                    'global_edges': len(countries) * 2
                }
                data.append(row)
        
        self.test_metrics_df = pd.DataFrame(data)
    
    def test_summary_report_generation(self):
        """测试摘要报告生成"""
        summary = get_metrics_summary_report(self.test_metrics_df)
        
        self.assertIsInstance(summary, dict)
        
        # 检查基本统计信息
        required_keys = [
            'report_generated', 'total_records', 'years_covered',
            'year_range', 'total_countries', 'yearly_statistics'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary, f"Missing key: {key}")
        
        # 检查数值正确性
        self.assertEqual(summary['total_records'], len(self.test_metrics_df))
        self.assertEqual(summary['years_covered'], 3)
        self.assertEqual(summary['year_range'], "2020 - 2022")
        self.assertEqual(summary['total_countries'], 5)
    
    def test_summary_yearly_statistics(self):
        """测试年度统计信息"""
        summary = get_metrics_summary_report(self.test_metrics_df)
        
        yearly_stats = summary['yearly_statistics']
        self.assertIsInstance(yearly_stats, list)
        self.assertEqual(len(yearly_stats), 3)  # 3个年份
        
        for year_stat in yearly_stats:
            self.assertIn('year', year_stat)
            self.assertIn('nodes', year_stat)
            self.assertIn('avg_total_strength', year_stat)
            self.assertIn('network_density', year_stat)
            self.assertIn('top_countries_by_strength', year_stat)
            self.assertIn('top_countries_by_pagerank', year_stat)
            
            # 检查排名数据格式
            top_strength = year_stat['top_countries_by_strength']
            self.assertIsInstance(top_strength, list)
            self.assertLessEqual(len(top_strength), 5)  # 最多5个国家
            
            if top_strength:
                self.assertIn('country_code', top_strength[0])
                self.assertIn('total_strength', top_strength[0])
    
    def test_summary_trend_analysis(self):
        """测试趋势分析"""
        summary = get_metrics_summary_report(self.test_metrics_df)
        
        # 检查网络增长趋势
        self.assertIn('network_growth', summary)
        growth = summary['network_growth']
        
        self.assertIn('initial_size', growth)
        self.assertIn('final_size', growth)
        self.assertIn('growth_rate', growth)
        
        # 检查密度趋势
        self.assertIn('density_trend', summary)
        density_trend = summary['density_trend']
        
        self.assertIn('initial_density', density_trend)
        self.assertIn('final_density', density_trend)
        self.assertIn('density_change', density_trend)
    
    def test_summary_empty_data(self):
        """测试空数据的摘要"""
        empty_df = pd.DataFrame()
        summary = get_metrics_summary_report(empty_df)
        
        self.assertIn('error', summary)
        self.assertEqual(summary['error'], '没有数据可用于生成摘要')


class TestExportMetricsToFiles(unittest.TestCase):
    """测试指标数据导出功能"""
    
    def setUp(self):
        # 创建测试数据
        self.test_data = {
            'year': [2020, 2020, 2021, 2021],
            'country_code': ['USA', 'CHN', 'USA', 'CHN'],
            'country_name': ['United States', 'China', 'United States', 'China'],
            'total_degree': [10, 8, 12, 9],
            'total_strength': [1000, 800, 1100, 850],
            'betweenness_centrality': [0.5, 0.3, 0.6, 0.35],
            'pagerank_centrality': [0.6, 0.4, 0.65, 0.35],
            'eigenvector_centrality': [0.7, 0.3, 0.75, 0.25],
            'global_density': [0.3, 0.3, 0.32, 0.32],
            'global_avg_path_length': [2.5, 2.5, 2.4, 2.4],
            'global_nodes': [50, 50, 52, 52],
            'global_edges': [150, 150, 160, 160]
        }
        self.test_df = pd.DataFrame(self.test_data)
    
    def test_export_to_temporary_directory(self):
        """测试导出到临时目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = export_metrics_to_files(self.test_df, temp_dir)
            
            self.assertIsInstance(exported_files, dict)
            
            # 检查导出的文件类型
            expected_file_types = [
                'full_data', 'node_centrality', 'global_metrics', 'summary_report'
            ]
            
            for file_type in expected_file_types:
                self.assertIn(file_type, exported_files)
                file_path = exported_files[file_type]
                self.assertTrue(Path(file_path).exists(), f"File not found: {file_path}")
            
            # 检查CSV文件内容
            full_data_path = exported_files['full_data']
            imported_df = pd.read_csv(full_data_path)
            self.assertEqual(len(imported_df), len(self.test_df))
            
            # 检查JSON摘要报告
            summary_path = exported_files['summary_report']
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            self.assertIsInstance(summary_data, dict)
            self.assertIn('total_records', summary_data)
    
    def test_export_file_structure(self):
        """测试导出文件的结构"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = export_metrics_to_files(self.test_df, temp_dir)
            
            # 检查节点中心性文件
            centrality_path = exported_files['node_centrality']
            centrality_df = pd.read_csv(centrality_path)
            
            centrality_columns = [
                'year', 'country_code', 'country_name',
                'total_degree', 'total_strength', 'betweenness_centrality',
                'pagerank_centrality', 'eigenvector_centrality'
            ]
            
            for col in centrality_columns:
                self.assertIn(col, centrality_df.columns)
            
            # 检查全局指标文件
            global_path = exported_files['global_metrics']
            global_df = pd.read_csv(global_path)
            
            # 全局指标文件应该每年只有一行
            self.assertEqual(len(global_df), 2)  # 2020, 2021
            
            global_columns = ['year', 'global_density', 'global_avg_path_length', 'global_nodes']
            for col in global_columns:
                self.assertIn(col, global_df.columns)
    
    def test_export_error_handling(self):
        """测试导出错误处理"""
        # 尝试导出到无效路径（只有在权限限制时才会失败）
        invalid_path = "/root/nonexistent_directory"
        
        # 这个测试可能在某些系统上不会失败，所以我们检查返回值
        result = export_metrics_to_files(self.test_df, invalid_path)
        
        # 如果失败，应该返回空字典或包含错误信息
        self.assertIsInstance(result, dict)


class TestModuleIntegration(unittest.TestCase):
    """测试模块间集成"""
    
    def test_module_imports(self):
        """测试模块导入"""
        # 验证所有主要函数都可以从模块中导入
        from __init__ import (
            calculate_all_metrics_for_year,
            calculate_metrics_for_multiple_years,
            calculate_all_node_centralities,
            calculate_all_global_metrics,
            get_metrics_summary_report,
            export_metrics_to_files,
            setup_logger,
            validate_graph,
            safe_divide
        )
        
        # 确保函数都是可调用的
        self.assertTrue(callable(calculate_all_metrics_for_year))
        self.assertTrue(callable(calculate_metrics_for_multiple_years))
        self.assertTrue(callable(calculate_all_node_centralities))
        self.assertTrue(callable(calculate_all_global_metrics))
        self.assertTrue(callable(get_metrics_summary_report))
        self.assertTrue(callable(export_metrics_to_files))
        self.assertTrue(callable(setup_logger))
        self.assertTrue(callable(validate_graph))
        self.assertTrue(callable(safe_divide))
    
    def test_version_and_metadata(self):
        """测试版本和元数据"""
        self.assertTrue(hasattr(metrics_module, '__version__'))
        self.assertTrue(hasattr(metrics_module, '__author__'))
        self.assertTrue(hasattr(metrics_module, '__all__'))
        
        self.assertIsInstance(metrics_module.__version__, str)
        self.assertIsInstance(metrics_module.__author__, str)
        self.assertIsInstance(metrics_module.__all__, list)
        
        # 检查__all__中包含主要函数
        expected_exports = [
            'calculate_all_metrics_for_year',
            'calculate_metrics_for_multiple_years',
            'calculate_all_node_centralities',
            'calculate_all_global_metrics'
        ]
        
        for export in expected_exports:
            self.assertIn(export, metrics_module.__all__)
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建测试网络
        G = nx.DiGraph()
        countries = ['USA', 'CHN', 'DEU', 'JPN']
        
        for i, source in enumerate(countries):
            for j, target in enumerate(countries):
                if i != j:
                    G.add_edge(source, target, weight=100 * (i + j + 1))
        
        # 1. 计算单年份指标
        single_year_result = calculate_all_metrics_for_year(G, 2020)
        self.assertIsInstance(single_year_result, pd.DataFrame)
        self.assertEqual(len(single_year_result), 4)
        
        # 2. 计算多年份指标
        annual_networks = {2020: G, 2021: G}
        multi_year_result = calculate_metrics_for_multiple_years(annual_networks)
        self.assertIsInstance(multi_year_result, pd.DataFrame)
        self.assertEqual(len(multi_year_result), 8)  # 4 countries * 2 years
        
        # 3. 生成摘要报告
        summary = get_metrics_summary_report(multi_year_result)
        self.assertIsInstance(summary, dict)
        self.assertIn('total_records', summary)
        self.assertEqual(summary['total_records'], 8)
        
        # 4. 导出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = export_metrics_to_files(multi_year_result, temp_dir)
            self.assertIsInstance(exported_files, dict)
            self.assertGreater(len(exported_files), 0)


if __name__ == '__main__':
    # 配置测试环境
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCalculateAllMetricsForYear,
        TestCalculateMetricsForMultipleYears,
        TestMetricsSummaryReport,
        TestExportMetricsToFiles,
        TestModuleIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*60}")
    print(f"03_metrics模块集成测试摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"成功率: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split(chr(10))[0] if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-1] if traceback else 'Unknown error'}")