#!/usr/bin/env python3
"""
03_metrics模块安装验证脚本

快速验证模块是否正确安装和配置。
"""

import sys
import networkx as nx
import pandas as pd
import numpy as np

# 测试基本导入
try:
    import utils
    import node_metrics
    import global_metrics
    from utils import setup_logger, validate_graph, safe_divide
    from node_metrics import calculate_all_node_centralities
    from global_metrics import calculate_all_global_metrics
    # 导入主模块的函数
    import __init__ as main_module
    calculate_all_metrics_for_year = main_module.calculate_all_metrics_for_year
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def create_test_network():
    """创建测试网络"""
    G = nx.DiGraph()
    edges = [
        ('USA', 'CHN', 1000),
        ('USA', 'DEU', 500), 
        ('CHN', 'USA', 800),
        ('CHN', 'JPN', 300),
        ('DEU', 'FRA', 400),
        ('JPN', 'KOR', 200),
        ('FRA', 'ITA', 150),
        ('KOR', 'JPN', 180)
    ]
    
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)
    
    return G

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    # 创建测试网络
    G = create_test_network()
    print(f"创建测试网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    try:
        # 测试节点指标计算
        node_metrics = calculate_all_node_centralities(G, 2020)
        print(f"✅ 节点指标计算成功: {len(node_metrics)} 行数据")
        
        # 测试全局指标计算
        global_metrics = calculate_all_global_metrics(G, 2020)
        print(f"✅ 全局指标计算成功: {len(global_metrics)} 个指标")
        
        # 测试统一接口
        all_metrics = calculate_all_metrics_for_year(G, 2020)
        print(f"✅ 统一接口计算成功: {len(all_metrics)} 行, {len(all_metrics.columns)} 列")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False

def test_data_consistency():
    """测试数据一致性"""
    print("\n🔍 测试数据一致性...")
    
    G = create_test_network()
    result_df = calculate_all_metrics_for_year(G, 2020)
    
    try:
        # 检查PageRank归一化
        pagerank_sum = result_df['pagerank_centrality'].sum()
        assert abs(pagerank_sum - 1.0) < 0.01, f"PageRank求和不为1: {pagerank_sum}"
        print("✅ PageRank归一化正确")
        
        # 检查度数一致性
        total_in_degree = result_df['in_degree'].sum()
        total_out_degree = result_df['out_degree'].sum()
        assert total_in_degree == total_out_degree, "入度总和不等于出度总和"
        print("✅ 度数一致性正确")
        
        # 检查强度计算
        for _, row in result_df.iterrows():
            assert row['total_strength'] == row['in_strength'] + row['out_strength']
        print("✅ 强度计算正确")
        
        return True
        
    except AssertionError as e:
        print(f"❌ 数据一致性测试失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 一致性测试异常: {e}")
        return False

def test_algorithm_correctness():
    """测试算法修正的正确性"""
    print("\n🎯 测试算法修正...")
    
    # 创建简单的三节点网络测试加权路径
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=1)    # 低权重 -> 长距离
    G.add_edge('B', 'C', weight=100)  # 高权重 -> 短距离  
    G.add_edge('A', 'C', weight=10)   # 中等权重 -> 中等距离
    
    try:
        global_metrics = calculate_all_global_metrics(G, 2020)
        
        # 检查加权路径长度是否合理
        weighted_path_length = global_metrics.get('avg_weighted_path_length', 0)
        assert weighted_path_length > 0, "加权路径长度应该大于0"
        print("✅ 加权路径计算正确")
        
        # 检查网络效率
        efficiency = global_metrics.get('weighted_global_efficiency', 0)
        assert 0 <= efficiency <= 1, f"网络效率超出范围: {efficiency}"
        print("✅ 网络效率计算正确")
        
        return True
        
    except Exception as e:
        print(f"❌ 算法测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 03_metrics 模块验证")
    print("=" * 60)
    
    # 设置日志级别
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    # 运行测试
    tests = [
        ("基本功能", test_basic_functionality),
        ("数据一致性", test_data_consistency),
        ("算法修正", test_algorithm_correctness)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed_tests += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 验证总结")
    print("=" * 60)
    print(f"通过测试: {passed_tests}/{total_tests}")
    success_rate = (passed_tests / total_tests) * 100
    print(f"成功率: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有验证通过！03_metrics 模块工作正常。")
        print("\n✨ 模块已准备好用于生产使用。")
        sys.exit(0)
    else:
        print("⚠️  部分验证失败，请检查模块配置。")
        sys.exit(1)

if __name__ == '__main__':
    main()