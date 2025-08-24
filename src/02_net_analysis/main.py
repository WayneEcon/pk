#!/usr/bin/env python3
"""
全球能源贸易网络构建主流程 (Network Construction Main Pipeline)
===========================================================

负责将清洗后的年度原始贸易数据转化为结构化的全球能源贸易网络时间序列。

核心功能：
1. 加载经01模块清洗的贸易数据
2. 应用"进口优先"原则处理数据一致性
3. 使用美国GDP平减指数进行通胀调整
4. 构建年度加权有向图网络
5. 计算网络统计指标
6. 保存多种格式的网络文件和统计报告

版本：v2.0 (Complete Network Construction Pipeline)
作者：Energy Network Analysis Team
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import networkx as nx
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入模块组件
from data_loader import load_cleaned_trade_data, load_gdp_deflator
from data_processor import process_trade_data_with_deflator, aggregate_trade_flows
from network_builder import build_network_from_data
from network_stats import calculate_network_statistics
from output_manager import save_networks_comprehensive, generate_summary_report

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_outputs_directory() -> Path:
    """创建模块输出目录"""
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir

def build_annual_networks(start_year: int = 2001, end_year: int = 2024) -> Dict[int, nx.DiGraph]:
    """
    构建指定年份范围的年度网络
    
    Args:
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        Dict[int, nx.DiGraph]: 年份到网络图的字典
    """
    logger.info(f"🚀 开始构建年度网络 ({start_year}-{end_year})")
    
    # 确保输出目录存在
    outputs_dir = create_outputs_directory()
    
    annual_networks = {}
    network_stats = []
    
    # 加载GDP平减指数
    logger.info("📊 加载GDP平减指数...")
    gdp_deflator = load_gdp_deflator()
    
    for year in range(start_year, end_year + 1):
        try:
            logger.info(f"🔄 处理 {year} 年数据...")
            
            # 1. 加载清洗后的贸易数据
            trade_data = load_cleaned_trade_data(year)
            
            if trade_data.empty:
                logger.warning(f"   ⚠️ {year} 年数据为空，跳过")
                continue
            
            # 2. 应用通胀调整和数据一致性处理
            processed_data = process_trade_data_with_deflator(trade_data, gdp_deflator, year)
            
            # 3. 聚合贸易流（应用"进口优先"原则）
            aggregated_data = aggregate_trade_flows(processed_data, year)
            
            # 4. 构建网络图
            network_graph = build_network_from_data(aggregated_data, year)
            
            # 5. 计算网络统计指标
            stats = calculate_network_statistics(network_graph, year)
            
            # 保存到集合中
            annual_networks[year] = network_graph
            network_stats.append(stats)
            
            logger.info(f"   ✅ {year} 年网络构建完成: {network_graph.number_of_nodes()} 节点, {network_graph.number_of_edges()} 边")
            
        except Exception as e:
            logger.error(f"   ❌ {year} 年网络构建失败: {str(e)}")
            continue
    
    logger.info(f"🎯 年度网络构建完成，成功构建 {len(annual_networks)} 个年度网络")
    
    return annual_networks, network_stats


def main():
    """主函数：执行完整的网络构建流程"""
    
    logger.info("=" * 60)
    logger.info("🌍 全球能源贸易网络构建系统 v2.0")
    logger.info("=" * 60)
    
    try:
        # 构建年度网络
        annual_networks, network_stats = build_annual_networks()
        
        if not annual_networks:
            logger.error("❌ 没有成功构建任何网络，程序退出")
            return
        
        # 保存网络数据和统计结果
        logger.info("💾 保存网络数据和统计结果...")
        save_networks_comprehensive(annual_networks, network_stats)
        
        # 生成摘要报告
        logger.info("📄 生成摘要报告...")
        generate_summary_report(annual_networks, network_stats)
        
        # 输出完成信息
        logger.info("🎉 网络构建流程完成!")
        logger.info(f"📊 成功构建 {len(annual_networks)} 个年度网络")
        logger.info(f"📁 所有输出文件已保存到相应目录")
        
        # 显示网络概况
        stats_df = pd.DataFrame(network_stats)
        logger.info(f"📈 网络规模范围: {stats_df['nodes'].min()}-{stats_df['nodes'].max()} 节点")
        logger.info(f"🔗 贸易关系范围: {stats_df['edges'].min()}-{stats_df['edges'].max()} 边")
        
    except Exception as e:
        logger.error(f"❌ 网络构建过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()