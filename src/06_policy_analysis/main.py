#!/usr/bin/env python3
"""
main.py - 主执行脚本
串联整个美国能源独立政策影响分析流程
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd

# 添加src路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入分析模块
from analysis import (
    load_and_prepare_data, 
    run_pre_post_analysis, 
    calculate_policy_impact_statistics,
    export_analysis_results
)
from plotting import (
    create_policy_impact_dashboard,
    plot_metric_timeseries,
    plot_period_comparison
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置常量
KEY_METRICS = [
    'in_strength', 'out_strength', 'total_strength',
    'betweenness_centrality', 'pagerank_centrality',
    'in_degree', 'out_degree', 'total_degree'
]

def determine_key_countries(df: pd.DataFrame, top_n: int = 10) -> List[str]:
    """
    根据整个研究周期的数据动态确定核心国家
    
    Args:
        df: 包含所有年份数据的DataFrame
        top_n: 选择前N个国家
        
    Returns:
        核心国家代码列表
    """
    
    # 按国家分组计算总进出口
    country_totals = df.groupby('country_code').agg({
        'in_strength': 'sum',
        'out_strength': 'sum'
    }).reset_index()
    
    # 获取进口前top_n国家
    top_importers = country_totals.nlargest(top_n, 'in_strength')['country_code'].tolist()
    
    # 获取出口前top_n国家  
    top_exporters = country_totals.nlargest(top_n, 'out_strength')['country_code'].tolist()
    
    # 合并并去重
    key_countries = list(set(top_importers + top_exporters))
    
    # 按总贸易额排序
    country_totals['total_trade'] = country_totals['in_strength'] + country_totals['out_strength']
    sorted_countries = country_totals.sort_values('total_trade', ascending=False)
    
    # 确保结果按重要性排序
    result = []
    for _, row in sorted_countries.iterrows():
        if row['country_code'] in key_countries:
            result.append(row['country_code'])
    
    logger.info(f"动态选定的核心国家: {result}")
    return result

def run_full_policy_analysis(data_filepath: str = None,
                           countries_list: Optional[List[str]] = None,
                           metrics_list: Optional[List[str]] = None,
                           output_tables_dir: str = None,
                           output_figures_dir: str = None,
                           generate_visualizations: bool = True) -> bool:
    """
    执行完整的政策影响分析流程
    
    Args:
        data_filepath: 数据文件路径
        countries_list: 分析的国家列表，None表示使用默认
        metrics_list: 分析的指标列表，None表示使用默认
        output_tables_dir: 表格输出目录
        output_figures_dir: 图表输出目录
        generate_visualizations: 是否生成可视化
        
    Returns:
        分析是否成功完成
    """
    logger.info("🚀 开始美国能源独立政策影响分析")
    logger.info("=" * 60)
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置默认路径
    if data_filepath is None:
        from pathlib import Path
        data_filepath = Path(__file__).parent.parent / "03_metrics" / "all_metrics.csv"
    if output_tables_dir is None:
        from pathlib import Path
        output_tables_dir = str(Path(__file__).parent)
    if output_figures_dir is None:
        from pathlib import Path
        output_figures_dir = str(Path(__file__).parent / "figures")
    
    # 第1步：加载和准备数据
    logger.info("\n📖 第1步：加载和准备数据...")
    df = load_and_prepare_data(data_filepath)
    logger.info(f"✅ 数据加载完成: {len(df)} 条记录")
    
    # 动态确定核心国家（如果未指定）
    if countries_list is None:
        countries_list = determine_key_countries(df, top_n=10)
    if metrics_list is None:
        metrics_list = KEY_METRICS
    
    logger.info(f"📊 分析配置:")
    logger.info(f"  数据文件: {data_filepath}")
    logger.info(f"  关注国家: {len(countries_list)} 个")
    logger.info(f"  分析指标: {len(metrics_list)} 个")
    logger.info(f"  生成可视化: {'是' if generate_visualizations else '否'}")
    
    try:
        # 第2步：执行事前-事后对比分析
        logger.info("\n🔍 第2步：执行事前-事后对比分析...")
        comparison_df = run_pre_post_analysis(df, countries_list, metrics_list)
        logger.info(f"✅ 对比分析完成: {len(comparison_df)} 个国家")
        
        # 第3步：计算政策影响统计量
        logger.info("\n📊 第3步：计算政策影响统计量...")
        statistics = calculate_policy_impact_statistics(df, comparison_df, metrics_list)
        logger.info("✅ 统计分析完成")
        
        # 第4步：导出分析结果
        logger.info("\n💾 第4步：导出分析结果...")
        exported_files = export_analysis_results(comparison_df, statistics, output_tables_dir)
        logger.info(f"✅ 结果导出完成: {len(exported_files)} 个文件")
        
        # 第5步：生成可视化（可选）
        if generate_visualizations:
            logger.info("\n📈 第5步：生成可视化...")
            try:
                visualization_files = create_policy_impact_dashboard(
                    df, comparison_df, statistics, countries_list, metrics_list, output_figures_dir
                )
                total_charts = sum(len(v) if isinstance(v, list) else 1 for v in visualization_files.values())
                logger.info(f"✅ 可视化完成: {total_charts} 个图表")
            except Exception as e:
                logger.warning(f"⚠️  可视化生成部分失败: {e}")
        
        # 输出结果摘要
        logger.info("\n🎯 分析结果摘要:")
        logger.info("-" * 40)
        
        # 显示显著变化的指标
        if 'significance_tests' in statistics:
            significant_metrics = [
                metric for metric, test in statistics['significance_tests'].items()
                if test.get('is_significant', False)
            ]
            logger.info(f"📊 统计显著的指标变化: {len(significant_metrics)} 个")
            for metric in significant_metrics[:5]:  # 显示前5个
                test_result = statistics['significance_tests'][metric]
                logger.info(f"  {metric}: p={test_result['p_value']:.4f}")
        
        # 显示变化最大的国家
        if len(comparison_df) > 0:
            # 以total_strength为例展示最大变化
            if 'total_strength_change' in comparison_df.columns:
                top_winners = comparison_df.nlargest(3, 'total_strength_change')
                top_losers = comparison_df.nsmallest(3, 'total_strength_change')
                
                logger.info(f"🏆 total_strength增长最大的国家:")
                for _, row in top_winners.iterrows():
                    logger.info(f"  {row['country_code']}: +{row['total_strength_change']:.2e}")
                
                logger.info(f"📉 total_strength下降最大的国家:")
                for _, row in top_losers.iterrows():
                    logger.info(f"  {row['country_code']}: {row['total_strength_change']:.2e}")
        
        # 显示输出文件位置
        logger.info(f"\n📁 输出文件位置:")
        for file_type, filepath in exported_files.items():
            logger.info(f"  {file_type}: {filepath}")
        
        if generate_visualizations:
            logger.info(f"  图表目录: {output_figures_dir}")
        
        logger.info("\n✅ 美国能源独立政策影响分析完成！")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_analysis(countries: List[str] = None, 
                      metrics: List[str] = None) -> bool:
    """
    快速分析：只生成核心结果，不生成大量可视化
    
    Args:
        countries: 分析的国家列表
        metrics: 分析的指标列表
        
    Returns:
        分析是否成功
    """
    logger.info("⚡ 执行快速政策影响分析...")
    
    return run_full_policy_analysis(
        countries_list=countries,
        metrics_list=metrics,
        generate_visualizations=False
    )

def run_visualization_only(data_filepath: str = None,
                          comparison_filepath: str = None,
                          statistics_filepath: str = None) -> bool:
    """
    仅生成可视化：基于已有的分析结果生成图表
    
    Args:
        data_filepath: 原始数据文件路径
        comparison_filepath: 对比分析结果文件路径
        statistics_filepath: 统计结果文件路径
        
    Returns:
        可视化是否成功
    """
    logger.info("📈 仅生成可视化...")
    
    # 设置默认路径
    if data_filepath is None:
        from pathlib import Path
        data_filepath = Path(__file__).parent.parent / "03_metrics" / "all_metrics.csv"
    if comparison_filepath is None:
        from pathlib import Path
        comparison_filepath = Path(__file__).parent / "policy_impact_summary.csv"
    if statistics_filepath is None:
        from pathlib import Path
        statistics_filepath = Path(__file__).parent / "policy_impact_statistics.json"
    
    try:
        # 加载数据
        import pandas as pd
        import json
        
        df = pd.read_csv(data_filepath)
        df['period'] = df['year'].apply(lambda x: 'pre' if 2001 <= x <= 2008 
                                       else ('transition' if 2009 <= x <= 2015 
                                            else 'post'))
        
        comparison_df = pd.read_csv(comparison_filepath)
        
        with open(statistics_filepath, 'r') as f:
            statistics = json.load(f)
        
        # 生成可视化
        visualization_files = create_policy_impact_dashboard(
            df, comparison_df, statistics, countries_list, metrics_list
        )
        
        total_charts = sum(len(v) if isinstance(v, list) else 1 for v in visualization_files.values())
        logger.info(f"✅ 可视化完成: {total_charts} 个图表")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 可视化生成失败: {e}")
        return False

def main():
    """主函数"""
    print("🌟 美国能源独立政策影响分析工具")
    print("=" * 50)
    print("请选择执行模式:")
    print("1. 完整分析 (包括可视化)")
    print("2. 快速分析 (仅核心结果)")
    print("3. 仅生成可视化")
    print("4. 使用默认设置执行")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        success = run_full_policy_analysis()
    elif choice == '2':
        success = run_quick_analysis()
    elif choice == '3':
        success = run_visualization_only()
    elif choice == '4':
        success = run_full_policy_analysis()
    else:
        print("❌ 无效选择")
        return
    
    if success:
        print("\n🎉 分析成功完成！")
    else:
        print("\n💥 分析过程中出现错误")

if __name__ == "__main__":
    # 直接运行完整分析
    success = run_full_policy_analysis()
    
    if success:
        print("\n🎉 政策影响分析成功完成！")
        print("📊 查看 06_policy_analysis 文件夹获取分析结果")
        print("📈 查看 06_policy_analysis 文件夹获取可视化图表")
    else:
        print("\n💥 分析过程中出现错误，请检查日志")
        sys.exit(1)