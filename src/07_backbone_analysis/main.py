#!/usr/bin/env python3
"""
骨干网络分析主程序
================

清晰的逻辑串联调用所有模块功能，实现"黄金中间点"的设计目标。
支持灵活的年份参数控制，兼顾"全面计算"和"重点可视化"。

核心流程：
1. 数据加载
2. 骨干网络提取（全面分析）
3. 稳健性检验（与轨道一对比）
4. 可视化生成（重点年份）
5. 报告生成

使用方法：
    python main.py                          # 默认分析
    python main.py --config config.json    # 使用配置文件
    python main.py --quick                  # 快速模式
    python main.py --full                   # 完整分析
    python main.py --years 2018,2020       # 指定年份

作者：Energy Network Analysis Team
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import traceback

# 导入所有整合模块
from config import (
    AnalysisConfig, 
    get_quick_demo_config, 
    get_full_analysis_config,
    get_validation_focused_config
)
from algorithms import batch_backbone_extraction
from validation import run_robustness_checks
from reporting import create_backbone_visualizations, generate_summary_report

# 数据加载工具
import networkx as nx
import numpy as np


def setup_logging(config: AnalysisConfig):
    """设置日志系统"""
    
    log_level = getattr(logging, config.log_level.upper())
    
    handlers = [logging.StreamHandler()]
    
    if config.log_to_file:
        log_file = config.output_path / f"backbone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 骨干网络分析系统启动")
    logger.info(f"   配置: {len(config.analysis_years)}年分析, {len(config.algorithms)}种算法")
    
    return logger


def load_network_data(config: AnalysisConfig, logger: logging.Logger) -> Dict[int, nx.Graph]:
    """
    加载网络数据
    
    Args:
        config: 分析配置
        logger: 日志器
        
    Returns:
        年份到网络的映射字典
    """
    
    logger.info("📂 开始加载网络数据...")
    
    networks = {}
    loaded_years = []
    
    for year in config.analysis_years:
        # 尝试从多个可能位置加载
        potential_paths = [
            config.data_path / "networks" / f"network_{year}.graphml",
            config.data_path / f"network_{year}.graphml", 
            config.data_path / f"{year}.graphml",
            Path("../02_net_analysis/outputs/networks") / f"network_{year}.graphml"
        ]
        
        loaded = False
        for path in potential_paths:
            if path.exists():
                try:
                    G = nx.read_graphml(path)
                    if G.number_of_nodes() > 0:
                        networks[year] = G
                        loaded_years.append(year)
                        logger.info(f"   ✅ {year}: {G.number_of_nodes()}节点, {G.number_of_edges()}边")
                        loaded = True
                        break
                except Exception as e:
                    logger.warning(f"   ⚠️ 加载{path}失败: {e}")
                    continue
        
        if not loaded:
            logger.warning(f"   ⚠️ {year}年数据未找到，将使用模拟数据")
    
    # 如果没有加载到真实数据，创建演示数据
    if not networks:
        logger.info("   创建演示数据...")
        networks = create_demo_networks(config.analysis_years, logger)
    
    logger.info(f"✅ 网络数据加载完成: {len(networks)}年")
    
    return networks


def create_demo_networks(years: List[int], logger: logging.Logger) -> Dict[int, nx.Graph]:
    """创建演示网络数据"""
    
    logger.info("   生成演示网络数据...")
    
    # 主要能源贸易国家
    countries = [
        'USA', 'CAN', 'MEX', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'NOR',
        'CHN', 'JPN', 'KOR', 'IND', 'SGP', 'AUS', 'SAU', 'ARE', 'QAT', 'KWT',
        'RUS', 'BRA', 'VEN', 'COL', 'ARG', 'NGA', 'AGO', 'LBY', 'DZA'
    ]
    
    # 地区映射
    region_map = {
        'North America': ['USA', 'CAN', 'MEX'],
        'Europe': ['GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'NOR', 'RUS'],
        'Asia': ['CHN', 'JPN', 'KOR', 'IND', 'SGP'],
        'Middle East': ['SAU', 'ARE', 'QAT', 'KWT'],
        'Latin America': ['BRA', 'VEN', 'COL', 'ARG'],
        'Africa': ['NGA', 'AGO', 'LBY', 'DZA'],
        'Oceania': ['AUS']
    }
    
    def same_region(c1, c2):
        for region, region_countries in region_map.items():
            if c1 in region_countries and c2 in region_countries:
                return True
        return False
    
    networks = {}
    
    for year in years:
        G = nx.Graph()
        G.add_nodes_from(countries)
        
        np.random.seed(42 + year)  # 确保可重现但有年份变化
        
        for i, c1 in enumerate(countries):
            for c2 in countries[i+1:]:
                # 贸易概率
                prob = 0.15
                
                # 美国相关贸易更频繁
                if 'USA' in [c1, c2]:
                    prob *= 2.5
                
                # 地区内贸易更频繁
                if same_region(c1, c2):
                    prob *= 1.8
                
                if np.random.random() < prob:
                    # 贸易量（美国在2011年后增长更快）
                    base_weight = np.random.exponential(50) * 1e6
                    
                    # 页岩革命效应
                    if 'USA' in [c1, c2] and year >= 2011:
                        growth_factor = 1.0 + (year - 2011) * 0.15
                        base_weight *= growth_factor
                    
                    G.add_edge(c1, c2, weight=base_weight)
        
        networks[year] = G
        logger.info(f"     {year}: {G.number_of_nodes()}节点, {G.number_of_edges()}边")
    
    return networks


def load_track1_results(config: AnalysisConfig, logger: logging.Logger) -> Optional[Dict]:
    """
    加载轨道一(03模块)的分析结果
    
    Args:
        config: 分析配置
        logger: 日志器
        
    Returns:
        轨道一结果字典，如果未找到则返回None
    """
    
    logger.info("🔗 尝试加载轨道一(03模块)分析结果...")
    
    # 尝试从多个可能位置加载03模块结果
    potential_paths = [
        Path("../03_metrics/all_metrics.csv"),
        Path("../03_metrics/node_centrality_metrics.csv"),
        config.data_path / "track1_results.json",
        config.output_path / "track1_results.json"
    ]
    
    for path in potential_paths:
        if path.exists():
            try:
                if path.suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(path)
                    logger.info(f"   ✅ 找到03模块结果: {path}")
                    # 将DataFrame转换为字典格式
                    return {'centrality_data': df.to_dict('records')}
                elif path.suffix == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"   ✅ 找到轨道一结果: {path}")
                    return data
            except Exception as e:
                logger.warning(f"   ⚠️ 加载{path}失败: {e}")
                continue
    
    logger.info("   ℹ️ 未找到轨道一结果，将跳过对比分析")
    return None


def run_backbone_analysis(networks: Dict[int, nx.Graph], 
                         config: AnalysisConfig, 
                         logger: logging.Logger) -> Dict[str, Dict[int, nx.Graph]]:
    """
    运行骨干网络提取分析
    
    Args:
        networks: 原始网络数据
        config: 分析配置
        logger: 日志器
        
    Returns:
        骨干网络结果
    """
    
    logger.info("🔗 开始骨干网络提取分析...")
    
    try:
        backbone_results = batch_backbone_extraction(
            networks=networks,
            alpha_values=config.alpha_values,
            beta=config.beta_value,
            weight_attr=config.weight_attr
        )
        
        # 统计结果
        total_backbones = sum(len(yearly_data) for yearly_data in backbone_results.values())
        logger.info(f"✅ 骨干网络提取完成: {total_backbones}个骨干网络")
        
        # 保存骨干网络（如果需要）
        if config.save_networks:
            save_backbone_networks(backbone_results, config, logger)
        
        return backbone_results
        
    except Exception as e:
        logger.error(f"❌ 骨干网络提取失败: {e}")
        logger.error(traceback.format_exc())
        raise


def save_backbone_networks(backbone_results: Dict[str, Dict[int, nx.Graph]], 
                          config: AnalysisConfig, 
                          logger: logging.Logger):
    """保存骨干网络到文件"""
    
    logger.info("💾 保存骨干网络...")
    
    networks_dir = config.output_path / "networks"
    networks_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    for algorithm, yearly_networks in backbone_results.items():
        alg_dir = networks_dir / algorithm
        alg_dir.mkdir(exist_ok=True)
        
        for year, network in yearly_networks.items():
            file_path = alg_dir / f"backbone_{algorithm}_{year}.graphml"
            try:
                nx.write_graphml(network, file_path)
                saved_count += 1
            except Exception as e:
                logger.warning(f"   ⚠️ 保存{file_path}失败: {e}")
    
    logger.info(f"   ✅ 已保存 {saved_count} 个骨干网络文件")


def run_analysis_pipeline(config: AnalysisConfig) -> Dict[str, Any]:
    """
    运行完整的分析流水线
    
    Args:
        config: 分析配置
        
    Returns:
        完整分析结果
    """
    
    # 设置日志和创建输出目录
    config.create_output_directories()
    logger = setup_logging(config)
    
    analysis_results = {
        'config': config.__dict__,
        'execution_time': {},
        'data_summary': {},
        'backbone_results': {},
        'validation_results': {},
        'visualization_paths': {},
        'report_path': '',
        'status': 'running',
        'start_time': datetime.now().isoformat()
    }
    
    try:
        # 1. 数据加载
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Phase 1: 数据加载")
        logger.info("=" * 60)
        
        networks = load_network_data(config, logger)
        track1_results = load_track1_results(config, logger)
        
        analysis_results['execution_time']['data_loading'] = (datetime.now() - start_time).total_seconds()
        analysis_results['data_summary'] = {
            'years_loaded': sorted(networks.keys()),
            'total_years': len(networks),
            'track1_available': track1_results is not None
        }
        
        # 2. 骨干网络提取
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Phase 2: 骨干网络提取")
        logger.info("=" * 60)
        
        backbone_networks = run_backbone_analysis(networks, config, logger)
        
        analysis_results['execution_time']['backbone_extraction'] = (datetime.now() - start_time).total_seconds()
        analysis_results['backbone_results'] = {
            'algorithms_applied': list(backbone_networks.keys()),
            'total_backbone_networks': sum(len(yearly) for yearly in backbone_networks.values())
        }
        
        # 3. 稳健性检验
        if config.run_validation:
            start_time = datetime.now()
            logger.info("=" * 60)
            logger.info("Phase 3: 稳健性检验")
            logger.info("=" * 60)
            
            validation_results = run_robustness_checks(
                full_networks=networks,
                backbone_networks=backbone_networks,
                track1_results=track1_results
            )
            
            analysis_results['execution_time']['validation'] = (datetime.now() - start_time).total_seconds()
            analysis_results['validation_results'] = validation_results
        else:
            logger.info("⏭️ 跳过稳健性检验（配置禁用）")
            analysis_results['validation_results'] = {}
        
        # 4. 可视化生成
        if config.create_visualizations:
            start_time = datetime.now()
            logger.info("=" * 60)
            logger.info("Phase 4: 可视化生成")
            logger.info("=" * 60)
            
            visualization_paths = create_backbone_visualizations(
                full_networks=networks,
                backbone_networks=backbone_networks,
                node_attributes=None,  # 将来可以从其他模块加载
                output_dir=config.figures_path,
                visualization_years=config.visualization_years
            )
            
            analysis_results['execution_time']['visualization'] = (datetime.now() - start_time).total_seconds()
            analysis_results['visualization_paths'] = visualization_paths
        else:
            logger.info("⏭️ 跳过可视化生成（配置禁用）")
            analysis_results['visualization_paths'] = {}
        
        # 5. 报告生成
        if config.generate_reports:
            start_time = datetime.now()
            logger.info("=" * 60)
            logger.info("Phase 5: 报告生成")
            logger.info("=" * 60)
            
            report_path = generate_summary_report(
                full_networks=networks,
                backbone_networks=backbone_networks,
                robustness_results=analysis_results.get('validation_results', {}),
                visualization_paths=analysis_results.get('visualization_paths', {}),
                output_dir=config.output_path
            )
            
            analysis_results['execution_time']['reporting'] = (datetime.now() - start_time).total_seconds()
            analysis_results['report_path'] = report_path
        else:
            logger.info("⏭️ 跳过报告生成（配置禁用）")
            analysis_results['report_path'] = ''
        
        # 分析完成
        analysis_results['status'] = 'completed'
        analysis_results['end_time'] = datetime.now().isoformat()
        analysis_results['total_time'] = sum(analysis_results['execution_time'].values())
        
        logger.info("=" * 60)
        logger.info("分析完成")
        logger.info("=" * 60)
        logger.info(f"✅ 总执行时间: {analysis_results['total_time']:.1f} 秒")
        
        # 保存分析摘要
        save_analysis_summary(analysis_results, config, logger)
        
    except Exception as e:
        logger.error(f"❌ 分析流程失败: {e}")
        logger.error(traceback.format_exc())
        analysis_results['status'] = 'failed'
        analysis_results['error'] = str(e)
        analysis_results['end_time'] = datetime.now().isoformat()
    
    return analysis_results


def save_analysis_summary(results: Dict[str, Any], 
                         config: AnalysisConfig, 
                         logger: logging.Logger):
    """保存分析摘要"""
    
    summary_path = config.output_path / "analysis_summary.json"
    
    try:
        # 处理不可序列化的对象
        serializable_results = {}
        for key, value in results.items():
            if key == 'config':
                # 转换Path对象为字符串
                config_dict = {}
                for k, v in value.items():
                    if isinstance(v, Path):
                        config_dict[k] = str(v)
                    else:
                        config_dict[k] = v
                serializable_results[key] = config_dict
            else:
                serializable_results[key] = value
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 分析摘要已保存: {summary_path}")
        
    except Exception as e:
        logger.error(f"❌ 摘要保存失败: {e}")


def print_results_summary(results: Dict[str, Any]):
    """打印结果摘要"""
    
    print("\n" + "=" * 70)
    print("                    骨干网络分析结果摘要")
    print("=" * 70)
    
    status = results.get('status', 'unknown')
    print(f"状态: {'✅ 成功完成' if status == 'completed' else '❌ 执行失败'}")
    
    if status == 'completed':
        total_time = results.get('total_time', 0)
        print(f"总时间: {total_time:.1f} 秒")
        
        # 数据摘要
        data_summary = results.get('data_summary', {})
        years = data_summary.get('years_loaded', [])
        if years:
            print(f"分析年份: {min(years)}-{max(years)} ({len(years)}年)")
        
        # 骨干网络结果
        backbone_summary = results.get('backbone_results', {})
        algorithms = backbone_summary.get('algorithms_applied', [])
        total_backbones = backbone_summary.get('total_backbone_networks', 0)
        print(f"算法: {', '.join(algorithms)}")
        print(f"骨干网络: {total_backbones}个")
        
        # 验证结果
        validation_summary = results.get('validation_results', {})
        if validation_summary:
            overall_assessment = validation_summary.get('overall_assessment', {})
            score = overall_assessment.get('total_score', 0)
            rating = overall_assessment.get('rating', 'unknown')
            print(f"稳健性: {score:.3f} ({rating.upper()})")
        
        # 可视化结果
        viz_paths = results.get('visualization_paths', {})
        total_viz = sum(len(paths) for paths in viz_paths.values())
        if total_viz > 0:
            print(f"可视化: {total_viz}个图表")
        
        # 报告路径
        report_path = results.get('report_path', '')
        if report_path:
            print(f"报告: {Path(report_path).name}")
    
    elif status == 'failed':
        error = results.get('error', '未知错误')
        print(f"错误: {error}")
    
    print("=" * 70)


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="骨干网络分析系统 v3.0 - 黄金中间点版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                           # 默认分析
  python main.py --config config.json     # 使用配置文件
  python main.py --quick                   # 快速演示模式
  python main.py --full                    # 完整分析模式
  python main.py --validation              # 验证重点模式
  python main.py --years 2018,2020        # 指定分析年份
  python main.py --viz-years 2018,2020    # 指定可视化年份
        """
    )
    
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--quick', action='store_true', help='快速演示模式')
    parser.add_argument('--full', action='store_true', help='完整分析模式')
    parser.add_argument('--validation', action='store_true', help='验证重点模式')
    parser.add_argument('--years', type=str, help='分析年份，逗号分隔 (如: 2018,2020)')
    parser.add_argument('--viz-years', type=str, help='可视化年份，逗号分隔')
    parser.add_argument('--output', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--data-path', type=str, help='数据路径')
    
    args = parser.parse_args()
    
    try:
        # 确定配置
        if args.config and Path(args.config).exists():
            config = AnalysisConfig.load_config(Path(args.config))
        elif args.quick:
            config = get_quick_demo_config()
        elif args.full:
            config = get_full_analysis_config()
        elif args.validation:
            config = get_validation_focused_config()
        else:
            config = AnalysisConfig()  # 默认配置
        
        # 覆盖命令行参数
        if args.years:
            try:
                years = [int(y.strip()) for y in args.years.split(',')]
                config.analysis_years = years
            except ValueError:
                print("❌ 年份格式错误，请使用逗号分隔的数字，如: 2018,2020")
                return 1
        
        if args.viz_years:
            try:
                viz_years = [int(y.strip()) for y in args.viz_years.split(',')]
                config.visualization_years = viz_years
            except ValueError:
                print("❌ 可视化年份格式错误")
                return 1
        
        if args.output:
            config.output_path = Path(args.output)
            config.figures_path = config.output_path / "figures"
        
        if args.data_path:
            config.data_path = Path(args.data_path)
        
        # 打印配置信息
        print("🚀 骨干网络分析系统 v3.0")
        print(f"   分析年份: {len(config.analysis_years)}年 ({min(config.analysis_years)}-{max(config.analysis_years)})")
        print(f"   可视化年份: {config.visualization_years}")
        print(f"   算法: {', '.join(config.algorithms)}")
        print(f"   输出目录: {config.output_path}")
        
        # 运行分析
        results = run_analysis_pipeline(config)
        
        # 打印结果摘要
        print_results_summary(results)
        
        return 0 if results['status'] == 'completed' else 1
        
    except KeyboardInterrupt:
        print("⚠️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)