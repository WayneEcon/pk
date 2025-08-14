#!/usr/bin/env python3
"""
analysis.py - 核心统计分析功能
实现美国能源独立政策影响的事前-事后对比分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from scipy import stats
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    加载并准备分析数据
    
    Args:
        filepath: all_metrics.csv文件路径
        
    Returns:
        添加了period列的DataFrame
        
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当数据格式不正确时
    """
    logger.info(f"📖 加载数据文件: {filepath}")
    
    # 检查文件是否存在
    if not Path(filepath).exists():
        raise FileNotFoundError(f"找不到数据文件: {filepath}")
    
    # 加载数据
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ 成功加载数据: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        raise ValueError(f"读取数据文件失败: {e}")
    
    # 验证必要列
    required_columns = ['year', 'country_code']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {missing_columns}")
    
    # 添加政策期间标识
    df = df.copy()
    
    def assign_period(year: int) -> str:
        """根据年份分配政策期间"""
        if 2001 <= year <= 2008:
            return 'pre'
        elif 2009 <= year <= 2015:
            return 'transition'
        elif 2016 <= year <= 2024:
            return 'post'
        else:
            return 'unknown'
    
    df['period'] = df['year'].apply(assign_period)
    
    # 统计各期间的数据
    period_counts = df['period'].value_counts()
    logger.info(f"📊 政策期间数据分布:")
    for period, count in period_counts.items():
        logger.info(f"  {period}: {count} 条记录")
    
    # 验证年份范围
    year_range = (df['year'].min(), df['year'].max())
    logger.info(f"📅 数据年份范围: {year_range[0]} - {year_range[1]}")
    
    return df

def run_pre_post_analysis(df: pd.DataFrame, 
                         countries_of_interest: List[str] = None,
                         metrics_of_interest: List[str] = None) -> pd.DataFrame:
    """
    执行事前-事后对比分析
    
    Args:
        df: 包含period列的完整数据
        countries_of_interest: 关注的国家列表，None表示所有国家
        metrics_of_interest: 关注的指标列表，None表示所有数值指标
        
    Returns:
        包含对比分析结果的DataFrame
    """
    logger.info("🔍 开始事前-事后对比分析...")
    
    # 筛选数据
    analysis_df = df.copy()
    
    if countries_of_interest:
        analysis_df = analysis_df[analysis_df['country_code'].isin(countries_of_interest)]
        logger.info(f"🌍 分析国家: {len(countries_of_interest)} 个")
    
    # 只保留pre和post期间
    analysis_df = analysis_df[analysis_df['period'].isin(['pre', 'post'])]
    
    if len(analysis_df) == 0:
        raise ValueError("筛选后没有可分析的数据")
    
    # 识别数值指标列
    if metrics_of_interest is None:
        # 自动识别数值列，排除标识列
        exclude_columns = ['year', 'country_code', 'country_name', 'period']
        numeric_columns = analysis_df.select_dtypes(include=[np.number]).columns
        metrics_of_interest = [col for col in numeric_columns if col not in exclude_columns]
    
    logger.info(f"📊 分析指标: {len(metrics_of_interest)} 个")
    
    # 按country_code和period分组计算均值
    logger.info("📈 计算期间均值...")
    grouped = analysis_df.groupby(['country_code', 'period'])[metrics_of_interest].mean().reset_index()
    
    # 重塑数据：将period作为列
    logger.info("🔄 重塑数据格式...")
    pivot_df = grouped.pivot(index='country_code', columns='period', values=metrics_of_interest)
    
    # 扁平化列名
    pivot_df.columns = [f'{metric}_{period}' for metric, period in pivot_df.columns]
    pivot_df.reset_index(inplace=True)
    
    # 计算变化量和变化率
    logger.info("📊 计算变化指标...")
    for metric in metrics_of_interest:
        pre_col = f'{metric}_pre'
        post_col = f'{metric}_post'
        
        if pre_col in pivot_df.columns and post_col in pivot_df.columns:
            # 绝对变化
            pivot_df[f'{metric}_change'] = pivot_df[post_col] - pivot_df[pre_col]
            
            # 相对变化（百分比）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pivot_df[f'{metric}_pct_change'] = (
                    (pivot_df[post_col] - pivot_df[pre_col]) / 
                    pivot_df[pre_col].abs() * 100
                )
                # 处理无穷大和NaN值
                pivot_df[f'{metric}_pct_change'] = pivot_df[f'{metric}_pct_change'].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"✅ 对比分析完成: {len(pivot_df)} 个国家")
    
    return pivot_df

def calculate_policy_impact_statistics(df: pd.DataFrame, 
                                     comparison_df: pd.DataFrame,
                                     metrics_of_interest: List[str] = None) -> Dict[str, Any]:
    """
    计算政策影响统计量
    
    Args:
        df: 原始数据（包含period列）
        comparison_df: 对比分析结果
        metrics_of_interest: 关注的指标列表
        
    Returns:
        包含统计结果的字典
    """
    logger.info("📊 计算政策影响统计量...")
    
    if metrics_of_interest is None:
        # 从comparison_df中提取指标名
        change_cols = [col for col in comparison_df.columns if col.endswith('_change')]
        metrics_of_interest = [col.replace('_change', '') for col in change_cols]
    
    statistics = {
        'summary': {},
        'significance_tests': {},
        'top_winners': {},
        'top_losers': {},
        'period_aggregates': {}
    }
    
    # 1. 基本统计摘要
    logger.info("📈 计算基本统计摘要...")
    for metric in metrics_of_interest:
        change_col = f'{metric}_change'
        pct_change_col = f'{metric}_pct_change'
        
        if change_col in comparison_df.columns:
            changes = comparison_df[change_col].dropna()
            
            # 检查百分比变化列是否存在
            pct_changes = pd.Series(dtype=float)
            if pct_change_col in comparison_df.columns:
                pct_changes = comparison_df[pct_change_col].dropna()
            
            statistics['summary'][metric] = {
                'mean_change': changes.mean(),
                'median_change': changes.median(),
                'std_change': changes.std(),
                'mean_pct_change': pct_changes.mean() if len(pct_changes) > 0 else np.nan,
                'median_pct_change': pct_changes.median() if len(pct_changes) > 0 else np.nan,
                'countries_increased': (changes > 0).sum(),
                'countries_decreased': (changes < 0).sum(),
                'countries_unchanged': (changes == 0).sum()
            }
    
    # 2. 显著性检验（配对t检验）
    logger.info("🔬 执行显著性检验...")
    for metric in metrics_of_interest:
        pre_col = f'{metric}_pre'
        post_col = f'{metric}_post'
        
        if pre_col in comparison_df.columns and post_col in comparison_df.columns:
            pre_values = comparison_df[pre_col].dropna()
            post_values = comparison_df[post_col].dropna()
            
            # 确保配对数据
            common_countries = comparison_df.dropna(subset=[pre_col, post_col])['country_code']
            if len(common_countries) > 3:  # 至少需要几个观测值
                pre_paired = comparison_df.loc[comparison_df['country_code'].isin(common_countries), pre_col]
                post_paired = comparison_df.loc[comparison_df['country_code'].isin(common_countries), post_col]
                
                try:
                    t_stat, p_value = stats.ttest_rel(post_paired, pre_paired)
                    statistics['significance_tests'][metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05,
                        'sample_size': len(common_countries)
                    }
                except Exception as e:
                    logger.warning(f"⚠️  {metric}显著性检验失败: {e}")
    
    # 3. 最大受益者和受损者
    logger.info("🏆 识别最大受益者和受损者...")
    for metric in metrics_of_interest:
        change_col = f'{metric}_change'
        
        if change_col in comparison_df.columns:
            # 按变化量排序
            sorted_df = comparison_df.sort_values(change_col, ascending=False).dropna(subset=[change_col])
            
            statistics['top_winners'][metric] = sorted_df.head(5)[['country_code', change_col]].to_dict('records')
            statistics['top_losers'][metric] = sorted_df.tail(5)[['country_code', change_col]].to_dict('records')
    
    # 4. 期间聚合统计
    logger.info("📊 计算期间聚合统计...")
    for period in ['pre', 'post']:
        period_data = df[df['period'] == period]
        if len(period_data) > 0:
            period_stats = {}
            for metric in metrics_of_interest:
                if metric in period_data.columns:
                    values = period_data[metric].dropna()
                    if len(values) > 0:
                        period_stats[metric] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max()
                        }
            statistics['period_aggregates'][period] = period_stats
    
    logger.info("✅ 政策影响统计量计算完成")
    
    return statistics

def export_analysis_results(comparison_df: pd.DataFrame, 
                          statistics: Dict[str, Any],
                          output_dir: str = "outputs/tables") -> Dict[str, str]:
    """
    导出分析结果
    
    Args:
        comparison_df: 对比分析结果
        statistics: 统计结果
        output_dir: 输出目录
        
    Returns:
        导出文件路径字典
    """
    logger.info(f"💾 导出分析结果到 {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    try:
        # 1. 导出对比分析表格
        comparison_file = output_path / "policy_impact_summary.csv"
        comparison_df.to_csv(comparison_file, index=False)
        exported_files['comparison'] = str(comparison_file)
        logger.info(f"✅ 对比分析表格: {comparison_file}")
        
        # 2. 导出统计摘要
        import json
        statistics_file = output_path / "policy_impact_statistics.json"
        
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换statistics字典
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        clean_statistics = convert_dict(statistics)
        
        with open(statistics_file, 'w', encoding='utf-8') as f:
            json.dump(clean_statistics, f, ensure_ascii=False, indent=2)
        exported_files['statistics'] = str(statistics_file)
        logger.info(f"✅ 统计结果: {statistics_file}")
        
        # 3. 创建可读的摘要报告
        summary_file = output_path / "policy_impact_report.md"
        create_summary_report(comparison_df, statistics, summary_file)
        exported_files['report'] = str(summary_file)
        logger.info(f"✅ 分析报告: {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ 导出过程中出错: {e}")
        
    return exported_files

def create_summary_report(comparison_df: pd.DataFrame, 
                        statistics: Dict[str, Any],
                        output_file: Path) -> None:
    """创建可读的Markdown格式摘要报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 美国能源独立政策影响分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 分析概要\n\n")
        f.write("本报告采用\"事前-事后\"对比分析方法，评估美国能源独立政策对全球能源贸易网络的影响。\n\n")
        
        f.write("### 时间窗口划分\n\n")
        f.write("- **事前期 (Pre-Period)**: 2001-2008年 - 基准期\n")
        f.write("- **转型期 (Transition)**: 2009-2015年 - 页岩油革命加速期\n")
        f.write("- **事后期 (Post-Period)**: 2016-2024年 - 美国成为能源出口国\n\n")
        
        f.write("## 主要发现\n\n")
        
        # 显著性检验结果
        if 'significance_tests' in statistics:
            f.write("### 统计显著性检验\n\n")
            f.write("| 指标 | t统计量 | p值 | 显著性 | 样本量 |\n")
            f.write("|------|---------|-----|--------|--------|\n")
            
            for metric, test_result in statistics['significance_tests'].items():
                significance = "✅ 显著" if test_result['is_significant'] else "❌ 不显著"
                f.write(f"| {metric} | {test_result['t_statistic']:.3f} | {test_result['p_value']:.3f} | {significance} | {test_result['sample_size']} |\n")
            f.write("\n")
        
        # 基本统计摘要
        if 'summary' in statistics:
            f.write("### 整体变化趋势\n\n")
            f.write("| 指标 | 平均变化 | 中位变化 | 平均变化率 | 上升国家数 | 下降国家数 |\n")
            f.write("|------|----------|----------|------------|------------|------------|\n")
            
            for metric, summary in statistics['summary'].items():
                f.write(f"| {metric} | {summary['mean_change']:.4f} | {summary['median_change']:.4f} | {summary['mean_pct_change']:.2f}% | {summary['countries_increased']} | {summary['countries_decreased']} |\n")
            f.write("\n")
        
        f.write("## 分析结论\n\n")
        f.write("1. **政策冲击效应**: 通过统计检验可以识别出具有显著影响的网络指标\n")
        f.write("2. **结构性变化**: 网络中心性指标的变化反映了全球能源贸易格局的重构\n")
        f.write("3. **国家差异化影响**: 不同国家在政策冲击下表现出不同的适应性和受影响程度\n\n")
        
        f.write("## 方法论说明\n\n")
        f.write("- **分析方法**: 事前-事后对比分析 (Pre-Post Comparison)\n")
        f.write("- **统计检验**: 配对t检验 (Paired t-test)\n")
        f.write("- **显著性水平**: α = 0.05\n")
        f.write("- **数据来源**: UN Comtrade全球贸易数据库\n")
        f.write("- **产品范围**: 能源产品 (HS编码: 2701, 2709, 2710, 2711)\n\n")

# 便捷函数
def quick_policy_analysis(filepath: str = "outputs/tables/all_metrics.csv",
                        countries: List[str] = None,
                        metrics: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    快速执行完整的政策影响分析
    
    Args:
        filepath: 数据文件路径
        countries: 关注的国家列表
        metrics: 关注的指标列表
        
    Returns:
        (对比分析结果, 统计结果)
    """
    logger.info("🚀 开始快速政策影响分析...")
    
    # 加载数据
    df = load_and_prepare_data(filepath)
    
    # 对比分析
    comparison_df = run_pre_post_analysis(df, countries, metrics)
    
    # 统计分析
    statistics = calculate_policy_impact_statistics(df, comparison_df, metrics)
    
    # 导出结果
    export_analysis_results(comparison_df, statistics)
    
    logger.info("✅ 快速分析完成")
    
    return comparison_df, statistics