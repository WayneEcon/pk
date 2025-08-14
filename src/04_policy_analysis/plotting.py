#!/usr/bin/env python3
"""
plotting.py - 可视化功能
创建政策影响分析的图表和可视化
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 政策期间配置（英文版）
PERIOD_CONFIG = {
    'pre': {'color': '#2E86AB', 'label': 'Pre-Policy (2001-2008)', 'alpha': 0.3},
    'transition': {'color': '#A23B72', 'label': 'Transition (2009-2015)', 'alpha': 0.3}, 
    'post': {'color': '#F18F01', 'label': 'Post-Policy (2016-2024)', 'alpha': 0.3}
}

# 指标英文名称映射
METRIC_LABELS = {
    'in_strength': 'Import Strength',
    'out_strength': 'Export Strength', 
    'total_strength': 'Total Trade Strength',
    'betweenness_centrality': 'Betweenness Centrality',
    'pagerank_centrality': 'PageRank Centrality',
    'eigenvector_centrality': 'Eigenvector Centrality',
    'in_degree': 'Import Degree',
    'out_degree': 'Export Degree',
    'total_degree': 'Total Degree'
}

def plot_country_dashboard(df: pd.DataFrame, 
                          country_code: str, 
                          metrics_to_plot: List[str],
                          output_dir: str = "outputs/figures/policy_impact",
                          figsize: Tuple = (16, 12)) -> str:
    """
    为单个Country生成包含多个子图的仪表盘式概览图
    
    Args:
        df: 完整的数据DataFrame（包含period列）
        country_code: Country代码
        metrics_to_plot: 要绘制的指标列表
        output_dir: 输出目录
        figsize: 图形大小
        
    Returns:
        保存的文件路径
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 筛选该Country的数据
    country_data = df[df['country_code'] == country_code].copy()
    if country_data.empty:
        logger.warning(f"No data found for country: {country_code}")
        return ""
    
    # 计算网格布局
    n_metrics = len(metrics_to_plot)
    if n_metrics <= 3:
        nrows, ncols = 1, n_metrics
    elif n_metrics <= 6:
        nrows, ncols = 2, 3
    elif n_metrics <= 9:
        nrows, ncols = 3, 3
    else:
        nrows = int(np.ceil(n_metrics / 4))
        ncols = 4
    
    # 创建图形和子图
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                           sharex=True if nrows > 1 else False)
    
    # 确保axes是二维数组
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)
    
    # 绘制每个指标
    for i, metric in enumerate(metrics_to_plot):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # 绘制时间序列
        ax.plot(country_data['year'], country_data[metric], 
               linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        
        # 添加期间背景色块
        for period, config in PERIOD_CONFIG.items():
            period_data = country_data[country_data['period'] == period]
            if not period_data.empty:
                ax.axvspan(period_data['year'].min(), period_data['year'].max(), 
                          alpha=config['alpha'], color=config['color'])
        
        # 设置子图标题和标签
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight='bold')
        
        # 只为边缘子图设置轴标签
        if row == nrows - 1:  # 底行
            ax.set_xlabel('Year', fontsize=10)
        if col == 0:  # 左列
            ax.set_ylabel('Value', fontsize=10)
            
        # 格式化轴
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # 科学计数法处理大数值
        if country_data[metric].max() > 1e9:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 隐藏多余的子图
    for i in range(n_metrics, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    # 设置整体标题
    fig.suptitle(f'{country_code}: Policy Impact Analysis', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 添加图例（仅一次）
    if n_metrics > 0:
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=config['color'], 
                                       alpha=config['alpha'], label=config['label']) 
                         for config in PERIOD_CONFIG.values()]
        fig.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.88), fontsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # 保存图片
    filename = f"{country_code}_dashboard.png"
    filepath = Path(output_dir) / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"✅ Dashboard saved: {filepath}")
    return str(filepath)

def plot_metric_timeseries(df: pd.DataFrame, 
                          country_code: str, 
                          metric_name: str,
                          output_dir: str = "outputs/figures/policy_impact",
                          figsize: Tuple[int, int] = (12, 8),
                          save_format: str = 'png') -> str:
    """
    绘制指定Country指定指标的时间序列图
    
    Args:
        df: 完整的数据DataFrame（包含period列）
        country_code: Country代码
        metric_name: 指标名称
        output_dir: 输出目录
        figsize: 图形大小
        save_format: 保存格式
        
    Returns:
        保存的文件路径
        
    Raises:
        ValueError: 当Country或指标不存在时
    """
    logger.info(f"📈 绘制时间序列图: {country_code} - {metric_name}")
    
    # 验证输入
    if country_code not in df['country_code'].values:
        raise ValueError(f"Country代码 {country_code} 不存在于数据中")
    
    if metric_name not in df.columns:
        raise ValueError(f"指标 {metric_name} 不存在于数据中")
    
    # 筛选数据
    country_data = df[df['country_code'] == country_code].copy()
    country_data = country_data.sort_values('year')
    
    if len(country_data) == 0:
        raise ValueError(f"没有找到Country {country_code} 的数据")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制时间序列线
    years = country_data['year']
    values = country_data[metric_name]
    
    ax.plot(years, values, linewidth=2.5, marker='o', markersize=6, 
           color='#2C3E50', alpha=0.8, label=f'{country_code} - {metric_name}')
    
    # 添加政策期间背景
    for period, config in PERIOD_CONFIG.items():
        if period == 'pre':
            ax.axvspan(2001, 2008, alpha=config['alpha'], color=config['color'], 
                      label=config['label'])
        elif period == 'transition':
            ax.axvspan(2009, 2015, alpha=config['alpha'], color=config['color'],
                      label=config['label'])
        elif period == 'post':
            ax.axvspan(2016, 2024, alpha=config['alpha'], color=config['color'],
                      label=config['label'])
    
    # 添加分割线
    ax.axvline(x=2008.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=2015.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 格式化图表
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{METRIC_LABELS.get(metric_name, metric_name)}', fontsize=12, fontweight='bold')
    ax.set_title(f'{country_code} - {METRIC_LABELS.get(metric_name, metric_name)} Time Series', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴刻度
    ax.set_xlim(2000, 2025)
    ax.set_xticks(range(2001, 2025, 2))
    ax.tick_params(axis='x', rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图形
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{country_code}_{metric_name}_timeseries.{save_format}"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"✅ 时间序列图已保存: {filepath}")
    
    return str(filepath)

def plot_period_comparison(comparison_df: pd.DataFrame,
                         metric_name: str,
                         top_n: int = 10,
                         output_dir: str = "outputs/figures/policy_impact",
                         figsize: Tuple[int, int] = (14, 10)) -> str:
    """
    绘制指标的事前-事后期间对比图
    
    Args:
        comparison_df: 对比分析结果DataFrame
        metric_name: 指标名称
        top_n: 显示变化最大的前N个Country
        output_dir: 输出目录
        figsize: 图形大小
        
    Returns:
        保存的文件路径
    """
    logger.info(f"📊 绘制期间对比图: {metric_name}")
    
    # 验证列名
    pre_col = f'{metric_name}_pre'
    post_col = f'{metric_name}_post'
    change_col = f'{metric_name}_change'
    
    required_cols = [pre_col, post_col, change_col]
    missing_cols = [col for col in required_cols if col not in comparison_df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")
    
    # 筛选数据（去除NaN并按Change排序）
    plot_data = comparison_df.dropna(subset=required_cols).copy()
    plot_data = plot_data.reindex(plot_data[change_col].abs().sort_values(ascending=False).index)
    
    # 取前N个变化最大的Country
    plot_data = plot_data.head(top_n)
    
    if len(plot_data) == 0:
        raise ValueError(f"没有有效数据用于绘制 {metric_name} 的对比图")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 子图1: 事前-事后值对比
    countries = plot_data['country_code']
    pre_values = plot_data[pre_col]
    post_values = plot_data[post_col]
    
    x = np.arange(len(countries))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pre_values, width, label='Pre-Policy (2001-2008)', 
                   color=PERIOD_CONFIG['pre']['color'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, post_values, width, label='Post-Policy (2016-2024)',
                   color=PERIOD_CONFIG['post']['color'], alpha=0.8)
    
    ax1.set_xlabel('Country', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
    ax1.set_title(f'{metric_name} - Pre vs Post Period', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, pre_values)
    add_value_labels(bars2, post_values)
    
    # 子图2: Change
    changes = plot_data[change_col]
    colors = ['green' if x > 0 else 'red' for x in changes]
    
    bars = ax2.bar(countries, changes, color=colors, alpha=0.7)
    ax2.set_xlabel('Country', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{metric_name} Change', fontsize=12, fontweight='bold')
    ax2.set_title(f'{metric_name} - Change (Post - Pre Period)', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加Change标签
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{change:.3f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{metric_name}_period_comparison.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"✅ 期间对比图已保存: {filepath}")
    
    return str(filepath)

def plot_correlation_heatmap(comparison_df: pd.DataFrame,
                           metrics_list: List[str],
                           output_dir: str = "outputs/figures/policy_impact",
                           figsize: Tuple[int, int] = (12, 10)) -> str:
    """
    绘制指标Change的相关性热力图
    
    Args:
        comparison_df: 对比分析结果DataFrame
        metrics_list: 指标列表
        output_dir: 输出目录
        figsize: 图形大小
        
    Returns:
        保存的文件路径
    """
    logger.info("🔥 绘制Change相关性热力图...")
    
    # 提取Change列
    change_cols = [f'{metric}_change' for metric in metrics_list]
    available_cols = [col for col in change_cols if col in comparison_df.columns]
    
    if len(available_cols) < 2:
        raise ValueError("至少需要2个有效的Change指标才能绘制相关性图")
    
    # 计算相关性矩阵
    corr_data = comparison_df[available_cols].dropna()
    correlation_matrix = corr_data.corr()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               fmt='.3f', ax=ax)
    
    # 设置标签（去掉_change后缀）
    labels = [col.replace('_change', '') for col in available_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    ax.set_title('指标Change相关性分析\n(Post - Pre Period)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图形
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = "metrics_change_correlation.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"✅ 相关性热力图已保存: {filepath}")
    
    return str(filepath)

def create_policy_impact_dashboard(df: pd.DataFrame,
                                 comparison_df: pd.DataFrame,
                                 statistics: Dict[str, Any],
                                 key_countries: List[str],
                                 key_metrics: List[str],
                                 output_dir: str = "outputs/figures/policy_impact") -> Dict[str, str]:
    """
    创建政策影响分析的完整仪表板（重构版本）
    
    Args:
        df: 原始数据
        comparison_df: 对比分析结果
        statistics: 统计结果
        key_countries: 重点Country列表
        key_metrics: 重点指标列表
        output_dir: 输出目录
        
    Returns:
        生成的图表文件路径字典
    """
    logger.info("📊 创建政策影响分析仪表板...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    try:
        # 1. 为每个重点Country生成仪表盘
        logger.info("📈 生成Country仪表盘...")
        dashboard_files = []
        
        for country in key_countries:
            dashboard_file = plot_country_dashboard(
                df, country, key_metrics, output_dir
            )
            if dashboard_file:
                dashboard_files.append(dashboard_file)
        
        generated_files['dashboards'] = dashboard_files
        logger.info(f"✅ 生成了 {len(dashboard_files)} 个Country仪表盘")
        
        # 2. 生成期间对比图
        logger.info("📊 生成期间对比图...")
        comparison_files = []
        for metric in key_metrics:
            try:
                filepath = plot_period_comparison(comparison_df, metric, output_dir=output_dir)
                comparison_files.append(filepath)
            except Exception as e:
                logger.warning(f"⚠️  生成 {metric} 对比图失败: {e}")
        
        generated_files['comparisons'] = comparison_files
        
        # 3. 生成相关性热力图
        logger.info("🔥 生成相关性热力图...")
        try:
            heatmap_file = plot_correlation_heatmap(comparison_df, key_metrics, output_dir)
            generated_files['heatmap'] = heatmap_file
        except Exception as e:
            logger.warning(f"⚠️  生成相关性热力图失败: {e}")
        
        # 4. 生成综合概览图
        logger.info("🌟 生成综合概览图...")
        try:
            overview_file = create_overview_plot(statistics, key_metrics, output_dir)
            generated_files['overview'] = overview_file
        except Exception as e:
            logger.warning(f"⚠️  生成综合概览图失败: {e}")
        
        logger.info(f"✅ 仪表板创建完成，共生成 {sum(len(v) if isinstance(v, list) else 1 for v in generated_files.values())} 个图表")
        
    except Exception as e:
        logger.error(f"❌ 仪表板创建过程中出错: {e}")
    
    return generated_files

def create_overview_plot(statistics: Dict[str, Any],
                        metrics: List[str],
                        output_dir: str) -> str:
    """
    创建统计概览图
    
    Args:
        statistics: 统计结果
        metrics: 指标列表  
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    if 'summary' not in statistics or 'significance_tests' not in statistics:
        raise ValueError("统计结果不完整，无法生成概览图")
    
    # 准备数据
    summary_data = []
    for metric in metrics:
        if metric in statistics['summary']:
            summary = statistics['summary'][metric]
            significance = statistics['significance_tests'].get(metric, {})
            
            summary_data.append({
                'metric': metric,
                'mean_change': summary['mean_change'],
                'countries_increased': summary['countries_increased'],
                'countries_decreased': summary['countries_decreased'],
                'is_significant': significance.get('is_significant', False),
                'p_value': significance.get('p_value', 1.0)
            })
    
    if not summary_data:
        raise ValueError("没有有效的统计数据用于生成概览图")
    
    summary_df = pd.DataFrame(summary_data)
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 平均Change
    colors = ['green' if x > 0 else 'red' for x in summary_df['mean_change']]
    bars1 = ax1.bar(summary_df['metric'], summary_df['mean_change'], color=colors, alpha=0.7)
    ax1.set_title('各指标平均Change', fontweight='bold')
    ax1.set_ylabel('平均Change')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 子图2: 显著性检验结果
    significance_colors = ['green' if x else 'gray' for x in summary_df['is_significant']]
    bars2 = ax2.bar(summary_df['metric'], summary_df['p_value'], color=significance_colors, alpha=0.7)
    ax2.set_title('统计显著性检验 (p值)', fontweight='bold')
    ax2.set_ylabel('p值')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
    ax2.legend()
    
    # 子图3: Country数量变化
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, summary_df['countries_increased'], width, 
                    label='指标上升', color='green', alpha=0.7)
    bars3b = ax3.bar(x + width/2, summary_df['countries_decreased'], width,
                    label='指标下降', color='red', alpha=0.7)
    
    ax3.set_title('各指标影响的Country数量', fontweight='bold')
    ax3.set_ylabel('Country数量')
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df['metric'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: Change分布散点图
    significant_metrics = summary_df[summary_df['is_significant']]
    non_significant_metrics = summary_df[~summary_df['is_significant']]
    
    if len(significant_metrics) > 0:
        ax4.scatter(significant_metrics['mean_change'], significant_metrics['p_value'],
                   color='red', s=100, alpha=0.7, label='显著变化')
    
    if len(non_significant_metrics) > 0:
        ax4.scatter(non_significant_metrics['mean_change'], non_significant_metrics['p_value'],
                   color='gray', s=100, alpha=0.7, label='非显著变化')
    
    ax4.set_xlabel('平均Change')
    ax4.set_ylabel('p值')
    ax4.set_title('Change vs 统计显著性', fontweight='bold')
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.suptitle('美国能源独立政策影响分析 - 统计概览', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图形
    output_path = Path(output_dir)
    filename = "policy_impact_overview.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    return str(filepath)