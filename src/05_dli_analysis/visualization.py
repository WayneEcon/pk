#!/usr/bin/env python3
"""
双向DLI可视化分析模块 (Bidirectional DLI Visualization Module)
=====================================================

本模块负责生成双向动态锁定指数(DLI)分析的核心可视化图表：
1. 权力反转图：展示锁定关系的时间演化和方向转换
2. 出口目标排名图：美国对不同国家的出口锁定力排名
3. 双向对比分析图：进口锁定vs出口锁定的综合对比

作者：Energy Network Analysis Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set English fonts and styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置图表风格
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_dli_panel_data(file_path: str = None) -> pd.DataFrame:
    """
    加载双向DLI面板数据
    
    Args:
        file_path: 数据文件路径，默认使用标准路径
        
    Returns:
        双向DLI面板数据DataFrame
    """
    
    if file_path is None:
        base_dir = Path(__file__).parent.parent.parent
        file_path = Path(__file__).parent / "dli_panel_data_v2.csv"
    
    logger.info(f"📂 加载双向DLI面板数据: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ 成功加载数据: {len(df):,} 条记录")
        
        # 数据概览
        locking_stats = df.groupby('locking_dimension_type').size()
        logger.info(f"📊 双向锁定数据分布:")
        for locking_type, count in locking_stats.items():
            logger.info(f"  {locking_type}: {count:,} 条记录")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        raise

def create_power_reversal_chart(df: pd.DataFrame = None, output_dir: str = None) -> str:
    """
    创建权力反转图：展示锁定关系的时间演化和方向转换
    
    重点展示页岩革命(2011年)前后美国在能源贸易中的权力关系变化：
    - 进口锁定：美国被供应商锁定的程度
    - 出口锁定：美国锁定其他国家的程度
    
    Args:
        df: 双向DLI面板数据，如果为None则自动加载
        output_dir: 输出目录，默认使用标准路径
        
    Returns:
        生成的图表文件路径
    """
    
    logger.info("🎨 开始创建权力反转图...")
    
    # 加载数据
    if df is None:
        df = load_dli_panel_data()
    
    # 设置输出路径
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "outputs" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按年份和锁定类型汇总数据
    yearly_trends = df.groupby(['year', 'locking_dimension_type']).agg({
        'dli_score': ['mean', 'std', 'count']
    }).round(4)
    yearly_trends.columns = ['mean_dli', 'std_dli', 'count']
    yearly_trends = yearly_trends.reset_index()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 颜色配置
    colors = {
        'import_locking': '#d62728',  # 红色：美国被锁定（负面）
        'export_locking': '#2ca02c'   # 绿色：美国锁定他国（正面）
    }
    
    # 上图：时间趋势线图
    for locking_type in ['import_locking', 'export_locking']:
        data = yearly_trends[yearly_trends['locking_dimension_type'] == locking_type]
        
        # 绘制主趋势线
        ax1.plot(data['year'], data['mean_dli'], 
                color=colors[locking_type], linewidth=3, 
                label=f'{"Import Locking (US Being Locked)" if locking_type == "import_locking" else "Export Locking (US Locking Others)"}',
                marker='o', markersize=5)
        
        # 添加置信区间
        ax1.fill_between(data['year'], 
                        data['mean_dli'] - data['std_dli'], 
                        data['mean_dli'] + data['std_dli'],
                        alpha=0.2, color=colors[locking_type])
    
    # 标记页岩革命时点
    ax1.axvline(x=2011, color='#ff7f0e', linestyle='--', linewidth=2, 
                label='Shale Revolution Policy Shock (2011)')
    
    # 添加关键时期标注
    ax1.axvspan(2001, 2010, alpha=0.1, color='gray', label='Traditional Energy Period')
    ax1.axvspan(2011, 2024, alpha=0.1, color='orange', label='Shale Revolution Period')
    
    ax1.set_title('Power Reversal Chart: Bidirectional Impact of Shale Revolution on US Energy Trade Locking', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Average DLI Score', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 设置x轴
    ax1.set_xlim(2001, 2024)
    ax1.set_xticks(range(2001, 2025, 3))
    
    # 下图：双向对比条形图（政策前后对比）
    pre_policy = df[df['year'] <= 2010].groupby('locking_dimension_type')['dli_score'].mean()
    post_policy = df[df['year'] >= 2011].groupby('locking_dimension_type')['dli_score'].mean()
    
    x = np.arange(len(pre_policy))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, pre_policy.values, width, 
                    label='Pre-Shale Revolution (2001-2010)', color='lightblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, post_policy.values, width,
                    label='Post-Shale Revolution (2011-2024)', color='darkblue', alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Bidirectional Locking Effects: Pre vs Post Shale Revolution', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Locking Type', fontsize=12)
    ax2.set_ylabel('Average DLI Score', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Import Locking\n(US Being Locked)', 'Export Locking\n(US Locking Others)'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加变化箭头和标注
    for i, locking_type in enumerate(['import_locking', 'export_locking']):
        change = post_policy[locking_type] - pre_policy[locking_type]
        color = 'green' if change > 0 else 'red'
        symbol = '↑' if change > 0 else '↓'
        ax2.annotate(f'{symbol} {change:+.3f}', 
                    xy=(i, max(pre_policy[locking_type], post_policy[locking_type]) + 0.05),
                    ha='center', va='bottom', fontsize=11, color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / "power_reversal_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ 权力反转图已保存至: {output_path}")
    return str(output_path)

def create_export_target_ranking(df: pd.DataFrame = None, output_dir: str = None, 
                               top_n: int = 20) -> str:
    """
    创建出口目标排名图：美国对不同国家的出口锁定力排名
    
    展示美国通过能源出口对各国的影响力排名，识别美国能源外交的重点目标
    
    Args:
        df: 双向DLI面板数据，如果为None则自动加载
        output_dir: 输出目录，默认使用标准路径
        top_n: 显示前N个国家，默认20
        
    Returns:
        生成的图表文件路径
    """
    
    logger.info("🎯 开始创建出口目标排名图...")
    
    # 加载数据
    if df is None:
        df = load_dli_panel_data()
    
    # 设置输出路径
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "outputs" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 筛选出口锁定数据
    export_data = df[df['locking_dimension_type'] == 'export_locking'].copy()
    
    if len(export_data) == 0:
        logger.warning("⚠️ 未找到出口锁定数据")
        return None
    
    # 按国家汇总出口锁定力
    country_ranking = export_data.groupby('us_partner').agg({
        'dli_score': ['mean', 'std', 'count'],
        'market_locking_power': ['mean', 'max']
    }).round(4)
    
    country_ranking.columns = ['avg_dli', 'std_dli', 'count', 'avg_market_power', 'max_market_power']
    country_ranking = country_ranking.reset_index()
    
    # 计算综合锁定力指标（加权平均，考虑记录数量）
    country_ranking['weighted_score'] = (
        country_ranking['avg_dli'] * np.log(country_ranking['count'] + 1)
    )
    
    # 按综合锁定力排序
    country_ranking = country_ranking.sort_values('weighted_score', ascending=False)
    
    # 取前N个国家
    top_countries = country_ranking.head(top_n)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # 左图：出口锁定力排名
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_countries)))
    bars = ax1.barh(range(len(top_countries)), top_countries['avg_dli'], 
                    color=colors, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, top_countries['avg_dli'])):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    ax1.set_yticks(range(len(top_countries)))
    ax1.set_yticklabels(top_countries['us_partner'], fontsize=10)
    ax1.set_xlabel('Average Export Locking DLI Score', fontsize=12)
    ax1.set_title(f'US Export Locking Power Ranking (Top {top_n} Countries)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 反转y轴使排名从上到下
    ax1.invert_yaxis()
    
    # 右图：市场锁定力 vs DLI得分散点图
    recent_data = export_data[export_data['year'] >= 2020]  # 使用近期数据
    
    # 按国家汇总近期数据
    recent_ranking = recent_data.groupby('us_partner').agg({
        'dli_score': 'mean',
        'market_locking_power': 'mean',
        'trade_value_usd': 'sum'
    }).reset_index()
    
    # 创建散点图，点的大小代表贸易额
    scatter = ax2.scatter(recent_ranking['market_locking_power'], 
                         recent_ranking['dli_score'],
                         s=np.sqrt(recent_ranking['trade_value_usd']) / 1000,  # 调整点大小
                         alpha=0.6, c=recent_ranking['dli_score'], 
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # 标注重要国家
    important_countries = recent_ranking.nlargest(8, 'dli_score')
    for _, country in important_countries.iterrows():
        ax2.annotate(country['us_partner'], 
                    (country['market_locking_power'], country['dli_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    ax2.set_xlabel('Market Locking Power', fontsize=12)
    ax2.set_ylabel('DLI Score', fontsize=12)
    ax2.set_title('Market Locking Power vs DLI Score Distribution\n(Point Size = Trade Value, Color = DLI Score)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('DLI Score', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / "export_target_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ 出口目标排名图已保存至: {output_path}")
    return str(output_path)

def create_bidirectional_comparison(df: pd.DataFrame = None, output_dir: str = None) -> str:
    """
    创建双向对比分析图：进口锁定vs出口锁定的综合对比
    
    展示美国在不同能源产品和贸易伙伴上的双向锁定格局
    
    Args:
        df: 双向DLI面板数据，如果为None则自动加载
        output_dir: 输出目录，默认使用标准路径
        
    Returns:
        生成的图表文件路径
    """
    
    logger.info("🔄 开始创建双向对比分析图...")
    
    # 加载数据
    if df is None:
        df = load_dli_panel_data()
    
    # 设置输出路径
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "outputs" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建2x2子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 按能源产品的双向对比
    product_stats = df.groupby(['energy_product', 'locking_dimension_type'])['dli_score'].mean().unstack()
    
    product_stats.plot(kind='bar', ax=ax1, color=['#d62728', '#2ca02c'], alpha=0.8)
    ax1.set_title('Bidirectional Locking Comparison by Energy Product', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Energy Product', fontsize=10)
    ax1.set_ylabel('Average DLI Score', fontsize=10)
    ax1.legend(['Import Locking', 'Export Locking'], fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. 按时间的双向趋势对比
    time_stats = df.groupby(['year', 'locking_dimension_type'])['dli_score'].mean().unstack()
    
    time_stats.plot(ax=ax2, color=['#d62728', '#2ca02c'], linewidth=2, marker='o')
    ax2.axvline(x=2011, color='orange', linestyle='--', alpha=0.7, label='Shale Revolution')
    ax2.set_title('Temporal Evolution of Bidirectional Locking', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Average DLI Score', fontsize=10)
    ax2.legend(['Import Locking', 'Export Locking', 'Shale Revolution'], fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Four-dimension radar chart comparison
    dimensions = ['continuity', 'infrastructure', 'stability', 'market_locking_power']
    dim_labels = ['Continuity', 'Infrastructure', 'Stability', 'Market Locking Power']
    
    # 计算两种锁定类型的各维度平均值
    import_dims = []
    export_dims = []
    
    for dim in dimensions:
        import_avg = df[df['locking_dimension_type'] == 'import_locking'][dim].mean()
        export_avg = df[df['locking_dimension_type'] == 'export_locking'][dim].mean()
        import_dims.append(import_avg)
        export_dims.append(export_avg)
    
    # 标准化到0-1范围
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    all_dims = np.array([import_dims, export_dims])
    scaled_dims = scaler.fit_transform(all_dims.T).T
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    import_scaled = scaled_dims[0].tolist() + [scaled_dims[0][0]]
    export_scaled = scaled_dims[1].tolist() + [scaled_dims[1][0]]
    
    ax3.plot(angles, import_scaled, 'o-', linewidth=2, label='Import Locking', color='#d62728')
    ax3.fill(angles, import_scaled, alpha=0.25, color='#d62728')
    ax3.plot(angles, export_scaled, 'o-', linewidth=2, label='Export Locking', color='#2ca02c')
    ax3.fill(angles, export_scaled, alpha=0.25, color='#2ca02c')
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(dim_labels)
    ax3.set_ylim(0, 1)
    ax3.set_title('Four-Dimension Radar Chart Comparison (Normalized)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True)
    
    # 4. 贸易伙伴数量和锁定强度分布
    partner_stats = df.groupby(['us_partner', 'locking_dimension_type']).agg({
        'dli_score': 'mean',
        'year': 'count'
    }).reset_index()
    
    import_partners = partner_stats[partner_stats['locking_dimension_type'] == 'import_locking']
    export_partners = partner_stats[partner_stats['locking_dimension_type'] == 'export_locking']
    
    ax4.hist(import_partners['dli_score'], bins=20, alpha=0.6, label='Import Locking', 
             color='#d62728', density=True)
    ax4.hist(export_partners['dli_score'], bins=20, alpha=0.6, label='Export Locking', 
             color='#2ca02c', density=True)
    
    ax4.set_title('DLI Score Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('DLI Score', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / "bidirectional_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ 双向对比分析图已保存至: {output_path}")
    return str(output_path)

def generate_all_visualizations(df: pd.DataFrame = None, output_dir: str = None) -> Dict[str, str]:
    """
    生成所有双向DLI可视化图表
    
    Args:
        df: 双向DLI面板数据，如果为None则自动加载
        output_dir: 输出目录，默认使用标准路径
        
    Returns:
        生成的图表文件路径字典
    """
    
    logger.info("🎨 开始生成所有双向DLI可视化图表...")
    
    # 加载数据
    if df is None:
        df = load_dli_panel_data()
    
    # 创建输出目录
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "outputs" / "figures"
    
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成所有图表
    results = {}
    
    try:
        results['power_reversal'] = create_power_reversal_chart(df, output_dir)
        results['export_ranking'] = create_export_target_ranking(df, output_dir)
        results['bidirectional_comparison'] = create_bidirectional_comparison(df, output_dir)
        
        logger.info("✅ 所有可视化图表生成完成!")
        logger.info("📊 生成的图表:")
        for chart_type, path in results.items():
            logger.info(f"  {chart_type}: {path}")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ 可视化生成失败: {e}")
        raise

if __name__ == "__main__":
    # 测试可视化功能
    try:
        # 生成所有可视化图表
        chart_paths = generate_all_visualizations()
        
        print("🎉 双向DLI可视化生成完成!")
        print("📊 生成的图表文件:")
        for chart_type, path in chart_paths.items():
            print(f"  {chart_type}: {path}")
            
    except Exception as e:
        logger.error(f"❌ 可视化测试失败: {e}")
        raise