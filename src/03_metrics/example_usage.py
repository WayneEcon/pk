#!/usr/bin/env python3
"""
个性化PageRank使用示例
====================

演示如何在04_dli_analysis模块中调用和使用个性化PageRank数据
"""

import pandas as pd
from pathlib import Path
import sys

def load_personalized_pagerank_data(metrics_dir: Path = None) -> pd.DataFrame:
    """
    加载个性化PageRank数据
    
    Args:
        metrics_dir: 03_metrics模块目录路径
        
    Returns:
        pd.DataFrame: 个性化PageRank数据
    """
    if metrics_dir is None:
        # 默认路径：从04模块调用时的相对路径
        current_dir = Path(__file__).parent
        metrics_dir = current_dir
        
    ppr_file = metrics_dir / "outputs" / "personalized_pagerank_panel.csv"
    
    if not ppr_file.exists():
        raise FileNotFoundError(f"个性化PageRank数据文件未找到: {ppr_file}")
    
    df = pd.read_csv(ppr_file)
    print(f"✅ 成功加载个性化PageRank数据: {len(df):,}条记录")
    return df

def example_integration_with_dli(year: int = 2020):
    """
    演示如何将个性化PageRank数据整合到DLI计算中
    
    Args:
        year: 目标年份
    """
    print(f"\n=== 演示：将个性化PageRank整合到{year}年DLI计算 ===")
    
    # 1. 加载个性化PageRank数据
    ppr_data = load_personalized_pagerank_data()
    
    # 2. 筛选指定年份数据
    year_data = ppr_data[ppr_data['year'] == year].copy()
    print(f"📊 {year}年数据: {len(year_data)}个国家")
    
    # 3. 重命名列以符合DLI分析约定
    year_data = year_data.rename(columns={
        'ppr_us_export_influence': 'pagerank_export_locking',
        'ppr_influence_on_us': 'pagerank_import_locking'
    })
    
    # 4. 添加合成指标（例子）
    year_data['pagerank_total_locking'] = (
        year_data['pagerank_export_locking'] + 
        year_data['pagerank_import_locking']
    )
    
    # 5. 展示美国相关的重要数据点
    usa_data = year_data[year_data['country_name'] == 'USA']
    if not usa_data.empty:
        print(f"\n🇺🇸 美国{year}年个性化PageRank数据:")
        print(f"  出口锁定影响力: {usa_data['pagerank_export_locking'].iloc[0]:.6f}")
        print(f"  进口锁定影响力: {usa_data['pagerank_import_locking'].iloc[0]:.6f}")
        print(f"  总锁定影响力: {usa_data['pagerank_total_locking'].iloc[0]:.6f}")
    
    # 6. 展示对美国影响力最大的前5个国家
    print(f"\n🔝 {year}年对美国进口锁定影响力最高的5个国家:")
    top_import_influence = year_data.nlargest(5, 'pagerank_import_locking')[
        ['country_name', 'pagerank_import_locking']
    ]
    for idx, row in top_import_influence.iterrows():
        print(f"  {row['country_name']}: {row['pagerank_import_locking']:.6f}")
    
    # 7. 展示美国出口锁定影响力最大的前5个国家
    print(f"\n🚀 {year}年美国出口锁定影响力最高的5个国家:")
    top_export_influence = year_data.nlargest(5, 'pagerank_export_locking')[
        ['country_name', 'pagerank_export_locking']
    ]
    for idx, row in top_export_influence.iterrows():
        print(f"  {row['country_name']}: {row['pagerank_export_locking']:.6f}")
    
    return year_data

def demonstrate_time_series_analysis():
    """演示时间序列分析"""
    print(f"\n=== 演示：美国网络影响力时间序列分析 ===")
    
    # 加载完整数据
    ppr_data = load_personalized_pagerank_data()
    
    # 提取美国数据的时间序列
    usa_timeseries = ppr_data[ppr_data['country_name'] == 'USA'].copy()
    usa_timeseries = usa_timeseries.sort_values('year')
    
    print(f"📈 美国网络影响力演变 (2001-2024):")
    print(f"  年份范围: {usa_timeseries['year'].min()}-{usa_timeseries['year'].max()}")
    
    # 关键时间点分析
    key_years = [2001, 2008, 2011, 2020, 2024]  # 9/11, 金融危机, 页岩革命, 疫情, 最新
    
    for year in key_years:
        year_data = usa_timeseries[usa_timeseries['year'] == year]
        if not year_data.empty:
            export_influence = year_data['ppr_us_export_influence'].iloc[0]
            print(f"  {year}年出口锁定影响力: {export_influence:.6f}")
    
    # 计算变化趋势
    first_year = usa_timeseries.iloc[0]
    last_year = usa_timeseries.iloc[-1]
    
    export_change = last_year['ppr_us_export_influence'] - first_year['ppr_us_export_influence']
    export_change_pct = (export_change / first_year['ppr_us_export_influence']) * 100
    
    print(f"\n📊 2001-2024年美国出口影响力变化:")
    print(f"  绝对变化: {export_change:+.6f}")
    print(f"  相对变化: {export_change_pct:+.2f}%")
    
    return usa_timeseries

def demonstrate_cross_country_comparison(year: int = 2024):
    """演示跨国比较分析"""
    print(f"\n=== 演示：{year}年跨国网络影响力比较 ===")
    
    ppr_data = load_personalized_pagerank_data()
    year_data = ppr_data[ppr_data['year'] == year].copy()
    
    # 重点关注能源大国
    energy_powers = ['USA', 'CHN', 'RUS', 'SAU', 'CAN', 'NOR', 'ARE', 'NLD']
    
    energy_data = year_data[year_data['country_name'].isin(energy_powers)].copy()
    energy_data = energy_data.sort_values('ppr_us_export_influence', ascending=False)
    
    print(f"🌍 主要能源国家的美国出口影响力接收情况:")
    for idx, row in energy_data.iterrows():
        country = row['country_name']
        influence = row['ppr_us_export_influence']
        print(f"  {country}: {influence:.6f}")
    
    # 对美影响力分析
    energy_data_import = energy_data.sort_values('ppr_influence_on_us', ascending=False)
    print(f"\n🇺🇸 主要能源国家对美国的进口锁定影响力:")
    for idx, row in energy_data_import.iterrows():
        country = row['country_name']
        influence = row['ppr_influence_on_us']
        print(f"  {country}: {influence:.6f}")
    
    return energy_data

def main():
    """主演示函数"""
    print("🌟 个性化PageRank数据使用演示")
    print("=" * 50)
    
    try:
        # 1. 基础数据加载演示
        print("\n【Step 1】基础数据加载")
        ppr_data = load_personalized_pagerank_data()
        print(f"数据形状: {ppr_data.shape}")
        print(f"列名: {list(ppr_data.columns)}")
        
        # 2. DLI整合演示
        print("\n【Step 2】DLI整合演示")
        dli_ready_data = example_integration_with_dli(2020)
        
        # 3. 时间序列分析演示
        print("\n【Step 3】时间序列分析演示")
        usa_timeseries = demonstrate_time_series_analysis()
        
        # 4. 跨国比较演示
        print("\n【Step 4】跨国比较演示")
        cross_country_data = demonstrate_cross_country_comparison(2024)
        
        print(f"\n✅ 演示完成！个性化PageRank数据已准备就绪，可以整合到DLI分析中。")
        
        # 5. 给出04模块的调用建议
        print(f"\n💡 在04_dli_analysis模块中的建议调用方式:")
        print(f"```python")
        print(f"# 在04模块的main.py或相关脚本中:")
        print(f"from pathlib import Path")
        print(f"import pandas as pd")
        print(f"")
        print(f"def load_personalized_pagerank():")
        print(f"    metrics_dir = Path('../03_metrics')")
        print(f"    ppr_file = metrics_dir / 'outputs' / 'personalized_pagerank_panel.csv'")
        print(f"    return pd.read_csv(ppr_file)")
        print(f"")
        print(f"# 在DLI计算中整合:")
        print(f"ppr_data = load_personalized_pagerank()")
        print(f"# 按年份和国家merge到现有的DLI数据中")
        print(f"enhanced_dli = existing_dli.merge(")
        print(f"    ppr_data[['year', 'country_name', 'ppr_us_export_influence', 'ppr_influence_on_us']],")
        print(f"    on=['year', 'country_name'], ")
        print(f"    how='left'")
        print(f")")
        print(f"```")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)