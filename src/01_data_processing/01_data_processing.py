#!/usr/bin/env python3
"""
美国能源独立政策的国际影响研究 - 数据处理脚本

功能：
1. 读取并合并2001-2024年UN Comtrade原始数据
2. 筛选四大能源产品(2701煤炭, 2709原油, 2710成品油, 2711天然气)
3. 标准化国家代码和清理异常值
4. 生成清洗后的数据集供网络分析使用

作者：研究团队
创建日期：2025-08-13
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """创建必要的输出目录"""
    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "processed_data",
        base_dir / "outputs" / "figures",
        base_dir / "outputs" / "tables",
        base_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_raw_data(raw_data_dir):
    """
    读取并合并所有年份的原始数据
    
    参数:
        raw_data_dir: 原始数据目录路径
    
    返回:
        DataFrame: 合并后的原始数据
    """
    print("正在读取原始数据...")
    
    all_data = []
    years = range(2001, 2025)
    
    for year in years:
        file_path = raw_data_dir / f"{year}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, low_memory=False)
                print(f"  - {year}: {len(df):,} 条记录")
                all_data.append(df)
            except Exception as e:
                print(f"  - {year}: 读取失败 - {e}")
        else:
            print(f"  - {year}: 文件不存在")
    
    if not all_data:
        raise ValueError("未找到任何有效的数据文件")
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n原始数据合并完成：{len(combined_data):,} 条记录")
    
    return combined_data

def filter_energy_products(df):
    """
    筛选四大能源产品的贸易数据
    
    参数:
        df: 原始贸易数据DataFrame
    
    返回:
        DataFrame: 筛选后的能源产品数据
    """
    print("\n正在筛选能源产品...")
    
    # 四大能源产品的HS代码（转换为整数类型）
    energy_codes = [2701, 2709, 2710, 2711]
    
    # 筛选条件
    energy_data = df[df['cmdCode'].isin(energy_codes)].copy()
    
    print(f"筛选后的能源数据：{len(energy_data):,} 条记录")
    
    # 各产品分布统计
    product_stats = energy_data.groupby(['cmdCode', 'cmdDesc']).size().reset_index(name='记录数')
    print("\n各能源产品分布：")
    for _, row in product_stats.iterrows():
        print(f"  - {row['cmdCode']}: {row['记录数']:,} 条 ({row['cmdDesc']})")
    
    return energy_data

def clean_country_codes(df):
    """
    清理和标准化国家代码
    
    参数:
        df: 能源贸易数据DataFrame
    
    返回:
        DataFrame: 清理后的数据
    """
    print("\n正在清理国家代码...")
    
    initial_count = len(df)
    
    # 移除自贸易记录（同一国家内部贸易）
    df = df[df['reporterISO'] != df['partnerISO']].copy()
    self_trade_removed = initial_count - len(df)
    print(f"  - 移除自贸易记录：{self_trade_removed:,} 条")
    
    # 移除缺失国家代码的记录
    before_iso_clean = len(df)
    df = df.dropna(subset=['reporterISO', 'partnerISO']).copy()
    iso_missing_removed = before_iso_clean - len(df)
    print(f"  - 移除缺失ISO代码记录：{iso_missing_removed:,} 条")
    
    # 移除ISO代码长度不等于3的记录
    before_iso_length = len(df)
    df = df[(df['reporterISO'].str.len() == 3) & (df['partnerISO'].str.len() == 3)].copy()
    iso_length_removed = before_iso_length - len(df)
    print(f"  - 移除ISO代码格式错误记录：{iso_length_removed:,} 条")
    
    # 统计唯一国家数
    unique_reporters = df['reporterISO'].nunique()
    unique_partners = df['partnerISO'].nunique()
    all_countries = pd.concat([df['reporterISO'], df['partnerISO']]).nunique()
    
    print(f"  - 报告国数量：{unique_reporters}")
    print(f"  - 贸易伙伴国数量：{unique_partners}")
    print(f"  - 总体国家数量：{all_countries}")
    
    return df

def clean_trade_values(df):
    """
    清理贸易价值数据
    
    参数:
        df: 能源贸易数据DataFrame
    
    返回:
        DataFrame: 清理后的数据
    """
    print("\n正在清理贸易价值数据...")
    
    initial_count = len(df)
    
    # 选择主要价值字段，优先使用primaryValue
    df['trade_value'] = df['primaryValue']
    
    # 如果primaryValue缺失，使用cifvalue或fobvalue
    mask_missing_primary = df['trade_value'].isna()
    df.loc[mask_missing_primary, 'trade_value'] = df.loc[mask_missing_primary, 'cifvalue']
    
    mask_still_missing = df['trade_value'].isna()
    df.loc[mask_still_missing, 'trade_value'] = df.loc[mask_still_missing, 'fobvalue']
    
    # 移除仍然缺失贸易价值的记录
    before_value_clean = len(df)
    df = df.dropna(subset=['trade_value']).copy()
    value_missing_removed = before_value_clean - len(df)
    print(f"  - 移除缺失贸易价值记录：{value_missing_removed:,} 条")
    
    # 移除负值和零值
    before_positive = len(df)
    df = df[df['trade_value'] > 0].copy()
    non_positive_removed = before_positive - len(df)
    print(f"  - 移除非正值记录：{non_positive_removed:,} 条")
    
    # 移除异常大值（使用99.9%分位数作为阈值）
    threshold = df['trade_value'].quantile(0.999)
    before_outlier = len(df)
    df = df[df['trade_value'] <= threshold].copy()
    outlier_removed = before_outlier - len(df)
    print(f"  - 移除异常大值记录（>{threshold:,.0f}美元）：{outlier_removed:,} 条")
    
    # 统计贸易价值分布
    print(f"\n贸易价值统计：")
    print(f"  - 最小值：${df['trade_value'].min():,.0f}")
    print(f"  - 中位数：${df['trade_value'].median():,.0f}")
    print(f"  - 平均值：${df['trade_value'].mean():,.0f}")
    print(f"  - 最大值：${df['trade_value'].max():,.0f}")
    
    return df

def create_final_dataset(df):
    """
    创建最终的清洗数据集
    
    参数:
        df: 清理后的能源贸易数据
    
    返回:
        DataFrame: 最终数据集
    """
    print("\n正在创建最终数据集...")
    
    # 选择核心字段
    core_columns = [
        'refYear', 'reporterISO', 'reporterDesc', 'partnerISO', 'partnerDesc',
        'flowCode', 'flowDesc', 'cmdCode', 'cmdDesc', 'trade_value'
    ]
    
    # 确保所有核心字段都存在
    available_columns = [col for col in core_columns if col in df.columns]
    final_df = df[available_columns].copy()
    
    # 重命名字段以便后续分析
    rename_mapping = {
        'refYear': 'year',
        'reporterISO': 'reporter',
        'reporterDesc': 'reporter_name',
        'partnerISO': 'partner',
        'partnerDesc': 'partner_name',
        'flowCode': 'flow',
        'flowDesc': 'flow_name',
        'cmdCode': 'product_code',
        'cmdDesc': 'product_name',
        'trade_value': 'trade_value_raw_usd'
    }
    
    final_df = final_df.rename(columns=rename_mapping)
    
    # 排序
    final_df = final_df.sort_values(['year', 'reporter', 'partner', 'product_code']).reset_index(drop=True)
    
    print(f"最终数据集：{len(final_df):,} 条记录")
    print(f"时间跨度：{final_df['year'].min()} - {final_df['year'].max()}")
    
    return final_df

def generate_data_summary(df, output_dir):
    """
    生成数据质量报告和统计摘要
    
    参数:
        df: 最终清洗后的数据集
        output_dir: 输出目录
    """
    print("\n正在生成数据统计摘要...")
    
    # 年度统计
    yearly_stats = df.groupby('year').agg({
        'trade_value_raw_usd': ['count', 'sum', 'mean'],
        'reporter': 'nunique',
        'partner': 'nunique'
    }).round(2)
    
    yearly_stats.columns = ['记录数', '总贸易额(美元)', '平均贸易额(美元)', '报告国数', '伙伴国数']
    yearly_stats['总贸易额(十亿美元)'] = yearly_stats['总贸易额(美元)'] / 1e9
    
    # 产品统计
    product_stats = df.groupby(['product_code', 'product_name']).agg({
        'trade_value_raw_usd': ['count', 'sum'],
        'year': ['min', 'max']
    }).round(2)
    
    product_stats.columns = ['记录数', '总贸易额(美元)', '开始年', '结束年']
    product_stats['总贸易额(十亿美元)'] = product_stats['总贸易额(美元)'] / 1e9
    
    # 主要贸易国统计（按总贸易额排序）
    country_stats = df.groupby(['reporter', 'reporter_name'])['trade_value_raw_usd'].sum().sort_values(ascending=False).head(20)
    country_stats = country_stats / 1e9  # 转换为十亿美元
    country_stats.name = '总出口额(十亿美元)'
    
    # 保存统计结果
    with pd.ExcelWriter(output_dir / 'tables' / 'data_summary.xlsx') as writer:
        yearly_stats.to_excel(writer, sheet_name='年度统计')
        product_stats.to_excel(writer, sheet_name='产品统计')
        country_stats.to_excel(writer, sheet_name='主要出口国')
    
    # 保存CSV格式
    yearly_stats.to_csv(output_dir / 'tables' / 'yearly_statistics.csv')
    product_stats.to_csv(output_dir / 'tables' / 'product_statistics.csv')
    country_stats.to_csv(output_dir / 'tables' / 'top_exporters.csv')
    
    print("统计摘要已保存到 outputs/tables/ 目录")
    
    return yearly_stats, product_stats, country_stats

def main():
    """主函数"""
    print("=" * 60)
    print("美国能源独立政策的国际影响研究 - 数据处理")
    print("=" * 60)
    
    # 设置目录
    base_dir = setup_directories()
    raw_data_dir = base_dir / "raw_data"
    processed_data_dir = base_dir / "processed_data"
    output_dir = base_dir / "outputs"
    
    try:
        # 步骤1：读取原始数据
        raw_data = load_raw_data(raw_data_dir)
        
        # 步骤2：筛选能源产品
        energy_data = filter_energy_products(raw_data)
        
        # 步骤3：清理国家代码
        cleaned_country_data = clean_country_codes(energy_data)
        
        # 步骤4：清理贸易价值
        cleaned_value_data = clean_trade_values(cleaned_country_data)
        
        # 步骤5：创建最终数据集
        final_dataset = create_final_dataset(cleaned_value_data)
        
        # 步骤6：按年度保存清洗后的数据
        print(f"\n正在按年度保存清洗后的数据...")
        for year in range(2001, 2025):
            year_data = final_dataset[final_dataset['year'] == year]
            if len(year_data) > 0:
                output_file = processed_data_dir / f"cleaned_energy_trade_{year}.csv"
                year_data.to_csv(output_file, index=False)
                print(f"  - {year}: {len(year_data):,} 条记录 -> {output_file.name}")
        
        # 步骤7：生成统计摘要
        yearly_stats, product_stats, country_stats = generate_data_summary(final_dataset, output_dir)
        
        # 打印关键统计信息
        print("\n" + "=" * 60)
        print("数据处理完成！关键统计信息：")
        print("=" * 60)
        print(f"总记录数：{len(final_dataset):,}")
        print(f"时间跨度：{final_dataset['year'].min()} - {final_dataset['year'].max()}")
        print(f"国家数量：{pd.concat([final_dataset['reporter'], final_dataset['partner']]).nunique()}")
        print(f"总贸易额：${final_dataset['trade_value_raw_usd'].sum()/1e12:.2f} 万亿美元")
        
        print("\n年度记录数分布（前5年和后5年）：")
        year_counts = final_dataset['year'].value_counts().sort_index()
        for year in [2001, 2002, 2003, 2004, 2005]:
            if year in year_counts.index:
                print(f"  {year}: {year_counts[year]:,}")
        print("  ...")
        for year in [2020, 2021, 2022, 2023, 2024]:
            if year in year_counts.index:
                print(f"  {year}: {year_counts[year]:,}")
        
        print("\n美国相关贸易记录数：")
        usa_records = final_dataset[(final_dataset['reporter'] == 'USA') | (final_dataset['partner'] == 'USA')]
        print(f"  总计：{len(usa_records):,} 条记录")
        
    except Exception as e:
        print(f"\n错误：{e}")
        raise

if __name__ == "__main__":
    main()