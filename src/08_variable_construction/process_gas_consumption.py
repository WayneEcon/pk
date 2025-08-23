#!/usr/bin/env python3
"""
天然气消费数据处理脚本 - 构建年度国家级消费面板

本脚本从BP世界能源统计年鉴的宽格式天然气消费数据中，构建标准的长格式年度国家级面板数据。

核心处理逻辑：
1. 识别并移除地区汇总行和注释行
2. 转换宽格式（年份为列）为长格式（年份为行）
3. 数据单位：十亿立方米 (bcm)
4. 处理特殊值：'-' 转为 0，'^' 等特殊标记转为 0

作者: Claude Code
创建时间: 2025-08-23
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(file_path):
    """加载原始天然气消费数据"""
    print("=== 第1步：加载原始数据 ===")
    
    df = pd.read_csv(file_path)
    print(f"原始数据维度: {df.shape}")
    
    # 重命名第一列为country
    df = df.rename(columns={df.columns[0]: 'country'})
    
    # 检查数据单位
    first_col_name = list(pd.read_csv(file_path).columns)[0]
    print(f"数据单位: {first_col_name}")
    
    if 'Billion cubic metres' not in first_col_name:
        print(f"⚠️ 警告：预期单位为'Billion cubic metres'，实际为'{first_col_name}'")
    
    print("✅ 原始数据加载完成")
    return df


def filter_valid_countries(df):
    """筛选有效的国家数据，移除地区汇总和注释行"""
    print("\n=== 第2步：筛选有效国家数据 ===")
    
    def is_valid_country_name(country_name):
        """判断是否为有效的国家名称"""
        if pd.isna(country_name) or str(country_name).strip() == '':
            return False
        
        country_str = str(country_name).strip()
        
        # 排除所有非国家行的模式
        invalid_patterns = [
            # 符号和特殊标记
            '^', '♦', 'Less than', 'not available', 'n/a',
            # 注释和说明
            '#', 'Source:', 'Excludes', 'Includes', 'Note:',
            # 地区汇总关键词
            'Total', 'Other', 'World', 'OECD', 'Non-OECD',
            # 地理区域描述
            'America', 'Europe', 'Africa', 'Asia', 'Middle East', 'CIS',
            'Central America', 'Caribbean', 'Pacific', 'Union'
        ]
        
        # 检查是否包含无效模式
        for pattern in invalid_patterns:
            if pattern in country_str:
                return False
        
        # 检查是否以特殊符号开头（通常是注释或标记）
        if country_str.startswith((' ', '^', '♦', '#')):
            return False
        
        # 长度检查（避免冗长的说明文字）
        if len(country_str) > 30:
            return False
        
        return True
    
    # 筛选有效国家
    valid_countries = df[df['country'].apply(is_valid_country_name)].copy().reset_index(drop=True)
    excluded_count = len(df) - len(valid_countries)
    
    print(f"筛选结果:")
    print(f"  保留国家: {len(valid_countries)} 个")
    print(f"  排除行数: {excluded_count} 行")
    
    # 显示前20个保留的国家
    print(f"\n保留的国家（前20个）:")
    for i, country in enumerate(valid_countries['country'].head(20)):
        print(f"  {i+1:2d}. {country}")
    
    print("✅ 国家数据筛选完成")
    return valid_countries


def convert_to_panel_format(df):
    """转换为长格式面板数据"""
    print("\n=== 第3步：转换为面板格式 ===")
    
    # 识别年份列
    year_columns = [col for col in df.columns if col != 'country' and col.isdigit()]
    year_columns.sort()  # 确保年份按顺序排列
    
    print(f"年份范围: {year_columns[0]}-{year_columns[-1]} ({len(year_columns)} 年)")
    
    # 转换为长格式
    panel_df = pd.melt(
        df,
        id_vars=['country'],
        value_vars=year_columns,
        var_name='year',
        value_name='gas_consumption_bcm'
    )
    
    # 转换数据类型
    panel_df['year'] = panel_df['year'].astype(int)
    
    print(f"面板数据维度: {panel_df.shape}")
    print(f"覆盖国家数: {panel_df['country'].nunique()}")
    print("✅ 格式转换完成")
    
    return panel_df


def clean_consumption_values(panel_df):
    """清洗消费数据的数值"""
    print("\n=== 第4步：清洗消费数值 ===")
    
    def process_value(val):
        """处理单个消费值"""
        if pd.isna(val):
            return np.nan
        
        if isinstance(val, (int, float)):
            return float(val) if val >= 0 else np.nan
        
        str_val = str(val).strip()
        
        # 处理特殊标记
        if str_val in ['-', '', '^']:
            return 0.0  # BP数据中'-'通常表示零或极小值
        
        # 转换为数值
        try:
            numeric_val = float(str_val)
            return numeric_val if numeric_val >= 0 else np.nan
        except ValueError:
            print(f"⚠️ 无法转换的值: {repr(str_val)}")
            return np.nan
    
    # 应用数值处理
    original_na_count = panel_df['gas_consumption_bcm'].isna().sum()
    panel_df['gas_consumption_bcm'] = panel_df['gas_consumption_bcm'].apply(process_value)
    final_na_count = panel_df['gas_consumption_bcm'].isna().sum()
    
    print(f"数值处理结果:")
    print(f"  原始空值: {original_na_count}")
    print(f"  处理后空值: {final_na_count}")
    print(f"  数据完整性: {(1 - final_na_count/len(panel_df)) * 100:.1f}%")
    
    print("✅ 数值清洗完成")
    return panel_df


def validate_and_save(panel_df, output_path):
    """验证数据质量并保存结果"""
    print(f"\n=== 第5步：数据验证与保存 ===")
    
    # 排序
    panel_df = panel_df.sort_values(['country', 'year']).reset_index(drop=True)
    
    # 数据质量统计
    total_rows = len(panel_df)
    valid_rows = panel_df['gas_consumption_bcm'].notna().sum()
    zero_rows = (panel_df['gas_consumption_bcm'] == 0).sum()
    positive_rows = (panel_df['gas_consumption_bcm'] > 0).sum()
    
    print(f"数据质量报告:")
    print(f"  总行数: {total_rows:,}")
    print(f"  有效数据: {valid_rows:,} ({valid_rows/total_rows*100:.1f}%)")
    print(f"  零值: {zero_rows:,} ({zero_rows/total_rows*100:.1f}%)")
    print(f"  正值: {positive_rows:,} ({positive_rows/total_rows*100:.1f}%)")
    
    # 年度全球消费量趋势检查
    annual_totals = panel_df.groupby('year')['gas_consumption_bcm'].sum()
    print(f"\n全球年度天然气消费量 (bcm):")
    key_years = [2000, 2010, 2020, 2024]
    for year in key_years:
        if year in annual_totals.index:
            print(f"  {year}年: {annual_totals[year]:,.1f} bcm")
    
    # 主要消费国验证 (2024年)
    print(f"\n主要消费国验证 (2024年):")
    major_consumers = ['US', 'China', 'Russian Federation', 'Iran', 'Japan', 'Germany', 'India']
    data_2024 = panel_df[panel_df['year'] == 2024]
    
    for country in major_consumers:
        country_data = data_2024[data_2024['country'] == country]
        if len(country_data) > 0:
            consumption = country_data['gas_consumption_bcm'].iloc[0]
            print(f"  ✅ {country}: {consumption:.1f} bcm")
        else:
            print(f"  ❌ {country}: 未找到数据")
    
    # 保存数据
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    panel_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ 数据已保存到: {output_path}")
    
    # 保存处理日志
    log_path = output_path.replace('.csv', '_processing_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("天然气消费面板数据处理日志\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"处理时间: {pd.Timestamp.now()}\\n")
        f.write(f"数据源单位: Billion cubic metres (bcm)\\n")
        f.write(f"最终面板规模: {len(panel_df):,} 行\\n")
        f.write(f"覆盖国家数: {panel_df['country'].nunique()}\\n")
        f.write(f"时间范围: {panel_df['year'].min()}-{panel_df['year'].max()}\\n")
        f.write(f"数据完整性: {valid_rows/total_rows*100:.1f}%\\n\\n")
        
        f.write("年度全球消费量:\\n")
        for year in key_years:
            if year in annual_totals.index:
                f.write(f"  {year}: {annual_totals[year]:,.1f} bcm\\n")
    
    print(f"✅ 处理日志已保存到: {log_path}")
    
    return True


def main():
    """主函数"""
    print("🚀 开始处理天然气消费数据")
    print("=" * 60)
    
    # 文件路径
    input_file = '08data/rawdata/gas_consumption.csv'
    output_file = 'outputs/gas_consumption_panel.csv'
    
    try:
        # 1. 加载原始数据
        raw_data = load_raw_data(input_file)
        
        # 2. 筛选有效国家
        country_data = filter_valid_countries(raw_data)
        
        # 3. 转换为面板格式
        panel_data = convert_to_panel_format(country_data)
        
        # 4. 清洗数值
        clean_data = clean_consumption_values(panel_data)
        
        # 5. 验证并保存
        validate_and_save(clean_data, output_file)
        
        print("\\n" + "=" * 60)
        print("🎉 天然气消费面板数据处理完成！")
        print(f"📁 输出文件: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ 处理过程中发生错误: {str(e)}")
        print("请检查数据质量或联系开发者")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)