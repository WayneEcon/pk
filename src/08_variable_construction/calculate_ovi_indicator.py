#!/usr/bin/env python3
"""
OVI指标计算脚本 - 构建最终的LNG进口脆弱性指标

本脚本整合LNG终端容量和天然气消费数据，生成用于计量分析的OVI_LNG指标。

核心计算流程：
1. 加载两个面板数据并进行国家名称标准化
2. 以消费数据为主表执行左合并
3. 计算原始OVI = 容量/消费量
4. 应用3年滚动平均平滑处理
5. 异常值裁剪到[0,10]区间
6. 生成滞后一期的最终指标

作者: Claude Code
创建时间: 2025-08-23
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径以导入country_standardizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from country_standardizer import CountryStandardizer
    print("✅ 成功导入country_standardizer模块")
except ImportError as e:
    print(f"❌ 导入country_standardizer失败: {e}")
    print("请确保country_standardizer.py在同一目录中")
    sys.exit(1)


def load_panel_datasets():
    """加载并验证两个面板数据"""
    print("=== 第1步：加载面板数据 ===")
    
    # 文件路径
    lng_file = 'outputs/lng_terminal_capacity_panel.csv'
    gas_file = 'outputs/gas_consumption_panel.csv'
    
    # 加载数据
    try:
        lng_df = pd.read_csv(lng_file)
        gas_df = pd.read_csv(gas_file)
        print(f"LNG容量面板: {lng_df.shape} ({lng_df['country'].nunique()} 国家)")
        print(f"天然气消费面板: {gas_df.shape} ({gas_df['country'].nunique()} 国家)")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        sys.exit(1)
    
    # 验证列结构
    required_lng_cols = ['country', 'year', 'lng_import_capacity_bcm']
    required_gas_cols = ['country', 'year', 'gas_consumption_bcm']
    
    if not all(col in lng_df.columns for col in required_lng_cols):
        print(f"❌ LNG数据缺失必要列: {required_lng_cols}")
        sys.exit(1)
        
    if not all(col in gas_df.columns for col in required_gas_cols):
        print(f"❌ 天然气数据缺失必要列: {required_gas_cols}")
        sys.exit(1)
    
    print("✅ 数据加载完成")
    return lng_df, gas_df


def standardize_country_names(lng_df, gas_df):
    """使用CountryStandardizer标准化国家名称"""
    print("\n=== 第2步：国家名称标准化 ===")
    
    standardizer = CountryStandardizer()
    
    # 标准化LNG数据的国家名称
    print("🔧 标准化LNG数据国家名称...")
    lng_df_std = standardizer.standardize_dataframe(lng_df, 'country', 'country_std')
    
    # 标准化天然气消费数据的国家名称
    print("🔧 标准化天然气消费数据国家名称...")
    gas_df_std = standardizer.standardize_dataframe(gas_df, 'country', 'country_std')
    
    # 移除无法标准化的国家
    lng_df_clean = lng_df_std[lng_df_std['country_std'].notna()].copy()
    gas_df_clean = gas_df_std[gas_df_std['country_std'].notna()].copy()
    
    print(f"\n📊 标准化结果:")
    print(f"LNG数据: {len(lng_df)} → {len(lng_df_clean)} 行 ({len(lng_df_clean['country_std'].unique())} 国家)")
    print(f"天然气数据: {len(gas_df)} → {len(gas_df_clean)} 行 ({len(gas_df_clean['country_std'].unique())} 国家)")
    
    # 使用标准化后的国家代码作为主键
    # 先删除原来的country列，再重命名
    lng_df_clean = lng_df_clean.drop(columns=['country']).rename(columns={'country_std': 'country'})
    gas_df_clean = gas_df_clean.drop(columns=['country']).rename(columns={'country_std': 'country'})
    
    print("✅ 国家名称标准化完成")
    return lng_df_clean, gas_df_clean


def check_country_coverage(lng_df, gas_df):
    """检查两个数据集的国家覆盖差异"""
    print("\n=== 第3步：国家覆盖分析 ===")
    
    lng_countries = set(lng_df['country'].unique())
    gas_countries = set(gas_df['country'].unique())
    
    overlap = lng_countries.intersection(gas_countries)
    lng_only = lng_countries - gas_countries
    gas_only = gas_countries - lng_countries
    
    print(f"🌍 覆盖分析结果:")
    print(f"  共同国家: {len(overlap)} 个")
    print(f"  仅LNG数据: {len(lng_only)} 个 {sorted(list(lng_only))[:10]}")
    print(f"  仅消费数据: {len(gas_only)} 个 {sorted(list(gas_only))[:10]}")
    
    if len(lng_only) > 0:
        print(f"\n⚠️ 注意：{len(lng_only)}个国家只有LNG容量数据但没有消费数据")
        print("这些国家在左合并后将被排除")
        
    print("✅ 国家覆盖分析完成")
    return True


def merge_panels(lng_df, gas_df):
    """执行左合并并处理缺失容量数据"""
    print("\n=== 第4步：面板数据合并 ===")
    
    # 数据清理：删除USSR数据
    print("🧹 数据清理：删除无效实体USSR...")
    original_gas_count = len(gas_df)
    gas_df = gas_df[gas_df['country'] != 'USSR'].copy()
    ussr_removed = original_gas_count - len(gas_df)
    if ussr_removed > 0:
        print(f"   删除USSR数据: {ussr_removed} 行")
    
    print("🔧 执行左合并 (以天然气消费为主表)...")
    
    # 以gas_df为主表执行左合并
    merged_df = gas_df.merge(
        lng_df[['country', 'year', 'lng_import_capacity_bcm']], 
        on=['country', 'year'], 
        how='left'
    )
    
    print(f"合并结果: {merged_df.shape}")
    
    # 处理缺失的LNG容量数据 - 填充为0
    missing_capacity = merged_df['lng_import_capacity_bcm'].isna().sum()
    print(f"缺失LNG容量数据: {missing_capacity} 行")
    
    if missing_capacity > 0:
        print("🔧 将缺失的LNG容量数据填充为0 (确实没有LNG进口设施)")
        merged_df['lng_import_capacity_bcm'] = merged_df['lng_import_capacity_bcm'].fillna(0.0)
    
    print("✅ 面板合并完成")
    return merged_df


def calculate_ovi_indicator(merged_df):
    """计算OVI指标的完整流程"""
    print("\n=== 第5步：OVI指标计算 ===")
    
    df = merged_df.copy()
    
    # Step 1: 按业务规则处理零消费数据
    print("🔍 应用业务规则处理零消费数据...")
    zero_consumption = (df['gas_consumption_bcm'] <= 0) | (df['gas_consumption_bcm'].isna())
    zero_count = zero_consumption.sum()
    positive_consumption = df['gas_consumption_bcm'] > 0
    positive_count = positive_consumption.sum()
    
    print(f"零消费/负消费数据: {zero_count} 行")
    print(f"正常消费数据: {positive_count} 行")
    
    if zero_count > 0:
        # 显示零消费国家统计
        zero_countries = df[zero_consumption]['country'].value_counts()
        print(f"涉及零消费国家: {zero_countries.to_dict()}")
    
    # Step 2: 按业务规则计算原始OVI
    print("🔧 按业务规则计算原始OVI...")
    df['OVI_LNG_raw'] = 0.0  # 初始化为0
    
    # 对于正消费的行，执行除法
    df.loc[positive_consumption, 'OVI_LNG_raw'] = (
        df.loc[positive_consumption, 'lng_import_capacity_bcm'] / 
        df.loc[positive_consumption, 'gas_consumption_bcm']
    )
    
    # 对于零消费的行，OVI直接设为0 (已经在初始化时设置)
    print(f"   正消费国家OVI计算: {positive_count} 行")
    print(f"   零消费国家OVI设为0: {zero_count} 行")
    
    # 验证计算结果
    inf_count = np.isinf(df['OVI_LNG_raw']).sum()
    nan_count = df['OVI_LNG_raw'].isna().sum()
    
    if inf_count > 0 or nan_count > 0:
        print(f"⚠️ 原始OVI计算异常: {inf_count} inf值, {nan_count} NaN值")
        return None
    
    print(f"原始OVI统计: min={df['OVI_LNG_raw'].min():.4f}, max={df['OVI_LNG_raw'].max():.4f}, mean={df['OVI_LNG_raw'].mean():.4f}")
    
    # Step 3: 3年滚动平均平滑处理
    print("🔧 应用3年滚动平均平滑...")
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    df['OVI_LNG_smoothed'] = df.groupby('country')['OVI_LNG_raw'].rolling(
        window=3, min_periods=1
    ).mean().reset_index(level=0, drop=True)
    
    # Step 4: 异常值裁剪到[0,10]区间
    print("🔧 异常值裁剪到[0,10]区间...")
    df['OVI_LNG_clipped'] = df['OVI_LNG_smoothed'].clip(0, 10)
    
    clipped_count = (df['OVI_LNG_smoothed'] != df['OVI_LNG_clipped']).sum()
    print(f"被裁剪的异常值: {clipped_count} 个")
    
    # Step 5: 生成滞后一期指标
    print("🔧 生成滞后一期最终指标...")
    df['OVI_LNG_final'] = df.groupby('country')['OVI_LNG_clipped'].shift(1)
    
    # 最终统计
    final_valid = df['OVI_LNG_final'].notna().sum()
    print(f"最终有效OVI观测值: {final_valid}")
    
    print("✅ OVI指标计算完成")
    return df


def validate_and_save_results(df, output_path):
    """验证最终指标并保存结果"""
    print("\n=== 第6步：结果验证与保存 ===")
    
    if df is None:
        print("❌ 无法保存，数据处理失败")
        return False
    
    # 验证最终指标
    print("📊 最终OVI指标描述性统计:")
    final_stats = df['OVI_LNG_final'].describe()
    print(final_stats)
    
    # 按年度统计
    print("\n📊 年度OVI指标统计 (平均值):")
    annual_ovi = df.groupby('year')['OVI_LNG_final'].mean()
    key_years = [2005, 2010, 2015, 2020, 2024]
    for year in key_years:
        if year in annual_ovi.index:
            print(f"  {year}年: {annual_ovi[year]:.4f}")
    
    # 主要国家验证
    print("\n📊 主要LNG进口国OVI指标 (2020年):")
    major_importers = ['JPN', 'KOR', 'CHN', 'IND', 'ESP', 'FRA', 'GBR', 'TUR']
    data_2020 = df[df['year'] == 2020]
    
    for country in major_importers:
        country_data = data_2020[data_2020['country'] == country]
        if len(country_data) > 0 and country_data['OVI_LNG_final'].notna().any():
            ovi_value = country_data['OVI_LNG_final'].iloc[0]
            print(f"  ✅ {country}: {ovi_value:.4f}")
        else:
            print(f"  ❌ {country}: 无数据")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存最终数据
    df_output = df.sort_values(['country', 'year']).reset_index(drop=True)
    df_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 最终数据已保存到: {output_path}")
    
    
    return True


def main():
    """主函数"""
    print("🚀 开始构建OVI_LNG脆弱性指标")
    print("=" * 60)
    
    # 输出路径
    output_file = 'outputs/ovi_gas.csv'
    
    try:
        # 1. 加载面板数据
        lng_df, gas_df = load_panel_datasets()
        
        # 2. 国家名称标准化
        lng_df_std, gas_df_std = standardize_country_names(lng_df, gas_df)
        
        # 3. 检查国家覆盖差异
        check_country_coverage(lng_df_std, gas_df_std)
        
        # 4. 合并面板数据
        merged_df = merge_panels(lng_df_std, gas_df_std)
        
        # 5. 计算OVI指标
        final_df = calculate_ovi_indicator(merged_df)
        
        # 6. 验证并保存结果
        success = validate_and_save_results(final_df, output_file)
        
        if success:
            print("\\n" + "=" * 60)
            print("🎉 OVI_LNG指标构建完成！")
            print(f"📁 输出文件: {output_file}")
            return True
        else:
            print("❌ OVI指标构建失败")
            return False
        
    except Exception as e:
        print(f"\\n❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)