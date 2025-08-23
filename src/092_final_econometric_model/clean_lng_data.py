#!/usr/bin/env python3
"""
LNG数据清理脚本
===============

专门处理lngdata.csv文件，构建干净的P_lng价格变量

输入：/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/08data/rawdata/lngdata.csv
输出：outputs/clean_lng_price_data.csv

处理步骤：
1. 加载原始LNG贸易数据
2. 计算单价 P_lng = primaryValue / netWgt  
3. 执行1%和99%缩尾处理
4. 按国家-年份聚合
5. 输出干净的价格数据

作者：Energy Network Analysis Team
版本：v1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def clean_lng_data():
    """主清理函数"""
    print("🚢 LNG数据清理脚本")
    print("=" * 50)
    
    # 输入输出路径
    input_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/08data/rawdata/lngdata.csv")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "clean_lng_price_data.csv"
    
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出文件: {output_path}")
    
    # 步骤1: 加载原始数据
    logger.info("📊 步骤1: 加载原始LNG数据...")
    
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return
    
    try:
        df_raw = pd.read_csv(input_path)
        logger.info(f"✅ 原始数据加载完成: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
        
        # 显示列名（用于调试）
        logger.info(f"原始列名: {list(df_raw.columns)}")
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {str(e)}")
        return
    
    # 步骤2: 数据预处理
    logger.info("🔧 步骤2: 数据预处理...")
    
    # 检查关键列是否存在
    required_cols = ['reporterISO', 'refYear', 'primaryValue', 'netWgt']
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    
    if missing_cols:
        logger.error(f"❌ 缺少必要列: {missing_cols}")
        logger.info(f"可用列: {list(df_raw.columns)}")
        return
    
    # 重命名列为标准名称
    df_clean = df_raw.rename(columns={
        'reporterISO': 'country',
        'refYear': 'year',
        'primaryValue': 'trade_value_usd',
        'netWgt': 'net_weight_kg'
    }).copy()
    
    logger.info(f"数据年份范围: {df_clean['year'].min()} - {df_clean['year'].max()}")
    logger.info(f"涵盖国家数: {df_clean['country'].nunique()}")
    
    # 步骤3: 价格计算前的数据清理
    logger.info("💰 步骤3: 计算LNG单价...")
    
    # 移除无效记录
    df_valid = df_clean[
        (df_clean['trade_value_usd'] > 0) & 
        (df_clean['net_weight_kg'] > 0) &
        (df_clean['trade_value_usd'].notna()) &
        (df_clean['net_weight_kg'].notna())
    ].copy()
    
    logger.info(f"有效记录数: {len(df_valid)} / {len(df_clean)} ({len(df_valid)/len(df_clean):.1%})")
    
    if len(df_valid) == 0:
        logger.error("❌ 没有有效的贸易记录")
        return
    
    # 计算原始单价 P_lng = Trade Value (US$) / Net Weight (kg)
    df_valid['P_lng_raw'] = df_valid['trade_value_usd'] / df_valid['net_weight_kg']
    
    # 显示原始价格统计
    logger.info(f"原始价格统计:")
    logger.info(f"  最小值: ${df_valid['P_lng_raw'].min():.4f}/kg")
    logger.info(f"  最大值: ${df_valid['P_lng_raw'].max():.4f}/kg")
    logger.info(f"  均值: ${df_valid['P_lng_raw'].mean():.4f}/kg")
    logger.info(f"  中位数: ${df_valid['P_lng_raw'].median():.4f}/kg")
    
    # 步骤4: 异常值处理（缩尾处理）
    logger.info("📐 步骤4: 执行1%和99%缩尾处理...")
    
    # 使用scipy的winsorize进行缩尾处理
    price_values = df_valid['P_lng_raw'].values
    winsorized_prices = mstats.winsorize(price_values, limits=[0.01, 0.01])
    
    df_valid['P_lng'] = winsorized_prices
    
    # 显示缩尾后价格统计
    logger.info(f"缩尾后价格统计:")
    logger.info(f"  最小值: ${df_valid['P_lng'].min():.4f}/kg")
    logger.info(f"  最大值: ${df_valid['P_lng'].max():.4f}/kg")
    logger.info(f"  均值: ${df_valid['P_lng'].mean():.4f}/kg")
    logger.info(f"  中位数: ${df_valid['P_lng'].median():.4f}/kg")
    
    # 步骤5: 按国家-年份聚合
    logger.info("📊 步骤5: 按国家-年份聚合数据...")
    
    # 聚合函数：对价格取均值，对贸易量求和
    df_aggregated = df_valid.groupby(['country', 'year']).agg({
        'P_lng': 'mean',  # 价格取均值
        'trade_value_usd': 'sum',  # 贸易额求和
        'net_weight_kg': 'sum',  # 重量求和
        'P_lng_raw': 'mean'  # 保留原始价格均值用于比较
    }).reset_index()
    
    # 重新计算聚合后的加权平均价格
    df_aggregated['P_lng_weighted'] = df_aggregated['trade_value_usd'] / df_aggregated['net_weight_kg']
    
    logger.info(f"聚合后记录数: {len(df_aggregated)}")
    logger.info(f"聚合后国家数: {df_aggregated['country'].nunique()}")
    logger.info(f"聚合后年份范围: {df_aggregated['year'].min()} - {df_aggregated['year'].max()}")
    
    # 步骤6: 最终数据质量检查
    logger.info("🔍 步骤6: 数据质量检查...")
    
    # 检查是否有异常价格
    high_price_threshold = df_aggregated['P_lng'].quantile(0.95)
    low_price_threshold = df_aggregated['P_lng'].quantile(0.05)
    
    high_price_count = (df_aggregated['P_lng'] > high_price_threshold).sum()
    low_price_count = (df_aggregated['P_lng'] < low_price_threshold).sum()
    
    logger.info(f"价格分布检查:")
    logger.info(f"  高价格记录 (>95分位): {high_price_count}")
    logger.info(f"  低价格记录 (<5分位): {low_price_count}")
    logger.info(f"  正常价格记录: {len(df_aggregated) - high_price_count - low_price_count}")
    
    # 检查每年数据覆盖
    yearly_coverage = df_aggregated.groupby('year')['country'].nunique().reset_index()
    yearly_coverage.columns = ['year', 'country_count']
    
    logger.info(f"年度数据覆盖:")
    for _, row in yearly_coverage.tail(10).iterrows():
        logger.info(f"  {int(row['year'])}: {int(row['country_count'])} 个国家")
    
    # 步骤7: 保存清理后的数据
    logger.info("💾 步骤7: 保存清理后的数据...")
    
    # 选择最终输出列
    final_columns = ['country', 'year', 'P_lng', 'trade_value_usd', 'net_weight_kg']
    df_final = df_aggregated[final_columns].copy()
    
    # 按国家和年份排序
    df_final = df_final.sort_values(['country', 'year'])
    
    try:
        df_final.to_csv(output_path, index=False)
        logger.info(f"✅ 清理后数据已保存: {output_path}")
        
        # 保存数据摘要
        summary_path = output_dir / "lng_data_cleaning_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("LNG数据清理摘要\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"原始数据: {len(df_raw)} 条记录\n")
            f.write(f"有效记录: {len(df_valid)} 条记录\n")
            f.write(f"最终聚合: {len(df_final)} 条记录\n")
            f.write(f"涵盖国家: {df_final['country'].nunique()} 个\n")
            f.write(f"时间范围: {df_final['year'].min()} - {df_final['year'].max()}\n\n")
            f.write(f"价格统计 (P_lng):\n")
            f.write(f"  最小值: ${df_final['P_lng'].min():.4f}/kg\n")
            f.write(f"  最大值: ${df_final['P_lng'].max():.4f}/kg\n")
            f.write(f"  均值: ${df_final['P_lng'].mean():.4f}/kg\n")
            f.write(f"  中位数: ${df_final['P_lng'].median():.4f}/kg\n")
            f.write(f"  标准差: ${df_final['P_lng'].std():.4f}/kg\n")
        
        logger.info(f"✅ 清理摘要已保存: {summary_path}")
        
    except Exception as e:
        logger.error(f"❌ 保存数据失败: {str(e)}")
        return
    
    # 步骤8: 显示示例数据
    logger.info("📋 步骤8: 显示示例数据...")
    
    print(f"\n清理后数据示例 (前10行):")
    print(df_final.head(10).to_string(index=False))
    
    print(f"\n主要统计信息:")
    print(f"• 最终记录数: {len(df_final):,}")
    print(f"• 涵盖国家: {df_final['country'].nunique()} 个")
    print(f"• 时间跨度: {df_final['year'].max() - df_final['year'].min() + 1} 年")
    print(f"• 平均LNG价格: ${df_final['P_lng'].mean():.4f}/kg")
    
    print(f"\n🎉 LNG数据清理完成！")
    print(f"✅ 输出文件: {output_path}")
    print(f"✅ 摘要文件: {summary_path}")

if __name__ == "__main__":
    clean_lng_data()