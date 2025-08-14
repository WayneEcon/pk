#!/usr/bin/env python3
"""
DLI指标计算模块 (DLI Calculator Module)
==========================================

本模块实现动态锁定指数(Dynamic Locking Index)的核心计算算法。
DLI通过四个维度量化国家间能源贸易关系的路径依赖和转换成本：

1. 贸易持续性 (Continuity): 衡量关系的长期性
2. 基础设施强度 (Infrastructure): 衡量专用性资产导致的锁定
3. 贸易稳定性 (Stability): 衡量关系的可靠性
4. 市场锁定力 (Market Locking Power): 衡量市场结构导致的锁定效应

最终通过主成分分析(PCA)确定权重，合成DLI总分。

作者：Energy Network Analysis Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 导入数据准备模块的全局数据加载功能
from data_preparation import load_global_trade_data_range
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算贸易持续性指标 (Continuity)
    
    公式: Continuity_ijt = (从2001年到t年存在贸易的年数) / (t - 2001 + 1)
    
    解释: 衡量美国与伙伴国j在能源产品i上的贸易关系持续性。
    值越接近1，表示贸易关系越稳定持续；值越接近0，表示贸易关系越不稳定。
    
    Args:
        df: 包含美国贸易数据的DataFrame
        必须包含列: year, us_partner, energy_product, trade_value_usd
        
    Returns:
        添加了continuity列的DataFrame
        
    示例:
        如果美国与加拿大的原油贸易在2001-2010年间有8年存在贸易，
        那么2010年的持续性 = 8 / (2010-2001+1) = 8/10 = 0.8
    """
    
    logger.info("🔄 开始计算贸易持续性指标...")
    
    df_continuity = df.copy()
    
    # 按伙伴国和能源产品分组计算持续性
    continuity_results = []
    
    # 获取所有unique组合（必须包含us_role以区分进出口）
    if 'us_role' in df_continuity.columns:
        groups = df_continuity.groupby(['us_partner', 'energy_product', 'us_role'])
    else:
        logger.warning("缺少us_role字段，将按国家-产品组合计算持续性（可能混合进出口数据）")
        groups = df_continuity.groupby(['us_partner', 'energy_product'])
    
    for group_key, group_data in groups:
        # 解包组合键
        if 'us_role' in df_continuity.columns:
            partner, product, us_role = group_key
        else:
            partner, product = group_key
            us_role = None
            
        # 获取该组合的所有年份
        trade_years = set(group_data['year'].unique())
        
        # 为每一年计算持续性
        for year in trade_years:
            # 计算从2001年到当前年份应有的年数
            total_possible_years = year - 2001 + 1
            
            # 计算从2001年到当前年份实际存在贸易的年数
            actual_trade_years = len([y for y in trade_years if 2001 <= y <= year])
            
            # 计算持续性
            continuity = actual_trade_years / total_possible_years if total_possible_years > 0 else 0
            
            result_record = {
                'year': year,
                'us_partner': partner,
                'energy_product': product,
                'continuity': continuity,
                'actual_trade_years': actual_trade_years,
                'total_possible_years': total_possible_years
            }
            
            if us_role is not None:
                result_record['us_role'] = us_role
                
            continuity_results.append(result_record)
    
    # 转换为DataFrame
    continuity_df = pd.DataFrame(continuity_results)
    
    # 与原数据合并（确保包含us_role字段避免进出口数据混淆）
    merge_keys = ['year', 'us_partner', 'energy_product']
    if 'us_role' in df_continuity.columns:
        merge_keys.append('us_role')
    
    df_with_continuity = pd.merge(
        df_continuity, 
        continuity_df[merge_keys + ['continuity']], 
        on=merge_keys, 
        how='left'
    )
    
    # 数据验证
    assert df_with_continuity['continuity'].isnull().sum() == 0, "持续性指标计算中存在缺失值"
    assert (df_with_continuity['continuity'] >= 0).all() and (df_with_continuity['continuity'] <= 1).all(), "持续性指标值超出[0,1]范围"
    
    # 统计摘要
    logger.info(f"📊 贸易持续性统计:")
    logger.info(f"  平均持续性: {df_with_continuity['continuity'].mean():.3f}")
    logger.info(f"  中位数持续性: {df_with_continuity['continuity'].median():.3f}")
    logger.info(f"  最高持续性: {df_with_continuity['continuity'].max():.3f}")
    logger.info(f"  最低持续性: {df_with_continuity['continuity'].min():.3f}")
    logger.info(f"  完全持续关系(=1): {(df_with_continuity['continuity'] == 1).sum()} 条记录")
    
    logger.info("✅ 贸易持续性指标计算完成!")
    return df_with_continuity

def calculate_infrastructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算基础设施强度指标 (Infrastructure)
    
    公式: Infrastructure_ijt = log(Trade_Volume_ijt / Distance_ij + 1)
    
    方法论说明:
    目标: 衡量由高沉没成本的专用性资产（如管道、专用港口）导致的锁定。
    挑战: 这些资产的价值是不可观测的。
    解决方案: 使用代理变量。经济学逻辑是：只有在地理距离近且贸易量巨大的情况下，
    投资昂贵的、不可移动的专用性基础设施才具有经济合理性。
    因此，贸易额/距离 这个比值，可以作为衡量这种"基础设施绑定"强度的有效代理。
    
    Args:
        df: 包含美国贸易数据的DataFrame
        必须包含列: trade_value_usd, distance_km
        
    Returns:
        添加了infrastructure列的DataFrame
        
    示例:
        美国与加拿大原油贸易额1000万美元，距离735公里
        Infrastructure = log(10000000 / 735 + 1) = log(13606 + 1) = log(13607) ≈ 9.52
    """
    
    logger.info("🏗️ 开始计算基础设施强度指标...")
    
    df_infrastructure = df.copy()
    
    # 数据验证
    assert 'trade_value_usd' in df_infrastructure.columns, "缺少trade_value_usd列"
    assert 'distance_km' in df_infrastructure.columns, "缺少distance_km列"
    assert (df_infrastructure['trade_value_usd'] > 0).all(), "贸易值必须大于0"
    assert (df_infrastructure['distance_km'] > 0).all(), "距离必须大于0"
    
    # 计算基础设施强度
    # 使用+1避免log(0)的情况，虽然在我们的数据中不太可能出现
    df_infrastructure['trade_distance_ratio'] = df_infrastructure['trade_value_usd'] / df_infrastructure['distance_km']
    df_infrastructure['infrastructure'] = np.log(df_infrastructure['trade_distance_ratio'] + 1)
    
    # 数据验证
    assert df_infrastructure['infrastructure'].isnull().sum() == 0, "基础设施指标计算中存在缺失值"
    assert (df_infrastructure['infrastructure'] >= 0).all(), "基础设施指标值不能为负"
    
    # 统计摘要
    logger.info(f"📊 基础设施强度统计:")
    logger.info(f"  平均强度: {df_infrastructure['infrastructure'].mean():.3f}")
    logger.info(f"  中位数强度: {df_infrastructure['infrastructure'].median():.3f}")
    logger.info(f"  最高强度: {df_infrastructure['infrastructure'].max():.3f}")
    logger.info(f"  最低强度: {df_infrastructure['infrastructure'].min():.3f}")
    logger.info(f"  标准差: {df_infrastructure['infrastructure'].std():.3f}")
    
    # 按距离区间分析
    if 'distance_category' in df_infrastructure.columns:
        distance_infra_stats = df_infrastructure.groupby('distance_category')['infrastructure'].agg(['mean', 'std', 'count'])
        logger.info(f"  按距离区间的基础设施强度:")
        for category, stats in distance_infra_stats.iterrows():
            logger.info(f"    {category}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, 记录数={stats['count']}")
    
    logger.info("✅ 基础设施强度指标计算完成!")
    return df_infrastructure

def calculate_stability(df: pd.DataFrame, window_years: int = 5) -> pd.DataFrame:
    """
    计算贸易稳定性指标 (Stability)
    
    公式: Stability_ijt = 1 / (CV_ijt + 0.1)
    其中 CV_ijt 是伙伴j在产品i上过去window_years年贸易额的变异系数 (标准差/平均值)
    
    解释: 衡量关系的可靠性，波动越小，锁定越强。
    变异系数越小，稳定性指标越大，表示贸易关系越稳定。
    
    Args:
        df: 包含美国贸易数据的DataFrame
        window_years: 计算稳定性的滑动窗口年数，默认为5年
        
    Returns:
        添加了stability列的DataFrame
        
    示例:
        如果某国某产品过去5年贸易额标准差为1000万，平均值为5000万
        CV = 1000/5000 = 0.2
        Stability = 1 / (0.2 + 0.1) = 1 / 0.3 = 3.33
    """
    
    logger.info(f"📈 开始计算贸易稳定性指标 (窗口期={window_years}年)...")
    
    df_stability = df.copy()
    
    # 按伙伴国、能源产品和贸易角色分组，计算每年的稳定性
    stability_results = []
    
    if 'us_role' in df_stability.columns:
        groups = df_stability.groupby(['us_partner', 'energy_product', 'us_role'])
    else:
        logger.warning("缺少us_role字段，将按国家-产品组合计算稳定性（可能混合进出口数据）")
        groups = df_stability.groupby(['us_partner', 'energy_product'])
    
    for group_key, group_data in groups:
        # 解包组合键
        if 'us_role' in df_stability.columns:
            partner, product, us_role = group_key
        else:
            partner, product = group_key
            us_role = None
        # 按年份排序
        group_data = group_data.sort_values('year')
        
        # 聚合每年的贸易额（因为可能有进口和出口）
        yearly_trade = group_data.groupby('year')['trade_value_usd'].sum().reset_index()
        
        # 为每一年计算稳定性
        for i, row in yearly_trade.iterrows():
            current_year = row['year']
            
            # 获取过去window_years年的历史数据（不包括当前年，避免前视偏误）
            start_year = current_year - window_years
            window_data = yearly_trade[
                (yearly_trade['year'] >= start_year) & 
                (yearly_trade['year'] < current_year)  # 严格小于当前年
            ]
            
            if len(window_data) >= 3:  # 至少需要3个历史观测值才可靠
                trade_values = window_data['trade_value_usd'].values
                
                # 计算变异系数 (CV = std / mean)
                mean_trade = np.mean(trade_values)
                std_trade = np.std(trade_values)
                
                if mean_trade > 0:
                    cv = std_trade / mean_trade
                    stability = 1 / (cv + 0.1)  # 加0.1避免分母为0
                else:
                    # 均值为0表示数据质量问题，标记为缺失
                    stability = np.nan
                    cv = np.nan
                
                result_record = {
                    'year': current_year,
                    'us_partner': partner,
                    'energy_product': product,
                    'stability': stability,
                    'cv': cv,
                    'window_years_used': len(window_data),
                    'mean_trade_value': mean_trade,
                    'std_trade_value': std_trade
                }
                
                if us_role is not None:
                    result_record['us_role'] = us_role
                    
                stability_results.append(result_record)
            elif len(window_data) == 2:
                # 历史数据不足但有一些信息，给予中等稳定性评分
                result_record = {
                    'year': current_year,
                    'us_partner': partner,
                    'energy_product': product,
                    'stability': 5.0,  # 中等水平稳定性
                    'cv': np.nan,
                    'window_years_used': len(window_data),
                    'mean_trade_value': np.mean(window_data['trade_value_usd'].values),
                    'std_trade_value': np.std(window_data['trade_value_usd'].values)
                }
                
                if us_role is not None:
                    result_record['us_role'] = us_role
                    
                stability_results.append(result_record)
            else:
                # 历史数据严重不足，标记为缺失值但记录当前年信息
                result_record = {
                    'year': current_year,
                    'us_partner': partner,
                    'energy_product': product,
                    'stability': np.nan,  # 数据不足，标记为缺失
                    'cv': np.nan,
                    'window_years_used': len(window_data),
                    'mean_trade_value': row['trade_value_usd'],
                    'std_trade_value': 0
                }
                
                if us_role is not None:
                    result_record['us_role'] = us_role
                    
                stability_results.append(result_record)
    
    # 转换为DataFrame
    stability_df = pd.DataFrame(stability_results)
    
    # 与原数据合并（确保包含us_role字段避免进出口数据混淆）
    merge_keys = ['year', 'us_partner', 'energy_product']
    if 'us_role' in df_stability.columns:
        merge_keys.append('us_role')
        
    df_with_stability = pd.merge(
        df_stability, 
        stability_df[merge_keys + ['stability']], 
        on=merge_keys, 
        how='left'
    )
    
    # 处理缺失值：对于历史数据不足的情况，使用全局平均稳定性
    missing_stability_count = df_with_stability['stability'].isnull().sum()
    if missing_stability_count > 0:
        logger.warning(f"发现 {missing_stability_count} 条稳定性指标缺失值，将使用全局均值填充")
        global_mean_stability = df_with_stability['stability'].mean()
        df_with_stability['stability'] = df_with_stability['stability'].fillna(global_mean_stability)
    
    # 数据验证（允许一定比例的缺失值，特别是在测试小批量数据时）
    missing_count = df_with_stability['stability'].isnull().sum()
    if missing_count > 0:
        missing_pct = missing_count / len(df_with_stability) * 100
        if missing_pct > 80:  # 如果超过80%缺失，抛出错误
            raise ValueError(f"稳定性指标缺失比例过高: {missing_pct:.1f}% ({missing_count}/{len(df_with_stability)})")
        elif missing_pct > 50:  # 如果超过50%缺失，给出警告
            logger.warning(f"稳定性指标缺失比例较高: {missing_pct:.1f}% ({missing_count}/{len(df_with_stability)})")
        else:
            logger.info(f"稳定性指标少量缺失: {missing_pct:.1f}% ({missing_count}/{len(df_with_stability)})")
    else:
        logger.info("✅ 稳定性指标无缺失值")
    valid_stability = df_with_stability['stability'][df_with_stability['stability'].notna()]
    if len(valid_stability) > 0:
        assert (valid_stability > 0).all(), "稳定性指标值必须大于0"
    
    # 统计摘要
    logger.info(f"📊 贸易稳定性统计:")
    logger.info(f"  平均稳定性: {df_with_stability['stability'].mean():.3f}")
    logger.info(f"  中位数稳定性: {df_with_stability['stability'].median():.3f}")
    logger.info(f"  最高稳定性: {df_with_stability['stability'].max():.3f}")
    logger.info(f"  最低稳定性: {df_with_stability['stability'].min():.3f}")
    logger.info(f"  标准差: {df_with_stability['stability'].std():.3f}")
    
    # 按产品分析稳定性
    product_stability = df_with_stability.groupby('energy_product')['stability'].agg(['mean', 'std', 'count'])
    logger.info(f"  按能源产品的稳定性:")
    for product, stats in product_stability.iterrows():
        logger.info(f"    {product}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, 记录数={stats['count']}")
    
    logger.info("✅ 贸易稳定性指标计算完成!")
    return df_with_stability

def calculate_import_locking_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算进口锁定力指标 (Import Locking Power)
    
    公式: Import_Locking_Power_ijt = HHI_it * share_ijt
    
    解释: 此指标衡量美国在进口时面临的市场结构锁定效应。
    美国在某产品i上的供应商市场越集中（赫芬达尔-赫希曼指数HHI_it越高），
    且当前伙伴j的贸易份额(share_ijt)越高，意味着替换该供应商的难度越大，
    因此其进口锁定力越强。
    
    Args:
        df: 包含美国贸易数据的DataFrame
        必须包含列: year, us_partner, energy_product, trade_value_usd, us_role
        
    Returns:
        添加了market_locking_power列的DataFrame（只计算进口部分）
        
    计算步骤:
        1. 筛选美国进口数据
        2. 按年份和产品分组计算供应商HHI
        3. 计算每个供应商在每种产品上的市场份额
        4. 进口锁定力 = 供应商HHI × 供应商份额
    """
    
    logger.info("📥 开始计算进口锁定力指标...")
    
    df_locking = df.copy()
    
    # 只处理美国作为进口方的数据
    import_data = df_locking[df_locking['us_role'] == 'importer'].copy()
    
    if len(import_data) == 0:
        logger.warning("没有找到美国进口数据，返回原数据")
        return df_locking
    
    locking_results = []
    
    # 按年份和产品计算HHI和市场份额
    for year in import_data['year'].unique():
        year_data = import_data[import_data['year'] == year]
        
        for product in year_data['energy_product'].unique():
            product_data = year_data[year_data['energy_product'] == product]
            
            # 计算总进口额
            total_import = product_data['trade_value_usd'].sum()
            
            if total_import > 0:
                # 计算每个供应商的市场份额
                supplier_shares = product_data.groupby('us_partner')['trade_value_usd'].sum() / total_import
                
                # 计算供应商HHI (Herfindahl-Hirschman Index)
                hhi = (supplier_shares ** 2).sum()
                
                # 为每个供应商计算进口锁定力
                for partner, share in supplier_shares.items():
                    import_locking_power = hhi * share
                    
                    locking_results.append({
                        'year': year,
                        'us_partner': partner,
                        'energy_product': product,
                        'us_role': 'importer',
                        'market_locking_power': import_locking_power,
                        'supplier_share': share,
                        'supplier_hhi': hhi,
                        'total_suppliers': len(supplier_shares)
                    })
    
    # 转换为DataFrame
    locking_df = pd.DataFrame(locking_results)
    
    # 与原数据合并（保持进口数据，出口数据稍后单独计算）
    df_with_locking = pd.merge(
        df_locking, 
        locking_df[['year', 'us_partner', 'energy_product', 'us_role', 'market_locking_power']], 
        on=['year', 'us_partner', 'energy_product', 'us_role'], 
        how='left'
    )
    
    # 填充缺失值为0
    df_with_locking['market_locking_power'] = df_with_locking['market_locking_power'].fillna(0)
    
    # 数据验证
    assert df_with_locking['market_locking_power'].isnull().sum() == 0, "市场锁定力指标计算中存在缺失值"
    assert (df_with_locking['market_locking_power'] >= 0).all(), "市场锁定力指标值不能为负"
    
    # 统计摘要
    logger.info(f"📊 市场锁定力统计:")
    logger.info(f"  平均锁定力: {df_with_locking['market_locking_power'].mean():.4f}")
    logger.info(f"  中位数锁定力: {df_with_locking['market_locking_power'].median():.4f}")
    logger.info(f"  最高锁定力: {df_with_locking['market_locking_power'].max():.4f}")
    logger.info(f"  最低锁定力: {df_with_locking['market_locking_power'].min():.4f}")
    logger.info(f"  非零锁定力记录: {(df_with_locking['market_locking_power'] > 0).sum()} 条")
    
    # 按产品分析锁定力（仅进口数据）
    if len(locking_df) > 0:
        product_locking = locking_df.groupby('energy_product').agg({
            'market_locking_power': ['mean', 'max'],
            'supplier_hhi': 'mean',
            'total_suppliers': 'mean'
        }).round(4)
        
        logger.info(f"  按能源产品的市场集中度:")
        for product in product_locking.index:
            stats = product_locking.loc[product]
            logger.info(f"    {product}: 平均HHI={stats[('supplier_hhi', 'mean')]:.4f}, " +
                       f"平均锁定力={stats[('market_locking_power', 'mean')]:.4f}, " +
                       f"平均供应商数={stats[('total_suppliers', 'mean')]:.1f}")
    
    logger.info("✅ 进口锁定力指标计算完成!")
    return df_with_locking


def calculate_export_locking_power(df: pd.DataFrame, global_trade_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    计算出口锁定力指标 (Export Locking Power) - 镜像计算逻辑
    
    理论框架：当美国向某国出口能源时，评估该国对美国的"被锁定"程度
    
    计算逻辑：
    1. 对于美国向国家X出口产品P的每一条记录
    2. 查询全球数据，找到国家X在该年份进口产品P的所有供应商
    3. 计算国家X在产品P上的进口集中度（供应商HHI）
    4. 计算美国在国家X的产品P进口中的份额
    5. 出口锁定力 = 国家X的进口HHI × 美国在X国市场的份额
    
    Args:
        df: 包含美国贸易数据的DataFrame
        global_trade_data: 全球贸易数据字典，格式{year: DataFrame}
        
    Returns:
        添加了market_locking_power列的DataFrame（只计算出口部分）
    """
    
    logger.info("📤 开始计算出口锁定力指标（镜像逻辑）...")
    
    df_locking = df.copy()
    
    # 只处理美国作为出口方的数据
    export_data = df_locking[df_locking['us_role'] == 'exporter'].copy()
    
    if len(export_data) == 0:
        logger.warning("没有找到美国出口数据，返回原数据")
        return df_locking
    
    if not global_trade_data:
        logger.warning("未提供全球贸易数据，出口锁定力将设为0")
        df_locking.loc[df_locking['us_role'] == 'exporter', 'market_locking_power'] = 0
        return df_locking
    
    locking_results = []
    
    # 为每个美国出口记录计算对应的出口锁定力
    for idx, row in export_data.iterrows():
        year = row['year']
        partner_country = row['us_partner']  # 美国的出口目标国
        product = row['energy_product']
        us_export_value = row['trade_value_usd']
        
        # 检查是否有该年份的全球数据
        if year not in global_trade_data:
            logger.debug(f"缺少{year}年全球数据，跳过")
            continue
        
        global_year_data = global_trade_data[year]
        
        # 查找目标国在该年份、该产品上的所有进口记录
        # 注意：在全球数据中，目标国作为reporter，流向为M(Import)
        partner_imports = global_year_data[
            (global_year_data['reporter'] == partner_country) & 
            (global_year_data['flow'] == 'M') & 
            (global_year_data['energy_product'] == product)
        ].copy()
        
        if len(partner_imports) == 0:
            # 目标国在该产品上没有进口记录，锁定力为0
            locking_results.append({
                'year': year,
                'us_partner': partner_country,
                'energy_product': product,
                'us_role': 'exporter',
                'market_locking_power': 0,
                'target_import_hhi': 0,
                'us_share_in_target': 0,
                'target_total_suppliers': 0,
                'target_total_imports': 0
            })
            continue
        
        # 计算目标国的总进口额
        total_imports = partner_imports['trade_value_usd'].sum()
        
        if total_imports <= 0:
            locking_results.append({
                'year': year,
                'us_partner': partner_country,
                'energy_product': product,
                'us_role': 'exporter',
                'market_locking_power': 0,
                'target_import_hhi': 0,
                'us_share_in_target': 0,
                'target_total_suppliers': 0,
                'target_total_imports': 0
            })
            continue
        
        # 计算目标国各供应商的市场份额
        supplier_shares = partner_imports.groupby('partner')['trade_value_usd'].sum() / total_imports
        
        # 计算目标国的进口集中度（供应商HHI）
        import_hhi = (supplier_shares ** 2).sum()
        
        # 计算美国在目标国市场中的份额
        us_share = supplier_shares.get('USA', 0)  # 如果美国不在供应商列表中，份额为0
        
        # 计算出口锁定力：目标国进口HHI × 美国在目标国市场的份额
        export_locking_power = import_hhi * us_share
        
        locking_results.append({
            'year': year,
            'us_partner': partner_country,
            'energy_product': product,
            'us_role': 'exporter',
            'market_locking_power': export_locking_power,
            'target_import_hhi': import_hhi,
            'us_share_in_target': us_share,
            'target_total_suppliers': len(supplier_shares),
            'target_total_imports': total_imports
        })
    
    # 转换为DataFrame
    locking_df = pd.DataFrame(locking_results)
    
    # 与原数据合并
    df_with_locking = pd.merge(
        df_locking, 
        locking_df[['year', 'us_partner', 'energy_product', 'us_role', 'market_locking_power']], 
        on=['year', 'us_partner', 'energy_product', 'us_role'], 
        how='left'
    )
    
    # 填充缺失值为0
    df_with_locking['market_locking_power'] = df_with_locking['market_locking_power'].fillna(0)
    
    # 统计摘要
    if len(locking_df) > 0:
        logger.info(f"📊 出口锁定力统计:")
        logger.info(f"  平均锁定力: {locking_df['market_locking_power'].mean():.4f}")
        logger.info(f"  最高锁定力: {locking_df['market_locking_power'].max():.4f}")
        logger.info(f"  非零锁定力记录: {(locking_df['market_locking_power'] > 0).sum()} 条")
        logger.info(f"  美国在目标市场平均份额: {locking_df['us_share_in_target'].mean():.4f}")
        logger.info(f"  目标国平均供应商数: {locking_df['target_total_suppliers'].mean():.1f}")
        
        # 按产品分析
        product_stats = locking_df.groupby('energy_product').agg({
            'market_locking_power': ['mean', 'max'],
            'target_import_hhi': 'mean',
            'us_share_in_target': 'mean'
        }).round(4)
        
        logger.info(f"  按能源产品的出口锁定力:")
        for product in product_stats.index:
            stats = product_stats.loc[product]
            logger.info(f"    {product}: 平均锁定力={stats[('market_locking_power', 'mean')]:.4f}, " +
                       f"目标国平均HHI={stats[('target_import_hhi', 'mean')]:.4f}")
    
    logger.info("✅ 出口锁定力指标计算完成!")
    return df_with_locking


def calculate_dli_composite(df: pd.DataFrame, 
                           use_pca: bool = True, 
                           custom_weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    计算DLI综合指标
    
    步骤:
    1. 标准化四个维度的分值（均值=0，标准差=1）
    2. 使用PCA确定权重（第一主成分的载荷作为权重）
    3. 计算加权综合得分
    
    Args:
        df: 包含四个DLI维度的DataFrame
        use_pca: 是否使用PCA确定权重，False则使用等权重或自定义权重
        custom_weights: 自定义权重字典，如 {'continuity': 0.3, 'infrastructure': 0.3, ...}
        
    Returns:
        添加了dli_composite列的DataFrame
        
    DLI公式:
        DLI_ijt = w1 * Continuity_ijt + w2 * Infrastructure_ijt + w3 * Stability_ijt + w4 * Market_Locking_Power_ijt
        权重由PCA第一主成分确定
    """
    
    logger.info("🎯 开始计算DLI综合指标...")
    
    df_composite = df.copy()
    
    # 检查必需的列
    required_columns = ['continuity', 'infrastructure', 'stability', 'market_locking_power']
    missing_columns = [col for col in required_columns if col not in df_composite.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的DLI维度列: {missing_columns}")
    
    # 提取四个维度的数据
    dli_dimensions = df_composite[required_columns].copy()
    
    # 数据验证：检查是否有缺失值或异常值
    logger.info("🔍 DLI维度数据质量检查:")
    for col in required_columns:
        missing_count = dli_dimensions[col].isnull().sum()
        if missing_count > 0:
            logger.warning(f"  {col}: {missing_count} 个缺失值")
        
        # 显示基本统计
        stats = dli_dimensions[col].describe()
        logger.info(f"  {col}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, " + 
                   f"范围=[{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # 处理缺失值（如果有）
    if dli_dimensions.isnull().any().any():
        logger.warning("发现缺失值，将使用各列均值填充")
        dli_dimensions = dli_dimensions.fillna(dli_dimensions.mean())
    
    # 第1步：标准化处理
    logger.info("📊 执行标准化处理...")
    scaler = StandardScaler()
    dli_standardized = scaler.fit_transform(dli_dimensions)
    dli_std_df = pd.DataFrame(dli_standardized, columns=required_columns, index=dli_dimensions.index)
    
    # 显示标准化后的统计
    logger.info("标准化后的数据统计:")
    for col in required_columns:
        mean_val = dli_std_df[col].mean()
        std_val = dli_std_df[col].std()
        logger.info(f"  {col}: 均值={mean_val:.6f}, 标准差={std_val:.6f}")
    
    # 第2步：确定权重
    if use_pca and custom_weights is None:
        logger.info("🔬 使用PCA确定权重...")
        
        # 执行主成分分析
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(dli_std_df)
        
        # 获取第一主成分的载荷作为权重
        pc1_loadings = pca.components_[0]
        
        # 确保权重为正值（取绝对值）并归一化
        weights = np.abs(pc1_loadings)
        weights = weights / weights.sum()
        
        weight_dict = dict(zip(required_columns, weights))
        
        # 显示PCA结果
        logger.info(f"📈 PCA分析结果:")
        logger.info(f"  第一主成分解释方差比: {pca.explained_variance_ratio_[0]:.3f}")
        logger.info(f"  累计解释方差比: {pca.explained_variance_ratio_[:2].sum():.3f} (前两个主成分)")
        logger.info(f"  PCA确定的权重:")
        for dim, weight in weight_dict.items():
            logger.info(f"    {dim}: {weight:.4f}")
            
    elif custom_weights is not None:
        logger.info("⚙️ 使用自定义权重...")
        
        # 验证自定义权重
        if set(custom_weights.keys()) != set(required_columns):
            raise ValueError(f"自定义权重必须包含所有维度: {required_columns}")
        
        if abs(sum(custom_weights.values()) - 1.0) > 1e-6:
            logger.warning("自定义权重之和不为1，将进行归一化")
            total_weight = sum(custom_weights.values())
            custom_weights = {k: v/total_weight for k, v in custom_weights.items()}
        
        weight_dict = custom_weights
        logger.info(f"  自定义权重:")
        for dim, weight in weight_dict.items():
            logger.info(f"    {dim}: {weight:.4f}")
    
    else:
        logger.info("⚖️ 使用等权重...")
        weight_dict = {dim: 0.25 for dim in required_columns}
        logger.info(f"  等权重: 每个维度权重 = 0.25")
    
    # 第3步：计算加权综合得分
    logger.info("🧮 计算DLI综合得分...")
    
    dli_composite_score = np.zeros(len(dli_std_df))
    
    for dim, weight in weight_dict.items():
        dli_composite_score += weight * dli_std_df[dim].values
    
    # 添加到原DataFrame
    df_composite['dli_composite'] = dli_composite_score
    
    # 为了便于解释，将DLI综合得分转换为正值（最小值映射为0）
    min_dli = df_composite['dli_composite'].min()
    if min_dli < 0:
        df_composite['dli_composite_adjusted'] = df_composite['dli_composite'] - min_dli
        logger.info(f"将DLI综合得分调整为非负值 (最小值调整: {min_dli:.4f})")
    else:
        df_composite['dli_composite_adjusted'] = df_composite['dli_composite']
    
    # 同时保存标准化后的各维度分值和权重信息
    for dim in required_columns:
        df_composite[f'{dim}_standardized'] = dli_std_df[dim]
    
    # 将权重信息保存为属性（用于后续分析）
    df_composite.attrs = {
        'dli_weights': weight_dict,
        'pca_explained_variance': pca.explained_variance_ratio_[0] if use_pca else None,
        'standardization_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    }
    
    # 统计摘要
    logger.info(f"📊 DLI综合指标统计:")
    logger.info(f"  原始DLI得分:")
    logger.info(f"    均值: {df_composite['dli_composite'].mean():.4f}")
    logger.info(f"    标准差: {df_composite['dli_composite'].std():.4f}")
    logger.info(f"    范围: [{df_composite['dli_composite'].min():.4f}, {df_composite['dli_composite'].max():.4f}]")
    
    if 'dli_composite_adjusted' in df_composite.columns:
        logger.info(f"  调整后DLI得分:")
        logger.info(f"    均值: {df_composite['dli_composite_adjusted'].mean():.4f}")
        logger.info(f"    标准差: {df_composite['dli_composite_adjusted'].std():.4f}")
        logger.info(f"    范围: [{df_composite['dli_composite_adjusted'].min():.4f}, {df_composite['dli_composite_adjusted'].max():.4f}]")
    
    # 显示各维度与综合得分的相关性
    logger.info(f"  各维度与DLI综合得分的相关性:")
    for dim in required_columns:
        corr = df_composite[dim].corr(df_composite['dli_composite'])
        logger.info(f"    {dim}: {corr:.4f}")
    
    logger.info("✅ DLI综合指标计算完成!")
    return df_composite

def calculate_dli_composite_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用统一权重计算双向DLI综合指标
    
    关键特点：
    1. 使用包含进口和出口的完整数据集来运行PCA
    2. 确保所有dli_score都在同一标尺下可比
    3. 权重统一性原则的具体实现
    
    Args:
        df: 包含四个DLI维度的完整DataFrame（进口+出口）
        
    Returns:
        添加了统一标尺dli_score列的DataFrame
    """
    
    logger.info("🎯 计算统一标尺DLI综合指标...")
    
    df_unified = df.copy()
    
    # 检查必需的四个维度
    required_dimensions = ['continuity', 'infrastructure', 'stability', 'market_locking_power']
    missing_dimensions = [dim for dim in required_dimensions if dim not in df_unified.columns]
    if missing_dimensions:
        raise ValueError(f"缺少DLI维度: {missing_dimensions}")
    
    logger.info("🔍 双向DLI维度数据质量检查:")
    for dim in required_dimensions:
        logger.info(f"  {dim}: 均值={df_unified[dim].mean():.4f}, 标准差={df_unified[dim].std():.4f}, " + 
                   f"范围=[{df_unified[dim].min():.4f}, {df_unified[dim].max():.4f}]")
    
    # 标准化处理（使用全部数据的均值和标准差）
    logger.info("📊 对完整双向数据集执行标准化...")
    scaler = StandardScaler()
    standardized_dimensions = scaler.fit_transform(df_unified[required_dimensions])
    
    standardized_df = pd.DataFrame(standardized_dimensions, columns=required_dimensions, index=df_unified.index)
    
    # 验证标准化效果
    logger.info("标准化后的数据统计:")
    for dim in required_dimensions:
        logger.info(f"  {dim}: 均值={standardized_df[dim].mean():.6f}, 标准差={standardized_df[dim].std():.6f}")
    
    # 使用PCA确定统一权重
    logger.info("🔬 使用完整数据集运行PCA确定统一权重...")
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(standardized_dimensions)
    
    # 获取第一主成分的载荷作为权重
    first_component = pca.components_[0]
    weights_raw = np.abs(first_component)  # 取绝对值
    weights_normalized = weights_raw / weights_raw.sum()  # 归一化
    
    # 创建权重字典
    weights_dict = dict(zip(required_dimensions, weights_normalized))
    
    logger.info("📈 统一PCA权重结果:")
    logger.info(f"  第一主成分解释方差比: {pca.explained_variance_ratio_[0]:.3f}")
    logger.info(f"  累计解释方差比: {pca.explained_variance_ratio_[:2].sum():.3f} (前两个主成分)")
    logger.info("  统一权重分配:")
    for dim, weight in weights_dict.items():
        logger.info(f"    {dim}: {weight:.4f}")
    
    # 计算统一标尺下的DLI综合得分
    logger.info("🧮 计算统一标尺DLI综合得分...")
    dli_scores = []
    for idx in df_unified.index:
        score = sum(standardized_df.loc[idx, dim] * weights_dict[dim] for dim in required_dimensions)
        dli_scores.append(score)
    
    df_unified['dli_score'] = dli_scores
    
    # 调整为非负值（如果需要）
    min_score = df_unified['dli_score'].min()
    if min_score < 0:
        df_unified['dli_score_adjusted'] = df_unified['dli_score'] - min_score
        logger.info(f"将DLI得分调整为非负值 (最小值调整: {min_score:.4f})")
    else:
        df_unified['dli_score_adjusted'] = df_unified['dli_score']
    
    # 统计摘要
    logger.info(f"📊 统一DLI综合指标统计:")
    logger.info(f"  原始DLI得分:")
    logger.info(f"    均值: {df_unified['dli_score'].mean():.4f}")
    logger.info(f"    标准差: {df_unified['dli_score'].std():.4f}")
    logger.info(f"    范围: [{df_unified['dli_score'].min():.4f}, {df_unified['dli_score'].max():.4f}]")
    
    if 'dli_score_adjusted' in df_unified.columns:
        logger.info(f"  调整后DLI得分:")
        logger.info(f"    均值: {df_unified['dli_score_adjusted'].mean():.4f}")
        logger.info(f"    标准差: {df_unified['dli_score_adjusted'].std():.4f}")
        logger.info(f"    范围: [{df_unified['dli_score_adjusted'].min():.4f}, {df_unified['dli_score_adjusted'].max():.4f}]")
    
    # 按锁定类型分析
    if 'locking_dimension_type' in df_unified.columns:
        type_stats = df_unified.groupby('locking_dimension_type')['dli_score'].agg(['count', 'mean', 'std']).round(4)
        logger.info("  按锁定类型统计:")
        for locking_type, stats in type_stats.iterrows():
            logger.info(f"    {locking_type}: {stats['count']} 条记录, 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    
    # 保存权重信息用于后续分析
    df_unified._pca_weights = weights_dict
    df_unified._pca_explained_variance = pca.explained_variance_ratio_[0]
    
    logger.info("✅ 统一DLI综合指标计算完成!")
    return df_unified


def generate_dli_panel_data_v2(trade_data: pd.DataFrame = None, 
                              data_file_path: str = None,
                              output_path: str = None,
                              enable_global_data: bool = True) -> pd.DataFrame:
    """
    生成双向DLI面板数据集 (Version 2.0)
    
    这是升级版的DLI计算模块主要接口，支持双向锁定分析：
    - 进口锁定 (Import Locking): 美国被供应商锁定的程度
    - 出口锁定 (Export Locking): 美国锁定其他国家的程度
    
    Args:
        trade_data: 预处理的美国贸易数据DataFrame，如果为None则从文件加载
        data_file_path: 数据文件路径，默认使用标准路径
        output_path: 输出文件路径，默认保存到outputs目录
        enable_global_data: 是否加载全局数据以计算出口锁定力，默认True
        
    Returns:
        包含双向DLI指标的面板数据DataFrame，增加locking_dimension_type列
        
    输出列包括：
        - 基础数据：year, us_partner, energy_product, trade_value_usd, distance_km等
        - 锁定维度类型：locking_dimension_type ('import_locking' 或 'export_locking')
        - DLI四维度：continuity, infrastructure, stability, market_locking_power  
        - 统一标尺综合指标：dli_score (使用统一PCA权重)
    """
    
    logger.info("🚀 开始生成双向DLI面板数据 (v2.0)...")
    
    # 第1步：加载美国贸易数据
    if trade_data is not None:
        df = trade_data.copy()
        logger.info(f"使用提供的贸易数据: {len(df)} 条记录")
    else:
        if data_file_path is None:
            base_dir = Path(__file__).parent.parent.parent
            data_file_path = base_dir / "outputs" / "tables" / "us_trade_prepared_for_dli.csv"
        
        if not Path(data_file_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file_path}")
        
        df = pd.read_csv(data_file_path)
        logger.info(f"从文件加载贸易数据: {data_file_path}, {len(df)} 条记录")
    
    # 数据验证
    required_base_columns = ['year', 'us_partner', 'energy_product', 'trade_value_usd', 'distance_km', 'us_role']
    missing_columns = [col for col in required_base_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必需列: {missing_columns}")
    
    # 第2步：加载全局贸易数据（用于计算出口锁定力）
    global_trade_data = {}
    if enable_global_data:
        try:
            logger.info("🌍 加载全球贸易数据以支持出口锁定力计算...")
            years_needed = sorted(df['year'].unique())
            global_trade_data = load_global_trade_data_range(
                start_year=min(years_needed), 
                end_year=max(years_needed)
            )
            logger.info(f"✅ 成功加载{len(global_trade_data)}年全球数据")
        except Exception as e:
            logger.error(f"❌ 加载全局数据失败: {e}")
            logger.warning("将只计算进口锁定力，出口锁定力设为0")
            global_trade_data = {}
    else:
        logger.info("⚠️ 全局数据加载已禁用，出口锁定力将设为0")
    
    # 第3步：分别计算进口和出口数据的前三个维度
    logger.info("🔄 计算基础DLI维度（持续性、基础设施、稳定性）...")
    
    # 基础三维度计算（进口和出口共享）
    df = calculate_continuity(df)
    df = calculate_infrastructure(df)  
    df = calculate_stability(df)
    
    # 第4步：分别计算进口和出口的市场锁定力
    logger.info("🔒 计算双向市场锁定力...")
    
    # 分离进口和出口数据
    import_data = df[df['us_role'] == 'importer'].copy()
    export_data = df[df['us_role'] == 'exporter'].copy()
    
    # 计算进口锁定力
    if len(import_data) > 0:
        import_data = calculate_import_locking_power(import_data)
        import_data['locking_dimension_type'] = 'import_locking'
    
    # 计算出口锁定力
    if len(export_data) > 0:
        export_data = calculate_export_locking_power(export_data, global_trade_data)
        export_data['locking_dimension_type'] = 'export_locking'
    
    # 合并进口和出口数据
    if len(import_data) > 0 and len(export_data) > 0:
        df_combined = pd.concat([import_data, export_data], ignore_index=True)
    elif len(import_data) > 0:
        df_combined = import_data
    elif len(export_data) > 0:
        df_combined = export_data
    else:
        raise ValueError("没有找到有效的进口或出口数据")
    
    logger.info(f"✅ 双向数据合并完成: {len(df_combined)} 条记录")
    logger.info(f"  进口锁定记录: {(df_combined['locking_dimension_type'] == 'import_locking').sum()}")
    logger.info(f"  出口锁定记录: {(df_combined['locking_dimension_type'] == 'export_locking').sum()}")
    
    # 第5步：使用全部数据重新运行PCA获得统一权重
    logger.info("🧮 使用完整双向数据重新计算统一PCA权重...")
    df_final = calculate_dli_composite_unified(df_combined)
    
    # 第6步：数据整理和验证
    logger.info("🔧 最终数据整理...")
    
    # 选择需要保存的列（适用于双向DLI分析）
    output_columns = [
        # 基础标识列
        'year', 'us_partner', 'energy_product', 'us_role', 'locking_dimension_type',
        # 基础数据列  
        'trade_value_usd', 'distance_km', 'distance_category',
        # DLI四个维度
        'continuity', 'infrastructure', 'stability', 'market_locking_power',
        # 统一标尺综合指标
        'dli_score', 'dli_score_adjusted'
    ]
    
    # 确保所有列都存在
    available_columns = [col for col in output_columns if col in df_final.columns]
    df_output = df_final[available_columns].copy()
    
    # 按关键字段排序（双向DLI排序）
    df_output = df_output.sort_values(['year', 'us_partner', 'energy_product', 'us_role', 'locking_dimension_type'])
    df_output = df_output.reset_index(drop=True)
    
    # 最终数据验证
    logger.info("🔍 双向DLI数据集最终验证:")
    logger.info(f"  总记录数: {len(df_output):,}")
    logger.info(f"  年份范围: {df_output['year'].min()}-{df_output['year'].max()}")
    logger.info(f"  贸易伙伴数: {df_output['us_partner'].nunique()}")
    logger.info(f"  能源产品数: {df_output['energy_product'].nunique()}")
    
    # 按锁定类型统计
    if 'locking_dimension_type' in df_output.columns:
        type_counts = df_output['locking_dimension_type'].value_counts()
        logger.info(f"  锁定维度类型分布:")
        for ltype, count in type_counts.items():
            logger.info(f"    {ltype}: {count:,} 条记录 ({count/len(df_output)*100:.1f}%)")
    
    # 检查缺失值
    missing_summary = df_output.isnull().sum()
    if missing_summary.any():
        logger.warning("发现缺失值:")
        for col, count in missing_summary[missing_summary > 0].items():
            logger.warning(f"  {col}: {count} 个缺失值")
    else:
        logger.info("✅ 无缺失值")
    
    # 双向DLI指标统计摘要
    dli_columns = ['continuity', 'infrastructure', 'stability', 'market_locking_power', 'dli_score_adjusted']
    logger.info(f"📊 双向DLI指标最终统计摘要:")
    for col in dli_columns:
        if col in df_output.columns:
            stats = df_output[col].describe()
            logger.info(f"  {col}:")
            logger.info(f"    均值±标准差: {stats['mean']:.4f}±{stats['std']:.4f}")
            logger.info(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"    分位数(25%,50%,75%): {stats['25%']:.4f}, {stats['50%']:.4f}, {stats['75%']:.4f}")
    
    # 第7步：导出数据（双向DLI版本）
    if output_path is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = Path(__file__).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dli_panel_data_v2.csv"
    
    df_output.to_csv(output_path, index=False)
    logger.info(f"💾 双向DLI面板数据已保存至: {output_path}")
    
    # 保存统一权重信息到json文件
    if hasattr(df_final, '_pca_weights'):
        import json
        weights_path = Path(output_path).parent / "dli_weights_and_params_v2.json"
        
        weights_info = {
            'version': '2.0',
            'description': '双向DLI分析统一权重系统',
            'unified_pca_weights': df_final._pca_weights,
            'pca_explained_variance': df_final._pca_explained_variance,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'total_records': len(df_output),
                'year_range': [int(df_output['year'].min()), int(df_output['year'].max())],
                'num_partners': int(df_output['us_partner'].nunique()),
                'num_products': int(df_output['energy_product'].nunique()),
                'import_locking_records': int((df_output['locking_dimension_type'] == 'import_locking').sum()),
                'export_locking_records': int((df_output['locking_dimension_type'] == 'export_locking').sum())
            },
            'methodology_notes': {
                'pca_basis': '使用包含进口和出口的完整数据集运行PCA',
                'weight_calculation': '第一主成分载荷的绝对值归一化',
                'score_comparability': '所有dli_score都在统一标尺下可比',
                'locking_types': {
                    'import_locking': '美国被供应商锁定的程度',
                    'export_locking': '美国锁定其他国家的程度（镜像计算）'
                }
            }
        }
        
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_info, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 双向DLI权重信息已保存至: {weights_path}")
    
    logger.info("🎉 双向DLI面板数据生成完成!")
    return df_output

if __name__ == "__main__":
    # 测试双向DLI计算功能
    try:
        dli_panel = generate_dli_panel_data_v2()
        print(f"✅ 双向DLI面板数据生成成功!")
        print(f"📊 数据维度: {dli_panel.shape}")
        print(f"🔗 DLI综合指标范围: [{dli_panel['dli_score'].min():.4f}, {dli_panel['dli_score'].max():.4f}]")
        
        # 显示双向数据统计
        locking_stats = dli_panel.groupby('locking_dimension_type').agg({
            'dli_score': ['count', 'mean', 'std']
        }).round(4)
        print(f"📈 双向锁定统计:")
        print(locking_stats)
        
    except Exception as e:
        logger.error(f"❌ 双向DLI计算失败: {e}")
        raise