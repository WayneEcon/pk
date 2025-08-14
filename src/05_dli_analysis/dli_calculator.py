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
    
    # 数据验证（修改后的验证逻辑）
    assert df_with_stability['stability'].isnull().sum() == 0, "稳定性指标计算中仍存在缺失值"
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

def calculate_market_locking_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场锁定力指标 (Market Locking Power)
    
    公式: Market_Locking_Power_ijt = HHI_it * share_ijt
    
    解释: 此指标直接衡量市场结构导致的锁定效应。
    美国在某产品i上的供应商市场越集中（赫芬达尔-赫希曼指数HHI_it越高），
    且当前伙伴j的贸易份额(share_ijt)越高，意味着替换该伙伴的难度越大，
    因此其市场锁定力越强。
    
    Args:
        df: 包含美国贸易数据的DataFrame
        必须包含列: year, us_partner, energy_product, trade_value_usd, us_role
        
    Returns:
        添加了market_locking_power列的DataFrame
        
    计算步骤:
        1. 按年份和产品分组计算HHI（基于进口份额）
        2. 计算每个伙伴国在每种产品上的市场份额
        3. 市场锁定力 = HHI × 市场份额
    """
    
    logger.info("🔒 开始计算市场锁定力指标...")
    
    df_locking = df.copy()
    
    # 只考虑美国作为进口方的数据来计算供应商集中度
    import_data = df_locking[df_locking['us_role'] == 'importer'].copy()
    
    if len(import_data) == 0:
        logger.warning("没有找到美国进口数据，市场锁定力将设为0")
        df_locking['market_locking_power'] = 0
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
                
                # 计算HHI (Herfindahl-Hirschman Index)
                # HHI = Σ(market_share_i)^2
                hhi = (supplier_shares ** 2).sum()
                
                # 为每个供应商计算市场锁定力
                for partner, share in supplier_shares.items():
                    market_locking_power = hhi * share
                    
                    locking_results.append({
                        'year': year,
                        'us_partner': partner,
                        'energy_product': product,
                        'us_role': 'importer',
                        'market_locking_power': market_locking_power,
                        'market_share': share,
                        'hhi': hhi,
                        'total_suppliers': len(supplier_shares)
                    })
    
    # 转换为DataFrame
    locking_df = pd.DataFrame(locking_results)
    
    # 计算出口数据的买方集中度（改进的双向计算）
    export_data = df_locking[df_locking['us_role'] == 'exporter'].copy()
    export_locking_results = []
    
    if len(export_data) > 0:
        logger.info("计算美国出口的买方集中度...")
        
        # 按年份和产品计算买方HHI和锁定力
        for year in export_data['year'].unique():
            year_data = export_data[export_data['year'] == year]
            
            for product in year_data['energy_product'].unique():
                product_data = year_data[year_data['energy_product'] == product]
                
                # 计算总出口额
                total_export = product_data['trade_value_usd'].sum()
                
                if total_export > 0:
                    # 计算每个买方的市场份额
                    buyer_shares = product_data.groupby('us_partner')['trade_value_usd'].sum() / total_export
                    
                    # 计算买方HHI (Herfindahl-Hirschman Index)
                    buyer_hhi = (buyer_shares ** 2).sum()
                    
                    # 为每个买方计算市场锁定力
                    for partner, share in buyer_shares.items():
                        buyer_locking_power = buyer_hhi * share
                        
                        export_locking_results.append({
                            'year': year,
                            'us_partner': partner,
                            'energy_product': product,
                            'us_role': 'exporter',
                            'market_locking_power': buyer_locking_power,
                            'market_share': share,
                            'hhi': buyer_hhi,
                            'total_buyers': len(buyer_shares)
                        })
        
        # 转换为DataFrame
        export_locking_df = pd.DataFrame(export_locking_results)
        
        # 合并进口和出口的锁定力数据
        if len(export_locking_df) > 0:
            all_locking = pd.concat([locking_df, export_locking_df], ignore_index=True)
        else:
            # 如果出口锁定力计算失败，回退到原来的简单处理
            export_locking_simple = export_data[['year', 'us_partner', 'energy_product', 'us_role']].copy()
            export_locking_simple['market_locking_power'] = 0
            all_locking = pd.concat([locking_df, export_locking_simple], ignore_index=True)
    else:
        all_locking = locking_df
    
    # 与原数据合并（已经包含完整键值包括us_role）
    df_with_locking = pd.merge(
        df_locking, 
        all_locking[['year', 'us_partner', 'energy_product', 'us_role', 'market_locking_power']], 
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
            'hhi': 'mean',
            'total_suppliers': 'mean'
        }).round(4)
        
        logger.info(f"  按能源产品的市场集中度:")
        for product in product_locking.index:
            stats = product_locking.loc[product]
            logger.info(f"    {product}: 平均HHI={stats[('hhi', 'mean')]:.4f}, " +
                       f"平均锁定力={stats[('market_locking_power', 'mean')]:.4f}, " +
                       f"平均供应商数={stats[('total_suppliers', 'mean')]:.1f}")
    
    logger.info("✅ 市场锁定力指标计算完成!")
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

def generate_dli_panel_data(trade_data: pd.DataFrame = None, 
                           data_file_path: str = None,
                           output_path: str = None) -> pd.DataFrame:
    """
    生成完整的DLI面板数据集
    
    这是DLI计算模块的主要接口函数，整合所有DLI维度计算步骤
    
    Args:
        trade_data: 预处理的美国贸易数据DataFrame，如果为None则从文件加载
        data_file_path: 数据文件路径，默认使用标准路径
        output_path: 输出文件路径，默认保存到outputs目录
        
    Returns:
        包含完整DLI指标的面板数据DataFrame
        
    输出列包括：
        - 基础数据：year, us_partner, energy_product, trade_value_usd, distance_km等
        - DLI四维度：continuity, infrastructure, stability, market_locking_power
        - 综合指标：dli_composite, dli_composite_adjusted
    """
    
    logger.info("🚀 开始生成DLI面板数据...")
    
    # 第1步：加载数据
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
    required_base_columns = ['year', 'us_partner', 'energy_product', 'trade_value_usd', 'distance_km']
    missing_columns = [col for col in required_base_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必需列: {missing_columns}")
    
    # 第2步：依次计算四个DLI维度
    logger.info("计算DLI四个维度指标...")
    
    # 贸易持续性
    df = calculate_continuity(df)
    
    # 基础设施强度
    df = calculate_infrastructure(df)
    
    # 贸易稳定性
    df = calculate_stability(df)
    
    # 市场锁定力
    df = calculate_market_locking_power(df)
    
    # 第3步：计算DLI综合指标
    df = calculate_dli_composite(df)
    
    # 第4步：数据整理和验证
    logger.info("🔧 最终数据整理...")
    
    # 选择需要保存的列
    output_columns = [
        # 基础标识列
        'year', 'us_partner', 'energy_product', 'us_role',
        # 基础数据列
        'trade_value_usd', 'distance_km', 'distance_category',
        # DLI四个维度
        'continuity', 'infrastructure', 'stability', 'market_locking_power',
        # DLI综合指标
        'dli_composite', 'dli_composite_adjusted'
    ]
    
    # 确保所有列都存在
    available_columns = [col for col in output_columns if col in df.columns]
    df_output = df[available_columns].copy()
    
    # 按关键字段排序
    df_output = df_output.sort_values(['year', 'us_partner', 'energy_product', 'us_role'])
    df_output = df_output.reset_index(drop=True)
    
    # 最终数据验证
    logger.info("🔍 最终数据验证:")
    logger.info(f"  总记录数: {len(df_output):,}")
    logger.info(f"  年份范围: {df_output['year'].min()}-{df_output['year'].max()}")
    logger.info(f"  贸易伙伴数: {df_output['us_partner'].nunique()}")
    logger.info(f"  能源产品数: {df_output['energy_product'].nunique()}")
    
    # 检查缺失值
    missing_summary = df_output.isnull().sum()
    if missing_summary.any():
        logger.warning("发现缺失值:")
        for col, count in missing_summary[missing_summary > 0].items():
            logger.warning(f"  {col}: {count} 个缺失值")
    else:
        logger.info("✅ 无缺失值")
    
    # DLI指标统计摘要
    dli_columns = ['continuity', 'infrastructure', 'stability', 'market_locking_power', 'dli_composite_adjusted']
    logger.info(f"📊 DLI指标最终统计摘要:")
    for col in dli_columns:
        if col in df_output.columns:
            stats = df_output[col].describe()
            logger.info(f"  {col}:")
            logger.info(f"    均值±标准差: {stats['mean']:.4f}±{stats['std']:.4f}")
            logger.info(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"    分位数(25%,50%,75%): {stats['25%']:.4f}, {stats['50%']:.4f}, {stats['75%']:.4f}")
    
    # 第5步：导出数据
    if output_path is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "outputs" / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dli_panel_data.csv"
    
    df_output.to_csv(output_path, index=False)
    logger.info(f"💾 DLI面板数据已保存至: {output_path}")
    
    # 同时保存权重信息到json文件
    if hasattr(df, 'attrs') and 'dli_weights' in df.attrs:
        import json
        weights_path = Path(output_path).parent / "dli_weights_and_params.json"
        
        weights_info = {
            'dli_weights': df.attrs['dli_weights'],
            'pca_explained_variance': df.attrs.get('pca_explained_variance'),
            'standardization_params': df.attrs.get('standardization_params'),
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'total_records': len(df_output),
                'year_range': [int(df_output['year'].min()), int(df_output['year'].max())],
                'num_partners': int(df_output['us_partner'].nunique()),
                'num_products': int(df_output['energy_product'].nunique())
            }
        }
        
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_info, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 DLI权重信息已保存至: {weights_path}")
    
    logger.info("✅ DLI面板数据生成完成!")
    return df_output

if __name__ == "__main__":
    # 测试DLI计算功能
    try:
        dli_panel = generate_dli_panel_data()
        print(f"✅ DLI面板数据生成成功!")
        print(f"📊 数据维度: {dli_panel.shape}")
        print(f"🔗 DLI综合指标范围: [{dli_panel['dli_composite_adjusted'].min():.4f}, {dli_panel['dli_composite_adjusted'].max():.4f}]")
        
    except Exception as e:
        logger.error(f"❌ DLI计算失败: {e}")
        raise