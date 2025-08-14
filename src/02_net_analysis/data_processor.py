#!/usr/bin/env python3
"""
数据处理模块
负责数据一致性处理和贸易流聚合
"""

import pandas as pd
import logging
from pathlib import Path
from .utils import setup_path, validate_dataframe_columns, log_dataframe_info, DataQualityReporter

# 确保路径设置
setup_path()

logger = logging.getLogger(__name__)

def resolve_trade_data_consistency(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    解决双边贸易数据不一致问题，实施"优先进口数据"策略
    
    Args:
        df: 包含进出口数据的DataFrame，必须包含'flow'列
        year: 数据年份，用于日志记录
        
    Returns:
        一致化处理后的DataFrame，包含source, target等标准字段
        
    Raises:
        ValueError: 当输入数据缺少必要字段时
        
    Example:
        >>> df = pd.DataFrame({...})  # 包含flow列的原始数据
        >>> consistent_data = resolve_trade_data_consistency(df, 2020)
    """
    logger.info(f"     {year}: 开始数据一致性处理...")
    
    # 数据验证
    required_cols = ['flow', 'reporter', 'partner', 'trade_value_raw_usd', 'reporter_name', 'partner_name', 'year']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"数据缺少必要字段: {missing_cols}")
    
    # 分离进口和出口数据
    imports = df[df['flow'] == 'M'].copy()
    exports = df[df['flow'] == 'X'].copy()
    
    logger.debug(f"     {year}: 原始数据 - 进口: {len(imports):,}, 出口: {len(exports):,}")
    
    if len(imports) == 0 and len(exports) == 0:
        logger.warning(f"     {year}: 没有有效的贸易数据")
        return pd.DataFrame()
    
    # 使用vectorized操作创建标准化的贸易流标识和字段
    consistent_flows = []
    
    # 1. 处理进口数据（vectorized操作）
    if len(imports) > 0:
        import_data = imports.copy()
        import_data['trade_flow_id'] = import_data['partner'] + '_to_' + import_data['reporter']
        import_data['source'] = import_data['partner']
        import_data['target'] = import_data['reporter']
        import_data['source_name'] = import_data['partner_name']
        import_data['target_name'] = import_data['reporter_name']
        import_data['data_source'] = 'import_reported'
        
        # 选择需要的字段并转为字典列表（避免逐行循环）
        import_records = import_data[['source', 'target', 'trade_value_raw_usd', 'source_name', 
                                   'target_name', 'year', 'data_source', 'trade_flow_id']].to_dict('records')
        consistent_flows.extend(import_records)
    
    # 2. 处理出口数据作为补充（vectorized操作）
    if len(exports) > 0:
        export_data = exports.copy()
        export_data['trade_flow_id'] = export_data['reporter'] + '_to_' + export_data['partner']
        export_data['source'] = export_data['reporter']
        export_data['target'] = export_data['partner']
        export_data['source_name'] = export_data['reporter_name']
        export_data['target_name'] = export_data['partner_name']
        export_data['data_source'] = 'export_mirrored'
        
        # 只添加进口数据中没有的贸易流
        if len(imports) > 0:
            import_flow_ids = set(imports['partner'] + '_to_' + imports['reporter'])
            export_supplement = export_data[~export_data['trade_flow_id'].isin(import_flow_ids)]
        else:
            export_supplement = export_data
            
        export_records = export_supplement[['source', 'target', 'trade_value_raw_usd', 'source_name',
                                          'target_name', 'year', 'data_source', 'trade_flow_id']].to_dict('records')
        consistent_flows.extend(export_records)
    
    # 3. 生成最终的一致化数据集
    if not consistent_flows:
        logger.warning(f"     {year}: 没有生成任何一致化贸易流")
        return pd.DataFrame()
        
    consistent_df = pd.DataFrame(consistent_flows)
    
    # 统计信息
    import_based = len(consistent_df[consistent_df['data_source'] == 'import_reported'])
    export_based = len(consistent_df[consistent_df['data_source'] == 'export_mirrored'])
    
    logger.info(f"     {year}: 一致性处理完成")
    logger.info(f"     {year}: 基于进口数据: {import_based:,} 条 ({import_based/len(consistent_df):.1%})")
    logger.info(f"     {year}: 基于出口镜像: {export_based:,} 条 ({export_based/len(consistent_df):.1%})")
    
    return consistent_df

def aggregate_trade_flows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    聚合贸易流：将同一国家对的多种能源产品贸易额合并
    
    Args:
        df: 一致化处理后的DataFrame，必须包含source, target等字段
        year: 数据年份，用于日志记录
        
    Returns:
        聚合后的DataFrame，每个国家对只有一条记录
        
    Raises:
        ValueError: 当输入数据为空或缺少必要字段时
        
    Example:
        >>> consistent_df = resolve_trade_data_consistency(raw_df, 2020)
        >>> aggregated_df = aggregate_trade_flows(consistent_df, 2020)
    """
    logger.info(f"     {year}: 开始贸易流聚合...")
    
    # 数据验证
    if df.empty:
        logger.warning(f"     {year}: 输入数据为空，跳过聚合")
        return pd.DataFrame()
        
    required_cols = ['source', 'target', 'trade_value_raw_usd', 'source_name', 'target_name', 'year', 'data_source']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"数据缺少必要字段: {missing_cols}")
    
    original_count = len(df)
    
    # 按贸易对聚合（优化：使用更高效的聚合操作）
    aggregated = df.groupby(['source', 'target'], as_index=False).agg({
        'trade_value_raw_usd': 'sum',
        'source_name': 'first',
        'target_name': 'first', 
        'year': 'first',
        'data_source': lambda x: list(x.unique())
    })
    
    # 使用vectorized操作确定主要数据源
    aggregated['primary_data_source'] = aggregated['data_source'].apply(
        lambda sources: 'mixed' if len(sources) > 1 else sources[0]
    )
    
    # 移除临时的data_source列表
    aggregated = aggregated.drop('data_source', axis=1)
    
    # 数据质量统计
    aggregated_count = len(aggregated)
    reduction_rate = (original_count - aggregated_count) / original_count if original_count > 0 else 0
    
    logger.info(f"     {year}: 聚合完成: {original_count:,} -> {aggregated_count:,} 条贸易流 (压缩率: {reduction_rate:.1%})")
    
    return aggregated