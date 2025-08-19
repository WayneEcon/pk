#!/usr/bin/env python3
"""
05模块标准数据接口 (DLI Data Interface)
===================================

为其他模块提供标准化的DLI数据访问接口
"""

import pandas as pd
from typing import List, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_dli_panel_data(countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
    """
    标准DLI面板数据接口
    
    Args:
        countries: 指定国家列表，None则返回所有国家
        years: 指定年份列表，None则返回所有年份
        
    Returns:
        pd.DataFrame: DLI面板数据
    """
    module_dir = Path(__file__).parent
    
    # 尝试多个可能的文件位置
    dli_files = [
        module_dir / "dli_panel_data.csv",
        module_dir / "outputs" / "dli_panel_data.csv",
        module_dir / "outputs" / "dli_results.csv",
        module_dir / "dli_panel_data_v2.csv"  # 兼容旧版本
    ]
    
    df = None
    for file in dli_files:
        if file.exists():
            try:
                df = pd.read_csv(file)
                logger.info(f"✅ 加载DLI数据: {file}")
                break
            except Exception as e:
                logger.warning(f"⚠️ 无法加载DLI文件 {file}: {e}")
                continue
    
    if df is None:
        logger.error("❌ 未找到DLI数据文件")
        return pd.DataFrame()
    
    # 标准化列名
    column_mapping = {
        'us_partner': 'country',
        'country_code': 'country',
        'dli_score_adjusted': 'dli_score'
    }
    df = df.rename(columns=column_mapping)
    
    # 确保有必需的列
    required_columns = ['year', 'country', 'dli_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"❌ DLI数据缺少必需列: {missing_columns}")
        return pd.DataFrame()
    
    # 数据过滤
    if countries is not None:
        df = df[df['country'].isin(countries)]
    
    if years is not None:
        df = df[df['year'].isin(years)]
    
    return df[required_columns].drop_duplicates()

def get_dli_summary() -> Dict:
    """获取DLI数据概览"""
    df = get_dli_panel_data()
    
    if df.empty:
        return {}
    
    return {
        'total_observations': len(df),
        'unique_countries': df['country'].nunique(),
        'unique_years': df['year'].nunique(),
        'year_range': [df['year'].min(), df['year'].max()],
        'countries': sorted(df['country'].unique()),
        'dli_stats': {
            'mean': df['dli_score'].mean(),
            'std': df['dli_score'].std(),
            'min': df['dli_score'].min(),
            'max': df['dli_score'].max()
        }
    }

def get_available_countries() -> List[str]:
    """获取可用的国家列表"""
    df = get_dli_panel_data()
    return sorted(df['country'].unique()) if not df.empty else []

def get_available_years() -> List[int]:
    """获取可用的年份列表"""
    df = get_dli_panel_data()
    return sorted(df['year'].unique()) if not df.empty else []

def get_country_dli_timeseries(country: str) -> pd.DataFrame:
    """获取特定国家的DLI时间序列"""
    df = get_dli_panel_data(countries=[country])
    return df.sort_values('year') if not df.empty else pd.DataFrame()

def validate_dli_data() -> Dict[str, bool]:
    """验证DLI数据的完整性和一致性"""
    df = get_dli_panel_data()
    
    validation = {
        'has_data': not df.empty,
        'has_usa_data': False,
        'balanced_panel': False,
        'reasonable_values': False,
        'no_missing_values': False
    }
    
    if not df.empty:
        # 检查是否有美国数据
        validation['has_usa_data'] = 'USA' in df['country'].values
        
        # 检查是否为平衡面板
        country_year_counts = df.groupby('country')['year'].nunique()
        if len(country_year_counts) > 0:
            validation['balanced_panel'] = country_year_counts.std() == 0
        
        # 检查DLI值是否合理(0-1之间)
        validation['reasonable_values'] = (
            df['dli_score'].between(0, 1).all() and
            df['dli_score'].notna().all()
        )
        
        # 检查是否有缺失值
        validation['no_missing_values'] = not df[['year', 'country', 'dli_score']].isnull().any().any()
    
    return validation