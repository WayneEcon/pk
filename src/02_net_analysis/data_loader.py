#!/usr/bin/env python3
"""
数据加载模块
负责加载年度数据和基础验证
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from .utils import setup_path, validate_dataframe_columns, log_dataframe_info, REQUIRED_COLUMNS, PROCESSED_DATA_DIR, FILE_TEMPLATES

# 确保路径设置
setup_path()

logger = logging.getLogger(__name__)

def load_yearly_data(year: int) -> Optional[pd.DataFrame]:
    """
    加载指定年份的清洗数据，包含数据验证
    
    Args:
        year: 要加载的数据年份
        
    Returns:
        成功时返回DataFrame，失败时返回None
        
    Raises:
        无异常抛出，所有错误通过日志记录并返回None
        
    Example:
        >>> df = load_yearly_data(2020)
        >>> if df is not None:
        ...     print(f"加载了 {len(df)} 条记录")
    """
    file_path = PROCESSED_DATA_DIR / FILE_TEMPLATES['cleaned_data'].format(year=year)
    
    if not file_path.exists():
        logger.warning(f"数据文件不存在: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # 使用工具函数进行数据验证
        try:
            validate_dataframe_columns(df, REQUIRED_COLUMNS.keys(), f"{year}年数据")
        except ValueError as e:
            logger.error(f"{year}: {e}")
            return None
        
        # 基础数据检查
        if len(df) == 0:
            logger.warning(f"{year}: 数据文件为空")
            return None
        
        # 数据质量统计
        total_records = len(df)
        non_zero_trade = (df['trade_value_raw_usd'] > 0).sum()
        zero_trade_rate = (total_records - non_zero_trade) / total_records if total_records > 0 else 0
        
        # 使用工具函数记录信息
        log_dataframe_info(df, f"原始数据(非零贸易: {non_zero_trade:,}, 零值率: {zero_trade_rate:.1%})", year, logger)
        
        return df
        
    except FileNotFoundError:
        logger.error(f"❌ {year}: 文件不存在 - {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"❌ {year}: 文件为空")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"❌ {year}: 文件格式错误 - {e}")
        return None
    except Exception as e:
        logger.error(f"❌ {year}: 意外错误 - {e}")
        return None