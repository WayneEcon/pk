#!/usr/bin/env python3
"""
统一国家编码标准化模块 (Unified Country Code Standardizer)
==========================================================

从08模块的simple_gas_ovi_builder.py提取并增强的国家名称→ISO代码标准化系统
用于解决跨模块数据整合中的国家编码不一致问题

作者：Energy Network Analysis Team  
版本：v1.0 - 统一标准化系统
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Set
from pathlib import Path

logger = logging.getLogger(__name__)

class CountryStandardizer:
    """
    统一的国家编码标准化器
    基于08模块的成熟映射系统，用于整个项目的国家名称标准化
    """
    
    def __init__(self):
        """初始化标准化器，加载完整的国家映射表"""
        
        # 完整的国家名称→ISO代码映射表（从simple_gas_ovi_builder.py提取）
        self.country_mapping = {
            # 北美洲
            'United States': 'USA', 'US': 'USA', 'United States of America': 'USA',
            'Canada': 'CAN', 'Mexico': 'MEX',
            
            # 欧洲主要国家
            'Germany': 'DEU', 'Federal Republic of Germany': 'DEU',
            'France': 'FRA', 'Italy': 'ITA', 'Spain': 'ESP',
            'United Kingdom': 'GBR', 'UK': 'GBR', 'Great Britain': 'GBR',
            'Netherlands': 'NLD', 'Belgium': 'BEL', 'Austria': 'AUT',
            'Switzerland': 'CHE', 'Norway': 'NOR', 'Sweden': 'SWE',
            'Denmark': 'DNK', 'Finland': 'FIN', 'Poland': 'POL',
            'Portugal': 'PRT', 'Greece': 'GRC', 'Ireland': 'IRL',
            'Iceland': 'ISL', 'Luxembourg': 'LUX', 'Cyprus': 'CYP', 'Malta': 'MLT',
            
            # 东欧和巴尔干
            'Czech Republic': 'CZE', 'Hungary': 'HUN', 'Slovakia': 'SVK',
            'Romania': 'ROU', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
            'Slovenia': 'SVN', 'Estonia': 'EST', 'Latvia': 'LVA',
            'Lithuania': 'LTU', 'Serbia': 'SRB', 'North Macedonia': 'MKD',
            'Moldova': 'MDA', 'Georgia': 'GEO', 'Armenia': 'ARM',
            
            # 亚洲主要国家
            'China': 'CHN', "China, People's Republic of": 'CHN',
            'China Hong Kong SAR': 'HKG', 'Hong Kong SAR, China': 'HKG', 'Hong Kong': 'HKG',
            'Japan': 'JPN', 'South Korea': 'KOR', "Korea, Rep.": 'KOR',
            'Korea, Republic of': 'KOR', 'Taiwan': 'TWN',
            'India': 'IND', 'Indonesia': 'IDN', 'Malaysia': 'MYS',
            'Singapore': 'SGP', 'Thailand': 'THA', 'Philippines': 'PHL',
            'Vietnam': 'VNM', 'Bangladesh': 'BGD', 'Pakistan': 'PAK',
            'Sri Lanka': 'LKA', 'Myanmar': 'MMR', 'Cambodia': 'KHM',
            
            # 前苏联国家
            'Russian Federation': 'RUS', 'Russia': 'RUS',
            'Ukraine': 'UKR', 'Kazakhstan': 'KAZ', 'Belarus': 'BLR',
            'Uzbekistan': 'UZB', 'Azerbaijan': 'AZE', 'Turkmenistan': 'TKM',
            'Kyrgyzstan': 'KGZ', 'Tajikistan': 'TJK', 'USSR': 'USSR',
            
            # 中东
            'Saudi Arabia': 'SAU', 'Iran': 'IRN', 'Islamic Republic of Iran': 'IRN',
            'Iraq': 'IRQ', 'Kuwait': 'KWT', 'United Arab Emirates': 'ARE',
            'Qatar': 'QAT', 'Oman': 'OMN', 'Bahrain': 'BHR', 'Israel': 'ISR',
            'Turkey': 'TUR', 'Turkiye': 'TUR', 'Türkiye': 'TUR',
            'Jordan': 'JOR', 'Lebanon': 'LBN',
            
            # 非洲
            'Nigeria': 'NGA', 'South Africa': 'ZAF', 'Algeria': 'DZA',
            'Egypt': 'EGY', 'Libya': 'LBY', 'Morocco': 'MAR',
            'Angola': 'AGO', 'Ghana': 'GHA', 'Kenya': 'KEN', 'Tunisia': 'TUN',
            'Sudan': 'SDN', "Côte d'Ivoire": 'CIV',
            
            # 大洋洲
            'Australia': 'AUS', 'New Zealand': 'NZL',
            
            # 南美洲
            'Brazil': 'BRA', 'Argentina': 'ARG', 'Chile': 'CHL',
            'Colombia': 'COL', 'Peru': 'PER', 'Venezuela': 'VEN',
            'Ecuador': 'ECU', 'Uruguay': 'URY', 'Bolivia': 'BOL',
            'Paraguay': 'PRY', 
            
            # 加勒比海和中美洲
            'Trinidad & Tobago': 'TTO', 'Trinidad and Tobago': 'TTO',
            'Dominican Republic': 'DOM', 'Panama': 'PAN', 'Jamaica': 'JAM',
            'El Salvador': 'SLV',
            
            # 其他
            'Gibraltar': 'GIB',
        }
        
        # 过滤关键词（用于排除非国家实体）
        self.filter_keywords = [
            'total', 'other', 'excludes', 'includes', '*', '#', 
            'world', 'global', 'unspecified', 'not specified',
            'bunkers', 'statistical difference', 'memo:'
        ]
        
        logger.info(f"🌍 国家标准化器初始化完成")
        logger.info(f"   支持国家映射: {len(self.country_mapping)} 个")
        
    def standardize_country_name(self, country: str) -> Optional[str]:
        """
        标准化国家名称 - 智能映射系统
        
        Args:
            country: 原始国家名称
            
        Returns:
            标准化的ISO 3位代码，如果无法标准化则返回None
        """
        if pd.isna(country):
            return None
        
        country_str = str(country).strip()
        
        # 过滤非国家实体
        if any(keyword in country_str.lower() for keyword in self.filter_keywords):
            return None
        
        # 首先检查是否已经是有效的ISO代码（3位字母）
        if len(country_str) == 3 and country_str.isupper() and country_str.isalpha():
            # 已经是ISO代码格式，直接返回
            logger.debug(f"国家代码 '{country_str}' 已是标准ISO格式")
            return country_str
        
        # 精确匹配
        if country_str in self.country_mapping:
            return self.country_mapping[country_str]
        
        # 模糊匹配（智能备选方案）
        country_lower = country_str.lower()
        
        # 中国相关
        if 'china' in country_lower and 'hong kong' not in country_lower and 'taiwan' not in country_lower:
            return 'CHN'
        if 'hong kong' in country_lower:
            return 'HKG'
        if 'taiwan' in country_lower:
            return 'TWN'
            
        # 韩国相关
        if 'korea' in country_lower and ('south' in country_lower or 'republic' in country_lower):
            return 'KOR'
            
        # 俄国相关
        if 'russia' in country_lower or 'russian' in country_lower:
            return 'RUS'
            
        # 德国相关
        if 'germany' in country_lower:
            return 'DEU'
            
        # 英国相关
        if ('united kingdom' in country_lower or country_lower == 'uk' or 
            'great britain' in country_lower or country_lower == 'britain'):
            return 'GBR'
            
        # 美国相关
        if 'united states' in country_lower or country_lower == 'us':
            return 'USA'
            
        # 伊朗相关
        if 'iran' in country_lower:
            return 'IRN'
            
        # 其他常见模糊匹配
        if 'netherlands' in country_lower:
            return 'NLD'
        if 'switzerland' in country_lower:
            return 'CHE'
            
        # 无法标准化
        logger.debug(f"国家名称 '{country_str}' 无法标准化")
        return None
    
    def standardize_dataframe(self, df: pd.DataFrame, country_column: str, 
                            new_column_name: str = 'country') -> pd.DataFrame:
        """
        标准化DataFrame中的国家列
        
        Args:
            df: 包含国家名称的DataFrame
            country_column: 原始国家名称列名
            new_column_name: 新的标准化列名
            
        Returns:
            添加了标准化国家代码列的DataFrame
        """
        if country_column not in df.columns:
            logger.error(f"列 '{country_column}' 不存在于DataFrame中")
            return df
        
        logger.info(f"🔧 标准化 '{country_column}' 列...")
        
        # 应用标准化
        df_result = df.copy()
        df_result[new_column_name] = df_result[country_column].apply(self.standardize_country_name)
        
        # 统计结果
        original_count = len(df)
        standardized_count = df_result[new_column_name].notna().sum()
        unique_countries = df_result[new_column_name].nunique()
        
        logger.info(f"✅ 标准化完成:")
        logger.info(f"   原始记录: {original_count}")
        logger.info(f"   成功标准化: {standardized_count} ({standardized_count/original_count:.1%})")
        logger.info(f"   唯一国家: {unique_countries}")
        
        # 显示无法标准化的国家（便于调试）
        failed_countries = df_result[df_result[new_column_name].isna()][country_column].unique()
        if len(failed_countries) > 0:
            logger.warning(f"⚠️ 无法标准化的国家 ({len(failed_countries)}个): {list(failed_countries)[:10]}")
        
        return df_result
    
    def get_country_mapping(self) -> Dict[str, str]:
        """返回完整的国家映射字典"""
        return self.country_mapping.copy()
    
    def add_custom_mapping(self, custom_mapping: Dict[str, str]) -> None:
        """
        添加自定义国家映射
        
        Args:
            custom_mapping: 自定义的国家名称→ISO代码映射
        """
        self.country_mapping.update(custom_mapping)
        logger.info(f"📝 添加自定义映射: {len(custom_mapping)} 个")

# 便捷函数
def standardize_country(country: str) -> Optional[str]:
    """便捷函数：标准化单个国家名称"""
    standardizer = CountryStandardizer()
    return standardizer.standardize_country_name(country)

def standardize_country_dataframe(df: pd.DataFrame, country_column: str) -> pd.DataFrame:
    """便捷函数：标准化DataFrame中的国家列"""
    standardizer = CountryStandardizer()
    return standardizer.standardize_dataframe(df, country_column)