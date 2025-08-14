#!/usr/bin/env python3
"""
权威国家/地区代码验证模块

功能: 
- 创建权威的ISO 3166-1 alpha-3国家代码白名单
- 验证贸易数据中的国家代码是否为真实的主权独立实体
- 剔除区域性汇总实体 (如 "North America, nes", "Africa, nes", "EU-27" 等)

确保数据源的纯净性，避免重复计算和节点数量虚高问题。
"""

import pandas as pd
from typing import Set, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_valid_country_codes() -> Set[str]:
    """
    获取权威的ISO 3166-1 alpha-3国家代码集合
    
    基于联合国统计司标准，包含真实的主权独立国家和我们认可的地区实体
    剔除所有区域性、概念性汇总实体
    
    Returns:
        Set[str]: 权威的ISO3代码集合
    """
    
    # 首先尝试使用pycountry库获取标准ISO国家代码
    valid_codes = set()
    
    try:
        import pycountry
        
        # 获取所有ISO 3166-1 alpha-3国家代码
        for country in pycountry.countries:
            if hasattr(country, 'alpha_3'):
                valid_codes.add(country.alpha_3)
        
        logger.info(f"使用pycountry库获取了 {len(valid_codes)} 个标准ISO国家代码")
        
    except ImportError:
        logger.warning("pycountry库未安装，使用硬编码的权威国家代码列表")
        
        # 硬编码的权威ISO 3166-1 alpha-3国家代码列表
        # 基于联合国统计司M49标准
        valid_codes = {
            'AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATA', 'ATG',
            'ARG', 'ARM', 'ABW', 'AUS', 'AUT', 'AZE', 'BHS', 'BHR', 'BGD',
            'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BES',
            'BIH', 'BWA', 'BVT', 'BRA', 'IOT', 'BRN', 'BGR', 'BFA', 'BDI',
            'CPV', 'KHM', 'CMR', 'CAN', 'CYM', 'CAF', 'TCD', 'CHL', 'CHN',
            'CXR', 'CCK', 'COL', 'COM', 'COG', 'COD', 'COK', 'CRI', 'CIV',
            'HRV', 'CUB', 'CUW', 'CYP', 'CZE', 'DNK', 'DJI', 'DMA', 'DOM',
            'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 'ETH', 'FLK',
            'FRO', 'FJI', 'FIN', 'FRA', 'GUF', 'PYF', 'ATF', 'GAB', 'GMB',
            'GEO', 'DEU', 'GHA', 'GIB', 'GRC', 'GRL', 'GRD', 'GLP', 'GUM',
            'GTM', 'GGY', 'GIN', 'GNB', 'GUY', 'HTI', 'HMD', 'VAT', 'HND',
            'HKG', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN',
            'ISR', 'ITA', 'JAM', 'JPN', 'JEY', 'JOR', 'KAZ', 'KEN', 'KIR',
            'PRK', 'KOR', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR',
            'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG', 'MWI', 'MYS', 'MDV',
            'MLI', 'MLT', 'MHL', 'MTQ', 'MRT', 'MUS', 'MYT', 'MEX', 'FSM',
            'MDA', 'MCO', 'MNG', 'MNE', 'MSR', 'MAR', 'MOZ', 'MMR', 'NAM',
            'NRU', 'NPL', 'NLD', 'NCL', 'NZL', 'NIC', 'NER', 'NGA', 'NIU',
            'NFK', 'MKD', 'MNP', 'NOR', 'OMN', 'PAK', 'PLW', 'PSE', 'PAN',
            'PNG', 'PRY', 'PER', 'PHL', 'PCN', 'POL', 'PRT', 'PRI', 'QAT',
            'REU', 'ROU', 'RUS', 'RWA', 'BLM', 'SHN', 'KNA', 'LCA', 'MAF',
            'SPM', 'VCT', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC',
            'SLE', 'SGP', 'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SGS',
            'SSD', 'ESP', 'LKA', 'SDN', 'SUR', 'SJM', 'SWE', 'CHE', 'SYR',
            'TWN', 'TJK', 'TZA', 'THA', 'TLS', 'TGO', 'TKL', 'TON', 'TTO',
            'TUN', 'TUR', 'TKM', 'TCA', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR',
            'USA', 'UMI', 'URY', 'UZB', 'VUT', 'VEN', 'VNM', 'VGB', 'VIR',
            'WLF', 'ESH', 'YEM', 'ZMB', 'ZWE'
        }
        
        logger.info(f"使用硬编码国家代码列表，共 {len(valid_codes)} 个代码")
    
    # 手动添加一些重要的非主权实体（如有必要）
    additional_valid_codes = {
        'TWN',  # Taiwan (重要贸易实体)
        'HKG',  # Hong Kong (重要金融中心)
        'MAC',  # Macao (特别行政区)
        'PSE',  # Palestine (UN观察员)
    }
    
    valid_codes.update(additional_valid_codes)
    
    # 定义需要明确排除的区域性汇总实体
    # 这些通常在UN Comtrade中出现但不是真实的国家/地区
    regional_aggregates_to_exclude = {
        # 地理区域汇总
        'AFR',  # Africa, nes
        'ASI',  # Asia, nes  
        'EUR',  # Europe, nes
        'NAM',  # North America, nes (注意与NAM纳米比亚区分)
        'SAM',  # South America, nes
        'OCE',  # Oceania, nes
        
        # 经济区域汇总
        'EU27', # European Union (27 countries)
        'EU28', # European Union (28 countries)
        'EU15', # European Union (15 countries)
        'ASEAN', # ASEAN
        'MERCOSUR', # Mercosur
        'NAFTA', # NAFTA
        'USMCA', # USMCA
        
        # 其他概念性实体
        'LDC',  # Least Developed Countries
        'LLDC', # Landlocked Developing Countries
        'SIDS', # Small Island Developing States
        'OTH',  # Other
        'UNK',  # Unknown
        'WLD',  # World
        'XXX',  # Unspecified
        
        # 可能的其他模糊实体
        'NES',  # Not elsewhere specified
        'OAS',  # Other Asia, nes
        'OSA',  # Other South America, nes
        'ONA',  # Other North America, nes
        'OEU',  # Other Europe, nes
        'OAF',  # Other Africa, nes
        'OOC',  # Other Oceania, nes
    }
    
    # 确保不包含任何区域性汇总实体
    valid_codes = valid_codes - regional_aggregates_to_exclude
    
    logger.info(f"最终权威国家代码集合包含 {len(valid_codes)} 个实体")
    
    return valid_codes

def is_valid_country_code(code: str) -> bool:
    """
    验证ISO3代码是否为权威认可的国家/地区实体
    
    Args:
        code: ISO 3166-1 alpha-3 国家代码 (如 'USA', 'CHN')
        
    Returns:
        bool: 如果是权威认可的国家/地区实体则返回True，否则False
    """
    if not isinstance(code, str):
        return False
    
    # 标准化处理（转大写，去除空白）
    code = code.strip().upper()
    
    if len(code) != 3:
        return False
    
    # 获取权威代码集合（使用缓存避免重复计算）
    if not hasattr(is_valid_country_code, '_valid_codes_cache'):
        is_valid_country_code._valid_codes_cache = get_valid_country_codes()
    
    return code in is_valid_country_code._valid_codes_cache

def filter_valid_trade_data(df: pd.DataFrame, 
                          reporter_col: str = 'reporterISO', 
                          partner_col: str = 'partnerISO') -> pd.DataFrame:
    """
    过滤贸易数据，只保留权威认可的国家/地区之间的贸易记录
    
    Args:
        df: 包含贸易数据的DataFrame
        reporter_col: 报告国ISO代码列名
        partner_col: 合作伙伴ISO代码列名
        
    Returns:
        pd.DataFrame: 过滤后的干净贸易数据
    """
    logger.info(f"开始过滤贸易数据，原始数据有 {len(df):,} 条记录")
    
    # 检查必要列是否存在
    if reporter_col not in df.columns or partner_col not in df.columns:
        raise ValueError(f"数据中缺少必要列: {reporter_col} 或 {partner_col}")
    
    original_count = len(df)
    unique_reporters_before = df[reporter_col].nunique()
    unique_partners_before = df[partner_col].nunique()
    
    logger.info(f"过滤前统计:")
    logger.info(f"  - 总记录数: {original_count:,}")
    logger.info(f"  - 唯一报告国数量: {unique_reporters_before}")
    logger.info(f"  - 唯一合作伙伴数量: {unique_partners_before}")
    
    # 应用双重过滤：报告国和合作伙伴都必须是权威认可的实体
    mask_reporter = df[reporter_col].apply(is_valid_country_code)
    mask_partner = df[partner_col].apply(is_valid_country_code)
    mask_both_valid = mask_reporter & mask_partner
    
    df_filtered = df[mask_both_valid].copy()
    
    filtered_count = len(df_filtered)
    unique_reporters_after = df_filtered[reporter_col].nunique()
    unique_partners_after = df_filtered[partner_col].nunique()
    
    logger.info(f"过滤后统计:")
    logger.info(f"  - 总记录数: {filtered_count:,} (保留 {filtered_count/original_count:.1%})")
    logger.info(f"  - 唯一报告国数量: {unique_reporters_after} (减少 {unique_reporters_before - unique_reporters_after})")
    logger.info(f"  - 唯一合作伙伴数量: {unique_partners_after} (减少 {unique_partners_before - unique_partners_after})")
    
    # 报告被过滤掉的实体示例
    invalid_reporters = set(df[~mask_reporter][reporter_col].unique())
    invalid_partners = set(df[~mask_partner][partner_col].unique())
    invalid_entities = invalid_reporters | invalid_partners
    
    if invalid_entities:
        logger.info(f"被过滤掉的区域性汇总实体示例: {list(invalid_entities)[:10]}")
    
    return df_filtered

def get_data_quality_report(df: pd.DataFrame, 
                          reporter_col: str = 'reporterISO', 
                          partner_col: str = 'partnerISO') -> dict:
    """
    生成数据质量报告，分析当前数据中的实体构成
    
    Args:
        df: 贸易数据DataFrame
        reporter_col: 报告国ISO代码列名
        partner_col: 合作伙伴ISO代码列名
        
    Returns:
        dict: 数据质量分析报告
    """
    all_entities = set(df[reporter_col].unique()) | set(df[partner_col].unique())
    
    valid_entities = {entity for entity in all_entities if is_valid_country_code(entity)}
    invalid_entities = all_entities - valid_entities
    
    report = {
        'total_entities': len(all_entities),
        'valid_entities_count': len(valid_entities),
        'invalid_entities_count': len(invalid_entities),
        'valid_ratio': len(valid_entities) / len(all_entities) if all_entities else 0,
        'invalid_entities_list': sorted(invalid_entities),
        'total_records': len(df),
        'records_with_invalid_reporter': len(df[~df[reporter_col].apply(is_valid_country_code)]),
        'records_with_invalid_partner': len(df[~df[partner_col].apply(is_valid_country_code)])
    }
    
    return report

if __name__ == "__main__":
    # 测试验证功能
    print("=== 国家代码验证模块测试 ===")
    
    # 测试有效代码
    valid_codes = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'TWN', 'HKG']
    print(f"测试有效代码: {valid_codes}")
    for code in valid_codes:
        result = is_valid_country_code(code)
        print(f"  {code}: {'✅' if result else '❌'}")
    
    # 测试无效代码（区域性汇总实体）
    invalid_codes = ['EU27', 'AFR', 'ASI', 'NAM', 'NES', 'OTH', 'WLD']
    print(f"\n测试无效代码: {invalid_codes}")
    for code in invalid_codes:
        result = is_valid_country_code(code)
        print(f"  {code}: {'❌' if not result else '⚠️'}")
    
    # 获取权威代码统计
    valid_country_codes = get_valid_country_codes()
    print(f"\n权威国家代码集合统计: {len(valid_country_codes)} 个实体")
    print(f"前20个代码示例: {sorted(list(valid_country_codes))[:20]}")