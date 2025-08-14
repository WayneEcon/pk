#!/usr/bin/env python3
"""
数据准备模块 (Data Preparation Module)
=====================================

本模块负责为DLI分析准备所需的数据：
1. 加载美国相关的能源贸易数据（作为进口国或出口国）
2. 补充地理距离数据
3. 数据清洗和标准化处理

作者：Energy Network Analysis Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 能源产品代码映射
ENERGY_PRODUCTS = {
    '2701': 'Coal',
    '2709': 'Crude_Oil', 
    '2710': 'Refined_Oil',
    '2711': 'Natural_Gas'
}

def load_country_distances() -> Dict[str, float]:
    """
    加载完整的国家距离数据
    
    使用CEPII数据源的精确距离数据，基于人口加权中心距离计算
    
    Returns:
        国家代码到美国距离的字典（单位：公里）
    """
    import json
    
    # 尝试加载完整的距离数据
    distance_file = Path(__file__).parent / "complete_us_distances_cepii.json"
    
    if distance_file.exists():
        try:
            with open(distance_file, 'r', encoding='utf-8') as f:
                distances = json.load(f)
            logger.info(f"成功加载完整距离数据: {len(distances)} 个国家")
            return distances
        except Exception as e:
            logger.warning(f"加载完整距离数据失败: {e}，使用备份数据")
    
    # 备份距离数据（基于首都到华盛顿DC的距离，单位：公里）
    backup_distances = {
        'CAN': 735,     # 渥太华-华盛顿
        'MEX': 1887,    # 墨西哥城-华盛顿  
        'SAU': 11140,   # 利雅得-华盛顿
        'QAT': 11235,   # 多哈-华盛顿
        'VEN': 3367,    # 加拉加斯-华盛顿
        'NOR': 6120,    # 奥斯陆-华盛顿
        'GBR': 5900,    # 伦敦-华盛顿
        'CHN': 11172,   # 北京-华盛顿
        'RUS': 7816,    # 莫斯科-华盛顿
        'ARE': 11575,   # 阿布扎比-华盛顿
        'IND': 12342,   # 新德里-华盛顿
        'JPN': 10906,   # 东京-华盛顿
        'KOR': 11014,   # 首尔-华盛顿
        'BRA': 6834,    # 巴西利亚-华盛顿
        'ARG': 8531,    # 布宜诺斯艾利斯-华盛顿
        'COL': 3593,    # 波哥大-华盛顿
        'ECU': 4406,    # 基多-华盛顿
        'TTO': 3458,    # 西班牙港-华盛顿
        'NLD': 5862,    # 阿姆斯特丹-华盛顿
        'AGO': 10152,   # 罗安达-华盛顿
        'NGA': 9568,    # 阿布贾-华盛顿
        'IRQ': 10327,   # 巴格达-华盛顿
        'IRN': 10856,   # 德黑兰-华盛顿
        'KWT': 10823,   # 科威特城-华盛顿
        'DZA': 7520,    # 阿尔及尔-华盛顿
        'LBY': 8850,    # 的黎波里-华盛顿
        'EGY': 9100,    # 开罗-华盛顿
    }
    
    logger.warning(f"使用备份距离数据: {len(backup_distances)} 个国家")
    return backup_distances

# 全局距离数据（在模块加载时初始化）
COUNTRY_DISTANCES = load_country_distances()

def robust_outlier_treatment(df: pd.DataFrame, column: str, 
                           method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """
    稳健的异常值检测和处理
    
    Args:
        df: 输入DataFrame
        column: 需要处理的列名
        method: 异常值检测方法 ('iqr' 或 'zscore')
        factor: 异常值阈值因子
        
    Returns:
        添加了异常值标记和温莎化处理列的DataFrame
        
    新增列：
        - {column}_is_outlier: 异常值标记
        - {column}_winsorized: 温莎化处理后的值
    """
    
    if column not in df.columns:
        logger.warning(f"列 {column} 不存在，跳过异常值处理")
        return df
    
    df_result = df.copy()
    
    if method == 'iqr':
        Q1 = df_result[column].quantile(0.25)
        Q3 = df_result[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 标记异常值
        df_result[f'{column}_is_outlier'] = (
            (df_result[column] < lower_bound) | 
            (df_result[column] > upper_bound)
        )
        
        # 温莎化处理
        df_result[f'{column}_winsorized'] = df_result[column].clip(lower_bound, upper_bound)
        
        logger.info(f"IQR异常值检测 - {column}: 阈值=[{lower_bound:.2e}, {upper_bound:.2e}]")
        
    elif method == 'zscore':
        mean_val = df_result[column].mean()
        std_val = df_result[column].std()
        z_scores = np.abs((df_result[column] - mean_val) / std_val)
        
        # 标记异常值
        df_result[f'{column}_is_outlier'] = z_scores > factor
        
        # 温莎化处理
        lower_bound = mean_val - factor * std_val
        upper_bound = mean_val + factor * std_val
        df_result[f'{column}_winsorized'] = df_result[column].clip(lower_bound, upper_bound)
        
        logger.info(f"Z-score异常值检测 - {column}: 阈值=±{factor}, 范围=[{lower_bound:.2e}, {upper_bound:.2e}]")
    
    else:
        logger.warning(f"未知的异常值检测方法: {method}")
        return df
    
    outlier_count = df_result[f'{column}_is_outlier'].sum()
    outlier_pct = outlier_count / len(df_result) * 100
    
    logger.info(f"{column}异常值: {outlier_count} 条 ({outlier_pct:.2f}%)")
    
    return df_result

def load_us_trade_data(data_dir: str = None) -> pd.DataFrame:
    """
    加载美国相关的能源贸易数据
    
    Args:
        data_dir: 数据目录路径，默认使用项目标准路径
        
    Returns:
        包含美国作为进口国或出口国的所有能源贸易数据的DataFrame
        
    数据结构：
        - year: 年份
        - reporter: 报告国代码
        - partner: 伙伴国代码  
        - flow: 贸易流向 (M=Import, X=Export)
        - product_code: 能源产品代码
        - product_name: 产品名称
        - trade_value_usd: 贸易值（美元）
        - us_role: 美国角色 ('importer' 或 'exporter')
        - us_partner: 美国的贸易伙伴国
        - energy_product: 标准化的能源产品名称
    """
    
    logger.info("🚀 开始加载美国能源贸易数据...")
    
    # 设置数据路径
    if data_dir is None:
        base_dir = Path(__file__).parent.parent.parent  # 到达energy_network目录
        data_dir = base_dir / "data" / "processed_data"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 加载所有年份的数据
    all_us_trade = []
    years_processed = 0
    
    for year in range(2001, 2025):
        file_path = data_dir / f"cleaned_energy_trade_{year}.csv"
        
        if not file_path.exists():
            logger.warning(f"❌ {year}年数据文件不存在: {file_path}")
            continue
            
        try:
            # 读取年度数据
            df = pd.read_csv(file_path)
            logger.info(f"📂 {year}: 加载了 {len(df):,} 条贸易记录")
            
            # 筛选美国相关的贸易数据
            # 情况1：美国作为报告国（进口商或出口商）
            usa_as_reporter = df[df['reporter'] == 'USA'].copy()
            if len(usa_as_reporter) > 0:
                usa_as_reporter['us_role'] = usa_as_reporter['flow'].map({'M': 'importer', 'X': 'exporter'})
                usa_as_reporter['us_partner'] = usa_as_reporter['partner']
                
            # 情况2：美国作为伙伴国
            usa_as_partner = df[df['partner'] == 'USA'].copy()
            if len(usa_as_partner) > 0:
                # 当USA作为伙伴国时，流向需要反向理解
                usa_as_partner['us_role'] = usa_as_partner['flow'].map({'M': 'exporter', 'X': 'importer'})
                usa_as_partner['us_partner'] = usa_as_partner['reporter']
                # 调整reporter和partner列，使USA始终在reporter位置
                usa_as_partner = usa_as_partner.rename(columns={
                    'reporter': 'temp_partner',
                    'partner': 'reporter'
                })
                usa_as_partner = usa_as_partner.rename(columns={
                    'temp_partner': 'partner'
                })
                # 相应地调整流向
                usa_as_partner['flow'] = usa_as_partner['us_role'].map({'importer': 'M', 'exporter': 'X'})
            
            # 合并两种情况的数据
            year_usa_trade = []
            if len(usa_as_reporter) > 0:
                year_usa_trade.append(usa_as_reporter)
            if len(usa_as_partner) > 0:
                year_usa_trade.append(usa_as_partner)
                
            if year_usa_trade:
                year_df = pd.concat(year_usa_trade, ignore_index=True)
                
                # 重命名贸易值列
                year_df = year_df.rename(columns={'trade_value_raw_usd': 'trade_value_usd'})
                
                # 标准化能源产品名称（确保product_code是字符串类型）
                year_df['product_code'] = year_df['product_code'].astype(str)
                year_df['energy_product'] = year_df['product_code'].map(ENERGY_PRODUCTS)
                
                # 选择需要的列
                columns_to_keep = [
                    'year', 'reporter', 'partner', 'flow', 'product_code', 
                    'product_name', 'trade_value_usd', 'us_role', 'us_partner', 'energy_product'
                ]
                year_df = year_df[columns_to_keep]
                
                all_us_trade.append(year_df)
                years_processed += 1
                logger.info(f"✅ {year}: 提取了 {len(year_df):,} 条美国相关贸易记录")
            else:
                logger.warning(f"⚠️  {year}: 未找到美国相关贸易数据")
                
        except Exception as e:
            logger.error(f"❌ 处理{year}年数据时出错: {e}")
            continue
    
    if not all_us_trade:
        raise ValueError("未找到任何美国相关的贸易数据")
    
    # 合并所有年份数据
    us_trade_df = pd.concat(all_us_trade, ignore_index=True)
    
    # 数据质量检查
    logger.info("🔍 执行数据质量检查...")
    
    # 检查缺失值
    missing_values = us_trade_df.isnull().sum()
    if missing_values.any():
        logger.warning(f"发现缺失值: {missing_values[missing_values > 0].to_dict()}")
    
    # 检查贸易值
    invalid_trade_values = us_trade_df[us_trade_df['trade_value_usd'] <= 0]
    if len(invalid_trade_values) > 0:
        logger.warning(f"发现 {len(invalid_trade_values)} 条无效贸易值记录（<=0），将被移除")
        us_trade_df = us_trade_df[us_trade_df['trade_value_usd'] > 0]
    
    # 数据摘要统计
    logger.info(f"📊 美国贸易数据摘要:")
    logger.info(f"  总记录数: {len(us_trade_df):,}")
    logger.info(f"  年份范围: {us_trade_df['year'].min()}-{us_trade_df['year'].max()}")
    logger.info(f"  贸易伙伴数: {us_trade_df['us_partner'].nunique()}")
    logger.info(f"  能源产品数: {us_trade_df['energy_product'].nunique()}")
    logger.info(f"  总贸易额: ${us_trade_df['trade_value_usd'].sum():,.0f}")
    
    # 按角色统计
    role_stats = us_trade_df.groupby('us_role')['trade_value_usd'].agg(['count', 'sum'])
    logger.info(f"  按美国角色统计:")
    for role, stats in role_stats.iterrows():
        logger.info(f"    {role}: {stats['count']:,} 条记录, ${stats['sum']:,.0f}")
    
    # 按产品统计  
    product_stats = us_trade_df.groupby('energy_product')['trade_value_usd'].agg(['count', 'sum'])
    logger.info(f"  按能源产品统计:")
    for product, stats in product_stats.iterrows():
        logger.info(f"    {product}: {stats['count']:,} 条记录, ${stats['sum']:,.0f}")
    
    logger.info("✅ 美国能源贸易数据加载完成!")
    return us_trade_df

def add_distance_data(trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    为美国贸易数据添加地理距离信息
    
    Args:
        trade_df: 包含美国贸易数据的DataFrame
        
    Returns:
        添加了distance_km列的DataFrame
        
    注意：
        - 距离数据基于各国首都到华盛顿DC的大圆距离
        - 对于未包含在距离字典中的国家，使用全球平均距离作为估计值
    """
    
    logger.info("🌍 开始添加地理距离数据...")
    
    df = trade_df.copy()
    
    # 添加距离列
    df['distance_km'] = df['us_partner'].map(COUNTRY_DISTANCES)
    
    # 处理未知国家
    unknown_countries = df[df['distance_km'].isnull()]['us_partner'].unique()
    total_countries_in_data = df['us_partner'].nunique()
    known_countries = total_countries_in_data - len(unknown_countries)
    
    logger.info(f"📍 距离数据匹配情况:")
    logger.info(f"  数据中总国家数: {total_countries_in_data}")
    logger.info(f"  成功匹配: {known_countries} 个国家 ({known_countries/total_countries_in_data*100:.1f}%)")
    logger.info(f"  需要估算: {len(unknown_countries)} 个国家 ({len(unknown_countries)/total_countries_in_data*100:.1f}%)")
    
    if len(unknown_countries) > 0:
        # 使用全球平均距离
        global_avg_distance = np.mean(list(COUNTRY_DISTANCES.values()))
        logger.info(f"对未知国家使用全球平均距离: {global_avg_distance:.0f}km")
        
        if len(unknown_countries) <= 10:
            logger.info(f"未知国家列表: {list(unknown_countries)}")
        else:
            logger.info(f"未知国家数量较多，显示前10个: {list(unknown_countries[:10])}")
        
        df['distance_km'] = df['distance_km'].fillna(global_avg_distance)
    
    # 数据验证
    assert df['distance_km'].isnull().sum() == 0, "距离数据中仍有缺失值"
    
    # 距离统计
    logger.info(f"📏 距离数据统计:")
    logger.info(f"  最近距离: {df['distance_km'].min():.0f} km")
    logger.info(f"  最远距离: {df['distance_km'].max():.0f} km") 
    logger.info(f"  平均距离: {df['distance_km'].mean():.0f} km")
    
    # 按距离区间统计贸易伙伴
    distance_bins = [0, 2000, 5000, 8000, 15000]
    distance_labels = ['邻近(<2000km)', '近距离(2-5000km)', '中距离(5-8000km)', '远距离(>8000km)']
    df['distance_category'] = pd.cut(df['distance_km'], bins=distance_bins, labels=distance_labels, include_lowest=True)
    
    distance_partner_stats = df.groupby('distance_category')['us_partner'].nunique()
    logger.info(f"  按距离区间的贸易伙伴数:")
    for category, count in distance_partner_stats.items():
        logger.info(f"    {category}: {count} 个国家")
    
    logger.info("✅ 地理距离数据添加完成!")
    return df

def prepare_dli_dataset(data_dir: str = None) -> pd.DataFrame:
    """
    准备DLI分析的完整数据集
    
    这是数据准备模块的主要接口函数，整合了所有数据准备步骤
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        完整的、准备好进行DLI分析的DataFrame
        
    包含以下列：
        - year, reporter, partner, flow, product_code, product_name
        - trade_value_usd, us_role, us_partner, energy_product
        - distance_km, distance_category
    """
    
    logger.info("🎯 开始准备DLI分析数据集...")
    
    # 第1步：加载美国贸易数据
    us_trade_df = load_us_trade_data(data_dir)
    
    # 第2步：添加距离数据
    complete_df = add_distance_data(us_trade_df)
    
    # 第3步：最终数据验证和清洗
    logger.info("🔧 执行最终数据验证和异常值处理...")
    
    # 移除重复记录
    initial_rows = len(complete_df)
    complete_df = complete_df.drop_duplicates()
    removed_duplicates = initial_rows - len(complete_df)
    if removed_duplicates > 0:
        logger.info(f"移除了 {removed_duplicates} 条重复记录")
    
    # 确保数据类型正确
    complete_df['year'] = complete_df['year'].astype(int)
    complete_df['trade_value_usd'] = complete_df['trade_value_usd'].astype(float)
    complete_df['distance_km'] = complete_df['distance_km'].astype(float)
    
    # 异常值检测和处理
    complete_df = robust_outlier_treatment(complete_df, 'trade_value_usd', method='iqr', factor=3.0)
    complete_df = robust_outlier_treatment(complete_df, 'distance_km', method='iqr', factor=2.0)
    
    # 报告异常值情况
    trade_outliers = complete_df['trade_value_usd_is_outlier'].sum()
    distance_outliers = complete_df['distance_km_is_outlier'].sum()
    
    if trade_outliers > 0 or distance_outliers > 0:
        logger.info(f"📊 异常值检测结果:")
        logger.info(f"  贸易额异常值: {trade_outliers} 条记录 ({trade_outliers/len(complete_df)*100:.2f}%)")
        logger.info(f"  距离异常值: {distance_outliers} 条记录 ({distance_outliers/len(complete_df)*100:.2f}%)")
        logger.info("  注意：异常值已标记但保留在数据中，可使用温莎化处理后的值")
    
    # 按年份、伙伴国、产品排序
    complete_df = complete_df.sort_values(['year', 'us_partner', 'energy_product', 'us_role'])
    complete_df = complete_df.reset_index(drop=True)
    
    # 最终数据摘要
    logger.info(f"✅ DLI数据集准备完成!")
    logger.info(f"📊 最终数据集规模:")
    logger.info(f"  总记录数: {len(complete_df):,}")
    logger.info(f"  时间跨度: {complete_df['year'].min()}-{complete_df['year'].max()}")
    logger.info(f"  贸易伙伴: {complete_df['us_partner'].nunique()} 个国家")
    logger.info(f"  能源产品: {complete_df['energy_product'].nunique()} 种")
    logger.info(f"  数据列数: {len(complete_df.columns)}")
    
    return complete_df

def load_global_trade_data_by_year(year: int, data_dir: str = None) -> pd.DataFrame:
    """
    加载指定年份的全局能源贸易数据
    
    这个函数专为双向DLI分析设计，支持计算出口锁定力时需要的全球贸易格局数据
    
    Args:
        year: 需要加载的年份
        data_dir: 数据目录路径，默认使用项目标准路径
        
    Returns:
        包含该年份所有能源贸易记录的DataFrame
        
    数据结构：
        - year: 年份
        - reporter: 报告国代码  
        - partner: 伙伴国代码
        - flow: 贸易流向 (M=Import, X=Export)
        - product_code: 能源产品代码
        - product_name: 产品名称
        - trade_value_usd: 贸易值（美元）
        - energy_product: 标准化的能源产品名称
    """
    
    logger.info(f"🌍 加载{year}年全球能源贸易数据...")
    
    # 设置数据路径
    if data_dir is None:
        base_dir = Path(__file__).parent.parent.parent  # 到达energy_network目录
        data_dir = base_dir / "data" / "processed_data"
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / f"cleaned_energy_trade_{year}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"❌ {year}年全球数据文件不存在: {file_path}")
    
    try:
        # 读取年度数据
        df = pd.read_csv(file_path)
        logger.info(f"📂 {year}: 成功加载 {len(df):,} 条全球贸易记录")
        
        # 重命名贸易值列以保持一致性
        if 'trade_value_raw_usd' in df.columns:
            df = df.rename(columns={'trade_value_raw_usd': 'trade_value_usd'})
        
        # 标准化能源产品名称
        df['product_code'] = df['product_code'].astype(str)
        df['energy_product'] = df['product_code'].map(ENERGY_PRODUCTS)
        
        # 筛选有效的能源产品
        df = df[df['energy_product'].notna()]
        
        # 确保数据类型正确
        df['trade_value_usd'] = df['trade_value_usd'].astype(float)
        
        # 移除无效贸易值
        df = df[df['trade_value_usd'] > 0]
        
        logger.info(f"✅ {year}: 清洗后保留 {len(df):,} 条有效能源贸易记录")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 处理{year}年全球数据时出错: {e}")
        raise


def get_global_trade_cache() -> Dict[int, pd.DataFrame]:
    """
    获取全局贸易数据缓存
    
    为了避免重复加载大量数据，提供一个简单的缓存机制
    
    Returns:
        年份到DataFrame的字典缓存
    """
    if not hasattr(get_global_trade_cache, '_cache'):
        get_global_trade_cache._cache = {}
    return get_global_trade_cache._cache


def load_global_trade_data_range(start_year: int = 2001, end_year: int = 2024, 
                               data_dir: str = None) -> Dict[int, pd.DataFrame]:
    """
    批量加载指定年份范围的全局贸易数据
    
    Args:
        start_year: 起始年份
        end_year: 结束年份  
        data_dir: 数据目录路径
        
    Returns:
        年份到DataFrame的字典
    """
    
    logger.info(f"🌍 批量加载{start_year}-{end_year}年全球能源贸易数据...")
    
    cache = get_global_trade_cache()
    global_data = {}
    
    for year in range(start_year, end_year + 1):
        if year in cache:
            logger.info(f"📋 {year}: 使用缓存数据")
            global_data[year] = cache[year]
        else:
            try:
                df = load_global_trade_data_by_year(year, data_dir)
                global_data[year] = df
                cache[year] = df
            except FileNotFoundError:
                logger.warning(f"⚠️  {year}: 数据文件不存在，跳过")
                continue
            except Exception as e:
                logger.error(f"❌ {year}: 加载失败 - {e}")
                continue
    
    total_records = sum(len(df) for df in global_data.values())
    logger.info(f"✅ 成功加载{len(global_data)}年数据，总计 {total_records:,} 条记录")
    
    return global_data


def export_prepared_data(df: pd.DataFrame, output_path: str = None) -> str:
    """
    导出准备好的数据集到CSV文件
    
    Args:
        df: 准备好的数据集
        output_path: 输出路径，默认保存到outputs目录
        
    Returns:
        实际的输出文件路径
    """
    
    if output_path is None:
        base_dir = Path(__file__).parent.parent.parent  # 到达energy_network目录
        output_dir = Path(__file__).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "us_trade_prepared_for_dli.csv"
    
    df.to_csv(output_path, index=False)
    logger.info(f"💾 数据已导出至: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    # 测试数据准备功能
    try:
        prepared_data = prepare_dli_dataset()
        output_file = export_prepared_data(prepared_data)
        print(f"✅ 数据准备完成，文件保存在: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ 数据准备失败: {e}")
        raise