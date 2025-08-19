#!/usr/bin/env python3
"""
简化天然气OVI构建器 v3.0 - 回归初心版
=================================

核心原则:
1. 只构建天然气OVI，彻底删除石油相关内容
2. 简单、清晰、可验证的逻辑
3. 确保数据一致性和可追溯性

输入数据:
- LNG接收站容量 (GEM-GGIT-LNG-Terminals-2024-09.xlsx)
- 天然气管道容量 (GEM-GGIT-Gas-Pipelines-2024-12.xlsx) 
- 天然气消费量 (EI-Stats-Review-all-data.xlsx)

输出:
- gas_ovi_clean.csv: 国别-年度天然气OVI面板数据

作者: Energy Network Analysis Team
版本: v3.0 - 回归初心版
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from universal_unit_converter import UniversalUnitConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGasOVIBuilder:
    """简化天然气OVI构建器 - 只关注天然气"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.rawdata_dir = self.data_dir / "rawdata"
        self.converter = UniversalUnitConverter()
        self.years = list(range(2000, 2025))  # 2000-2024年
        
        # 国家名称标准化映射 (精简版)
        self.country_mapping = {
            'United States': 'USA', 'US': 'USA', 'United States of America': 'USA',
            'Russia': 'RUS', 'Russian Federation': 'RUS',
            'China': 'CHN', "China, People's Republic of": 'CHN',
            'Germany': 'DEU', 'Japan': 'JPN', 'United Kingdom': 'GBR',
            'France': 'FRA', 'Italy': 'ITA', 'Canada': 'CAN',
            'India': 'IND', 'Brazil': 'BRA', 'South Korea': 'KOR',
            'Australia': 'AUS', 'Netherlands': 'NLD', 'Norway': 'NOR',
            'Saudi Arabia': 'SAU', 'Iran': 'IRN', 'Iraq': 'IRQ',
            'Kuwait': 'KWT', 'United Arab Emirates': 'ARE',
            'Qatar': 'QAT', 'Nigeria': 'NGA', 'Algeria': 'DZA',
            'Indonesia': 'IDN', 'Egypt': 'EGY', 'Singapore': 'SGP',
            'Mexico': 'MEX', 'Argentina': 'ARG', 'Poland': 'POL',
            'Turkey': 'TUR', 'Turkiye': 'TUR', 'Türkiye': 'TUR',
            'Thailand': 'THA', 'Malaysia': 'MYS', 'South Africa': 'ZAF',
            'Ukraine': 'UKR', 'Kazakhstan': 'KAZ', 'Venezuela': 'VEN',
            'Spain': 'ESP', 'Belgium': 'BEL', 'Austria': 'AUT',
            'Switzerland': 'CHE', 'Greece': 'GRC', 'Portugal': 'PRT'
        }
    
    def standardize_country_name(self, country: str) -> Optional[str]:
        """标准化国家名称"""
        if pd.isna(country):
            return None
        
        country_str = str(country).strip()
        
        # 过滤掉注释行和汇总行
        filter_keywords = ['total', 'other', 'excludes', 'includes', '*', '#']
        
        if any(keyword in country_str.lower() for keyword in filter_keywords):
            return None
        
        return self.country_mapping.get(country_str, country_str)
    
    def process_lng_terminals(self) -> pd.DataFrame:
        """处理LNG接收站数据"""
        logger.info("🏭 处理LNG接收站数据...")
        
        lng_file = self.rawdata_dir / "GEM-GGIT-LNG-Terminals-2024-09.xlsx"
        if not lng_file.exists():
            logger.error(f"❌ LNG文件不存在: {lng_file}")
            return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
        
        try:
            # 读取数据
            df = pd.read_excel(lng_file, sheet_name='LNG Terminals')
            logger.info(f"📊 原始LNG数据: {len(df)}条记录")
            
            # 筛选进口终端
            import_lng = df[
                (df['FacilityType'].str.contains('Import|Terminal', na=False)) &
                (df['Status'].str.lower() == 'operating')
            ].copy()
            logger.info(f"✅ 运营中的进口LNG终端: {len(import_lng)}个")
            
            if import_lng.empty:
                return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
            
            # 标准化国家名称
            import_lng['country'] = import_lng['Country'].apply(self.standardize_country_name)
            import_lng = import_lng.dropna(subset=['country'])
            
            # 处理年份和容量
            import_lng['start_year'] = pd.to_numeric(import_lng['StartYear1'], errors='coerce').fillna(0).astype(int)
            import_lng = import_lng[(import_lng['start_year'] >= 2000) & (import_lng['start_year'] <= 2024)]
            
            # 转换容量单位到BCM
            capacity_data = []
            for _, row in import_lng.iterrows():
                try:
                    capacity_bcm = self.converter.convert_gas_to_bcm(row['Capacity'], row['CapacityUnits'])
                    if pd.notna(capacity_bcm) and capacity_bcm > 0:
                        capacity_data.append({
                            'country': row['country'],
                            'start_year': row['start_year'],
                            'capacity_bcm': capacity_bcm
                        })
                except Exception as e:
                    continue  # 跳过转换失败的记录
            
            if not capacity_data:
                logger.warning("⚠️ 没有成功转换的LNG容量数据")
                return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
            
            # 转换为DataFrame并聚合
            capacity_df = pd.DataFrame(capacity_data)
            agg_df = capacity_df.groupby(['country', 'start_year'], as_index=False)['capacity_bcm'].sum()
            
            # 创建时间序列面板
            all_countries = agg_df['country'].unique()
            panel_data = []
            for country in all_countries:
                for year in self.years:
                    panel_data.append({'country': country, 'year': year})
            
            panel = pd.DataFrame(panel_data)
            
            # 合并容量增量数据
            panel = panel.merge(
                agg_df.rename(columns={'start_year': 'year', 'capacity_bcm': 'cap_add'}),
                on=['country', 'year'], how='left'
            )
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            
            # 计算累积容量
            panel = panel.sort_values(['country', 'year'])
            panel['lng_capacity_bcm'] = panel.groupby('country')['cap_add'].cumsum()
            
            result = panel[panel['year'] <= 2024][['country', 'year', 'lng_capacity_bcm']].copy()
            
            logger.info(f"✅ LNG容量数据完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"❌ LNG数据处理失败: {str(e)}")
            return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
    
    def process_gas_pipelines(self) -> pd.DataFrame:
        """处理天然气管道数据"""
        logger.info("🚇 处理天然气管道数据...")
        
        pipeline_file = self.rawdata_dir / "GEM-GGIT-Gas-Pipelines-2024-12.xlsx"
        if not pipeline_file.exists():
            logger.error(f"❌ 管道文件不存在: {pipeline_file}")
            return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
        
        try:
            # 读取数据
            df = pd.read_excel(pipeline_file, sheet_name='Gas Pipelines 2024-12-17')
            logger.info(f"📊 原始管道数据: {len(df)}条记录")
            
            # 筛选天然气管道
            gas_pipelines = df[
                (df['Fuel'] == 'Gas') &
                (df['Status'].str.lower() == 'operating')
            ].copy()
            logger.info(f"✅ 运营中的天然气管道: {len(gas_pipelines)}条")
            
            if gas_pipelines.empty:
                return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
            
            # 标准化国家名称（使用EndCountry作为进口国）
            gas_pipelines['country'] = gas_pipelines['EndCountry'].apply(self.standardize_country_name)
            gas_pipelines = gas_pipelines.dropna(subset=['country'])
            
            # 处理年份和容量
            gas_pipelines['start_year'] = pd.to_numeric(gas_pipelines['StartYear1'], errors='coerce').fillna(0).astype(int)
            gas_pipelines = gas_pipelines[(gas_pipelines['start_year'] >= 2000) & (gas_pipelines['start_year'] <= 2024)]
            
            # 转换容量单位到BCM
            capacity_data = []
            for _, row in gas_pipelines.iterrows():
                try:
                    capacity_bcm = self.converter.convert_gas_to_bcm(row['Capacity'], row['CapacityUnits'])
                    if pd.notna(capacity_bcm) and capacity_bcm > 0:
                        capacity_data.append({
                            'country': row['country'],
                            'start_year': row['start_year'],
                            'capacity_bcm': capacity_bcm
                        })
                except Exception as e:
                    continue  # 跳过转换失败的记录
            
            if not capacity_data:
                logger.warning("⚠️ 没有成功转换的管道容量数据")
                return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
            
            # 转换为DataFrame并聚合
            capacity_df = pd.DataFrame(capacity_data)
            agg_df = capacity_df.groupby(['country', 'start_year'], as_index=False)['capacity_bcm'].sum()
            
            # 创建时间序列面板
            all_countries = agg_df['country'].unique()
            panel_data = []
            for country in all_countries:
                for year in self.years:
                    panel_data.append({'country': country, 'year': year})
            
            panel = pd.DataFrame(panel_data)
            
            # 合并容量增量数据
            panel = panel.merge(
                agg_df.rename(columns={'start_year': 'year', 'capacity_bcm': 'cap_add'}),
                on=['country', 'year'], how='left'
            )
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            
            # 计算累积容量
            panel = panel.sort_values(['country', 'year'])
            panel['pipeline_capacity_bcm'] = panel.groupby('country')['cap_add'].cumsum()
            
            result = panel[panel['year'] <= 2024][['country', 'year', 'pipeline_capacity_bcm']].copy()
            
            logger.info(f"✅ 管道容量数据完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"❌ 管道数据处理失败: {str(e)}")
            return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
    
    def process_gas_consumption(self) -> pd.DataFrame:
        """处理天然气消费数据"""
        logger.info("📈 处理天然气消费数据...")
        
        gas_file = self.rawdata_dir / "EI-Stats-Review-all-data.xlsx"
        if not gas_file.exists():
            logger.error(f"❌ 消费数据文件不存在: {gas_file}")
            return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
        
        try:
            # 读取消费数据
            df = pd.read_excel(gas_file, sheet_name='Gas Consumption - Bcm', skiprows=2)
            logger.info(f"📊 原始消费数据shape: {df.shape}")
            
            # 识别年份列
            id_col = df.columns[0]
            year_cols = []
            for col in df.columns[1:]:
                try:
                    year_int = int(str(col).replace('.0', ''))
                    if 2000 <= year_int <= 2024:
                        year_cols.append(col)
                except:
                    continue
            
            if not year_cols:
                logger.error("❌ 未找到有效的年份列")
                return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
            
            logger.info(f"✅ 找到{len(year_cols)}个年份列: {min(year_cols)}-{max(year_cols)}")
            
            # 只保留需要的列
            df_clean = df[[id_col] + year_cols].copy()
            
            # 转换为长格式
            consumption_long = pd.melt(
                df_clean,
                id_vars=[id_col],
                value_vars=year_cols,
                var_name='year',
                value_name='gas_consumption_bcm'
            )
            consumption_long.columns = ['country', 'year', 'gas_consumption_bcm']
            
            # 标准化国家名称
            consumption_long['country'] = consumption_long['country'].apply(self.standardize_country_name)
            consumption_long = consumption_long.dropna(subset=['country'])
            
            # 清理年份和数值
            consumption_long['year'] = consumption_long['year'].astype(str).str.replace('.0', '').astype(int)
            consumption_long['gas_consumption_bcm'] = pd.to_numeric(consumption_long['gas_consumption_bcm'], errors='coerce')
            
            # 筛选有效数据
            result = consumption_long[
                (consumption_long['year'] >= 2000) & 
                (consumption_long['year'] <= 2024) &
                (consumption_long['gas_consumption_bcm'] > 0)
            ].copy()
            
            logger.info(f"✅ 消费数据完成: {len(result)}条记录，{result['country'].nunique()}个国家")
            return result
            
        except Exception as e:
            logger.error(f"❌ 消费数据处理失败: {str(e)}")
            return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
    
    def build_gas_ovi(self) -> pd.DataFrame:
        """构建天然气OVI"""
        logger.info("🚀 开始构建天然气OVI...")
        
        # 1. 处理各个组件
        lng_capacity = self.process_lng_terminals()
        pipeline_capacity = self.process_gas_pipelines() 
        gas_consumption = self.process_gas_consumption()
        
        # 2. 合并容量数据
        logger.info("🔧 合并容量数据...")
        capacity_data = lng_capacity.merge(
            pipeline_capacity, 
            on=['country', 'year'], 
            how='outer'
        ).fillna(0)
        
        capacity_data['total_capacity_bcm'] = (
            capacity_data['lng_capacity_bcm'] + 
            capacity_data['pipeline_capacity_bcm']
        )
        
        logger.info(f"📊 容量数据: {len(capacity_data)}条记录，{capacity_data['country'].nunique()}个国家")
        
        # 3. 合并消费数据
        logger.info("🔧 合并消费数据...")
        final_data = capacity_data.merge(
            gas_consumption, 
            on=['country', 'year'], 
            how='inner'  # 只保留有消费数据的记录
        )
        
        logger.info(f"📊 合并后数据: {len(final_data)}条记录，{final_data['country'].nunique()}个国家")
        
        # 4. 计算OVI
        logger.info("🔧 计算OVI...")
        final_data['ovi_gas'] = final_data['total_capacity_bcm'] / final_data['gas_consumption_bcm']
        
        # 异常值处理
        final_data['ovi_gas'] = final_data['ovi_gas'].replace([np.inf, -np.inf], np.nan)
        
        # 筛选合理范围的OVI值
        final_data = final_data[
            (final_data['ovi_gas'] >= 0.001) & 
            (final_data['ovi_gas'] <= 50) &
            (final_data['ovi_gas'].notna())
        ].copy()
        
        # 5. 输出结果
        result = final_data[[
            'country', 'year', 'ovi_gas', 
            'lng_capacity_bcm', 'pipeline_capacity_bcm', 'total_capacity_bcm',
            'gas_consumption_bcm'
        ]].copy()
        
        result = result.sort_values(['country', 'year']).reset_index(drop=True)
        
        logger.info(f"✅ 天然气OVI构建完成:")
        logger.info(f"   📊 最终数据: {len(result)}条记录")
        logger.info(f"   🌍 覆盖国家: {result['country'].nunique()}个")
        logger.info(f"   📅 年份范围: {result['year'].min()}-{result['year'].max()}")
        logger.info(f"   📈 OVI范围: {result['ovi_gas'].min():.3f} - {result['ovi_gas'].max():.3f}")
        
        return result

def main():
    """主函数：构建简化的天然气OVI"""
    logger.info("🎯 简化天然气OVI构建器 v3.0 - 回归初心版")
    logger.info("="*50)
    
    try:
        # 初始化构建器
        data_dir = Path("08data")
        builder = SimpleGasOVIBuilder(data_dir)
        
        # 构建OVI
        ovi_data = builder.build_gas_ovi()
        
        if len(ovi_data) > 0:
            # 保存结果
            output_path = data_dir / "gas_ovi_clean.csv"
            ovi_data.to_csv(output_path, index=False)
            
            logger.info("="*50)
            logger.info("🎉 构建完成!")
            logger.info(f"💾 结果保存至: {output_path}")
            logger.info(f"📊 数据规模: {len(ovi_data)}行 x {len(ovi_data.columns)}列")
            
            # 数据质量检查
            logger.info("📊 数据质量报告:")
            logger.info(f"   国家数: {ovi_data['country'].nunique()}")
            logger.info(f"   年份范围: {ovi_data['year'].min()}-{ovi_data['year'].max()}")
            logger.info(f"   OVI统计: 均值={ovi_data['ovi_gas'].mean():.3f}, 中位数={ovi_data['ovi_gas'].median():.3f}")
            
            # 显示部分数据
            logger.info("📝 数据样例:")
            sample_countries = ovi_data['country'].unique()[:3]
            for country in sample_countries:
                country_data = ovi_data[ovi_data['country'] == country].head(3)
                logger.info(f"   {country}: {len(country_data)}条记录")
        else:
            logger.error("❌ 未能构建任何OVI数据")
            
    except Exception as e:
        logger.error(f"❌ 构建失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()