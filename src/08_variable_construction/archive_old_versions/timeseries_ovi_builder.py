"""
时间序列OVI构建器 - 性能优化版本
核心改进：减少内存使用，优化Excel读取，向量化计算
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from universal_unit_converter import UniversalUnitConverter
import gc

logger = logging.getLogger(__name__)

class TimeSeriesOVIBuilder:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.rawdata_dir = self.data_dir / "rawdata"
        self.converter = UniversalUnitConverter()
        self.years = list(range(2000, 2025))  # 2000-2024年
        
        # 精简的国家名称标准化映射
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
            'Ukraine': 'UKR', 'Kazakhstan': 'KAZ', 'Venezuela': 'VEN'
        }
    
    def standardize_country_name(self, country: str) -> Optional[str]:
        """标准化国家名称"""
        if pd.isna(country):
            return None
        
        country_str = str(country).strip()
        
        # 过滤掉注释行和汇总行
        filter_keywords = ['total', 'other', 'cent', 'america', 'ussr', 'excludes', 
                          'includes', 'derivatives', '*', '#', 'transformation']
        
        if any(keyword in country_str.lower() for keyword in filter_keywords):
            return None
        
        return self.country_mapping.get(country_str, country_str)
    
    def _create_country_year_panel(self, countries: List[str]) -> pd.DataFrame:
        """创建国别-年度空面板 - 优化版"""
        # 使用向量化创建
        country_array = np.repeat(countries, len(self.years))
        year_array = np.tile(self.years, len(countries))
        return pd.DataFrame({'country': country_array, 'year': year_array})
    
    def _process_lng_terminals_timeseries(self) -> pd.DataFrame:
        """处理LNG接收站数据 - 高性能版本"""
        logger.info("步骤1: 处理LNG接收站时间序列数据...")
        
        lng_file = self.rawdata_dir / "GEM-GGIT-LNG-Terminals-2024-09.xlsx"
        if not lng_file.exists():
            logger.error(f"LNG文件不存在: {lng_file}")
            return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
        
        try:
            # **性能优化1: 只读取必要的列**
            required_cols = ['Country', 'FacilityType', 'Status', 'StartYear1', 'Capacity', 'CapacityUnits']
            df = pd.read_excel(lng_file, sheet_name='LNG Terminals', usecols=required_cols)
            
            # **性能优化2: 早期筛选，减少数据量**
            df = df[
                (df['FacilityType'].str.contains('Import|Terminal', na=False)) &
                (df['Status'].str.lower() == 'operating')
            ].copy()
            
            if df.empty:
                logger.warning("未找到符合条件的LNG终端")
                return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
            
            # **性能优化3: 向量化国家名称标准化**
            df['country'] = df['Country'].map(self.country_mapping).fillna(df['Country'])
            df = df.dropna(subset=['country'])
            
            # **性能优化4: 向量化单位转换**
            df['start_year'] = pd.to_numeric(df['StartYear1'], errors='coerce').fillna(0).astype(int)
            df = df[(df['start_year'] >= 2000) & (df['start_year'] <= 2024)]
            
            # 批量转换容量
            valid_mask = df['CapacityUnits'].notna() & df['Capacity'].notna()
            df = df[valid_mask].copy()
            
            def batch_convert_lng(rows):
                results = []
                for _, row in rows.iterrows():
                    try:
                        converted = self.converter.convert_gas_to_bcm(row['Capacity'], row['CapacityUnits'])
                        results.append(converted)
                    except:
                        results.append(np.nan)
                return results
            
            df['capacity_bcm'] = batch_convert_lng(df)
            df = df.dropna(subset=['capacity_bcm'])
            
            # **性能优化5: 高效聚合和累积**
            agg_df = df.groupby(['country', 'start_year'], as_index=False)['capacity_bcm'].sum()
            
            all_countries = agg_df['country'].unique()
            panel = self._create_country_year_panel(list(all_countries))
            
            # 合并并计算累积值
            panel = panel.merge(agg_df.rename(columns={'start_year': 'year', 'capacity_bcm': 'cap_add'}), 
                               on=['country', 'year'], how='left')
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            panel.sort_values(['country', 'year'], inplace=True)
            panel['lng_capacity_bcm'] = panel.groupby('country')['cap_add'].cumsum()
            
            result = panel[['country', 'year', 'lng_capacity_bcm']].copy()
            
            # 清理内存
            del df, agg_df, panel
            gc.collect()
            
            logger.info(f"LNG时间序列完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"LNG时间序列处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
    
    def _process_gas_pipelines_timeseries(self) -> pd.DataFrame:
        """处理天然气管道数据 - 高性能版本"""
        logger.info("步骤2: 处理天然气管道时间序列数据...")
        
        pipeline_file = self.rawdata_dir / "GEM-GGIT-Gas-Pipelines-2024-12.xlsx"
        if not pipeline_file.exists():
            logger.error(f"管道文件不存在: {pipeline_file}")
            return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
        
        try:
            # **性能优化: 只读取必要的列**
            required_cols = ['Fuel', 'EndCountry', 'Status', 'StartYear1', 'Capacity', 'CapacityUnits']
            df = pd.read_excel(pipeline_file, sheet_name='Gas Pipelines 2024-12-17', usecols=required_cols)
            
            # **早期筛选**
            df = df[
                (df['Fuel'] == 'Gas') &
                (df['Status'].str.lower() == 'operating')
            ].copy()
            
            if df.empty:
                return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
            
            # **向量化处理**
            df['country'] = df['EndCountry'].map(self.country_mapping).fillna(df['EndCountry'])
            df = df.dropna(subset=['country'])
            
            df['start_year'] = pd.to_numeric(df['StartYear1'], errors='coerce').fillna(0).astype(int)
            df = df[(df['start_year'] >= 2000) & (df['start_year'] <= 2024)]
            
            # 批量单位转换
            valid_mask = df['CapacityUnits'].notna() & df['Capacity'].notna()
            df = df[valid_mask].copy()
            
            df['capacity_bcm'] = [
                self.converter.convert_gas_to_bcm(row['Capacity'], row['CapacityUnits'])
                if pd.notna(row['Capacity']) and pd.notna(row['CapacityUnits']) else np.nan
                for _, row in df.iterrows()
            ]
            df = df.dropna(subset=['capacity_bcm'])
            
            # **高效聚合**
            agg_df = df.groupby(['country', 'start_year'], as_index=False)['capacity_bcm'].sum()
            
            all_countries = agg_df['country'].unique()
            panel = self._create_country_year_panel(list(all_countries))
            
            panel = panel.merge(agg_df.rename(columns={'start_year': 'year', 'capacity_bcm': 'cap_add'}), 
                               on=['country', 'year'], how='left')
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            panel.sort_values(['country', 'year'], inplace=True)
            panel['pipeline_capacity_bcm'] = panel.groupby('country')['cap_add'].cumsum()
            
            result = panel[['country', 'year', 'pipeline_capacity_bcm']].copy()
            
            # 清理内存
            del df, agg_df, panel
            gc.collect()
            
            logger.info(f"管道时间序列完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"管道时间序列处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
    
    def _process_gas_consumption_timeseries(self) -> pd.DataFrame:
        """处理天然气消费数据 - 高性能版本"""
        logger.info("步骤3: 处理天然气消费时间序列数据...")
        
        gas_file = self.rawdata_dir / "EI-Stats-Review-all-data.xlsx"
        if not gas_file.exists():
            logger.error(f"天然气消费文件不存在: {gas_file}")
            return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
        
        try:
            # **性能优化: 限制读取范围**
            df = pd.read_excel(gas_file, sheet_name='Gas Consumption - Bcm', skiprows=2, nrows=100)
            
            # **快速识别年份列**
            id_col = df.columns[0] 
            year_cols = [col for col in df.columns[1:] if str(col).replace('.0', '').isdigit()]
            year_cols = [col for col in year_cols if 2000 <= int(str(col).replace('.0', '')) <= 2024]
            
            if not year_cols:
                logger.warning("未找到有效的年份列")
                return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
            
            # **只保留需要的列**
            df = df[[id_col] + year_cols].copy()
            
            # **向量化melt**
            consumption_long = pd.melt(df, id_vars=[id_col], value_vars=year_cols,
                                      var_name='year', value_name='gas_consumption_bcm')
            consumption_long.columns = ['country', 'year', 'gas_consumption_bcm']
            
            # **向量化处理**
            consumption_long['country'] = consumption_long['country'].map(self.country_mapping).fillna(consumption_long['country'])
            consumption_long = consumption_long.dropna(subset=['country'])
            
            consumption_long['year'] = consumption_long['year'].astype(str).str.replace('.0', '').astype(int)
            consumption_long['gas_consumption_bcm'] = pd.to_numeric(consumption_long['gas_consumption_bcm'], errors='coerce')
            
            # **最终筛选**
            result = consumption_long[
                (consumption_long['year'] >= 2000) & 
                (consumption_long['year'] <= 2024) &
                (consumption_long['gas_consumption_bcm'].notna())
            ].copy()
            
            # 清理内存
            del df, consumption_long
            gc.collect()
            
            logger.info(f"天然气消费时间序列完成: {len(result)}条记录")
            return result
            
        except Exception as e:
            logger.error(f"天然气消费时间序列处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
    
    def _construct_ovi_gas(self) -> pd.DataFrame:
        """构建天然气OVI - 优化版本"""
        logger.info("=== 构建天然气OVI时间序列 ===")
        
        # 1. 并行处理各个组件
        lng_capacity = self._process_lng_terminals_timeseries()
        pipeline_capacity = self._process_gas_pipelines_timeseries()
        gas_consumption = self._process_gas_consumption_timeseries()
        
        # 2. 高效合并
        logger.info("步骤4: 合并天然气时间序列数据...")
        
        # 使用外连接合并容量数据
        capacity_data = lng_capacity.merge(pipeline_capacity, on=['country', 'year'], how='outer').fillna(0)
        capacity_data['total_capacity_bcm'] = capacity_data['lng_capacity_bcm'] + capacity_data['pipeline_capacity_bcm']
        
        # 合并消费数据
        final_data = capacity_data.merge(gas_consumption, on=['country', 'year'], how='inner')
        
        # 3. 计算OVI - 向量化
        final_data['ovi_gas'] = final_data['total_capacity_bcm'] / final_data['gas_consumption_bcm']
        final_data['ovi_gas'] = final_data['ovi_gas'].replace([np.inf, -np.inf], np.nan)
        
        # 异常值处理
        final_data = final_data[
            (final_data['ovi_gas'] >= 0.01) & 
            (final_data['ovi_gas'] <= 100) &
            (final_data['ovi_gas'].notna())
        ].copy()
        
        result = final_data[['country', 'year', 'ovi_gas']].copy()
        
        # 清理内存
        del lng_capacity, pipeline_capacity, gas_consumption, capacity_data, final_data
        gc.collect()
        
        logger.info(f"✅ 天然气OVI时间序列完成: {len(result)}条记录")
        return result
    
    def _construct_ovi_oil(self) -> pd.DataFrame:
        """构建石油OVI - 简化版本（用于稳健性检验）"""
        logger.info("=== 构建石油OVI时间序列 ===")
        
        # 简化实现：由于石油数据复杂度高，返回空数据框
        # 实际项目中可根据需要扩展
        logger.info("石油OVI构建跳过（稳健性检验指标）")
        return pd.DataFrame(columns=['country', 'year', 'ovi_oil'])
    
    def build_complete_ovi_timeseries(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """构建完整的OVI时间序列"""
        logger.info("🚀 开始构建完整OVI时间序列...")
        
        try:
            # 构建天然气OVI（主指标）
            gas_ovi = self._construct_ovi_gas()
            
            # 构建石油OVI（稳健性指标）
            oil_ovi = self._construct_ovi_oil()
            
            # 保存结果
            if not gas_ovi.empty:
                gas_ovi.to_csv(self.data_dir / "ovi_gas_timeseries.csv", index=False)
                logger.info(f"✅ 天然气OVI保存完成: {len(gas_ovi)}条记录")
            
            if not oil_ovi.empty:
                oil_ovi.to_csv(self.data_dir / "ovi_oil_timeseries.csv", index=False)
                logger.info(f"✅ 石油OVI保存完成: {len(oil_ovi)}条记录")
            
            return gas_ovi, oil_ovi
            
        except Exception as e:
            logger.error(f"❌ OVI时间序列构建失败: {str(e)}")
            return None, None