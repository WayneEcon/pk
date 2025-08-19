"""
时间序列OVI构建器 - 彻底重建版本
核心原则：构建国别-年度面板数据，正确处理基础设施的时间序列特性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from universal_unit_converter import UniversalUnitConverter

logger = logging.getLogger(__name__)

class TimeSeriesOVIBuilder:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.rawdata_dir = self.data_dir / "rawdata"
        self.converter = UniversalUnitConverter()
        self.years = list(range(2000, 2025))  # 2000-2024年
        
        # 详尽的国家名称标准化映射
        self.country_mapping = {
            'United States': 'USA', 'United States of America': 'USA', 'US': 'USA',
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
            'Israel': 'ISR', 'Chile': 'CHL', 'Peru': 'PER',
            'Belgium': 'BEL', 'Spain': 'ESP', 'Portugal': 'PRT',
            'Greece': 'GRC', 'Czech Republic': 'CZE', 'Hungary': 'HUN',
            'Romania': 'ROU', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
            'Serbia': 'SRB', 'Slovakia': 'SVK', 'Slovenia': 'SVN',
            'Lithuania': 'LTU', 'Latvia': 'LVA', 'Estonia': 'EST',
            'Finland': 'FIN', 'Sweden': 'SWE', 'Denmark': 'DNK',
            'Austria': 'AUT', 'Switzerland': 'CHE', 'Ireland': 'IRL',
            'New Zealand': 'NZL', 'Philippines': 'PHL', 'Vietnam': 'VNM',
            'Bangladesh': 'BGD', 'Pakistan': 'PAK', 'Sri Lanka': 'LKA',
            'Myanmar': 'MMR', 'Cambodia': 'KHM', 'Laos': 'LAO',
            'Mongolia': 'MNG', 'Uzbekistan': 'UZB', 'Turkmenistan': 'TKM',
            'Azerbaijan': 'AZE', 'Georgia': 'GEO', 'Armenia': 'ARM',
            'Belarus': 'BLR', 'Moldova': 'MDA', 'Libya': 'LBY',
            'Tunisia': 'TUN', 'Morocco': 'MAR', 'Sudan': 'SDN',
            'Ethiopia': 'ETH', 'Kenya': 'KEN', 'Ghana': 'GHA',
            'Angola': 'AGO', 'Mozambique': 'MOZ', 'Tanzania': 'TZA',
            'Zambia': 'ZMB', 'Zimbabwe': 'ZWE', 'Botswana': 'BWA',
            'Ecuador': 'ECU', 'Colombia': 'COL', 'Bolivia': 'BOL',
            'Paraguay': 'PRY', 'Uruguay': 'URY', 'Costa Rica': 'CRI',
            'Panama': 'PAN', 'Dominican Republic': 'DOM', 'Trinidad and Tobago': 'TTO',
            'Jamaica': 'JAM', 'Barbados': 'BRB', 'Bahrain': 'BHR',
            'Oman': 'OMN', 'Jordan': 'JOR', 'Lebanon': 'LBN',
            'Syria': 'SYR', 'Yemen': 'YEM', 'Afghanistan': 'AFG',
            'Nepal': 'NPL', 'Bhutan': 'BTN', 'Maldives': 'MDV'
        }
    
    def standardize_country_name(self, country: str) -> Optional[str]:
        """标准化国家名称"""
        if pd.isna(country):
            return None
        
        country_str = str(country).strip()
        
        # 过滤掉注释行和汇总行
        filter_keywords = [
            'total', 'other', 'cent', 'america', 'ussr', 'excludes', 
            'includes', 'derivatives', '*', '#', 'transformation',
            'prior to', 'liquid fuels', 'of which', 'bunkers'
        ]
        
        if any(keyword in country_str.lower() for keyword in filter_keywords):
            return None
        
        return self.country_mapping.get(country_str, country_str)
    
    def _create_country_year_panel(self, countries: List[str]) -> pd.DataFrame:
        """创建国别-年度空面板"""
        panel_data = []
        for country in countries:
            for year in self.years:
                panel_data.append({'country': country, 'year': year})
        return pd.DataFrame(panel_data)
    
    def _process_lng_terminals_timeseries(self) -> pd.DataFrame:
        """处理LNG接收站数据 - 时间序列版本"""
        logger.info("步骤1: 处理LNG接收站时间序列数据...")
        
        lng_file = self.rawdata_dir / "GEM-GGIT-LNG-Terminals-2024-09.xlsx"
        if not lng_file.exists():
            logger.error(f"LNG文件不存在: {lng_file}")
            return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
        
        try:
            df = pd.read_excel(lng_file, sheet_name='LNG Terminals')
            
            # 筛选进口类型的LNG终端 (使用FacilityType字段)
            import_lng = df[df['FacilityType'].str.contains('Import|Terminal', na=False)].copy()
            logger.info(f"进口LNG终端总数: {len(import_lng)}个")

            # 标准化国家名称
            import_lng['country'] = import_lng['Country'].apply(self.standardize_country_name)
            import_lng = import_lng.dropna(subset=['country'])
            
            # 筛选运营项目并清理年份
            operating_lng = import_lng[import_lng['Status'].str.lower() == 'operating'].copy()
            operating_lng['start_year'] = pd.to_numeric(operating_lng['StartYear1'], errors='coerce').dropna().astype(int)
            
            # 单位转换
            def safe_convert(row):
                try:
                    return self.converter.convert_gas_to_bcm(row['Capacity'], row['CapacityUnits'])
                except Exception:
                    return np.nan
            operating_lng['capacity_bcm'] = operating_lng.apply(safe_convert, axis=1)
            
            # 过滤掉转换失败的项目
            operating_lng = operating_lng.dropna(subset=['start_year', 'capacity_bcm'])
            
            # ---------- 向量化累积 ----------
            if operating_lng.empty:
                logger.warning("未找到符合条件的 LNG 项目")
                return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])

            agg_df = operating_lng.groupby(['country', 'start_year'])['capacity_bcm'].sum().reset_index()
            
            all_countries = agg_df['country'].unique()
            panel = self._create_country_year_panel(list(all_countries))
            panel = panel.merge(
                agg_df.rename(columns={'start_year': 'year', 'capacity_bcm': 'cap_add'}),
                on=['country', 'year'],
                how='left'
            )
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            panel['lng_capacity_bcm'] = panel.groupby('country')['cap_add'].cumsum()

            result = panel[
                (panel['year'] >= 2000) & (panel['year'] <= 2024)
            ][['country', 'year', 'lng_capacity_bcm']]
            
            logger.info(f"LNG时间序列数据完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"LNG时间序列数据处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'lng_capacity_bcm'])
    
    def _process_gas_pipelines_timeseries(self) -> pd.DataFrame:
        """处理天然气管道数据 - 时间序列版本"""
        logger.info("步骤2: 处理天然气管道时间序列数据...")
        
        pipeline_file = self.rawdata_dir / "GEM-GGIT-Gas-Pipelines-2024-12.xlsx"
        if not pipeline_file.exists():
            logger.error(f"管道文件不存在: {pipeline_file}")
            return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
        
        try:
            df = pd.read_excel(pipeline_file, sheet_name='Gas Pipelines 2024-12-17')
            
            # 筛选天然气管道
            gas_pipelines = df[df['Fuel'] == 'Gas'].copy()
            logger.info(f"天然气管道总数: {len(gas_pipelines)}条")

            # 标准化国家名称
            gas_pipelines['country'] = gas_pipelines['EndCountry'].apply(self.standardize_country_name)
            gas_pipelines = gas_pipelines.dropna(subset=['country'])

            # 筛选运营项目并清理年份
            operating_pipelines = gas_pipelines[gas_pipelines['Status'].str.lower() == 'operating'].copy()
            operating_pipelines['start_year'] = pd.to_numeric(operating_pipelines['StartYear'], errors='coerce').dropna().astype(int)

            # 单位转换
            def safe_convert(row):
                try:
                    return self.converter.convert_gas_to_bcm(row['Capacity'], row['CapacityUnits'])
                except Exception:
                    return np.nan
            operating_pipelines['capacity_bcm'] = operating_pipelines.apply(safe_convert, axis=1)

            # 过滤掉转换失败的项目
            operating_pipelines = operating_pipelines.dropna(subset=['start_year', 'capacity_bcm'])

            # ---------- 向量化累积 ----------
            if operating_pipelines.empty:
                logger.warning("未找到符合条件的天然气管道项目")
                return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])

            agg_df = operating_pipelines.groupby(['country', 'start_year'])['capacity_bcm'].sum().reset_index()

            all_countries = agg_df['country'].unique()
            panel = self._create_country_year_panel(list(all_countries))
            panel = panel.merge(
                agg_df.rename(columns={'start_year': 'year', 'capacity_bcm': 'cap_add'}),
                on=['country', 'year'],
                how='left'
            )
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            panel['pipeline_capacity_bcm'] = panel.groupby('country')['cap_add'].cumsum()

            result = panel[
                (panel['year'] >= 2000) & (panel['year'] <= 2024)
            ][['country', 'year', 'pipeline_capacity_bcm']]
            
            logger.info(f"管道时间序列数据完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"管道时间序列数据处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'pipeline_capacity_bcm'])
    
    def _process_gas_consumption_timeseries(self) -> pd.DataFrame:
        """处理天然气消费数据 - 时间序列版本"""
        logger.info("步骤3: 处理天然气消费时间序列数据...")
        
        gas_file = self.rawdata_dir / "EI-Stats-Review-all-data.xlsx"
        if not gas_file.exists():
            logger.error(f"天然气消费文件不存在: {gas_file}")
            return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
        
        try:
            # 读取BCM格式的消费数据，跳过前2行标题
            df = pd.read_excel(gas_file, sheet_name='Gas Consumption - Bcm', skiprows=2)
            
            # 使用pd.melt转换为长格式
            id_cols = df.columns[0]  # 第一列是国家
            year_cols = [col for col in df.columns[1:] if str(col).replace('.0', '').isdigit()]
            
            consumption_long = pd.melt(
                df[[id_cols] + year_cols],
                id_vars=[id_cols],
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
            
            # 筛选2000-2024年数据并去除缺失值
            result = consumption_long[
                (consumption_long['year'] >= 2000) & 
                (consumption_long['year'] <= 2024)
            ].dropna()
            
            logger.info(f"天然气消费时间序列数据完成: {len(result)}条记录")
            return result
            
        except Exception as e:
            logger.error(f"天然气消费时间序列数据处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'gas_consumption_bcm'])
    
    def _construct_ovi_gas(self) -> pd.DataFrame:
        """构建天然气OVI - 国别-年度面板数据版本"""
        logger.info("=== 构建天然气OVI时间序列 ===")
        
        # 1. 处理LNG接收站时间序列数据
        lng_capacity = self._process_lng_terminals_timeseries()
        
        # 2. 处理天然气管道时间序列数据  
        pipeline_capacity = self._process_gas_pipelines_timeseries()
        
        # 3. 处理天然气消费时间序列数据
        gas_consumption = self._process_gas_consumption_timeseries()
        
        # 4. 合并所有时间序列数据
        logger.info("步骤4: 合并时间序列数据...")
        
        # 外连接合并容量数据
        capacity_data = lng_capacity.merge(
            pipeline_capacity, 
            on=['country', 'year'], 
            how='outer'
        ).fillna(0)
        
        capacity_data['total_gas_capacity_bcm'] = (
            capacity_data['lng_capacity_bcm'] + capacity_data['pipeline_capacity_bcm']
        )
        
        # 内连接合并消费数据（只保留有消费数据的国家-年份）
        ovi_data = gas_consumption.merge(
            capacity_data, 
            on=['country', 'year'], 
            how='inner'
        )
        
        # 5. 计算OVI指标
        ovi_data['ovi_gas'] = ovi_data['total_gas_capacity_bcm'] / ovi_data['gas_consumption_bcm']
        ovi_data['ovi_gas'] = ovi_data['ovi_gas'].replace([np.inf, -np.inf], np.nan)
        ovi_data['ovi_gas'] = ovi_data['ovi_gas'].clip(lower=0)  # 确保非负
        
        # 6. 返回最终结果
        result = ovi_data[['country', 'year', 'ovi_gas']].copy()
        
        logger.info(f"天然气OVI时间序列构建完成:")
        logger.info(f"  总记录数: {len(result)}")
        logger.info(f"  覆盖国家: {result['country'].nunique()}个")
        logger.info(f"  时间范围: {result['year'].min()}-{result['year'].max()}")
        
        return result
    
    def _process_refinery_capacity_timeseries(self) -> pd.DataFrame:
        """处理炼油能力数据 - 时间序列版本"""
        logger.info("步骤1: 处理炼油能力时间序列数据...")
        
        oil_file = self.rawdata_dir / "EI-Stats-Review-all-data.xlsx"
        if not oil_file.exists():
            logger.error(f"石油数据文件不存在: {oil_file}")
            return pd.DataFrame(columns=['country', 'year', 'refinery_capacity_kbpd'])
        
        try:
            # 读取炼油能力数据，跳过前2行标题
            df = pd.read_excel(oil_file, sheet_name='Oil refinery - capacity', skiprows=2)
            
            # 使用pd.melt转换为长格式
            id_cols = df.columns[0]  # 第一列是国家
            year_cols = [col for col in df.columns[1:] if str(col).replace('.0', '').isdigit()]
            
            refinery_long = pd.melt(
                df[[id_cols] + year_cols],
                id_vars=[id_cols],
                value_vars=year_cols,
                var_name='year',
                value_name='refinery_capacity_kbpd'
            )
            
            refinery_long.columns = ['country', 'year', 'refinery_capacity_kbpd']
            
            # 标准化国家名称
            refinery_long['country'] = refinery_long['country'].apply(self.standardize_country_name)
            refinery_long = refinery_long.dropna(subset=['country'])
            
            # 清理年份和数值
            refinery_long['year'] = refinery_long['year'].astype(str).str.replace('.0', '').astype(int)
            refinery_long['refinery_capacity_kbpd'] = pd.to_numeric(refinery_long['refinery_capacity_kbpd'], errors='coerce')
            
            # 筛选2000-2024年数据并去除缺失值
            result = refinery_long[
                (refinery_long['year'] >= 2000) & 
                (refinery_long['year'] <= 2024)
            ].dropna()
            
            logger.info(f"炼油能力时间序列数据完成: {len(result)}条记录")
            return result
            
        except Exception as e:
            logger.error(f"炼油能力时间序列数据处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'refinery_capacity_kbpd'])
    
    def _process_oil_pipelines_timeseries(self) -> pd.DataFrame:
        """处理石油管道数据 - 时间序列版本"""
        logger.info("步骤2: 处理石油管道时间序列数据...")
        
        pipeline_file = self.rawdata_dir / "GEM-GOIT-Oil-NGL-Pipelines-2025-03.xlsx"
        if not pipeline_file.exists():
            logger.error(f"石油管道文件不存在: {pipeline_file}")
            return pd.DataFrame(columns=['country', 'year', 'oil_pipeline_capacity_bpd'])
        
        try:
            df = pd.read_excel(pipeline_file, sheet_name='Pipelines')
            
            # 筛选运营中的原油管道
            operating_oil_pipelines = df[df['Status'].str.lower() == 'operating'].copy()
            logger.info(f"运营中的石油管道总数: {len(operating_oil_pipelines)}条")

            # 标准化国家名称
            operating_oil_pipelines['country'] = operating_oil_pipelines['EndCountry'].apply(self.standardize_country_name)
            operating_oil_pipelines = operating_oil_pipelines.dropna(subset=['country'])

            # 清理年份
            operating_oil_pipelines['start_year'] = pd.to_numeric(operating_oil_pipelines['StartYear'], errors='coerce').dropna().astype(int)

            # 单位转换
            def convert_to_bpd(row):
                unit = row['CapacityUnits']
                capacity = row['Capacity']
                if unit == 'bpd':
                    return capacity
                if unit == 'mtpa':
                    return capacity * 1000000 / (365 * 0.137)
                return np.nan
            
            operating_oil_pipelines['capacity_bpd'] = operating_oil_pipelines.apply(convert_to_bpd, axis=1)
            
            # 过滤掉转换失败的项目
            operating_oil_pipelines = operating_oil_pipelines.dropna(subset=['start_year', 'capacity_bpd'])

            # ---------- 向量化累积 ----------
            if operating_oil_pipelines.empty:
                logger.warning("未找到符合条件的石油管道项目")
                return pd.DataFrame(columns=['country', 'year', 'oil_pipeline_capacity_bpd'])

            agg_df = operating_oil_pipelines.groupby(['country', 'start_year'])['capacity_bpd'].sum().reset_index()

            all_countries = agg_df['country'].unique()
            panel = self._create_country_year_panel(list(all_countries))
            panel = panel.merge(
                agg_df.rename(columns={'start_year': 'year', 'capacity_bpd': 'cap_add'}),
                on=['country', 'year'],
                how='left'
            )
            panel['cap_add'] = panel['cap_add'].fillna(0.0)
            panel['oil_pipeline_capacity_bpd'] = panel.groupby('country')['cap_add'].cumsum()

            result = panel[
                (panel['year'] >= 2000) & (panel['year'] <= 2024)
            ][['country', 'year', 'oil_pipeline_capacity_bpd']]

            logger.info(f"石油管道时间序列数据完成: {len(result)}条记录，{len(all_countries)}个国家")
            return result
            
        except Exception as e:
            logger.error(f"石油管道时间序列数据处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'oil_pipeline_capacity_bpd'])
    
    def _process_oil_consumption_timeseries(self) -> pd.DataFrame:
        """处理石油消费数据 - 时间序列版本"""
        logger.info("步骤3: 处理石油消费时间序列数据...")
        
        oil_file = self.rawdata_dir / "EI-Stats-Review-all-data.xlsx"
        if not oil_file.exists():
            logger.error(f"石油消费文件不存在: {oil_file}")
            return pd.DataFrame(columns=['country', 'year', 'oil_consumption_tonnes'])
        
        try:
            # 读取石油消费数据，跳过前2行标题
            df = pd.read_excel(oil_file, sheet_name='Oil Consumption - Tonnes', skiprows=2)
            
            # 使用pd.melt转换为长格式
            id_cols = df.columns[0]  # 第一列是国家
            year_cols = [col for col in df.columns[1:] if str(col).replace('.0', '').isdigit()]
            
            consumption_long = pd.melt(
                df[[id_cols] + year_cols],
                id_vars=[id_cols],
                value_vars=year_cols,
                var_name='year',
                value_name='oil_consumption_tonnes'
            )
            
            consumption_long.columns = ['country', 'year', 'oil_consumption_tonnes']
            
            # 标准化国家名称
            consumption_long['country'] = consumption_long['country'].apply(self.standardize_country_name)
            consumption_long = consumption_long.dropna(subset=['country'])
            
            # 清理年份和数值
            consumption_long['year'] = consumption_long['year'].astype(str).str.replace('.0', '').astype(int)
            consumption_long['oil_consumption_tonnes'] = pd.to_numeric(consumption_long['oil_consumption_tonnes'], errors='coerce')
            
            # 筛选2000-2024年数据并去除缺失值
            result = consumption_long[
                (consumption_long['year'] >= 2000) & 
                (consumption_long['year'] <= 2024)
            ].dropna()
            
            logger.info(f"石油消费时间序列数据完成: {len(result)}条记录")
            return result
            
        except Exception as e:
            logger.error(f"石油消费时间序列数据处理失败: {e}")
            return pd.DataFrame(columns=['country', 'year', 'oil_consumption_tonnes'])
    
    def _construct_ovi_oil(self) -> pd.DataFrame:
        """构建石油OVI - 国别-年度面板数据版本"""
        logger.info("=== 构建石油OVI时间序列 ===")
        
        # 1. 处理炼油能力时间序列数据
        refinery_capacity = self._process_refinery_capacity_timeseries()
        
        # 2. 处理石油管道时间序列数据
        pipeline_capacity = self._process_oil_pipelines_timeseries()
        
        # 3. 处理石油消费时间序列数据
        oil_consumption = self._process_oil_consumption_timeseries()
        
        # 4. 合并所有时间序列数据
        logger.info("步骤4: 合并石油时间序列数据...")
        
        # 外连接合并容量数据
        capacity_data = refinery_capacity.merge(
            pipeline_capacity, 
            on=['country', 'year'], 
            how='outer'
        ).fillna(0)
        
        # 单位转换：统一转换为mtpa
        capacity_data['refinery_capacity_mtpa'] = capacity_data['refinery_capacity_kbpd'] * 365 * 0.137 / 1000  # kbpd到mtpa
        capacity_data['oil_pipeline_capacity_mtpa'] = capacity_data['oil_pipeline_capacity_bpd'] * 365 * 0.137 / 1000000  # bpd到mtpa
        
        capacity_data['total_oil_capacity_mtpa'] = (
            capacity_data['refinery_capacity_mtpa'] + capacity_data['oil_pipeline_capacity_mtpa']
        )
        
        # 内连接合并消费数据（只保留有消费数据的国家-年份）
        ovi_data = oil_consumption.merge(
            capacity_data, 
            on=['country', 'year'], 
            how='inner'
        )
        
        # 5. 计算OVI指标（消费数据单位转换：Million tonnes直接使用）
        ovi_data['oil_consumption_mtpa'] = ovi_data['oil_consumption_tonnes']  # 数据已经是Million tonnes per year
        ovi_data['ovi_oil'] = ovi_data['total_oil_capacity_mtpa'] / ovi_data['oil_consumption_mtpa']
        ovi_data['ovi_oil'] = ovi_data['ovi_oil'].replace([np.inf, -np.inf], np.nan)
        ovi_data['ovi_oil'] = ovi_data['ovi_oil'].clip(lower=0)  # 确保非负
        
        # 6. 返回最终结果
        result = ovi_data[['country', 'year', 'ovi_oil']].copy()
        
        logger.info(f"石油OVI时间序列构建完成:")
        logger.info(f"  总记录数: {len(result)}")
        logger.info(f"  覆盖国家: {result['country'].nunique()}个")
        logger.info(f"  时间范围: {result['year'].min()}-{result['year'].max()}")
        
        return result
    
    def build_complete_ovi_timeseries(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """构建完整的OVI时间序列数据"""
        logger.info("=== 开始构建完整OVI时间序列 ===")
        
        # 构建天然气OVI
        gas_ovi = self._construct_ovi_gas()
        
        # 构建石油OVI
        oil_ovi = self._construct_ovi_oil()
        
        logger.info("=== OVI时间序列构建完成 ===")
        
        return gas_ovi, oil_ovi


def main():
    """测试时间序列OVI构建器"""
    print("🔧 时间序列OVI构建器测试")
    print("="*50)
    
    logging.basicConfig(level=logging.INFO)
    
    builder = TimeSeriesOVIBuilder("src/08_variable_construction/08data")
    
    # 构建完整OVI时间序列
    gas_ovi, oil_ovi = builder.build_complete_ovi_timeseries()
    
    print(f"\n✅ 时间序列OVI构建完成:")
    print(f"   天然气OVI: {len(gas_ovi)}条记录，{gas_ovi['country'].nunique()}个国家")
    print(f"   石油OVI: {len(oil_ovi)}条记录，{oil_ovi['country'].nunique()}个国家")
    
    # 保存时间序列数据
    gas_ovi.to_csv("data/processed_data/ovi_gas_timeseries.csv", index=False)
    oil_ovi.to_csv("data/processed_data/ovi_oil_timeseries.csv", index=False)
    
    print(f"   💾 时间序列数据已保存")

if __name__ == "__main__":
    main()