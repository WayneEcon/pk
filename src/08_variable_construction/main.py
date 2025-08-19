#!/usr/bin/env python3
"""
变量构建模块 (Variable Construction Module)
========================================

本模块是项目研究范式更新后的数据奠基模块。
核心目标：从基础数据源出发，搜集、计算并整合所有研究需要的变量，
最终生成一份干净、完整的国别-年度面板数据集 analytical_panel.csv。

主要功能：
1. 搜集宏观经济控制变量 (World Bank API)
2. 加载基础数据 (01, 03, 04模块输出)
3. 构建核心变量 (Node-DLI_US, Vul_US, OVI, US_ProdShock)
4. 整合输出最终分析面板

作者：Energy Network Analysis Team
版本：v1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import json
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
# 集成新版时间序列 OVI 计算器
from timeseries_ovi_builder import TimeSeriesOVIBuilder

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('variable_construction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 可选依赖
try:
    import wbdata
    HAS_WBDATA = True
    logger.info("✅ wbdata库可用，将使用World Bank API")
except ImportError:
    HAS_WBDATA = False
    logger.warning("⚠️ wbdata库未安装，将使用模拟数据")

try:
    import requests
    HAS_REQUESTS = True
    logger.info("✅ requests库可用，将使用EIA API")
except ImportError:
    HAS_REQUESTS = False
    logger.warning("⚠️ requests库未安装，将使用模拟数据")

try:
    from scipy.signal import savgol_filter
    from statsmodels.tsa.filters.hp_filter import hpfilter
    HAS_FILTERING = True
    logger.info("✅ 滤波库可用，将使用HP滤波")
except ImportError:
    HAS_FILTERING = False
    logger.warning("⚠️ 滤波库未安装，将使用简化处理")

class VariableConstructor:
    """变量构建主类"""
    
    def __init__(self, base_dir: str = None):
        """初始化变量构建器"""
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.data_dir = self.base_dir / "data"
        self.output_dir = Path(__file__).parent / "outputs"
        self.temp_data_dir = Path(__file__).parent / "08data"  # 使用08data目录
        
        # 创建必要目录
        self.output_dir.mkdir(exist_ok=True)
        self.temp_data_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.macro_data = None
        self.base_data = {}
        self.core_variables = {}
        self.final_panel = None
        
        logger.info("🏗️ 变量构建器初始化完成")
        logger.info(f"   项目根目录: {self.base_dir}")
        logger.info(f"   输出目录: {self.output_dir}")
    
    def fetch_macro_controls(self) -> pd.DataFrame:
        """
        搜集宏观经济控制变量
        
        Returns:
            包含宏观经济变量的DataFrame
        """
        logger.info("🌍 开始搜集宏观经济控制变量...")
        
        # 优先尝试从缓存加载
        cache_path = self.temp_data_dir / "macro_controls.csv"
        if cache_path.exists():
            try:
                logger.info("   从缓存加载宏观数据...")
                macro_data = pd.read_csv(cache_path)
                
                # 确保年份为整数类型
                if 'year' in macro_data.columns:
                    # 处理可能的datetime字符串
                    macro_data['year'] = pd.to_datetime(macro_data['year']).dt.year
                
                logger.info(f"✅ 从缓存加载宏观数据: {len(macro_data)} 行记录")
                logger.info(f"   数据范围: {macro_data['year'].min()}-{macro_data['year'].max()}")
                
                self.macro_data = macro_data
                return macro_data
                
            except Exception as e:
                logger.warning(f"⚠️ 缓存数据加载失败: {str(e)}，将重新获取")
        
        if not HAS_WBDATA:
            logger.warning("⚠️ wbdata库不可用，无法获取World Bank数据")
            return None
        
        try:
            # 定义变量映射
            indicators = {
                'NY.GDP.MKTP.CD': 'gdp_current_usd',
                'SP.POP.TOTL': 'population_total', 
                'NE.TRD.GNFS.ZS': 'trade_openness_gdp_pct'
            }
            
            # 定义时间范围和国家
            import datetime
            date_range = (datetime.datetime(2000, 1, 1), datetime.datetime(2024, 12, 31))
            
            # 从现有数据推断国家列表
            countries = self._get_country_list()
            
            # 从World Bank API获取数据
            logger.info(f"   从World Bank获取 {len(indicators)} 个指标，{len(countries)} 个国家")
            
            macro_data = wbdata.get_dataframe(
                indicators, 
                country=countries,
                date=date_range,
                parse_dates=True
            ).reset_index()
            
            # 数据清洗
            macro_data = macro_data.rename(columns={
                'country': 'country_name',
                'date': 'year'
            })
            
            # 转换年份为整数（解决与其他数据合并时的类型冲突）
            if 'year' in macro_data.columns:
                macro_data['year'] = pd.to_datetime(macro_data['year']).dt.year
            
            # 计算对数变换
            if 'gdp_current_usd' in macro_data.columns:
                macro_data['log_gdp'] = np.log(macro_data['gdp_current_usd'].replace(0, np.nan))
            
            if 'population_total' in macro_data.columns:
                macro_data['log_population'] = np.log(macro_data['population_total'].replace(0, np.nan))
            
            # 保存中间结果
            output_path = self.temp_data_dir / "macro_controls.csv"
            macro_data.to_csv(output_path, index=False)
            
            logger.info(f"✅ 宏观数据搜集完成: {len(macro_data)} 行记录")
            logger.info(f"   数据范围: {macro_data['year'].min()}-{macro_data['year'].max()}")
            logger.info(f"   保存至: {output_path}")
            
            self.macro_data = macro_data
            return macro_data
            
        except Exception as e:
            logger.error(f"❌ World Bank API调用失败: {str(e)}")
            logger.warning("⚠️ 无法获取World Bank数据，宏观变量将标记为缺失值")
            return None
    
    def _get_country_list(self) -> List[str]:
        """从现有数据推断国家列表"""
        try:
            # 尝试从01模块的输出获取国家列表
            trade_data_path = self.base_dir / "src" / "01_data_processing" / "cleaned_trade_flow.csv"
            if trade_data_path.exists():
                trade_data = pd.read_csv(trade_data_path, nrows=1000)  # 只读前1000行推断
                countries = set()
                if 'exporter_iso3' in trade_data.columns:
                    countries.update(trade_data['exporter_iso3'].unique())
                if 'importer_iso3' in trade_data.columns:
                    countries.update(trade_data['importer_iso3'].unique())
                countries = list(countries)
                logger.info(f"   从贸易数据推断出 {len(countries)} 个国家")
                return countries
        except Exception as e:
            logger.warning(f"⚠️ 无法从现有数据推断国家列表: {str(e)}")
        
        # 使用默认主要国家列表
        default_countries = [
            'USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'ITA', 'BRA', 'CAN',
            'RUS', 'AUS', 'KOR', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'CHE'
        ]
        logger.info(f"   使用默认国家列表: {len(default_countries)} 个国家")
        return default_countries

    def load_base_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载基础数据 (01, 03, 04模块输出)
        
        Returns:
            包含各模块数据的字典
        """
        logger.info("📁 开始加载基础数据...")
        
        base_data = {}
        
        # 加载01模块输出 - 贸易流数据
        trade_data_files = list((self.base_dir / "data" / "processed_data").glob("cleaned_energy_trade_*.csv"))
        
        if trade_data_files:
            try:
                # 合并所有年份的贸易数据
                trade_data_list = []
                for file_path in sorted(trade_data_files):
                    yearly_data = pd.read_csv(file_path)
                    trade_data_list.append(yearly_data)
                
                trade_data = pd.concat(trade_data_list, ignore_index=True)
                base_data['trade_flow'] = trade_data
                logger.info(f"✅ 加载贸易流数据: 合并 {len(trade_data_files)} 个文件 ({len(trade_data)} 行)")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载贸易数据: {str(e)}")
        else:
            logger.warning("⚠️ 未找到cleaned_energy_trade数据文件")
        
        # 加载03模块输出 - 网络指标
        metrics_data_path = self.base_dir / "src" / "03_metrics"
        
        # 节点中心性指标
        node_metrics_path = metrics_data_path / "node_centrality_metrics.csv"
        if node_metrics_path.exists():
            try:
                node_metrics = pd.read_csv(node_metrics_path)
                base_data['node_metrics'] = node_metrics
                logger.info(f"✅ 加载节点指标: {len(node_metrics)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载节点指标: {str(e)}")
        
        # 全局网络指标
        global_metrics_path = metrics_data_path / "global_network_metrics.csv"
        if global_metrics_path.exists():
            try:
                global_metrics = pd.read_csv(global_metrics_path)
                base_data['global_metrics'] = global_metrics
                logger.info(f"✅ 加载全局指标: {len(global_metrics)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载全局指标: {str(e)}")
        
        # 加载04模块输出 - DLI数据
        dli_data_path = self.base_dir / "src" / "04_dli_analysis" / "dli_panel_data.csv"
        if dli_data_path.exists():
            try:
                dli_data = pd.read_csv(dli_data_path)
                base_data['dli_panel'] = dli_data
                logger.info(f"✅ 加载DLI面板数据: {len(dli_data)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载DLI数据: {str(e)}")
        
        # 如果关键数据缺失，记录缺失但不生成虚假数据
        if not base_data:
            logger.warning("⚠️ 未找到任何基础数据，相关变量将标记为缺失值")
        
        self.base_data = base_data
        logger.info(f"✅ 基础数据加载完成，共 {len(base_data)} 个数据集")
        
        return base_data
    
    def construct_core_variables(self) -> Dict[str, pd.DataFrame]:
        """
        构建核心变量
        
        Returns:
            包含核心变量的字典
        """
        logger.info("⚙️ 开始构建核心变量...")
        
        core_vars = {}
        
        # 构建 Node-DLI_US
        node_dli_us = self._construct_node_dli_us()
        if node_dli_us is not None:
            core_vars['node_dli_us'] = node_dli_us
        
        # 构建 Vul_US
        vul_us = self._construct_vul_us()
        if vul_us is not None:
            core_vars['vul_us'] = vul_us
        
        # 构建OVI（新版时间序列）
        try:
            builder = TimeSeriesOVIBuilder(self.temp_data_dir)
            gas_ovi, oil_ovi = builder.build_complete_ovi_timeseries()
            
            if gas_ovi is not None:
                core_vars['ovi_gas'] = gas_ovi
                logger.info(f"✅ OVI_gas 时间序列已生成: {len(gas_ovi)} 行")
            
            if oil_ovi is not None:
                core_vars['ovi_oil'] = oil_ovi
                logger.info(f"✅ OVI_oil 时间序列已生成: {len(oil_ovi)} 行")
                
        except Exception as e:
            logger.error(f"❌ OVI 时间序列构建失败: {e}")
        
        # 构建 US_ProdShock
        us_prod_shock = self._construct_us_prod_shock()
        if us_prod_shock is not None:
            core_vars['us_prod_shock'] = us_prod_shock
        
        self.core_variables = core_vars
        logger.info(f"✅ 核心变量构建完成，共 {len(core_vars)} 个变量")
        
        return core_vars
    
    def _construct_node_dli_us(self) -> Optional[pd.DataFrame]:
        """构建 Node-DLI_US (美国锚定动态锁定指数)"""
        logger.info("   构建 Node-DLI_US...")
        
        try:
            if 'dli_panel' not in self.base_data or 'trade_flow' not in self.base_data:
                logger.warning("⚠️ 缺少DLI或贸易数据，跳过Node-DLI_US构建")
                return None
            
            dli_data = self.base_data['dli_panel'].copy()
            trade_data = self.base_data['trade_flow'].copy()
            
            # 筛选与美国相关的贸易 (适配实际数据格式)
            us_trade = trade_data[
                (trade_data['reporter'] == 'USA') | 
                (trade_data['partner'] == 'USA')
            ].copy()
            
            if len(us_trade) == 0:
                logger.warning("⚠️ 未找到美国相关贸易数据")
                return None
            
            # 计算贸易份额 (适配实际格式)
            us_trade['partner_country'] = np.where(
                us_trade['reporter'] == 'USA',
                us_trade['partner'],
                us_trade['reporter']
            )
            
            # 确定美国角色 (Export 或 Import from 美国)
            us_trade['us_role'] = np.where(
                (us_trade['reporter'] == 'USA') & (us_trade['flow'] == 'X'),
                'exporter',
                'importer'
            )
            
            # 计算真实的进口份额
            logger.info("   计算真实贸易份额...")
            
            # 计算各国总进口额（从所有国家）
            total_imports = trade_data[trade_data['flow'] == 'M'].groupby(['year', 'reporter']).agg({
                'trade_value_raw_usd': 'sum'
            }).reset_index()
            total_imports.columns = ['year', 'country', 'total_imports']
            
            # 计算各国从美国的进口额
            us_imports = us_trade[
                (us_trade['partner'] == 'USA') & (us_trade['flow'] == 'M')
            ].groupby(['year', 'reporter']).agg({
                'trade_value_raw_usd': 'sum'
            }).reset_index()
            us_imports.columns = ['year', 'country', 'us_imports']
            
            # 合并计算真实进口份额
            trade_shares = total_imports.merge(us_imports, on=['year', 'country'], how='left')
            trade_shares['us_imports'] = trade_shares['us_imports'].fillna(0)
            trade_shares['import_share_from_us'] = trade_shares['us_imports'] / trade_shares['total_imports']
            trade_shares['import_share_from_us'] = trade_shares['import_share_from_us'].fillna(0).clip(0, 1)
            
            logger.info(f"   计算了 {len(trade_shares)} 个国家-年份的真实贸易份额")
            
            # 基于真实DLI数据构建Node-DLI_US
            logger.info("   基于真实DLI数据构建Node-DLI_US...")
            node_dli_records = []
            
            for _, trade_row in trade_shares.iterrows():
                year = trade_row['year']
                country = trade_row['country']
                s_imp = trade_row['import_share_from_us']
                
                # 查找对应的DLI数据
                # DLI_{US->i,t}: 美国出口到该国的锁定指数
                dli_us_to_i = dli_data[
                    (dli_data['year'] == year) &
                    (dli_data['us_partner'] == country) &
                    (dli_data['us_role'] == 'exporter')
                ]['dli_score_adjusted'].mean()
                
                # DLI_{i->US,t}: 该国出口到美国的锁定指数  
                dli_i_to_us = dli_data[
                    (dli_data['year'] == year) &
                    (dli_data['us_partner'] == country) &
                    (dli_data['us_role'] == 'importer')
                ]['dli_score_adjusted'].mean()
                
                # 应用Node-DLI公式
                if pd.isna(dli_us_to_i):
                    dli_us_to_i = 0
                if pd.isna(dli_i_to_us):
                    dli_i_to_us = 0
                
                # NodeDLI^US_{i,t} = s^{imp}_{i,US,t} × DLI_{US→i,t} + (1-s^{imp}_{i,US,t}) × DLI_{i→US,t}
                node_dli_us = s_imp * dli_us_to_i + (1 - s_imp) * dli_i_to_us
                
                node_dli_records.append({
                    'year': year,
                    'country': country,
                    'node_dli_us': node_dli_us,
                    'import_share_from_us': s_imp,
                    'dli_us_to_i': dli_us_to_i,
                    'dli_i_to_us': dli_i_to_us,
                    'us_imports': trade_row['us_imports'],
                    'total_imports': trade_row['total_imports']
                })
            
            node_dli_df = pd.DataFrame(node_dli_records)
            
            # 数据质量检查
            non_zero_dli = node_dli_df[node_dli_df['node_dli_us'] > 0]
            logger.info(f"   有效DLI记录: {len(non_zero_dli)}/{len(node_dli_df)}")
            logger.info(f"   Node-DLI_US范围: {node_dli_df['node_dli_us'].min():.3f} - {node_dli_df['node_dli_us'].max():.3f}")
            logger.info(f"   平均进口份额: {node_dli_df['import_share_from_us'].mean():.3f}")
            
            # 只保留核心变量用于合并
            final_node_dli = node_dli_df[['year', 'country', 'node_dli_us', 'import_share_from_us']].copy()
            
            # 保存中间结果
            output_path = self.temp_data_dir / "node_dli_us.csv"
            node_dli_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ 真实Node-DLI_US构建完成: {len(final_node_dli)} 行记录")
            return final_node_dli
            
        except Exception as e:
            logger.error(f"❌ Node-DLI_US构建失败: {str(e)}")
            return None
    
    def _construct_vul_us(self) -> Optional[pd.DataFrame]:
        """构建 Vul_US (美国锚定脆弱性指数)"""
        logger.info("   构建 Vul_US...")
        
        try:
            if 'trade_flow' not in self.base_data:
                logger.warning("⚠️ 缺少贸易数据，跳过Vul_US构建")
                return None
            
            trade_data = self.base_data['trade_flow'].copy()
            
            # 计算各国的进口依赖度和多样化程度 (适配实际格式)
            import_data = trade_data[trade_data['flow'] == 'M'].copy()  # 只要进口数据
            import_data = import_data.groupby(['year', 'reporter', 'partner']).agg({
                'trade_value_raw_usd': 'sum'
            }).reset_index()
            
            # 计算HHI指数（简化版）
            total_imports = import_data.groupby(['year', 'reporter']).agg({
                'trade_value_raw_usd': 'sum'
            }).reset_index().rename(columns={'trade_value_raw_usd': 'total_imports'})
            
            import_data = import_data.merge(total_imports, on=['year', 'reporter'])
            import_data['import_share'] = import_data['trade_value_raw_usd'] / import_data['total_imports']
            
            # 计算HHI
            hhi_data = import_data.groupby(['year', 'reporter']).apply(
                lambda x: (x['import_share'] ** 2).sum()
            ).reset_index(name='hhi_imports')
            
            # 计算对美依赖度
            us_imports = import_data[import_data['partner'] == 'USA'].copy()
            us_imports = us_imports.rename(columns={
                'import_share': 'us_import_share',
                'reporter': 'country'
            })[['year', 'country', 'us_import_share']]
            
            # 合并数据计算Vul_US
            vul_data = hhi_data.merge(us_imports, left_on=['year', 'reporter'], 
                                    right_on=['year', 'country'], how='left')
            
            vul_data['us_import_share'] = vul_data['us_import_share'].fillna(0)
            vul_data['vul_us'] = vul_data['us_import_share'] * vul_data['hhi_imports']
            
            vul_df = vul_data[['year', 'country', 'vul_us', 'us_import_share', 'hhi_imports']].copy()
            vul_df = vul_df.dropna()
            
            # 保存中间结果
            output_path = self.temp_data_dir / "vul_us.csv"
            vul_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Vul_US构建完成: {len(vul_df)} 行记录")
            return vul_df
            
        except Exception as e:
            logger.error(f"❌ Vul_US构建失败: {str(e)}")
            return None

    
    def _construct_us_prod_shock(self) -> Optional[pd.DataFrame]:
        """构建综合产量冲击指数 (原油+天然气)"""
        logger.info("   构建综合US_ProdShock（原油+天然气）...")
        
        try:
            if not HAS_REQUESTS:
                logger.warning("⚠️ requests库不可用，无法获取EIA数据")
                return None
            
            # 使用提供的EIA API密钥
            eia_api_key = "kCKMXECZ7EZxHpYPXekyOhSdccpNc85aeOpDGIwm"
            logger.info(f"   使用EIA API Key: {eia_api_key[:8]}...")
            
            # 第1步：获取美国原油产量数据
            logger.info("   获取美国原油产量数据...")
            oil_url = "https://api.eia.gov/v2/petroleum/crd/crpdn/data/"
            oil_params = {
                'api_key': eia_api_key,
                'frequency': 'annual',
                'data[0]': 'value',
                'start': '2000',
                'end': '2023',
                'length': 1000
            }
            
            oil_response = requests.get(oil_url, params=oil_params, timeout=30)
            oil_data = None
            if oil_response.status_code == 200:
                oil_json = oil_response.json()
                if 'response' in oil_json and 'data' in oil_json['response']:
                    oil_df = pd.DataFrame(oil_json['response']['data'])
                    # 筛选美国数据
                    us_oil = oil_df[oil_df['area-name'].str.contains('USA', na=False)].copy()
                    us_oil['year'] = us_oil['period'].astype(int)
                    us_oil['value'] = pd.to_numeric(us_oil['value'], errors='coerce')
                    oil_data = us_oil.groupby('year')['value'].sum().reset_index()
                    oil_data.columns = ['year', 'us_production_oil']
                    logger.info(f"   原油数据: {len(oil_data)} 年，范围 {oil_data['year'].min()}-{oil_data['year'].max()}")
            
            # 第2步：获取美国天然气产量数据
            logger.info("   获取美国天然气产量数据...")
            gas_url = "https://api.eia.gov/v2/natural-gas/prod/sum/data/"
            gas_params = {
                'api_key': eia_api_key,
                'frequency': 'annual',
                'data[0]': 'value',
                'start': '2000',
                'end': '2023',
                'length': 500
            }
            
            gas_response = requests.get(gas_url, params=gas_params, timeout=30)
            gas_data = None
            if gas_response.status_code == 200:
                gas_json = gas_response.json()
                if 'response' in gas_json and 'data' in gas_json['response']:
                    gas_df = pd.DataFrame(gas_json['response']['data'])
                    gas_df['year'] = gas_df['period'].astype(int)
                    gas_df['value'] = pd.to_numeric(gas_df['value'], errors='coerce')
                    gas_data = gas_df.groupby('year')['value'].sum().reset_index()
                    gas_data.columns = ['year', 'us_production_gas']
                    logger.info(f"   天然气数据: {len(gas_data)} 年，范围 {gas_data['year'].min()}-{gas_data['year'].max()}")
            
            # 检查是否成功获取两种数据
            if oil_data is None or gas_data is None:
                logger.warning("⚠️ EIA API数据获取不完整，US_ProdShock变量将标记为缺失值")
                return None
            
            # 第3步：合并数据
            combined_data = oil_data.merge(gas_data, on='year', how='outer').sort_values('year')
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 10:
                logger.warning("⚠️ 合并后数据点过少，无法计算可靠的HP滤波冲击")
                return None
            
            # 第4步：分别计算HP滤波冲击
            logger.info("   计算HP滤波冲击...")
            if HAS_FILTERING:
                # 原油冲击
                oil_cycle, oil_trend = hpfilter(combined_data['us_production_oil'].values, lamb=100)
                # 天然气冲击
                gas_cycle, gas_trend = hpfilter(combined_data['us_production_gas'].values, lamb=100)
            else:
                # 简化冲击计算
                oil_cycle = (combined_data['us_production_oil'] - 
                           combined_data['us_production_oil'].rolling(3).mean()).fillna(0).values
                gas_cycle = (combined_data['us_production_gas'] - 
                           combined_data['us_production_gas'].rolling(3).mean()).fillna(0).values
            
            # 第5步：标准化冲击序列
            z_shock_oil = (oil_cycle - oil_cycle.mean()) / oil_cycle.std()
            z_shock_gas = (gas_cycle - gas_cycle.mean()) / gas_cycle.std()
            
            # 第6步：等权重合成综合冲击指数
            us_prod_shock = 0.5 * z_shock_oil + 0.5 * z_shock_gas
            
            # 构建最终数据框
            shock_df = pd.DataFrame({
                'year': combined_data['year'].values,
                'us_production_oil': combined_data['us_production_oil'].values,
                'us_production_gas': combined_data['us_production_gas'].values,
                'us_prod_shock': us_prod_shock
            })
            
            # 保存中间结果
            output_path = self.temp_data_dir / "us_prod_shock.csv"
            shock_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ 综合US_ProdShock构建完成: {len(shock_df)} 年数据")
            logger.info(f"   原油产量范围: {shock_df['us_production_oil'].min():.0f} - {shock_df['us_production_oil'].max():.0f}")
            logger.info(f"   天然气产量范围: {shock_df['us_production_gas'].min():.0f} - {shock_df['us_production_gas'].max():.0f}")
            logger.info(f"   综合冲击范围: {shock_df['us_prod_shock'].min():.3f} - {shock_df['us_prod_shock'].max():.3f}")
            
            return shock_df
                
        except Exception as e:
            logger.error(f"❌ 综合US_ProdShock构建失败: {str(e)}")
            return None
    
    def create_analytical_panel(self) -> pd.DataFrame:
        """
        创建最终分析面板
        
        Returns:
            整合后的分析面板DataFrame
        """
        logger.info("🔗 开始创建最终分析面板...")
        
        # 检查必要数据并构建基础面板
        if self.macro_data is not None:
            # 从宏观数据开始构建面板
            panel = self.macro_data.copy()
            
            # 添加country列（从country_name提取ISO3代码）
            if 'country_name' in panel.columns and 'country' not in panel.columns:
                panel['country'] = panel['country_name'].str[:3].str.upper()
                logger.info(f"   从country_name提取country列: {panel['country'].nunique()} 个国家")
            
            logger.info(f"   基于宏观数据构建起始面板: {len(panel)} 行")
        else:
            # 宏观数据缺失，创建基础框架面板
            logger.warning("⚠️ 宏观数据缺失，创建基础国家-年份面板框架")
            countries = self._get_country_list()
            years = list(range(2000, 2025))
            
            # 创建国家-年份笛卡尔积
            country_year_pairs = []
            for country in countries:
                for year in years:
                    country_year_pairs.append({'country': country, 'year': year})
            
            panel = pd.DataFrame(country_year_pairs)
            logger.info(f"   创建基础面板框架: {len(panel)} 行")
        
        # 标准化国家名称列
        if 'country_name' in panel.columns:
            panel['country'] = panel['country_name']
        
        logger.info(f"   起始面板: {len(panel)} 行")
        
        # 逐步合并其他数据
        merge_count = 0
        
        # 合并核心变量
        for var_name, var_data in self.core_variables.items():
            if var_data is not None and len(var_data) > 0:
                try:
                    before_len = len(panel)
                    
                    # 特殊处理US_ProdShock - 只有年份数据，需要为所有国家复制
                    if var_name == 'us_prod_shock':
                        # US_ProdShock数据只有年份，为所有国家复制
                        panel = panel.merge(var_data, on='year', how='left')
                    else:
                        # 其他变量按year和country合并
                        panel = panel.merge(var_data, on=['year', 'country'], how='left')
                    
                    after_len = len(panel)
                    
                    if after_len == before_len:
                        merge_count += 1
                        logger.info(f"   ✅ 合并 {var_name}: {len(var_data)} 行")
                    else:
                        logger.warning(f"   ⚠️ 合并 {var_name} 改变了面板行数: {before_len} -> {after_len}")
                        
                except Exception as e:
                    logger.warning(f"   ❌ 无法合并 {var_name}: {str(e)}")
        
        # 合并网络指标
        if 'node_metrics' in self.base_data:
            try:
                node_metrics = self.base_data['node_metrics'].copy()
                
                # 标准化列名 - 如果有country_code，重命名为country
                if 'country_code' in node_metrics.columns:
                    node_metrics['country'] = node_metrics['country_code']
                
                before_len = len(panel)
                panel = panel.merge(node_metrics, on=['year', 'country'], how='left')
                after_len = len(panel)
                
                if after_len == before_len:
                    merge_count += 1
                    logger.info(f"   ✅ 合并网络指标: {len(node_metrics)} 行")
                else:
                    logger.warning(f"   ⚠️ 合并网络指标改变了面板行数: {before_len} -> {after_len}")
                    
            except Exception as e:
                logger.warning(f"   ❌ 无法合并网络指标: {str(e)}")
        
        # 数据清洗和最终处理
        panel = self._clean_final_panel(panel)
        
        # 保存最终面板
        output_path = self.base_dir / "data" / "processed_data" / "analytical_panel.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
        
        # 同时保存到模块输出目录
        module_output_path = self.output_dir / "analytical_panel.csv"
        panel.to_csv(module_output_path, index=False)
        
        self.final_panel = panel
        
        logger.info(f"✅ 最终分析面板创建完成:")
        logger.info(f"   行数: {len(panel)}")
        logger.info(f"   列数: {len(panel.columns)}")
        logger.info(f"   年份范围: {panel['year'].min()}-{panel['year'].max()}")
        logger.info(f"   国家数量: {panel['country'].nunique()}")
        logger.info(f"   成功合并: {merge_count} 个数据集")
        logger.info(f"   保存至: {output_path}")
        
        return panel
    
    def _clean_final_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """清洗最终面板数据"""
        logger.info("   清洗最终面板数据...")
        
        # 删除重复行
        initial_len = len(panel)
        panel = panel.drop_duplicates(subset=['year', 'country'])
        if len(panel) != initial_len:
            logger.info(f"   删除重复行: {initial_len} -> {len(panel)}")
        
        # 确保年份为整数
        panel['year'] = panel['year'].astype(int)
        
        # 限制年份范围
        panel = panel[(panel['year'] >= 2000) & (panel['year'] <= 2024)]
        
        # 清理无效值
        numeric_columns = panel.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 替换无穷大值为NaN
            panel[col] = panel[col].replace([np.inf, -np.inf], np.nan)
        
        # 按年份和国家排序
        panel = panel.sort_values(['year', 'country']).reset_index(drop=True)
        
        return panel
    
    def create_data_dictionary(self) -> None:
        """创建数据字典"""
        logger.info("📖 创建数据字典...")
        
        if self.final_panel is None:
            logger.warning("⚠️ 最终面板未创建，无法生成数据字典")
            return
        
        # 构建数据字典内容
        dictionary_content = f"""# 分析面板数据字典
## Analytical Panel Data Dictionary

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**模块**: 08_variable_construction v1.0  
**总行数**: {len(self.final_panel):,}  
**总列数**: {len(self.final_panel.columns)}  
**年份范围**: {self.final_panel['year'].min()}-{self.final_panel['year'].max()}  
**国家数量**: {self.final_panel['country'].nunique()}

---

## 变量详细说明

### 基础标识变量
"""
        
        # 基础变量
        basic_vars = {
            'year': '年份 (整数)',
            'country': '国家ISO3代码 (字符串)',
            'country_name': '国家全名 (字符串，来源：World Bank)'
        }
        
        for var, desc in basic_vars.items():
            if var in self.final_panel.columns:
                dictionary_content += f"- **{var}**: {desc}\n"
        
        # 宏观经济变量
        dictionary_content += "\n### 宏观经济控制变量 (来源：World Bank WDI API)\n"
        macro_vars = {
            'gdp_current_usd': 'GDP，现价美元 (NY.GDP.MKTP.CD)',
            'population_total': '总人口数 (SP.POP.TOTL)',
            'trade_openness_gdp_pct': '贸易开放度，占GDP百分比 (NE.TRD.GNFS.ZS)',
            'log_gdp': 'GDP的自然对数',
            'log_population': '人口的自然对数'
        }
        
        for var, desc in macro_vars.items():
            if var in self.final_panel.columns:
                dictionary_content += f"- **{var}**: {desc}\n"
        
        # 核心研究变量
        dictionary_content += "\n### 核心研究变量 (本模块构建)\n"
        core_vars = {
            'node_dli_us': 'Node-DLI_US: 美国锚定动态锁定指数，基于04_dli_analysis的边级DLI聚合',
            'vul_us': 'Vul_US: 美国锚定脆弱性指数，基于进口份额×HHI指数',
            'ovi_gas': 'OVI_gas: 天然气物理冗余指数 (主指标)，基于LNG接收站和管道容量',
            'ovi_oil': 'OVI_oil: 石油物理冗余指数 (稳健性检验指标)，基于炼油厂和管道容量',
            'us_prod_shock': 'US_ProdShock: 美国页岩油气产量冲击，HP滤波后的周期成分'
        }
        
        for var, desc in core_vars.items():
            if var in self.final_panel.columns:
                dictionary_content += f"- **{var}**: {desc}\n"
        
        # 网络指标变量
        dictionary_content += "\n### 网络拓扑指标 (来源：03_metrics)\n"
        network_vars = {
            'betweenness_centrality': '介数中心性，衡量节点在网络中的桥梁作用',
            'closeness_centrality': '接近中心性，衡量节点到其他节点的平均距离',
            'eigenvector_centrality': '特征向量中心性，考虑邻居重要性的中心性',
            'degree_centrality': '度中心性，衡量节点的连接数量'
        }
        
        for var, desc in network_vars.items():
            if var in self.final_panel.columns:
                dictionary_content += f"- **{var}**: {desc}\n"
        
        # 统计摘要
        dictionary_content += "\n---\n\n## 数据质量摘要\n\n"
        
        # 缺失值统计
        missing_stats = self.final_panel.isnull().sum()
        missing_pct = (missing_stats / len(self.final_panel) * 100).round(2)
        
        dictionary_content += "### 缺失值统计\n\n"
        dictionary_content += "| 变量名 | 缺失值数量 | 缺失率(%) |\n"
        dictionary_content += "|--------|------------|----------|\n"
        
        for var in self.final_panel.columns:
            if missing_stats[var] > 0:
                dictionary_content += f"| {var} | {missing_stats[var]} | {missing_pct[var]}% |\n"
        
        # 数值变量统计
        numeric_cols = self.final_panel.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            dictionary_content += "\n### 数值变量基础统计\n\n"
            stats_df = self.final_panel[numeric_cols].describe()
            dictionary_content += stats_df.round(4).to_markdown()
        
        # 数据来源说明
        dictionary_content += "\n\n---\n\n## 数据来源与构建方法\n\n"
        dictionary_content += """
1. **宏观经济数据**: 通过wbdata包从世界银行WDI数据库获取
2. **贸易网络数据**: 基于01_data_processing模块的清洗贸易流数据
3. **网络拓扑指标**: 基于03_metrics模块计算的中心性指标
4. **DLI指标**: 基于04_dli_analysis模块的边级动态锁定指数
5. **物理基础设施数据**: 手动收集的LNG接收站和管道容量数据
6. **美国产量数据**: 通过EIA API获取的美国石油天然气产量数据

## 使用建议

1. **因变量选择**: 建议使用vul_us作为主要的脆弱性指标
2. **解释变量**: node_dli_us和ovi是核心解释变量
3. **控制变量**: 建议控制log_gdp, log_population, trade_openness_gdp_pct
4. **工具变量**: us_prod_shock可作为外生冲击的工具变量
5. **网络控制**: 可加入网络中心性指标作为额外控制

---

*本数据字典由08_variable_construction模块自动生成*  
*Energy Network Analysis Team*
"""
        
        # 保存数据字典
        dict_path = self.output_dir / "data_dictionary.md"
        with open(dict_path, 'w', encoding='utf-8') as f:
            f.write(dictionary_content)
        
        logger.info(f"✅ 数据字典创建完成: {dict_path}")
    
    def run_full_pipeline(self) -> None:
        """运行完整的变量构建流水线"""
        logger.info("🚀 开始运行完整的变量构建流水线...")
        
        try:
            # 步骤1: 搜集宏观控制变量
            logger.info("\n" + "="*50)
            logger.info("步骤1: 搜集宏观经济控制变量")
            logger.info("="*50)
            self.fetch_macro_controls()
            
            # 步骤2: 加载基础数据
            logger.info("\n" + "="*50)
            logger.info("步骤2: 加载基础数据")
            logger.info("="*50)
            self.load_base_data()
            
            # 步骤3: 构建核心变量
            logger.info("\n" + "="*50)
            logger.info("步骤3: 构建核心变量")
            logger.info("="*50)
            self.construct_core_variables()
            
            # 步骤4: 创建最终面板
            logger.info("\n" + "="*50)
            logger.info("步骤4: 创建最终分析面板")
            logger.info("="*50)
            self.create_analytical_panel()
            
            # 步骤5: 生成数据字典
            logger.info("\n" + "="*50)
            logger.info("步骤5: 生成数据字典")
            logger.info("="*50)
            self.create_data_dictionary()
            
            logger.info("\n" + "="*60)
            logger.info("🎉 变量构建流水线执行完成！")
            logger.info("="*60)
            logger.info(f"✅ 最终输出:")
            logger.info(f"   - 分析面板: analytical_panel.csv ({len(self.final_panel)} 行)")
            logger.info(f"   - 数据字典: data_dictionary.md")
            logger.info(f"   - 中间文件: {self.temp_data_dir}")
            logger.info(f"   - 输出目录: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {str(e)}")
            raise

def main():
    """主函数：演示变量构建流水线"""
    print("🏗️ 08_variable_construction - 超级数据工厂")
    print("="*60)
    
    try:
        # 初始化变量构建器
        constructor = VariableConstructor()
        
        # 运行完整流水线
        constructor.run_full_pipeline()
        
        print("\n✅ 变量构建模块执行成功！")
        print("📄 查看输出文件:")
        print(f"   - {constructor.output_dir / 'analytical_panel.csv'}")
        print(f"   - {constructor.output_dir / 'data_dictionary.md'}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()