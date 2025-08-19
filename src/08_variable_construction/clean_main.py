#!/usr/bin/env python3
"""
清晰版主构建器 v3.0 - 回归初心版
===============================

核心目标:
根据research_outline.md，构建以下4个核心变量：
1. Node-DLI_US: 美国锚定动态锁定指数 (来自04_dli_analysis)
2. Vul_US: 美国锚定脆弱性指数 (基于贸易份额×HHI)
3. OVI: 天然气物理冗余指数 (LNG+管道/消费)
4. US_ProdShock: 美国产量冲击 (页岩革命外生冲击)

输出:
- analytical_panel.csv: 国别-年度面板数据 (500行×核心变量)
- data_dictionary.md: 数据字典

作者: Energy Network Analysis Team  
版本: v3.0 - 回归初心版
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import requests

# 导入简化的天然气OVI构建器
from simple_gas_ovi_builder import SimpleGasOVIBuilder

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_variable_construction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CleanVariableConstructor:
    """清晰版变量构建器 - 只构建研究大纲要求的核心变量"""
    
    def __init__(self, base_dir: str = None):
        """初始化构建器"""
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.data_dir = self.base_dir / "data"
        self.output_dir = Path(__file__).parent / "outputs"
        self.temp_data_dir = Path(__file__).parent / "08data"
        
        # 创建必要目录
        self.output_dir.mkdir(exist_ok=True)
        self.temp_data_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.macro_data = None
        self.base_data = {}
        self.core_variables = {}
        self.final_panel = None
        
        logger.info("🏗️ 清晰版变量构建器初始化完成")
        logger.info(f"   项目根目录: {self.base_dir}")
        logger.info(f"   输出目录: {self.output_dir}")
    
    def load_cached_macro_data(self) -> pd.DataFrame:
        """加载缓存的宏观数据"""
        logger.info("🌍 加载宏观经济控制变量...")
        
        cache_path = self.temp_data_dir / "macro_controls.csv"
        if cache_path.exists():
            try:
                macro_data = pd.read_csv(cache_path)
                macro_data['year'] = pd.to_datetime(macro_data['year']).dt.year
                
                logger.info(f"✅ 从缓存加载宏观数据: {len(macro_data)} 行记录")
                logger.info(f"   数据范围: {macro_data['year'].min()}-{macro_data['year'].max()}")
                
                self.macro_data = macro_data
                return macro_data
                
            except Exception as e:
                logger.warning(f"⚠️ 缓存数据加载失败: {str(e)}")
        
        logger.warning("⚠️ 未找到宏观数据缓存，将创建基础模板")
        return None
    
    def load_base_data(self) -> Dict[str, pd.DataFrame]:
        """加载基础数据 (01, 03, 04模块输出)"""
        logger.info("📁 开始加载基础数据...")
        
        base_data = {}
        
        # 加载01模块输出 - 贸易流数据
        trade_data_files = list((self.base_dir / "data" / "processed_data").glob("cleaned_energy_trade_*.csv"))
        
        if trade_data_files:
            try:
                trade_data_list = []
                for file_path in sorted(trade_data_files):
                    yearly_data = pd.read_csv(file_path)
                    trade_data_list.append(yearly_data)
                
                trade_data = pd.concat(trade_data_list, ignore_index=True)
                base_data['trade_flow'] = trade_data
                logger.info(f"✅ 加载贸易流数据: 合并 {len(trade_data_files)} 个文件 ({len(trade_data)} 行)")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载贸易数据: {str(e)}")
        
        # 加载03模块输出 - 网络指标
        metrics_data_path = self.base_dir / "src" / "03_metrics"
        
        node_metrics_path = metrics_data_path / "node_centrality_metrics.csv"
        if node_metrics_path.exists():
            try:
                node_metrics = pd.read_csv(node_metrics_path)
                base_data['node_metrics'] = node_metrics
                logger.info(f"✅ 加载节点指标: {len(node_metrics)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载节点指标: {str(e)}")
        
        # 加载04模块输出 - DLI数据
        dli_data_path = self.base_dir / "src" / "04_dli_analysis" / "dli_panel_data.csv"
        if dli_data_path.exists():
            try:
                dli_data = pd.read_csv(dli_data_path)
                base_data['dli_panel'] = dli_data
                logger.info(f"✅ 加载DLI面板数据: {len(dli_data)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载DLI数据: {str(e)}")
        
        self.base_data = base_data
        logger.info(f"✅ 基础数据加载完成，共 {len(base_data)} 个数据集")
        
        return base_data
    
    def construct_node_dli_us(self) -> Optional[pd.DataFrame]:
        """构建 Node-DLI_US (美国锚定动态锁定指数)"""
        logger.info("   构建 Node-DLI_US...")
        
        try:
            if 'dli_panel' not in self.base_data or 'trade_flow' not in self.base_data:
                logger.warning("⚠️ 缺少DLI或贸易数据，跳过Node-DLI_US构建")
                return None
            
            dli_data = self.base_data['dli_panel'].copy()
            trade_data = self.base_data['trade_flow'].copy()
            
            # 筛选与美国相关的贸易
            us_trade = trade_data[
                (trade_data['reporter'] == 'USA') | 
                (trade_data['partner'] == 'USA')
            ].copy()
            
            if len(us_trade) == 0:
                logger.warning("⚠️ 未找到美国相关贸易数据")
                return None
            
            # 计算贸易份额
            us_trade['partner_country'] = np.where(
                us_trade['reporter'] == 'USA',
                us_trade['partner'],
                us_trade['reporter']
            )
            
            # 计算各国总进口额
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
            node_dli_records = []
            
            for _, trade_row in trade_shares.iterrows():
                year = trade_row['year']
                country = trade_row['country']
                s_imp = trade_row['import_share_from_us']
                
                # 查找对应的DLI数据
                dli_us_to_i = dli_data[
                    (dli_data['year'] == year) &
                    (dli_data['us_partner'] == country) &
                    (dli_data['us_role'] == 'exporter')
                ]['dli_score_adjusted'].mean()
                
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
                    'import_share_from_us': s_imp
                })
            
            node_dli_df = pd.DataFrame(node_dli_records)
            
            # 数据质量检查
            non_zero_dli = node_dli_df[node_dli_df['node_dli_us'] > 0]
            logger.info(f"   有效DLI记录: {len(non_zero_dli)}/{len(node_dli_df)}")
            logger.info(f"   Node-DLI_US范围: {node_dli_df['node_dli_us'].min():.3f} - {node_dli_df['node_dli_us'].max():.3f}")
            
            # 保存中间结果
            output_path = self.temp_data_dir / "node_dli_us_clean.csv"
            node_dli_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Node-DLI_US构建完成: {len(node_dli_df)} 行记录")
            return node_dli_df[['year', 'country', 'node_dli_us', 'import_share_from_us']].copy()
            
        except Exception as e:
            logger.error(f"❌ Node-DLI_US构建失败: {str(e)}")
            return None
    
    def construct_vul_us(self) -> Optional[pd.DataFrame]:
        """构建 Vul_US (美国锚定脆弱性指数)"""
        logger.info("   构建 Vul_US...")
        
        try:
            if 'trade_flow' not in self.base_data:
                logger.warning("⚠️ 缺少贸易数据，跳过Vul_US构建")
                return None
            
            trade_data = self.base_data['trade_flow'].copy()
            
            # 计算各国的进口依赖度和多样化程度
            import_data = trade_data[trade_data['flow'] == 'M'].copy()
            import_data = import_data.groupby(['year', 'reporter', 'partner']).agg({
                'trade_value_raw_usd': 'sum'
            }).reset_index()
            
            # 计算HHI指数
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
            output_path = self.temp_data_dir / "vul_us_clean.csv"
            vul_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Vul_US构建完成: {len(vul_df)} 行记录")
            return vul_df
            
        except Exception as e:
            logger.error(f"❌ Vul_US构建失败: {str(e)}")
            return None
    
    def construct_gas_ovi(self) -> Optional[pd.DataFrame]:
        """构建天然气OVI"""
        logger.info("   构建天然气OVI...")
        
        try:
            # 使用简化的天然气OVI构建器
            builder = SimpleGasOVIBuilder(self.temp_data_dir)
            ovi_data = builder.build_gas_ovi()
            
            if len(ovi_data) > 0:
                # 只保留核心列用于合并
                result = ovi_data[['country', 'year', 'ovi_gas']].copy()
                logger.info(f"✅ 天然气OVI构建完成: {len(result)} 行记录")
                return result
            else:
                logger.warning("⚠️ 未能构建天然气OVI数据")
                return None
                
        except Exception as e:
            logger.error(f"❌ 天然气OVI构建失败: {str(e)}")
            return None
    
    def construct_us_prod_shock(self) -> Optional[pd.DataFrame]:
        """构建综合产量冲击指数 (原油+天然气)"""
        logger.info("   构建综合US_ProdShock（原油+天然气）...")
        
        try:
            # 使用EIA API获取数据
            eia_api_key = "kCKMXECZ7EZxHpYPXekyOhSdccpNc85aeOpDGIwm"
            logger.info(f"   使用EIA API Key: {eia_api_key[:8]}...")
            
            # 获取美国原油产量数据
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
                    us_oil = oil_df[oil_df['area-name'].str.contains('USA', na=False)].copy()
                    us_oil['year'] = us_oil['period'].astype(int)
                    us_oil['value'] = pd.to_numeric(us_oil['value'], errors='coerce')
                    oil_data = us_oil.groupby('year')['value'].sum().reset_index()
                    oil_data.columns = ['year', 'us_production_oil']
                    logger.info(f"   原油数据: {len(oil_data)} 年")
            
            # 获取美国天然气产量数据
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
                    logger.info(f"   天然气数据: {len(gas_data)} 年")
            
            if oil_data is None or gas_data is None:
                logger.warning("⚠️ EIA API数据获取不完整")
                return None
            
            # 合并数据
            combined_data = oil_data.merge(gas_data, on='year', how='outer').sort_values('year')
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 10:
                logger.warning("⚠️ 合并后数据点过少")
                return None
            
            # 计算HP滤波冲击
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                oil_cycle, oil_trend = hpfilter(combined_data['us_production_oil'].values, lamb=100)
                gas_cycle, gas_trend = hpfilter(combined_data['us_production_gas'].values, lamb=100)
            except ImportError:
                # 简化冲击计算
                oil_cycle = (combined_data['us_production_oil'] - 
                           combined_data['us_production_oil'].rolling(3).mean()).fillna(0).values
                gas_cycle = (combined_data['us_production_gas'] - 
                           combined_data['us_production_gas'].rolling(3).mean()).fillna(0).values
            
            # 标准化冲击序列
            z_shock_oil = (oil_cycle - oil_cycle.mean()) / oil_cycle.std()
            z_shock_gas = (gas_cycle - gas_cycle.mean()) / gas_cycle.std()
            
            # 等权重合成综合冲击指数
            us_prod_shock = 0.5 * z_shock_oil + 0.5 * z_shock_gas
            
            # 构建最终数据框
            shock_df = pd.DataFrame({
                'year': combined_data['year'].values,
                'us_production_oil': combined_data['us_production_oil'].values,
                'us_production_gas': combined_data['us_production_gas'].values,
                'us_prod_shock': us_prod_shock
            })
            
            # 保存中间结果
            output_path = self.temp_data_dir / "us_prod_shock_clean.csv"
            shock_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ 综合US_ProdShock构建完成: {len(shock_df)} 年数据")
            
            return shock_df
                
        except Exception as e:
            logger.error(f"❌ 综合US_ProdShock构建失败: {str(e)}")
            return None
    
    def construct_core_variables(self) -> Dict[str, pd.DataFrame]:
        """构建核心变量"""
        logger.info("⚙️ 开始构建核心变量...")
        
        core_vars = {}
        
        # 1. 构建 Node-DLI_US
        node_dli_us = self.construct_node_dli_us()
        if node_dli_us is not None:
            core_vars['node_dli_us'] = node_dli_us
        
        # 2. 构建 Vul_US
        vul_us = self.construct_vul_us()
        if vul_us is not None:
            core_vars['vul_us'] = vul_us
        
        # 3. 构建天然气OVI
        gas_ovi = self.construct_gas_ovi()
        if gas_ovi is not None:
            core_vars['ovi_gas'] = gas_ovi
        
        # 4. 构建 US产量冲击
        us_shock = self.construct_us_prod_shock()
        if us_shock is not None:
            core_vars['us_prod_shock'] = us_shock
        
        self.core_variables = core_vars
        logger.info(f"✅ 核心变量构建完成，共 {len(core_vars)} 个变量")
        
        return core_vars
    
    def create_analytical_panel(self) -> pd.DataFrame:
        """创建最终分析面板"""
        logger.info("🔗 开始创建最终分析面板...")
        
        # 从宏观数据开始构建面板
        if self.macro_data is not None:
            panel = self.macro_data.copy()
            if 'country_name' in panel.columns and 'country' not in panel.columns:
                panel['country'] = panel['country_name'].str[:3].str.upper()
            logger.info(f"   基于宏观数据构建起始面板: {len(panel)} 行")
        else:
            # 创建基础框架面板
            logger.info("⚠️ 宏观数据缺失，创建基础国家-年份面板框架")
            countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'ITA', 'BRA', 'CAN',
                        'RUS', 'AUS', 'KOR', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'ARE']
            years = list(range(2000, 2025))
            
            country_year_pairs = []
            for country in countries:
                for year in years:
                    country_year_pairs.append({'country': country, 'year': year})
            
            panel = pd.DataFrame(country_year_pairs)
            logger.info(f"   创建基础面板框架: {len(panel)} 行")
        
        # 合并核心变量
        merge_count = 0
        for var_name, var_data in self.core_variables.items():
            if var_data is not None and len(var_data) > 0:
                try:
                    before_len = len(panel)
                    
                    if var_name == 'us_prod_shock':
                        # US_ProdShock只有年份数据，为所有国家复制
                        panel = panel.merge(var_data, on='year', how='left')
                    else:
                        # 其他变量按year和country合并
                        panel = panel.merge(var_data, on=['year', 'country'], how='left')
                    
                    after_len = len(panel)
                    
                    if after_len == before_len:
                        merge_count += 1
                        # 统计覆盖率
                        if var_name != 'us_prod_shock':
                            coverage = var_data.shape[0]
                            total_possible = len(panel)
                            logger.info(f"   ✅ 合并 {var_name}: {coverage} 行数据")
                        else:
                            logger.info(f"   ✅ 合并 {var_name}: {len(var_data)} 年数据（全面板复制）")
                            
                except Exception as e:
                    logger.warning(f"   ❌ 无法合并 {var_name}: {str(e)}")
        
        # 数据清洗
        panel = self._clean_final_panel(panel)
        
        # 保存最终面板
        output_path = self.output_dir / "analytical_panel.csv"
        panel.to_csv(output_path, index=False)
        
        self.final_panel = panel
        
        logger.info(f"✅ 最终分析面板创建完成:")
        logger.info(f"   行数: {len(panel)}")
        logger.info(f"   列数: {len(panel.columns)}")
        logger.info(f"   年份范围: {panel['year'].min()}-{panel['year'].max()}")
        logger.info(f"   国家数量: {panel['country'].nunique()}")
        logger.info(f"   成功合并: {merge_count} 个核心变量")
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
            panel[col] = panel[col].replace([np.inf, -np.inf], np.nan)
        
        # 按年份和国家排序
        panel = panel.sort_values(['year', 'country']).reset_index(drop=True)
        
        return panel
    
    def run_clean_pipeline(self) -> None:
        """运行清晰版变量构建流水线"""
        logger.info("🚀 开始运行清晰版变量构建流水线...")
        
        try:
            # 步骤1: 加载宏观控制变量
            logger.info("\n" + "="*50)
            logger.info("步骤1: 加载宏观经济控制变量")
            logger.info("="*50)
            self.load_cached_macro_data()
            
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
            
            logger.info("\n" + "="*60)
            logger.info("🎉 清晰版变量构建流水线执行完成！")
            logger.info("="*60)
            logger.info(f"✅ 最终输出:")
            logger.info(f"   - 分析面板: analytical_panel.csv ({len(self.final_panel)} 行)")
            logger.info(f"   - 中间文件: {self.temp_data_dir}")
            logger.info(f"   - 输出目录: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {str(e)}")
            raise

def main():
    """主函数：演示清晰版变量构建流水线"""
    print("🏗️ 08_variable_construction - 清晰版数据工厂 v3.0")
    print("="*60)
    
    try:
        # 初始化变量构建器
        constructor = CleanVariableConstructor()
        
        # 运行清晰版流水线
        constructor.run_clean_pipeline()
        
        print("\n✅ 清晰版变量构建模块执行成功！")
        print("📄 查看输出文件:")
        print(f"   - {constructor.output_dir / 'analytical_panel.csv'}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()