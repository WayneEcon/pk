#!/usr/bin/env python3
"""
主构建器 - 回归初心版
===============================

核心目标:
根据research_outline.md（重构版），构建以下4个核心变量：
1. Node-DLI_US: 美国锚定动态锁定指数 (来自04_dli_analysis)
2. HHI_imports: 进口来源多样化指数 (替代vul_us，避免构造内生性)
3. OVI: 天然气物理冗余指数 (LNG+管道/消费)
4. US_ProdShock: 美国产量冲击 - AR(2)残差方法 (页岩革命外生冲击)

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

# 导入新的US产量冲击构建器 (AR(2)残差方法)
from us_prod_shock_builder import USProdShockBuilder

# 导入新的宏观控制变量构建器
from macro_controls_builder import MacroControlsBuilder

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


class VariableConstructor:
    """变量构建器 - 只构建研究大纲要求的核心变量"""
    
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

    def _ensure_macro_data(self) -> None:
        """确保宏观经济数据可用，使用独立的构建器。"""
        logger.info("🌍 正在准备宏观经济控制变量...")
        
        # 使用独立的宏观控制变量构建器
        macro_builder = MacroControlsBuilder(
            data_dir=self.temp_data_dir,
            output_dir=self.output_dir
        )
        
        self.macro_data = macro_builder.build_macro_controls()
        if self.macro_data is None:
            logger.error("❌ 无法获取宏观数据，后续步骤可能失败。")
    
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
        dli_panel_path = self.base_dir / "src" / "04_dli_analysis" / "dli_panel_data.csv"
        if dli_panel_path.exists():
            try:
                dli_data = pd.read_csv(dli_panel_path)
                base_data['dli_panel'] = dli_data
                logger.info(f"✅ 加载DLI面板数据: {len(dli_data)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载DLI面板数据: {str(e)}")

        # 修复路径：从08data目录加载Node-DLI_US数据
        node_dli_us_path = self.temp_data_dir / "node_dli_us.csv"
        if node_dli_us_path.exists():
            try:
                node_dli_us_data = pd.read_csv(node_dli_us_path)
                base_data['node_dli_us'] = node_dli_us_data
                logger.info(f"✅ 加载Node-DLI_US数据: {len(node_dli_us_data)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载Node-DLI_US数据: {str(e)}")

        # 使用独立的HHI_imports构建器
        from hhi_imports_builder import HHIImportsBuilder
        hhi_builder = HHIImportsBuilder(self.base_dir, self.output_dir, self.temp_data_dir)
        hhi_imports_data = hhi_builder.load_hhi_imports()
        
        if hhi_imports_data is not None:
            base_data['hhi_imports'] = hhi_imports_data
        else:
            logger.warning("⚠️ 未能加载HHI_imports数据")
        
        self.base_data = base_data
        logger.info(f"✅ 基础数据加载完成，共 {len(base_data)} 个数据集")
        
        return base_data
    
    def construct_gas_ovi(self) -> Optional[pd.DataFrame]:
        """构建天然气OVI"""
        logger.info("   构建天然气OVI...")
        
        try:
            # 使用简化的天然气OVI构建器
            builder = SimpleGasOVIBuilder(self.temp_data_dir)
            ovi_data = builder.build_gas_ovi()
            
            if len(ovi_data) > 0:
                # OVI数据已经使用标准ISO代码，直接使用，不做转换
                
                # 只保留核心列用于合并
                result = ovi_data[['country', 'year', 'ovi_gas']].copy()
                
                logger.info(f"✅ 天然气OVI构建完成: {len(result)} 行记录")
                logger.info(f"   使用标准ISO代码: {sorted(result['country'].unique())}")
                return result
            else:
                logger.warning("⚠️ 未能构建天然气OVI数据")
                return None
                
        except Exception as e:
            logger.error(f"❌ 天然气OVI构建失败: {str(e)}")
            return None
    
    def construct_us_prod_shock(self) -> Optional[pd.DataFrame]:
        """构建美国天然气产量冲击 - AR(2)残差方法"""
        logger.info("   构建US_ProdShock（AR(2)残差方法）...")
        
        try:
            # 使用新的专门构建器
            shock_builder = USProdShockBuilder()
            
            # 构建AR(2)残差冲击
            shock_data = shock_builder.build_us_prod_shock(
                start_year=2000, 
                end_year=2024,
                save_path=self.output_dir / "us_prod_shock_ar2.csv"
            )
            
            if shock_data is None:
                logger.error("❌ AR(2)残差冲击构建失败")
                return None
            
            # 选择需要的列
            result_columns = ['year', 'us_gas_production', 'us_prod_shock']
            available_columns = [col for col in result_columns if col in shock_data.columns]
            
            result_df = shock_data[available_columns].copy()
            
            # 重命名列以保持向后兼容
            if 'us_gas_production' in result_df.columns:
                result_df = result_df.rename(columns={'us_gas_production': 'us_production_gas'})
            
            logger.info(f"✅ AR(2)残差US_ProdShock构建完成: {len(result_df)} 年数据")
            logger.info(f"   有效冲击值: {result_df['us_prod_shock'].notna().sum()} 个")
            logger.info(f"   缺失值(前2年): {result_df['us_prod_shock'].isna().sum()} 个")
            
            return result_df
                
        except Exception as e:
            logger.error(f"❌ AR(2)残差US_ProdShock构建失败: {str(e)}")
            return None

    def construct_price_quantity_variables(self) -> Optional[pd.DataFrame]:
        """构建价格代理变量P_it和数量增长变量g_it (基于UN Comtrade数据)"""
        logger.info("   构建价格代理变量P_it和数量增长变量g_it...")
        
        try:
            raw_data_dir = self.base_dir / "data" / "raw_data"
            if not raw_data_dir.exists():
                logger.error(f"❌ UN Comtrade原始数据目录不存在: {raw_data_dir}")
                return None
            
            # 获取所有年份的CSV文件
            csv_files = list(raw_data_dir.glob("*.csv"))
            if not csv_files:
                logger.error(f"❌ 未找到UN Comtrade CSV文件")
                return None
            
            logger.info(f"📁 找到 {len(csv_files)} 个年份的数据文件")
            
            all_gas_data = []
            
            for csv_file in sorted(csv_files):
                year = csv_file.stem
                if not year.isdigit() or int(year) < 2000 or int(year) > 2024:
                    continue
                    
                logger.info(f"   处理 {year} 年数据...")
                
                try:
                    # 读取年度数据
                    df_year = pd.read_csv(csv_file)
                    
                    # 筛选天然气进口记录 (商品编码2711: Petroleum gases and other gaseous hydrocarbons)
                    gas_imports = df_year[
                        (df_year['flowCode'] == 'M') &  # 进口
                        (df_year['cmdCode'] == 2711) &  # 天然气
                        (df_year['cifvalue'].notna()) & (df_year['cifvalue'] > 0) &  # 有效价值
                        (df_year['qty'].notna()) & (df_year['qty'] > 0)  # 有效数量
                    ].copy()
                    
                    if len(gas_imports) == 0:
                        logger.info(f"     {year}年无有效天然气进口记录")
                        continue
                    
                    logger.info(f"     {year}年: {len(gas_imports)}条天然气进口记录")
                    
                    # 标准化数量单位到kg
                    gas_imports['qty_standardized_kg'] = gas_imports.apply(
                        self._standardize_quantity_to_kg, axis=1
                    )
                    
                    # 过滤无效的标准化数量
                    gas_imports = gas_imports[gas_imports['qty_standardized_kg'] > 0]
                    
                    if len(gas_imports) == 0:
                        logger.info(f"     {year}年: 标准化后无有效记录")
                        continue
                    
                    # 按国家(reporterISO)和年份聚合
                    yearly_agg = gas_imports.groupby('reporterISO').agg({
                        'cifvalue': 'sum',      # 总进口价值(美元)
                        'qty_standardized_kg': 'sum'  # 总进口量(kg)
                    }).reset_index()
                    
                    yearly_agg['year'] = int(year)
                    yearly_agg['country'] = yearly_agg['reporterISO']  # 使用ISO代码作为国家标识
                    
                    # 计算年度平均进口价格 P_it (单位价值 = 美元/kg)
                    yearly_agg['P_it'] = yearly_agg['cifvalue'] / yearly_agg['qty_standardized_kg']
                    
                    # 保留需要的列
                    yearly_result = yearly_agg[['country', 'year', 'P_it', 'qty_standardized_kg']].copy()
                    all_gas_data.append(yearly_result)
                    
                    logger.info(f"     {year}年: {len(yearly_result)}个国家有天然气进口数据")
                    
                except Exception as e:
                    logger.warning(f"     ❌ {year}年数据处理失败: {str(e)}")
                    continue
            
            if not all_gas_data:
                logger.error("❌ 未能处理任何年份的天然气进口数据")
                return None
            
            # 合并所有年份数据
            combined_data = pd.concat(all_gas_data, ignore_index=True)
            logger.info(f"✅ 合并数据: {len(combined_data)}条记录，{combined_data['country'].nunique()}个国家")
            
            # 排序数据以便计算增长率
            combined_data = combined_data.sort_values(['country', 'year']).reset_index(drop=True)
            
            # 计算数量增长率 g_it = ln(qty_t) - ln(qty_{t-1})
            combined_data['qty_ln'] = np.log(combined_data['qty_standardized_kg'])
            combined_data['g_it'] = combined_data.groupby('country')['qty_ln'].diff()
            
            # 保存中间结果
            output_path = self.temp_data_dir / "gas_price_quantity_data.csv"
            combined_data.to_csv(output_path, index=False)
            
            # 返回最终结果
            result = combined_data[['country', 'year', 'P_it', 'g_it']].copy()
            
            # 同时保存到outputs目录供09模块使用
            final_output_path = self.output_dir / "price_quantity_variables.csv"
            result.to_csv(final_output_path, index=False)
            logger.info(f"💾 P_it和g_it变量保存至: {final_output_path}")
            
            logger.info(f"✅ 价格数量变量构建完成:")
            logger.info(f"   📊 数据记录: {len(result)} 条")
            logger.info(f"   🌍 覆盖国家: {result['country'].nunique()} 个")
            logger.info(f"   📅 年份范围: {result['year'].min()}-{result['year'].max()}")
            logger.info(f"   💾 中间文件: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 价格数量变量构建失败: {str(e)}")
            return None

    def _standardize_quantity_to_kg(self, row) -> float:
        """
        将不同单位的数量标准化为千克(kg)
        
        根据UN Comtrade数据中的qtyUnitAbbr字段进行单位转换
        """
        qty = row.get('qty', 0)
        unit = row.get('qtyUnitAbbr', '')
        
        if pd.isna(qty) or qty <= 0:
            return 0
        
        if pd.isna(unit):
            unit = ''
        
        unit = str(unit).strip().lower()
        
        # 单位转换映射 (转换为kg)
        unit_conversion = {
            'kg': 1.0,           # 千克
            't': 1000.0,         # 吨 = 1000kg
            'l': 0.5,           # 升 (LNG密度约0.5kg/L)
            'm3': 0.8,          # 立方米 (天然气密度约0.8kg/m3)
            'mt': 1000.0,       # 公吨 = 1000kg
            'g': 0.001,         # 克 = 0.001kg
        }
        
        # 查找匹配的单位转换因子
        conversion_factor = unit_conversion.get(unit, 1.0)  # 默认假设已经是kg
        
        return float(qty) * conversion_factor
    
    def construct_core_variables(self) -> Dict[str, pd.DataFrame]:
        """构建核心变量"""
        logger.info("⚙️ 开始构建核心变量...")
        
        core_vars = {}
        
        # 1. 加载 Node-DLI_US (由04模块构建)
        if 'node_dli_us' in self.base_data:
            core_vars['node_dli_us'] = self.base_data['node_dli_us']
        else:
            logger.warning("⚠️ 未能加载Node-DLI_US数据，将跳过此变量。")

        # 2. 加载 HHI_imports (由05模块构建，替代vul_us)
        if 'hhi_imports' in self.base_data:
            core_vars['hhi_imports'] = self.base_data['hhi_imports']
        else:
            logger.warning("⚠️ 未能加载HHI_imports数据，将跳过此变量。")
        
        # 3. 构建天然气OVI
        gas_ovi = self.construct_gas_ovi()
        if gas_ovi is not None:
            core_vars['ovi_gas'] = gas_ovi
        
        # 4. 构建 US产量冲击
        us_shock = self.construct_us_prod_shock()
        if us_shock is not None:
            core_vars['us_prod_shock'] = us_shock
        
        # 5. 构建价格和数量变量 (P_it 和 g_it)
        price_qty_data = self.construct_price_quantity_variables()
        if price_qty_data is not None:
            core_vars['price_quantity'] = price_qty_data
        
        self.core_variables = core_vars
        logger.info(f"✅ 核心变量构建完成，共 {len(core_vars)} 个变量")
        
        return core_vars
    
    def create_analytical_panel(self) -> pd.DataFrame:
        """创建最终分析面板"""
        logger.info("🔗 开始创建最终分析面板...")
        
        # 从宏观数据开始构建面板
        if self.macro_data is not None:
            panel = self.macro_data.copy()
            
            # 处理country列的映射
            if 'country_code' in panel.columns:
                # 世界银行数据已有标准ISO代码，直接使用
                panel['country'] = panel['country_code']
                logger.info(f"   基于世界银行宏观数据构建起始面板: {len(panel)} 行")
                logger.info(f"   覆盖国家: {panel['country'].nunique()} 个（使用标准ISO代码）")
            elif 'country_name' in panel.columns and 'country' not in panel.columns:
                # 旧版本数据，需要映射
                country_name_to_iso = {
                    'Australia': 'AUS', 'Brazil': 'BRA', 'Canada': 'CAN',
                    'China': 'CHN', 'France': 'FRA', 'Germany': 'DEU',
                    'Indonesia': 'IDN', 'Italy': 'ITA', 'Japan': 'JPN',
                    'Korea, Rep.': 'KOR', 'Mexico': 'MEX', 'Netherlands': 'NLD',
                    'Russian Federation': 'RUS', 'Saudi Arabia': 'SAU',
                    'Spain': 'ESP', 'Switzerland': 'CHE', 'Turkiye': 'TUR',
                    'United Kingdom': 'GBR', 'United States': 'USA'
                }
                panel['country'] = panel['country_name'].map(country_name_to_iso)
                panel = panel.dropna(subset=['country'])  # 移除无法映射的国家
                logger.info(f"   基于宏观数据构建起始面板: {len(panel)} 行（使用标准ISO代码）")
        else:
            # 创建基础框架面板
            logger.info("⚠️ 宏观数据缺失，创建基础国家-年份面板框架（使用标准ISO代码）")
            countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'ITA', 'BRA', 'CAN',
                        'RUS', 'AUS', 'KOR', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'CHE']
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
                    elif var_name == 'price_quantity':
                        # 价格数量变量单独保存，不合并到主面板
                        logger.info(f"   📊 {var_name}: 单独保存，不合并到主面板")
                    else:
                        # 其他变量按year和country合并
                        panel = panel.merge(var_data, on=['year', 'country'], how='left')
                    
                    after_len = len(panel)
                    
                    if after_len == before_len:
                        merge_count += 1
                        # 统计覆盖率 - 改进版本
                        if var_name != 'us_prod_shock' and var_name != 'price_quantity':
                            # 获取合并后实际的非空值数量
                            var_columns = [col for col in var_data.columns if col not in ['year', 'country']]
                            if var_columns:
                                main_var_col = var_columns[0]  # 使用第一个非索引列作为主要变量
                                actual_coverage = panel[main_var_col].notna().sum()
                                total_possible = len(panel)
                                coverage_rate = actual_coverage / total_possible * 100
                                logger.info(f"   ✅ 合并 {var_name}: {actual_coverage}/{total_possible} 观测值 ({coverage_rate:.1f}% 覆盖率)")
                        else:
                            logger.info(f"   ✅ 合并 {var_name}: {len(var_data)} 年数据（全面板复制）")
                            
                except Exception as e:
                    logger.warning(f"   ❌ 无法合并 {var_name}: {str(e)}")
        
        # 合并地理距离数据（新增：用于地理异质性分析）
        try:
            import json
            logger.info("🌍 合并地理距离数据...")
            
            # 读取地理距离JSON文件
            distance_json_path = Path("../04_dli_analysis/complete_us_distances_cepii.json")
            if distance_json_path.exists():
                with open(distance_json_path, 'r', encoding='utf-8') as f:
                    distance_data = json.load(f)
                
                # 转换为DataFrame
                distance_df = pd.DataFrame(list(distance_data.items()), columns=['country', 'distance_to_us'])
                distance_df['distance_to_us'] = pd.to_numeric(distance_df['distance_to_us'])
                
                # 左连接到主面板（距离数据对所有年份都相同）
                before_merge = len(panel)
                panel = panel.merge(distance_df, on='country', how='left')
                after_merge = len(panel)
                
                if after_merge == before_merge:
                    coverage = panel['distance_to_us'].notna().sum()
                    coverage_rate = coverage / len(panel) * 100
                    logger.info(f"   ✅ 地理距离数据合并成功: {coverage}/{len(panel)} 观测值 ({coverage_rate:.1f}% 覆盖率)")
                    logger.info(f"   覆盖国家样例: {sorted(distance_df['country'].head(10).tolist())}")
                else:
                    logger.warning(f"   ⚠️  合并后数据行数变化: {before_merge} -> {after_merge}")
                    
            else:
                logger.warning(f"   ⚠️  地理距离数据文件不存在: {distance_json_path}")
                # 添加空的distance_to_us列以保持数据结构一致
                panel['distance_to_us'] = np.nan
                
        except Exception as e:
            logger.warning(f"   ❌ 地理距离数据合并失败: {str(e)}")
            # 添加空的distance_to_us列以保持数据结构一致
            panel['distance_to_us'] = np.nan
        
        # 数据清洗
        panel = self._clean_final_panel(panel)
        
        # 验证关键变量完整性
        self._validate_panel_completeness(panel)
        
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
    
    def _validate_panel_completeness(self, panel: pd.DataFrame) -> None:
        """验证分析面板的关键变量完整性"""
        logger.info("🔍 验证分析面板关键变量完整性...")
        
        # 定义期望的核心变量（更新：hhi_imports替代vul_us）
        expected_core_vars = ['node_dli_us', 'hhi_imports', 'ovi_gas', 'us_prod_shock']
        
        validation_results = {}
        total_observations = len(panel)
        
        for var in expected_core_vars:
            if var in panel.columns:
                non_null_count = panel[var].notna().sum()
                coverage_rate = non_null_count / total_observations * 100
                validation_results[var] = {
                    'present': True,
                    'non_null_count': non_null_count,
                    'coverage_rate': coverage_rate
                }
                
                # 评估覆盖率
                if coverage_rate >= 70:
                    status = "✅ 优秀"
                elif coverage_rate >= 50:
                    status = "⚠️ 良好"
                elif coverage_rate >= 30:
                    status = "⚠️ 一般"
                else:
                    status = "❌ 不足"
                
                logger.info(f"   {var}: {non_null_count}/{total_observations} ({coverage_rate:.1f}%) {status}")
            else:
                validation_results[var] = {'present': False}
                logger.error(f"   ❌ {var}: 变量缺失！")
        
        # 总体评估
        present_vars = sum(1 for v in validation_results.values() if v.get('present', False))
        logger.info(f"📊 变量完整性总结: {present_vars}/{len(expected_core_vars)} 个核心变量存在")
        
        if present_vars < len(expected_core_vars):
            logger.warning(f"⚠️ 警告：{len(expected_core_vars) - present_vars} 个核心变量缺失，可能影响后续分析")
        else:
            logger.info("✅ 所有核心变量均已成功整合到分析面板")
        
        # 保存验证报告
        validation_report_path = self.output_dir / "data_validation_report.txt"
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            f.write("08模块数据完整性验证报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总观测数: {total_observations}\n")
            f.write(f"总国家数: {panel['country'].nunique()}\n")
            f.write(f"年份范围: {panel['year'].min()}-{panel['year'].max()}\n\n")
            
            f.write("核心变量覆盖率:\n")
            for var, result in validation_results.items():
                if result.get('present', False):
                    f.write(f"  {var}: {result['non_null_count']}/{total_observations} ({result['coverage_rate']:.1f}%)\n")
                else:
                    f.write(f"  {var}: 变量缺失\n")
        
        logger.info(f"📄 验证报告已保存: {validation_report_path}")

    def run_pipeline(self) -> None:
        """运行变量构建流水线"""
        logger.info("🚀 开始运行变量构建流水线...")
        
        try:
            # 步骤1: 准备宏观控制变量 (按需下载和清理)
            logger.info("\n" + "="*50)
            logger.info("步骤1: 准备宏观经济控制变量")
            logger.info("="*50)
            self._ensure_macro_data()
            
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
            logger.info("🎉 变量构建流水线执行完成！")
            logger.info("="*60)
            logger.info(f"✅ 最终输出:")
            logger.info(f"   - 分析面板: analytical_panel.csv ({len(self.final_panel)} 行)")
            logger.info(f"   - 中间文件: {self.temp_data_dir}")
            logger.info(f"   - 输出目录: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {str(e)}")
            raise

def main():
    """主函数：变量构建流水线"""
    print("🏗️ 08_variable_construction - 数据工厂")
    print("="*60)
    
    try:
        # 初始化变量构建器
        constructor = VariableConstructor()
        
        # 运行流水线
        constructor.run_pipeline()
        
        print("\n✅ 变量构建模块执行成功！")
        print("📄 查看输出文件:")
        print(f"   - {constructor.output_dir / 'analytical_panel.csv'}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()