#!/usr/bin/env python3
"""
092_final_econometric_model 数据加载器
================================

最终计量分析模块的数据整合组件
- 加载基础分析面板数据
- 整合地理距离数据
- 构建纯净LNG价格变量
- 为最终LP-IRF模型准备完整数据

作者：Energy Network Analysis Team
版本：v1.0 - 决定性因果推断版本
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FinalDataLoader:
    """
    最终数据加载器 - 整合所有数据源为LP-IRF分析做准备
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            project_root: 项目根目录，如果为None则使用默认路径
        """
        if project_root is None:
            self.project_root = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network")
        else:
            self.project_root = Path(project_root)
        
        # 定义新的数据路径 - 按照最新要求更新
        # 1. HHI数据
        self.hhi_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/outputs/hhi_imports_extracted.csv")
        
        # 2. 宏观控制变量
        self.macro_controls_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/outputs/macro_controls.csv")
        
        # 3. 核心OVI
        self.ovi_gas_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/outputs/ovi_gas.csv")
        
        # 4. 美国天然气产量
        self.us_prod_shock_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/outputs/us_prod_shock_ar2.csv")
        
        # 5. LNG价格和消费数据
        self.price_quantity_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/08_variable_construction/outputs/price_quantity_variables.csv")
        
        # 6. DLI数据（两个选择）
        self.dli_pagerank_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/04_dli_analysis/outputs/dli_pagerank.csv")
        self.dli_export_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/04_dli_analysis/outputs/export_dli.csv")
        
        # 保留原有的距离数据路径
        self.distance_data_path = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network/src/04_dli_analysis/complete_us_distances_cepii.json")
        
        logger.info(f"092模块数据加载器初始化完成")
        logger.info(f"项目根目录: {self.project_root}")
        
    def load_hhi_data(self) -> pd.DataFrame:
        """加载HHI进口数据"""
        logger.info("📊 加载HHI进口数据...")
        
        if not self.hhi_path.exists():
            logger.warning(f"⚠️ HHI数据不存在: {self.hhi_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.hhi_path)
            logger.info(f"✅ HHI数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
            return df
        except Exception as e:
            logger.error(f"❌ 加载HHI数据失败: {str(e)}")
            return pd.DataFrame()
    
    def load_macro_controls(self) -> pd.DataFrame:
        """加载宏观控制变量数据"""
        logger.info("🌍 加载宏观控制变量数据...")
        
        if not self.macro_controls_path.exists():
            logger.warning(f"⚠️ 宏观控制变量数据不存在: {self.macro_controls_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.macro_controls_path)
            logger.info(f"✅ 宏观控制变量加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
            return df
        except Exception as e:
            logger.error(f"❌ 加载宏观控制变量失败: {str(e)}")
            return pd.DataFrame()
    
    def load_ovi_gas_data(self) -> pd.DataFrame:
        """加载核心OVI天然气数据"""
        logger.info("⚡ 加载核心OVI天然气数据...")
        
        if not self.ovi_gas_path.exists():
            logger.error(f"❌ 核心OVI数据不存在: {self.ovi_gas_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.ovi_gas_path)
            logger.info(f"✅ OVI天然气数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
            return df
        except Exception as e:
            logger.error(f"❌ 加载OVI数据失败: {str(e)}")
            return pd.DataFrame()
    
    def load_us_prod_shock_data(self) -> pd.DataFrame:
        """加载美国天然气产量冲击数据"""
        logger.info("🇺🇸 加载美国产量冲击数据...")
        
        if not self.us_prod_shock_path.exists():
            logger.error(f"❌ 美国产量冲击数据不存在: {self.us_prod_shock_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.us_prod_shock_path)
            logger.info(f"✅ 美国产量冲击数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
            return df
        except Exception as e:
            logger.error(f"❌ 加载美国产量冲击数据失败: {str(e)}")
            return pd.DataFrame()
    
    def load_dli_data(self, use_pagerank_version: bool = None) -> pd.DataFrame:
        """
        加载DLI数据，支持两个版本的选择
        
        Args:
            use_pagerank_version: True使用pagerank版本，False使用export版本，None自动选择
            
        Returns:
            DLI数据DataFrame
        """
        logger.info("🔗 加载DLI数据...")
        
        # 检查两个版本的可用性
        pagerank_available = self.dli_pagerank_path.exists()
        export_available = self.dli_export_path.exists()
        
        logger.info(f"   DLI PageRank版本可用: {pagerank_available}")
        logger.info(f"   DLI Export版本可用: {export_available}")
        
        # 自动选择逻辑
        if use_pagerank_version is None:
            if pagerank_available and export_available:
                logger.info("   📊 两个版本都可用，优先使用PageRank增强版")
                use_pagerank_version = True
            elif pagerank_available:
                use_pagerank_version = True
            elif export_available:
                use_pagerank_version = False
            else:
                logger.error("❌ 两个DLI版本都不可用")
                return pd.DataFrame()
        
        # 加载选定版本
        try:
            if use_pagerank_version:
                if not pagerank_available:
                    logger.error(f"❌ PageRank版本不存在: {self.dli_pagerank_path}")
                    return pd.DataFrame()
                
                df = pd.read_csv(self.dli_pagerank_path)
                logger.info(f"✅ DLI PageRank版本加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
                df['dli_version'] = 'pagerank_enhanced'
                
            else:
                if not export_available:
                    logger.error(f"❌ Export版本不存在: {self.dli_export_path}")
                    return pd.DataFrame()
                
                df = pd.read_csv(self.dli_export_path)
                logger.info(f"✅ DLI Export版本加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")
                df['dli_version'] = 'export_only'
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 加载DLI数据失败: {str(e)}")
            return pd.DataFrame()
    
    def _merge_base_components(self, df_ovi: pd.DataFrame, df_hhi: pd.DataFrame, 
                              df_macro: pd.DataFrame, df_prod_shock: pd.DataFrame, 
                              df_dli: pd.DataFrame) -> pd.DataFrame:
        """
        合并所有基础数据组件
        
        Args:
            df_ovi: 核心OVI数据 
            df_hhi: HHI数据
            df_macro: 宏观控制变量
            df_prod_shock: 美国产量冲击数据
            df_dli: DLI数据
            
        Returns:
            合并后的基础面板数据
        """
        logger.info("🔗 开始合并基础数据组件...")
        
        # 以OVI数据作为主表
        df_merged = df_ovi.copy()
        logger.info(f"   主表(OVI): {df_merged.shape}")
        
        # 合并HHI数据
        if not df_hhi.empty:
            merge_keys = ['country', 'year'] if 'country' in df_hhi.columns else ['year']
            df_merged = df_merged.merge(df_hhi, on=merge_keys, how='left', suffixes=('', '_hhi'))
            logger.info(f"   合并HHI后: {df_merged.shape}")
        else:
            logger.warning("   ⚠️ HHI数据为空，跳过")
        
        # 合并宏观控制变量
        if not df_macro.empty:
            merge_keys = ['country', 'year'] if 'country' in df_macro.columns else ['year']
            df_merged = df_merged.merge(df_macro, on=merge_keys, how='left', suffixes=('', '_macro'))
            logger.info(f"   合并宏观变量后: {df_merged.shape}")
        else:
            logger.warning("   ⚠️ 宏观控制变量数据为空，跳过")
        
        # 合并美国产量冲击数据
        if not df_prod_shock.empty:
            # 产量冲击数据通常只有年份维度
            df_merged = df_merged.merge(df_prod_shock, on='year', how='left', suffixes=('', '_prod'))
            logger.info(f"   合并产量冲击后: {df_merged.shape}")
        else:
            logger.warning("   ⚠️ 美国产量冲击数据为空，跳过")
        
        # 合并DLI数据
        if not df_dli.empty:
            # DLI数据可能有不同的列名结构，需要适配
            dli_merge_keys = []
            if 'country' in df_dli.columns and 'country' in df_merged.columns:
                dli_merge_keys.extend(['country', 'year'])
            elif 'us_partner' in df_dli.columns:
                # DLI数据使用us_partner字段
                df_dli_adapted = df_dli.rename(columns={'us_partner': 'country'})
                dli_merge_keys = ['country', 'year']
                df_merged = df_merged.merge(df_dli_adapted, on=dli_merge_keys, how='left', suffixes=('', '_dli'))
            else:
                dli_merge_keys = ['year']
                df_merged = df_merged.merge(df_dli, on=dli_merge_keys, how='left', suffixes=('', '_dli'))
            
            logger.info(f"   合并DLI数据后: {df_merged.shape}")
            logger.info(f"   使用的DLI版本: {df_dli.get('dli_version', ['未知'])[0] if 'dli_version' in df_dli.columns else '未知'}")
        else:
            logger.warning("   ⚠️ DLI数据为空，跳过")
        
        logger.info(f"✅ 基础数据组件合并完成: {df_merged.shape}")
        
        # 检查关键变量
        key_vars = ['year', 'country', 'ovi_gas']
        available_vars = [var for var in key_vars if var in df_merged.columns]
        missing_vars = [var for var in key_vars if var not in df_merged.columns]
        
        logger.info(f"   可用关键变量: {available_vars}")
        if missing_vars:
            logger.warning(f"   ⚠️ 缺失关键变量: {missing_vars}")
        
        return df_merged
    
    def load_distance_data(self) -> Dict:
        """
        加载地理距离数据
        
        Returns:
            距离数据字典 {country_code: distance_to_us}
        """
        logger.info("🌍 加载地理距离数据...")
        
        if not self.distance_data_path.exists():
            logger.error(f"❌ 距离数据文件不存在: {self.distance_data_path}")
            return {}
        
        try:
            with open(self.distance_data_path, 'r', encoding='utf-8') as f:
                distance_data = json.load(f)
            
            logger.info(f"✅ 距离数据加载完成: {len(distance_data)} 个国家")
            
            # 显示示例数据
            sample_countries = list(distance_data.items())[:5]
            logger.info(f"   示例数据: {sample_countries}")
            
            return distance_data
            
        except Exception as e:
            logger.error(f"❌ 加载距离数据失败: {str(e)}")
            return {}
    
    def load_lng_data(self) -> pd.DataFrame:
        """
        加载LNG贸易数据
        
        Returns:
            LNG数据DataFrame
        """
        logger.info("🚢 加载LNG贸易数据...")
        
        if not self.lng_data_path.exists():
            logger.error(f"❌ LNG数据文件不存在: {self.lng_data_path}")
            return pd.DataFrame()
        
        try:
            df_lng = pd.read_csv(self.lng_data_path)
            logger.info(f"✅ LNG数据加载完成: {df_lng.shape[0]} 行 × {df_lng.shape[1]} 列")
            
            # 检查关键列
            required_cols = ['reporterISO', 'refYear', 'primaryValue', 'netWgt']
            missing_cols = [col for col in required_cols if col not in df_lng.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ LNG数据缺少列: {missing_cols}")
                return pd.DataFrame()
            
            # 显示数据范围
            if 'refYear' in df_lng.columns:
                year_range = f"{df_lng['refYear'].min()}-{df_lng['refYear'].max()}"
                logger.info(f"   时间范围: {year_range}")
            
            if 'reporterISO' in df_lng.columns:
                country_count = df_lng['reporterISO'].nunique()
                logger.info(f"   涵盖国家: {country_count} 个")
            
            return df_lng
            
        except Exception as e:
            logger.error(f"❌ 加载LNG数据失败: {str(e)}")
            return pd.DataFrame()
    
    def merge_distance_data(self, df_panel: pd.DataFrame, distance_data: Dict) -> pd.DataFrame:
        """
        将地理距离数据合并到分析面板
        
        Args:
            df_panel: 基础分析面板
            distance_data: 距离数据字典
            
        Returns:
            合并了距离数据的DataFrame
        """
        logger.info("🔗 合并地理距离数据...")
        
        if df_panel.empty or not distance_data:
            logger.warning("⚠️ 输入数据为空，跳过距离数据合并")
            return df_panel
        
        try:
            df_with_distance = df_panel.copy()
            
            # 添加距离列
            df_with_distance['distance_to_us'] = df_with_distance['country'].map(distance_data)
            
            # 统计合并结果
            matched_countries = df_with_distance['distance_to_us'].notna().sum()
            total_records = len(df_with_distance)
            match_rate = matched_countries / total_records if total_records > 0 else 0
            
            logger.info(f"✅ 距离数据合并完成:")
            logger.info(f"   • 成功匹配: {matched_countries}/{total_records} 条记录 ({match_rate:.1%})")
            
            # 显示未匹配的国家
            unmatched_countries = df_with_distance[df_with_distance['distance_to_us'].isna()]['country'].unique()
            if len(unmatched_countries) > 0:
                logger.info(f"   • 未匹配国家: {list(unmatched_countries)[:10]}...")
            
            return df_with_distance
            
        except Exception as e:
            logger.error(f"❌ 距离数据合并失败: {str(e)}")
            return df_panel
    
    def construct_lng_price(self, df_lng: pd.DataFrame) -> pd.DataFrame:
        """
        构建纯净LNG价格变量 P_it_lng
        
        Args:
            df_lng: LNG贸易数据
            
        Returns:
            包含P_it_lng的清洁数据
        """
        logger.info("💰 构建纯净LNG价格变量...")
        
        if df_lng.empty:
            logger.warning("⚠️ LNG数据为空，无法构建价格变量")
            return pd.DataFrame()
        
        try:
            # 复制数据
            df_lng_clean = df_lng.copy()
            
            # 标准化列名
            column_mapping = {
                'reporterISO': 'country',
                'refYear': 'year',
                'primaryValue': 'trade_value_usd',
                'netWgt': 'net_weight_kg'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df_lng_clean.columns:
                    df_lng_clean = df_lng_clean.rename(columns={old_col: new_col})
            
            # 检查必要列是否存在
            required_cols = ['country', 'year', 'trade_value_usd', 'net_weight_kg']
            missing_cols = [col for col in required_cols if col not in df_lng_clean.columns]
            
            if missing_cols:
                logger.error(f"❌ 缺少必要列: {missing_cols}")
                return pd.DataFrame()
            
            # 计算单价 P_it_lng = Trade Value (US$) / Net Weight (kg)
            logger.info("   计算原始LNG单价...")
            df_lng_clean = df_lng_clean[
                (df_lng_clean['trade_value_usd'] > 0) & 
                (df_lng_clean['net_weight_kg'] > 0)
            ].copy()
            
            df_lng_clean['P_it_lng_raw'] = (
                df_lng_clean['trade_value_usd'] / df_lng_clean['net_weight_kg']
            )
            
            # 移除明显异常值
            valid_prices = df_lng_clean[df_lng_clean['P_it_lng_raw'] > 0]
            logger.info(f"   原始价格观测数: {len(valid_prices)}")
            
            # 1%和99%缩尾处理
            logger.info("   执行1%和99%缩尾处理...")
            price_values = valid_prices['P_it_lng_raw'].values
            
            # 使用scipy的mstats.winsorize进行缩尾
            winsorized_prices = mstats.winsorize(price_values, limits=[0.01, 0.01])
            
            # 创建最终价格数据
            df_price_final = valid_prices.copy()
            df_price_final['P_it_lng'] = winsorized_prices
            
            # 按国家-年份聚合（取均值）
            logger.info("   按国家-年份聚合价格数据...")
            df_price_agg = df_price_final.groupby(['country', 'year']).agg({
                'P_it_lng': 'mean',
                'trade_value_usd': 'sum',
                'net_weight_kg': 'sum'
            }).reset_index()
            
            # 统计最终结果
            final_countries = df_price_agg['country'].nunique()
            final_records = len(df_price_agg)
            year_range = f"{df_price_agg['year'].min()}-{df_price_agg['year'].max()}"
            
            logger.info(f"✅ LNG价格构建完成:")
            logger.info(f"   • 最终记录数: {final_records}")
            logger.info(f"   • 涵盖国家: {final_countries} 个")
            logger.info(f"   • 时间范围: {year_range}")
            logger.info(f"   • 价格范围: ${df_price_agg['P_it_lng'].min():.4f} - ${df_price_agg['P_it_lng'].max():.4f} /kg")
            
            return df_price_agg
            
        except Exception as e:
            logger.error(f"❌ 构建LNG价格失败: {str(e)}")
            return pd.DataFrame()
    
    def merge_lng_price_data(self, df_panel: pd.DataFrame, df_lng_price: pd.DataFrame) -> pd.DataFrame:
        """
        将LNG价格数据合并到主分析面板
        
        Args:
            df_panel: 主分析面板（已含距离数据）
            df_lng_price: LNG价格数据
            
        Returns:
            最终的完整分析数据
        """
        logger.info("🔗 合并LNG价格数据...")
        
        if df_panel.empty:
            logger.warning("⚠️ 主分析面板为空")
            return df_panel
        
        if df_lng_price.empty:
            logger.warning("⚠️ LNG价格数据为空，将添加空的P_it_lng列")
            df_panel['P_it_lng'] = np.nan
            return df_panel
        
        try:
            # 执行左连接合并
            df_final = df_panel.merge(
                df_lng_price[['country', 'year', 'P_it_lng']], 
                on=['country', 'year'], 
                how='left'
            )
            
            # 统计合并结果
            total_records = len(df_final)
            lng_matched = df_final['P_it_lng'].notna().sum()
            lng_countries = df_final[df_final['P_it_lng'].notna()]['country'].nunique()
            match_rate = lng_matched / total_records if total_records > 0 else 0
            
            logger.info(f"✅ LNG价格数据合并完成:")
            logger.info(f"   • LNG价格覆盖: {lng_matched}/{total_records} 条记录 ({match_rate:.1%})")
            logger.info(f"   • 有LNG数据的国家: {lng_countries} 个")
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ LNG价格数据合并失败: {str(e)}")
            return df_panel
    
    def load_price_quantity_data(self) -> pd.DataFrame:
        """
        加载价格数量变量数据 (P_it, g_it)
        
        Returns:
            价格数量数据DataFrame
        """
        logger.info("📊 加载价格数量变量数据...")
        
        if not self.price_quantity_path.exists():
            logger.error(f"❌ 价格数量文件不存在: {self.price_quantity_path}")
            return pd.DataFrame()
        
        try:
            df_pq = pd.read_csv(self.price_quantity_path)
            logger.info(f"✅ 价格数量数据加载完成: {df_pq.shape[0]} 行 × {df_pq.shape[1]} 列")
            
            # 检查关键列
            required_cols = ['country', 'year', 'P_it', 'g_it']
            missing_cols = [col for col in required_cols if col not in df_pq.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ 价格数量数据缺少列: {missing_cols}")
                return pd.DataFrame()
            
            # 显示数据范围
            if 'year' in df_pq.columns:
                year_range = f"{df_pq['year'].min()}-{df_pq['year'].max()}"
                logger.info(f"   时间范围: {year_range}")
            
            if 'country' in df_pq.columns:
                country_count = df_pq['country'].nunique()
                logger.info(f"   涵盖国家: {country_count} 个")
            
            # 数据质量统计
            p_it_valid = df_pq['P_it'].notna().sum()
            g_it_valid = df_pq['g_it'].notna().sum()
            logger.info(f"   P_it有效观测: {p_it_valid}")
            logger.info(f"   g_it有效观测: {g_it_valid}")
            
            return df_pq
            
        except Exception as e:
            logger.error(f"❌ 加载价格数量数据失败: {str(e)}")
            return pd.DataFrame()
    
    def merge_price_quantity_data(self, df_panel: pd.DataFrame, df_pq: pd.DataFrame) -> pd.DataFrame:
        """
        将价格数量数据合并到主分析面板
        
        Args:
            df_panel: 主分析面板
            df_pq: 价格数量数据
            
        Returns:
            合并后的DataFrame
        """
        logger.info("🔗 合并价格数量数据...")
        
        if df_panel.empty:
            logger.warning("⚠️ 主分析面板为空")
            return df_panel
        
        if df_pq.empty:
            logger.warning("⚠️ 价格数量数据为空，将添加空的P_it和g_it列")
            df_panel['P_it'] = np.nan
            df_panel['g_it'] = np.nan
            return df_panel
        
        try:
            # 执行左连接合并
            df_merged = df_panel.merge(
                df_pq[['country', 'year', 'P_it', 'g_it']], 
                on=['country', 'year'], 
                how='left'
            )
            
            # 统计合并结果
            total_records = len(df_merged)
            p_it_matched = df_merged['P_it'].notna().sum()
            g_it_matched = df_merged['g_it'].notna().sum()
            pq_countries = df_merged[df_merged['P_it'].notna() | df_merged['g_it'].notna()]['country'].nunique()
            
            logger.info(f"✅ 价格数量数据合并完成:")
            logger.info(f"   • P_it覆盖: {p_it_matched}/{total_records} 条记录 ({p_it_matched/total_records:.1%})")
            logger.info(f"   • g_it覆盖: {g_it_matched}/{total_records} 条记录 ({g_it_matched/total_records:.1%})")
            logger.info(f"   • 有价格数量数据的国家: {pq_countries} 个")
            
            return df_merged
            
        except Exception as e:
            logger.error(f"❌ 价格数量数据合并失败: {str(e)}")
            return df_panel

    def load_clean_lng_price(self) -> pd.DataFrame:
        """
        加载清理后的LNG价格数据
        
        Returns:
            清理后的LNG价格DataFrame
        """
        logger.info("🚢 加载清理后的LNG价格数据...")
        
        clean_lng_path = Path("outputs/clean_lng_price_data.csv")
        
        if not clean_lng_path.exists():
            logger.warning(f"⚠️ 清理后的LNG数据不存在: {clean_lng_path}")
            logger.info("   请先运行 clean_lng_data.py 脚本")
            return pd.DataFrame()
        
        try:
            df_lng = pd.read_csv(clean_lng_path)
            logger.info(f"✅ 清理后LNG数据加载完成: {df_lng.shape[0]} 行 × {df_lng.shape[1]} 列")
            
            # 检查必要列
            required_cols = ['country', 'year', 'P_lng']
            missing_cols = [col for col in required_cols if col not in df_lng.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ LNG数据缺少列: {missing_cols}")
                return pd.DataFrame()
            
            # 数据质量统计
            valid_prices = df_lng['P_lng'].notna().sum()
            logger.info(f"   有效价格记录: {valid_prices}")
            logger.info(f"   涵盖国家: {df_lng['country'].nunique()} 个")
            logger.info(f"   时间范围: {df_lng['year'].min()}-{df_lng['year'].max()}")
            logger.info(f"   价格范围: ${df_lng['P_lng'].min():.4f} - ${df_lng['P_lng'].max():.4f} /kg")
            
            return df_lng
            
        except Exception as e:
            logger.error(f"❌ 加载清理后LNG数据失败: {str(e)}")
            return pd.DataFrame()
    
    def merge_clean_lng_price(self, df_panel: pd.DataFrame, df_lng: pd.DataFrame) -> pd.DataFrame:
        """
        将清理后的LNG价格数据合并到主分析面板
        
        Args:
            df_panel: 主分析面板
            df_lng: 清理后的LNG价格数据
            
        Returns:
            最终的完整分析数据
        """
        logger.info("🔗 合并清理后的LNG价格数据...")
        
        if df_panel.empty:
            logger.warning("⚠️ 主分析面板为空")
            return df_panel
        
        if df_lng.empty:
            logger.warning("⚠️ LNG价格数据为空，将添加空的P_lng列")
            df_panel['P_lng'] = np.nan
            return df_panel
        
        try:
            # 执行左连接合并
            df_final = df_panel.merge(
                df_lng[['country', 'year', 'P_lng']], 
                on=['country', 'year'], 
                how='left'
            )
            
            # 统计合并结果
            total_records = len(df_final)
            lng_matched = df_final['P_lng'].notna().sum()
            lng_countries = df_final[df_final['P_lng'].notna()]['country'].nunique()
            match_rate = lng_matched / total_records if total_records > 0 else 0
            
            logger.info(f"✅ LNG价格数据合并完成:")
            logger.info(f"   • P_lng覆盖: {lng_matched}/{total_records} 条记录 ({match_rate:.1%})")
            logger.info(f"   • 有LNG价格数据的国家: {lng_countries} 个")
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ LNG价格数据合并失败: {str(e)}")
            return df_panel

    def load_complete_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """
        加载完整的最终分析数据集
        
        Returns:
            (完整数据集, 数据统计信息)
        """
        logger.info("🚀 开始加载完整的最终分析数据集...")
        
        # 步骤1: 分别加载各个数据源（替代原有的analytical_panel）
        logger.info("📊 开始分别加载各个数据源...")
        
        # 加载核心OVI数据作为主表
        df_main = self.load_ovi_gas_data()
        if df_main.empty:
            return pd.DataFrame(), {'status': 'failed', 'message': '核心OVI数据加载失败'}
        
        # 加载其他数据源
        df_hhi = self.load_hhi_data()
        df_macro = self.load_macro_controls()
        df_prod_shock = self.load_us_prod_shock_data()
        df_dli = self.load_dli_data()  # 自动选择最佳版本
        
        # 构建基础面板数据集
        df_panel = self._merge_base_components(df_main, df_hhi, df_macro, df_prod_shock, df_dli)
        
        if df_panel.empty:
            return pd.DataFrame(), {'status': 'failed', 'message': '基础数据合并失败'}
        
        # 步骤2: 加载价格数量数据 (P_it, g_it)
        df_pq = self.load_price_quantity_data()
        
        # 步骤3: 合并价格数量数据
        df_with_pq = self.merge_price_quantity_data(df_panel, df_pq)
        
        # 步骤4: 加载地理距离数据
        distance_data = self.load_distance_data()
        
        # 步骤5: 合并距离数据
        df_with_distance = self.merge_distance_data(df_with_pq, distance_data)
        
        # 步骤6: 加载清理后的LNG价格数据
        df_lng_price = self.load_clean_lng_price()
        
        # 步骤7: 最终合并
        df_final = self.merge_clean_lng_price(df_with_distance, df_lng_price)
        
        # 生成数据统计
        stats = self._generate_dataset_stats(df_final)
        
        logger.info(f"🎉 完整数据集构建完成:")
        logger.info(f"   • 最终形状: {df_final.shape}")
        logger.info(f"   • 核心变量完整性: {stats['core_variables_status']}")
        
        return df_final, stats
    
    def _generate_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        生成数据集统计信息
        
        Args:
            df: 最终数据集
            
        Returns:
            统计信息字典
        """
        if df.empty:
            return {'status': 'empty', 'message': '数据集为空'}
        
        # 核心变量检查
        core_vars = ['country', 'year', 'ovi_gas', 'us_prod_shock', 'distance_to_us', 'P_it', 'g_it', 'P_lng', 'log_gdp', 'log_population']
        core_status = {}
        
        for var in core_vars:
            if var in df.columns:
                non_null_count = df[var].notna().sum()
                total_count = len(df)
                core_status[var] = {
                    'available': True,
                    'coverage': f"{non_null_count}/{total_count} ({non_null_count/total_count:.1%})"
                }
            else:
                core_status[var] = {'available': False, 'coverage': '0/0 (0.0%)'}
        
        return {
            'status': 'success',
            'total_observations': len(df),
            'total_countries': df['country'].nunique() if 'country' in df.columns else 0,
            'year_range': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns and not df['year'].isna().all() else 'N/A',
            'core_variables_status': core_status,
            'columns': list(df.columns)
        }


def main():
    """测试数据加载功能"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    print("🔬 092_final_econometric_model 数据加载器测试")
    print("=" * 60)
    
    # 创建数据加载器
    loader = FinalDataLoader()
    
    # 加载完整数据集
    df_final, stats = loader.load_complete_dataset()
    
    print(f"\n📊 最终数据集统计:")
    print(f"   • 数据形状: {df_final.shape}")
    print(f"   • 状态: {stats.get('status', 'unknown')}")
    
    if stats['status'] == 'success':
        print(f"   • 总观测数: {stats['total_observations']}")
        print(f"   • 国家数: {stats['total_countries']}")
        print(f"   • 时间范围: {stats['year_range']}")
        
        print(f"\n📋 核心变量状态:")
        for var, info in stats['core_variables_status'].items():
            status_icon = "✅" if info['available'] else "❌"
            print(f"   {status_icon} {var}: {info['coverage']}")
    
    print(f"\n🎉 数据加载器测试完成!")


if __name__ == "__main__":
    main()