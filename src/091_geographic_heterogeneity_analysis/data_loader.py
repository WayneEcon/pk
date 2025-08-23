#!/usr/bin/env python3
"""
数据加载模块 (Data Loader Module)
============================================

09_econometric_analysis 模块的数据加载组件

作者：Energy Network Analysis Team
版本：v1.1 - 兼容 ovi_gas 并增强诊断
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class DataLoader:
    """
    数据加载器 - 专门处理空数据和缺失文件的健壮加载逻辑
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            project_root: 项目根目录，如果为None则自动推断
        """
        if project_root is None:
            # 使用绝对路径确保能找到正确的项目根目录
            self.project_root = Path("/Users/ywz/Desktop/pku/美国能源独立/project/energy_network")
        else:
            self.project_root = Path(project_root)
        
        # 更新路径配置：analytical_panel从08模块outputs目录加载
        self.analytical_panel_path = self.project_root / "src" / "08_variable_construction" / "outputs" / "analytical_panel.csv"
        # 价格数量变量文件路径
        self.price_quantity_path = self.project_root / "src" / "08_variable_construction" / "outputs" / "price_quantity_variables.csv"
        # DLI和VUL变量文件路径
        self.node_dli_path = self.project_root / "src" / "08_variable_construction" / "08data" / "node_dli_us.csv"
        self.vul_us_path = self.project_root / "src" / "08_variable_construction" / "08data" / "vul_us.csv"
        
        logger.info(f"数据加载器初始化完成")
        logger.info(f"项目根目录: {self.project_root}")
        logger.info(f"分析面板路径: {self.analytical_panel_path}")
        logger.info(f"价格数量变量路径: {self.price_quantity_path}")
        
        # 定义预期的列名（基于08模块的输出规范）
        self.expected_columns = self._get_expected_columns()
    
    def _get_expected_columns(self) -> List[str]:
        """
        定义预期的数据列名
        
        Returns:
            预期的列名列表
        """
        return [
            # 基础标识变量
            'year', 'country', 'country_name',
            
            # 宏观经济控制变量
            'gdp_current_usd', 'population_total', 'trade_openness_gdp_pct',
            'log_gdp', 'log_population',
            
            # 核心研究变量（来自08模块）
            'node_dli_us',         # Node-DLI_US: 美国锚定动态锁定指数
            'vul_us',              # Vul_US: 美国锚定脆弱性指数
            'ovi_gas',             # 已从 'ovi' 更新
            'us_prod_shock',       # US_ProdShock: 美国产量冲击
            
            # 网络拓扑指标（来自03模块）
            'betweenness_centrality', 'eigenvector_centrality',
            'in_degree', 'out_degree', 'total_degree',
            
            # 辅助变量
            'us_production_oil', 'us_production_gas'
        ]
    
    def _fix_ovi_gas_data_integration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        智能修复：使用统一标准化系统整合08模块原始gas_ovi数据
        解决国家编码不一致导致的数据丢失问题
        """
        logger.info("🔧 智能修复：整合丢失的ovi_gas数据...")
        
        try:
            # 导入统一标准化系统
            from country_standardizer import CountryStandardizer
            standardizer = CountryStandardizer()
            
            # 尝试加载原始gas_ovi数据
            gas_ovi_path = self.project_root / "src" / "08_variable_construction" / "08data" / "gas_ovi.csv"
            
            if not gas_ovi_path.exists():
                logger.warning(f"⚠️ 原始gas_ovi文件不存在: {gas_ovi_path}")
                return df
            
            df_gas_ovi = pd.read_csv(gas_ovi_path)
            logger.info(f"📊 加载原始gas_ovi数据: {df_gas_ovi.shape}")
            
            # 使用统一标准化系统处理国家编码
            df_gas_ovi_standardized = standardizer.standardize_dataframe(
                df_gas_ovi, 
                country_column='country', 
                new_column_name='country_standardized'
            )
            
            # 保留标准化成功的数据
            df_gas_ovi_clean = df_gas_ovi_standardized.dropna(subset=['country_standardized'])
            
            # 准备合并数据
            ovi_data = df_gas_ovi_clean[['country_standardized', 'year', 'ovi_gas']].rename(
                columns={'country_standardized': 'country'}
            )
            
            # 合并到主数据框
            df_before = df.copy()
            
            # 先移除原有的ovi_gas列（如果存在）
            if 'ovi_gas' in df.columns:
                df = df.drop(columns=['ovi_gas'])
            
            # 左连接合并ovi_gas数据
            df_merged = df.merge(ovi_data, on=['country', 'year'], how='left')
            
            # 统计修复效果
            original_ovi_count = df_before['ovi_gas'].notna().sum() if 'ovi_gas' in df_before.columns else 0
            new_ovi_count = df_merged['ovi_gas'].notna().sum()
            countries_with_ovi = df_merged[df_merged['ovi_gas'].notna()]['country'].nunique()
            
            logger.info(f"✅ ovi_gas数据智能修复完成:")
            logger.info(f"   • 修复前有效观测: {original_ovi_count}")
            logger.info(f"   • 修复后有效观测: {new_ovi_count}")
            logger.info(f"   • 新增有效观测: {new_ovi_count - original_ovi_count}")
            logger.info(f"   • 有数据的国家: {countries_with_ovi}")
            
            return df_merged
            
        except Exception as e:
            logger.error(f"❌ ovi_gas数据智能修复失败: {str(e)}")
            return df

    def _load_and_merge_price_quantity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        加载并合并价格数量变量数据
        
        Args:
            df: 主要分析面板数据
            
        Returns:
            合并了P_it和g_it变量的数据框
        """
        logger.info("🔗 开始加载并合并价格数量变量...")
        
        try:
            if not self.price_quantity_path.exists():
                logger.warning(f"⚠️ 价格数量变量文件不存在: {self.price_quantity_path}")
                return df
            
            # 加载价格数量变量数据
            df_pq = pd.read_csv(self.price_quantity_path)
            logger.info(f"📊 价格数量变量数据: {df_pq.shape[0]} 行 × {df_pq.shape[1]} 列")
            
            # 检查必要的列
            required_cols = ['country', 'year', 'P_it', 'g_it']
            missing_cols = [col for col in required_cols if col not in df_pq.columns]
            if missing_cols:
                logger.warning(f"⚠️ 价格数量数据缺少必要列: {missing_cols}")
                return df
            
            # 左连接合并数据（保留主面板的所有记录）
            df_before = df.copy()
            df_merged = df.merge(df_pq[required_cols], on=['country', 'year'], how='left')
            
            # 统计合并效果
            pit_count = df_merged['P_it'].notna().sum()
            git_count = df_merged['g_it'].notna().sum()
            
            logger.info(f"✅ 价格数量变量合并完成:")
            logger.info(f"   • P_it有效观测: {pit_count}")
            logger.info(f"   • g_it有效观测: {git_count}")
            logger.info(f"   • 覆盖国家: {df_merged[df_merged['P_it'].notna()]['country'].nunique()} 个")
            
            return df_merged
            
        except Exception as e:
            logger.error(f"❌ 价格数量变量合并失败: {str(e)}")
            return df

    def _load_and_merge_dli_vul_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        加载并合并node_dli_us和vul_us变量数据
        
        Args:
            df: 主要分析面板数据
            
        Returns:
            合并了DLI和VUL变量的数据框
        """
        logger.info("🔗 开始加载并合并DLI和VUL变量...")
        
        try:
            # 加载node_dli_us数据
            if self.node_dli_path.exists():
                df_dli = pd.read_csv(self.node_dli_path)
                logger.info(f"📊 Node-DLI数据: {df_dli.shape[0]} 行")
                
                # 合并DLI数据
                df_before = df.copy()
                df = df.merge(df_dli[['country', 'year', 'node_dli_us']], on=['country', 'year'], how='left')
                dli_count = df['node_dli_us'].notna().sum()
                logger.info(f"   ✅ node_dli_us合并完成: {dli_count}条有效观测")
            else:
                logger.warning(f"⚠️ Node-DLI文件不存在: {self.node_dli_path}")
            
            # 加载vul_us数据
            if self.vul_us_path.exists():
                df_vul = pd.read_csv(self.vul_us_path)
                logger.info(f"📊 VUL-US数据: {df_vul.shape[0]} 行")
                
                # 合并VUL数据
                df = df.merge(df_vul[['country', 'year', 'vul_us']], on=['country', 'year'], how='left')
                vul_count = df['vul_us'].notna().sum()
                logger.info(f"   ✅ vul_us合并完成: {vul_count}条有效观测")
            else:
                logger.warning(f"⚠️ VUL-US文件不存在: {self.vul_us_path}")
                
            logger.info("✅ DLI和VUL变量合并完成")
            return df
            
        except Exception as e:
            logger.error(f"❌ DLI和VUL变量合并失败: {str(e)}")
            return df

    def load_analytical_panel(self) -> pd.DataFrame:
        """加载分析面板数据 - 实现严格的平衡面板数据清洗逻辑"""
        logger.info("🔍 开始加载分析面板数据...")
        
        if not self.analytical_panel_path.exists():
            logger.error(f"❌ 分析面板文件不存在: {self.analytical_panel_path}")
            logger.error("这是致命错误！必须使用08模块输出的正确数据文件。")
            return self._create_empty_dataframe()
        
        try:
            # 步骤1：加载原始数据
            df_raw = pd.read_csv(self.analytical_panel_path)
            logger.info(f"📊 原始数据加载完成: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
            
            if len(df_raw) == 0:
                logger.error("❌ 分析面板文件为空 - 这是致命错误！")
                return self._create_empty_dataframe()
            
            # 步骤1.5：紧急修复ovi_gas数据丢失问题
            df_fixed = self._fix_ovi_gas_data_integration(df_raw)
            
            # 步骤1.6：加载并合并价格数量变量
            df_with_price_qty = self._load_and_merge_price_quantity_data(df_fixed)
            
            # 步骤1.7：加载并合并node_dli_us和vul_us变量
            df_with_dli_vul = self._load_and_merge_dli_vul_data(df_with_price_qty)
            
            # 步骤2：严格的平衡面板数据清洗
            df_cleaned = self._enforce_balanced_panel_constraints(df_with_dli_vul)
            
            # 步骤3：确保列完整性
            df_final = self._ensure_expected_columns(df_cleaned)
            
            logger.info(f"✅ 平衡面板数据准备完成: {df_final.shape}")
            return df_final
            
        except Exception as e:
            logger.error(f"❌ 加载分析面板数据失败: {str(e)}")
            return self._create_empty_dataframe()
    
    def _enforce_balanced_panel_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行严格的平衡面板约束条件
        确保所有用于回归分析的核心变量都没有缺失值
        """
        logger.info("🔧 开始执行严格的平衡面板约束...")
        
        # 采用分层清洗策略：仅对核心分析变量要求严格无缺失
        # 第一层：核心回归分析变量（绝对不能有缺失值）
        core_analysis_variables = [
            'country', 'year',                    # 面板标识
            'node_dli_us', 'vul_us',              # 核心因变量
            'ovi_gas', 'us_prod_shock',           # 核心解释变量
            'log_gdp', 'log_population'           # 基础控制变量
        ]
        
        # 第二层：可选控制变量（允许部分缺失，但会影响样本量）
        optional_control_variables = [
            'trade_openness_gdp_pct',
            'betweenness_centrality', 'eigenvector_centrality',
            'in_degree', 'out_degree', 'total_degree'
        ]
        
        # 检查核心变量是否存在
        missing_core_vars = [var for var in core_analysis_variables if var not in df.columns]
        if missing_core_vars:
            logger.error(f"❌ 核心变量缺失: {missing_core_vars}")
            logger.error("这违反了08模块的数据输出约定！")
        
        # 报告原始数据统计
        original_countries = df['country'].nunique() if 'country' in df.columns else 0
        original_years = df['year'].nunique() if 'year' in df.columns else 0
        logger.info(f"📈 原始数据统计: {original_countries} 个国家, {original_years} 个年份")
        
        # 步骤1：对核心变量执行严格清洗
        available_core_vars = [var for var in core_analysis_variables if var in df.columns]
        logger.info(f"🎯 第一层清洗：对 {len(available_core_vars)} 个核心变量执行严格缺失值剔除...")
        
        df_before = df.copy()
        df_core_cleaned = df.dropna(subset=available_core_vars)
        
        # 报告第一层清洗结果
        core_rows_dropped = len(df_before) - len(df_core_cleaned)
        logger.info(f"   • 核心变量清洗剔除: {core_rows_dropped} 行")
        logger.info(f"   • 核心清洗后样本: {len(df_core_cleaned)} 观测值")
        
        # 步骤2：评估可选变量的可用性（不强制剔除）
        logger.info(f"🔍 第二层评估：检查可选控制变量的数据完整性...")
        available_optional_vars = []
        for var in optional_control_variables:
            if var in df_core_cleaned.columns:
                non_null_count = df_core_cleaned[var].notna().sum()
                coverage_rate = non_null_count / len(df_core_cleaned)
                logger.info(f"   • {var}: {non_null_count}/{len(df_core_cleaned)} ({coverage_rate:.1%})")
                if coverage_rate >= 0.7:  # 至少70%的数据可用
                    available_optional_vars.append(var)
        
        logger.info(f"   • 高质量可选变量: {available_optional_vars}")
        
        # 最终清洗决策：基于核心变量的严格清洗结果
        df_cleaned = df_core_cleaned.copy()
        
        # 报告清洗结果
        rows_dropped = len(df_before) - len(df_cleaned)
        final_countries = df_cleaned['country'].nunique() if 'country' in df_cleaned.columns else 0
        final_years = df_cleaned['year'].nunique() if 'year' in df_cleaned.columns else 0
        
        logger.info(f"🧹 数据清洗完成:")
        logger.info(f"   • 剔除观测值: {rows_dropped} 行")
        logger.info(f"   • 最终样本: {len(df_cleaned)} 观测值")
        logger.info(f"   • 最终国家数: {final_countries}")
        logger.info(f"   • 最终年份数: {final_years}")
        
        # 验证是否符合预期的平衡面板规格（45国家 × 669观测值）
        if len(df_cleaned) == 669 and final_countries == 45:
            logger.info("✅ 数据完全符合预期的平衡面板规格 (45国家 × 669观测值)")
        else:
            logger.warning(f"⚠️ 数据不符合预期规格:")
            logger.warning(f"   期望: 45国家 × 669观测值")
            logger.warning(f"   实际: {final_countries}国家 × {len(df_cleaned)}观测值")
        
        return df_cleaned
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """
        创建空但结构正确的DataFrame
        
        Returns:
            包含所有预期列但无数据的DataFrame
        """
        logger.info("   构建空DataFrame框架...")
        
        # 创建空DataFrame但包含所有预期列
        df = pd.DataFrame(columns=self.expected_columns)
        
        # 设置正确的数据类型
        type_mapping = {
            'year': 'int64',
            'country': 'str',
            'country_name': 'str',
            'gdp_current_usd': 'float64',
            'population_total': 'float64',
            'trade_openness_gdp_pct': 'float64',
            'log_gdp': 'float64',
            'log_population': 'float64',
            'node_dli_us': 'float64',
            'vul_us': 'float64',
            'ovi_gas': 'float64', # 已从 'ovi' 更新
            'us_prod_shock': 'float64'
        }
        
        for col, dtype in type_mapping.items():
            if col in df.columns:
                if dtype == 'str':
                    df[col] = df[col].astype('object')
                else:
                    df[col] = df[col].astype(dtype)
        
        logger.info(f"   空DataFrame框架创建完成: {len(self.expected_columns)} 列")
        return df
    
    def _ensure_expected_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保DataFrame包含所有预期列
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含所有预期列的DataFrame
        """
        missing_cols = set(self.expected_columns) - set(df.columns)
        
        if missing_cols:
            logger.info(f"   添加缺失列: {sorted(missing_cols)}")
            for col in missing_cols:
                df[col] = np.nan
        
        # 重新排序列以匹配预期顺序
        available_expected_cols = [col for col in self.expected_columns if col in df.columns]
        other_cols = [col for col in df.columns if col not in self.expected_columns]
        df = df[available_expected_cols + other_cols]
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """获取数据摘要统计，包含描述性统计"""
        if len(df) == 0:
            return {
                'total_rows': 0, 'total_cols': len(df.columns), 'year_range': 'N/A',
                'countries': 0, 'key_variables_available': [], 'data_status': 'empty',
                'descriptive_stats': {}
            }
        
        key_variables = ['node_dli_us', 'vul_us', 'ovi_gas', 'us_prod_shock']
        available_key_vars_info = [f"{var} ({df[var].notna().sum()}/{len(df)})" for var in key_variables if var in df.columns and df[var].notna().any()]
        
        desc_stats = {}
        if available_key_vars_info:
            desc_df = df[key_variables].dropna(how='all').describe().transpose()
            desc_stats = desc_df[['count', 'mean', 'std', 'min', 'max']].to_dict('index')

        return {
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'year_range': f"{int(df['year'].min())}-{int(df['year'].max())}" if 'year' in df.columns and df['year'].notna().any() else 'N/A',
            'countries': df['country'].nunique() if 'country' in df.columns else 0,
            'key_variables_available': available_key_vars_info,
            'data_status': 'available' if available_key_vars_info else 'missing_key_vars',
            'descriptive_stats': desc_stats,
            'raw_panel_data': df # 传递原始数据用于可视化
        }
    
    def validate_data_for_analysis(self, df: pd.DataFrame) -> Dict:
        """验证数据是否适合进行计量分析"""
        validation = {'is_valid_for_analysis': False, 'issues': [], 'recommendations': []}
        if len(df) == 0:
            validation['issues'].append("数据集为空")
            return validation

        key_variables = ['node_dli_us', 'vul_us', 'ovi_gas', 'us_prod_shock']
        missing_key_vars = [var for var in key_variables if var not in df.columns or df[var].isna().all()]
        if missing_key_vars:
            validation['issues'].append(f"关键研究变量缺失或全为空: {missing_key_vars}")
        
        if 'year' not in df.columns or 'country' not in df.columns:
            validation['issues'].append("缺少面板数据必需的 'year' 或 'country' 标识")
        
        validation['is_valid_for_analysis'] = not validation['issues']
        return validation


def get_data_status() -> Dict:
    """便捷函数：加载数据并获取其状态摘要"""
    loader = DataLoader()
    df = loader.load_analytical_panel()
    summary = loader.get_data_summary(df)
    validation = loader.validate_data_for_analysis(df)
    
    return {'summary': summary, 'validation': validation}


if __name__ == "__main__":
    # 测试数据加载功能
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🔬 09_econometric_analysis 数据加载器测试")
    print("=" * 50)
    
    # 测试数据加载
    loader = DataLoader()
    df = loader.load_analytical_panel()
    
    print(f"\n📊 数据加载结果:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 数据摘要
    summary = loader.get_data_summary(df)
    print(f"\n📈 数据摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 数据验证
    validation = loader.validate_data_for_analysis(df)
    print(f"\n✅ 数据验证:")
    print(f"  适合分析: {validation['is_valid_for_analysis']}")
    if validation['issues']:
        print(f"  问题: {validation['issues']}")
    if validation['recommendations']:
        print(f"  建议: {validation['recommendations']}")
    
    print("\n🎉 数据加载器测试完成!")