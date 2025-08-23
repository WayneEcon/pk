#!/usr/bin/env python3
"""
计量模型模块 (Econometric Models Module)
=====================================

09_econometric_analysis 模块的核心计量模型实现

理论框架：锚定多样化假说 (Anchored Diversification Hypothesis)
============================================================

核心研究问题：
一个国家与美国建立更紧密的能源网络关系，是否有助于其整体能源进口来源的多元化？

核心模型设定：
HHI_imports_{i,t} = β·NodeDLI^{US}_{i,t} + Γ·Controls_{i,t} + α_i + λ_t + ε_{i,t}

理论预期：β < 0
- 锚定多样化假说认为，与主要能源出口国（美国）建立稳定的能源网络关系，
  为进口国提供了一个可靠的"锚点"，从而降低了对任何单一供应商的依赖，
  鼓励进口多样化策略，最终导致hhi_imports指数下降。

学术严谨性考虑：
1. 聚类稳健标准误：所有面板回归使用按国家聚类的稳健标准误，校正国家内观测值的序列相关性
2. 避免构造内生性：核心模型直接使用hhi_imports，避免之前vul_us=f(us_import_share, hhi_imports)的循环论证
3. 双向固定效应：控制国家异质性和时间趋势的影响

模型局限性与未来方向：
警告：本模型结果应被严谨地解读为相关性证据而非严格因果效应，主要挑战包括：
1. 遗漏变量偏误：制度质量、政治立场等不可观测因素可能同时影响NodeDLI_US和HHI
2. 反向因果：多样化程度高的国家可能更容易吸引美国建立能源合作关系
3. 未来研究方向：考虑使用工具变量法（Bartik式或重力模型式）进行因果识别

作者：Energy Network Analysis Team
版本：v2.0 - 锚定多样化假说版本（含局限性讨论）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import warnings

# 计量分析库导入 (条件导入以处理缺失依赖)
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from linearmodels import PanelOLS, PooledOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

# 简化的配置设置
class SimpleConfig:
    def __init__(self):
        self.validation = type('obj', (object,), {'DATA_QUALITY_THRESHOLDS': {'min_observations': 20}})
        self.analysis = type('obj', (object,), {'RESEARCH_MODELS': {}})
    
    def get_control_variables(self, control_type):
        if control_type == 'macro_controls':
            return ['log_gdp', 'log_population']
        return []

# 使用简化配置
config = SimpleConfig()

logger = logging.getLogger(__name__)

class EconometricModels:
    """
    计量分析模型类 - 实现空数据兼容的计量分析框架
    """
    
    def __init__(self):
        """初始化计量模型分析器"""
        self.config = config
        self.results = {}
        
        logger.info("🔬 计量模型分析器初始化完成")
        
        # 检查依赖库可用性
        if not HAS_STATSMODELS:
            logger.warning("⚠️ statsmodels库不可用，部分功能受限")
        if not HAS_LINEARMODELS:
            logger.warning("⚠️ linearmodels库不可用，面板数据分析受限")
    
    def _check_data_availability(self, df: pd.DataFrame, required_vars: List[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        检查数据可用性 - 只检查存在性和非空性，不执行dropna()
        
        Args:
            df: 输入数据
            required_vars: 必需变量列表
            
        Returns:
            (检查结果字典, 原始未修改的DataFrame)
        """
        check_result = {
            'data_available': False,
            'missing_vars': [],
            'empty_vars': [],
            'total_obs': 0,
            'status_message': ''
        }
        
        # 基础检查：数据是否为空
        if df is None or len(df) == 0:
            check_result['status_message'] = '数据集为空或不存在'
            return check_result, df
        
        check_result['total_obs'] = len(df)
        
        # 检查变量是否存在
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            check_result['missing_vars'] = missing_vars
            check_result['status_message'] = f'缺少必需变量: {missing_vars}'
            return check_result, df
        
        # 检查变量是否全为空
        empty_vars = [var for var in required_vars if df[var].isna().all()]
        if empty_vars:
            check_result['empty_vars'] = empty_vars
            check_result['status_message'] = f'变量数据全为空: {empty_vars}'
            return check_result, df
        
        # 数据检查通过
        check_result['data_available'] = True
        check_result['status_message'] = '数据检查通过'
        
        return check_result, df
    
    def _create_empty_result(self, model_name: str, status_message: str) -> Dict[str, Any]:
        """
        创建空结果字典
        
        Args:
            model_name: 模型名称
            status_message: 状态信息
            
        Returns:
            空结果字典
        """
        return {
            'model_name': model_name,
            'model_type': self.config.analysis.RESEARCH_MODELS.get(model_name, {}).get('method', 'unknown'),
            'status': 'failed',
            'status_message': status_message,
            'coefficients': {},
            'std_errors': {},
            'p_values': {},
            'r_squared': np.nan,
            'n_obs': 0,
            'n_entities': 0,
            'diagnostics': {},
            'estimation_time': 0.0,
            'formula': self.config.analysis.RESEARCH_MODELS.get(model_name, {}).get('formula', ''),
            'data_available': False
        }
    
    def run_dli_hhi_association(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        核心模型: DLI-HHI关联检验（锚定多样化假说）
        双向固定效应面板模型: HHI_{i,t} = β·NodeDLI^{US}_{i,t} + Γ·Controls_{i,t} + α_i + λ_t + ε_{i,t}
        
        研究问题: 一个国家与美国建立更紧密的能源网络关系，是否有助于其整体能源进口来源的多元化？
        
        理论预期: β < 0 (与美国关系越紧密，进口来源越多样化，HHI越低)
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_1_dli_hhi_association'
        logger.info(f"🔍 运行核心模型1: DLI-HHI关联检验（锚定多样化假说）...")
        
        # 动态构建必需变量列表  
        dependent_var = 'hhi_imports'
        explanatory_vars = ['node_dli_us'] + self.config.get_control_variables('macro_controls')
        required_vars = [dependent_var] + explanatory_vars + ['year', 'country']
        
        # 检查数据可用性 (只检查存在性，不执行dropna)
        data_check, df_checked = self._check_data_availability(df, required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        # 创建该模型的专用数据子集
        analysis_data = df_checked[required_vars].dropna().copy()
        
        # 记录数据处理的影响
        logger.info(f"   Data for Model 1: Started with {len(df_checked)} obs, using {len(analysis_data)} after handling missing values.")
        
        # 检查处理后的观测数是否足够
        if len(analysis_data) < self.config.validation.DATA_QUALITY_THRESHOLDS['min_observations']:
            error_msg = f'处理缺失值后观测数不足: {len(analysis_data)} < {self.config.validation.DATA_QUALITY_THRESHOLDS["min_observations"]}'
            logger.warning(f"   ⚠️ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
        
        # 如果没有statsmodels或linearmodels，返回空结果
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 设置面板数据索引
            analysis_data = analysis_data.set_index(['country', 'year'])
            
            # 动态构建公式 (用于记录目的)
            formula = f"{dependent_var} ~ {' + '.join(explanatory_vars)} + EntityEffects + TimeEffects"
            
            # 运行双向固定效应模型（聚类稳健标准误）
            logger.info(f"   估计锚定多样化假说模型...")
            
            model = PanelOLS(
                dependent=analysis_data[dependent_var],
                exog=analysis_data[explanatory_vars],
                entity_effects=True,    # 个体固定效应
                time_effects=True,      # 时间固定效应
                check_rank=False        # 跳过rank检查以处理小样本
            )
            
            # 使用按国家聚类的稳健标准误来校正国家内观测值的序列相关性
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            # 提取结果
            result_dict = {
                'model_name': model_name,
                'model_type': 'two_way_fixed_effects',
                'status': 'success',
                'status_message': '模型估计成功',
                'coefficients': dict(results.params),
                'std_errors': dict(results.std_errors),
                'p_values': dict(results.pvalues),
                'r_squared': float(results.rsquared),
                'r_squared_within': float(results.rsquared_within) if hasattr(results, 'rsquared_within') else np.nan,
                'n_obs': int(results.nobs),
                'n_entities': len(analysis_data.index.get_level_values('country').unique()),
                'f_statistic': float(results.f_statistic.stat) if hasattr(results, 'f_statistic') else np.nan,
                'formula': formula,
                'data_available': True,
                'estimation_time': 0.0,  # 可以添加计时功能
                'diagnostics': self._run_model_diagnostics(analysis_data, results)
            }
            
            logger.info(f"   ✅ 核心模型估计完成: R²={result_dict['r_squared']:.4f}, N={result_dict['n_obs']}")
            
            # 检验锚定多样化假说
            node_dli_coef = result_dict['coefficients'].get('node_dli_us', np.nan)
            if not np.isnan(node_dli_coef):
                hypothesis_supported = node_dli_coef < 0
                logger.info(f"   锚定多样化假说检验: β={node_dli_coef:.4f}, 假说{'支持' if hypothesis_supported else '不支持'}")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"模型估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
    
    
    def run_local_projection_shock_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        模型3: 局部投影因果验证
        局部投影模型: ΔY(t+h) ~ US_ProdShock(t) * OVI_gas(t-1) + Controls
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_3_local_projection_validation'
        logger.info(f"🔍 运行模型3: 局部投影因果验证...")
        
        # 动态构建必需变量列表 - 重构后使用hhi_imports替代vul_us
        base_required_vars = ['hhi_imports', 'us_prod_shock', 'ovi_gas', 'year', 'country'] + self.config.get_control_variables('macro_controls')
        
        # 检查数据可用性 (只检查存在性，不执行dropna)
        data_check, df_checked = self._check_data_availability(df, base_required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 准备数据并创建转换变量 (在dropna之前)
            analysis_data = df_checked[base_required_vars].copy()
            analysis_data = analysis_data.sort_values(['country', 'year'])
            
            # 创建滞后变量和未来变化
            analysis_data['ovi_gas_lag1'] = analysis_data.groupby('country')['ovi_gas'].shift(1)
            
            # 获取预测期数设定
            horizons = [0, 1, 2, 3]  # 默认预测期数
            
            # 为不同预测期创建因变量 - 使用hhi_imports替代vul_us
            for h in horizons:
                if h == 0:
                    analysis_data[f'delta_hhi_h{h}'] = analysis_data.groupby('country')['hhi_imports'].diff()
                else:
                    analysis_data[f'delta_hhi_h{h}'] = (
                        analysis_data.groupby('country')['hhi_imports'].shift(-h) - 
                        analysis_data['hhi_imports']
                    )
            
            # 创建交互项
            analysis_data['us_prod_shock_x_ovi_gas_lag1'] = (
                analysis_data['us_prod_shock'] * analysis_data['ovi_gas_lag1']
            )
            
            # 构建基础解释变量列表 (用于所有预测期)
            base_explanatory_vars = ['us_prod_shock', 'ovi_gas_lag1', 'us_prod_shock_x_ovi_gas_lag1'] + self.config.get_control_variables('macro_controls')
            
            # 删除基础变量的缺失数据
            analysis_data = analysis_data.dropna(subset=base_explanatory_vars)
            
            # 记录数据处理的影响
            logger.info(f"   Data for Model 3: Started with {len(df_checked)} obs, using {len(analysis_data)} after handling missing values.")
            
            if len(analysis_data) < self.config.validation.DATA_QUALITY_THRESHOLDS['min_observations']:
                error_msg = f'创建局部投影变量后观测数不足: {len(analysis_data)} < {self.config.validation.DATA_QUALITY_THRESHOLDS["min_observations"]}'
                logger.warning(f"   ⚠️ {error_msg}")
                return self._create_empty_result(model_name, error_msg)
            
            logger.info(f"   估计局部投影模型，预测期数: {horizons}")
            
            # 运行不同时间窗口的局部投影
            horizon_results = {}
            overall_diagnostics = {}
            
            for h in horizons:
                dependent_var = f'delta_hhi_h{h}'
                
                # 检查该期数的因变量是否可用
                if dependent_var not in analysis_data.columns or analysis_data[dependent_var].isna().all():
                    logger.warning(f"   跳过预测期 h={h}: 因变量不可用")
                    continue
                
                # 准备该期数的数据
                horizon_data = analysis_data.dropna(subset=[dependent_var])
                
                if len(horizon_data) < 20:  # 最少观测数
                    logger.warning(f"   跳过预测期 h={h}: 观测数不足({len(horizon_data)})")
                    continue
                
                # 设置索引
                horizon_data = horizon_data.set_index(['country', 'year'])
                
                # 估计模型
                explanatory_vars = base_explanatory_vars
                
                try:
                    model = PanelOLS(
                        dependent=horizon_data[dependent_var],
                        exog=horizon_data[explanatory_vars],
                        entity_effects=True,
                        time_effects=False,  # 局部投影通常不包含时间效应
                        check_rank=False
                    )
                    
                    results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    horizon_results[f'h{h}'] = {
                        'horizon': h,
                        'coefficients': dict(results.params),
                        'std_errors': dict(results.std_errors),
                        'p_values': dict(results.pvalues),
                        'r_squared': float(results.rsquared),
                        'n_obs': int(results.nobs)
                    }
                    
                    logger.info(f"     ✓ h={h}: R²={results.rsquared:.4f}, N={results.nobs}")
                    
                except Exception as e:
                    logger.warning(f"     ✗ h={h}: 估计失败 - {str(e)}")
                    continue
            
            if not horizon_results:
                return self._create_empty_result(model_name, '所有预测期数的模型估计都失败')
            
            # 构建动态公式字符串
            formula = f"Δhhi_imports(t+h) ~ {' + '.join(base_explanatory_vars)} + EntityEffects"
            
            # 聚合结果
            result_dict = {
                'model_name': model_name,
                'model_type': 'local_projections',
                'status': 'success',
                'status_message': f'局部投影模型估计成功，{len(horizon_results)}个预测期',
                'horizon_results': horizon_results,
                'horizons_estimated': list(horizon_results.keys()),
                'n_horizons': len(horizon_results),
                'formula': formula,
                'data_available': True,
                'estimation_time': 0.0,
                'diagnostics': overall_diagnostics
            }
            
            logger.info(f"   ✅ 模型3估计完成: {len(horizon_results)} 个预测期")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"局部投影模型估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)

    def run_lp_irf_price_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        模型4A: LP-IRF价格通道模型 (第4章物理本质检验)
        P_{i,t+h} = θ_h * (US_ProdShock_t × OVI_{i,t-1}) + Controls + α_i + λ_t + η_{i,t+h}
        
        预期结果: θ_h 显著为负 (OVI高的国家在美国供应增加冲击下价格下降更多)
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_4a_lp_irf_price_channel'
        logger.info(f"🔍 运行模型4A: LP-IRF价格通道检验...")
        
        # 动态构建必需变量列表
        base_required_vars = ['P_it', 'us_prod_shock', 'ovi_gas', 'year', 'country'] + self.config.get_control_variables('macro_controls')
        
        # 检查数据可用性
        data_check, df_checked = self._check_data_availability(df, base_required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 准备数据
            analysis_data = df_checked[base_required_vars].copy()
            analysis_data = analysis_data.sort_values(['country', 'year'])
            
            # 创建滞后OVI变量
            analysis_data['ovi_gas_lag1'] = analysis_data.groupby('country')['ovi_gas'].shift(1)
            
            # 创建交互项: US_ProdShock × OVI(t-1)
            analysis_data['shock_x_ovi'] = (
                analysis_data['us_prod_shock'] * analysis_data['ovi_gas_lag1']
            )
            
            # 获取预测期数设定 (0到4年)
            horizons = [0, 1, 2, 3, 4]
            
            # 为不同预测期创建因变量 P_{i,t+h}
            for h in horizons:
                if h == 0:
                    analysis_data[f'P_it_h{h}'] = analysis_data['P_it']
                else:
                    analysis_data[f'P_it_h{h}'] = analysis_data.groupby('country')['P_it'].shift(-h)
            
            # 获取控制变量
            control_vars = self.config.get_control_variables('macro_controls')
            available_controls = [var for var in control_vars if var in analysis_data.columns]
            
            # 构建解释变量列表
            explanatory_vars = ['shock_x_ovi'] + available_controls
            required_for_regression = explanatory_vars + ['ovi_gas_lag1', 'year', 'country']
            
            # 最终清理数据
            for h in horizons:
                required_for_regression.append(f'P_it_h{h}')
            
            final_data = analysis_data[required_for_regression].dropna()
            
            if len(final_data) < 50:  # 最小样本量要求
                error_msg = f'样本量不足: {len(final_data)} < 50'
                return self._create_empty_result(model_name, error_msg)
            
            # 设置面板数据索引
            final_data = final_data.set_index(['country', 'year'])
            
            horizon_results = {}
            
            # 对每个预测期运行回归
            for h in horizons:
                try:
                    logger.info(f"   估计预测期 h={h}...")
                    
                    # 运行双向固定效应模型
                    model = PanelOLS(
                        dependent=final_data[f'P_it_h{h}'],
                        exog=final_data[explanatory_vars],
                        entity_effects=True,    # 国家固定效应
                        time_effects=True,      # 年份固定效应
                        check_rank=False
                    )
                    
                    results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    # 提取核心系数 θ_h (交互项系数)
                    theta_h = results.params['shock_x_ovi']
                    theta_h_se = results.std_errors['shock_x_ovi']
                    theta_h_pvalue = results.pvalues['shock_x_ovi']
                    
                    horizon_results[f'h{h}'] = {
                        'horizon': h,
                        'theta_coefficient': float(theta_h),
                        'theta_std_error': float(theta_h_se),
                        'theta_p_value': float(theta_h_pvalue),
                        'theta_significant': theta_h_pvalue < 0.05,
                        'theta_sign_correct': theta_h < 0,  # 预期为负
                        'r_squared': float(results.rsquared),
                        'n_obs': int(results.nobs),
                        'all_coefficients': dict(results.params),
                        'all_p_values': dict(results.pvalues)
                    }
                    
                    logger.info(f"     h={h}: θ_h={theta_h:.4f} (p={theta_h_pvalue:.3f})")
                    
                except Exception as e:
                    logger.warning(f"     预测期 h={h} 估计失败: {str(e)}")
                    continue
            
            if not horizon_results:
                return self._create_empty_result(model_name, '所有预测期估计失败')
            
            # 汇总结果
            formula = f"P_it(t+h) ~ shock_x_ovi + {' + '.join(available_controls)} + EntityEffects + TimeEffects"
            
            result_dict = {
                'model_name': model_name,
                'model_type': 'lp_irf_price_channel',
                'status': 'success',
                'status_message': f'LP-IRF价格通道模型估计成功，{len(horizon_results)}个预测期',
                'horizon_results': horizon_results,
                'horizons_estimated': [int(k[1:]) for k in horizon_results.keys()],
                'n_horizons': len(horizon_results),
                'formula': formula,
                'economic_interpretation': '负的θ_h表示OVI高的国家在美国供应冲击下价格下降更多',
                'expected_sign': 'negative',
                'data_available': True,
                'sample_period': f"{final_data.index.get_level_values('year').min()}-{final_data.index.get_level_values('year').max()}",
                'total_observations': len(final_data)
            }
            
            logger.info(f"   ✅ LP-IRF价格通道模型完成: {len(horizon_results)} 个预测期")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"LP-IRF价格通道模型估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)

    def run_lp_irf_quantity_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        模型4B: LP-IRF数量通道模型 (第4章物理本质检验)
        g_{i,t+h} = θ_h * (US_ProdShock_t × OVI_{i,t-1}) + Controls + α_i + λ_t + η_{i,t+h}
        
        预期结果: θ_h 显著为正 (OVI高的国家在美国供应增加冲击下进口量增长更多)
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_4b_lp_irf_quantity_channel'
        logger.info(f"🔍 运行模型4B: LP-IRF数量通道检验...")
        
        # 动态构建必需变量列表
        base_required_vars = ['g_it', 'us_prod_shock', 'ovi_gas', 'year', 'country'] + self.config.get_control_variables('macro_controls')
        
        # 检查数据可用性
        data_check, df_checked = self._check_data_availability(df, base_required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 准备数据
            analysis_data = df_checked[base_required_vars].copy()
            analysis_data = analysis_data.sort_values(['country', 'year'])
            
            # 创建滞后OVI变量
            analysis_data['ovi_gas_lag1'] = analysis_data.groupby('country')['ovi_gas'].shift(1)
            
            # 创建交互项: US_ProdShock × OVI(t-1)
            analysis_data['shock_x_ovi'] = (
                analysis_data['us_prod_shock'] * analysis_data['ovi_gas_lag1']
            )
            
            # 获取预测期数设定 (0到4年)
            horizons = [0, 1, 2, 3, 4]
            
            # 为不同预测期创建因变量 g_{i,t+h}
            for h in horizons:
                if h == 0:
                    analysis_data[f'g_it_h{h}'] = analysis_data['g_it']
                else:
                    analysis_data[f'g_it_h{h}'] = analysis_data.groupby('country')['g_it'].shift(-h)
            
            # 获取控制变量
            control_vars = self.config.get_control_variables('macro_controls')
            available_controls = [var for var in control_vars if var in analysis_data.columns]
            
            # 构建解释变量列表
            explanatory_vars = ['shock_x_ovi'] + available_controls
            required_for_regression = explanatory_vars + ['ovi_gas_lag1', 'year', 'country']
            
            # 最终清理数据
            for h in horizons:
                required_for_regression.append(f'g_it_h{h}')
            
            final_data = analysis_data[required_for_regression].dropna()
            
            if len(final_data) < 50:  # 最小样本量要求
                error_msg = f'样本量不足: {len(final_data)} < 50'
                return self._create_empty_result(model_name, error_msg)
            
            # 设置面板数据索引
            final_data = final_data.set_index(['country', 'year'])
            
            horizon_results = {}
            
            # 对每个预测期运行回归
            for h in horizons:
                try:
                    logger.info(f"   估计预测期 h={h}...")
                    
                    # 运行双向固定效应模型
                    model = PanelOLS(
                        dependent=final_data[f'g_it_h{h}'],
                        exog=final_data[explanatory_vars],
                        entity_effects=True,    # 国家固定效应
                        time_effects=True,      # 年份固定效应
                        check_rank=False
                    )
                    
                    results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    # 提取核心系数 θ_h (交互项系数)
                    theta_h = results.params['shock_x_ovi']
                    theta_h_se = results.std_errors['shock_x_ovi']
                    theta_h_pvalue = results.pvalues['shock_x_ovi']
                    
                    horizon_results[f'h{h}'] = {
                        'horizon': h,
                        'theta_coefficient': float(theta_h),
                        'theta_std_error': float(theta_h_se),
                        'theta_p_value': float(theta_h_pvalue),
                        'theta_significant': theta_h_pvalue < 0.05,
                        'theta_sign_correct': theta_h > 0,  # 预期为正
                        'r_squared': float(results.rsquared),
                        'n_obs': int(results.nobs),
                        'all_coefficients': dict(results.params),
                        'all_p_values': dict(results.pvalues)
                    }
                    
                    logger.info(f"     h={h}: θ_h={theta_h:.4f} (p={theta_h_pvalue:.3f})")
                    
                except Exception as e:
                    logger.warning(f"     预测期 h={h} 估计失败: {str(e)}")
                    continue
            
            if not horizon_results:
                return self._create_empty_result(model_name, '所有预测期估计失败')
            
            # 汇总结果
            formula = f"g_it(t+h) ~ shock_x_ovi + {' + '.join(available_controls)} + EntityEffects + TimeEffects"
            
            result_dict = {
                'model_name': model_name,
                'model_type': 'lp_irf_quantity_channel',
                'status': 'success',
                'status_message': f'LP-IRF数量通道模型估计成功，{len(horizon_results)}个预测期',
                'horizon_results': horizon_results,
                'horizons_estimated': [int(k[1:]) for k in horizon_results.keys()],
                'n_horizons': len(horizon_results),
                'formula': formula,
                'economic_interpretation': '正的θ_h表示OVI高的国家在美国供应冲击下进口量增长更多',
                'expected_sign': 'positive',
                'data_available': True,
                'sample_period': f"{final_data.index.get_level_values('year').min()}-{final_data.index.get_level_values('year').max()}",
                'total_observations': len(final_data)
            }
            
            logger.info(f"   ✅ LP-IRF数量通道模型完成: {len(horizon_results)} 个预测期")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"LP-IRF数量通道模型估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
    
    def _run_model_diagnostics(self, data: pd.DataFrame, results) -> Dict[str, Any]:
        """
        运行模型诊断检验
        
        Args:
            data: 分析数据
            results: 模型结果
            
        Returns:
            诊断结果字典
        """
        diagnostics = {}
        
        try:
            # 基础统计
            diagnostics['n_obs'] = len(data)
            diagnostics['n_vars'] = len(data.columns)
            
            # 如果有残差，进行进一步诊断
            if hasattr(results, 'resids'):
                residuals = results.resids
                
                # 残差基础统计
                diagnostics['residual_mean'] = float(residuals.mean())
                diagnostics['residual_std'] = float(residuals.std())
                
                # Durbin-Watson检验 (如果有statsmodels)
                if HAS_STATSMODELS:
                    try:
                        diagnostics['durbin_watson'] = float(durbin_watson(residuals))
                    except:
                        diagnostics['durbin_watson'] = np.nan
            
        except Exception as e:
            logger.warning(f"诊断检验失败: {str(e)}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def run_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        运行核心模型1: DLI-HHI关联分析
        
        Args:
            df: 分析数据
            
        Returns:
            模型1结果的汇总字典
        """
        logger.info("🚀 运行核心模型1: DLI-HHI关联分析...")
        
        all_results = {
            'overview': {
                'total_models': 1,  # 只有1个核心模型
                'completed_models': 0,
                'failed_models': 0,
                'data_available': len(df) > 0 if df is not None else False
            },
            'models': {}
        }
        
        # 模型1: DLI-HHI关联（唯一核心模型）
        try:
            result1 = self.run_dli_hhi_association(df)
            all_results['models']['model_1_dli_hhi_association'] = result1
            if result1['status'] == 'success':
                all_results['overview']['completed_models'] += 1
            else:
                all_results['overview']['failed_models'] += 1
        except Exception as e:
            logger.error(f"核心模型运行异常: {str(e)}")
            all_results['models']['model_1_dli_hhi_association'] = self._create_empty_result('model_1_dli_hhi_association', f'运行异常: {str(e)}')
            all_results['overview']['failed_models'] += 1
        
        logger.info(f"✅ 核心模型运行完成: 成功 {all_results['overview']['completed_models']}/{all_results['overview']['total_models']}")
        
        return all_results

    def run_surface_association_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        第3章：表面关联检验 - Node-DLI_US与HHI的关联性测试
        
        模型设定：
        HHI_{i,t} = β × NodeDLI^US_{i,t} + Γ × Controls_{i,t} + α_i + λ_t + ε_{i,t}
        
        锚定多样化假说：β < 0 (与美国关系越紧密，进口来源越多样化)
        
        Args:
            df: 分析数据集
            
        Returns:
            Dict包含估计结果和统计信息
        """
        model_name = 'model_3_surface_association'
        logger.info(f"🎯 开始运行表面关联检验: {model_name}")
        
        # 检查数据可用性  
        required_variables = ['hhi_imports', 'node_dli_us', 'log_gdp', 'log_population', 'year', 'country']
        check_result, analysis_data = self._check_data_availability(df, required_variables)
        
        if not check_result['data_available']:
            return self._create_empty_result(model_name, check_result['message'])
        
        logger.info(f"   数据检查通过: {len(analysis_data)} 观测值")
        
        try:
            # 去除缺失值
            clean_data = analysis_data.dropna(subset=['hhi_imports', 'node_dli_us', 'log_gdp', 'log_population'])
            
            if len(clean_data) < 20:
                return self._create_empty_result(model_name, f'有效观测数不足: {len(clean_data)}')
            
            logger.info(f"   清理后样本量: {len(clean_data)} 观测值")
            
            # 设置面板索引
            clean_data = clean_data.set_index(['country', 'year'])
            
            # 定义解释变量
            explanatory_vars = ['node_dli_us', 'log_gdp', 'log_population']
            
            # 估计双向固定效应模型
            model = PanelOLS(
                dependent=clean_data['hhi_imports'],
                exog=clean_data[explanatory_vars],
                entity_effects=True,   # 国家固定效应
                time_effects=True,     # 年份固定效应
                check_rank=False
            )
            
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            # 提取关键系数
            node_dli_coef = float(results.params.get('node_dli_us', np.nan))
            node_dli_pval = float(results.pvalues.get('node_dli_us', 1.0))
            node_dli_stderr = float(results.std_errors.get('node_dli_us', np.nan))
            
            # 构建结果
            result_dict = {
                'model_name': model_name,
                'status': 'success',
                'status_message': f'表面关联检验完成，样本量{results.nobs}',
                'model_formula': 'hhi_imports ~ node_dli_us + log_gdp + log_population + EntityEffects + TimeEffects',
                'sample_size': int(results.nobs),
                'r_squared': float(results.rsquared),
                'node_dli_coefficient': node_dli_coef,
                'node_dli_p_value': node_dli_pval,
                'node_dli_std_error': node_dli_stderr,
                'expected_sign_correct': node_dli_coef < 0,  # 锚定多样化假说预期为负
                'all_coefficients': dict(results.params),
                'all_p_values': dict(results.pvalues),
                'all_std_errors': dict(results.std_errors),
                'economic_interpretation': self._get_surface_association_interpretation(node_dli_coef, node_dli_pval)
            }
            
            logger.info(f"   ✅ 表面关联检验完成: β={node_dli_coef:.4f} (p={node_dli_pval:.3f})")
            return result_dict
            
        except Exception as e:
            logger.error(f"   ❌ 表面关联检验失败: {str(e)}")
            return self._create_empty_result(model_name, f'估计过程失败: {str(e)}')

    def _get_surface_association_interpretation(self, beta: float, p_value: float) -> str:
        """生成表面关联检验的经济学解读（锚定多样化假说）"""
        significance = "显著" if p_value < 0.05 else "不显著" if p_value < 0.1 else "不显著"
        sign = "负" if beta < 0 else "正"
        
        if beta < 0:
            base_interpretation = f"Node-DLI_US对hhi_imports的影响为{sign}({significance})，支持锚定多样化假说：与美国建立更紧密网络关系的国家，进口来源更加多样化（hhi_imports更低）。"
        else:
            base_interpretation = f"Node-DLI_US对hhi_imports的影响为{sign}({significance})，不支持锚定多样化假说，可能存在其他经济机制或需要考虑内生性问题。"
        
        return base_interpretation


# 便捷函数
def run_single_model(model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    运行单个模型的便捷函数
    
    Args:
        model_name: 模型名称
        df: 分析数据
        
    Returns:
        模型结果
    """
    models = EconometricModels()
    
    if model_name == 'model_1_dli_hhi_association':
        return models.run_dli_hhi_association(df)
    elif model_name == 'model_2_ovi_dli_causality':
        return models.run_ovi_dli_causality(df)
    elif model_name == 'model_3_local_projection_validation':
        return models.run_local_projection_shock_validation(df)
    elif model_name == 'model_4a_lp_irf_price_channel':
        return models.run_lp_irf_price_channel(df)
    elif model_name == 'model_4b_lp_irf_quantity_channel':
        return models.run_lp_irf_quantity_channel(df)
    elif model_name == 'model_3_surface_association':
        return models.run_surface_association_test(df)
    else:
        raise ValueError(f"未知模型名称: {model_name}")


if __name__ == "__main__":
    # 测试模型模块
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🔬 09_econometric_analysis 模型模块测试")
    print("=" * 50)
    
    # 创建空DataFrame测试
    test_df = pd.DataFrame()
    
    models = EconometricModels()
    results = models.run_all_models(test_df)
    
    print(f"\n📊 测试结果:")
    print(f"总模型数: {results['overview']['total_models']}")
    print(f"完成模型数: {results['overview']['completed_models']}")
    print(f"失败模型数: {results['overview']['failed_models']}")
    
    for model_name, result in results['models'].items():
        print(f"\n• {model_name}:")
        print(f"  状态: {result['status']}")
        print(f"  信息: {result['status_message']}")
    
    print("\n🎉 模型模块测试完成!")