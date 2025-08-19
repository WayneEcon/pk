#!/usr/bin/env python3
"""
计量模型模块 (Econometric Models Module)
=====================================

09_econometric_analysis 模块的核心计量模型实现

作者：Energy Network Analysis Team
版本：v1.0 - 计量分析框架
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

from .config import config

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
    
    def _check_data_availability(self, df: pd.DataFrame, required_vars: List[str]) -> Dict[str, Any]:
        """
        检查数据可用性
        
        Args:
            df: 输入数据
            required_vars: 必需变量列表
            
        Returns:
            数据可用性检查结果
        """
        check_result = {
            'data_available': False,
            'missing_vars': [],
            'empty_vars': [],
            'total_obs': 0,
            'usable_obs': 0,
            'status_message': ''
        }
        
        # 基础检查：数据是否为空
        if df is None or len(df) == 0:
            check_result['status_message'] = '数据集为空或不存在'
            return check_result
        
        check_result['total_obs'] = len(df)
        
        # 检查变量是否存在
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            check_result['missing_vars'] = missing_vars
            check_result['status_message'] = f'缺少必需变量: {missing_vars}'
            return check_result
        
        # 检查变量是否全为空
        empty_vars = [var for var in required_vars if df[var].isna().all()]
        if empty_vars:
            check_result['empty_vars'] = empty_vars
            check_result['status_message'] = f'变量数据全为空: {empty_vars}'
            return check_result
        
        # 检查可用观测数
        subset_df = df[required_vars].dropna()
        check_result['usable_obs'] = len(subset_df)
        
        if check_result['usable_obs'] < self.config.validation.DATA_QUALITY_THRESHOLDS['min_observations']:
            check_result['status_message'] = f'可用观测数不足: {check_result["usable_obs"]} < {self.config.validation.DATA_QUALITY_THRESHOLDS["min_observations"]}'
            return check_result
        
        # 数据可用
        check_result['data_available'] = True
        check_result['status_message'] = '数据可用于分析'
        
        return check_result
    
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
    
    def run_dli_vul_association(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        模型1: DLI-脆弱性关联检验
        双向固定效应面板模型: vul_us ~ node_dli_us + Controls + FE(country,year)
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_1_dli_vul_association'
        logger.info(f"🔍 运行模型1: DLI-脆弱性关联检验...")
        
        # 定义必需变量
        required_vars = ['vul_us', 'node_dli_us', 'year', 'country'] + self.config.get_control_variables('macro_controls')
        
        # 检查数据可用性
        data_check = self._check_data_availability(df, required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        logger.info(f"   数据检查通过: {data_check['usable_obs']} 个可用观测")
        
        # 如果没有statsmodels或linearmodels，返回空结果
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 准备数据
            analysis_data = df[required_vars].dropna().copy()
            
            # 设置面板数据索引
            analysis_data = analysis_data.set_index(['country', 'year'])
            
            # 构建模型公式
            dependent_var = 'vul_us'
            explanatory_vars = ['node_dli_us'] + self.config.get_control_variables('macro_controls')
            
            # 运行双向固定效应模型
            logger.info(f"   估计双向固定效应模型...")
            
            model = PanelOLS(
                dependent=analysis_data[dependent_var],
                exog=analysis_data[explanatory_vars],
                entity_effects=True,    # 个体固定效应
                time_effects=True,      # 时间固定效应
                check_rank=False        # 跳过rank检查以处理小样本
            )
            
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
                'formula': f"{dependent_var} ~ {' + '.join(explanatory_vars)} + EntityEffects + TimeEffects",
                'data_available': True,
                'estimation_time': 0.0,  # 可以添加计时功能
                'diagnostics': self._run_model_diagnostics(analysis_data, results)
            }
            
            logger.info(f"   ✅ 模型1估计完成: R²={result_dict['r_squared']:.4f}, N={result_dict['n_obs']}")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"模型估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
    
    def run_ovi_dli_causality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        模型2: OVI对DLI的因果效应
        双向固定效应面板模型: node_dli_us ~ ovi(t-1) + Controls + FE(country,year)
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_2_ovi_dli_causality'
        logger.info(f"🔍 运行模型2: OVI对DLI的因果效应...")
        
        # 定义必需变量
        required_vars = ['node_dli_us', 'ovi', 'year', 'country'] + self.config.get_control_variables('macro_controls')
        
        # 检查数据可用性
        data_check = self._check_data_availability(df, required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        logger.info(f"   数据检查通过: {data_check['usable_obs']} 个可用观测")
        
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 准备数据并创建滞后变量
            analysis_data = df[required_vars].dropna().copy()
            analysis_data = analysis_data.sort_values(['country', 'year'])
            
            # 创建OVI的滞后项
            analysis_data['ovi_lag1'] = analysis_data.groupby('country')['ovi'].shift(1)
            
            # 删除无法计算滞后的观测
            analysis_data = analysis_data.dropna(subset=['ovi_lag1'])
            
            if len(analysis_data) < self.config.validation.DATA_QUALITY_THRESHOLDS['min_observations']:
                return self._create_empty_result(model_name, '创建滞后变量后观测数不足')
            
            # 设置面板数据索引
            analysis_data = analysis_data.set_index(['country', 'year'])
            
            # 构建模型
            dependent_var = 'node_dli_us'
            explanatory_vars = ['ovi_lag1'] + self.config.get_control_variables('macro_controls')
            
            logger.info(f"   估计OVI滞后效应模型...")
            
            model = PanelOLS(
                dependent=analysis_data[dependent_var],
                exog=analysis_data[explanatory_vars],
                entity_effects=True,
                time_effects=True,
                check_rank=False
            )
            
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            # 提取结果
            result_dict = {
                'model_name': model_name,
                'model_type': 'two_way_fixed_effects_lagged',
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
                'formula': f"{dependent_var} ~ {' + '.join(explanatory_vars)} + EntityEffects + TimeEffects",
                'data_available': True,
                'lag_structure': 'ovi_lag1',
                'estimation_time': 0.0,
                'diagnostics': self._run_model_diagnostics(analysis_data, results)
            }
            
            logger.info(f"   ✅ 模型2估计完成: R²={result_dict['r_squared']:.4f}, N={result_dict['n_obs']}")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"模型估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
    
    def run_local_projection_shock_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        模型3: 局部投影因果验证
        局部投影模型: ΔY(t+h) ~ US_ProdShock(t) * OVI(t-1) + Controls
        
        Args:
            df: 分析数据
            
        Returns:
            模型结果字典
        """
        model_name = 'model_3_local_projection_validation'
        logger.info(f"🔍 运行模型3: 局部投影因果验证...")
        
        # 定义必需变量
        required_vars = ['vul_us', 'us_prod_shock', 'ovi', 'year', 'country'] + self.config.get_control_variables('macro_controls')
        
        # 检查数据可用性
        data_check = self._check_data_availability(df, required_vars)
        if not data_check['data_available']:
            logger.warning(f"   ⚠️ {data_check['status_message']}")
            return self._create_empty_result(model_name, data_check['status_message'])
        
        logger.info(f"   数据检查通过: {data_check['usable_obs']} 个可用观测")
        
        if not HAS_STATSMODELS or not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, '缺少必需的计量分析库')
        
        try:
            # 准备数据
            analysis_data = df[required_vars].dropna().copy()
            analysis_data = analysis_data.sort_values(['country', 'year'])
            
            # 创建滞后变量和未来变化
            analysis_data['ovi_lag1'] = analysis_data.groupby('country')['ovi'].shift(1)
            
            # 获取预测期数设定
            horizons = self.config.analysis.RESEARCH_MODELS[model_name].get('horizons', [0, 1, 2, 3])
            
            # 为不同预测期创建因变量
            for h in horizons:
                if h == 0:
                    analysis_data[f'delta_vul_h{h}'] = analysis_data.groupby('country')['vul_us'].diff()
                else:
                    analysis_data[f'delta_vul_h{h}'] = (
                        analysis_data.groupby('country')['vul_us'].shift(-h) - 
                        analysis_data['vul_us']
                    )
            
            # 创建交互项
            analysis_data['us_prod_shock_x_ovi_lag1'] = (
                analysis_data['us_prod_shock'] * analysis_data['ovi_lag1']
            )
            
            # 删除缺失数据
            required_for_lp = ['us_prod_shock', 'ovi_lag1', 'us_prod_shock_x_ovi_lag1'] + self.config.get_control_variables('macro_controls')
            analysis_data = analysis_data.dropna(subset=required_for_lp)
            
            if len(analysis_data) < self.config.validation.DATA_QUALITY_THRESHOLDS['min_observations']:
                return self._create_empty_result(model_name, '创建局部投影变量后观测数不足')
            
            logger.info(f"   估计局部投影模型，预测期数: {horizons}")
            
            # 运行不同时间窗口的局部投影
            horizon_results = {}
            overall_diagnostics = {}
            
            for h in horizons:
                dependent_var = f'delta_vul_h{h}'
                
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
                explanatory_vars = ['us_prod_shock', 'ovi_lag1', 'us_prod_shock_x_ovi_lag1'] + self.config.get_control_variables('macro_controls')
                
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
            
            # 聚合结果
            result_dict = {
                'model_name': model_name,
                'model_type': 'local_projections',
                'status': 'success',
                'status_message': f'局部投影模型估计成功，{len(horizon_results)}个预测期',
                'horizon_results': horizon_results,
                'horizons_estimated': list(horizon_results.keys()),
                'n_horizons': len(horizon_results),
                'formula': f"Δvul_us(t+h) ~ us_prod_shock(t) * ovi_lag1(t-1) + Controls + EntityEffects",
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
        运行所有核心模型
        
        Args:
            df: 分析数据
            
        Returns:
            所有模型结果的汇总字典
        """
        logger.info("🚀 开始运行所有核心计量模型...")
        
        all_results = {
            'overview': {
                'total_models': 3,
                'completed_models': 0,
                'failed_models': 0,
                'data_available': len(df) > 0 if df is not None else False
            },
            'models': {}
        }
        
        # 模型1: DLI-脆弱性关联
        try:
            result1 = self.run_dli_vul_association(df)
            all_results['models']['model_1_dli_vul_association'] = result1
            if result1['status'] == 'success':
                all_results['overview']['completed_models'] += 1
            else:
                all_results['overview']['failed_models'] += 1
        except Exception as e:
            logger.error(f"模型1运行异常: {str(e)}")
            all_results['models']['model_1_dli_vul_association'] = self._create_empty_result('model_1_dli_vul_association', f'运行异常: {str(e)}')
            all_results['overview']['failed_models'] += 1
        
        # 模型2: OVI因果效应
        try:
            result2 = self.run_ovi_dli_causality(df)
            all_results['models']['model_2_ovi_dli_causality'] = result2
            if result2['status'] == 'success':
                all_results['overview']['completed_models'] += 1
            else:
                all_results['overview']['failed_models'] += 1
        except Exception as e:
            logger.error(f"模型2运行异常: {str(e)}")
            all_results['models']['model_2_ovi_dli_causality'] = self._create_empty_result('model_2_ovi_dli_causality', f'运行异常: {str(e)}')
            all_results['overview']['failed_models'] += 1
        
        # 模型3: 局部投影验证
        try:
            result3 = self.run_local_projection_shock_validation(df)
            all_results['models']['model_3_local_projection_validation'] = result3
            if result3['status'] == 'success':
                all_results['overview']['completed_models'] += 1
            else:
                all_results['overview']['failed_models'] += 1
        except Exception as e:
            logger.error(f"模型3运行异常: {str(e)}")
            all_results['models']['model_3_local_projection_validation'] = self._create_empty_result('model_3_local_projection_validation', f'运行异常: {str(e)}')
            all_results['overview']['failed_models'] += 1
        
        logger.info(f"✅ 所有模型运行完成: 成功 {all_results['overview']['completed_models']}/{all_results['overview']['total_models']}")
        
        return all_results


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
    
    if model_name == 'model_1_dli_vul_association':
        return models.run_dli_vul_association(df)
    elif model_name == 'model_2_ovi_dli_causality':
        return models.run_ovi_dli_causality(df)
    elif model_name == 'model_3_local_projection_validation':
        return models.run_local_projection_shock_validation(df)
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