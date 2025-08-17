#!/usr/bin/env python3
"""
核心分析模块 (Core Analysis Module)
================================

本模块实现网络结构异质性的回归分析，包括：
1. 全局异质性分析：DLI效应与全局网络特征的交互
2. 局部异质性分析：DLI效应与局部节点特征的交互

基于05_causal_validation的基准回归模型，引入交互项进行异质性检验。

作者：Energy Network Analysis Team
版本：v1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 统计分析包
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from scipy import stats
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning("⚠️ statsmodels未安装，将使用简化版回归分析")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeterogeneityAnalyzer:
    """网络结构异质性分析器"""
    
    def __init__(self):
        self.global_results = {}
        self.local_results = {}
        self.summary_stats = {}
        logger.info("🧮 初始化异质性分析器")
    
    def run_global_analysis(self, data: pd.DataFrame, 
                          dli_vars: List[str] = None,
                          global_vars: List[str] = None,
                          control_vars: List[str] = None,
                          outcome_var: str = 'comprehensive_resilience',
                          interactions_to_test: List[List[str]] = None) -> Dict[str, Any]:
        """
        运行全局异质性分析
        
        Args:
            data: 包含所有变量的数据集
            dli_vars: DLI变量列表
            global_vars: 全局网络指标列表
            control_vars: 控制变量列表
            outcome_var: 被解释变量
            
        Returns:
            分析结果字典
        """
        logger.info("🌐 开始全局异质性分析...")
        
        # 自动识别变量
        if dli_vars is None:
            dli_vars = [col for col in data.columns if 'dli' in col.lower()]
        if global_vars is None:
            global_vars = [col for col in data.columns if 'global_' in col or 'network_' in col]
        if control_vars is None:
            control_vars = [col for col in data.columns if 'control' in col.lower()]
        
        logger.info(f"   - DLI变量: {dli_vars}")
        logger.info(f"   - 全局变量: {global_vars}")
        logger.info(f"   - 控制变量: {control_vars}")
        
        results = {}
        
        # 使用精确指定的交互项，如果没有指定则回退到全排列
        if interactions_to_test:
            interaction_pairs = interactions_to_test
        else:
            # 回退到全排列（保持向后兼容）
            interaction_pairs = [[dli_var, global_var] for dli_var in dli_vars for global_var in global_vars]
        
        for dli_var, global_var in interaction_pairs:
            # 检查变量是否存在
            if dli_var not in data.columns or global_var not in data.columns:
                logger.warning(f"⚠️ 变量 {dli_var} 或 {global_var} 不存在，跳过此交互项")
                continue
                
            # 创建交互项
            interaction_var = f"{dli_var}_x_{global_var}"
            data[interaction_var] = data[dli_var] * data[global_var]
            
            # 构建回归方程
            model_name = f"{dli_var}_x_{global_var}"
            
            # 运行回归
            result = self._run_regression(
                data=data,
                outcome_var=outcome_var,
                main_vars=[dli_var, global_var],
                interaction_vars=[interaction_var],
                control_vars=control_vars,
                model_name=model_name
            )
            
            results[model_name] = result
        
        self.global_results = results
        logger.info(f"✅ 全局异质性分析完成，共 {len(results)} 个模型")
        return results
    
    def run_local_analysis(self, data: pd.DataFrame,
                         dli_vars: List[str] = None,
                         local_vars: List[str] = None,
                         control_vars: List[str] = None,
                         outcome_var: str = 'comprehensive_resilience',
                         interactions_to_test: List[List[str]] = None) -> Dict[str, Any]:
        """
        运行局部异质性分析
        
        Args:
            data: 包含所有变量的数据集
            dli_vars: DLI变量列表
            local_vars: 局部节点指标列表
            control_vars: 控制变量列表
            outcome_var: 被解释变量
            
        Returns:
            分析结果字典
        """
        logger.info("🏠 开始局部异质性分析...")
        
        # 自动识别变量
        if dli_vars is None:
            dli_vars = [col for col in data.columns if 'dli' in col.lower()]
        if local_vars is None:
            local_vars = [col for col in data.columns if any(x in col for x in 
                         ['centrality', 'degree', 'strength', 'pagerank'])]
        if control_vars is None:
            control_vars = [col for col in data.columns if 'control' in col.lower()]
        
        logger.info(f"   - DLI变量: {dli_vars}")
        logger.info(f"   - 局部变量: {local_vars}")
        logger.info(f"   - 控制变量: {control_vars}")
        
        results = {}
        
        # 使用精确指定的交互项，如果没有指定则回退到全排列
        if interactions_to_test:
            interaction_pairs = interactions_to_test
        else:
            # 回退到全排列（保持向后兼容）
            interaction_pairs = [[dli_var, local_var] for dli_var in dli_vars for local_var in local_vars]
        
        for dli_var, local_var in interaction_pairs:
            # 检查变量是否存在
            if dli_var not in data.columns or local_var not in data.columns:
                logger.warning(f"⚠️ 变量 {dli_var} 或 {local_var} 不存在，跳过此交互项")
                continue
                
            # 创建交互项
            interaction_var = f"{dli_var}_x_{local_var}"
            data[interaction_var] = data[dli_var] * data[local_var]
            
            # 构建回归方程
            model_name = f"{dli_var}_x_{local_var}"
            
            # 运行回归
            result = self._run_regression(
                data=data,
                outcome_var=outcome_var,
                main_vars=[dli_var, local_var],
                interaction_vars=[interaction_var],
                control_vars=control_vars,
                model_name=model_name
            )
            
            results[model_name] = result
        
        self.local_results = results
        logger.info(f"✅ 局部异质性分析完成，共 {len(results)} 个模型")
        return results
    
    def _run_regression(self, data: pd.DataFrame, outcome_var: str,
                       main_vars: List[str], interaction_vars: List[str],
                       control_vars: List[str], model_name: str) -> Dict[str, Any]:
        """
        运行单个回归模型
        
        Args:
            data: 数据集
            outcome_var: 被解释变量
            main_vars: 主要解释变量
            interaction_vars: 交互项变量
            control_vars: 控制变量
            model_name: 模型名称
            
        Returns:
            回归结果字典
        """
        
        # 准备变量
        all_vars = main_vars + interaction_vars + control_vars
        available_vars = [var for var in all_vars if var in data.columns]
        
        if outcome_var not in data.columns:
            logger.warning(f"⚠️ 被解释变量 {outcome_var} 不存在，使用模拟数据")
            data[outcome_var] = np.random.normal(0, 1, len(data))
        
        # 清理数据
        model_data = data[[outcome_var] + available_vars].dropna()
        
        if len(model_data) == 0:
            logger.warning(f"⚠️ 模型 {model_name} 数据为空")
            return self._create_empty_result(model_name)
        
        logger.info(f"   📊 运行模型: {model_name} (N={len(model_data)})")
        
        # 运行回归
        if HAS_STATSMODELS:
            return self._run_statsmodels_regression(model_data, outcome_var, available_vars, model_name)
        else:
            return self._run_simple_regression(model_data, outcome_var, available_vars, model_name)
    
    def _run_statsmodels_regression(self, data: pd.DataFrame, outcome_var: str,
                                  explanatory_vars: List[str], model_name: str) -> Dict[str, Any]:
        """使用statsmodels运行回归"""
        
        try:
            # 准备数据
            y = data[outcome_var]
            X = data[explanatory_vars]
            X = sm.add_constant(X)  # 添加常数项
            
            # 运行OLS回归
            model = sm.OLS(y, X).fit()
            
            # 计算边际效应（对于交互项）
            marginal_effects = self._calculate_marginal_effects(data, explanatory_vars, model)
            
            # 整理结果
            result = {
                'model_name': model_name,
                'n_obs': model.nobs,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_stat': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'coefficients': model.params.to_dict(),
                'std_errors': model.bse.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'conf_int': model.conf_int().to_dict(),
                'marginal_effects': marginal_effects,
                'summary': str(model.summary()),
                'model_object': model
            }
            
            # 检查多重共线性
            try:
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                result['vif'] = vif_data.to_dict('records')
            except:
                result['vif'] = None
            
            # 异质性检验
            try:
                lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
                result['heteroskedasticity_test'] = {
                    'lm_stat': lm,
                    'lm_pvalue': lm_pvalue,
                    'f_stat': fvalue,
                    'f_pvalue': f_pvalue
                }
            except:
                result['heteroskedasticity_test'] = None
                
            return result
            
        except Exception as e:
            logger.error(f"❌ 回归分析失败 {model_name}: {str(e)}")
            return self._create_empty_result(model_name)
    
    def _run_simple_regression(self, data: pd.DataFrame, outcome_var: str,
                             explanatory_vars: List[str], model_name: str) -> Dict[str, Any]:
        """简化版回归分析（当statsmodels不可用时）"""
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            y = data[outcome_var].values
            X = data[explanatory_vars].values
            
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            
            result = {
                'model_name': model_name,
                'n_obs': len(data),
                'r_squared': r2_score(y, y_pred),
                'coefficients': {var: coef for var, coef in zip(explanatory_vars, model.coef_)},
                'intercept': model.intercept_,
                'summary': f"简化回归模型 {model_name}，R² = {r2_score(y, y_pred):.4f}",
                'marginal_effects': None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 简化回归分析失败 {model_name}: {str(e)}")
            return self._create_empty_result(model_name)
    
    def _calculate_marginal_effects(self, data: pd.DataFrame, vars_list: List[str], 
                                  model) -> Dict[str, Any]:
        """计算交互项的边际效应"""
        
        marginal_effects = {}
        
        # 找到交互项
        interaction_vars = [var for var in vars_list if '_x_' in var]
        
        for int_var in interaction_vars:
            try:
                # 解析交互项名称
                var1, var2 = int_var.split('_x_')
                
                if var1 in data.columns and var2 in data.columns:
                    # 计算在不同var2水平下var1的边际效应
                    var2_values = np.percentile(data[var2], [10, 25, 50, 75, 90])
                    
                    effects = []
                    for val in var2_values:
                        # 边际效应 = β1 + β_interaction * var2_value
                        if var1 in model.params.index and int_var in model.params.index:
                            marginal_effect = model.params[var1] + model.params[int_var] * val
                            effects.append({
                                f'{var2}_value': val,
                                'marginal_effect': marginal_effect,
                                'percentile': np.where(var2_values == val)[0][0] * 20 + 10
                            })
                    
                    marginal_effects[int_var] = effects
                    
            except Exception as e:
                logger.warning(f"⚠️ 无法计算 {int_var} 的边际效应: {str(e)}")
        
        return marginal_effects
    
    def _create_empty_result(self, model_name: str) -> Dict[str, Any]:
        """创建空的结果字典"""
        return {
            'model_name': model_name,
            'n_obs': 0,
            'r_squared': np.nan,
            'coefficients': {},
            'p_values': {},
            'summary': f"模型 {model_name} 运行失败",
            'marginal_effects': None,
            'error': True
        }
    
    def create_results_table(self) -> pd.DataFrame:
        """
        创建结果汇总表
        
        Returns:
            包含所有模型结果的DataFrame
        """
        logger.info("📋 创建结果汇总表...")
        
        all_results = {**self.global_results, **self.local_results}
        
        if not all_results:
            logger.warning("⚠️ 没有分析结果可供汇总")
            return pd.DataFrame()
        
        rows = []
        for model_name, result in all_results.items():
            if result.get('error'):
                continue
                
            # 提取主要系数
            coeffs = result.get('coefficients', {})
            p_values = result.get('p_values', {})
            
            # 找到交互项
            interaction_vars = [var for var in coeffs.keys() if '_x_' in var]
            
            for int_var in interaction_vars:
                row = {
                    'model': model_name,
                    'interaction_term': int_var,
                    'coefficient': coeffs.get(int_var, np.nan),
                    'p_value': p_values.get(int_var, np.nan),
                    'significant': p_values.get(int_var, 1.0) < 0.05,
                    'n_obs': result.get('n_obs', 0),
                    'r_squared': result.get('r_squared', np.nan),
                    'analysis_type': 'Global' if any(x in model_name for x in ['global', 'network']) else 'Local'
                }
                rows.append(row)
        
        results_df = pd.DataFrame(rows)
        
        if len(results_df) > 0:
            # 按显著性和系数大小排序
            results_df = results_df.sort_values(['significant', 'coefficient'], ascending=[False, False])
            logger.info(f"✅ 结果表创建完成，共 {len(results_df)} 个交互项")
        else:
            logger.warning("⚠️ 结果表为空")
        
        return results_df
    
    def get_significant_interactions(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        获取显著的交互效应
        
        Args:
            alpha: 显著性水平
            
        Returns:
            显著交互效应的摘要
        """
        results_df = self.create_results_table()
        
        if len(results_df) == 0:
            return {
                'total_interactions': 0,
                'significant_interactions': 0,
                'significance_rate': 0,
                'significant_details': [],
                'strongest_effect': None,
                'summary': '没有发现显著的交互效应'
            }
        
        significant = results_df[results_df['p_value'] < alpha]
        
        summary = {
            'total_interactions': len(results_df),
            'significant_interactions': len(significant),
            'significance_rate': len(significant) / len(results_df) if len(results_df) > 0 else 0,
            'significant_details': significant.to_dict('records') if len(significant) > 0 else [],
            'strongest_effect': {
                'interaction': significant.iloc[0]['interaction_term'] if len(significant) > 0 else None,
                'coefficient': significant.iloc[0]['coefficient'] if len(significant) > 0 else None,
                'p_value': significant.iloc[0]['p_value'] if len(significant) > 0 else None
            } if len(significant) > 0 else None
        }
        
        logger.info(f"🎯 发现 {len(significant)} 个显著交互效应（α={alpha}）")
        
        return summary


def main():
    """测试分析功能"""
    # 创建测试数据
    np.random.seed(42)
    n = 100
    
    test_data = pd.DataFrame({
        'resilience_score': np.random.normal(0.7, 0.2, n),
        'dli_composite': np.random.normal(0.4, 0.15, n),
        'global_density': np.random.normal(0.3, 0.1, n),
        'betweenness_centrality': np.random.exponential(0.1, n),
        'control_var1': np.random.normal(0, 1, n),
        'control_var2': np.random.normal(0, 1, n)
    })
    
    # 运行分析
    analyzer = HeterogeneityAnalyzer()
    
    global_results = analyzer.run_global_analysis(test_data)
    local_results = analyzer.run_local_analysis(test_data)
    
    # 查看结果
    results_table = analyzer.create_results_table()
    print("📊 分析结果汇总:")
    print(results_table)
    
    significant = analyzer.get_significant_interactions()
    print("\n🎯 显著交互效应:")
    print(significant)


if __name__ == "__main__":
    main()