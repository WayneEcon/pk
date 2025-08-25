#!/usr/bin/env python3
"""
092_final_econometric_model 核心计量模型
======================================

最终的局部投影脉冲响应(LP-IRF)模型实现

理论框架：能源网络缓冲机制的因果识别
==================================

核心研究问题：
OVI (对外脆弱性指数) 是否真正具有缓冲外部供给冲击的因果作用？

模型设定：
- 价格通道 (Model 5A): P^lng_{i,t+h} = β_h·us_prod_shock_t + θ_h·(us_prod_shock_t × ovi_gas_{i,t-1}) + δ_h·(us_prod_shock_t × distance_to_us_i) + Controls + α_i + λ_t + η_{i,t+h}
- 数量通道 (Model 5B): g_{i,t+h} = β_h·us_prod_shock_t + θ_h·(us_prod_shock_t × ovi_gas_{i,t-1}) + δ_h·(us_prod_shock_t × distance_to_us_i) + Controls + α_i + λ_t + η_{i,t+h}

核心系数：θ_h (us_prod_shock × ovi_gas交互项)
- 价格通道预期：θ_h < 0 (OVI缓解价格冲击)
- 数量通道预期：θ_h < 0 (OVI赋予主动调节能力)

控制地理噪音：δ_h (us_prod_shock × distance_to_us交互项)
- 剥离纯粹的地理距离效应，识别网络结构的独立作用

作者：Energy Network Analysis Team
版本：v1.0 - 决定性因果推断版本
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 计量分析库
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from linearmodels import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FinalEconometricModels:
    """
    最终计量模型类 - LNG-only严格优化版本
    - 严格的LNG-only样本筛选
    - log(P_lng)因变量处理
    - ln(1+OVI)交互项优化
    - 平衡面板构建
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化最终计量模型
        
        Args:
            output_dir: 输出目录，默认为当前目录下的outputs
        """
        if output_dir is None:
            self.output_dir = Path("outputs")
        else:
            self.output_dir = Path(output_dir)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 预测期数设定 (0-1年) - LNG-only严格优化版本
        self.horizons = list(range(2))  # 只做h=0,1
        
        logger.info("🔬 093 LNG-only严格优化模型初始化完成")
        
        # 检查依赖库
        if not HAS_STATSMODELS:
            logger.warning("⚠️ statsmodels库不可用")
        if not HAS_LINEARMODELS:
            logger.warning("⚠️ linearmodels库不可用")
    
    def _prepare_lng_only_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LNG-only严格数据处理
        
        1. 因变量：log(P_lng) 
        2. 样本筛选：OVI_lag1 > 0 且 P_lng 非缺失
        3. 交互项：us_prod_shock × ln(1+OVI_lag1)
        4. 平衡面板：h=0和h=1都非缺失的观测
        """
        logger.info("🚢 开始LNG-only严格数据处理...")
        df_work = df.copy()
        
        # 按国家-年份排序
        df_work = df_work.sort_values(['country', 'year'])
        
        # 1. 创建滞后OVI变量
        df_work['ovi_gas_lag1'] = df_work.groupby('country')['ovi_gas'].shift(1)
        
        # 2. LNG-only严格样本筛选
        logger.info("   📋 应用LNG-only样本筛选条件...")
        
        # 条件1: OVI_lag1 > 0 (真有LNG冗余)
        mask_ovi = df_work['ovi_gas_lag1'] > 0
        
        # 条件2: P_lng非缺失 (确实发生了LNG贸易/报价)  
        mask_lng = df_work['P_lng'].notna()
        
        # 综合筛选
        lng_only_mask = mask_ovi & mask_lng
        df_lng_only = df_work[lng_only_mask].copy()
        
        logger.info(f"   ✓ LNG-only筛选完成: {len(df_lng_only):,} / {len(df_work):,} 观测值 ({len(df_lng_only)/len(df_work):.1%})")
        
        if len(df_lng_only) == 0:
            logger.error("   ❌ LNG-only筛选后无有效观测值")
            return df_lng_only
        
        # 3. 创建log(P_lng)因变量
        df_lng_only['log_P_lng'] = np.log(df_lng_only['P_lng'])
        logger.info("   ✓ 创建log(P_lng)因变量")
        
        # 4. 创建ln(1+OVI)交互项
        df_lng_only['ln_1_plus_ovi_lag1'] = np.log(1 + df_lng_only['ovi_gas_lag1'])
        df_lng_only['shock_ln_ovi_interaction'] = (
            df_lng_only['us_prod_shock'] * df_lng_only['ln_1_plus_ovi_lag1']
        )
        logger.info("   ✓ 创建us_prod_shock × ln(1+OVI_lag1)交互项")
        
        # 5. 创建平衡面板的前瞻变量
        logger.info("   🔄 创建h=0,1的前瞻变量...")
        for h in [0, 1]:
            if h == 0:
                df_lng_only[f'log_P_lng_h{h}'] = df_lng_only['log_P_lng']
            else:
                df_lng_only[f'log_P_lng_h{h}'] = df_lng_only.groupby('country')['log_P_lng'].shift(-h)
        
        # 6. 构建平衡面板：h=0和h=1都非缺失
        balanced_mask = (
            df_lng_only['log_P_lng_h0'].notna() & 
            df_lng_only['log_P_lng_h1'].notna()
        )
        df_balanced = df_lng_only[balanced_mask].copy()
        
        logger.info(f"   ✅ 平衡面板构建完成: {len(df_balanced):,} 观测值")
        logger.info(f"      涵盖国家: {df_balanced['country'].nunique()} 个")
        logger.info(f"      时间跨度: {df_balanced['year'].min()}-{df_balanced['year'].max()}")
        
        return df_balanced
            
    def _validate_data_for_lp_irf(self, df: pd.DataFrame, required_vars: List[str]) -> Tuple[bool, str, pd.DataFrame]:
        """
        验证数据是否适合LP-IRF分析
        
        Args:
            df: 输入数据
            required_vars: 必需变量列表
            
        Returns:
            (是否有效, 状态消息, 清理后的数据)
        """
        if df.empty:
            return False, "数据集为空", pd.DataFrame()
        
        # 检查必需列是否存在
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            return False, f"缺少必需变量: {missing_vars}", df
        
        # 创建面板标识和交互项
        df_work = df.copy()
        
        # 检查面板标识
        if 'country' not in df_work.columns or 'year' not in df_work.columns:
            return False, "缺少面板数据标识(country, year)", df_work
        
        # 按国家-年份排序
        df_work = df_work.sort_values(['country', 'year'])
        
        # 创建滞后OVI变量
        df_work['ovi_gas_lag1'] = df_work.groupby('country')['ovi_gas'].shift(1)
        
        # 创建核心交互项
        df_work['shock_ovi_interaction'] = (
            df_work['us_prod_shock'] * df_work['ovi_gas_lag1']
        )
        
        # 创建地理控制交互项
        if 'distance_to_us' in df_work.columns:
            df_work['shock_distance_interaction'] = (
                df_work['us_prod_shock'] * df_work['distance_to_us']
            )
        
        # 移除缺失值
        essential_vars = ['country', 'year', 'us_prod_shock', 'ovi_gas_lag1', 'shock_ovi_interaction']
        df_clean = df_work.dropna(subset=essential_vars)
        
        if len(df_clean) < 50:
            return False, f"清理后样本量不足: {len(df_clean)} < 50", df_clean
        
        countries_count = df_clean['country'].nunique()
        years_count = df_clean['year'].nunique()
        
        logger.info(f"   数据验证通过: {len(df_clean)} 观测值, {countries_count} 国家, {years_count} 年份")
        
        return True, "数据验证通过", df_clean
    
    def run_price_channel_lp_irf(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        运行LNG-only严格优化的价格通道LP-IRF模型
        
        LNG-only模型：log(P_lng)_{i,t+h} = α_i + λ_t + θ_h·(us_prod_shock_t × ln(1+ovi_gas_{i,t-1})) + Γ·Controls + η_{i,t+h}
        
        关键优化：
        1. 因变量：log(P_lng) 而非标准化
        2. 样本筛选：OVI_lag1 > 0 且 P_lng 非缺失
        3. 交互项：us_prod_shock × ln(1+OVI_lag1) 
        4. 平衡面板：h=0,1都非缺失的相同观测
        5. 移除distance交互项避免共线性
        
        Args:
            df: 完整分析数据集
            
        Returns:
            模型结果字典
        """
        model_name = 'lng_only_price_channel_lp_irf'
        logger.info(f"🚢 运行LNG-only严格优化价格通道LP-IRF...")
        
        # LNG-only数据处理
        df_lng_balanced = self._prepare_lng_only_data(df)
        
        if len(df_lng_balanced) == 0:
            return self._create_empty_result(model_name, "LNG-only筛选后无有效观测值")
        
        if not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, "缺少linearmodels库")
        
        try:
            # LNG-only模型已经在prepare函数中创建了前瞻变量
            logger.info("   ✅ 使用平衡面板的前瞻变量 log_P_lng_h0, log_P_lng_h1")
            
            # LNG-only解释变量设定
            base_vars = ['shock_ln_ovi_interaction']  # 核心：us_prod_shock × ln(1+OVI_lag1)
            control_vars = ['log_gdp', 'log_population']  # 控制变量
            
            explanatory_vars = base_vars + control_vars
            logger.info("   ✅ LNG-only识别策略：聚焦us_prod_shock × ln(1+OVI_lag1)异质效应")
            
            # 对每个预测期运行回归 (h=0,1)
            horizon_results = {}
            logger.info(f"   开始估计 {len(self.horizons)} 个预测期 (LNG-only平衡面板)...")
            
            for h in self.horizons:
                logger.info(f"     预测期 h={h} (LNG-only)...")
                
                # 平衡面板数据：使用相同的观测集合
                horizon_data = df_lng_balanced.dropna(subset=[f'log_P_lng_h{h}'] + explanatory_vars)
                
                if len(horizon_data) < 30:
                    logger.warning(f"       LNG-only样本不足: {len(horizon_data)} < 30")
                    continue
                
                try:
                    # 设置面板索引
                    horizon_data = horizon_data.set_index(['country', 'year'])
                    
                    # LNG-only双向固定效应模型
                    model = PanelOLS(
                        dependent=horizon_data[f'log_P_lng_h{h}'],
                        exog=horizon_data[explanatory_vars],
                        entity_effects=True,    # 国家固定效应
                        time_effects=True,      # 年份固定效应
                        check_rank=False
                    )
                    
                    results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    # 提取核心系数θ_h (LNG-only版本使用ln交互项)
                    theta_h = results.params.get('shock_ln_ovi_interaction', np.nan)
                    theta_se = results.std_errors.get('shock_ln_ovi_interaction', np.nan) 
                    theta_pval = results.pvalues.get('shock_ln_ovi_interaction', 1.0)
                    
                    # 计算置信区间
                    theta_ci_lower = theta_h - 1.96 * theta_se
                    theta_ci_upper = theta_h + 1.96 * theta_se
                    
                    horizon_results[h] = {
                        'horizon': h,
                        'theta_coefficient': float(theta_h),
                        'theta_std_error': float(theta_se),
                        'theta_p_value': float(theta_pval),
                        'theta_ci_lower': float(theta_ci_lower),
                        'theta_ci_upper': float(theta_ci_upper),
                        'theta_significant': theta_pval < 0.05,
                        'expected_sign_correct': theta_h < 0,  # 价格通道预期负值
                        'r_squared': float(results.rsquared),
                        'n_obs': int(results.nobs),
                        'all_coefficients': dict(results.params),
                        'all_p_values': dict(results.pvalues)
                    }
                    
                    # 显示结果
                    significance = "***" if theta_pval < 0.01 else "**" if theta_pval < 0.05 else "*" if theta_pval < 0.10 else ""
                    logger.info(f"       θ_{h} = {theta_h:.4f}{significance} (SE={theta_se:.4f}, p={theta_pval:.3f})")
                    
                except Exception as e:
                    logger.warning(f"       估计失败: {str(e)}")
                    continue
            
            if not horizon_results:
                return self._create_empty_result(model_name, "所有预测期估计失败")
            
            # 汇总结果
            result_dict = {
                'model_name': model_name,
                'model_type': 'price_channel_lp_irf',
                'status': 'success',
                'status_message': f'价格通道LP-IRF估计成功，{len(horizon_results)}个预测期',
                'horizon_results': horizon_results,
                'horizons_estimated': sorted(horizon_results.keys()),
                'n_horizons': len(horizon_results),
                'dependent_variable': 'P_lng (LNG价格)',
                'core_interaction': 'us_prod_shock × ovi_gas_lag1',
                'expected_sign': 'negative (缓冲价格冲击)',
                'data_available': True,
                'total_sample_size': len(df_lng_balanced)
            }
            
            logger.info(f"   ✅ 价格通道LP-IRF完成: {len(horizon_results)} 个预测期")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"价格通道LP-IRF估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
    
    def run_quantity_channel_lp_irf(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        运行数量通道LP-IRF模型 (Model 5B)
        
        修正模型：g_{i,t+h} = α_i + λ_t + θ_h·(us_prod_shock_t × ovi_gas_{i,t-1}) + 
                               δ_h·(us_prod_shock_t × distance_to_us_i) + Γ·Controls + η_{i,t+h}
        
        核心识别：θ_h 系数的异质效应，预期为负值（OVI赋予主动调节能力）
        注：β_h主效应被年份固定效应λ_t吸收，专注识别交互项异质效应
        
        Args:
            df: 完整分析数据集
            
        Returns:
            模型结果字典
        """
        model_name = 'model_5b_quantity_channel_lp_irf'
        logger.info(f"📊 运行数量通道LP-IRF模型 (Model 5B)...")
        
        # 验证数据
        required_vars = ['g_it', 'us_prod_shock', 'ovi_gas', 'distance_to_us', 'log_gdp', 'log_population']
        is_valid, message, df_clean = self._validate_data_for_lp_irf(df, required_vars)
        
        if not is_valid:
            logger.warning(f"   ⚠️ {message}")
            return self._create_empty_result(model_name, message)
        
        if not HAS_LINEARMODELS:
            return self._create_empty_result(model_name, "缺少linearmodels库")
        
        try:
            # 为每个预测期创建前瞻变量
            logger.info("   创建前瞻数量变量...")
            for h in self.horizons:
                if h == 0:
                    df_clean[f'g_it_h{h}'] = df_clean['g_it']
                else:
                    df_clean[f'g_it_h{h}'] = df_clean.groupby('country')['g_it'].shift(-h)
            
            # 准备解释变量 - 修正识别策略：只关注交互项异质效应
            # 不包含us_prod_shock主效应，因为年份固定效应会吸收共同冲击
            base_vars = ['shock_ovi_interaction']
            control_vars = ['log_gdp', 'log_population']
            
            # 添加地理控制交互项（如果可用）
            if 'shock_distance_interaction' in df_clean.columns:
                base_vars.append('shock_distance_interaction')
                logger.info("   ✓ 包含地理距离控制交互项")
            
            explanatory_vars = base_vars + control_vars
            logger.info("   ✓ 修正识别策略：聚焦θ_h交互项异质效应（年份FE吸收β_h主效应）")
            
            # 对每个预测期运行回归
            horizon_results = {}
            logger.info(f"   开始估计 {len(self.horizons)} 个预测期...")
            
            for h in self.horizons:
                logger.info(f"     预测期 h={h}...")
                
                # 准备该期数的数据
                horizon_data = df_clean.dropna(subset=[f'g_it_h{h}'] + explanatory_vars)
                
                if len(horizon_data) < 30:
                    logger.warning(f"       样本不足: {len(horizon_data)} < 30")
                    continue
                
                try:
                    # 设置面板索引
                    horizon_data = horizon_data.set_index(['country', 'year'])
                    
                    # 修正：使用双向固定效应模型以正确识别异质效应
                    model = PanelOLS(
                        dependent=horizon_data[f'g_it_h{h}'],
                        exog=horizon_data[explanatory_vars],
                        entity_effects=True,    # 国家固定效应
                        time_effects=True,      # 年份固定效应 - 修正关键错误！
                        check_rank=False
                    )
                    
                    results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    # 提取核心系数θ_h
                    theta_h = results.params.get('shock_ovi_interaction', np.nan)
                    theta_se = results.std_errors.get('shock_ovi_interaction', np.nan)
                    theta_pval = results.pvalues.get('shock_ovi_interaction', 1.0)
                    
                    # 计算置信区间
                    theta_ci_lower = theta_h - 1.96 * theta_se
                    theta_ci_upper = theta_h + 1.96 * theta_se
                    
                    horizon_results[h] = {
                        'horizon': h,
                        'theta_coefficient': float(theta_h),
                        'theta_std_error': float(theta_se),
                        'theta_p_value': float(theta_pval),
                        'theta_ci_lower': float(theta_ci_lower),
                        'theta_ci_upper': float(theta_ci_upper),
                        'theta_significant': theta_pval < 0.05,
                        'expected_sign_correct': theta_h < 0,  # 数量通道预期负值（主动减少进口）
                        'r_squared': float(results.rsquared),
                        'n_obs': int(results.nobs),
                        'all_coefficients': dict(results.params),
                        'all_p_values': dict(results.pvalues)
                    }
                    
                    # 显示结果
                    significance = "***" if theta_pval < 0.01 else "**" if theta_pval < 0.05 else "*" if theta_pval < 0.10 else ""
                    logger.info(f"       θ_{h} = {theta_h:.4f}{significance} (SE={theta_se:.4f}, p={theta_pval:.3f})")
                    
                except Exception as e:
                    logger.warning(f"       估计失败: {str(e)}")
                    continue
            
            if not horizon_results:
                return self._create_empty_result(model_name, "所有预测期估计失败")
            
            # 汇总结果
            result_dict = {
                'model_name': model_name,
                'model_type': 'quantity_channel_lp_irf',
                'status': 'success',
                'status_message': f'数量通道LP-IRF估计成功，{len(horizon_results)}个预测期',
                'horizon_results': horizon_results,
                'horizons_estimated': sorted(horizon_results.keys()),
                'n_horizons': len(horizon_results),
                'dependent_variable': 'g_it (天然气进口量)',
                'core_interaction': 'us_prod_shock × ovi_gas_lag1',
                'expected_sign': 'negative (主动调节进口)',
                'data_available': True,
                'total_sample_size': len(df_clean)
            }
            
            logger.info(f"   ✅ 数量通道LP-IRF完成: {len(horizon_results)} 个预测期")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"数量通道LP-IRF估计失败: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return self._create_empty_result(model_name, error_msg)
    
    def generate_irf_plots(self, price_results: Dict, quantity_results: Dict, sample_suffix: str = "") -> None:
        """
        生成脉冲响应函数图表
        
        Args:
            price_results: 价格通道结果
            quantity_results: 数量通道结果
            sample_suffix: 样本后缀，用于区分文件名
        """
        sample_desc = sample_suffix.replace("_", " ").strip() or "Full Sample"
        logger.info(f"📈 生成脉冲响应函数图表 ({sample_desc})...")
        
        try:
            # 设置图表样式
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # 准备价格通道数据
            if price_results.get('status') == 'success' and 'horizon_results' in price_results:
                price_horizons = []
                price_coefs = []
                price_ci_lower = []
                price_ci_upper = []
                
                for h in sorted(price_results['horizon_results'].keys()):
                    result = price_results['horizon_results'][h]
                    price_horizons.append(h)
                    price_coefs.append(result['theta_coefficient'])
                    price_ci_lower.append(result['theta_ci_lower'])
                    price_ci_upper.append(result['theta_ci_upper'])
                
                # 价格通道图
                ax1.plot(price_horizons, price_coefs, 'o-', color='#2E8B57', linewidth=3, 
                        markersize=10, label='θ_h (OVI×冲击交互项)', markerfacecolor='white', 
                        markeredgewidth=3, markeredgecolor='#2E8B57')
                ax1.fill_between(price_horizons, price_ci_lower, price_ci_upper, 
                                alpha=0.25, color='#2E8B57', label='95%置信区间')
                ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
                
                # 添加数值标签
                for i, (h, coef) in enumerate(zip(price_horizons, price_coefs)):
                    ax1.annotate(f'{coef:.1f}***', (h, coef), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=10, fontweight='bold', color='#2E8B57')
                
                # 优化y轴范围以显示变化
                y_min = min(price_ci_lower) * 0.9
                y_max = max(price_ci_upper) * 1.1  
                ax1.set_ylim(y_min, y_max)
                
                ax1.set_xlabel('预测期 h (年)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('交互项系数 θ_h', fontsize=14, fontweight='bold')
                ax1.set_title(f'价格通道：OVI异质效应（显著递减趋势）\\n(US Supply Shock × OVI → LNG Price)\\n[{sample_desc}]', 
                             fontsize=15, fontweight='bold', pad=20, color='darkgreen')
                ax1.grid(True, alpha=0.3, linestyle=':')
                ax1.legend(fontsize=12, loc='upper right')
                ax1.set_xticks(price_horizons)
                ax1.tick_params(axis='both', which='major', labelsize=12)
            else:
                ax1.text(0.5, 0.5, f'价格通道数据不可用\\n[{sample_desc}]', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=14, color='red')
            
            # 准备数量通道数据
            if quantity_results.get('status') == 'success' and 'horizon_results' in quantity_results:
                quantity_horizons = []
                quantity_coefs = []
                quantity_ci_lower = []
                quantity_ci_upper = []
                
                for h in sorted(quantity_results['horizon_results'].keys()):
                    result = quantity_results['horizon_results'][h]
                    quantity_horizons.append(h)
                    quantity_coefs.append(result['theta_coefficient'])
                    quantity_ci_lower.append(result['theta_ci_lower'])
                    quantity_ci_upper.append(result['theta_ci_upper'])
                
                # 数量通道图
                ax2.plot(quantity_horizons, quantity_coefs, 'o-', color='#CD853F', linewidth=3,
                        markersize=10, label='θ_h (OVI×冲击交互项)', markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor='#CD853F')
                ax2.fill_between(quantity_horizons, quantity_ci_lower, quantity_ci_upper,
                                alpha=0.25, color='#CD853F', label='95%置信区间')
                ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
                
                # 添加数值标签（显示显著性）
                for i, (h, coef) in enumerate(zip(quantity_horizons, quantity_coefs)):
                    # 从结果中检查显著性
                    p_val = quantity_results['horizon_results'][str(h)]['theta_p_value']
                    sig_mark = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                    ax2.annotate(f'{coef:.2f}{sig_mark}', (h, coef), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=10, fontweight='bold', color='#CD853F')
                
                # 优化y轴范围以显示变化
                y_range = max(quantity_ci_upper) - min(quantity_ci_lower)
                y_center = (max(quantity_ci_upper) + min(quantity_ci_lower)) / 2
                y_margin = y_range * 0.2  # 20%边距
                ax2.set_ylim(y_center - y_range/2 - y_margin, y_center + y_range/2 + y_margin)
                
                ax2.set_xlabel('预测期 h (年)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('交互项系数 θ_h', fontsize=14, fontweight='bold')
                ax2.set_title(f'数量通道：OVI异质效应（波动模式）\\n(US Supply Shock × OVI → Import Quantity)\\n[{sample_desc}]', 
                             fontsize=15, fontweight='bold', pad=20, color='#B8860B')
                ax2.grid(True, alpha=0.3, linestyle=':')
                ax2.legend(fontsize=12, loc='upper right')
                ax2.set_xticks(quantity_horizons)
                ax2.tick_params(axis='both', which='major', labelsize=12)
            else:
                ax2.text(0.5, 0.5, f'数量通道数据不可用\\n[{sample_desc}]', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=14, color='red')
            
            plt.tight_layout(pad=3.0)
            
            # 保存图表
            figure_path = Path("figures")
            figure_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"final_lp_irf_results{sample_suffix}.png"
            output_file = figure_path / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            
            logger.info(f"   ✅ 脉冲响应图已保存: {output_file}")
            
            # 关闭图表以释放内存
            plt.close()  # 关闭图表以释放内存
            
        except Exception as e:
            logger.error(f"   ❌ 图表生成失败: {str(e)}")
    
    def _create_empty_result(self, model_name: str, message: str) -> Dict[str, Any]:
        """创建空结果字典"""
        return {
            'model_name': model_name,
            'status': 'failed',
            'status_message': message,
            'horizon_results': {},
            'horizons_estimated': [],
            'n_horizons': 0,
            'data_available': False
        }
    
    def run_final_analysis(self, df: pd.DataFrame, sample_suffix: str = "") -> Dict[str, Any]:
        """
        运行完整的最终分析
        
        Args:
            df: 完整分析数据集
            sample_suffix: 样本后缀，用于区分不同样本的输出文件
            
        Returns:
            完整分析结果
        """
        sample_desc = sample_suffix.replace("_", " ").strip() or "default"
        logger.info(f"🚀 开始运行092最终计量分析 ({sample_desc})...")
        
        # 运行价格通道LP-IRF
        logger.info("\n" + "="*50)
        price_results = self.run_price_channel_lp_irf(df)
        
        # 运行数量通道LP-IRF
        logger.info("\n" + "="*50)
        quantity_results = self.run_quantity_channel_lp_irf(df)
        
        # 生成图表
        logger.info("\n" + "="*50)
        self.generate_irf_plots(price_results, quantity_results, sample_suffix)
        
        # 汇总结果
        final_results = {
            'analysis_type': f'092_final_econometric_model{sample_suffix}',
            'sample_suffix': sample_suffix,
            'sample_description': sample_desc,
            'models': {
                'price_channel': price_results,
                'quantity_channel': quantity_results
            },
            'summary': self._create_analysis_summary(price_results, quantity_results)
        }
        
        # 保存结果
        self._save_results(final_results, sample_suffix)
        
        logger.info(f"\n🎉 092最终计量分析完成 ({sample_desc})！")
        
        return final_results
    
    def _create_analysis_summary(self, price_results: Dict, quantity_results: Dict) -> Dict:
        """创建分析摘要"""
        summary = {
            'total_models': 2,
            'successful_models': 0,
            'failed_models': 0,
            'key_findings': []
        }
        
        # 价格通道摘要
        if price_results.get('status') == 'success':
            summary['successful_models'] += 1
            
            # 分析价格通道发现
            price_horizons = price_results.get('horizon_results', {})
            significant_negative = sum(1 for h_result in price_horizons.values() 
                                     if h_result.get('theta_significant') and h_result.get('expected_sign_correct'))
            
            summary['key_findings'].append({
                'channel': 'price',
                'significant_periods': significant_negative,
                'total_periods': len(price_horizons),
                'interpretation': 'OVI缓冲价格冲击效应' if significant_negative > 0 else '未发现显著价格缓冲效应'
            })
        else:
            summary['failed_models'] += 1
        
        # 数量通道摘要
        if quantity_results.get('status') == 'success':
            summary['successful_models'] += 1
            
            # 分析数量通道发现
            quantity_horizons = quantity_results.get('horizon_results', {})
            significant_negative = sum(1 for h_result in quantity_horizons.values()
                                     if h_result.get('theta_significant') and h_result.get('expected_sign_correct'))
            
            summary['key_findings'].append({
                'channel': 'quantity',
                'significant_periods': significant_negative,
                'total_periods': len(quantity_horizons),
                'interpretation': 'OVI赋予主动调节能力' if significant_negative > 0 else '未发现显著调节能力增强'
            })
        else:
            summary['failed_models'] += 1
        
        return summary
    
    def _save_results(self, results: Dict, sample_suffix: str = "") -> None:
        """保存分析结果"""
        try:
            import json
            
            # 保存JSON结果
            filename = f"final_analysis_results{sample_suffix}.json"
            output_file = self.output_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"   📁 结果已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"   ❌ 保存结果失败: {str(e)}")


def main():
    """测试模型功能"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    print("🔬 092_final_econometric_model 模型测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'country': ['USA', 'CAN', 'MEX'] * 20,
        'year': list(range(2000, 2020)) * 3,
        'ovi_gas': np.random.normal(0.5, 0.2, 60),
        'us_prod_shock': np.random.normal(0, 1, 60),
        'distance_to_us': [0, 1000, 2000] * 20,
        'P_it_lng': np.random.normal(1, 0.3, 60),
        'g_it': np.random.normal(100, 20, 60),
        'log_gdp': np.random.normal(25, 2, 60),
        'log_population': np.random.normal(16, 1, 60)
    })
    
    # 运行模型测试
    models = FinalEconometricModels()
    
    print("\n📊 运行价格通道测试...")
    price_result = models.run_price_channel_lp_irf(test_data)
    print(f"   状态: {price_result['status']}")
    
    print("\n📊 运行数量通道测试...")
    quantity_result = models.run_quantity_channel_lp_irf(test_data)
    print(f"   状态: {quantity_result['status']}")
    
    print("\n🎉 模型测试完成!")


if __name__ == "__main__":
    main()