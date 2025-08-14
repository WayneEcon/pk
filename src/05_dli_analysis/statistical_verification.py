#!/usr/bin/env python3
"""
统计验证模块 (Statistical Verification Module)
==============================================

本模块使用双重差分法(DID)等准实验方法，对"页岩革命是否显著改变了DLI格局"
这一核心假说进行严谨的统计验证。

DID模型设定：
- 处理组 (Treatment Group): 通过管道进行原油和天然气贸易的美-加、美-墨关系
  这些关系受高沉没成本的专用性基础设施锁定，是政策冲击最直接的传导渠道
- 控制组 (Control Group): 通过海运进行LNG、原油及成品油贸易的关系
  如与沙特、卡塔尔、委内瑞拉等，基础设施专用性较低，转换成本更灵活
- 政策冲击时间点: 页岩革命产生显著产出效应的年份（2011年或之后）

作者：Energy Network Analysis Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 尝试导入statsmodels，如果没有则使用简化版本
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning("statsmodels not available, using simplified regression")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DID实验设计常量
TREATMENT_COUNTRIES = ['CAN', 'MEX']  # 处理组：管道贸易国家
CONTROL_COUNTRIES = ['SAU', 'QAT', 'VEN', 'NOR', 'GBR', 'RUS', 'ARE']  # 控制组：海运贸易主要国家
PIPELINE_PRODUCTS = ['Crude_Oil', 'Natural_Gas']  # 管道运输的主要产品
POLICY_SHOCK_YEAR = 2011  # 页岩革命显著产出效应年份
PRE_PERIOD = (2001, 2010)  # 政策前期间
POST_PERIOD = (2011, 2024)  # 政策后期间

def prepare_did_dataset(dli_data: pd.DataFrame = None, 
                       data_file_path: str = None) -> pd.DataFrame:
    """
    准备DID分析数据集
    
    Args:
        dli_data: DLI面板数据，如果为None则从文件加载
        data_file_path: 数据文件路径
        
    Returns:
        准备好的DID分析数据集
        
    包含列：
        - 基础标识：year, us_partner, energy_product, us_role
        - DLI指标：dli_composite_adjusted + 四个维度
        - DID变量：treatment, post, treatment_post
        - 控制变量：trade_value_usd, distance_km等
    """
    
    logger.info("🎯 开始准备DID分析数据集...")
    
    # 第1步：加载DLI数据
    if dli_data is not None:
        df = dli_data.copy()
        logger.info(f"使用提供的DLI数据: {len(df)} 条记录")
    else:
        if data_file_path is None:
            base_dir = Path(__file__).parent.parent.parent
            data_file_path = base_dir / "outputs" / "tables" / "dli_panel_data.csv"
        
        if not Path(data_file_path).exists():
            raise FileNotFoundError(f"DLI数据文件不存在: {data_file_path}")
        
        df = pd.read_csv(data_file_path)
        logger.info(f"从文件加载DLI数据: {data_file_path}, {len(df)} 条记录")
    
    # 第2步：定义处理组和控制组
    logger.info("🔍 定义处理组和控制组...")
    
    # 处理组：管道贸易关系（美-加、美-墨的原油和天然气）
    treatment_condition = (
        df['us_partner'].isin(TREATMENT_COUNTRIES) & 
        df['energy_product'].isin(PIPELINE_PRODUCTS)
    )
    
    # 控制组：海运贸易关系（其他主要贸易伙伴）
    control_condition = (
        df['us_partner'].isin(CONTROL_COUNTRIES) & 
        ~df['energy_product'].isin(['Coal'])  # 排除煤炭，因为主要是海运但性质不同
    )
    
    # 筛选实验样本
    did_sample = df[treatment_condition | control_condition].copy()
    
    if len(did_sample) == 0:
        raise ValueError("未找到符合条件的DID分析样本")
    
    logger.info(f"📊 DID样本构成:")
    logger.info(f"  总样本: {len(did_sample)} 条记录")
    
    # 第3步：创建DID变量
    logger.info("⚙️ 创建DID实验变量...")
    
    # 处理组指示变量 (Treatment)
    did_sample['treatment'] = treatment_condition[treatment_condition | control_condition].astype(int)
    
    # 政策后时期指示变量 (Post)
    did_sample['post'] = (did_sample['year'] >= POLICY_SHOCK_YEAR).astype(int)
    
    # DID交互项 (Treatment × Post)
    did_sample['treatment_post'] = did_sample['treatment'] * did_sample['post']
    
    # 第4步：创建时期变量
    def assign_period(year):
        if year < POLICY_SHOCK_YEAR:
            return 'pre'
        else:
            return 'post'
    
    did_sample['period'] = did_sample['year'].apply(assign_period)
    
    # 第5步：数据验证和统计
    logger.info("🔍 DID实验设计验证:")
    
    # 按组和时期统计
    group_period_stats = did_sample.groupby(['treatment', 'period']).agg({
        'us_partner': 'nunique',
        'energy_product': 'nunique', 
        'dli_composite_adjusted': ['count', 'mean', 'std']
    }).round(4)
    
    logger.info("实验组构成统计:")
    print(group_period_stats)
    
    # 处理组国家统计
    treatment_countries_actual = did_sample[did_sample['treatment'] == 1]['us_partner'].unique()
    control_countries_actual = did_sample[did_sample['treatment'] == 0]['us_partner'].unique()
    
    logger.info(f"  实际处理组国家: {sorted(treatment_countries_actual)}")
    logger.info(f"  实际控制组国家: {sorted(control_countries_actual)}")
    
    # 产品分布统计
    product_by_group = did_sample.groupby(['treatment', 'energy_product']).size().unstack(fill_value=0)
    logger.info("产品分布统计:")
    print(product_by_group)
    
    # 时间平衡性检查
    time_balance = did_sample.groupby(['treatment', 'year']).size().unstack(fill_value=0)
    logger.info(f"时间跨度: {did_sample['year'].min()}-{did_sample['year'].max()}")
    logger.info(f"政策冲击年份: {POLICY_SHOCK_YEAR}")
    
    # 第6步：创建控制变量
    logger.info("📈 创建控制变量...")
    
    # 对数化贸易值（处理极值）
    did_sample['log_trade_value'] = np.log(did_sample['trade_value_usd'] + 1)
    
    # 对数化距离
    did_sample['log_distance'] = np.log(did_sample['distance_km'])
    
    # 年份趋势变量
    did_sample['year_trend'] = did_sample['year'] - 2001
    
    # 创建国家和产品固定效应变量
    did_sample['country_product'] = did_sample['us_partner'] + '_' + did_sample['energy_product']
    
    # 第7步：最终数据验证
    logger.info("✅ DID数据集验证:")
    logger.info(f"  最终样本量: {len(did_sample):,} 观测")
    logger.info(f"  国家数: {did_sample['us_partner'].nunique()}")
    logger.info(f"  产品数: {did_sample['energy_product'].nunique()}")
    logger.info(f"  年份数: {did_sample['year'].nunique()}")
    logger.info(f"  处理组观测: {did_sample['treatment'].sum():,} ({did_sample['treatment'].mean()*100:.1f}%)")
    logger.info(f"  政策后观测: {did_sample['post'].sum():,} ({did_sample['post'].mean()*100:.1f}%)")
    
    # 检查关键变量的缺失值
    key_variables = ['dli_composite_adjusted', 'treatment', 'post', 'treatment_post', 
                    'log_trade_value', 'log_distance']
    missing_summary = did_sample[key_variables].isnull().sum()
    if missing_summary.any():
        logger.warning("发现缺失值:")
        for var, count in missing_summary[missing_summary > 0].items():
            logger.warning(f"  {var}: {count} 个缺失值")
    else:
        logger.info("✅ 关键变量无缺失值")
    
    logger.info("✅ DID数据集准备完成!")
    return did_sample

def run_did_analysis(did_data: pd.DataFrame = None,
                    outcome_vars: List[str] = None,
                    control_vars: List[str] = None,
                    use_fixed_effects: bool = True) -> Dict[str, Dict]:
    """
    执行双重差分(DID)分析
    
    基本模型：
    Y_ijt = α + β₁×Treatment_ij + β₂×Post_t + β₃×(Treatment_ij × Post_t) + γ×X_ijt + ε_ijt
    
    其中：
    - Y_ijt: DLI相关结果变量
    - Treatment_ij: 处理组指示变量（1=管道贸易国家，0=海运贸易国家）
    - Post_t: 政策后时期指示变量（1=2011年及以后，0=2010年及以前）
    - β₃: DID估计量，表示政策对处理组的净影响
    - X_ijt: 控制变量
    
    Args:
        did_data: DID分析数据集
        outcome_vars: 结果变量列表，默认为DLI相关指标
        control_vars: 控制变量列表
        use_fixed_effects: 是否使用固定效应
        
    Returns:
        包含所有模型结果的字典
    """
    
    logger.info("📊 开始执行DID分析...")
    
    # 数据准备
    if did_data is None:
        did_data = prepare_did_dataset()
    
    if outcome_vars is None:
        outcome_vars = [
            'dli_composite_adjusted',
            'continuity', 
            'infrastructure', 
            'stability', 
            'market_locking_power'
        ]
    
    if control_vars is None:
        control_vars = ['log_trade_value', 'log_distance', 'year_trend']
    
    # 验证变量存在
    all_vars = outcome_vars + control_vars + ['treatment', 'post', 'treatment_post']
    missing_vars = [var for var in all_vars if var not in did_data.columns]
    if missing_vars:
        raise ValueError(f"数据中缺少变量: {missing_vars}")
    
    results = {}
    
    # 为每个结果变量运行DID回归
    for outcome_var in outcome_vars:
        logger.info(f"🔍 分析结果变量: {outcome_var}")
        
        try:
            # 准备回归数据（移除缺失值）
            reg_vars = [outcome_var, 'treatment', 'post', 'treatment_post'] + control_vars
            cluster_vars = ['us_partner']  # 聚类变量
            all_vars = reg_vars + cluster_vars
            reg_data = did_data[all_vars].dropna()
            
            if len(reg_data) == 0:
                logger.warning(f"  {outcome_var}: 无有效观测，跳过分析")
                continue
            
            logger.info(f"  有效观测数: {len(reg_data):,}")
            
            if HAS_STATSMODELS:
                # 使用statsmodels进行专业回归分析
                
                # 构建回归公式
                formula = f"{outcome_var} ~ treatment + post + treatment_post"
                if control_vars:
                    formula += " + " + " + ".join(control_vars)
                
                logger.info(f"  回归公式: {formula}")
                
                # 运行回归 - 使用聚类稳健标准误
                # 这是面板数据DID分析的标准做法，避免同一实体观测值的序列相关性
                model = smf.ols(formula, data=reg_data).fit(
                    cov_type='cluster', 
                    cov_kwds={'groups': reg_data['us_partner']}
                )
                
                # 提取关键结果
                did_coef = model.params['treatment_post']
                did_pvalue = model.pvalues['treatment_post']
                did_stderr = model.bse['treatment_post']
                did_tstat = model.tvalues['treatment_post']
                
                # 计算置信区间
                conf_int = model.conf_int().loc['treatment_post']
                did_ci_lower = conf_int[0]
                did_ci_upper = conf_int[1]
                
                # 模型诊断统计
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj
                f_statistic = model.fvalue
                f_pvalue = model.f_pvalue
                
                # 异方差检验（Breusch-Pagan）
                try:
                    bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
                except:
                    bp_stat, bp_pvalue = None, None
                
                # Durbin-Watson统计量（序列相关检验）
                try:
                    dw_stat = durbin_watson(model.resid)
                except:
                    dw_stat = None
                
                # 保存详细结果
                var_results = {
                    # DID核心结果
                    'did_coefficient': did_coef,
                    'did_std_error': did_stderr,
                    'did_t_statistic': did_tstat,
                    'did_p_value': did_pvalue,
                    'did_ci_lower': did_ci_lower,
                    'did_ci_upper': did_ci_upper,
                    'is_significant_5pct': did_pvalue < 0.05,
                    'is_significant_10pct': did_pvalue < 0.10,
                    
                    # 其他系数
                    'treatment_coef': model.params.get('treatment', None),
                    'post_coef': model.params.get('post', None),
                    'treatment_pvalue': model.pvalues.get('treatment', None),
                    'post_pvalue': model.pvalues.get('post', None),
                    
                    # 模型拟合统计
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'f_statistic': f_statistic,
                    'f_pvalue': f_pvalue,
                    'n_observations': len(reg_data),
                    
                    # 诊断统计
                    'breusch_pagan_stat': bp_stat,
                    'breusch_pagan_pvalue': bp_pvalue,
                    'durbin_watson_stat': dw_stat,
                    
                    # 完整模型对象（用于后续分析）
                    'full_model': model
                }
                
            else:
                # 简化版回归分析（使用numpy）
                logger.info("  使用简化回归方法（建议安装statsmodels以获得完整统计）")
                
                # 准备设计矩阵
                X = reg_data[['treatment', 'post', 'treatment_post'] + control_vars].values
                X = np.column_stack([np.ones(len(X)), X])  # 添加常数项
                y = reg_data[outcome_var].values
                
                # OLS估计
                beta = np.linalg.inv(X.T @ X) @ X.T @ y
                y_pred = X @ beta
                residuals = y - y_pred
                
                # 标准误计算
                mse = np.sum(residuals**2) / (len(y) - X.shape[1])
                var_cov_matrix = mse * np.linalg.inv(X.T @ X)
                std_errors = np.sqrt(np.diag(var_cov_matrix))
                
                # DID系数是第4个系数（treatment_post）
                did_coef = beta[3]
                did_stderr = std_errors[3]
                did_tstat = did_coef / did_stderr
                did_pvalue = 2 * (1 - stats.t.cdf(abs(did_tstat), len(y) - X.shape[1]))
                
                # R平方
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)
                adj_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1])
                
                var_results = {
                    'did_coefficient': did_coef,
                    'did_std_error': did_stderr,
                    'did_t_statistic': did_tstat,
                    'did_p_value': did_pvalue,
                    'is_significant_5pct': did_pvalue < 0.05,
                    'is_significant_10pct': did_pvalue < 0.10,
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'n_observations': len(reg_data),
                    'method': 'simplified_ols'
                }
            
            results[outcome_var] = var_results
            
            # 打印主要结果
            logger.info(f"  ✅ {outcome_var} DID结果:")
            logger.info(f"    系数: {did_coef:.6f}")
            logger.info(f"    标准误: {did_stderr:.6f}")
            logger.info(f"    t统计量: {did_tstat:.4f}")
            logger.info(f"    p值: {did_pvalue:.6f}")
            logger.info(f"    5%显著性: {'是' if did_pvalue < 0.05 else '否'}")
            if HAS_STATSMODELS:
                logger.info(f"    95%置信区间: [{did_ci_lower:.6f}, {did_ci_upper:.6f}]")
            logger.info(f"    R²: {r_squared:.4f}")
            
        except Exception as e:
            logger.error(f"  ❌ {outcome_var} 分析失败: {e}")
            results[outcome_var] = {'error': str(e)}
            continue
    
    # 计算总体统计
    successful_analyses = [k for k, v in results.items() if 'error' not in v]
    significant_5pct = [k for k, v in results.items() 
                       if 'error' not in v and v.get('is_significant_5pct', False)]
    significant_10pct = [k for k, v in results.items() 
                        if 'error' not in v and v.get('is_significant_10pct', False)]
    
    # 添加汇总信息
    results['_summary'] = {
        'total_variables_analyzed': len(outcome_vars),
        'successful_analyses': len(successful_analyses),
        'significant_5pct': len(significant_5pct),
        'significant_10pct': len(significant_10pct),
        'significant_variables_5pct': significant_5pct,
        'significant_variables_10pct': significant_10pct,
        'policy_shock_year': POLICY_SHOCK_YEAR,
        'treatment_countries': TREATMENT_COUNTRIES,
        'control_countries': CONTROL_COUNTRIES,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    logger.info("📊 DID分析汇总:")
    logger.info(f"  成功分析变量: {len(successful_analyses)}/{len(outcome_vars)}")
    logger.info(f"  5%水平显著: {len(significant_5pct)} 个变量 {significant_5pct}")
    logger.info(f"  10%水平显著: {len(significant_10pct)} 个变量 {significant_10pct}")
    
    logger.info("✅ DID分析完成!")
    return results

def generate_verification_report(did_results: Dict = None,
                                did_data: pd.DataFrame = None,
                                output_dir: str = None) -> str:
    """
    生成DID验证报告
    
    Args:
        did_results: DID分析结果
        did_data: DID数据集
        output_dir: 输出目录
        
    Returns:
        报告文件路径
    """
    
    logger.info("📝 开始生成DID验证报告...")
    
    # 设置输出路径
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "outputs" / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果没有提供结果，则运行分析
    if did_results is None:
        did_results = run_did_analysis(did_data)
    
    # 生成报告时间戳
    timestamp = pd.Timestamp.now()
    
    # 创建Markdown报告
    md_report_path = output_dir / "dli_verification_report.md"
    
    with open(md_report_path, 'w', encoding='utf-8') as f:
        # 报告标题和概述
        f.write("# DLI动态锁定指数统计验证报告\n\n")
        f.write(f"**报告生成时间**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 研究假说与实验设计\n\n")
        f.write("### 1.1 核心假说\n")
        f.write("页岩革命是否显著改变了美国与贸易伙伴之间的能源贸易锁定格局（DLI）？\n\n")
        
        f.write("### 1.2 实验设计 (双重差分法)\n")
        f.write("- **处理组**: 通过管道进行原油和天然气贸易的美-加、美-墨关系\n")
        f.write("  - 国家: 加拿大(CAN)、墨西哥(MEX)\n")
        f.write("  - 产品: 原油(Crude_Oil)、天然气(Natural_Gas)\n")
        f.write("  - 特征: 高沉没成本的专用性基础设施锁定\n\n")
        
        f.write("- **控制组**: 通过海运进行贸易的关系\n")
        f.write("  - 国家: 沙特阿拉伯(SAU)、卡塔尔(QAT)、委内瑞拉(VEN)、挪威(NOR)、英国(GBR)、俄罗斯(RUS)、阿联酋(ARE)\n")
        f.write("  - 特征: 基础设施专用性较低，转换成本更灵活\n\n")
        
        f.write(f"- **政策冲击时点**: {POLICY_SHOCK_YEAR}年（页岩革命显著产出效应年份）\n\n")
        
        f.write("### 1.3 DID模型\n")
        f.write("```\n")
        f.write("DLI_ijt = α + β₁×Treatment_ij + β₂×Post_t + β₃×(Treatment_ij × Post_t) + γ×X_ijt + ε_ijt\n")
        f.write("```\n")
        f.write("其中 β₃ 为DID估计量，衡量政策对处理组的净影响。\n\n")
        
        # 数据描述性统计
        if did_data is not None:
            f.write("## 2. 数据概况\n\n")
            
            summary_stats = did_results.get('_summary', {})
            f.write(f"- **总观测数**: {len(did_data):,} 条记录\n")
            f.write(f"- **时间跨度**: {did_data['year'].min()}-{did_data['year'].max()}\n")
            f.write(f"- **贸易伙伴**: {did_data['us_partner'].nunique()} 个国家\n")
            f.write(f"- **能源产品**: {did_data['energy_product'].nunique()} 种\n")
            f.write(f"- **处理组观测**: {did_data['treatment'].sum():,} ({did_data['treatment'].mean()*100:.1f}%)\n")
            f.write(f"- **政策后观测**: {did_data['post'].sum():,} ({did_data['post'].mean()*100:.1f}%)\n\n")
            
            # 按组和时期的描述性统计
            desc_stats = did_data.groupby(['treatment', 'period'])['dli_composite_adjusted'].agg(['count', 'mean', 'std']).round(4)
            f.write("### 2.1 按组和时期的DLI均值\n\n")
            f.write("| 组别 | 时期 | 观测数 | 均值 | 标准差 |\n")
            f.write("|------|------|--------|------|--------|\n")
            for (treatment, period), row in desc_stats.iterrows():
                group_name = "处理组" if treatment == 1 else "控制组"
                period_name = "政策前" if period == 'pre' else "政策后"
                f.write(f"| {group_name} | {period_name} | {row['count']} | {row['mean']:.4f} | {row['std']:.4f} |\n")
            f.write("\n")
        
        # DID分析结果
        f.write("## 3. DID分析结果\n\n")
        
        summary = did_results.get('_summary', {})
        f.write(f"- **成功分析变量数**: {summary.get('successful_analyses', 0)}/{summary.get('total_variables_analyzed', 0)}\n")
        f.write(f"- **5%水平显著变量**: {summary.get('significant_5pct', 0)} 个\n")
        f.write(f"- **10%水平显著变量**: {summary.get('significant_10pct', 0)} 个\n\n")
        
        # 详细结果表
        f.write("### 3.1 详细回归结果\n\n")
        f.write("| 被解释变量 | DID系数 | 标准误 | t统计量 | p值 | R² | 观测数 | 5%显著 |\n")
        f.write("|------------|---------|--------|---------|-----|-----|--------|--------|\n")
        
        for var, results in did_results.items():
            if var.startswith('_') or 'error' in results:
                continue
            
            coef = results.get('did_coefficient', 0)
            stderr = results.get('did_std_error', 0)
            t_stat = results.get('did_t_statistic', 0)
            p_val = results.get('did_p_value', 1)
            r_sq = results.get('r_squared', 0)
            n_obs = results.get('n_observations', 0)
            is_sig = "✓" if results.get('is_significant_5pct', False) else ""
            
            f.write(f"| {var} | {coef:.6f} | {stderr:.6f} | {t_stat:.4f} | {p_val:.6f} | {r_sq:.4f} | {n_obs:,} | {is_sig} |\n")
        
        f.write("\n")
        
        # 关键发现
        f.write("## 4. 关键发现\n\n")
        
        significant_vars_5 = summary.get('significant_variables_5pct', [])
        significant_vars_10 = summary.get('significant_variables_10pct', [])
        
        if significant_vars_5:
            f.write("### 4.1 统计显著的政策效应 (5%水平)\n\n")
            for var in significant_vars_5:
                if var in did_results:
                    result = did_results[var]
                    coef = result.get('did_coefficient', 0)
                    f.write(f"- **{var}**: DID系数 = {coef:.6f}")
                    if coef > 0:
                        f.write(" (政策增强了锁定效应)\n")
                    else:
                        f.write(" (政策减弱了锁定效应)\n")
            f.write("\n")
        else:
            f.write("### 4.1 统计显著性\n")
            f.write("在5%显著性水平下，未发现页岩革命对DLI指标的显著影响。\n\n")
        
        if significant_vars_10:
            f.write("### 4.2 边际显著的政策效应 (10%水平)\n\n")
            for var in significant_vars_10:
                if var in did_results and var not in significant_vars_5:
                    result = did_results[var]
                    coef = result.get('did_coefficient', 0)
                    f.write(f"- **{var}**: DID系数 = {coef:.6f}")
                    if coef > 0:
                        f.write(" (政策可能增强了锁定效应)\n")
                    else:
                        f.write(" (政策可能减弱了锁定效应)\n")
            f.write("\n")
        
        # 结论
        f.write("## 5. 结论\n\n")
        
        if significant_vars_5:
            f.write("基于双重差分分析，我们发现页岩革命对美国能源贸易锁定格局产生了统计显著的影响。")
            f.write("具体而言，管道贸易关系（美-加、美-墨）相较于海运贸易关系，")
            f.write("在页岩革命后表现出了不同的锁定模式变化。这一发现支持了我们关于")
            f.write("基础设施专用性在政策传导中重要作用的理论假说。\n\n")
        else:
            f.write("基于双重差分分析，我们未能在5%显著性水平下发现页岩革命对美国能源贸易锁定格局的统计显著影响。")
            f.write("这可能表明：(1) 政策效应确实不存在；(2) 效应存在但相对较小，需要更大样本才能检测到；")
            f.write("(3) 实验设计需要进一步优化。建议后续研究考虑更精细的分组策略或更长的观测期。\n\n")
        
        f.write("## 6. 方法论注记\n\n")
        f.write("- **因果推断方法**: 双重差分法(Difference-in-Differences)\n")
        f.write("- **标准误估计**: 异方差稳健标准误\n")
        if HAS_STATSMODELS:
            f.write("- **诊断检验**: Breusch-Pagan异方差检验, Durbin-Watson序列相关检验\n")
        f.write("- **显著性水平**: 5%和10%\n")
        f.write("- **软件工具**: Python statsmodels\n\n")
        
        f.write("---\n")
        f.write("*本报告由DLI分析模块自动生成*\n")
    
    logger.info(f"📄 Markdown报告已生成: {md_report_path}")
    
    # 同时生成CSV结果表
    csv_report_path = output_dir / "dli_verification_results.csv"
    
    results_for_csv = []
    for var, results in did_results.items():
        if var.startswith('_') or 'error' in results:
            continue
        
        row = {
            'variable': var,
            'did_coefficient': results.get('did_coefficient', np.nan),
            'did_std_error': results.get('did_std_error', np.nan),
            'did_t_statistic': results.get('did_t_statistic', np.nan),
            'did_p_value': results.get('did_p_value', np.nan),
            'significant_5pct': results.get('is_significant_5pct', False),
            'significant_10pct': results.get('is_significant_10pct', False),
            'r_squared': results.get('r_squared', np.nan),
            'n_observations': results.get('n_observations', 0)
        }
        
        if HAS_STATSMODELS and 'did_ci_lower' in results:
            row['ci_lower'] = results['did_ci_lower']
            row['ci_upper'] = results['did_ci_upper']
        
        results_for_csv.append(row)
    
    results_df = pd.DataFrame(results_for_csv)
    results_df.to_csv(csv_report_path, index=False)
    logger.info(f"📊 CSV结果已生成: {csv_report_path}")
    
    logger.info("✅ 验证报告生成完成!")
    return str(md_report_path)

def run_full_verification_analysis(dli_data_path: str = None,
                                  output_dir: str = None) -> Dict[str, str]:
    """
    执行完整的DLI统计验证分析流程
    
    这是统计验证模块的主要接口函数
    
    Args:
        dli_data_path: DLI面板数据文件路径
        output_dir: 输出目录
        
    Returns:
        包含输出文件路径的字典
    """
    
    logger.info("🚀 开始完整的DLI统计验证分析...")
    
    try:
        # 第1步：准备DID数据集
        logger.info("📋 第1步：准备DID分析数据集...")
        did_data = prepare_did_dataset(data_file_path=dli_data_path)
        
        # 第2步：执行DID分析
        logger.info("📊 第2步：执行双重差分分析...")
        did_results = run_did_analysis(did_data)
        
        # 第3步：生成验证报告
        logger.info("📝 第3步：生成统计验证报告...")
        report_path = generate_verification_report(did_results, did_data, output_dir)
        
        # 返回输出文件
        output_files = {
            'verification_report_md': report_path,
            'verification_results_csv': report_path.replace('.md', '_results.csv')
        }
        
        logger.info("✅ 完整的DLI统计验证分析完成!")
        logger.info(f"📄 报告文件: {output_files['verification_report_md']}")
        logger.info(f"📊 结果文件: {output_files['verification_results_csv']}")
        
        return output_files
        
    except Exception as e:
        logger.error(f"❌ DLI统计验证分析失败: {e}")
        raise

# 简化版统计函数（当没有statsmodels时使用）
if not HAS_STATSMODELS:
    from scipy import stats
    logger.warning("statsmodels未安装，将使用scipy进行基础统计分析")

if __name__ == "__main__":
    # 测试统计验证功能
    try:
        output_files = run_full_verification_analysis()
        print("✅ DLI统计验证分析成功完成!")
        for file_type, path in output_files.items():
            print(f"📁 {file_type}: {path}")
        
    except Exception as e:
        logger.error(f"❌ 统计验证分析失败: {e}")
        raise