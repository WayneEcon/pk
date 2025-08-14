#!/usr/bin/env python3
"""
双向DLI统计验证模块 v2.0 (Bidirectional DLI Statistical Verification Module)
============================================================================

本模块专为双向DLI分析系统设计，使用双重差分法(DID)等准实验方法，
对"页岩革命是否显著改变了美国能源贸易的双向锁定格局"进行严谨的统计验证。

核心功能：
1. 进口锁定DID分析：验证美国被供应商锁定程度的变化
2. 出口锁定DID分析：验证美国锁定其他国家程度的变化
3. 双向对比分析：量化权力关系反转效应

DID实验设计：
- 处理组：美-加、美-墨的管道贸易关系（高专用性基础设施）
- 控制组：与沙特、卡塔尔等的海运贸易关系（低专用性）
- 政策冲击时点：2011年（页岩革命显著产出效应年份）

作者：Energy Network Analysis Team
版本：2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
import json
warnings.filterwarnings('ignore')

# 导入statsmodels
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
CONTROL_COUNTRIES = ['SAU', 'QAT', 'VEN', 'NOR', 'GBR', 'RUS', 'ARE']  # 控制组
PIPELINE_PRODUCTS = ['Crude_Oil', 'Natural_Gas']  # 管道运输产品
POLICY_SHOCK_YEAR = 2011  # 页岩革命冲击年份
PRE_PERIOD = (2001, 2010)  # 政策前期间
POST_PERIOD = (2011, 2024)  # 政策后期间

def load_bidirectional_dli_data(data_file_path: str = None) -> pd.DataFrame:
    """
    加载双向DLI面板数据
    
    Args:
        data_file_path: 数据文件路径，默认使用v2数据文件
        
    Returns:
        双向DLI面板数据DataFrame
    """
    
    if data_file_path is None:
        base_dir = Path(__file__).parent.parent.parent
        data_file_path = Path(__file__).parent / "dli_panel_data_v2.csv"
    
    if not Path(data_file_path).exists():
        raise FileNotFoundError(f"双向DLI数据文件不存在: {data_file_path}")
    
    df = pd.read_csv(data_file_path)
    logger.info(f"📂 成功加载双向DLI数据: {len(df):,} 条记录")
    
    # 验证数据结构
    required_columns = ['year', 'us_partner', 'energy_product', 'locking_dimension_type', 'dli_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {missing_columns}")
    
    # 数据概览
    locking_stats = df.groupby('locking_dimension_type').agg({
        'dli_score': ['count', 'mean', 'std']
    }).round(4)
    logger.info("📊 双向锁定数据分布:")
    print(locking_stats)
    
    return df

def prepare_did_dataset_v2(df: pd.DataFrame, locking_type: str) -> pd.DataFrame:
    """
    为指定锁定类型准备DID分析数据集
    
    Args:
        df: 双向DLI面板数据
        locking_type: 锁定类型 ('import_locking' 或 'export_locking')
        
    Returns:
        准备好的DID分析数据集
    """
    
    logger.info(f"🎯 准备{locking_type}的DID分析数据集...")
    
    # 筛选指定锁定类型的数据
    df_filtered = df[df['locking_dimension_type'] == locking_type].copy()
    logger.info(f"筛选{locking_type}数据: {len(df_filtered):,} 条记录")
    
    if len(df_filtered) == 0:
        raise ValueError(f"未找到{locking_type}类型的数据")
    
    # 定义处理组和控制组
    if locking_type == 'import_locking':
        # 进口锁定：处理组为管道贸易，控制组为海运贸易
        treatment_condition = (
            df_filtered['us_partner'].isin(TREATMENT_COUNTRIES) & 
            df_filtered['energy_product'].isin(PIPELINE_PRODUCTS)
        )
        control_condition = (
            df_filtered['us_partner'].isin(CONTROL_COUNTRIES) & 
            ~df_filtered['energy_product'].isin(['Coal'])
        )
    else:
        # 出口锁定：处理组为对邻国出口，控制组为对远距离国家出口
        treatment_condition = (
            df_filtered['us_partner'].isin(TREATMENT_COUNTRIES)
        )
        control_condition = (
            df_filtered['us_partner'].isin(CONTROL_COUNTRIES)
        )
    
    # 创建DID样本
    did_sample = df_filtered[treatment_condition | control_condition].copy()
    
    if len(did_sample) == 0:
        raise ValueError(f"{locking_type}的DID样本为空")
    
    # 创建DID变量
    did_sample['treatment'] = treatment_condition[treatment_condition | control_condition].astype(int)
    did_sample['post'] = (did_sample['year'] >= POLICY_SHOCK_YEAR).astype(int)
    did_sample['treatment_post'] = did_sample['treatment'] * did_sample['post']
    did_sample['period'] = did_sample['year'].apply(
        lambda x: 'pre' if x < POLICY_SHOCK_YEAR else 'post'
    )
    
    # 创建控制变量
    did_sample['log_trade_value'] = np.log(did_sample['trade_value_usd'] + 1)
    if 'distance_km' in did_sample.columns:
        did_sample['log_distance'] = np.log(did_sample['distance_km'])
    did_sample['year_trend'] = did_sample['year'] - 2001
    did_sample['country_product'] = did_sample['us_partner'] + '_' + did_sample['energy_product']
    
    # 实验设计验证
    logger.info("🔍 DID实验设计验证:")
    
    # 按组和时期统计
    group_stats = did_sample.groupby(['treatment', 'period']).agg({
        'us_partner': 'nunique',
        'energy_product': 'nunique',
        'dli_score': ['count', 'mean', 'std']
    }).round(4)
    
    logger.info("实验组构成统计:")
    print(group_stats)
    
    # 处理组和控制组国家
    treatment_countries = did_sample[did_sample['treatment'] == 1]['us_partner'].unique()
    control_countries = did_sample[did_sample['treatment'] == 0]['us_partner'].unique()
    
    logger.info(f"  处理组国家: {sorted(treatment_countries)}")
    logger.info(f"  控制组国家: {sorted(control_countries)}")
    
    logger.info(f"✅ {locking_type} DID数据集准备完成: {len(did_sample):,} 观测")
    
    return did_sample

def run_did_regression_v2(did_data: pd.DataFrame, 
                         outcome_vars: List[str] = None,
                         control_vars: List[str] = None,
                         locking_type: str = 'import_locking') -> Dict[str, Dict]:
    """
    执行DID回归分析（使用聚类稳健标准误）
    
    Args:
        did_data: DID分析数据集
        outcome_vars: 结果变量列表
        control_vars: 控制变量列表
        locking_type: 锁定类型标识
        
    Returns:
        DID分析结果字典
    """
    
    logger.info(f"🧮 开始执行{locking_type}的DID回归分析...")
    
    if not HAS_STATSMODELS:
        raise ImportError("需要安装statsmodels库进行回归分析")
    
    # 默认结果变量
    if outcome_vars is None:
        potential_outcomes = ['dli_score', 'continuity', 'infrastructure', 'stability', 'market_locking_power']
        outcome_vars = [var for var in potential_outcomes if var in did_data.columns]
    
    # 默认控制变量
    if control_vars is None:
        potential_controls = ['log_trade_value', 'log_distance', 'year_trend']
        control_vars = [var for var in potential_controls if var in did_data.columns]
    
    logger.info(f"结果变量: {outcome_vars}")
    logger.info(f"控制变量: {control_vars}")
    
    results = {}
    
    for outcome_var in outcome_vars:
        logger.info(f"分析 {outcome_var}...")
        
        # 构建回归公式
        formula = f"{outcome_var} ~ treatment + post + treatment_post"
        if control_vars:
            formula += " + " + " + ".join(control_vars)
        
        logger.info(f"回归公式: {formula}")
        
        # 准备回归数据（移除缺失值）
        reg_vars = [outcome_var, 'treatment', 'post', 'treatment_post'] + control_vars
        cluster_vars = ['us_partner']  # 聚类变量
        all_vars = reg_vars + cluster_vars
        
        reg_data = did_data[all_vars].dropna()
        
        if len(reg_data) < 50:  # 最少样本量检查
            logger.warning(f"⚠️ {outcome_var}的有效样本量过少: {len(reg_data)}")
            continue
        
        try:
            # 运行回归 - 使用聚类稳健标准误
            model = smf.ols(formula, data=reg_data).fit(
                cov_type='cluster', 
                cov_kwds={'groups': reg_data['us_partner']}
            )
            
            # 提取DID系数及统计量
            did_coef = model.params['treatment_post']
            did_se = model.bse['treatment_post']
            did_tstat = model.tvalues['treatment_post']
            did_pvalue = model.pvalues['treatment_post']
            did_ci = model.conf_int().loc['treatment_post'].tolist()
            
            # 判断显著性
            significant_5pct = did_pvalue < 0.05
            significant_10pct = did_pvalue < 0.10
            
            # 保存结果
            results[outcome_var] = {
                'did_coefficient': did_coef,
                'did_std_error': did_se,
                'did_t_statistic': did_tstat,
                'did_p_value': did_pvalue,
                'significant_5pct': significant_5pct,
                'significant_10pct': significant_10pct,
                'r_squared': model.rsquared,
                'n_observations': len(reg_data),
                'ci_lower': did_ci[0],
                'ci_upper': did_ci[1],
                'locking_type': locking_type,
                'formula': formula
            }
            
            # 输出结果
            significance = "***" if did_pvalue < 0.01 else "**" if did_pvalue < 0.05 else "*" if did_pvalue < 0.10 else ""
            direction = "↑" if did_coef > 0 else "↓"
            
            logger.info(f"  {outcome_var}: {did_coef:+.4f} {significance} (p={did_pvalue:.4f}) {direction}")
            
        except Exception as e:
            logger.error(f"❌ {outcome_var}回归分析失败: {e}")
            continue
    
    logger.info(f"✅ {locking_type} DID回归分析完成，成功分析 {len(results)} 个指标")
    return results

def run_full_bidirectional_did_analysis(dli_data: pd.DataFrame = None) -> Dict[str, Dict]:
    """
    执行完整的双向DID分析
    
    Args:
        dli_data: 双向DLI面板数据，如果为None则自动加载
        
    Returns:
        包含进口锁定和出口锁定DID分析结果的字典
    """
    
    logger.info("🚀 开始完整的双向DID分析...")
    
    # 加载数据
    if dli_data is None:
        dli_data = load_bidirectional_dli_data()
    
    results = {}
    
    # 1. 进口锁定DID分析
    logger.info("📥 执行进口锁定DID分析...")
    try:
        import_data = prepare_did_dataset_v2(dli_data, 'import_locking')
        results['import_locking'] = run_did_regression_v2(
            import_data, locking_type='import_locking'
        )
        logger.info(f"✅ 进口锁定DID分析完成: {len(results['import_locking'])} 个指标")
    except Exception as e:
        logger.error(f"❌ 进口锁定DID分析失败: {e}")
    
    # 2. 出口锁定DID分析  
    logger.info("📤 执行出口锁定DID分析...")
    try:
        export_data = prepare_did_dataset_v2(dli_data, 'export_locking')
        results['export_locking'] = run_did_regression_v2(
            export_data, locking_type='export_locking'
        )
        logger.info(f"✅ 出口锁定DID分析完成: {len(results['export_locking'])} 个指标")
    except Exception as e:
        logger.error(f"❌ 出口锁定DID分析失败: {e}")
    
    # 3. 总结分析结果
    logger.info("📊 双向DID分析总结:")
    for locking_type, type_results in results.items():
        significant_vars = [var for var, res in type_results.items() 
                          if res.get('significant_5pct', False)]
        logger.info(f"  {locking_type}: {len(significant_vars)}/{len(type_results)} 个指标在5%水平显著")
        for var in significant_vars:
            coef = type_results[var]['did_coefficient']
            p_val = type_results[var]['did_p_value']
            direction = "增强" if coef > 0 else "减弱"
            logger.info(f"    {var}: {coef:+.4f} (p={p_val:.4f}) - 锁定效应{direction}")
    
    logger.info(f"🎉 完整双向DID分析完成！成功分析 {len(results)} 个锁定维度")
    return results

def save_bidirectional_results(results: Dict[str, Dict], 
                              output_dir: str = None) -> Dict[str, str]:
    """
    保存双向DID分析结果
    
    Args:
        results: 双向DID分析结果
        output_dir: 输出目录，默认使用标准路径
        
    Returns:
        保存的文件路径字典
    """
    
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = Path(__file__).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = {}
    
    # 1. 保存详细结果为CSV
    all_results = []
    for locking_type, type_results in results.items():
        for variable, result in type_results.items():
            result_row = {
                'locking_type': locking_type,
                'variable': variable,
                **result
            }
            all_results.append(result_row)
    
    results_df = pd.DataFrame(all_results)
    csv_path = Path(output_dir) / "dli_verification_results_v2.csv"
    results_df.to_csv(csv_path, index=False)
    output_paths['results_csv'] = str(csv_path)
    logger.info(f"💾 详细结果已保存至: {csv_path}")
    
    # 2. 保存JSON格式结果
    json_path = Path(output_dir) / "dli_verification_results_v2.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        serializable_results = {}
        for locking_type, type_results in results.items():
            serializable_results[locking_type] = {}
            for var, result in type_results.items():
                serializable_results[locking_type][var] = {
                    k: convert_numpy(v) for k, v in result.items()
                }
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    output_paths['results_json'] = str(json_path)
    logger.info(f"💾 JSON结果已保存至: {json_path}")
    
    return output_paths

def generate_verification_report_v2(results: Dict[str, Dict], 
                                   output_dir: str = None) -> str:
    """
    生成双向DLI验证报告
    
    Args:
        results: 双向DID分析结果
        output_dir: 输出目录
        
    Returns:
        生成的报告文件路径
    """
    
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = Path(__file__).parent
    
    report_path = Path(output_dir) / "dli_verification_report_v2.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 双向动态锁定指数(DLI)统计验证报告 v2.0\\n\\n")
        f.write("**生成时间**: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\\n")
        f.write("**分析方法**: 双重差分法(DID)，聚类稳健标准误\\n")
        f.write("**政策冲击**: 页岩革命(2011年)\\n\\n")
        
        f.write("---\\n\\n")
        
        # 分析概述
        f.write("## 📊 分析概述\\n\\n")
        f.write("本报告基于双向DLI分析系统，使用DID方法验证页岩革命对美国能源贸易锁定关系的双向影响：\\n\\n")
        f.write("- **进口锁定**: 美国被供应商锁定的程度变化\\n")
        f.write("- **出口锁定**: 美国锁定其他国家的程度变化\\n\\n")
        
        # 实验设计
        f.write("## 🧪 实验设计\\n\\n")
        f.write("### 处理组与控制组\\n")
        f.write(f"- **处理组**: {', '.join(TREATMENT_COUNTRIES)}（管道贸易，高专用性基础设施）\\n")
        f.write(f"- **控制组**: {', '.join(CONTROL_COUNTRIES)}（海运贸易，低专用性基础设施）\\n")
        f.write(f"- **政策冲击时点**: {POLICY_SHOCK_YEAR}年\\n\\n")
        
        # 主要发现
        f.write("## 🔍 主要发现\\n\\n")
        
        for locking_type, type_results in results.items():
            type_name = "进口锁定" if locking_type == "import_locking" else "出口锁定"
            f.write(f"### {type_name}分析结果\\n\\n")
            
            # 创建结果表格
            f.write("| 指标 | DID系数 | 标准误 | t统计量 | p值 | 显著性 | 95%置信区间 |\\n")
            f.write("|------|---------|--------|---------|-----|--------|-------------|\\n")
            
            for variable, result in type_results.items():
                coef = result['did_coefficient']
                se = result['did_std_error']
                t_stat = result['did_t_statistic'] 
                p_val = result['did_p_value']
                ci_lower = result['ci_lower']
                ci_upper = result['ci_upper']
                
                # 显著性标记
                if p_val < 0.01:
                    sig = "***"
                elif p_val < 0.05:
                    sig = "**"
                elif p_val < 0.10:
                    sig = "*"
                else:
                    sig = ""
                
                f.write(f"| {variable} | {coef:+.4f} | {se:.4f} | {t_stat:+.2f} | {p_val:.4f} | {sig} | [{ci_lower:+.4f}, {ci_upper:+.4f}] |\\n")
            
            f.write("\\n")
            
            # 显著性解释
            significant_vars = [var for var, res in type_results.items() if res.get('significant_5pct', False)]
            if significant_vars:
                f.write(f"**{type_name}关键发现**：\\n")
                for var in significant_vars:
                    coef = type_results[var]['did_coefficient']
                    direction = "增强" if coef > 0 else "减弱"
                    f.write(f"- {var}: 锁定效应显著{direction} ({coef:+.4f})\\n")
                f.write("\\n")
        
        # 统计说明
        f.write("## 📝 统计说明\\n\\n")
        f.write("- **聚类稳健标准误**: 按国家聚类校正面板数据序列相关性\\n")
        f.write("- **显著性水平**: *** p<0.01, ** p<0.05, * p<0.10\\n")
        f.write("- **DID系数**: treatment_post交互项系数，表示政策对处理组的净效应\\n")
        f.write("- **正系数**: 锁定效应增强；负系数: 锁定效应减弱\\n\\n")
        
        f.write("---\\n\\n")
        f.write("*本报告由双向DLI统计验证模块v2.0自动生成*\\n")
    
    logger.info(f"📄 双向DLI验证报告已生成: {report_path}")
    return str(report_path)

if __name__ == "__main__":
    # 测试双向DID分析
    try:
        logger.info("🚀 开始双向DLI统计验证测试...")
        
        # 执行完整分析
        results = run_full_bidirectional_did_analysis()
        
        # 保存结果
        output_paths = save_bidirectional_results(results)
        
        # 生成报告
        report_path = generate_verification_report_v2(results)
        
        print("🎉 双向DLI统计验证完成！")
        print("📊 输出文件:")
        for desc, path in output_paths.items():
            print(f"  {desc}: {path}")
        print(f"  verification_report: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ 双向DLI统计验证失败: {e}")
        raise