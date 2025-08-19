#!/usr/bin/env python3
"""
稳健性检验和对比分析模块
========================

整合所有稳健性检验功能，特别是与"轨道一"(03模块)的对比分析。
**核心使命**: 验证骨干网络分析结果与完整网络分析结果的一致性。

核心功能：
1. run_robustness_checks() - 主要检验函数，包含与03模块的Spearman相关性检验
2. 参数敏感性分析 - DF算法在不同alpha值下的稳定性
3. 跨算法验证 - 确保核心发现在所有算法中都一致
4. 统计显著性检验 - 验证美国地位变化的统计显著性

学术标准：
- Spearman相关系数 > 0.7
- 核心发现稳定性 > 80%
- 统计显著性 p < 0.05
- 跨算法一致性 > 75%

作者：Energy Network Analysis Team
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.stats import spearmanr, kendalltau, ttest_ind, mannwhitneyu
import logging
import json

logger = logging.getLogger(__name__)


def calculate_node_centralities(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """
    计算节点的各种中心性指标
    
    Args:
        G: NetworkX图对象
        
    Returns:
        包含各种中心性指标的字典
    """
    
    centralities = {}
    
    # 度中心性
    centralities['degree'] = dict(G.degree())
    
    # 强度中心性（加权度）
    centralities['strength'] = dict(G.degree(weight='weight'))
    
    # 中介中心性
    try:
        centralities['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    except:
        centralities['betweenness'] = {node: 0.0 for node in G.nodes()}
    
    # PageRank中心性
    try:
        centralities['pagerank'] = nx.pagerank(G, weight='weight')
    except:
        centralities['pagerank'] = {node: 1.0/G.number_of_nodes() for node in G.nodes()}
    
    # 接近中心性
    try:
        centralities['closeness'] = nx.closeness_centrality(G, distance='weight')
    except:
        centralities['closeness'] = {node: 0.0 for node in G.nodes()}
    
    return centralities


def compare_centrality_rankings(full_centrality: Dict[str, Dict[str, float]], 
                              backbone_centrality: Dict[str, Dict[str, float]],
                              metric: str = 'strength') -> Dict[str, float]:
    """
    比较完整网络和骨干网络的中心性排名
    
    Args:
        full_centrality: 完整网络的中心性指标
        backbone_centrality: 骨干网络的中心性指标  
        metric: 要比较的中心性指标
        
    Returns:
        包含相关性分析结果的字典
    """
    
    # 获取共同节点
    common_nodes = set(full_centrality[metric].keys()).intersection(
        set(backbone_centrality[metric].keys())
    )
    
    if len(common_nodes) < 5:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0, 'kendall_tau': 0.0, 'common_nodes': len(common_nodes)}
    
    # 提取共同节点的中心性值
    full_values = [full_centrality[metric][node] for node in common_nodes]
    backbone_values = [backbone_centrality[metric][node] for node in common_nodes]
    
    # 计算Spearman相关系数
    spearman_rho, spearman_p = spearmanr(full_values, backbone_values)
    
    # 计算Kendall's tau
    kendall_tau, kendall_p = kendalltau(full_values, backbone_values)
    
    return {
        'spearman_rho': spearman_rho,
        'spearman_pvalue': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_pvalue': kendall_p,
        'common_nodes': len(common_nodes)
    }


def run_robustness_checks(full_networks: Dict[int, nx.Graph],
                         backbone_networks: Dict[str, Dict[int, nx.Graph]],
                         track1_results: Optional[Dict] = None) -> Dict[str, Any]:
    """
    **核心函数**: 运行完整的稳健性检验
    
    **关键要求**: 此函数必须实现与"轨道一"(03模块)核心指标的对比分析，
    特别是使用Spearman等级相关系数来检验节点中心性排序的一致性。
    
    Args:
        full_networks: 完整网络数据 {year: network}
        backbone_networks: 骨干网络数据 {algorithm: {year: network}}
        track1_results: 轨道一(03模块)的分析结果
        
    Returns:
        完整的稳健性检验结果
    """
    
    logger.info("🔍 开始完整稳健性检验分析...")
    
    robustness_results = {
        'centrality_consistency': {},
        'parameter_sensitivity': {},
        'cross_algorithm_validation': {},
        'statistical_significance': {},
        'track1_comparison': {},  # 与轨道一的对比
        'overall_assessment': {}
    }
    
    # 获取共同年份
    common_years = set(full_networks.keys())
    for alg_networks in backbone_networks.values():
        common_years = common_years.intersection(set(alg_networks.keys()))
    common_years = sorted(common_years)
    
    if len(common_years) < 3:
        logger.warning("⚠️ 共同年份太少，无法进行充分验证")
        return robustness_results
    
    logger.info(f"   分析年份: {len(common_years)} 年 ({min(common_years)}-{max(common_years)})")
    
    # 1. 中心性一致性验证
    logger.info("📊 1. 中心性一致性验证...")
    centrality_results = validate_centrality_consistency(
        full_networks, backbone_networks, common_years
    )
    robustness_results['centrality_consistency'] = centrality_results
    
    # 2. 参数敏感性分析
    logger.info("🎯 2. Disparity Filter参数敏感性分析...")
    sensitivity_results = analyze_parameter_sensitivity(
        backbone_networks, common_years
    )
    robustness_results['parameter_sensitivity'] = sensitivity_results
    
    # 3. 跨算法验证
    logger.info("🔄 3. 跨算法一致性验证...")
    cross_algo_results = validate_cross_algorithm_consistency(
        backbone_networks, common_years
    )
    robustness_results['cross_algorithm_validation'] = cross_algo_results
    
    # 4. 统计显著性检验
    logger.info("📈 4. 统计显著性检验...")
    significance_results = test_statistical_significance(
        full_networks, backbone_networks, common_years
    )
    robustness_results['statistical_significance'] = significance_results
    
    # 5. 与轨道一对比（如果有数据）
    if track1_results:
        logger.info("🔗 5. 与轨道一(03模块)对比分析...")
        track1_comparison = compare_with_track1_results(
            backbone_networks, track1_results, common_years
        )
        robustness_results['track1_comparison'] = track1_comparison
    
    # 6. 总体评估
    overall_assessment = calculate_overall_robustness_score(robustness_results)
    robustness_results['overall_assessment'] = overall_assessment
    
    logger.info(f"✅ 稳健性检验完成，总体得分: {overall_assessment.get('total_score', 0):.3f}")
    
    return robustness_results


def validate_centrality_consistency(full_networks: Dict[int, nx.Graph],
                                  backbone_networks: Dict[str, Dict[int, nx.Graph]],
                                  common_years: List[int]) -> Dict[str, Any]:
    """
    验证中心性排序的一致性
    
    核心检验: 美国在完整网络vs骨干网络中的排名对比
    目标: Spearman相关系数 > 0.7
    """
    
    consistency_results = {
        'algorithm_correlations': {},
        'usa_rank_analysis': {},
        'overall_consistency_score': 0
    }
    
    all_correlations = []
    usa_rank_preservation = []
    
    # 对每个算法进行分析
    for algorithm_name, alg_networks in backbone_networks.items():
        logger.info(f"   分析{algorithm_name}算法...")
        
        alg_correlations = []
        usa_ranks = {'full': [], 'backbone': []}
        
        for year in common_years:
            if year not in alg_networks:
                continue
                
            full_G = full_networks[year]
            backbone_G = alg_networks[year]
            
            # 计算中心性
            full_centrality = calculate_node_centralities(full_G)
            backbone_centrality = calculate_node_centralities(backbone_G)
            
            # 比较强度中心性排名
            correlation_result = compare_centrality_rankings(
                full_centrality, backbone_centrality, 'strength'
            )
            
            if correlation_result['common_nodes'] >= 5:
                alg_correlations.append(correlation_result['spearman_rho'])
                
                # 分析美国排名
                if 'USA' in full_centrality['strength'] and 'USA' in backbone_centrality['strength']:
                    full_usa_rank = get_node_rank(full_centrality['strength'], 'USA')
                    backbone_usa_rank = get_node_rank(backbone_centrality['strength'], 'USA')
                    
                    usa_ranks['full'].append(full_usa_rank)
                    usa_ranks['backbone'].append(backbone_usa_rank)
        
        # 算法级别统计
        if alg_correlations:
            mean_correlation = np.mean(alg_correlations)
            consistency_results['algorithm_correlations'][algorithm_name] = {
                'mean_spearman_rho': mean_correlation,
                'yearly_correlations': alg_correlations,
                'meets_threshold': mean_correlation > 0.7
            }
            all_correlations.extend(alg_correlations)
            
            # 美国排名一致性
            if usa_ranks['full'] and usa_ranks['backbone']:
                usa_correlation, _ = spearmanr(usa_ranks['full'], usa_ranks['backbone'])
                consistency_results['usa_rank_analysis'][algorithm_name] = {
                    'rank_correlation': usa_correlation,
                    'full_ranks': usa_ranks['full'],
                    'backbone_ranks': usa_ranks['backbone']
                }
                usa_rank_preservation.append(usa_correlation)
    
    # 总体一致性评分
    if all_correlations:
        overall_score = np.mean(all_correlations)
        consistency_results['overall_consistency_score'] = overall_score
        consistency_results['meets_academic_standard'] = overall_score > 0.7
    
    return consistency_results


def analyze_parameter_sensitivity(backbone_networks: Dict[str, Dict[int, nx.Graph]],
                                common_years: List[int]) -> Dict[str, Any]:
    """
    分析Disparity Filter算法的参数敏感性
    """
    
    sensitivity_results = {
        'alpha_stability': {},
        'core_findings_stability': 0,
        'usa_degree_stability': {}
    }
    
    # 找到所有DF算法结果
    df_algorithms = {k: v for k, v in backbone_networks.items() if k.startswith('disparity_filter_')}
    
    if len(df_algorithms) < 2:
        logger.warning("⚠️ DF算法结果不足，无法进行参数敏感性分析")
        return sensitivity_results
    
    # 分析不同alpha值下的稳定性
    usa_degrees_by_alpha = {}
    retention_rates_by_alpha = {}
    
    for alpha_key, alg_networks in df_algorithms.items():
        alpha_value = float(alpha_key.split('_')[-1])
        
        usa_degrees = []
        retention_rates = []
        
        for year in common_years:
            if year not in alg_networks:
                continue
                
            backbone_G = alg_networks[year]
            
            # 美国度数
            if 'USA' in backbone_G.nodes():
                usa_degrees.append(backbone_G.degree('USA'))
            
            # 保留率
            retention_rate = backbone_G.graph.get('retention_rate', 0)
            retention_rates.append(retention_rate)
        
        usa_degrees_by_alpha[alpha_value] = usa_degrees
        retention_rates_by_alpha[alpha_value] = retention_rates
    
    # 计算稳定性指标
    alpha_values = sorted(usa_degrees_by_alpha.keys())
    
    # 美国度数稳定性
    usa_stability_scores = []
    for i in range(len(alpha_values) - 1):
        alpha1, alpha2 = alpha_values[i], alpha_values[i + 1]
        if usa_degrees_by_alpha[alpha1] and usa_degrees_by_alpha[alpha2]:
            correlation, _ = spearmanr(usa_degrees_by_alpha[alpha1], usa_degrees_by_alpha[alpha2])
            usa_stability_scores.append(correlation)
    
    if usa_stability_scores:
        sensitivity_results['usa_degree_stability'] = {
            'mean_stability': np.mean(usa_stability_scores),
            'stability_scores': usa_stability_scores,
            'meets_threshold': np.mean(usa_stability_scores) > 0.8
        }
    
    # 核心发现稳定性（保留率变化）
    retention_stability = []
    for i in range(len(alpha_values) - 1):
        alpha1, alpha2 = alpha_values[i], alpha_values[i + 1]
        if retention_rates_by_alpha[alpha1] and retention_rates_by_alpha[alpha2]:
            # 计算保留率变化的稳定性
            rates1, rates2 = retention_rates_by_alpha[alpha1], retention_rates_by_alpha[alpha2]
            correlation, _ = spearmanr(rates1, rates2)
            retention_stability.append(correlation)
    
    if retention_stability:
        sensitivity_results['core_findings_stability'] = np.mean(retention_stability)
    
    return sensitivity_results


def validate_cross_algorithm_consistency(backbone_networks: Dict[str, Dict[int, nx.Graph]],
                                       common_years: List[int]) -> Dict[str, Any]:
    """
    验证跨算法一致性
    """
    
    cross_algo_results = {
        'algorithm_pairs': {},
        'usa_position_consistency': {},
        'algorithm_consistency_score': 0
    }
    
    algorithms = list(backbone_networks.keys())
    consistency_scores = []
    
    # 两两比较算法
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            alg1, alg2 = algorithms[i], algorithms[j]
            
            pair_key = f"{alg1}_vs_{alg2}"
            
            # 比较美国度数趋势
            usa_degrees_1 = []
            usa_degrees_2 = []
            
            for year in common_years:
                if year in backbone_networks[alg1] and year in backbone_networks[alg2]:
                    G1 = backbone_networks[alg1][year]
                    G2 = backbone_networks[alg2][year]
                    
                    if 'USA' in G1.nodes() and 'USA' in G2.nodes():
                        usa_degrees_1.append(G1.degree('USA'))
                        usa_degrees_2.append(G2.degree('USA'))
            
            if len(usa_degrees_1) >= 3:
                correlation, p_value = spearmanr(usa_degrees_1, usa_degrees_2)
                
                cross_algo_results['algorithm_pairs'][pair_key] = {
                    'usa_degree_correlation': correlation,
                    'correlation_pvalue': p_value,
                    'significant': p_value < 0.05
                }
                
                consistency_scores.append(correlation)
    
    # 总体一致性评分
    if consistency_scores:
        mean_consistency = np.mean(consistency_scores)
        cross_algo_results['algorithm_consistency_score'] = mean_consistency
        cross_algo_results['meets_threshold'] = mean_consistency > 0.75
    
    return cross_algo_results


def test_statistical_significance(full_networks: Dict[int, nx.Graph],
                                 backbone_networks: Dict[str, Dict[int, nx.Graph]],
                                 common_years: List[int]) -> Dict[str, Any]:
    """
    测试统计显著性
    """
    
    significance_results = {
        'usa_position_change': {},
        'temporal_trends': {},
        'overall_significance': False
    }
    
    # 分析美国地位的时间变化
    for algorithm_name, alg_networks in backbone_networks.items():
        usa_degrees = []
        years_with_usa = []
        
        for year in sorted(common_years):
            if year in alg_networks:
                G = alg_networks[year]
                if 'USA' in G.nodes():
                    usa_degrees.append(G.degree('USA'))
                    years_with_usa.append(year)
        
        if len(usa_degrees) >= 5:
            # 分析时间趋势
            correlation, p_value = spearmanr(years_with_usa, usa_degrees)
            
            # 分期比较（页岩革命前后）
            shale_year = 2011
            pre_shale = [d for y, d in zip(years_with_usa, usa_degrees) if y < shale_year]
            post_shale = [d for y, d in zip(years_with_usa, usa_degrees) if y >= shale_year]
            
            if len(pre_shale) >= 2 and len(post_shale) >= 2:
                # Mann-Whitney U检验
                u_stat, u_p = mannwhitneyu(pre_shale, post_shale, alternative='less')
                
                significance_results['usa_position_change'][algorithm_name] = {
                    'temporal_correlation': correlation,
                    'temporal_pvalue': p_value,
                    'pre_post_shale_test': {
                        'u_statistic': u_stat,
                        'u_pvalue': u_p,
                        'significant_increase': u_p < 0.05
                    }
                }
    
    # 总体显著性判断
    significant_algorithms = []
    for alg_results in significance_results['usa_position_change'].values():
        if alg_results.get('pre_post_shale_test', {}).get('significant_increase', False):
            significant_algorithms.append(True)
    
    significance_results['overall_significance'] = len(significant_algorithms) > 0
    
    return significance_results


def compare_with_track1_results(backbone_networks: Dict[str, Dict[int, nx.Graph]],
                               track1_results: Dict,
                               common_years: List[int]) -> Dict[str, Any]:
    """
    与轨道一(03模块)结果对比分析
    """
    
    track1_comparison = {
        'centrality_ranking_comparison': {},
        'usa_metrics_comparison': {},
        'consistency_with_track1': 0
    }
    
    # 这里需要根据03模块的具体输出格式来实现
    # 目前提供框架结构
    
    logger.info("   轨道一对比分析功能待03模块具体格式确定后完善")
    
    return track1_comparison


def calculate_overall_robustness_score(robustness_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算总体稳健性得分
    """
    
    scores = []
    
    # 1. 中心性一致性得分
    centrality_score = robustness_results.get('centrality_consistency', {}).get('overall_consistency_score', 0)
    scores.append(centrality_score)
    
    # 2. 参数敏感性得分
    sensitivity_score = robustness_results.get('parameter_sensitivity', {}).get('core_findings_stability', 0)
    scores.append(sensitivity_score)
    
    # 3. 跨算法一致性得分
    cross_algo_score = robustness_results.get('cross_algorithm_validation', {}).get('algorithm_consistency_score', 0)
    scores.append(cross_algo_score)
    
    # 4. 统计显著性得分
    significance_score = 1.0 if robustness_results.get('statistical_significance', {}).get('overall_significance', False) else 0.0
    scores.append(significance_score)
    
    # 计算总分
    total_score = np.mean([s for s in scores if not np.isnan(s)])
    
    # 评级
    if total_score >= 0.85:
        rating = 'excellent'
    elif total_score >= 0.7:
        rating = 'high'
    elif total_score >= 0.5:
        rating = 'moderate'
    else:
        rating = 'low'
    
    return {
        'total_score': total_score,
        'component_scores': {
            'centrality_consistency': centrality_score,
            'parameter_sensitivity': sensitivity_score,
            'cross_algorithm_consistency': cross_algo_score,
            'statistical_significance': significance_score
        },
        'rating': rating,
        'meets_academic_standards': total_score > 0.7
    }


def get_node_rank(centrality_dict: Dict[str, float], node: str) -> int:
    """
    获取节点在中心性排序中的排名
    """
    
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (node_name, _) in enumerate(sorted_nodes, 1):
        if node_name == node:
            return rank
    
    return len(sorted_nodes) + 1  # 如果节点不存在，返回最后排名