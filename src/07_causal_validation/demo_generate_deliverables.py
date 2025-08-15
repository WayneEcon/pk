#!/usr/bin/env python3
"""
因果验证分析 - 关键产出生成演示
===============================

快速生成核心产出：
1. network_resilience.csv - 网络韧性数据库
2. causal_validation_report.md - 学术级验证报告

本脚本使用模拟数据演示完整分析流程。
"""

import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入分析模块
from resilience_calculator import generate_resilience_database, NetworkResilienceCalculator
from causal_model import run_causal_validation, CausalAnalyzer
import pandas as pd
import numpy as np
import networkx as nx

def create_demo_networks():
    """创建演示用的能源网络数据"""
    
    logger.info("🎭 创建演示网络数据...")
    
    # 主要能源国家
    countries = ['USA', 'CHN', 'RUS', 'SAU', 'CAN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 
                'BRA', 'MEX', 'AUS', 'NOR', 'ARE', 'KWT', 'IRN', 'IRQ', 'VEN', 'QAT']
    years = list(range(2010, 2025))
    
    networks = {}
    np.random.seed(42)  # 确保可重现性
    
    for year in years:
        G = nx.DiGraph()
        G.add_nodes_from(countries)
        
        # 生成现实的异质性贸易网络 - 修复共线性问题
        # 定义国家角色和特征（增加差异性）
        country_roles = {
            'USA': {'type': 'hub_importer', 'centrality_factor': 1.5, 'volatility': 0.1},
            'CHN': {'type': 'growing_hub', 'centrality_factor': 1.3, 'volatility': 0.15},
            'RUS': {'type': 'major_exporter', 'centrality_factor': 1.2, 'volatility': 0.2},
            'SAU': {'type': 'oil_exporter', 'centrality_factor': 1.0, 'volatility': 0.25},
            'DEU': {'type': 'industrial_hub', 'centrality_factor': 1.1, 'volatility': 0.08},
            'JPN': {'type': 'island_importer', 'centrality_factor': 0.9, 'volatility': 0.12},
            'CAN': {'type': 'resource_exporter', 'centrality_factor': 0.8, 'volatility': 0.15},
            'GBR': {'type': 'financial_hub', 'centrality_factor': 0.7, 'volatility': 0.1},
            'FRA': {'type': 'european_hub', 'centrality_factor': 0.75, 'volatility': 0.1},
            'IND': {'type': 'emerging_importer', 'centrality_factor': 0.85, 'volatility': 0.18},
            'BRA': {'type': 'regional_hub', 'centrality_factor': 0.6, 'volatility': 0.2},
            'MEX': {'type': 'transit_hub', 'centrality_factor': 0.5, 'volatility': 0.15},
            'AUS': {'type': 'resource_exporter', 'centrality_factor': 0.45, 'volatility': 0.12},
            'NOR': {'type': 'oil_exporter', 'centrality_factor': 0.4, 'volatility': 0.2},
            'ARE': {'type': 'oil_hub', 'centrality_factor': 0.5, 'volatility': 0.3},
            'KWT': {'type': 'oil_exporter', 'centrality_factor': 0.35, 'volatility': 0.25},
            'IRN': {'type': 'constrained_exporter', 'centrality_factor': 0.3, 'volatility': 0.4},
            'IRQ': {'type': 'unstable_exporter', 'centrality_factor': 0.25, 'volatility': 0.5},
            'VEN': {'type': 'declining_exporter', 'centrality_factor': 0.2, 'volatility': 0.6},
            'QAT': {'type': 'lng_exporter', 'centrality_factor': 0.4, 'volatility': 0.2}
        }
        
        # 时间效应：不同事件对不同国家的影响
        time_effects = {
            2014: {'RUS': -0.3, 'IRN': -0.2},  # 制裁
            2016: {'USA': 0.1, 'CAN': 0.05},   # 页岩油革命
            2018: {'IRN': -0.4, 'VEN': -0.3},  # 制裁加剧
            2020: {'all': -0.15},               # 疫情冲击
            2022: {'RUS': -0.5, 'USA': 0.2}    # 地缘政治冲击
        }
        
        # 添加异质性贸易边
        for exporter in countries:
            if exporter not in country_roles:
                continue
                
            exporter_role = country_roles[exporter]
            
            for importer in countries:
                if exporter == importer or importer not in country_roles:
                    continue
                    
                importer_role = country_roles[importer]
                
                # 基于角色确定贸易概率
                trade_prob = 0.1  # 基础概率
                
                # 出口国类型影响
                if exporter_role['type'] in ['major_exporter', 'oil_exporter', 'resource_exporter']:
                    trade_prob += 0.3
                    
                # 进口国类型影响  
                if importer_role['type'] in ['hub_importer', 'growing_hub', 'industrial_hub']:
                    trade_prob += 0.2
                
                # 地理和政治关系
                if (exporter == 'CAN' and importer == 'USA') or \
                   (exporter == 'MEX' and importer == 'USA') or \
                   (exporter in ['SAU', 'ARE', 'KWT'] and importer in ['CHN', 'JPN', 'IND']):
                    trade_prob += 0.4
                
                if np.random.random() < trade_prob:
                    # 基础贸易流量（基于国家角色）
                    base_flow = np.random.lognormal(
                        1 + exporter_role['centrality_factor'], 
                        exporter_role['volatility']
                    )
                    
                    # 时间趋势效应
                    year_trend = 1 + 0.02 * (year - 2010)
                    
                    # 特定年份冲击
                    shock_effect = 1.0
                    if year in time_effects:
                        if exporter in time_effects[year]:
                            shock_effect *= (1 + time_effects[year][exporter])
                        elif 'all' in time_effects[year]:
                            shock_effect *= (1 + time_effects[year]['all'])
                    
                    # 最终贸易流量
                    final_flow = base_flow * year_trend * shock_effect * (1 + np.random.normal(0, 0.2))
                    
                    if final_flow > 0:
                        G.add_edge(exporter, importer, weight=final_flow)
        
        # 添加一些反向流动（成品油、LNG等）
        major_importers = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND']
        major_exporters = ['SAU', 'RUS', 'CAN', 'NOR', 'IRN', 'IRQ', 'VEN', 'ARE', 'KWT', 'QAT']
        
        for importer in major_importers:
            for exporter in major_exporters:
                if importer in countries and exporter in countries:
                    if np.random.random() < 0.15:  # 15%概率有反向贸易
                        reverse_flow = np.random.lognormal(1, 0.5)
                        if G.has_edge(importer, exporter):
                            G[importer][exporter]['weight'] += reverse_flow
                        else:
                            G.add_edge(importer, exporter, weight=reverse_flow)
        
        networks[year] = G
        logger.info(f"   {year}年: {G.number_of_nodes()}节点, {G.number_of_edges()}边")
    
    return networks

def create_demo_dli_data(networks):
    """基于网络数据创建演示DLI数据"""
    
    logger.info("📊 创建演示DLI数据...")
    
    dli_data = []
    np.random.seed(42)
    
    # 国家基础DLI特征（增加差异性）
    country_base_dli = {
        'USA': 0.25,  # 能源独立性较强
        'CHN': 0.65,  # 高度依赖进口
        'RUS': 0.15,  # 主要出口国
        'SAU': 0.10,  # 石油出口大国
        'DEU': 0.55,  # 工业国家，依赖进口
        'JPN': 0.70,  # 岛国，高度依赖
        'CAN': 0.20,  # 资源丰富
        'GBR': 0.60,  # 后工业化，依赖进口
        'FRA': 0.50,  # 混合能源结构
        'IND': 0.58,  # 新兴经济体，需求增长
        'BRA': 0.35,  # 南美地区大国
        'MEX': 0.45,  # 过渡型经济
        'AUS': 0.18,  # 资源出口国
        'NOR': 0.12,  # 石油出口国
        'ARE': 0.08,  # 海湾石油国
        'KWT': 0.06,  # 石油出口国
        'IRN': 0.22,  # 受制裁影响
        'IRQ': 0.30,  # 政局不稳
        'VEN': 0.40,  # 经济困难
        'QAT': 0.14   # 小型富国
    }
    
    for year, G in networks.items():
        for country in G.nodes():
            if country not in country_base_dli:
                continue
                
            # 基础DLI（国家特征）
            base_dli = country_base_dli[country]
            
            # 网络结构影响（增加现实性）
            network_effect = 0
            if G.number_of_edges() > 0:
                # 入度集中度（供应商依赖）
                in_edges = [(s, d['weight']) for s, t, d in G.in_edges(country, data=True)]
                if in_edges:
                    total_imports = sum(weight for _, weight in in_edges)
                    if total_imports > 0:
                        import_shares = [weight/total_imports for _, weight in in_edges]
                        supply_concentration = sum(share**2 for share in import_shares)
                        network_effect += 0.2 * supply_concentration
                
                # 出度多样化（出口能力）
                out_edges = [(t, d['weight']) for s, t, d in G.out_edges(country, data=True)]
                if out_edges:
                    total_exports = sum(weight for _, weight in out_edges)
                    export_diversity = len(out_edges) / max(1, G.number_of_nodes() - 1)
                    network_effect -= 0.1 * export_diversity  # 出口多样化降低锁定
            
            # 时间趋势效应（非线性）
            time_base = (year - 2010) / 15.0  # 标准化到[0,1]
            
            # 不同类型国家的时间趋势不同
            if country in ['USA', 'CAN', 'NOR']:  # 能源独立改善
                time_trend = -0.05 * time_base + 0.02 * time_base**2
            elif country in ['CHN', 'IND']:  # 依赖度先升后降
                time_trend = 0.1 * time_base - 0.08 * time_base**2
            elif country in ['IRN', 'VEN', 'IRQ']:  # 制裁和危机影响
                time_trend = 0.15 * time_base * np.sin(2 * np.pi * time_base)
            else:  # 其他国家缓慢变化
                time_trend = 0.02 * time_base * (1 + 0.5 * np.sin(np.pi * time_base))
            
            # 特定事件冲击（增加时间异质性）
            event_shock = 0
            if year == 2014 and country in ['RUS', 'IRN']:  # 制裁开始
                event_shock = 0.1
            elif year == 2016 and country == 'USA':  # 页岩油革命
                event_shock = -0.08
            elif year == 2018 and country in ['IRN', 'VEN']:  # 制裁加剧
                event_shock = 0.15
            elif year == 2020:  # 疫情影响（全球）
                event_shock = 0.05 * (1 + np.random.normal(0, 0.5))
            elif year >= 2022 and country == 'RUS':  # 地缘政治冲击
                event_shock = 0.2
            
            # 随机波动（国家特定的波动性）
            volatility = {
                'USA': 0.02, 'CHN': 0.03, 'RUS': 0.08, 'SAU': 0.06,
                'DEU': 0.02, 'JPN': 0.025, 'CAN': 0.03, 'GBR': 0.025,
                'FRA': 0.02, 'IND': 0.04, 'BRA': 0.05, 'MEX': 0.04,
                'AUS': 0.03, 'NOR': 0.05, 'ARE': 0.07, 'KWT': 0.06,
                'IRN': 0.12, 'IRQ': 0.15, 'VEN': 0.18, 'QAT': 0.05
            }.get(country, 0.04)
            
            random_shock = np.random.normal(0, volatility)
            
            # 最终DLI得分（确保有足够变异）
            dli_score = base_dli + network_effect + time_trend + event_shock + random_shock
            dli_score = np.clip(dli_score, 0.05, 0.95)  # 避免极值
            
            dli_data.append({
                'year': year,
                'country': country,
                'dli_score': dli_score
            })
    
    dli_df = pd.DataFrame(dli_data)
    logger.info(f"   DLI数据: {dli_df.shape}")
    
    return dli_df

def main():
    """主演示流程"""
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🎯 因果验证分析 - 关键产出生成演示                       ║
║   Causal Validation Analysis - Key Deliverables Demo    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 1. 创建演示数据
        logger.info("=" * 50)
        logger.info("第一步：创建演示数据")
        logger.info("=" * 50)
        
        networks = create_demo_networks()
        dli_data = create_demo_dli_data(networks)
        
        # 2. 生成网络韧性数据库
        logger.info("\n" + "=" * 50)
        logger.info("第二步：生成网络韧性数据库")
        logger.info("=" * 50)
        
        resilience_db = generate_resilience_database(
            networks, 
            output_path="outputs/network_resilience.csv",
            countries=['USA', 'CHN', 'RUS', 'SAU', 'DEU', 'JPN']  # 重点国家
        )
        
        print(f"✅ 网络韧性数据库已生成: outputs/network_resilience.csv")
        print(f"   数据维度: {resilience_db.shape}")
        print(f"   时间跨度: {resilience_db['year'].min()}-{resilience_db['year'].max()}")
        
        # 3. 运行因果验证分析
        logger.info("\n" + "=" * 50)  
        logger.info("第三步：因果验证分析")
        logger.info("=" * 50)
        
        causal_results = run_causal_validation(
            resilience_db,
            dli_data,
            output_dir="outputs"
        )
        
        # 4. 生成学术报告
        logger.info("\n" + "=" * 50)
        logger.info("第四步：生成学术报告")
        logger.info("=" * 50)
        
        from main import CausalValidationPipeline
        pipeline = CausalValidationPipeline(output_dir="outputs")
        pipeline.resilience_data = resilience_db
        pipeline.dli_data = dli_data
        pipeline.causal_results = causal_results
        
        report_file = pipeline.generate_academic_report()
        
        # 5. 总结输出
        print("\n" + "🎉" + "=" * 50 + "🎉")
        print("    关键产出生成完成！")
        print("  Key Deliverables Generated Successfully!")
        print("=" * 52)
        
        print(f"\n📁 输出文件:")
        print(f"   1. 网络韧性数据库: outputs/network_resilience.csv")
        print(f"   2. 因果验证报告: {Path(report_file).name}")
        print(f"   3. 回归结果表格: outputs/regression_results.csv")
        print(f"   4. 原始分析结果: outputs/causal_validation_results.json")
        
        # 显示关键统计
        overall_assessment = causal_results.get('overall_assessment', {})
        evidence_strength = overall_assessment.get('causal_evidence_strength', 'unknown')
        
        print(f"\n🎯 分析摘要:")
        print(f"   • 因果证据强度: {evidence_strength.upper()}")
        print(f"   • 统计显著性: {'通过' if overall_assessment.get('statistical_significance') else '未通过'}")
        print(f"   • 分析了 {len(networks)} 年网络数据")
        print(f"   • 生成了 {len(resilience_db)} 个韧性观测")
        
        print(f"\n💡 下一步: 查看 outputs/ 目录中的所有分析结果")
        
    except Exception as e:
        logger.error(f"❌ 演示执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()