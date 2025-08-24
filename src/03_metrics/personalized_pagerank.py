#!/usr/bin/env python3
"""
个性化PageRank计算模块 (Personalized PageRank Calculator)
==========================================================

计算美国在全球能源贸易网络中的方向性PageRank影响力，为DLI分析提供新的网络中心性维度。

核心功能：
1. 出口锁定影响力：以美国为种子节点，计算美国对其他国家的网络影响力
2. 进口锁定影响力：以其他国家为种子节点，计算它们对美国的网络影响力

输出文件：personalized_pagerank_panel.csv
- year: 年份
- country_name: 国家名称  
- ppr_us_export_influence: 美国对该国的出口锁定网络影响力
- ppr_influence_on_us: 该国对美国的进口锁定网络影响力

作者：Energy Network Analysis Team
版本：v1.0
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import logging
import argparse
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PersonalizedPageRankCalculator:
    """个性化PageRank计算器"""
    
    def __init__(self, networks_dir: Path, output_dir: Path):
        """
        初始化计算器
        
        Args:
            networks_dir: 网络数据目录
            output_dir: 输出目录
        """
        self.networks_dir = Path(networks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 网络数据目录: {self.networks_dir}")
        logger.info(f"📁 输出目录: {self.output_dir}")
    
    def load_annual_networks(self) -> Dict[int, nx.Graph]:
        """
        加载年度网络数据
        
        Returns:
            Dict[int, nx.Graph]: 年份到网络图的映射
        """
        logger.info("📂 开始加载年度网络数据...")
        
        networks = {}
        
        # 尝试多种可能的文件格式和命名
        potential_patterns = [
            "network_*.graphml",
            "network_*.gexf", 
            "*_network.graphml",
            "*.graphml",
            "*.gexf"
        ]
        
        for pattern in potential_patterns:
            network_files = list(self.networks_dir.glob(pattern))
            if network_files:
                logger.info(f"   找到网络文件模式: {pattern}")
                break
        else:
            # 尝试从pickle文件加载
            pkl_files = list(self.networks_dir.glob("*.pkl"))
            if pkl_files:
                logger.info("   尝试从pickle文件加载网络数据...")
                import pickle
                for pkl_file in pkl_files:
                    try:
                        with open(pkl_file, 'rb') as f:
                            annual_networks = pickle.load(f)
                        if isinstance(annual_networks, dict):
                            networks.update(annual_networks)
                            logger.info(f"   ✅ 从{pkl_file.name}加载了{len(annual_networks)}个年度网络")
                            return networks
                    except Exception as e:
                        logger.warning(f"   ⚠️ 加载{pkl_file}失败: {e}")
            
            raise FileNotFoundError(f"在{self.networks_dir}中未找到网络文件")
        
        # 从GraphML/GEXF文件加载
        for network_file in network_files:
            try:
                # 从文件名提取年份
                filename = network_file.stem
                year = None
                
                # 尝试多种年份提取方式
                for part in filename.split('_'):
                    if part.isdigit() and len(part) == 4:
                        potential_year = int(part)
                        if 2000 <= potential_year <= 2030:
                            year = potential_year
                            break
                
                if year is None:
                    logger.warning(f"   ⚠️ 无法从文件名{filename}提取年份，跳过")
                    continue
                
                # 加载网络
                if network_file.suffix == '.graphml':
                    G = nx.read_graphml(network_file)
                elif network_file.suffix == '.gexf':
                    G = nx.read_gexf(network_file)
                else:
                    continue
                
                if G.number_of_nodes() > 0:
                    networks[year] = G
                    logger.info(f"   ✅ {year}: {G.number_of_nodes()}节点, {G.number_of_edges()}边")
                
            except Exception as e:
                logger.warning(f"   ⚠️ 加载{network_file}失败: {e}")
                continue
        
        if not networks:
            raise ValueError("未成功加载任何网络数据")
        
        logger.info(f"✅ 成功加载{len(networks)}个年度网络 ({min(networks.keys())}-{max(networks.keys())})")
        return networks
    
    def calculate_us_export_influence(self, G: nx.Graph) -> Dict[str, float]:
        """
        计算美国的出口锁定影响力
        
        以美国为种子节点，计算美国对其他国家的网络影响力
        
        Args:
            G: 网络图
            
        Returns:
            Dict[str, float]: 国家到影响力分数的映射
        """
        if 'USA' not in G.nodes():
            logger.warning("   ⚠️ 网络中未找到USA节点")
            return {}
        
        # 以美国为唯一种子节点
        personalization = {node: 1.0 if node == 'USA' else 0.0 for node in G.nodes()}
        
        try:
            # 计算个性化PageRank
            pagerank_scores = nx.pagerank(
                G, 
                personalization=personalization,
                max_iter=1000,
                tol=1e-6
            )
            
            return pagerank_scores
            
        except Exception as e:
            logger.warning(f"   ⚠️ 计算美国出口影响力失败: {e}")
            return {}
    
    def calculate_influence_on_us(self, G: nx.Graph) -> Dict[str, float]:
        """
        计算其他国家对美国的进口锁定影响力
        
        遍历每个国家作为种子节点，计算其对美国的网络影响力
        
        Args:
            G: 网络图
            
        Returns:
            Dict[str, float]: 国家到对美影响力分数的映射
        """
        if 'USA' not in G.nodes():
            logger.warning("   ⚠️ 网络中未找到USA节点")
            return {}
        
        influence_on_us = {}
        non_us_nodes = [node for node in G.nodes() if node != 'USA']
        
        for country in non_us_nodes:
            try:
                # 以当前国家为唯一种子节点
                personalization = {node: 1.0 if node == country else 0.0 for node in G.nodes()}
                
                # 计算个性化PageRank
                pagerank_scores = nx.pagerank(
                    G,
                    personalization=personalization,
                    max_iter=1000,
                    tol=1e-6
                )
                
                # 提取该国家对美国的影响力分数
                influence_on_us[country] = pagerank_scores.get('USA', 0.0)
                
            except Exception as e:
                logger.warning(f"   ⚠️ 计算{country}对美国影响力失败: {e}")
                influence_on_us[country] = 0.0
        
        return influence_on_us
    
    def calculate_annual_personalized_pagerank(self, year: int, G: nx.Graph) -> pd.DataFrame:
        """
        计算单年度的个性化PageRank指标
        
        Args:
            year: 年份
            G: 网络图
            
        Returns:
            pd.DataFrame: 包含该年所有国家个性化PageRank分数的DataFrame
        """
        logger.info(f"📊 计算{year}年个性化PageRank...")
        
        # 1. 计算美国出口影响力
        logger.info(f"   计算美国出口影响力...")
        us_export_influence = self.calculate_us_export_influence(G)
        
        # 2. 计算各国对美国进口影响力  
        logger.info(f"   计算各国对美进口影响力...")
        influence_on_us = self.calculate_influence_on_us(G)
        
        # 3. 构建结果DataFrame
        results = []
        all_countries = set(us_export_influence.keys()) | set(influence_on_us.keys())
        
        for country in all_countries:
            if country == 'USA':  # 美国自己的数据特殊处理
                results.append({
                    'year': year,
                    'country_name': country,
                    'ppr_us_export_influence': us_export_influence.get(country, 0.0),
                    'ppr_influence_on_us': 0.0  # 美国对自己的影响力设为0
                })
            else:
                results.append({
                    'year': year,
                    'country_name': country,
                    'ppr_us_export_influence': us_export_influence.get(country, 0.0),
                    'ppr_influence_on_us': influence_on_us.get(country, 0.0)
                })
        
        year_df = pd.DataFrame(results)
        logger.info(f"   ✅ {year}年完成: {len(year_df)}个国家")
        
        return year_df
    
    def calculate_all_years(self, networks: Dict[int, nx.Graph]) -> pd.DataFrame:
        """
        计算所有年份的个性化PageRank
        
        Args:
            networks: 年度网络字典
            
        Returns:
            pd.DataFrame: 包含所有年份所有国家的完整面板数据
        """
        logger.info(f"🚀 开始计算{len(networks)}个年份的个性化PageRank...")
        
        all_results = []
        
        for year in sorted(networks.keys()):
            G = networks[year]
            try:
                year_results = self.calculate_annual_personalized_pagerank(year, G)
                all_results.append(year_results)
            except Exception as e:
                logger.error(f"   ❌ {year}年计算失败: {e}")
                continue
        
        if not all_results:
            raise ValueError("所有年份的计算都失败了")
        
        # 合并所有年份数据
        complete_df = pd.concat(all_results, ignore_index=True)
        
        logger.info(f"✅ 个性化PageRank计算完成: {len(complete_df)}条记录，覆盖{len(complete_df['year'].unique())}年")
        
        return complete_df
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        生成摘要统计
        
        Args:
            df: 个性化PageRank结果DataFrame
            
        Returns:
            Dict: 摘要统计字典
        """
        stats = {
            'total_records': len(df),
            'years_covered': sorted(df['year'].unique().tolist()),
            'countries_count': df['country_name'].nunique(),
            'year_range': f"{df['year'].min()}-{df['year'].max()}",
            
            # 美国出口影响力统计
            'us_export_influence': {
                'mean': df['ppr_us_export_influence'].mean(),
                'std': df['ppr_us_export_influence'].std(),
                'max': df['ppr_us_export_influence'].max(),
                'min': df['ppr_us_export_influence'].min()
            },
            
            # 对美进口影响力统计
            'influence_on_us': {
                'mean': df['ppr_influence_on_us'].mean(),
                'std': df['ppr_influence_on_us'].std(), 
                'max': df['ppr_influence_on_us'].max(),
                'min': df['ppr_influence_on_us'].min()
            }
        }
        
        # 美国出口影响力最高的5个国家（最新年份）
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        top_us_export_targets = latest_data.nlargest(5, 'ppr_us_export_influence')[
            ['country_name', 'ppr_us_export_influence']
        ].to_dict('records')
        
        top_influence_on_us = latest_data.nlargest(5, 'ppr_influence_on_us')[
            ['country_name', 'ppr_influence_on_us']  
        ].to_dict('records')
        
        stats['top_rankings'] = {
            'latest_year': latest_year,
            'top_us_export_influence': top_us_export_targets,
            'top_influence_on_us': top_influence_on_us
        }
        
        return stats
    
    def save_results(self, df: pd.DataFrame) -> Tuple[Path, Path]:
        """
        保存计算结果
        
        Args:
            df: 结果DataFrame
            
        Returns:
            Tuple[Path, Path]: CSV文件路径和摘要JSON文件路径
        """
        # 保存主要数据文件
        csv_path = self.output_dir / "personalized_pagerank_panel.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 主数据文件已保存: {csv_path}")
        
        # 生成并保存摘要统计
        stats = self.generate_summary_stats(df)
        
        import json
        json_path = self.output_dir / "personalized_pagerank_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📊 摘要统计已保存: {json_path}")
        
        return csv_path, json_path
    
    def run_full_calculation(self) -> Tuple[pd.DataFrame, Path, Path]:
        """
        运行完整的个性化PageRank计算流程
        
        Returns:
            Tuple[pd.DataFrame, Path, Path]: 结果DataFrame、CSV路径、JSON路径
        """
        logger.info("=" * 60)
        logger.info("🌟 个性化PageRank计算系统启动")
        logger.info("=" * 60)
        
        try:
            # 1. 加载网络数据
            networks = self.load_annual_networks()
            
            # 2. 计算个性化PageRank
            results_df = self.calculate_all_years(networks)
            
            # 3. 保存结果
            csv_path, json_path = self.save_results(results_df)
            
            # 4. 输出完成信息
            logger.info("=" * 60)
            logger.info("🎉 个性化PageRank计算完成!")
            logger.info("=" * 60)
            logger.info(f"📊 总记录数: {len(results_df):,}")
            logger.info(f"📅 覆盖年份: {results_df['year'].min()}-{results_df['year'].max()}")
            logger.info(f"🌍 覆盖国家: {results_df['country_name'].nunique()}")
            logger.info(f"📁 输出文件: {csv_path.name}, {json_path.name}")
            
            return results_df, csv_path, json_path
            
        except Exception as e:
            logger.error(f"❌ 计算过程失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="个性化PageRank计算系统 v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python personalized_pagerank.py                                              # 使用默认路径
  python personalized_pagerank.py --networks-dir ../02_net_analysis/outputs/networks  # 指定网络数据目录
  python personalized_pagerank.py --output-dir ./outputs                       # 指定输出目录
        """
    )
    
    # 设置默认路径
    current_dir = Path(__file__).parent
    default_networks_dir = current_dir.parent / "02_net_analysis" / "outputs" / "networks"
    default_output_dir = current_dir / "outputs"
    
    parser.add_argument(
        '--networks-dir', 
        type=str, 
        default=str(default_networks_dir),
        help=f'网络数据目录 (默认: {default_networks_dir})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str, 
        default=str(default_output_dir),
        help=f'输出目录 (默认: {default_output_dir})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建计算器并执行
        calculator = PersonalizedPageRankCalculator(
            networks_dir=Path(args.networks_dir),
            output_dir=Path(args.output_dir)
        )
        
        results_df, csv_path, json_path = calculator.run_full_calculation()
        
        print(f"\n✅ 个性化PageRank计算成功完成!")
        print(f"📊 结果文件: {csv_path}")
        print(f"📈 摘要文件: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)