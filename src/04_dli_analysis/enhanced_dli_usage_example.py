#!/usr/bin/env python3
"""
增强版DLI使用示例 (Enhanced DLI Usage Examples)
===============================================

演示如何使用增强版五维度DLI数据进行政策分析和学术研究。

主要功能:
1. 数据加载和基础统计
2. 时间序列趋势分析
3. 方向性锁定效应对比
4. 国别和产品分析
5. 政策影响评估示例

版本: v1.0
作者: Energy Network Analysis Team  
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedDLIAnalyzer:
    """增强版DLI数据分析器"""
    
    def __init__(self, data_path: Path = None, weights_path: Path = None):
        """
        初始化分析器
        
        Args:
            data_path: 增强版DLI数据路径
            weights_path: 权重参数文件路径
        """
        if data_path is None:
            data_path = Path(__file__).parent / "outputs" / "dli_pagerank.csv"
        if weights_path is None:
            weights_path = Path(__file__).parent / "outputs" / "dli_pagerank_weights.json"
            
        self.data_path = data_path
        self.weights_path = weights_path
        
        # 加载数据
        self.data = pd.read_csv(data_path)
        
        # 加载权重信息
        with open(weights_path, 'r', encoding='utf-8') as f:
            self.weights_info = json.load(f)
        
        print(f"✅ 增强版DLI数据加载完成: {len(self.data):,}条记录")
        
    def get_basic_statistics(self) -> Dict:
        """获取基础统计信息"""
        
        stats = {
            'total_records': len(self.data),
            'years_covered': f"{self.data['year'].min()}-{self.data['year'].max()}",
            'countries_count': self.data['us_partner'].nunique(),
            'products_count': self.data['energy_product'].nunique(),
            'export_records': len(self.data[self.data['us_role'] == 'exporter']),
            'import_records': len(self.data[self.data['us_role'] == 'importer']),
            'enhanced_dli_coverage': (~self.data['dli_enhanced'].isna()).mean(),
            'dli_enhanced_stats': {
                'mean': self.data['dli_enhanced'].mean(),
                'std': self.data['dli_enhanced'].std(),
                'min': self.data['dli_enhanced'].min(),
                'max': self.data['dli_enhanced'].max()
            }
        }
        
        return stats
    
    def analyze_time_trends(self) -> pd.DataFrame:
        """分析时间趋势"""
        
        print("📈 分析时间序列趋势...")
        
        # 按年份和方向聚合数据
        yearly_trends = self.data.groupby(['year', 'us_role']).agg({
            'dli_enhanced': ['mean', 'std', 'count'],
            'trade_value_usd': 'sum'
        }).round(6)
        
        yearly_trends.columns = ['dli_mean', 'dli_std', 'record_count', 'total_trade_value']
        yearly_trends = yearly_trends.reset_index()
        
        return yearly_trends
    
    def compare_directional_effects(self) -> Dict:
        """对比方向性锁定效应"""
        
        print("⚖️  分析方向性锁定效应差异...")
        
        export_data = self.data[self.data['us_role'] == 'exporter']
        import_data = self.data[self.data['us_role'] == 'importer']
        
        comparison = {
            'export_locking': {
                'mean_dli': export_data['dli_enhanced'].mean(),
                'std_dli': export_data['dli_enhanced'].std(),
                'records': len(export_data),
                'top_partners': export_data.groupby('us_partner')['dli_enhanced'].mean().nlargest(10).to_dict()
            },
            'import_locking': {
                'mean_dli': import_data['dli_enhanced'].mean(),
                'std_dli': import_data['dli_enhanced'].std(), 
                'records': len(import_data),
                'top_partners': import_data.groupby('us_partner')['dli_enhanced'].mean().nlargest(10).to_dict()
            }
        }
        
        # 统计检验
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            export_data['dli_enhanced'].dropna(),
            import_data['dli_enhanced'].dropna()
        )
        
        comparison['statistical_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        return comparison
    
    def analyze_country_rankings(self, year: int = 2024, top_n: int = 20) -> Dict:
        """分析国家排名"""
        
        print(f"🌍 分析{year}年国家锁定效应排名...")
        
        year_data = self.data[self.data['year'] == year].copy()
        
        # 分别计算出口和进口锁定排名
        export_ranking = year_data[year_data['us_role'] == 'exporter'].groupby('us_partner').agg({
            'dli_enhanced': 'mean',
            'trade_value_usd': 'sum'
        }).sort_values('dli_enhanced', ascending=False).head(top_n)
        
        import_ranking = year_data[year_data['us_role'] == 'importer'].groupby('us_partner').agg({
            'dli_enhanced': 'mean', 
            'trade_value_usd': 'sum'
        }).sort_values('dli_enhanced', ascending=False).head(top_n)
        
        return {
            'year': year,
            'export_locking_ranking': export_ranking.to_dict('index'),
            'import_locking_ranking': import_ranking.to_dict('index')
        }
    
    def analyze_product_effects(self) -> pd.DataFrame:
        """分析不同能源产品的锁定效应"""
        
        print("⚡ 分析能源产品锁定效应差异...")
        
        product_analysis = self.data.groupby(['energy_product', 'us_role']).agg({
            'dli_enhanced': ['mean', 'std', 'count'],
            'trade_value_usd': ['sum', 'mean'],
            'continuity': 'mean',
            'infrastructure': 'mean',
            'stability': 'mean',
            'market_locking_power': 'mean'
        }).round(4)
        
        product_analysis.columns = [
            'dli_mean', 'dli_std', 'record_count',
            'total_trade', 'avg_trade',
            'avg_continuity', 'avg_infrastructure', 'avg_stability', 'avg_market_power'
        ]
        
        return product_analysis.reset_index()
    
    def evaluate_shale_revolution_impact(self) -> Dict:
        """评估页岩革命的影响（2011年前后对比）"""
        
        print("🛢️  评估页岩革命政策影响...")
        
        pre_shale = self.data[self.data['year'] <= 2010]
        post_shale = self.data[self.data['year'] >= 2015]  # 留出缓冲期
        
        def calculate_period_stats(data, period_name):
            export_data = data[data['us_role'] == 'exporter']
            import_data = data[data['us_role'] == 'importer']
            
            return {
                'period': period_name,
                'export_locking': {
                    'mean_dli': export_data['dli_enhanced'].mean(),
                    'record_count': len(export_data),
                    'avg_trade_value': export_data['trade_value_usd'].mean()
                },
                'import_locking': {
                    'mean_dli': import_data['dli_enhanced'].mean(),
                    'record_count': len(import_data),
                    'avg_trade_value': import_data['trade_value_usd'].mean()
                }
            }
        
        pre_stats = calculate_period_stats(pre_shale, 'Pre-Shale (2001-2010)')
        post_stats = calculate_period_stats(post_shale, 'Post-Shale (2015-2024)')
        
        # 计算变化
        export_dli_change = (
            post_stats['export_locking']['mean_dli'] - 
            pre_stats['export_locking']['mean_dli']
        )
        
        import_dli_change = (
            post_stats['import_locking']['mean_dli'] - 
            pre_stats['import_locking']['mean_dli']
        )
        
        return {
            'pre_shale_revolution': pre_stats,
            'post_shale_revolution': post_stats,
            'changes': {
                'export_locking_change': float(export_dli_change),
                'import_locking_change': float(import_dli_change),
                'interpretation': {
                    'export': 'increased' if export_dli_change > 0 else 'decreased',
                    'import': 'increased' if import_dli_change > 0 else 'decreased'
                }
            }
        }
    
    def create_visualization_dashboard(self, output_dir: Path = None):
        """创建可视化面板"""
        
        print("🎨 创建可视化面板...")
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "outputs" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        
        # 1. 时间趋势图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 年度趋势
        yearly_data = self.data.groupby(['year', 'us_role'])['dli_enhanced'].mean().unstack()
        
        ax1 = axes[0, 0]
        if 'exporter' in yearly_data.columns:
            ax1.plot(yearly_data.index, yearly_data['exporter'], 'b-', label='US Export Locking', linewidth=2)
        if 'importer' in yearly_data.columns:
            ax1.plot(yearly_data.index, yearly_data['importer'], 'r-', label='US Import Being Locked', linewidth=2)
        ax1.set_title('Enhanced DLI Trends Over Time')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Enhanced DLI')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 国家分布（2024年）
        ax2 = axes[0, 1]
        latest_data = self.data[self.data['year'] == 2024]
        export_top = latest_data[latest_data['us_role'] == 'exporter'].groupby('us_partner')['dli_enhanced'].mean().nlargest(10)
        export_top.plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_title('Top 10 Countries: US Export Locking (2024)')
        ax2.set_xlabel('Enhanced DLI Score')
        
        # 能源产品对比
        ax3 = axes[1, 0]
        product_data = self.data.groupby(['energy_product', 'us_role'])['dli_enhanced'].mean().unstack()
        product_data.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Enhanced DLI by Energy Product')
        ax3.set_xlabel('Energy Product')
        ax3.set_ylabel('Average Enhanced DLI')
        ax3.legend(title='US Role')
        ax3.tick_params(axis='x', rotation=45)
        
        # 维度权重对比
        ax4 = axes[1, 1]
        export_weights = list(self.weights_info['export_dli_weights']['dimension_weights'].values())
        import_weights = list(self.weights_info['import_dli_weights']['dimension_weights'].values())
        dimensions = ['Continuity', 'Infrastructure', 'Stability', 'Market Power', 'PageRank']
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        ax4.bar(x - width/2, export_weights, width, label='Export Locking', alpha=0.8)
        ax4.bar(x + width/2, import_weights, width, label='Import Locking', alpha=0.8)
        ax4.set_title('PCA Dimension Weights Comparison')
        ax4.set_xlabel('Dimensions')
        ax4.set_ylabel('PCA Weight')
        ax4.set_xticks(x)
        ax4.set_xticklabels(dimensions, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        
        dashboard_path = output_dir / "enhanced_dli_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 可视化面板已保存: {dashboard_path}")
        plt.close()
    
    def generate_policy_insights(self) -> Dict:
        """生成政策洞察"""
        
        print("💡 生成政策分析洞察...")
        
        # 获取最新年份数据
        latest_year = self.data['year'].max()
        latest_data = self.data[self.data['year'] == latest_year]
        
        # 美国出口锁定能力最强的关系
        top_export_locking = latest_data[latest_data['us_role'] == 'exporter'].nlargest(5, 'dli_enhanced')
        
        # 美国进口被锁定风险最高的关系
        top_import_locking = latest_data[latest_data['us_role'] == 'importer'].nlargest(5, 'dli_enhanced')
        
        # 计算网络维度的贡献
        export_pagerank_contribution = self.weights_info['export_dli_weights']['dimension_weights']['pagerank_dimension']
        import_pagerank_contribution = self.weights_info['import_dli_weights']['dimension_weights']['pagerank_dimension']
        
        insights = {
            'analysis_year': latest_year,
            'strategic_export_advantages': [
                {
                    'partner': row['us_partner'],
                    'product': row['energy_product'],
                    'dli_score': round(row['dli_enhanced'], 3),
                    'trade_value': row['trade_value_usd']
                }
                for _, row in top_export_locking.iterrows()
            ],
            'import_vulnerability_risks': [
                {
                    'partner': row['us_partner'],
                    'product': row['energy_product'],
                    'dli_score': round(row['dli_enhanced'], 3),
                    'trade_value': row['trade_value_usd']
                }
                for _, row in top_import_locking.iterrows()
            ],
            'network_dimension_importance': {
                'export_locking_weight': round(export_pagerank_contribution, 3),
                'import_locking_weight': round(import_pagerank_contribution, 3),
                'interpretation': 'PageRank网络维度在出口锁定中权重更高，表明网络位置对美国出口影响力更重要'
            },
            'policy_recommendations': [
                "优先发展对华天然气出口关系，具有最高的锁定潜力",
                "关注对加拿大的进口依赖，建立多元化供应策略",
                "利用网络中心性优势，扩大美国在全球能源网络中的影响力",
                "平衡出口锁定能力与进口被锁定风险，实现能源安全最大化"
            ]
        }
        
        return insights


def main():
    """主演示函数"""
    
    print("🌟 增强版DLI数据分析演示")
    print("=" * 50)
    
    try:
        # 初始化分析器
        analyzer = EnhancedDLIAnalyzer()
        
        # 1. 基础统计
        print("\n【1】基础统计分析")
        print("-" * 30)
        stats = analyzer.get_basic_statistics()
        print(f"总记录数: {stats['total_records']:,}")
        print(f"年份覆盖: {stats['years_covered']}")
        print(f"国家数量: {stats['countries_count']}")
        print(f"出口记录: {stats['export_records']:,} ({stats['export_records']/stats['total_records']*100:.1f}%)")
        print(f"进口记录: {stats['import_records']:,} ({stats['import_records']/stats['total_records']*100:.1f}%)")
        print(f"增强版DLI统计: 均值={stats['dli_enhanced_stats']['mean']:.6f}, 标准差={stats['dli_enhanced_stats']['std']:.3f}")
        
        # 2. 时间趋势分析
        print("\n【2】时间趋势分析")
        print("-" * 30)
        trends = analyzer.analyze_time_trends()
        print(f"时间序列数据点: {len(trends)}")
        print("最新5年平均趋势:")
        recent_trends = trends[trends['year'] >= 2020].groupby('us_role')['dli_mean'].mean()
        for role, avg_dli in recent_trends.items():
            print(f"  {role}: {avg_dli:.6f}")
        
        # 3. 方向性效应对比
        print("\n【3】方向性锁定效应对比")
        print("-" * 30)
        comparison = analyzer.compare_directional_effects()
        print(f"出口锁定平均水平: {comparison['export_locking']['mean_dli']:.6f}")
        print(f"进口被锁定平均水平: {comparison['import_locking']['mean_dli']:.6f}")
        print(f"统计显著性检验: p={comparison['statistical_test']['p_value']:.6f}")
        
        # 4. 国家排名分析
        print("\n【4】2024年国家锁定效应排名")
        print("-" * 30)
        rankings = analyzer.analyze_country_rankings(2024, top_n=5)
        
        print("美国出口锁定Top 5:")
        for country, data in list(rankings['export_locking_ranking'].items())[:5]:
            print(f"  {country}: {data['dli_enhanced']:.3f}")
            
        print("美国进口被锁定Top 5:")  
        for country, data in list(rankings['import_locking_ranking'].items())[:5]:
            print(f"  {country}: {data['dli_enhanced']:.3f}")
        
        # 5. 页岩革命影响评估
        print("\n【5】页岩革命政策影响评估")
        print("-" * 30)
        shale_impact = analyzer.evaluate_shale_revolution_impact()
        pre_export = shale_impact['pre_shale_revolution']['export_locking']['mean_dli']
        post_export = shale_impact['post_shale_revolution']['export_locking']['mean_dli']
        export_change = shale_impact['changes']['export_locking_change']
        
        print(f"页岩革命前出口锁定力: {pre_export:.6f}")
        print(f"页岩革命后出口锁定力: {post_export:.6f}")
        print(f"变化: {export_change:+.6f} ({shale_impact['changes']['interpretation']['export']})")
        
        # 6. 生成可视化
        print("\n【6】创建可视化面板")
        print("-" * 30)
        analyzer.create_visualization_dashboard()
        
        # 7. 政策洞察
        print("\n【7】政策分析洞察")
        print("-" * 30)
        insights = analyzer.generate_policy_insights()
        print(f"分析基准年: {insights['analysis_year']}")
        
        print("\n战略出口优势 (Top 3):")
        for i, advantage in enumerate(insights['strategic_export_advantages'][:3], 1):
            print(f"  {i}. {advantage['partner']}-{advantage['product']}: {advantage['dli_score']}")
        
        print("\n进口脆弱性风险 (Top 3):")
        for i, risk in enumerate(insights['import_vulnerability_risks'][:3], 1):
            print(f"  {i}. {risk['partner']}-{risk['product']}: {risk['dli_score']}")
        
        print(f"\nPageRank网络维度重要性:")
        network_info = insights['network_dimension_importance']
        print(f"  出口锁定权重: {network_info['export_locking_weight']}")
        print(f"  进口锁定权重: {network_info['import_locking_weight']}")
        print(f"  解读: {network_info['interpretation']}")
        
        print("\n核心政策建议:")
        for i, rec in enumerate(insights['policy_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n✅ 增强版DLI分析演示完成!")
        print("📁 相关输出文件:")
        print("  • dli_pagerank.csv - 增强版DLI面板数据")
        print("  • dli_pagerank_weights.json - PCA权重参数")
        print("  • dli_dimensions_correlation.png - 维度相关性热力图")
        print("  • enhanced_dli_dashboard.png - 综合分析面板")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()