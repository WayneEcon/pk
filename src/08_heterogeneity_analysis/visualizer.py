#!/usr/bin/env python3
"""
可视化模块 (Visualization Module)
==============================

本模块负责生成网络结构异质性分析的各类图表，包括：
1. 边际效应图 (Marginal Effect Plots)
2. 交互效应可视化
3. 回归结果汇总图表

作者：Energy Network Analysis Team
版本：v1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 可视化包
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
except ImportError:
    HAS_PLOTTING = False
    logging.warning("⚠️ matplotlib/seaborn未安装，无法生成图表")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeterogeneityVisualizer:
    """网络结构异质性可视化器"""
    
    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PLOTTING:
            logger.error("❌ 缺少绘图依赖包，请安装matplotlib和seaborn")
            return
            
        self.figures = {}
        logger.info(f"🎨 初始化可视化器，输出目录: {self.output_dir}")
    
    def plot_marginal_effects(self, analysis_results: Dict[str, Any], 
                            data: pd.DataFrame,
                            save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        绘制边际效应图
        
        Args:
            analysis_results: 分析结果字典
            data: 原始数据
            save_plots: 是否保存图片
            
        Returns:
            图形对象字典
        """
        if not HAS_PLOTTING:
            logger.error("❌ 无法生成图表：缺少绘图依赖")
            return {}
            
        logger.info("📈 生成边际效应图...")
        
        figures = {}
        
        for model_name, result in analysis_results.items():
            marginal_effects = result.get('marginal_effects')
            if not marginal_effects:
                continue
                
            for interaction_var, effects in marginal_effects.items():
                if not effects:
                    continue
                    
                # 创建边际效应图
                fig = self._create_marginal_effect_plot(
                    interaction_var, effects, model_name, data
                )
                
                if fig:
                    figures[f"{model_name}_{interaction_var}"] = fig
                    
                    if save_plots:
                        filename = f"marginal_effect_{model_name}_{interaction_var}.png"
                        filepath = self.output_dir / filename
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        logger.info(f"💾 保存图表: {filename}")
        
        logger.info(f"✅ 边际效应图生成完成，共 {len(figures)} 个图表")
        return figures
    
    def plot_interaction_heatmap(self, results_table: pd.DataFrame,
                               save_plot: bool = True) -> Optional[plt.Figure]:
        """
        绘制交互效应热力图
        
        Args:
            results_table: 结果汇总表
            save_plot: 是否保存图片
            
        Returns:
            图形对象
        """
        if not HAS_PLOTTING or len(results_table) == 0:
            return None
            
        logger.info("🔥 生成交互效应热力图...")
        
        try:
            # 准备数据
            pivot_data = self._prepare_heatmap_data(results_table)
            
            if pivot_data.empty:
                logger.warning("⚠️ 无数据可供绘制热力图")
                return None
            
            # 创建热力图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制系数热力图
            sns.heatmap(
                pivot_data, 
                annot=True, 
                cmap='RdBu_r',
                center=0,
                fmt='.3f',
                cbar_kws={'label': 'Interaction Coefficient'},
                ax=ax
            )
            
            ax.set_title('Network Heterogeneity: Interaction Effects Heatmap', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Network Characteristics', fontsize=12)
            ax.set_ylabel('DLI Variables', fontsize=12)
            
            plt.tight_layout()
            
            if save_plot:
                filepath = self.output_dir / "interaction_heatmap.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"💾 保存热力图: interaction_heatmap.png")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ 生成热力图失败: {str(e)}")
            return None
    
    def plot_significance_overview(self, results_table: pd.DataFrame,
                                 save_plot: bool = True) -> Optional[plt.Figure]:
        """
        绘制显著性概览图
        
        Args:
            results_table: 结果汇总表
            save_plot: 是否保存图片
            
        Returns:
            图形对象
        """
        if not HAS_PLOTTING or len(results_table) == 0:
            return None
            
        logger.info("📊 生成显著性概览图...")
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 显著性分布
            significance_counts = results_table['significant'].value_counts()
            ax1.pie(significance_counts.values, 
                   labels=['Non-significant', 'Significant'], 
                   autopct='%1.1f%%',
                   colors=['lightcoral', 'lightblue'])
            ax1.set_title('Significance Distribution', fontweight='bold')
            
            # 2. 系数分布
            ax2.hist(results_table['coefficient'], bins=20, alpha=0.7, color='skyblue')
            ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Interaction Coefficient')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Coefficient Distribution', fontweight='bold')
            
            # 3. P值分布
            ax3.hist(results_table['p_value'], bins=20, alpha=0.7, color='lightgreen')
            ax3.axvline(0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
            ax3.set_xlabel('P-value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('P-value Distribution', fontweight='bold')
            ax3.legend()
            
            # 4. 分析类型对比
            type_stats = results_table.groupby('analysis_type').agg({
                'significant': 'sum',
                'coefficient': 'mean'
            }).round(3)
            
            x_pos = np.arange(len(type_stats))
            ax4.bar(x_pos - 0.2, type_stats['significant'], 0.4, 
                   label='Significant Count', alpha=0.7)
            
            ax4_twin = ax4.twinx()
            ax4_twin.bar(x_pos + 0.2, type_stats['coefficient'], 0.4, 
                        label='Mean Coefficient', alpha=0.7, color='orange')
            
            ax4.set_xlabel('Analysis Type')
            ax4.set_ylabel('Significant Count', color='blue')
            ax4_twin.set_ylabel('Mean Coefficient', color='orange')
            ax4.set_title('Global vs Local Analysis', fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(type_stats.index)
            
            plt.suptitle('Network Heterogeneity Analysis Overview', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_plot:
                filepath = self.output_dir / "significance_overview.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"💾 保存概览图: significance_overview.png")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ 生成概览图失败: {str(e)}")
            return None
    
    def plot_regression_diagnostics(self, analysis_results: Dict[str, Any],
                                  save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        绘制回归诊断图
        
        Args:
            analysis_results: 分析结果
            save_plots: 是否保存图片
            
        Returns:
            诊断图字典
        """
        if not HAS_PLOTTING:
            return {}
            
        logger.info("🔍 生成回归诊断图...")
        
        figures = {}
        
        for model_name, result in analysis_results.items():
            model_obj = result.get('model_object')
            if not model_obj:
                continue
                
            try:
                fig = self._create_diagnostic_plots(model_obj, model_name)
                if fig:
                    figures[model_name] = fig
                    
                    if save_plots:
                        filename = f"diagnostics_{model_name}.png"
                        filepath = self.output_dir / filename
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        logger.info(f"💾 保存诊断图: {filename}")
                        
            except Exception as e:
                logger.warning(f"⚠️ 无法为模型 {model_name} 生成诊断图: {str(e)}")
        
        return figures
    
    def _create_marginal_effect_plot(self, interaction_var: str, effects: List[Dict],
                                   model_name: str, data: pd.DataFrame) -> Optional[plt.Figure]:
        """创建单个边际效应图"""
        
        try:
            # 解析变量名
            parts = interaction_var.split('_x_')
            if len(parts) != 2:
                return None
                
            dli_var, moderator_var = parts
            
            # 提取边际效应数据
            moderator_values = [e[f'{moderator_var}_value'] for e in effects]
            marginal_effects = [e['marginal_effect'] for e in effects]
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制边际效应线
            ax.plot(moderator_values, marginal_effects, 'o-', linewidth=2, markersize=8)
            
            # 添加零线
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 设置标签和标题
            ax.set_xlabel(f'{moderator_var} Values', fontsize=12)
            ax.set_ylabel(f'Marginal Effect of {dli_var}', fontsize=12)
            ax.set_title(f'Marginal Effect: {dli_var} conditional on {moderator_var}\n({model_name})', 
                        fontsize=14, fontweight='bold')
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 美化
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ 创建边际效应图失败 {interaction_var}: {str(e)}")
            return None
    
    def _prepare_heatmap_data(self, results_table: pd.DataFrame) -> pd.DataFrame:
        """准备热力图数据"""
        
        # 解析交互项名称
        results_table = results_table.copy()
        results_table[['dli_var', 'network_var']] = results_table['interaction_term'].str.split('_x_', expand=True)
        
        # 创建透视表
        pivot_data = results_table.pivot_table(
            values='coefficient',
            index='dli_var',
            columns='network_var',
            aggfunc='mean'
        )
        
        return pivot_data.fillna(0)
    
    def _create_diagnostic_plots(self, model, model_name: str) -> Optional[plt.Figure]:
        """创建回归诊断图"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 残差 vs 拟合值
            fitted_values = model.fittedvalues
            residuals = model.resid
            
            ax1.scatter(fitted_values, residuals, alpha=0.6)
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_xlabel('Fitted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Fitted')
            
            # 2. QQ图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Normal Q-Q Plot')
            
            # 3. 标准化残差
            standardized_residuals = residuals / np.sqrt(model.mse_resid)
            ax3.scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
            ax3.set_xlabel('Fitted Values')
            ax3.set_ylabel('√|Standardized Residuals|')
            ax3.set_title('Scale-Location')
            
            # 4. 杠杆值 vs 标准化残差
            if hasattr(model, 'get_influence'):
                influence = model.get_influence()
                leverage = influence.hat_matrix_diag
                ax4.scatter(leverage, standardized_residuals, alpha=0.6)
                ax4.set_xlabel('Leverage')
                ax4.set_ylabel('Standardized Residuals')
                ax4.set_title('Residuals vs Leverage')
            else:
                ax4.text(0.5, 0.5, 'Leverage plot not available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Residuals vs Leverage')
            
            plt.suptitle(f'Regression Diagnostics: {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ 创建诊断图失败: {str(e)}")
            return None
    
    def create_summary_report_figure(self, summary_stats: Dict[str, Any],
                                   save_plot: bool = True) -> Optional[plt.Figure]:
        """
        创建分析摘要报告图
        
        Args:
            summary_stats: 摘要统计数据
            save_plot: 是否保存图片
            
        Returns:
            图形对象
        """
        if not HAS_PLOTTING:
            return None
            
        logger.info("📋 生成分析摘要报告图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 关键统计数据
            stats_text = f"""
Network Structure Heterogeneity Analysis Summary

Total Interactions Tested: {summary_stats.get('total_interactions', 'N/A')}
Significant Interactions: {summary_stats.get('significant_interactions', 'N/A')}
Significance Rate: {summary_stats.get('significance_rate', 0):.1%}

Strongest Effect:
  Variable: {summary_stats.get('strongest_effect', {}).get('interaction', 'None')}
  Coefficient: {summary_stats.get('strongest_effect', {}).get('coefficient', 'N/A')}
  P-value: {summary_stats.get('strongest_effect', {}).get('p_value', 'N/A')}

Key Findings:
• DLI effects show heterogeneity across network structures
• {summary_stats.get('significance_rate', 0):.1%} of interactions are statistically significant
• Results suggest network topology moderates lock-in effects
            """
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.title('Network Structure Heterogeneity Analysis\nSummary Report', 
                     fontsize=16, fontweight='bold', pad=20)
            
            if save_plot:
                filepath = self.output_dir / "summary_report.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"💾 保存摘要报告: summary_report.png")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ 生成摘要报告图失败: {str(e)}")
            return None


def main():
    """测试可视化功能"""
    # 创建测试数据
    test_results_table = pd.DataFrame({
        'model': ['dli_x_density', 'dli_x_clustering', 'dli_x_centrality'],
        'interaction_term': ['dli_composite_x_global_density', 
                           'dli_composite_x_global_clustering',
                           'dli_composite_x_betweenness_centrality'],
        'coefficient': [0.15, -0.08, 0.22],
        'p_value': [0.03, 0.12, 0.01],
        'significant': [True, False, True],
        'n_obs': [100, 100, 100],
        'r_squared': [0.45, 0.32, 0.52],
        'analysis_type': ['Global', 'Global', 'Local']
    })
    
    test_summary = {
        'total_interactions': 3,
        'significant_interactions': 2,
        'significance_rate': 0.67,
        'strongest_effect': {
            'interaction': 'dli_composite_x_betweenness_centrality',
            'coefficient': 0.22,
            'p_value': 0.01
        }
    }
    
    # 测试可视化器
    visualizer = HeterogeneityVisualizer()
    
    # 生成图表
    heatmap = visualizer.plot_interaction_heatmap(test_results_table)
    overview = visualizer.plot_significance_overview(test_results_table)
    summary = visualizer.create_summary_report_figure(test_summary)
    
    print("✅ 可视化测试完成")


if __name__ == "__main__":
    main()