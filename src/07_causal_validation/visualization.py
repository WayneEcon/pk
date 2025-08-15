#!/usr/bin/env python3
"""
因果验证分析可视化模块 (Causal Validation Visualization Module)
=============================================================

本模块提供因果分析结果的专业级可视化功能，包括：
1. 韧性指标时间序列图
2. 因果关系诊断图表
3. 回归诊断图
4. 网络韧性分布图
5. DLI与韧性关系散点图

作者：Energy Network Analysis Team
版本：v1.0 (Academic Visualization Edition)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class CausalVisualization:
    """因果分析可视化类"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建图表子目录
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        logger.info(f"📊 初始化可视化模块，输出目录: {self.figures_dir}")
    
    def plot_resilience_time_series(self, resilience_data: pd.DataFrame, 
                                  countries: List[str] = None,
                                  save_path: str = None) -> str:
        """绘制韧性指标时间序列图"""
        
        if countries is None:
            countries = ['USA', 'CHN', 'RUS', 'SAU', 'DEU', 'JPN']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('网络韧性指标时间序列分析\nNetwork Resilience Indicators Time Series', 
                    fontsize=16, fontweight='bold')
        
        # 1. 综合韧性指标
        ax1 = axes[0, 0]
        for country in countries:
            country_data = resilience_data[resilience_data['country'] == country]
            if not country_data.empty:
                ax1.plot(country_data['year'], country_data['comprehensive_resilience'], 
                        marker='o', label=country, linewidth=2, markersize=4)
        
        ax1.set_title('综合韧性指标 (Comprehensive Resilience)', fontweight='bold')
        ax1.set_xlabel('年份 (Year)')
        ax1.set_ylabel('韧性得分 (Resilience Score)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 拓扑韧性指标
        ax2 = axes[0, 1]
        for country in countries:
            country_data = resilience_data[resilience_data['country'] == country]
            if not country_data.empty:
                ax2.plot(country_data['year'], country_data['topological_resilience_avg'], 
                        marker='s', label=country, linewidth=2, markersize=4)
        
        ax2.set_title('拓扑韧性指标 (Topological Resilience)', fontweight='bold')
        ax2.set_xlabel('年份 (Year)')
        ax2.set_ylabel('拓扑韧性 (Topological Resilience)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. 供应吸收率
        ax3 = axes[1, 0]
        for country in countries:
            country_data = resilience_data[resilience_data['country'] == country]
            if not country_data.empty:
                ax3.plot(country_data['year'], country_data['supply_absorption_rate'], 
                        marker='^', label=country, linewidth=2, markersize=4)
        
        ax3.set_title('供应吸收率 (Supply Absorption Rate)', fontweight='bold')
        ax3.set_xlabel('年份 (Year)')
        ax3.set_ylabel('吸收率 (Absorption Rate)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. 网络位置稳定性
        ax4 = axes[1, 1]
        for country in countries:
            country_data = resilience_data[resilience_data['country'] == country]
            if not country_data.empty:
                ax4.plot(country_data['year'], country_data['network_position_stability'], 
                        marker='d', label=country, linewidth=2, markersize=4)
        
        ax4.set_title('网络位置稳定性 (Network Position Stability)', fontweight='bold')
        ax4.set_xlabel('年份 (Year)')
        ax4.set_ylabel('稳定性 (Stability)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figures_dir / "resilience_time_series.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ 韧性时间序列图已保存: {save_path}")
        return str(save_path)
    
    def plot_dli_resilience_scatter(self, merged_data: pd.DataFrame,
                                   save_path: str = None) -> str:
        """绘制DLI与韧性关系散点图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DLI与网络韧性关系分析\nDLI vs Network Resilience Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. DLI vs 综合韧性
        ax1 = axes[0, 0]
        countries = merged_data['country'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
        
        for i, country in enumerate(countries):
            country_data = merged_data[merged_data['country'] == country]
            ax1.scatter(country_data['dli_score'], country_data['comprehensive_resilience'],
                       alpha=0.7, s=50, label=country, color=colors[i])
        
        # 添加回归线
        z = np.polyfit(merged_data['dli_score'], merged_data['comprehensive_resilience'], 1)
        p = np.poly1d(z)
        x_reg = np.linspace(merged_data['dli_score'].min(), merged_data['dli_score'].max(), 100)
        ax1.plot(x_reg, p(x_reg), "r--", alpha=0.8, linewidth=2)
        
        # 计算相关系数
        corr = merged_data[['dli_score', 'comprehensive_resilience']].corr().iloc[0, 1]
        ax1.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('DLI得分 (DLI Score)')
        ax1.set_ylabel('综合韧性 (Comprehensive Resilience)')
        ax1.set_title('DLI vs 综合韧性', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. DLI vs 拓扑韧性
        ax2 = axes[0, 1]
        for i, country in enumerate(countries):
            country_data = merged_data[merged_data['country'] == country]
            ax2.scatter(country_data['dli_score'], country_data['topological_resilience_avg'],
                       alpha=0.7, s=50, label=country, color=colors[i])
        
        z2 = np.polyfit(merged_data['dli_score'], merged_data['topological_resilience_avg'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_reg, p2(x_reg), "r--", alpha=0.8, linewidth=2)
        
        corr2 = merged_data[['dli_score', 'topological_resilience_avg']].corr().iloc[0, 1]
        ax2.text(0.05, 0.95, f'相关系数: {corr2:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('DLI得分 (DLI Score)')
        ax2.set_ylabel('拓扑韧性 (Topological Resilience)')
        ax2.set_title('DLI vs 拓扑韧性', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. DLI vs 供应吸收率
        ax3 = axes[1, 0]
        for i, country in enumerate(countries):
            country_data = merged_data[merged_data['country'] == country]
            ax3.scatter(country_data['dli_score'], country_data['supply_absorption_rate'],
                       alpha=0.7, s=50, label=country, color=colors[i])
        
        z3 = np.polyfit(merged_data['dli_score'], merged_data['supply_absorption_rate'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(x_reg, p3(x_reg), "r--", alpha=0.8, linewidth=2)
        
        corr3 = merged_data[['dli_score', 'supply_absorption_rate']].corr().iloc[0, 1]
        ax3.text(0.05, 0.95, f'相关系数: {corr3:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        ax3.set_xlabel('DLI得分 (DLI Score)')
        ax3.set_ylabel('供应吸收率 (Supply Absorption Rate)')
        ax3.set_title('DLI vs 供应吸收率', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 年份分布热力图
        ax4 = axes[1, 1]
        pivot_data = merged_data.pivot_table(values='comprehensive_resilience', 
                                           index='country', columns='year', 
                                           aggfunc='mean')
        
        sns.heatmap(pivot_data, ax=ax4, cmap='RdYlBu_r', cbar=True, 
                   fmt='.3f', square=False, linewidths=0.5)
        ax4.set_title('综合韧性年度热力图', fontweight='bold')
        ax4.set_xlabel('年份 (Year)')
        ax4.set_ylabel('国家 (Country)')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figures_dir / "dli_resilience_scatter.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ DLI-韧性散点图已保存: {save_path}")
        return str(save_path)
    
    def plot_regression_diagnostics(self, causal_results: Dict[str, Any],
                                   save_path: str = None) -> str:
        """绘制回归诊断图"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('回归模型诊断图表\nRegression Model Diagnostics', 
                    fontsize=16, fontweight='bold')
        
        # 提取回归结果
        try:
            fe_results = causal_results.get('fixed_effects_results', {})
            iv_results = causal_results.get('instrumental_variables_results', {})
            
            # 1. 系数比较图
            ax1 = axes[0, 0]
            models = []
            coefficients = []
            conf_intervals = []
            
            if 'coefficient' in fe_results:
                models.append('Fixed Effects')
                coefficients.append(fe_results['coefficient'])
                ci_lower = fe_results['coefficient'] - 1.96 * fe_results.get('std_error', 0)
                ci_upper = fe_results['coefficient'] + 1.96 * fe_results.get('std_error', 0)
                conf_intervals.append((ci_lower, ci_upper))
            
            if 'coefficient' in iv_results:
                models.append('IV (2SLS)')
                coefficients.append(iv_results['coefficient'])
                ci_lower = iv_results['coefficient'] - 1.96 * iv_results.get('std_error', 0)
                ci_upper = iv_results['coefficient'] + 1.96 * iv_results.get('std_error', 0)
                conf_intervals.append((ci_lower, ci_upper))
            
            if models:
                y_pos = np.arange(len(models))
                ax1.barh(y_pos, coefficients, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
                
                # 添加置信区间
                for i, (lower, upper) in enumerate(conf_intervals):
                    ax1.plot([lower, upper], [i, i], 'k-', linewidth=2)
                    ax1.plot([lower, lower], [i-0.1, i+0.1], 'k-', linewidth=2)
                    ax1.plot([upper, upper], [i-0.1, i+0.1], 'k-', linewidth=2)
                
                ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(models)
                ax1.set_xlabel('系数估计值 (Coefficient Estimate)')
                ax1.set_title('模型系数比较', fontweight='bold')
                ax1.grid(True, alpha=0.3)
            
            # 2. 显著性测试结果
            ax2 = axes[0, 1]
            if models:
                p_values = []
                if 'p_value' in fe_results:
                    p_values.append(fe_results['p_value'])
                if 'p_value' in iv_results:
                    p_values.append(iv_results['p_value'])
                
                colors = ['red' if p < 0.05 else 'gray' for p in p_values]
                bars = ax2.bar(models, p_values, color=colors, alpha=0.7)
                
                ax2.axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
                ax2.axhline(y=0.01, color='orange', linestyle='--', label='α=0.01')
                ax2.set_ylabel('P值 (P-value)')
                ax2.set_title('统计显著性测试', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 添加p值标签
                for bar, p_val in zip(bars, p_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{p_val:.4f}', ha='center', va='bottom')
            
            # 3. 模型拟合度比较
            ax3 = axes[0, 2]
            if models:
                r_squared_values = []
                if 'r_squared' in fe_results:
                    r_squared_values.append(fe_results['r_squared'])
                if 'r_squared' in iv_results:
                    r_squared_values.append(iv_results['r_squared'])
                
                if r_squared_values:
                    ax3.bar(models, r_squared_values, color=['#2ca02c', '#d62728'], alpha=0.7)
                    ax3.set_ylabel('R² 值')
                    ax3.set_title('模型拟合度', fontweight='bold')
                    ax3.set_ylim(0, 1)
                    ax3.grid(True, alpha=0.3)
                    
                    # 添加R²标签
                    for i, r2 in enumerate(r_squared_values):
                        ax3.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom')
            
            # 4. 诊断统计
            ax4 = axes[1, 0]
            diagnostics = causal_results.get('diagnostics', {})
            
            diag_names = []
            diag_values = []
            
            for key, value in diagnostics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    diag_names.append(key.replace('_', ' ').title())
                    diag_values.append(value)
            
            if diag_names:
                ax4.barh(range(len(diag_names)), diag_values, alpha=0.7)
                ax4.set_yticks(range(len(diag_names)))
                ax4.set_yticklabels(diag_names)
                ax4.set_xlabel('统计量值')
                ax4.set_title('诊断统计量', fontweight='bold')
                ax4.grid(True, alpha=0.3)
            
            # 5. 稳健性检验
            ax5 = axes[1, 1]
            robustness = causal_results.get('robustness_tests', {})
            
            if robustness:
                rob_methods = list(robustness.keys())
                rob_coeffs = [robustness[method].get('coefficient', 0) for method in rob_methods]
                
                ax5.scatter(range(len(rob_methods)), rob_coeffs, s=100, alpha=0.7)
                ax5.set_xticks(range(len(rob_methods)))
                ax5.set_xticklabels(rob_methods, rotation=45)
                ax5.set_ylabel('系数估计值')
                ax5.set_title('稳健性检验', fontweight='bold')
                ax5.grid(True, alpha=0.3)
                
                # 添加基准线
                if coefficients:
                    ax5.axhline(y=coefficients[0], color='red', linestyle='--', 
                               label=f'基准模型: {coefficients[0]:.3f}')
                    ax5.legend()
            
            # 6. 总体评估
            ax6 = axes[1, 2]
            overall = causal_results.get('overall_assessment', {})
            
            assessment_items = [
                ('统计显著性', overall.get('statistical_significance', False)),
                ('经济显著性', overall.get('economic_significance', False)),
                ('稳健性', overall.get('robustness', False)),
                ('工具变量有效性', overall.get('instrument_validity', False))
            ]
            
            colors = ['green' if item[1] else 'red' for item in assessment_items]
            values = [1 if item[1] else 0 for item in assessment_items]
            labels = [item[0] for item in assessment_items]
            
            wedges, texts, autotexts = ax6.pie(values, labels=labels, colors=colors, 
                                              autopct='%1.0f%%', startangle=90)
            ax6.set_title('总体评估', fontweight='bold')
            
        except Exception as e:
            logger.warning(f"绘制诊断图时出错: {e}")
            # 如果出错，显示占位图
            for ax in axes.flat:
                ax.text(0.5, 0.5, '数据不可用\nData Not Available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figures_dir / "regression_diagnostics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ 回归诊断图已保存: {save_path}")
        return str(save_path)
    
    def plot_network_resilience_distribution(self, resilience_data: pd.DataFrame,
                                           save_path: str = None) -> str:
        """绘制网络韧性分布图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('网络韧性分布分析\nNetwork Resilience Distribution Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. 综合韧性分布箱线图
        ax1 = axes[0, 0]
        countries = resilience_data['country'].unique()
        resilience_by_country = [resilience_data[resilience_data['country'] == country]['comprehensive_resilience'].values 
                               for country in countries]
        
        box_plot = ax1.boxplot(resilience_by_country, labels=countries, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('综合韧性分布 (按国家)', fontweight='bold')
        ax1.set_ylabel('综合韧性得分')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. 韧性指标相关性热力图
        ax2 = axes[0, 1]
        resilience_cols = [col for col in resilience_data.columns 
                          if 'resilience' in col or 'supply' in col or 'stability' in col]
        corr_matrix = resilience_data[resilience_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('韧性指标相关性', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        # 3. 年度韧性变化趋势
        ax3 = axes[1, 0]
        yearly_stats = resilience_data.groupby('year')['comprehensive_resilience'].agg(['mean', 'std'])
        
        ax3.fill_between(yearly_stats.index, 
                        yearly_stats['mean'] - yearly_stats['std'],
                        yearly_stats['mean'] + yearly_stats['std'],
                        alpha=0.3, label='±1σ 区间')
        ax3.plot(yearly_stats.index, yearly_stats['mean'], 'o-', 
                linewidth=2, markersize=6, label='平均值')
        
        ax3.set_title('年度韧性变化趋势', fontweight='bold')
        ax3.set_xlabel('年份')
        ax3.set_ylabel('综合韧性得分')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 韧性得分分布直方图
        ax4 = axes[1, 1]
        ax4.hist(resilience_data['comprehensive_resilience'], bins=20, 
                alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加统计信息
        mean_val = resilience_data['comprehensive_resilience'].mean()
        std_val = resilience_data['comprehensive_resilience'].std()
        ax4.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
        ax4.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
        ax4.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
        
        ax4.set_title('综合韧性得分分布', fontweight='bold')
        ax4.set_xlabel('综合韧性得分')
        ax4.set_ylabel('频次')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figures_dir / "resilience_distribution.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ 韧性分布图已保存: {save_path}")
        return str(save_path)
    
    def generate_visualization_summary(self, causal_results: Dict[str, Any],
                                     resilience_data: pd.DataFrame,
                                     dli_data: pd.DataFrame) -> Dict[str, str]:
        """生成完整的可视化报告"""
        
        logger.info("🎨 生成完整可视化报告...")
        
        # 合并数据用于散点图
        merged_data = pd.merge(resilience_data, dli_data, 
                             on=['year', 'country'], how='inner')
        
        visualization_files = {}
        
        try:
            # 1. 韧性时间序列图
            resilience_ts_path = self.plot_resilience_time_series(resilience_data)
            visualization_files['resilience_time_series'] = resilience_ts_path
            
            # 2. DLI-韧性关系图
            dli_scatter_path = self.plot_dli_resilience_scatter(merged_data)
            visualization_files['dli_resilience_scatter'] = dli_scatter_path
            
            # 3. 回归诊断图
            diagnostics_path = self.plot_regression_diagnostics(causal_results)
            visualization_files['regression_diagnostics'] = diagnostics_path
            
            # 4. 韧性分布图
            distribution_path = self.plot_network_resilience_distribution(resilience_data)
            visualization_files['resilience_distribution'] = distribution_path
            
            logger.info(f"✅ 可视化报告生成完成，共 {len(visualization_files)} 个图表")
            
        except Exception as e:
            logger.error(f"❌ 可视化生成过程中出错: {e}")
            
        return visualization_files

def create_visualizations(causal_results: Dict[str, Any],
                         resilience_data: pd.DataFrame,
                         dli_data: pd.DataFrame,
                         output_dir: str = "outputs") -> Dict[str, str]:
    """便捷函数：创建所有可视化图表"""
    
    visualizer = CausalVisualization(output_dir)
    return visualizer.generate_visualization_summary(
        causal_results, resilience_data, dli_data
    )