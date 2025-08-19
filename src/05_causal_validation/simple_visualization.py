#!/usr/bin/env python3
"""
精简可视化模块 (Simple Visualization Module)
==========================================

专注于因果分析的核心图表：
1. 韧性时间序列图
2. DLI与韧性关系散点图

版本：v2.1 (Simplified & Focused)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path

# 设置绘图样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 11

logger = logging.getLogger(__name__)

class SimpleCausalVisualization:
    """精简的因果分析可视化类"""
    
    def __init__(self, output_dir: Path):
        self.figures_dir = output_dir
        self.figures_dir.mkdir(exist_ok=True)
        logger.info(f"📊 初始化精简可视化模块: {self.figures_dir}")
    
    def create_all_visualizations(self, resilience_data: pd.DataFrame, 
                                 dli_data: pd.DataFrame, 
                                 causal_results: Dict[str, Any]):
        """生成所有核心可视化图表"""
        logger.info("🎨 生成可视化图表...")
        
        try:
            # 1. 韧性时间序列图
            self._plot_resilience_time_series(resilience_data)
            
            # 2. DLI与韧性关系散点图
            self._plot_dli_resilience_scatter(resilience_data, dli_data)
            
            logger.info("✅ 可视化图表生成完成")
            
        except Exception as e:
            logger.error(f"❌ 可视化生成失败: {e}")
    
    def _plot_resilience_time_series(self, resilience_data: pd.DataFrame):
        """绘制韧性指标时间序列图"""
        try:
            # 选择主要国家
            major_countries = ['USA', 'CHN', 'RUS', 'DEU', 'JPN']
            data = resilience_data[resilience_data['country'].isin(major_countries)]
            
            if data.empty:
                logger.warning("⚠️ 没有找到主要国家的数据，使用所有可用数据")
                data = resilience_data.head(50)  # 限制数据量
            
            # 关键韧性指标
            resilience_metrics = [
                'comprehensive_resilience',
                'topological_resilience_avg', 
                'supply_absorption_rate',
                'network_position_stability'
            ]
            
            # 确保指标存在
            available_metrics = [m for m in resilience_metrics if m in data.columns]
            
            if not available_metrics:
                logger.warning("⚠️ 未找到韧性指标，跳过时间序列图")
                return
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics[:4]):
                ax = axes[i]
                
                # 按国家绘制时间序列
                for country in data['country'].unique():
                    country_data = data[data['country'] == country].sort_values('year')
                    if len(country_data) > 1:
                        ax.plot(country_data['year'], country_data[metric], 
                               marker='o', label=country, linewidth=2, markersize=4)
                
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Resilience Score')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            output_file = self.figures_dir / "resilience_time_series.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ 韧性时间序列图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ 韧性时间序列图生成失败: {e}")
    
    def _plot_dli_resilience_scatter(self, resilience_data: pd.DataFrame, 
                                   dli_data: pd.DataFrame):
        """绘制DLI与韧性关系散点图"""
        try:
            # 合并数据
            merged_data = pd.merge(
                resilience_data, 
                dli_data, 
                on=['year', 'country'], 
                how='inner'
            )
            
            if merged_data.empty:
                logger.warning("⚠️ 无法合并DLI和韧性数据，跳过散点图")
                return
            
            # 创建散点图矩阵
            resilience_cols = [
                'comprehensive_resilience',
                'topological_resilience_avg',
                'supply_absorption_rate'
            ]
            
            # 确保列存在
            available_cols = [col for col in resilience_cols if col in merged_data.columns]
            
            if not available_cols or 'dli_score' not in merged_data.columns:
                logger.warning("⚠️ 缺少必要的数据列，跳过散点图")
                return
            
            # 创建子图
            n_cols = len(available_cols)
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
            
            if n_cols == 1:
                axes = [axes]
            
            for i, resilience_col in enumerate(available_cols):
                ax = axes[i]
                
                # 按国家着色的散点图
                countries = merged_data['country'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
                
                for country, color in zip(countries, colors):
                    country_data = merged_data[merged_data['country'] == country]
                    ax.scatter(country_data['dli_score'], country_data[resilience_col],
                             c=[color], label=country, alpha=0.7, s=50)
                
                # 添加趋势线
                if len(merged_data) > 5:
                    z = np.polyfit(merged_data['dli_score'], merged_data[resilience_col], 1)
                    p = np.poly1d(z)
                    ax.plot(merged_data['dli_score'].sort_values(), 
                           p(merged_data['dli_score'].sort_values()), 
                           "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel('DLI Score')
                ax.set_ylabel(resilience_col.replace('_', ' ').title())
                ax.set_title(f'DLI vs {resilience_col.replace("_", " ").title()}')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            output_file = self.figures_dir / "dli_resilience_scatter.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ DLI散点图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ DLI散点图生成失败: {e}")

# 为了兼容性，创建别名
CausalVisualization = SimpleCausalVisualization