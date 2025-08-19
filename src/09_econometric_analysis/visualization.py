#!/usr/bin/env python3
"""
可视化模块 (Visualization Module)
===============================

09_econometric_analysis 模块的可视化组件

作者：Energy Network Analysis Team
版本：v1.0 - 计量分析框架
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

# 可视化库导入 (条件导入以处理缺失依赖)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    import seaborn as sns
    HAS_MATPLOTLIB = True
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from .config import config
except ImportError:
    import config
    config = config.config

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    可视化引擎 - 专门处理空数据和失败结果的图表生成逻辑
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化可视化引擎
        
        Args:
            output_dir: 图表输出目录，如果为None则使用配置中的默认目录
        """
        self.config = config
        self.figures_dir = output_dir or self.config.output.FIGURE_PATHS['coefficient_comparison'].parent
        self.figures_dir.mkdir(exist_ok=True)
        
        logger.info(f"📊 可视化引擎初始化完成")
        logger.info(f"图表目录: {self.figures_dir}")
        
        if not HAS_MATPLOTLIB:
            logger.warning("⚠️ matplotlib/seaborn库不可用，将生成占位符图表")
        
        # 图表样式配置
        self.color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        self.figure_size = (12, 8)
        self.dpi = 300
    
    def plot_coefficient_comparison(self, model_results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        绘制系数对比图
        
        Args:
            model_results: 模型结果字典
            output_path: 输出路径，如果为None则使用默认路径
            
        Returns:
            图表文件路径
        """
        logger.info("📈 生成系数对比图...")
        
        if output_path is None:
            output_path = self.figures_dir / "coefficient_comparison.png"
        
        if not HAS_MATPLOTLIB:
            return self._create_placeholder_figure(output_path, "系数对比图", "等待数据和matplotlib库")
        
        try:
            # 提取系数数据
            coef_data = self._extract_coefficient_data(model_results)
            
            if coef_data.empty:
                return self._create_no_data_figure(output_path, "系数对比图", "没有可用的系数数据")
            
            # 创建图表
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # 绘制系数点图
            self._plot_coefficient_points(ax, coef_data)
            
            # 设置图表样式
            ax.set_title('Coefficient Comparison Across Models\n系数对比（跨模型）', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Variables 变量', fontsize=12)
            ax.set_ylabel('Coefficient Value 系数值', fontsize=12)
            
            # 添加零线
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            logger.info(f"✅ 系数对比图已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"生成系数对比图失败: {str(e)}")
            return self._create_error_figure(output_path, "系数对比图", f"生成失败: {str(e)}")
        
        return output_path
    
    def plot_diagnostic_plots(self, model_results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        绘制模型诊断图
        
        Args:
            model_results: 模型结果字典
            output_path: 输出路径
            
        Returns:
            图表文件路径
        """
        logger.info("🔧 生成模型诊断图...")
        
        if output_path is None:
            output_path = self.figures_dir / "diagnostic_plots.png"
        
        if not HAS_MATPLOTLIB:
            return self._create_placeholder_figure(output_path, "模型诊断图", "等待数据和matplotlib库")
        
        try:
            # 检查是否有可用的诊断数据
            diagnostic_available = self._check_diagnostic_data_availability(model_results)
            
            if not diagnostic_available:
                return self._create_no_data_figure(output_path, "模型诊断图", "没有可用的诊断统计数据")
            
            # 创建2x2子图布局
            fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
            fig.suptitle('Model Diagnostic Plots\n模型诊断图表', fontsize=16, fontweight='bold')
            
            # 绘制各种诊断图
            self._plot_model_fit_comparison(axes[0, 0], model_results)
            self._plot_sample_size_comparison(axes[0, 1], model_results)
            self._plot_significance_summary(axes[1, 0], model_results)
            self._plot_model_status_summary(axes[1, 1], model_results)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            logger.info(f"✅ 模型诊断图已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"生成模型诊断图失败: {str(e)}")
            return self._create_error_figure(output_path, "模型诊断图", f"生成失败: {str(e)}")
        
        return output_path
    
    def plot_impulse_response(self, model_results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        绘制脉冲响应图（局部投影结果）
        
        Args:
            model_results: 模型结果字典
            output_path: 输出路径
            
        Returns:
            图表文件路径
        """
        logger.info("⚡ 生成脉冲响应图...")
        
        if output_path is None:
            output_path = self.figures_dir / "impulse_response.png"
        
        if not HAS_MATPLOTLIB:
            return self._create_placeholder_figure(output_path, "脉冲响应图", "等待数据和matplotlib库")
        
        try:
            # 检查局部投影结果
            lp_model = model_results.get('models', {}).get('model_3_local_projection_validation')
            
            if not lp_model or lp_model.get('status') != 'success' or not lp_model.get('horizon_results'):
                return self._create_no_data_figure(output_path, "脉冲响应图", "没有可用的局部投影结果")
            
            # 提取脉冲响应数据
            impulse_data = self._extract_impulse_response_data(lp_model)
            
            if impulse_data.empty:
                return self._create_no_data_figure(output_path, "脉冲响应图", "无法提取脉冲响应数据")
            
            # 创建图表
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # 绘制脉冲响应函数
            self._plot_impulse_response_function(ax, impulse_data)
            
            # 设置图表样式
            ax.set_title('Impulse Response Functions\n脉冲响应函数（局部投影法）', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Horizon (periods) 期数', fontsize=12)
            ax.set_ylabel('Response 响应', fontsize=12)
            
            # 添加零线
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            logger.info(f"✅ 脉冲响应图已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"生成脉冲响应图失败: {str(e)}")
            return self._create_error_figure(output_path, "脉冲响应图", f"生成失败: {str(e)}")
        
        return output_path
    
    def plot_robustness_charts(self, model_results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        绘制稳健性检验图表
        
        Args:
            model_results: 模型结果字典
            output_path: 输出路径
            
        Returns:
            图表文件路径
        """
        logger.info("🛡️ 生成稳健性检验图表...")
        
        if output_path is None:
            output_path = self.figures_dir / "robustness_charts.png"
        
        if not HAS_MATPLOTLIB:
            return self._create_placeholder_figure(output_path, "稳健性检验图", "等待数据和matplotlib库")
        
        # 目前稳健性检验功能待实现，生成占位符
        return self._create_placeholder_figure(output_path, "稳健性检验图", "功能开发中，敬请期待")
    
    def _extract_coefficient_data(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """提取系数数据用于可视化"""
        coef_rows = []
        
        models_dict = model_results.get('models', {})
        
        for model_name, result in models_dict.items():
            if result.get('status') != 'success':
                continue
            
            coefficients = result.get('coefficients', {})
            std_errors = result.get('std_errors', {})
            p_values = result.get('p_values', {})
            
            # 关键变量
            key_vars = ['node_dli_us', 'ovi', 'ovi_lag1', 'us_prod_shock', 'us_prod_shock_x_ovi_lag1']
            
            for var in key_vars:
                if var in coefficients:
                    coef_rows.append({
                        'model': self._get_model_display_name(model_name),
                        'variable': var,
                        'coefficient': coefficients[var],
                        'std_error': std_errors.get(var, np.nan),
                        'p_value': p_values.get(var, np.nan),
                        'significant': p_values.get(var, 1) < 0.05
                    })
            
            # 处理局部投影的特殊情况
            if model_name == 'model_3_local_projection_validation' and 'horizon_results' in result:
                for horizon_key, horizon_result in result['horizon_results'].items():
                    horizon_coefs = horizon_result.get('coefficients', {})
                    horizon_ses = horizon_result.get('std_errors', {})
                    horizon_pvals = horizon_result.get('p_values', {})
                    
                    for var in ['us_prod_shock', 'us_prod_shock_x_ovi_lag1']:
                        if var in horizon_coefs:
                            coef_rows.append({
                                'model': f"LP-{horizon_key}",
                                'variable': var,
                                'coefficient': horizon_coefs[var],
                                'std_error': horizon_ses.get(var, np.nan),
                                'p_value': horizon_pvals.get(var, np.nan),
                                'significant': horizon_pvals.get(var, 1) < 0.05
                            })
        
        return pd.DataFrame(coef_rows)
    
    def _plot_coefficient_points(self, ax, coef_data: pd.DataFrame):
        """绘制系数点图"""
        if coef_data.empty:
            ax.text(0.5, 0.5, 'No coefficient data available\n没有可用的系数数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
        
        # 为每个变量分配颜色
        unique_vars = coef_data['variable'].unique()
        colors = dict(zip(unique_vars, self.color_palette[:len(unique_vars)]))
        
        # 绘制点图
        for i, (var, var_data) in enumerate(coef_data.groupby('variable')):
            x_positions = np.arange(len(var_data)) + i * 0.1
            
            # 绘制系数点
            for j, (_, row) in enumerate(var_data.iterrows()):
                color = colors[var]
                marker = 'o' if row['significant'] else 's'
                size = 100 if row['significant'] else 60
                alpha = 1.0 if row['significant'] else 0.6
                
                ax.scatter(j + i * 0.1, row['coefficient'], 
                          c=color, marker=marker, s=size, alpha=alpha, label=var if j == 0 else "")
                
                # 添加置信区间（如果有标准误）
                if not pd.isna(row['std_error']):
                    ci_lower = row['coefficient'] - 1.96 * row['std_error']
                    ci_upper = row['coefficient'] + 1.96 * row['std_error']
                    ax.plot([j + i * 0.1, j + i * 0.1], [ci_lower, ci_upper], 
                           color=color, alpha=0.5, linewidth=2)
        
        # 设置x轴标签
        ax.set_xticks(range(len(coef_data['model'].unique())))
        ax.set_xticklabels(coef_data['model'].unique(), rotation=45)
        
        # 添加图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _extract_impulse_response_data(self, lp_model: Dict[str, Any]) -> pd.DataFrame:
        """提取脉冲响应数据"""
        impulse_rows = []
        
        horizon_results = lp_model.get('horizon_results', {})
        
        for horizon_key, horizon_result in horizon_results.items():
            horizon = horizon_result.get('horizon', 0)
            coefficients = horizon_result.get('coefficients', {})
            std_errors = horizon_result.get('std_errors', {})
            
            # 提取交互项系数（这是脉冲响应的核心）
            if 'us_prod_shock_x_ovi_lag1' in coefficients:
                impulse_rows.append({
                    'horizon': horizon,
                    'response': coefficients['us_prod_shock_x_ovi_lag1'],
                    'std_error': std_errors.get('us_prod_shock_x_ovi_lag1', np.nan)
                })
        
        return pd.DataFrame(impulse_rows).sort_values('horizon')
    
    def _plot_impulse_response_function(self, ax, impulse_data: pd.DataFrame):
        """绘制脉冲响应函数"""
        if impulse_data.empty:
            ax.text(0.5, 0.5, 'No impulse response data available\n没有可用的脉冲响应数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
        
        horizons = impulse_data['horizon']
        responses = impulse_data['response']
        std_errors = impulse_data['std_error']
        
        # 绘制主响应线
        ax.plot(horizons, responses, 'o-', color=self.color_palette[0], 
               linewidth=2, markersize=8, label='Impulse Response')
        
        # 绘制置信区间
        if not std_errors.isna().all():
            ci_lower = responses - 1.96 * std_errors
            ci_upper = responses + 1.96 * std_errors
            ax.fill_between(horizons, ci_lower, ci_upper, alpha=0.3, color=self.color_palette[0])
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _check_diagnostic_data_availability(self, model_results: Dict[str, Any]) -> bool:
        """检查诊断数据是否可用"""
        models_dict = model_results.get('models', {})
        return any(result.get('status') == 'success' for result in models_dict.values())
    
    def _plot_model_fit_comparison(self, ax, model_results: Dict[str, Any]):
        """绘制模型拟合度对比"""
        models_dict = model_results.get('models', {})
        
        model_names = []
        r_squared_values = []
        
        for model_name, result in models_dict.items():
            if result.get('status') == 'success' and 'r_squared' in result:
                model_names.append(self._get_model_display_name(model_name))
                r_squared_values.append(result['r_squared'])
        
        if not model_names:
            ax.text(0.5, 0.5, 'No R² data available\n没有可用的R²数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Fit Comparison (R²)')
            return
        
        bars = ax.bar(model_names, r_squared_values, color=self.color_palette[:len(model_names)])
        ax.set_title('Model Fit Comparison (R²)\n模型拟合度对比')
        ax.set_ylabel('R²')
        ax.set_ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, r_squared_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_sample_size_comparison(self, ax, model_results: Dict[str, Any]):
        """绘制样本量对比"""
        models_dict = model_results.get('models', {})
        
        model_names = []
        sample_sizes = []
        
        for model_name, result in models_dict.items():
            if result.get('status') == 'success' and 'n_obs' in result:
                model_names.append(self._get_model_display_name(model_name))
                sample_sizes.append(result['n_obs'])
        
        if not model_names:
            ax.text(0.5, 0.5, 'No sample size data available\n没有可用的样本量数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Sample Size Comparison')
            return
        
        bars = ax.bar(model_names, sample_sizes, color=self.color_palette[1:len(model_names)+1])
        ax.set_title('Sample Size Comparison\n样本量对比')
        ax.set_ylabel('Number of Observations')
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, sample_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_sizes)*0.01, 
                   f'{value}', ha='center', va='bottom')
    
    def _plot_significance_summary(self, ax, model_results: Dict[str, Any]):
        """绘制显著性汇总"""
        coef_data = self._extract_coefficient_data(model_results)
        
        if coef_data.empty:
            ax.text(0.5, 0.5, 'No significance data available\n没有可用的显著性数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Significance Summary')
            return
        
        # 统计显著性结果
        sig_summary = coef_data.groupby('variable')['significant'].agg(['sum', 'count']).reset_index()
        sig_summary['sig_rate'] = sig_summary['sum'] / sig_summary['count']
        
        bars = ax.bar(sig_summary['variable'], sig_summary['sig_rate'], 
                     color=self.color_palette[:len(sig_summary)])
        ax.set_title('Significance Rate by Variable\n各变量显著性比例')
        ax.set_ylabel('Significance Rate')
        ax.set_ylim(0, 1)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_model_status_summary(self, ax, model_results: Dict[str, Any]):
        """绘制模型状态汇总"""
        overview = model_results.get('overview', {})
        
        if not overview:
            ax.text(0.5, 0.5, 'No status data available\n没有可用的状态数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Status Summary')
            return
        
        # 准备饼图数据
        completed = overview.get('completed_models', 0)
        failed = overview.get('failed_models', 0)
        
        if completed + failed == 0:
            ax.text(0.5, 0.5, 'No models run\n没有运行任何模型', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Status Summary')
            return
        
        sizes = [completed, failed]
        labels = ['Successful', 'Failed']
        colors = ['#6A994E', '#C73E1D']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        ax.set_title('Model Status Summary\n模型状态汇总')
    
    def _get_model_display_name(self, model_name: str) -> str:
        """获取模型显示名称"""
        model_config = self.config.analysis.RESEARCH_MODELS.get(model_name, {})
        return model_config.get('name', model_name).replace('模型', 'M').replace(':', '')
    
    def _create_placeholder_figure(self, output_path: Path, title: str, message: str) -> Path:
        """创建占位符图表"""
        if not HAS_MATPLOTLIB:
            # 如果没有matplotlib，创建一个简单的文本文件说明
            with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(f"图表占位符: {title}\n")
                f.write(f"原因: {message}\n")
                f.write(f"需要安装matplotlib和seaborn库才能生成图表\n")
            return output_path.with_suffix('.txt')
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # 创建占位符框
            rect = Rectangle((0.1, 0.3), 0.8, 0.4, linewidth=2, 
                           edgecolor=self.color_palette[0], facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            
            # 添加文本
            ax.text(0.5, 0.5, f"{title}\n{message}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, fontweight='bold')
            
            # 隐藏坐标轴
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            logger.info(f"✅ 占位符图表已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"创建占位符图表失败: {str(e)}")
        
        return output_path
    
    def _create_no_data_figure(self, output_path: Path, title: str, message: str) -> Path:
        """创建无数据图表"""
        return self._create_placeholder_figure(output_path, f"📊 {title}", f"⚠️ {message}")
    
    def _create_error_figure(self, output_path: Path, title: str, error_message: str) -> Path:
        """创建错误图表"""
        return self._create_placeholder_figure(output_path, f"❌ {title}", f"错误: {error_message}")
    
    def generate_all_visualizations(self, model_results: Dict[str, Any]) -> Dict[str, Path]:
        """
        生成所有可视化图表
        
        Args:
            model_results: 模型结果字典
            
        Returns:
            生成图表路径字典
        """
        logger.info("🎨 开始生成所有可视化图表...")
        
        generated_figures = {}
        
        try:
            # 1. 系数对比图
            generated_figures['coefficient_comparison'] = self.plot_coefficient_comparison(model_results)
        except Exception as e:
            logger.error(f"系数对比图生成失败: {str(e)}")
        
        try:
            # 2. 模型诊断图
            generated_figures['diagnostic_plots'] = self.plot_diagnostic_plots(model_results)
        except Exception as e:
            logger.error(f"模型诊断图生成失败: {str(e)}")
        
        try:
            # 3. 脉冲响应图
            generated_figures['impulse_response'] = self.plot_impulse_response(model_results)
        except Exception as e:
            logger.error(f"脉冲响应图生成失败: {str(e)}")
        
        try:
            # 4. 稳健性检验图
            generated_figures['robustness_charts'] = self.plot_robustness_charts(model_results)
        except Exception as e:
            logger.error(f"稳健性检验图生成失败: {str(e)}")
        
        logger.info(f"✅ 可视化生成完成，共 {len(generated_figures)} 个图表")
        
        return generated_figures


# 便捷函数
def generate_visualizations(model_results: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    生成可视化的便捷函数
    
    Args:
        model_results: 模型结果字典
        output_dir: 输出目录
        
    Returns:
        生成图表路径字典
    """
    visualizer = VisualizationEngine(output_dir)
    return visualizer.generate_all_visualizations(model_results)


if __name__ == "__main__":
    # 测试可视化功能
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🎨 09_econometric_analysis 可视化模块测试")
    print("=" * 50)
    
    # 创建测试用的空结果
    test_results = {
        'overview': {
            'total_models': 3,
            'completed_models': 0,
            'failed_models': 3,
            'data_available': False
        },
        'models': {
            'model_1_dli_vul_association': {
                'status': 'failed',
                'status_message': '数据不可用'
            },
            'model_2_ovi_dli_causality': {
                'status': 'failed',
                'status_message': '数据不可用'
            },
            'model_3_local_projection_validation': {
                'status': 'failed',
                'status_message': '数据不可用'
            }
        }
    }
    
    # 测试可视化生成
    visualizer = VisualizationEngine()
    figures = visualizer.generate_all_visualizations(test_results)
    
    print(f"\n📊 测试结果:")
    for figure_type, file_path in figures.items():
        print(f"  {figure_type}: {file_path}")
    
    print("\n🎉 可视化模块测试完成!")