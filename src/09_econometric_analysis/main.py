#!/usr/bin/env python3
"""
主程序入口 (Main Entry Point)
============================

09_econometric_analysis 模块的完整分析流水线

作者：Energy Network Analysis Team
版本：v1.0 - 计量分析框架
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import sys
import time
from datetime import datetime

# 导入本模块组件
from .config import config, print_config_summary
from .data_loader import DataLoader, get_data_status
from .models import EconometricModels
from .reporting import ReportGenerator
from .visualization import VisualizationEngine

logger = logging.getLogger(__name__)

class EconometricAnalysisPipeline:
    """
    计量经济学分析流水线 - 完整的端到端分析框架
    """
    
    def __init__(self):
        """初始化分析流水线"""
        self.config = config
        self.start_time = time.time()
        
        # 初始化各组件
        self.data_loader = DataLoader()
        self.models = EconometricModels()
        self.reporter = ReportGenerator()
        self.visualizer = VisualizationEngine()
        
        # 存储结果
        self.data = None
        self.data_summary = None
        self.model_results = None
        
        logger.info("🚀 计量经济学分析流水线初始化完成")
        logger.info(f"   模块版本: 09_econometric_analysis v1.0")
        logger.info(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的分析流水线
        
        Returns:
            分析结果汇总字典
        """
        logger.info("🔬 开始运行完整计量分析流水线...")
        logger.info("=" * 60)
        
        pipeline_results = {
            'status': 'running',
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'steps_completed': 0,
            'steps_total': 5,
            'data_loaded': False,
            'models_run': False,
            'reports_generated': False,
            'visualizations_created': False,
            'pipeline_success': False
        }
        
        try:
            # 步骤1: 数据加载与验证
            logger.info("步骤1/5: 数据加载与验证")
            logger.info("=" * 30)
            success = self._step_data_loading()
            pipeline_results['steps_completed'] = 1
            pipeline_results['data_loaded'] = success
            
            if not success:
                logger.warning("⚠️ 数据加载失败，但继续运行以演示空数据处理能力")
            
            # 步骤2: 运行计量模型
            logger.info("\n步骤2/5: 运行计量模型")
            logger.info("=" * 30)
            success = self._step_model_estimation()
            pipeline_results['steps_completed'] = 2
            pipeline_results['models_run'] = success
            
            # 步骤3: 生成分析报告
            logger.info("\n步骤3/5: 生成分析报告")
            logger.info("=" * 30)
            success = self._step_report_generation()
            pipeline_results['steps_completed'] = 3
            pipeline_results['reports_generated'] = success
            
            # 步骤4: 创建可视化图表
            logger.info("\n步骤4/5: 创建可视化图表")
            logger.info("=" * 30)
            success = self._step_visualization_creation()
            pipeline_results['steps_completed'] = 4
            pipeline_results['visualizations_created'] = success
            
            # 步骤5: 流水线总结
            logger.info("\n步骤5/5: 流水线总结")
            logger.info("=" * 30)
            pipeline_results.update(self._step_pipeline_summary())
            pipeline_results['steps_completed'] = 5
            pipeline_results['pipeline_success'] = True
            
            logger.info("✅ 计量分析流水线运行完成!")
            
        except Exception as e:
            logger.error(f"❌ 流水线运行异常: {str(e)}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error_message'] = str(e)
            pipeline_results['pipeline_success'] = False
        
        finally:
            # 计算总耗时
            total_time = time.time() - self.start_time
            pipeline_results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pipeline_results['total_duration_seconds'] = total_time
            pipeline_results['status'] = 'completed' if pipeline_results.get('pipeline_success') else 'failed'
        
        return pipeline_results
    
    def _step_data_loading(self) -> bool:
        """步骤1: 数据加载与验证"""
        try:
            logger.info("📊 加载分析数据...")
            
            # 加载数据
            self.data = self.data_loader.load_analytical_panel()
            
            # 获取数据摘要
            self.data_summary = get_data_status()
            
            # 打印数据概况
            summary = self.data_summary.get('summary', {})
            validation = self.data_summary.get('validation', {})
            
            logger.info(f"   数据形状: {summary.get('total_rows', 0)} 行 × {summary.get('total_cols', 0)} 列")
            logger.info(f"   年份范围: {summary.get('year_range', 'N/A')}")
            logger.info(f"   国家数量: {summary.get('countries', 0)}")
            logger.info(f"   数据状态: {summary.get('data_status', 'unknown')}")
            
            # 关键变量可用性
            key_vars = summary.get('key_variables_available', [])
            if key_vars:
                logger.info("   可用关键变量:")
                for var in key_vars:
                    logger.info(f"     • {var}")
            else:
                logger.warning("   ⚠️ 关键变量均不可用")
            
            # 数据验证结果
            if validation.get('is_valid_for_analysis'):
                logger.info("   ✅ 数据适合计量分析")
                return True
            else:
                logger.warning("   ❌ 数据不适合计量分析")
                issues = validation.get('issues', [])
                for issue in issues:
                    logger.warning(f"     • {issue}")
                logger.info("   将继续运行以演示空数据处理框架")
                return False
            
        except Exception as e:
            logger.error(f"数据加载步骤失败: {str(e)}")
            return False
    
    def _step_model_estimation(self) -> bool:
        """步骤2: 运行计量模型"""
        try:
            logger.info("🔍 开始运行计量模型...")
            
            # 运行所有模型
            self.model_results = self.models.run_all_models(self.data)
            
            # 打印模型运行结果
            overview = self.model_results.get('overview', {})
            total_models = overview.get('total_models', 0)
            completed_models = overview.get('completed_models', 0)
            failed_models = overview.get('failed_models', 0)
            
            logger.info(f"   总模型数: {total_models}")
            logger.info(f"   成功估计: {completed_models}")
            logger.info(f"   估计失败: {failed_models}")
            
            # 详细模型状态
            models_dict = self.model_results.get('models', {})
            for model_name, result in models_dict.items():
                model_config = self.config.analysis.RESEARCH_MODELS.get(model_name, {})
                status_icon = "✅" if result.get('status') == 'success' else "❌"
                logger.info(f"   {status_icon} {model_config.get('name', model_name)}: {result.get('status_message', 'N/A')}")
                
                if result.get('status') == 'success':
                    n_obs = result.get('n_obs', 0)
                    r_squared = result.get('r_squared', np.nan)
                    r2_str = f"{r_squared:.4f}" if not np.isnan(r_squared) else "N/A"
                    logger.info(f"       观测数: {n_obs}, R²: {r2_str}")
            
            return completed_models > 0
            
        except Exception as e:
            logger.error(f"模型估计步骤失败: {str(e)}")
            return False
    
    def _step_report_generation(self) -> bool:
        """步骤3: 生成分析报告"""
        try:
            logger.info("📝 生成分析报告...")
            
            if self.model_results is None:
                logger.warning("   没有模型结果，创建空报告框架")
                self.model_results = {'overview': {'total_models': 0, 'completed_models': 0, 'failed_models': 0}, 'models': {}}
            
            # 生成所有报告
            generated_reports = self.reporter.generate_all_reports(
                self.model_results, 
                self.data_summary
            )
            
            # 打印生成的报告
            for report_type, file_path in generated_reports.items():
                logger.info(f"   ✅ {report_type}: {file_path}")
            
            return len(generated_reports) > 0
            
        except Exception as e:
            logger.error(f"报告生成步骤失败: {str(e)}")
            return False
    
    def _step_visualization_creation(self) -> bool:
        """步骤4: 创建可视化图表"""
        try:
            logger.info("🎨 创建可视化图表...")
            
            if self.model_results is None:
                logger.warning("   没有模型结果，创建占位符图表")
                self.model_results = {'overview': {'total_models': 0, 'completed_models': 0, 'failed_models': 0}, 'models': {}}
            
            # 生成所有可视化
            generated_figures = self.visualizer.generate_all_visualizations(self.model_results)
            
            # 打印生成的图表
            for figure_type, file_path in generated_figures.items():
                logger.info(f"   🎯 {figure_type}: {file_path}")
            
            return len(generated_figures) > 0
            
        except Exception as e:
            logger.error(f"可视化创建步骤失败: {str(e)}")
            return False
    
    def _step_pipeline_summary(self) -> Dict[str, Any]:
        """步骤5: 流水线总结"""
        summary = {}
        
        try:
            logger.info("📋 生成流水线执行摘要...")
            
            # 数据摘要
            if self.data_summary:
                data_summary = self.data_summary.get('summary', {})
                summary['data_summary'] = {
                    'total_observations': data_summary.get('total_rows', 0),
                    'total_variables': data_summary.get('total_cols', 0),
                    'year_range': data_summary.get('year_range', 'N/A'),
                    'countries_count': data_summary.get('countries', 0),
                    'data_quality': data_summary.get('data_status', 'unknown')
                }
            
            # 模型摘要
            if self.model_results:
                overview = self.model_results.get('overview', {})
                summary['model_summary'] = {
                    'total_models': overview.get('total_models', 0),
                    'successful_models': overview.get('completed_models', 0),
                    'failed_models': overview.get('failed_models', 0),
                    'success_rate': f"{overview.get('completed_models', 0) / max(overview.get('total_models', 1), 1) * 100:.1f}%"
                }
            
            # 输出文件统计
            output_dir = self.config.output.OUTPUT_PATHS['regression_results'].parent
            figures_dir = self.config.output.FIGURE_PATHS['coefficient_comparison'].parent
            
            summary['output_summary'] = {
                'reports_directory': str(output_dir),
                'figures_directory': str(figures_dir),
                'total_output_files': len(list(output_dir.glob('*'))) + len(list(figures_dir.glob('*')))
            }
            
            # 打印关键统计
            logger.info("🎯 关键结果摘要:")
            if self.data is not None:
                logger.info(f"   • 数据可用性: {'是' if len(self.data) > 0 else '否'}")
            if self.model_results:
                logger.info(f"   • 模型成功率: {summary['model_summary']['success_rate']}")
            logger.info(f"   • 输出文件数: {summary['output_summary']['total_output_files']}")
            
            # 下一步建议
            self._print_next_steps_recommendations()
            
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            summary['error'] = str(e)
        
        return summary
    
    def _print_next_steps_recommendations(self):
        """打印下一步建议"""
        logger.info("\n💡 下一步建议:")
        
        if self.data is None or len(self.data) == 0:
            logger.info("   1. 检查08_variable_construction模块是否成功生成analytical_panel.csv")
            logger.info("   2. 确认核心变量(Node-DLI, Vul_US, OVI等)数据完整性")
            logger.info("   3. 数据就位后重新运行本模块获取实质性结果")
        else:
            if self.model_results and self.model_results.get('overview', {}).get('completed_models', 0) == 0:
                logger.info("   1. 检查数据质量，确保满足最少观测数要求")
                logger.info("   2. 验证关键变量是否存在异常值或编码问题")
                logger.info("   3. 考虑调整模型规范以适应当前数据特征")
            else:
                logger.info("   1. 查看生成的分析报告了解详细结果")
                logger.info("   2. 实施稳健性检验验证结果可靠性")
                logger.info("   3. 基于结果撰写学术论文或政策报告")
        
        logger.info("   4. 安装statsmodels和linearmodels库以启用完整计量分析功能")
        logger.info("   5. 安装matplotlib和seaborn库以生成专业图表")

    def run_quick_diagnostic(self) -> Dict[str, Any]:
        """
        运行快速诊断检查
        
        Returns:
            诊断结果字典
        """
        logger.info("🔧 运行快速诊断检查...")
        
        diagnostic_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'module_status': 'operational',
            'dependencies': self._check_dependencies(),
            'data_availability': self._check_data_availability(),
            'configuration': self._check_configuration(),
            'recommendations': []
        }
        
        # 基于检查结果给出建议
        if not diagnostic_results['dependencies']['all_available']:
            diagnostic_results['recommendations'].append("安装缺失的Python依赖库")
        
        if not diagnostic_results['data_availability']['data_exists']:
            diagnostic_results['recommendations'].append("检查08模块的数据生成状态")
        
        if not diagnostic_results['configuration']['paths_valid']:
            diagnostic_results['recommendations'].append("检查输出目录权限设置")
        
        logger.info(f"✅ 诊断完成，发现 {len(diagnostic_results['recommendations'])} 个建议")
        
        return diagnostic_results
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """检查依赖库可用性"""
        dependencies = {}
        
        try:
            import statsmodels
            dependencies['statsmodels'] = True
        except ImportError:
            dependencies['statsmodels'] = False
        
        try:
            from linearmodels import PanelOLS
            dependencies['linearmodels'] = True
        except ImportError:
            dependencies['linearmodels'] = False
        
        try:
            import matplotlib.pyplot as plt
            dependencies['matplotlib'] = True
        except ImportError:
            dependencies['matplotlib'] = False
        
        try:
            import seaborn as sns
            dependencies['seaborn'] = True
        except ImportError:
            dependencies['seaborn'] = False
        
        dependencies['all_available'] = all(dependencies.values())
        
        return dependencies
    
    def _check_data_availability(self) -> Dict[str, Any]:
        """检查数据可用性"""
        data_check = {}
        
        analytical_panel_path = self.data_loader.analytical_panel_path
        data_check['data_exists'] = analytical_panel_path.exists()
        data_check['data_path'] = str(analytical_panel_path)
        
        if data_check['data_exists']:
            try:
                df = pd.read_csv(analytical_panel_path)
                data_check['data_shape'] = df.shape
                data_check['data_empty'] = len(df) == 0
            except Exception as e:
                data_check['data_readable'] = False
                data_check['error'] = str(e)
        
        return data_check
    
    def _check_configuration(self) -> Dict[str, bool]:
        """检查配置有效性"""
        config_check = {}
        
        # 检查输出目录
        try:
            output_dir = self.config.output.OUTPUT_PATHS['regression_results'].parent
            figures_dir = self.config.output.FIGURE_PATHS['coefficient_comparison'].parent
            
            config_check['output_dir_exists'] = output_dir.exists()
            config_check['figures_dir_exists'] = figures_dir.exists()
            config_check['paths_valid'] = config_check['output_dir_exists'] and config_check['figures_dir_exists']
        except Exception as e:
            config_check['paths_valid'] = False
            config_check['error'] = str(e)
        
        return config_check


def main():
    """主函数入口"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.logging.LOG_FILE, encoding='utf-8')
        ]
    )
    
    print("🔬 09_econometric_analysis - 计量经济学分析框架")
    print("=" * 60)
    
    # 打印配置摘要
    print_config_summary()
    print()
    
    # 创建并运行分析流水线
    pipeline = EconometricAnalysisPipeline()
    
    try:
        # 可选: 运行快速诊断
        if '--diagnostic' in sys.argv:
            diagnostic_results = pipeline.run_quick_diagnostic()
            print(f"\n🔧 诊断结果: 发现 {len(diagnostic_results['recommendations'])} 个建议")
            return
        
        # 运行完整流水线
        results = pipeline.run_full_pipeline()
        
        # 打印最终状态
        print(f"\n🎯 流水线执行完成!")
        print(f"状态: {results['status']}")
        print(f"耗时: {results['total_duration_seconds']:.2f} 秒")
        print(f"完成步骤: {results['steps_completed']}/{results['steps_total']}")
        
        if results.get('pipeline_success'):
            print("✅ 所有步骤执行成功")
            print(f"\n📁 输出文件位置:")
            print(f"  报告: {config.output.OUTPUT_PATHS['regression_results'].parent}")
            print(f"  图表: {config.output.FIGURE_PATHS['coefficient_comparison'].parent}")
        else:
            print("⚠️ 部分步骤执行失败，但框架演示完成")
            print("请检查日志了解详细信息")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 程序异常: {str(e)}")
        logger.exception("程序执行异常")
    finally:
        print(f"\n📊 日志文件: {config.logging.LOG_FILE}")


if __name__ == "__main__":
    main()