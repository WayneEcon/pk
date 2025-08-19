#!/usr/bin/env python3
"""
结果报告模块 (Reporting Module)
==============================

09_econometric_analysis 模块的结果报告生成组件

作者：Energy Network Analysis Team
版本：v1.0 - 计量分析框架
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

try:
    from .config import config
except ImportError:
    import config
    config = config.config

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    报告生成器 - 专门处理空结果和缺失数据的报告生成逻辑
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录，如果为None则使用配置中的默认目录
        """
        self.config = config
        self.output_dir = output_dir or self.config.output.OUTPUT_PATHS['regression_results'].parent
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"📝 报告生成器初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_regression_results_csv(self, model_results: Dict[str, Any]) -> Path:
        """
        生成回归结果CSV文件
        
        Args:
            model_results: 模型结果字典
            
        Returns:
            CSV文件路径
        """
        logger.info("📊 生成回归结果CSV...")
        
        # 准备结果数据列表
        results_data = []
        
        if 'models' in model_results:
            models_dict = model_results['models']
        else:
            models_dict = model_results
        
        for model_name, result in models_dict.items():
            
            # 处理普通模型结果
            if model_name != 'model_3_local_projection_validation':
                row_data = self._extract_model_row(model_name, result)
                results_data.append(row_data)
            
            # 处理局部投影模型的多期结果
            else:
                if result.get('status') == 'success' and 'horizon_results' in result:
                    for horizon_key, horizon_result in result['horizon_results'].items():
                        row_data = self._extract_horizon_row(model_name, horizon_key, horizon_result)
                        results_data.append(row_data)
                else:
                    # 局部投影模型失败的情况
                    row_data = self._extract_model_row(model_name, result)
                    results_data.append(row_data)
        
        # 如果没有任何结果，创建空行
        if not results_data:
            results_data.append(self._create_empty_row())
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results_data)
        
        # 保存CSV
        csv_path = self.output_dir / "regression_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"✅ 回归结果CSV已生成: {csv_path}")
        logger.info(f"   包含 {len(results_df)} 行结果")
        
        return csv_path
    
    def _extract_model_row(self, model_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从模型结果中提取一行数据
        
        Args:
            model_name: 模型名称
            result: 模型结果
            
        Returns:
            行数据字典
        """
        # 获取模型配置信息
        model_config = self.config.analysis.RESEARCH_MODELS.get(model_name, {})
        
        row_data = {
            'model_name': model_name,
            'model_description': model_config.get('name', model_name),
            'chapter': model_config.get('chapter', 'N/A'),
            'method': result.get('model_type', 'N/A'),
            'status': result.get('status', 'unknown'),
            'status_message': result.get('status_message', ''),
            'formula': result.get('formula', model_config.get('formula', '')),
            'n_obs': result.get('n_obs', 0),
            'n_entities': result.get('n_entities', 0),
            'r_squared': result.get('r_squared', np.nan),
            'r_squared_within': result.get('r_squared_within', np.nan),
            'f_statistic': result.get('f_statistic', np.nan)
        }
        
        # 提取关键系数
        coefficients = result.get('coefficients', {})
        std_errors = result.get('std_errors', {})
        p_values = result.get('p_values', {})
        
        # 定义关键变量
        key_vars = ['node_dli_us', 'ovi', 'ovi_lag1', 'us_prod_shock', 'us_prod_shock_x_ovi_lag1']
        
        for var in key_vars:
            row_data[f'{var}_coef'] = coefficients.get(var, np.nan)
            row_data[f'{var}_se'] = std_errors.get(var, np.nan)
            row_data[f'{var}_pvalue'] = p_values.get(var, np.nan)
            
            # 计算显著性星号
            p_val = p_values.get(var, np.nan)
            if pd.isna(p_val):
                row_data[f'{var}_significance'] = ''
            elif p_val < 0.01:
                row_data[f'{var}_significance'] = '***'
            elif p_val < 0.05:
                row_data[f'{var}_significance'] = '**'
            elif p_val < 0.10:
                row_data[f'{var}_significance'] = '*'
            else:
                row_data[f'{var}_significance'] = ''
        
        return row_data
    
    def _extract_horizon_row(self, model_name: str, horizon_key: str, horizon_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从局部投影模型的单期结果中提取一行数据
        
        Args:
            model_name: 模型名称
            horizon_key: 期数标识
            horizon_result: 该期结果
            
        Returns:
            行数据字典
        """
        model_config = self.config.analysis.RESEARCH_MODELS.get(model_name, {})
        
        row_data = {
            'model_name': f"{model_name}_{horizon_key}",
            'model_description': f"{model_config.get('name', model_name)} - {horizon_key}",
            'chapter': model_config.get('chapter', 'N/A'),
            'method': 'local_projection',
            'status': 'success',
            'status_message': f'局部投影估计成功 - {horizon_key}',
            'formula': f"Δvul_us(t+{horizon_result.get('horizon', 0)}) ~ us_prod_shock(t) * ovi_lag1(t-1) + Controls",
            'n_obs': horizon_result.get('n_obs', 0),
            'n_entities': 0,  # 局部投影结果中可能没有这个信息
            'r_squared': horizon_result.get('r_squared', np.nan),
            'r_squared_within': np.nan,  # 局部投影通常没有within R²
            'f_statistic': np.nan,
            'horizon': horizon_result.get('horizon', 0)
        }
        
        # 提取系数
        coefficients = horizon_result.get('coefficients', {})
        std_errors = horizon_result.get('std_errors', {})
        p_values = horizon_result.get('p_values', {})
        
        key_vars = ['us_prod_shock', 'ovi_lag1', 'us_prod_shock_x_ovi_lag1']
        
        for var in key_vars:
            row_data[f'{var}_coef'] = coefficients.get(var, np.nan)
            row_data[f'{var}_se'] = std_errors.get(var, np.nan)
            row_data[f'{var}_pvalue'] = p_values.get(var, np.nan)
            
            # 显著性星号
            p_val = p_values.get(var, np.nan)
            if pd.isna(p_val):
                row_data[f'{var}_significance'] = ''
            elif p_val < 0.01:
                row_data[f'{var}_significance'] = '***'
            elif p_val < 0.05:
                row_data[f'{var}_significance'] = '**'
            elif p_val < 0.10:
                row_data[f'{var}_significance'] = '*'
            else:
                row_data[f'{var}_significance'] = ''
        
        return row_data
    
    def _create_empty_row(self) -> Dict[str, Any]:
        """
        创建空结果行
        
        Returns:
            空行数据字典
        """
        return {
            'model_name': 'no_models_run',
            'model_description': '没有运行任何模型',
            'chapter': 'N/A',
            'method': 'N/A',
            'status': 'no_data',
            'status_message': '数据不可用，未运行任何模型',
            'formula': 'N/A',
            'n_obs': 0,
            'n_entities': 0,
            'r_squared': np.nan,
            'r_squared_within': np.nan,
            'f_statistic': np.nan,
            **{f'{var}_{stat}': np.nan for var in ['node_dli_us', 'ovi', 'ovi_lag1', 'us_prod_shock', 'us_prod_shock_x_ovi_lag1'] 
               for stat in ['coef', 'se', 'pvalue']},
            **{f'{var}_significance': '' for var in ['node_dli_us', 'ovi', 'ovi_lag1', 'us_prod_shock', 'us_prod_shock_x_ovi_lag1']}
        }
    
    def generate_analysis_report_md(self, model_results: Dict[str, Any], data_summary: Optional[Dict] = None) -> Path:
        """
        生成Markdown格式的分析报告
        
        Args:
            model_results: 模型结果字典
            data_summary: 数据摘要（可选）
            
        Returns:
            Markdown文件路径
        """
        logger.info("📄 生成Markdown分析报告...")
        
        # 开始构建报告内容
        report_content = self._build_markdown_report(model_results, data_summary)
        
        # 保存Markdown文件
        md_path = self.output_dir / "analysis_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✅ Markdown报告已生成: {md_path}")
        
        return md_path
    
    def _build_markdown_report(self, model_results: Dict[str, Any], data_summary: Optional[Dict] = None) -> str:
        """
        构建Markdown报告内容
        
        Args:
            model_results: 模型结果字典
            data_summary: 数据摘要
            
        Returns:
            Markdown报告文本
        """
        report_lines = [
            "# 计量经济学分析报告",
            "## Econometric Analysis Report",
            "",
            f"**生成时间**: {self.timestamp}",
            f"**模块版本**: 09_econometric_analysis v1.0",
            "",
            "---",
            ""
        ]
        
        # 1. 执行摘要
        report_lines.extend(self._build_executive_summary(model_results))
        
        # 2. 数据概况
        if data_summary:
            report_lines.extend(self._build_data_overview(data_summary))
        
        # 3. 模型结果
        report_lines.extend(self._build_model_results_section(model_results))
        
        # 4. 稳健性检验（如果有的话）
        report_lines.extend(self._build_robustness_section(model_results))
        
        # 5. 结论与政策含义
        report_lines.extend(self._build_conclusions_section(model_results))
        
        # 6. 技术附录
        report_lines.extend(self._build_technical_appendix(model_results))
        
        return "\n".join(report_lines)
    
    def _build_executive_summary(self, model_results: Dict[str, Any]) -> List[str]:
        """构建执行摘要部分"""
        section = [
            "## 1. 执行摘要 (Executive Summary)",
            ""
        ]
        
        if 'overview' in model_results:
            overview = model_results['overview']
            section.extend([
                f"本研究运行了 **{overview.get('total_models', 0)}** 个核心计量模型，其中:",
                f"- ✅ 成功估计: {overview.get('completed_models', 0)} 个",
                f"- ❌ 估计失败: {overview.get('failed_models', 0)} 个",
                f"- 📊 数据可用性: {'是' if overview.get('data_available', False) else '否'}",
                ""
            ])
            
            if overview.get('completed_models', 0) == 0:
                section.extend([
                    "⚠️ **重要提示**: 由于数据不可用或不完整，所有模型估计均失败。",
                    "这通常表明08模块的数据构建过程尚未完成或存在问题。",
                    "建议检查数据构建流程后重新运行分析。",
                    ""
                ])
        else:
            section.extend([
                "⚠️ **数据状态**: 模型结果不可用，可能由于数据缺失或模块运行异常。",
                ""
            ])
        
        section.extend([
            "### 1.1 研究模型概览",
            "",
            "| 模型 | 研究问题 | 方法 | 状态 |",
            "|------|----------|------|------|"
        ])
        
        models_dict = model_results.get('models', {})
        for model_name, result in models_dict.items():
            model_config = self.config.analysis.RESEARCH_MODELS.get(model_name, {})
            status_emoji = "✅" if result.get('status') == 'success' else "❌"
            section.append(f"| {model_config.get('name', model_name)} | {model_config.get('description', 'N/A')} | {result.get('model_type', 'N/A')} | {status_emoji} {result.get('status', 'unknown')} |")
        
        section.extend(["", "---", ""])
        
        return section
    
    def _build_data_overview(self, data_summary: Dict) -> List[str]:
        """构建数据概览部分"""
        section = [
            "## 2. 数据概览 (Data Overview)",
            ""
        ]
        
        if 'summary' in data_summary:
            summary = data_summary['summary']
            section.extend([
                "### 2.1 基础统计",
                "",
                f"- **总行数**: {summary.get('total_rows', 0):,}",
                f"- **总列数**: {summary.get('total_cols', 0)}",
                f"- **年份范围**: {summary.get('year_range', 'N/A')}",
                f"- **国家数量**: {summary.get('countries', 0)}",
                f"- **数据状态**: {summary.get('data_status', 'unknown')}",
                ""
            ])
            
            # 关键变量可用性
            key_vars = summary.get('key_variables_available', [])
            if key_vars:
                section.extend([
                    "### 2.2 关键变量可用性",
                    ""
                ])
                for var_info in key_vars:
                    section.append(f"- {var_info}")
                section.append("")
            else:
                section.extend([
                    "### 2.2 关键变量可用性",
                    "",
                    "❌ **关键研究变量均不可用**",
                    "",
                    "核心变量 (node_dli_us, vul_us, ovi, us_prod_shock) 数据缺失或全为空值。",
                    "建议检查08_variable_construction模块的运行状态。",
                    ""
                ])
        
        if 'validation' in data_summary:
            validation = data_summary['validation']
            section.extend([
                "### 2.3 数据验证结果",
                "",
                f"- **适合计量分析**: {'是' if validation.get('is_valid_for_analysis', False) else '否'}",
                ""
            ])
            
            issues = validation.get('issues', [])
            if issues:
                section.extend(["**发现的问题**:", ""])
                for issue in issues:
                    section.append(f"- ❌ {issue}")
                section.append("")
            
            recommendations = validation.get('recommendations', [])
            if recommendations:
                section.extend(["**建议**:", ""])
                for rec in recommendations:
                    section.append(f"- 💡 {rec}")
                section.append("")
        
        section.extend(["---", ""])
        return section
    
    def _build_model_results_section(self, model_results: Dict[str, Any]) -> List[str]:
        """构建模型结果部分"""
        section = [
            "## 3. 模型结果 (Model Results)",
            ""
        ]
        
        models_dict = model_results.get('models', {})
        
        if not models_dict:
            section.extend([
                "⚠️ **没有可用的模型结果**",
                "",
                "原因可能包括:",
                "- 分析数据不可用或为空",
                "- 关键变量缺失",
                "- 模型估计过程中发生错误",
                ""
            ])
            return section
        
        # 为每个模型创建详细结果
        for i, (model_name, result) in enumerate(models_dict.items(), 1):
            model_config = self.config.analysis.RESEARCH_MODELS.get(model_name, {})
            
            section.extend([
                f"### 3.{i} {model_config.get('name', model_name)}",
                "",
                f"**研究问题**: {model_config.get('description', 'N/A')}",
                f"**估计方法**: {result.get('model_type', 'N/A')}",
                f"**状态**: {result.get('status', 'unknown')}",
                ""
            ])
            
            if result.get('status') == 'success':
                section.extend(self._format_successful_model_result(result))
            else:
                section.extend([
                    f"❌ **估计失败**: {result.get('status_message', '未知错误')}",
                    "",
                    "**可能的解决方案**:",
                    "- 检查数据的完整性和质量",
                    "- 确认所需变量都已正确构建",
                    "- 检查样本量是否满足最少观测要求",
                    ""
                ])
        
        section.extend(["---", ""])
        return section
    
    def _format_successful_model_result(self, result: Dict[str, Any]) -> List[str]:
        """格式化成功的模型结果"""
        section = []
        
        # 基础统计信息
        section.extend([
            "#### 基础统计",
            "",
            f"- **观测数**: {result.get('n_obs', 0):,}",
            f"- **个体数**: {result.get('n_entities', 0)}",
            f"- **R²**: {result.get('r_squared', np.nan):.4f}" if not pd.isna(result.get('r_squared', np.nan)) else "- **R²**: N/A",
            ""
        ])
        
        # 关键系数表格
        coefficients = result.get('coefficients', {})
        std_errors = result.get('std_errors', {})
        p_values = result.get('p_values', {})
        
        if coefficients:
            section.extend([
                "#### 关键系数估计",
                "",
                "| 变量 | 系数 | 标准误 | P值 | 显著性 |",
                "|------|------|--------|-----|--------|"
            ])
            
            # 只展示关键变量
            key_vars = ['node_dli_us', 'ovi', 'ovi_lag1', 'us_prod_shock', 'us_prod_shock_x_ovi_lag1']
            
            for var in key_vars:
                if var in coefficients:
                    coef = coefficients[var]
                    se = std_errors.get(var, np.nan)
                    p_val = p_values.get(var, np.nan)
                    
                    # 格式化系数
                    coef_str = f"{coef:.4f}" if not pd.isna(coef) else "N/A"
                    se_str = f"({se:.4f})" if not pd.isna(se) else "(N/A)"
                    p_str = f"{p_val:.3f}" if not pd.isna(p_val) else "N/A"
                    
                    # 显著性星号
                    if not pd.isna(p_val):
                        if p_val < 0.01:
                            sig = "***"
                        elif p_val < 0.05:
                            sig = "**"
                        elif p_val < 0.10:
                            sig = "*"
                        else:
                            sig = ""
                    else:
                        sig = ""
                    
                    section.append(f"| {var} | {coef_str} | {se_str} | {p_str} | {sig} |")
            
            section.extend([
                "",
                "*注: *** p<0.01, ** p<0.05, * p<0.10*",
                ""
            ])
        
        # 处理局部投影的特殊情况
        if 'horizon_results' in result:
            section.extend([
                "#### 局部投影结果",
                "",
                f"估计了 {len(result['horizon_results'])} 个预测期的脉冲响应。",
                ""
            ])
        
        return section
    
    def _build_robustness_section(self, model_results: Dict[str, Any]) -> List[str]:
        """构建稳健性检验部分"""
        return [
            "## 4. 稳健性检验 (Robustness Checks)",
            "",
            "⚠️ **稳健性检验功能待实现**",
            "",
            "计划包含的稳健性检验:",
            "- 排除异常值重新估计",
            "- 变量缩尾处理",
            "- 替代控制变量",
            "- 分时期子样本分析",
            "- Bootstrap推断",
            "",
            "---",
            ""
        ]
    
    def _build_conclusions_section(self, model_results: Dict[str, Any]) -> List[str]:
        """构建结论部分"""
        section = [
            "## 5. 结论与政策含义 (Conclusions & Policy Implications)",
            ""
        ]
        
        overview = model_results.get('overview', {})
        completed = overview.get('completed_models', 0)
        total = overview.get('total_models', 0)
        
        if completed == 0:
            section.extend([
                "### 5.1 主要发现",
                "",
                "❌ **由于数据不可用，暂时无法得出实质性结论。**",
                "",
                "当前状态表明:",
                "- 08_variable_construction模块的数据构建过程可能尚未完成",
                "- 需要等待核心研究变量 (Node-DLI, Vul_US, OVI等) 的数据填充",
                "- 分析框架已就绪，一旦数据到位即可产出结果",
                "",
                "### 5.2 下一步工作",
                "",
                "1. **数据完善**: 确保08模块成功生成完整的analytical_panel.csv",
                "2. **模型验证**: 数据到位后重新运行本模块验证模型框架",
                "3. **结果解读**: 基于实际估计结果解读经济学含义",
                "4. **稳健性检验**: 实施多种稳健性检验确保结果可靠",
                ""
            ])
        else:
            section.extend([
                "### 5.1 主要发现",
                "",
                f"基于 {completed}/{total} 个成功估计的模型，主要发现包括:",
                "",
                "**核心结果** (待数据完善后更新):",
                "- DLI与能源脆弱性的关联性",
                "- OVI对DLI的因果效应",
                "- 美国产量冲击的动态影响",
                "",
                "### 5.2 政策含义",
                "",
                "**能源安全政策建议** (基于分析框架):",
                "- 多元化能源进口来源以降低依赖性锁定",
                "- 投资物理冗余基础设施以增强韧性",
                "- 建立早期预警机制应对供给冲击",
                ""
            ])
        
        section.extend(["---", ""])
        return section
    
    def _build_technical_appendix(self, model_results: Dict[str, Any]) -> List[str]:
        """构建技术附录"""
        return [
            "## 6. 技术附录 (Technical Appendix)",
            "",
            "### 6.1 模型规范",
            "",
            "**模型1: DLI-脆弱性关联检验**",
            "```",
            "vul_us_it = β₀ + β₁ × node_dli_us_it + β₂ × Controls_it + α_i + δ_t + ε_it",
            "```",
            "",
            "**模型2: OVI因果效应**",
            "```", 
            "node_dli_us_it = γ₀ + γ₁ × ovi_i(t-1) + γ₂ × Controls_it + α_i + δ_t + ε_it",
            "```",
            "",
            "**模型3: 局部投影验证**",
            "```",
            "Δvul_us_i(t+h) = θ₀ᵸ + θ₁ᵸ × us_prod_shock_t × ovi_i(t-1) + θ₂ᵸ × Controls_it + α_i + ε_it",
            "```",
            "",
            "### 6.2 估计方法",
            "",
            "- **面板数据**: 双向固定效应模型 (Two-Way Fixed Effects)",
            "- **标准误**: 个体聚类稳健标准误",
            "- **局部投影**: Jordà (2005) 方法",
            "- **软件**: Python + linearmodels + statsmodels",
            "",
            "### 6.3 数据来源",
            "",
            "- **基础数据**: 08_variable_construction模块输出",
            "- **时间范围**: 2000-2024",
            "- **国家范围**: 基于贸易网络分析的重要能源国家",
            "",
            "---",
            "",
            f"*本报告由 09_econometric_analysis 模块自动生成于 {self.timestamp}*",
            "",
            f"*Energy Network Analysis Team - Claude Code Framework*"
        ]
    
    def generate_model_diagnostics_json(self, model_results: Dict[str, Any]) -> Path:
        """
        生成模型诊断JSON文件
        
        Args:
            model_results: 模型结果字典
            
        Returns:
            JSON文件路径
        """
        logger.info("🔧 生成模型诊断JSON...")
        
        diagnostics_data = {
            'timestamp': self.timestamp,
            'module_version': '09_econometric_analysis v1.0',
            'overview': model_results.get('overview', {}),
            'model_diagnostics': {}
        }
        
        models_dict = model_results.get('models', {})
        for model_name, result in models_dict.items():
            diagnostics_data['model_diagnostics'][model_name] = {
                'status': result.get('status', 'unknown'),
                'estimation_method': result.get('model_type', 'unknown'),
                'sample_size': result.get('n_obs', 0),
                'r_squared': result.get('r_squared', None),
                'diagnostics': result.get('diagnostics', {}),
                'data_available': result.get('data_available', False)
            }
        
        # 保存JSON
        json_path = self.output_dir / "model_diagnostics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 模型诊断JSON已生成: {json_path}")
        
        return json_path
    
    def generate_all_reports(self, model_results: Dict[str, Any], data_summary: Optional[Dict] = None) -> Dict[str, Path]:
        """
        生成所有报告文件
        
        Args:
            model_results: 模型结果字典
            data_summary: 数据摘要
            
        Returns:
            生成文件路径字典
        """
        logger.info("📚 开始生成所有报告...")
        
        generated_files = {}
        
        try:
            # 1. CSV结果表
            generated_files['csv'] = self.generate_regression_results_csv(model_results)
        except Exception as e:
            logger.error(f"CSV报告生成失败: {str(e)}")
        
        try:
            # 2. Markdown报告
            generated_files['markdown'] = self.generate_analysis_report_md(model_results, data_summary)
        except Exception as e:
            logger.error(f"Markdown报告生成失败: {str(e)}")
        
        try:
            # 3. 诊断JSON
            generated_files['diagnostics'] = self.generate_model_diagnostics_json(model_results)
        except Exception as e:
            logger.error(f"诊断JSON生成失败: {str(e)}")
        
        logger.info(f"✅ 报告生成完成，共 {len(generated_files)} 个文件")
        
        return generated_files


# 便捷函数
def generate_reports(model_results: Dict[str, Any], data_summary: Optional[Dict] = None, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    生成报告的便捷函数
    
    Args:
        model_results: 模型结果字典
        data_summary: 数据摘要
        output_dir: 输出目录
        
    Returns:
        生成文件路径字典
    """
    reporter = ReportGenerator(output_dir)
    return reporter.generate_all_reports(model_results, data_summary)


if __name__ == "__main__":
    # 测试报告生成功能
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("📝 09_econometric_analysis 报告模块测试")
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
                'status_message': '数据不可用',
                'model_type': 'two_way_fixed_effects',
                'data_available': False,
                'n_obs': 0,
                'coefficients': {},
                'std_errors': {},
                'p_values': {}
            },
            'model_2_ovi_dli_causality': {
                'status': 'failed', 
                'status_message': '数据不可用',
                'model_type': 'two_way_fixed_effects_lagged',
                'data_available': False,
                'n_obs': 0,
                'coefficients': {},
                'std_errors': {},
                'p_values': {}
            },
            'model_3_local_projection_validation': {
                'status': 'failed',
                'status_message': '数据不可用',
                'model_type': 'local_projections',
                'data_available': False,
                'horizon_results': {}
            }
        }
    }
    
    test_data_summary = {
        'summary': {
            'total_rows': 0,
            'total_cols': 25,
            'year_range': 'N/A',
            'countries': 0,
            'key_variables_available': [],
            'data_status': 'empty'
        },
        'validation': {
            'is_valid_for_analysis': False,
            'issues': ['数据集为空'],
            'recommendations': ['等待08模块生成数据']
        }
    }
    
    # 测试报告生成
    reporter = ReportGenerator()
    files = reporter.generate_all_reports(test_results, test_data_summary)
    
    print(f"\n📊 测试结果:")
    for report_type, file_path in files.items():
        print(f"  {report_type}: {file_path}")
    
    print("\n🎉 报告模块测试完成!")