#!/usr/bin/env python3
"""
网络结构异质性分析主程序 (Network Heterogeneity Analysis Main)
==========================================================

本程序是模块08的主执行接口，整合数据加载、核心分析和可视化功能，
完成"双向动态锁定效应(DLI)网络结构异质性"的完整分析流程。

核心研究问题：
Q1 (全局异质性): DLI对网络韧性的因果效应，是否在更稠密、更集聚、或更中心化的网络中表现得不同？
Q2 (局部异质性): 贸易关系的锁定效应，是否会因贸易双方在网络中的重要性而得到放大或缩小？

执行流程：
1. 数据加载与预处理
2. 全局异质性分析 (DLI × 全局网络指标)
3. 局部异质性分析 (DLI × 局部节点指标)
4. 可视化生成
5. 结果输出与汇总

作者：Energy Network Analysis Team
版本：v1.0
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback

# 导入分析模块
from data_loader import HeterogeneityDataLoader
from analysis import HeterogeneityAnalyzer
from visualizer import HeterogeneityVisualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkHeterogeneityPipeline:
    """网络结构异质性分析管道"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # 初始化组件
        self.data_loader = HeterogeneityDataLoader()
        self.analyzer = HeterogeneityAnalyzer()
        self.visualizer = HeterogeneityVisualizer(str(self.output_dir / "figures"))
        
        self.results = {}
        self.execution_time = None
        
        logger.info(f"🚀 初始化网络异质性分析管道，输出目录: {self.output_dir}")
    
    def run_complete_analysis(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行完整的异质性分析
        
        Args:
            config: 配置参数字典
            
        Returns:
            分析结果摘要
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("🎯 开始网络结构异质性分析")
        logger.info("=" * 60)
        
        try:
            # 1. 数据加载
            logger.info("\n📊 第1步：数据加载与预处理")
            global_data, local_data = self.data_loader.create_analysis_dataset()
            
            if len(global_data) == 0 or len(local_data) == 0:
                raise ValueError("数据集为空，无法进行分析")
            
            logger.info(f"   ✅ 全局分析数据集: {len(global_data)} 行")
            logger.info(f"   ✅ 局部分析数据集: {len(local_data)} 行")
            
            # 2. 全局异质性分析
            logger.info("\n🌐 第2步：全局异质性分析 (DLI × 全局网络指标)")
            global_results = self.analyzer.run_global_analysis(
                global_data,
                dli_vars=config.get('dli_vars') if config else None,
                global_vars=config.get('global_vars') if config else None,
                control_vars=config.get('control_vars') if config else None,
                interactions_to_test=config.get('interactions_to_test', {}).get('global') if config else None
            )
            
            # 3. 局部异质性分析
            logger.info("\n🏠 第3步：局部异质性分析 (DLI × 局部节点指标)")
            local_results = self.analyzer.run_local_analysis(
                local_data,
                dli_vars=config.get('dli_vars') if config else None,
                local_vars=config.get('local_vars') if config else None,
                control_vars=config.get('control_vars') if config else None,
                interactions_to_test=config.get('interactions_to_test', {}).get('local') if config else None
            )
            
            # 4. 结果汇总
            logger.info("\n📋 第4步：结果汇总与分析")
            results_table = self.analyzer.create_results_table()
            significant_interactions = self.analyzer.get_significant_interactions()
            
            # 5. 可视化生成
            logger.info("\n🎨 第5步：可视化生成")
            self._generate_visualizations(global_results, local_results, results_table, significant_interactions)
            
            # 6. 输出结果
            logger.info("\n💾 第6步：保存分析结果")
            self._save_results(global_results, local_results, results_table, significant_interactions)
            
            # 记录执行时间
            end_time = datetime.now()
            self.execution_time = end_time - start_time
            
            # 生成摘要
            summary = self._create_analysis_summary(significant_interactions, results_table)
            
            logger.info("=" * 60)
            logger.info(f"✅ 网络结构异质性分析完成！耗时: {self.execution_time}")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 分析过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
    
    def run_quick_demo(self) -> Dict[str, Any]:
        """
        运行快速演示分析
        
        Returns:
            演示分析结果
        """
        logger.info("🚀 运行快速演示模式...")
        
        # 使用默认配置和示例数据 - 精确指定要测试的交互项
        demo_config = {
            'dli_vars': ['dli_composite'],
            'global_vars': ['global_density'],
            'local_vars': ['betweenness_centrality'],
            'control_vars': [],
            # 精确指定要测试的交互项组合
            'interactions_to_test': {
                'global': [
                    ['dli_composite', 'global_density']
                ],
                'local': [
                    ['dli_composite', 'betweenness_centrality']
                ]
            }
        }
        
        return self.run_complete_analysis(demo_config)
    
    def _generate_visualizations(self, global_results: Dict, local_results: Dict,
                               results_table, significant_interactions: Dict):
        """生成所有可视化图表"""
        
        try:
            # 合并结果
            all_results = {**global_results, **local_results}
            
            # 边际效应图
            if all_results:
                self.visualizer.plot_marginal_effects(all_results, self.data_loader.causal_data or {})
            
            # 交互效应热力图
            if len(results_table) > 0:
                self.visualizer.plot_interaction_heatmap(results_table)
            
            # 显著性概览图
            if len(results_table) > 0:
                self.visualizer.plot_significance_overview(results_table)
            
            # 回归诊断图
            if all_results:
                self.visualizer.plot_regression_diagnostics(all_results)
            
            # 摘要报告图
            if significant_interactions:
                self.visualizer.create_summary_report_figure(significant_interactions)
            
            logger.info("✅ 所有可视化图表生成完成")
            
        except Exception as e:
            logger.error(f"❌ 可视化生成失败: {str(e)}")
    
    def _save_results(self, global_results: Dict, local_results: Dict,
                     results_table, significant_interactions: Dict):
        """保存分析结果"""
        
        try:
            # 保存结果表格
            if len(results_table) > 0:
                results_path = self.output_dir / "tables" / "heterogeneity_results.csv"
                results_table.to_csv(results_path, index=False)
                logger.info(f"💾 保存结果表格: {results_path}")
            
            # 保存显著交互效应
            if significant_interactions:
                sig_path = self.output_dir / "tables" / "significant_interactions.json"
                with open(sig_path, 'w', encoding='utf-8') as f:
                    json.dump(significant_interactions, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 保存显著交互效应: {sig_path}")
            
            # 保存全部回归结果
            all_results = {**global_results, **local_results}
            if all_results:
                # 简化结果用于JSON保存
                simplified_results = {}
                for model_name, result in all_results.items():
                    simplified_results[model_name] = {
                        'model_name': result.get('model_name'),
                        'n_obs': result.get('n_obs'),
                        'r_squared': result.get('r_squared'),
                        'coefficients': result.get('coefficients', {}),
                        'p_values': result.get('p_values', {}),
                        'summary_stats': {
                            'significant_interactions': len([p for p in result.get('p_values', {}).values() 
                                                           if isinstance(p, (int, float)) and p < 0.05])
                        }
                    }
                
                results_path = self.output_dir / "tables" / "full_regression_results.json"
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_results, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"💾 保存完整回归结果: {results_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {str(e)}")
    
    def _create_analysis_summary(self, significant_interactions: Dict, 
                               results_table) -> Dict[str, Any]:
        """创建分析摘要"""
        
        summary = {
            'analysis_type': 'Network Structure Heterogeneity Analysis',
            'execution_time': str(self.execution_time),
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.analyzer.global_results) + len(self.analyzer.local_results),
            'global_models': len(self.analyzer.global_results),
            'local_models': len(self.analyzer.local_results),
            'data_summary': {
                'global_dataset_size': len(self.data_loader.global_metrics) if self.data_loader.global_metrics is not None else 0,
                'local_dataset_size': len(self.data_loader.local_metrics) if self.data_loader.local_metrics is not None else 0
            },
            'significant_interactions': significant_interactions,
            'key_findings': self._extract_key_findings(significant_interactions, results_table),
            'output_files': {
                'tables': list((self.output_dir / "tables").glob("*.csv")) + list((self.output_dir / "tables").glob("*.json")),
                'figures': list((self.output_dir / "figures").glob("*.png"))
            }
        }
        
        return summary
    
    def _extract_key_findings(self, significant_interactions: Dict, results_table) -> List[str]:
        """提取关键发现"""
        
        findings = []
        
        if significant_interactions:
            total = significant_interactions.get('total_interactions', 0)
            significant = significant_interactions.get('significant_interactions', 0)
            rate = significant_interactions.get('significance_rate', 0)
            
            findings.append(f"共测试了 {total} 个交互效应，其中 {significant} 个具有统计显著性 ({rate:.1%})")
            
            if significant > 0:
                strongest = significant_interactions.get('strongest_effect')
                if strongest:
                    findings.append(f"最强的交互效应来自 {strongest.get('interaction')}，系数为 {strongest.get('coefficient'):.3f}")
            
            # 分析类型对比
            if len(results_table) > 0:
                global_sig = len(results_table[(results_table['analysis_type'] == 'Global') & 
                                             (results_table['significant'] == True)])
                local_sig = len(results_table[(results_table['analysis_type'] == 'Local') & 
                                            (results_table['significant'] == True)])
                
                if global_sig > local_sig:
                    findings.append("全局网络特征对DLI效应的调节作用更为显著")
                elif local_sig > global_sig:
                    findings.append("局部节点特征对DLI效应的调节作用更为显著")
                else:
                    findings.append("全局和局部网络特征对DLI效应的调节作用相当")
        
        if not findings:
            findings.append("未发现显著的网络结构异质性效应")
        
        return findings
    
    def generate_final_report(self) -> str:
        """生成最终分析报告"""
        
        report_path = self.output_dir / "heterogeneity_analysis_report.md"
        
        # 读取分析结果
        results_table = None
        significant_interactions = None
        
        try:
            results_path = self.output_dir / "tables" / "heterogeneity_results.csv"
            if results_path.exists():
                import pandas as pd
                results_table = pd.read_csv(results_path)
        except:
            pass
            
        try:
            sig_path = self.output_dir / "tables" / "significant_interactions.json"
            if sig_path.exists():
                with open(sig_path, 'r', encoding='utf-8') as f:
                    significant_interactions = json.load(f)
        except:
            pass
        
        # 生成报告内容
        report_content = f"""# 网络结构异质性分析报告
## Network Structure Heterogeneity Analysis Report

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析模块**: 08_heterogeneity_analysis v1.0

---

## 🎯 研究目标

本分析旨在探究双向动态锁定效应(DLI)是否会因能源贸易网络的拓扑结构不同而表现出异质性：

1. **全局异质性**: DLI对网络韧性的因果效应是否在更稠密、更集聚或更中心化的网络中表现不同？
2. **局部异质性**: 贸易关系的锁定效应是否会因贸易双方在网络中的重要性而得到放大或缩小？

---

## 📊 分析结果摘要

"""
        
        if significant_interactions:
            total = significant_interactions.get('total_interactions', 0)
            significant = significant_interactions.get('significant_interactions', 0)
            rate = significant_interactions.get('significance_rate', 0)
            
            report_content += f"""### 总体发现

- **交互效应测试总数**: {total}
- **显著交互效应数量**: {significant}
- **显著性比例**: {rate:.1%}

"""
            
            if significant > 0:
                strongest = significant_interactions.get('strongest_effect', {})
                report_content += f"""### 最强交互效应

- **变量组合**: {strongest.get('interaction', 'N/A')}
- **效应系数**: {strongest.get('coefficient', 'N/A')}
- **显著性水平**: {strongest.get('p_value', 'N/A')}

"""
        
        if results_table is not None and len(results_table) > 0:
            global_count = len(results_table[results_table['analysis_type'] == 'Global'])
            local_count = len(results_table[results_table['analysis_type'] == 'Local'])
            
            report_content += f"""### 分析类型分布

- **全局分析模型**: {global_count}
- **局部分析模型**: {local_count}

"""
        
        report_content += f"""---

## 📈 可视化结果

本分析生成了以下可视化图表：

1. **交互效应热力图** (`interaction_heatmap.png`)
   - 展示不同DLI变量与网络特征的交互效应强度
   
2. **显著性概览图** (`significance_overview.png`)
   - 显示显著性分布、系数分布等统计概览
   
3. **边际效应图** (`marginal_effect_*.png`)
   - 展示在不同网络特征水平下DLI效应的变化
   
4. **回归诊断图** (`diagnostics_*.png`)
   - 回归模型的残差分析和诊断检验

---

## 🔍 方法论说明

### 分析方法

本研究基于05_causal_validation的基准回归模型，引入DLI指标与网络特征的交互项：

**全局分析模型**:
```
Y ~ DLI + Global_Metric + DLI × Global_Metric + Controls
```

**局部分析模型**:
```
Y ~ DLI + Local_Metric + DLI × Local_Metric + Controls
```

### 数据来源

- **DLI效应指标**: 来自 `04_dli_analysis` 模块
- **全局网络指标**: 来自 `03_metrics` 模块的网络整体拓扑指标
- **局部节点指标**: 来自 `03_metrics` 模块的节点中心性指标
- **因果分析数据**: 来自 `05_causal_validation` 模块的基准回归变量

---

## 📁 输出文件

### 数据表格
- `heterogeneity_results.csv`: 完整的回归结果汇总表
- `significant_interactions.json`: 显著交互效应的详细信息
- `full_regression_results.json`: 所有回归模型的完整结果

### 可视化图表
- 所有图表保存在 `outputs/figures/` 目录下
- 支持高分辨率PNG格式，适合学术发表

---

## 💡 研究意义

本分析揭示了网络结构对DLI效应的调节作用，为理解能源贸易锁定效应的复杂性提供了新的视角。研究发现有助于：

1. **理论贡献**: 丰富了动态锁定理论的网络维度
2. **政策启示**: 为能源政策制定提供网络结构的考虑因素
3. **方法创新**: 建立了网络异质性分析的标准化框架

---

*本报告由 Network Heterogeneity Analysis Pipeline v1.0 自动生成*  
*Energy Network Analysis Team*
"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 最终报告已生成: {report_path}")
        return str(report_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='网络结构异质性分析')
    parser.add_argument('--mode', choices=['full', 'demo'], default='demo',
                       help='分析模式: full=完整分析, demo=快速演示')
    parser.add_argument('--output-dir', default='outputs',
                       help='输出目录路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径(JSON格式)')
    
    args = parser.parse_args()
    
    # 创建分析管道
    pipeline = NetworkHeterogeneityPipeline(args.output_dir)
    
    try:
        # 加载配置
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 运行分析
        if args.mode == 'demo':
            summary = pipeline.run_quick_demo()
        else:
            summary = pipeline.run_complete_analysis(config)
        
        # 生成最终报告
        report_path = pipeline.generate_final_report()
        
        # 输出摘要
        print("\n" + "="*60)
        print("📊 网络结构异质性分析完成!")
        print("="*60)
        print(f"执行时间: {summary.get('execution_time', 'N/A')}")
        print(f"总模型数: {summary.get('total_models', 0)}")
        print(f"显著交互: {summary.get('significant_interactions', {}).get('significant_interactions', 0)}")
        print(f"最终报告: {report_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()