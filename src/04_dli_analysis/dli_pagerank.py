#!/usr/bin/env python3
"""
增强版动态锁定指数计算模块 (Enhanced DLI with Personalized PageRank)
==================================================================

本模块实现集成个性化PageRank的五维度动态锁定指数计算：

原有四个维度：
1. 贸易持续性 (Continuity): 衡量关系的长期性
2. 基础设施强度 (Infrastructure): 衡量专用性资产导致的锁定
3. 贸易稳定性 (Stability): 衡量关系的可靠性  
4. 市场锁定力 (Market Locking Power): 衡量市场结构导致的锁定效应

新增网络维度：
5. 个性化PageRank影响力: 衡量方向性网络锁定能力

核心创新：
- 严格区分方向性：美国出口锁定他国 vs 美国进口被他国锁定
- 使用PCA处理五维度多重共线性问题
- 学术规范的诊断分析和相关性检验
- 统一标尺的综合指数计算

版本：v1.0 - Enhanced DLI with Network Centrality
作者：Energy Network Analysis Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDLICalculator:
    """增强版DLI计算器，集成个性化PageRank网络维度"""
    
    def __init__(self, dli_data_path: Path, pagerank_data_path: Path, output_dir: Path):
        """
        初始化增强版DLI计算器
        
        Args:
            dli_data_path: 原有DLI四维度数据路径
            pagerank_data_path: 个性化PageRank数据路径
            output_dir: 输出目录
        """
        self.dli_data_path = Path(dli_data_path)
        self.pagerank_data_path = Path(pagerank_data_path)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建figures目录（在04_dli_analysis根目录下）
        self.figures_dir = Path(__file__).parent / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        logger.info(f"📁 DLI数据路径: {self.dli_data_path}")
        logger.info(f"📁 PageRank数据路径: {self.pagerank_data_path}")
        logger.info(f"📁 输出目录: {self.output_dir}")
        
        # 五个维度列名（用于PCA分析）
        self.original_dimensions = ['continuity', 'infrastructure', 'stability', 'market_locking_power']
        self.pagerank_export_col = 'ppr_us_export_influence' 
        self.pagerank_import_col = 'ppr_influence_on_us'
        
        # 存储权重和分析结果
        self.pca_weights = {}
        self.correlation_matrices = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载原有DLI数据和个性化PageRank数据
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DLI数据和PageRank数据
        """
        logger.info("📂 开始加载数据...")
        
        # 加载原有DLI四维度数据
        if not self.dli_data_path.exists():
            raise FileNotFoundError(f"DLI数据文件不存在: {self.dli_data_path}")
        
        dli_data = pd.read_csv(self.dli_data_path)
        logger.info(f"✅ DLI数据加载完成: {len(dli_data):,}条记录")
        
        # 加载个性化PageRank数据  
        if not self.pagerank_data_path.exists():
            raise FileNotFoundError(f"PageRank数据文件不存在: {self.pagerank_data_path}")
        
        pagerank_data = pd.read_csv(self.pagerank_data_path)
        logger.info(f"✅ PageRank数据加载完成: {len(pagerank_data):,}条记录")
        
        # 数据质量检查
        logger.info("🔍 数据质量检查:")
        logger.info(f"  DLI数据年份范围: {dli_data['year'].min()}-{dli_data['year'].max()}")
        logger.info(f"  PageRank数据年份范围: {pagerank_data['year'].min()}-{pagerank_data['year'].max()}")
        logger.info(f"  DLI数据列: {list(dli_data.columns)}")
        logger.info(f"  PageRank数据列: {list(pagerank_data.columns)}")
        
        return dli_data, pagerank_data
        
    def integrate_pagerank_data(self, dli_data: pd.DataFrame, pagerank_data: pd.DataFrame) -> pd.DataFrame:
        """
        整合个性化PageRank数据到DLI数据中
        
        Args:
            dli_data: 原有DLI四维度数据
            pagerank_data: 个性化PageRank数据
            
        Returns:
            pd.DataFrame: 整合后的五维度数据
        """
        logger.info("🔗 开始整合PageRank数据到DLI数据...")
        
        # 准备合并的键
        # DLI数据使用us_partner字段，PageRank数据使用country_name字段
        pagerank_for_merge = pagerank_data[['year', 'country_name', 
                                          self.pagerank_export_col, 
                                          self.pagerank_import_col]].copy()
        
        pagerank_for_merge = pagerank_for_merge.rename(columns={
            'country_name': 'us_partner'
        })
        
        # 执行左连接合并
        enhanced_data = dli_data.merge(
            pagerank_for_merge,
            on=['year', 'us_partner'],
            how='left'
        )
        
        logger.info(f"🔗 数据合并完成: {len(enhanced_data):,}条记录")
        
        # 检查合并效果
        pagerank_missing = enhanced_data[self.pagerank_export_col].isna().sum()
        total_records = len(enhanced_data)
        missing_rate = pagerank_missing / total_records * 100
        
        logger.info(f"📊 PageRank数据覆盖率: {(100-missing_rate):.1f}% ({total_records-pagerank_missing:,}/{total_records:,})")
        
        if missing_rate > 10:
            logger.warning(f"⚠️  PageRank数据缺失率较高: {missing_rate:.1f}%")
        
        return enhanced_data
        
    def create_directional_datasets(self, enhanced_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建方向性数据集：区分美国出口锁定和进口被锁定
        
        Args:
            enhanced_data: 整合后的五维度数据
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 出口锁定数据, 进口被锁定数据
        """
        logger.info("↔️  创建方向性数据集...")
        
        # 1. 美国出口锁定他国数据 (美国为exporter)
        export_locking_data = enhanced_data[
            enhanced_data['us_role'] == 'exporter'
        ].copy()
        
        # 为出口锁定数据选择相应的PageRank维度
        export_locking_data['pagerank_dimension'] = export_locking_data[self.pagerank_export_col]
        
        # 2. 美国进口被他国锁定数据 (美国为importer)
        import_locking_data = enhanced_data[
            enhanced_data['us_role'] == 'importer'  
        ].copy()
        
        # 为进口锁定数据选择相应的PageRank维度
        import_locking_data['pagerank_dimension'] = import_locking_data[self.pagerank_import_col]
        
        logger.info(f"📤 美国出口锁定数据: {len(export_locking_data):,}条记录")
        logger.info(f"📥 美国进口被锁定数据: {len(import_locking_data):,}条记录")
        
        return export_locking_data, import_locking_data
    
    def diagnose_correlations(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        诊断五个维度之间的相关性
        
        Args:
            data: 包含五个维度的数据
            direction: 方向标识 ('export' 或 'import')
            
        Returns:
            pd.DataFrame: 相关系数矩阵
        """
        logger.info(f"🔍 开始{direction}锁定维度相关性诊断...")
        
        # 准备五个维度数据
        five_dimensions = self.original_dimensions + ['pagerank_dimension']
        
        # 筛选有效数据（去除缺失值）
        valid_data = data[five_dimensions].dropna()
        
        if len(valid_data) == 0:
            raise ValueError(f"{direction}锁定数据中没有完整的五维度观测值")
        
        logger.info(f"  有效观测数: {len(valid_data):,}")
        
        # 计算相关系数矩阵
        correlation_matrix = valid_data.corr()
        
        # 输出相关系数统计
        logger.info(f"  {direction}锁定维度相关性统计:")
        for i, dim1 in enumerate(five_dimensions):
            for j, dim2 in enumerate(five_dimensions):
                if i < j:  # 只输出上三角
                    corr = correlation_matrix.loc[dim1, dim2]
                    logger.info(f"    {dim1} vs {dim2}: {corr:.3f}")
        
        return correlation_matrix
    
    def create_correlation_heatmap(self, export_corr: pd.DataFrame, import_corr: pd.DataFrame):
        """
        创建相关系数矩阵热力图
        
        Args:
            export_corr: 出口锁定相关矩阵
            import_corr: 进口锁定相关矩阵
        """
        logger.info("🎨 创建相关系数矩阵热力图...")
        
        # 设置图形参数
        plt.style.use('default')
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 维度标签（用于显示）
        dimension_labels = [
            'Continuity', 'Infrastructure', 'Stability', 
            'Market Locking', 'PageRank Network'
        ]
        
        # 1. 出口锁定相关矩阵热力图
        ax1 = axes[0]
        sns.heatmap(export_corr, 
                   annot=True, 
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   xticklabels=dimension_labels,
                   yticklabels=dimension_labels,
                   ax=ax1)
        ax1.set_title('US Export Locking Dimensions\nCorrelation Matrix', fontsize=12, pad=20)
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # 2. 进口锁定相关矩阵热力图
        ax2 = axes[1]
        sns.heatmap(import_corr,
                   annot=True,
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   xticklabels=dimension_labels,
                   yticklabels=dimension_labels,
                   ax=ax2)
        ax2.set_title('US Import Locking Dimensions\nCorrelation Matrix', fontsize=12, pad=20)
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # 保存图形到正确的figures目录
        heatmap_path = self.figures_dir / "dli_dimensions_correlation.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"📊 相关矩阵热力图已保存: {heatmap_path}")
        
        plt.close()
        
        # 存储相关矩阵供后续使用
        self.correlation_matrices['export'] = export_corr
        self.correlation_matrices['import'] = import_corr
    
    def calculate_enhanced_dli_with_pca(self, data: pd.DataFrame, direction: str) -> Tuple[pd.DataFrame, Dict]:
        """
        使用PCA计算增强版五维度DLI
        
        Args:
            data: 包含五个维度的数据
            direction: 方向标识 ('export' 或 'import')
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 带有DLI得分的数据, PCA权重信息
        """
        logger.info(f"🎯 计算{direction}锁定增强版DLI...")
        
        enhanced_data = data.copy()
        
        # 准备五个维度数据
        five_dimensions = self.original_dimensions + ['pagerank_dimension']
        
        # 筛选有效数据
        valid_mask = enhanced_data[five_dimensions].notna().all(axis=1)
        valid_data = enhanced_data[valid_mask].copy()
        
        if len(valid_data) == 0:
            raise ValueError(f"{direction}锁定数据中没有完整的五维度观测值")
        
        logger.info(f"  有效观测数: {len(valid_data):,}")
        
        # 提取五维度矩阵进行PCA
        dimensions_matrix = valid_data[five_dimensions].values
        
        # 标准化（PCA前的必要步骤）
        scaler = StandardScaler()
        dimensions_standardized = scaler.fit_transform(dimensions_matrix)
        
        # 执行PCA分析
        pca = PCA(n_components=5)
        pca_scores = pca.fit_transform(dimensions_standardized)
        
        # 提取第一主成分作为综合DLI
        first_pc_scores = pca_scores[:, 0]
        
        # 获取权重（第一主成分的载荷）
        first_pc_loadings = pca.components_[0]
        
        # 创建权重字典
        pca_weights = {
            'dimensions': five_dimensions,
            'loadings': first_pc_loadings.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'first_pc_variance_explained': float(pca.explained_variance_ratio_[0])
        }
        
        # 将权重与维度名配对
        dimension_weights = {
            dim: float(weight) for dim, weight in zip(five_dimensions, first_pc_loadings)
        }
        pca_weights['dimension_weights'] = dimension_weights
        
        logger.info(f"  {direction}锁定PCA分析结果:")
        logger.info(f"    第一主成分解释方差比: {pca_weights['first_pc_variance_explained']:.3f}")
        logger.info(f"    五个维度权重:")
        for dim, weight in dimension_weights.items():
            logger.info(f"      {dim}: {weight:.3f}")
        
        # 将PCA得分添加到数据中
        valid_data[f'dli_enhanced_{direction}'] = first_pc_scores
        
        # 为所有数据分配DLI得分（包括缺失值的观测）
        enhanced_data[f'dli_enhanced_{direction}'] = np.nan
        enhanced_data.loc[valid_mask, f'dli_enhanced_{direction}'] = first_pc_scores
        
        # 存储权重信息
        self.pca_weights[direction] = pca_weights
        
        logger.info(f"✅ {direction}锁定增强版DLI计算完成")
        
        return enhanced_data, pca_weights
    
    def combine_directional_results(self, export_data: pd.DataFrame, import_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并双向DLI结果
        
        Args:
            export_data: 出口锁定DLI数据
            import_data: 进口锁定DLI数据
            
        Returns:
            pd.DataFrame: 合并后的完整增强版DLI数据
        """
        logger.info("🔄 合并双向DLI结果...")
        
        # 合并数据
        combined_data = pd.concat([export_data, import_data], ignore_index=True)
        
        # 创建统一的增强版DLI列
        combined_data['dli_enhanced'] = combined_data['dli_enhanced_export'].fillna(
            combined_data['dli_enhanced_import']
        )
        
        # 添加元信息列
        combined_data['calculation_method'] = 'Enhanced_5D_PCA'
        combined_data['pagerank_integrated'] = True
        combined_data['analysis_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"🎯 双向DLI合并完成: {len(combined_data):,}条记录")
        
        # 输出基本统计
        export_count = len(combined_data[combined_data['us_role'] == 'exporter'])
        import_count = len(combined_data[combined_data['us_role'] == 'importer'])
        
        logger.info(f"  美国出口锁定记录: {export_count:,}")
        logger.info(f"  美国进口被锁定记录: {import_count:,}")
        
        return combined_data
    
    def save_results(self, enhanced_data: pd.DataFrame) -> Tuple[Path, Path]:
        """
        保存增强版DLI结果
        
        Args:
            enhanced_data: 增强版DLI数据
            
        Returns:
            Tuple[Path, Path]: CSV文件路径, JSON权重文件路径
        """
        logger.info("💾 保存增强版DLI结果...")
        
        # 1. 保存增强版DLI面板数据
        csv_path = self.output_dir / "dli_pagerank.csv"
        enhanced_data.to_csv(csv_path, index=False)
        logger.info(f"📊 增强版DLI数据已保存: {csv_path}")
        
        # 2. 保存权重和分析参数
        weights_and_params = {
            'analysis_metadata': {
                'calculation_method': 'Enhanced_5D_PCA',
                'dimensions_count': 5,
                'original_dli_dimensions': self.original_dimensions,
                'pagerank_dimensions': [self.pagerank_export_col, self.pagerank_import_col],
                'analysis_timestamp': datetime.now().isoformat(),
                'total_records': len(enhanced_data)
            },
            'export_dli_weights': self.pca_weights.get('export', {}),
            'import_dli_weights': self.pca_weights.get('import', {}),
            'correlation_analysis': {
                'export_correlation_summary': self._summarize_correlation(
                    self.correlation_matrices.get('export')
                ),
                'import_correlation_summary': self._summarize_correlation(
                    self.correlation_matrices.get('import')
                )
            }
        }
        
        json_path = self.output_dir / "dli_pagerank_weights.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(weights_and_params, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📈 权重和参数已保存: {json_path}")
        
        return csv_path, json_path
    
    def _summarize_correlation(self, corr_matrix: pd.DataFrame) -> Dict:
        """
        总结相关矩阵的关键统计信息
        
        Args:
            corr_matrix: 相关系数矩阵
            
        Returns:
            Dict: 相关性摘要统计
        """
        if corr_matrix is None:
            return {'error': '相关矩阵不存在'}
        
        # 提取上三角相关系数（排除对角线）
        upper_triangle = []
        n = len(corr_matrix)
        for i in range(n):
            for j in range(i+1, n):
                upper_triangle.append(corr_matrix.iloc[i, j])
        
        upper_triangle = np.array(upper_triangle)
        
        return {
            'mean_correlation': float(np.mean(upper_triangle)),
            'max_correlation': float(np.max(upper_triangle)),
            'min_correlation': float(np.min(upper_triangle)),
            'std_correlation': float(np.std(upper_triangle)),
            'high_correlation_pairs': self._find_high_correlation_pairs(corr_matrix)
        }
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """
        找出高相关性的维度对
        
        Args:
            corr_matrix: 相关矩阵
            threshold: 高相关性阈值
            
        Returns:
            List[Dict]: 高相关性维度对列表
        """
        high_corr_pairs = []
        n = len(corr_matrix)
        
        for i in range(n):
            for j in range(i+1, n):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'dimension_1': corr_matrix.index[i],
                        'dimension_2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        return high_corr_pairs
    
    def generate_summary_report(self, enhanced_data: pd.DataFrame) -> str:
        """
        生成增强版DLI分析摘要报告
        
        Args:
            enhanced_data: 增强版DLI数据
            
        Returns:
            str: 报告内容
        """
        report = []
        report.append("# 增强版动态锁定指数(Enhanced DLI)分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## 1. 数据概览")
        report.append(f"- 总记录数: {len(enhanced_data):,}")
        report.append(f"- 年份范围: {enhanced_data['year'].min()}-{enhanced_data['year'].max()}")
        report.append(f"- 覆盖国家: {enhanced_data['us_partner'].nunique()}")
        report.append(f"- 能源产品: {enhanced_data['energy_product'].nunique()}")
        report.append("")
        
        report.append("## 2. 方法论创新")
        report.append("- 五维度分析框架: 传统4维 + 个性化PageRank网络维度")
        report.append("- 严格方向性区分: 美国出口锁定 vs 美国进口被锁定")  
        report.append("- PCA处理多重共线性: 数据驱动的权重确定")
        report.append("- 学术规范诊断: 相关性分析和可视化")
        report.append("")
        
        # 添加权重分析
        if 'export' in self.pca_weights and 'import' in self.pca_weights:
            report.append("## 3. PCA权重分析")
            
            export_weights = self.pca_weights['export']['dimension_weights']
            import_weights = self.pca_weights['import']['dimension_weights']
            
            report.append("### 美国出口锁定权重:")
            for dim, weight in export_weights.items():
                report.append(f"- {dim}: {weight:.3f}")
            
            report.append("\n### 美国进口被锁定权重:")
            for dim, weight in import_weights.items():
                report.append(f"- {dim}: {weight:.3f}")
            
            report.append(f"\n- 出口锁定第一主成分解释方差: {self.pca_weights['export']['first_pc_variance_explained']:.3f}")
            report.append(f"- 进口锁定第一主成分解释方差: {self.pca_weights['import']['first_pc_variance_explained']:.3f}")
        
        report.append("\n## 4. 核心发现")
        
        # 统计分析
        export_data = enhanced_data[enhanced_data['us_role'] == 'exporter']
        import_data = enhanced_data[enhanced_data['us_role'] == 'importer']
        
        if len(export_data) > 0:
            export_mean = export_data['dli_enhanced'].mean()
            report.append(f"- 美国出口锁定平均水平: {export_mean:.3f}")
        
        if len(import_data) > 0:
            import_mean = import_data['dli_enhanced'].mean()
            report.append(f"- 美国进口被锁定平均水平: {import_mean:.3f}")
        
        report.append("\n## 5. 文件输出")
        report.append("- dli_pagerank.csv: 增强版DLI面板数据")
        report.append("- dli_pagerank_weights.json: PCA权重和分析参数")
        report.append("- dli_dimensions_correlation.png: 维度相关性热力图")
        
        return "\n".join(report)
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Path, Path]:
        """
        运行完整的增强版DLI分析流程
        
        Returns:
            Tuple[pd.DataFrame, Path, Path]: 增强版数据, CSV路径, JSON路径
        """
        logger.info("=" * 60)
        logger.info("🌟 增强版DLI计算系统启动")
        logger.info("=" * 60)
        
        try:
            # 1. 数据加载
            dli_data, pagerank_data = self.load_data()
            
            # 2. 数据整合
            enhanced_data = self.integrate_pagerank_data(dli_data, pagerank_data)
            
            # 3. 创建方向性数据集
            export_data, import_data = self.create_directional_datasets(enhanced_data)
            
            # 4. 相关性诊断分析
            logger.info("📊 执行学术规范诊断分析...")
            export_corr = self.diagnose_correlations(export_data, 'export')
            import_corr = self.diagnose_correlations(import_data, 'import')
            
            # 5. 创建相关系数矩阵热力图
            self.create_correlation_heatmap(export_corr, import_corr)
            
            # 6. 计算增强版DLI（分别为两个方向）
            export_enhanced, export_weights = self.calculate_enhanced_dli_with_pca(export_data, 'export')
            import_enhanced, import_weights = self.calculate_enhanced_dli_with_pca(import_data, 'import')
            
            # 7. 合并双向结果
            final_enhanced_data = self.combine_directional_results(export_enhanced, import_enhanced)
            
            # 8. 保存结果
            csv_path, json_path = self.save_results(final_enhanced_data)
            
            # 9. 生成摘要报告
            logger.info("📄 生成分析摘要报告...")
            report_content = self.generate_summary_report(final_enhanced_data)
            report_path = self.output_dir / "enhanced_dli_analysis_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"📝 摘要报告已保存: {report_path}")
            
            # 10. 输出完成信息
            logger.info("=" * 60)
            logger.info("🎉 增强版DLI分析完成!")
            logger.info("=" * 60)
            logger.info(f"📊 总记录数: {len(final_enhanced_data):,}")
            logger.info(f"📅 覆盖年份: {final_enhanced_data['year'].min()}-{final_enhanced_data['year'].max()}")
            logger.info(f"🌍 覆盖国家: {final_enhanced_data['us_partner'].nunique()}")
            logger.info(f"📁 主要输出文件:")
            logger.info(f"  • {csv_path.name}")
            logger.info(f"  • {json_path.name}")
            logger.info(f"  • dli_dimensions_correlation.png")
            logger.info(f"  • enhanced_dli_analysis_report.md")
            
            return final_enhanced_data, csv_path, json_path
            
        except Exception as e:
            logger.error(f"❌ 分析过程失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强版动态锁定指数计算系统 v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python dli_pagerank.py                                                        # 使用默认路径
  python dli_pagerank.py --dli-data ./outputs/dli_panel_data.csv               # 指定DLI数据
  python dli_pagerank.py --pagerank-data ../03_metrics/outputs/personalized_pagerank_panel.csv  # 指定PageRank数据
  python dli_pagerank.py --output-dir ./enhanced_outputs                       # 指定输出目录
        """
    )
    
    # 设置默认路径
    current_dir = Path(__file__).parent
    default_dli_data = current_dir / "outputs" / "dli_panel_data.csv"
    default_pagerank_data = current_dir.parent / "03_metrics" / "outputs" / "personalized_pagerank_panel.csv" 
    default_output_dir = current_dir / "outputs"
    
    parser.add_argument(
        '--dli-data',
        type=str,
        default=str(default_dli_data),
        help=f'DLI四维度数据文件路径 (默认: {default_dli_data})'
    )
    
    parser.add_argument(
        '--pagerank-data', 
        type=str,
        default=str(default_pagerank_data),
        help=f'个性化PageRank数据文件路径 (默认: {default_pagerank_data})'
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
        # 创建增强版DLI计算器
        calculator = EnhancedDLICalculator(
            dli_data_path=Path(args.dli_data),
            pagerank_data_path=Path(args.pagerank_data),
            output_dir=Path(args.output_dir)
        )
        
        # 运行完整分析
        enhanced_data, csv_path, json_path = calculator.run_full_analysis()
        
        print(f"\n✅ 增强版DLI计算成功完成!")
        print(f"📊 增强版数据文件: {csv_path}")
        print(f"⚖️  权重参数文件: {json_path}")
        print(f"📈 相关性热力图: dli_dimensions_correlation.png")
        
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