#!/usr/bin/env python3
"""
数据加载与预处理模块 (Data Loader & Preprocessing)
================================================

本模块负责整合来自项目其他模块的数据，为异质性分析提供统一的数据接口。
整合数据包括：DLI效应指标、全局/局部网络指标、因果分析基准数据。

作者：Energy Network Analysis Team
版本：v1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import pickle
import json
from typing import Dict, Tuple, Optional, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 项目根路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"


class HeterogeneityDataLoader:
    """网络结构异质性分析数据加载器"""
    
    def __init__(self):
        self.dli_data = None
        self.global_metrics = None
        self.local_metrics = None
        self.causal_data = None
        logger.info("🚀 初始化异质性分析数据加载器")
    
    def load_dli_data(self) -> pd.DataFrame:
        """
        加载DLI效应指标数据
        
        Returns:
            包含DLI指标的DataFrame
        """
        logger.info("📊 加载DLI效应指标数据...")
        
        dli_paths = [
            SRC_ROOT / "04_dli_analysis" / "dli_panel_data.csv",
            SRC_ROOT / "04_dli_analysis" / "outputs" / "dli_panel_data.csv"
        ]
        
        for path in dli_paths:
            if path.exists():
                self.dli_data = pd.read_csv(path)
                logger.info(f"✅ 成功加载DLI数据: {len(self.dli_data)} 行")
                return self.dli_data
        
        # 如果找不到文件，生成示例数据
        logger.warning("⚠️ 未找到DLI数据文件，生成示例数据")
        self.dli_data = self._generate_demo_dli_data()
        return self.dli_data
    
    def load_global_metrics(self) -> pd.DataFrame:
        """
        加载全局网络指标数据
        
        Returns:
            包含全局网络指标的DataFrame
        """
        logger.info("🌐 加载全局网络指标数据...")
        
        metrics_paths = [
            SRC_ROOT / "03_metrics" / "global_network_metrics.csv",
            SRC_ROOT / "03_metrics" / "all_metrics.csv"
        ]
        
        for path in metrics_paths:
            if path.exists():
                df = pd.read_csv(path)
                # 筛选全局指标
                global_cols = [col for col in df.columns if 'global_' in col or 'network_' in col]
                if 'year' in df.columns:
                    global_cols.append('year')
                
                if global_cols:
                    self.global_metrics = df[global_cols].drop_duplicates(subset=['year'] if 'year' in global_cols else None)
                    logger.info(f"✅ 成功加载全局指标数据: {len(self.global_metrics)} 行, {len(global_cols)} 列")
                    return self.global_metrics
        
        # 生成示例数据
        logger.warning("⚠️ 未找到全局指标数据，生成示例数据")
        self.global_metrics = self._generate_demo_global_metrics()
        return self.global_metrics
    
    def load_local_metrics(self) -> pd.DataFrame:
        """
        加载局部节点指标数据
        
        Returns:
            包含节点中心性指标的DataFrame
        """
        logger.info("🏠 加载局部节点指标数据...")
        
        metrics_paths = [
            SRC_ROOT / "03_metrics" / "node_centrality_metrics.csv",
            SRC_ROOT / "03_metrics" / "all_metrics.csv"
        ]
        
        for path in metrics_paths:
            if path.exists():
                df = pd.read_csv(path)
                # 筛选节点指标
                if 'country_code' in df.columns:
                    required_cols = ['year', 'country_code']
                    centrality_cols = [col for col in df.columns if any(x in col for x in 
                                     ['degree', 'centrality', 'strength', 'pagerank'])]
                    
                    node_cols = required_cols + centrality_cols
                    available_cols = [col for col in node_cols if col in df.columns]
                    
                    if len(available_cols) > 2:  # 至少有year, country_code和一个指标
                        self.local_metrics = df[available_cols]
                        logger.info(f"✅ 成功加载局部指标数据: {len(self.local_metrics)} 行")
                        return self.local_metrics
        
        # 生成示例数据
        logger.warning("⚠️ 未找到局部指标数据，生成示例数据")
        self.local_metrics = self._generate_demo_local_metrics()
        return self.local_metrics
    
    def load_causal_data(self) -> pd.DataFrame:
        """
        加载因果分析基准数据
        
        Returns:
            包含因果分析变量的DataFrame
        """
        logger.info("🔗 加载因果分析基准数据...")
        
        causal_paths = [
            SRC_ROOT / "05_causal_validation" / "outputs" / "network_resilience.csv",
            SRC_ROOT / "05_causal_validation" / "network_resilience.csv"
        ]
        
        for path in causal_paths:
            if path.exists():
                self.causal_data = pd.read_csv(path)
                logger.info(f"✅ 成功加载因果分析数据: {len(self.causal_data)} 行")
                return self.causal_data
        
        # 生成示例数据
        logger.warning("⚠️ 未找到因果分析数据，生成示例数据")
        self.causal_data = self._generate_demo_causal_data()
        return self.causal_data
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有必要数据
        
        Returns:
            包含所有数据集的字典
        """
        logger.info("📦 开始加载所有数据...")
        
        data = {
            'dli': self.load_dli_data(),
            'global_metrics': self.load_global_metrics(),
            'local_metrics': self.load_local_metrics(),
            'causal': self.load_causal_data()
        }
        
        logger.info("✅ 所有数据加载完成")
        return data
    
    def create_analysis_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建分析数据集，整合所有数据源
        
        Returns:
            (global_dataset, local_dataset): 全局分析和局部分析数据集
        """
        logger.info("🔧 构建分析数据集...")
        
        # 加载所有数据
        data = self.load_all_data()
        
        # 构建全局分析数据集
        global_dataset = self._build_global_dataset(data)
        
        # 构建局部分析数据集  
        local_dataset = self._build_local_dataset(data)
        
        logger.info(f"✅ 分析数据集构建完成:")
        logger.info(f"   - 全局分析数据集: {len(global_dataset)} 行")
        logger.info(f"   - 局部分析数据集: {len(local_dataset)} 行")
        
        return global_dataset, local_dataset
    
    def _build_global_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """构建全局分析数据集 - 分析单位：国家-年份"""
        # 从因果分析数据开始（已经是国家-年份格式）
        global_df = data['causal'].copy()
        
        # 处理DLI数据：从双边关系聚合到国家层面
        if 'year' in data['dli'].columns:
            # 为美国构建国家层面的DLI指标
            if 'us_partner' in data['dli'].columns:
                # 按年份和美国角色聚合DLI（进口锁定+出口锁定）
                dli_country_level = data['dli'].groupby(['year', 'us_role']).agg({
                    'dli_score': ['mean', 'sum', 'std'],
                    'trade_value_usd': 'sum'
                }).reset_index()
                
                # 重命名列
                dli_country_level.columns = ['year', 'us_role', 'dli_mean', 'dli_total', 'dli_volatility', 'total_trade_value']
                
                # 透视转换：进口锁定和出口锁定分开
                dli_pivot = dli_country_level.pivot(index='year', columns='us_role', 
                                                   values=['dli_mean', 'dli_total', 'dli_volatility'])
                dli_pivot.columns = [f'{metric}_{role}' for metric, role in dli_pivot.columns]
                dli_pivot = dli_pivot.reset_index()
                
                # 计算综合锁定指标
                if 'dli_mean_importer' in dli_pivot.columns and 'dli_mean_exporter' in dli_pivot.columns:
                    dli_pivot['dli_composite'] = (dli_pivot['dli_mean_importer'] + dli_pivot['dli_mean_exporter']) / 2
                
                # 合并到全局数据集
                if 'year' in global_df.columns:
                    global_df = pd.merge(global_df, dli_pivot, on='year', how='left')
        
        # 合并全局网络指标
        if 'year' in global_df.columns and 'year' in data['global_metrics'].columns:
            global_df = pd.merge(global_df, data['global_metrics'], on='year', how='left')
        
        return global_df
    
    def _build_local_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """构建局部分析数据集 - 分析单位：国家-年份"""
        # 从因果分析数据开始（已经是国家-年份格式）
        local_df = data['causal'].copy()
        
        # 处理DLI数据：从双边关系聚合到国家层面
        if 'year' in data['dli'].columns and 'us_partner' in data['dli'].columns:
            # 计算每个国家作为美国伙伴时的DLI暴露度
            dli_partner_level = data['dli'].groupby(['year', 'us_partner', 'us_role']).agg({
                'dli_score': 'mean',
                'trade_value_usd': 'sum'
            }).reset_index()
            
            # 透视：分别计算该国从美国进口和向美国出口的锁定度
            dli_partner_pivot = dli_partner_level.pivot_table(
                index=['year', 'us_partner'], 
                columns='us_role',
                values='dli_score',
                aggfunc='mean'
            ).reset_index()
            dli_partner_pivot.columns.name = None
            
            # 重命名列
            col_mapping = {}
            for col in dli_partner_pivot.columns:
                if col == 'exporter':  # 美国作为出口商，伙伴国被锁定为进口商
                    col_mapping[col] = 'partner_import_locking'
                elif col == 'importer':  # 美国作为进口商，伙伴国被锁定为出口商  
                    col_mapping[col] = 'partner_export_locking'
            dli_partner_pivot.rename(columns=col_mapping, inplace=True)
            
            # 计算综合锁定指数
            if 'partner_import_locking' in dli_partner_pivot.columns and 'partner_export_locking' in dli_partner_pivot.columns:
                dli_partner_pivot['partner_total_locking'] = (
                    dli_partner_pivot['partner_import_locking'].fillna(0) + 
                    dli_partner_pivot['partner_export_locking'].fillna(0)
                ) / 2
            
            # 合并到局部数据集（匹配country字段与us_partner）
            if 'country' in local_df.columns:
                local_df = pd.merge(local_df, dli_partner_pivot, 
                                  left_on=['year', 'country'], 
                                  right_on=['year', 'us_partner'], 
                                  how='left')
        
        # 合并局部节点指标（按年份和国家精确匹配）
        if 'year' in local_df.columns and 'country' in local_df.columns:
            if 'year' in data['local_metrics'].columns and 'country_code' in data['local_metrics'].columns:
                local_df = pd.merge(local_df, data['local_metrics'], 
                                  left_on=['year', 'country'], 
                                  right_on=['year', 'country_code'], 
                                  how='left')
        
        return local_df
    
    def _generate_demo_dli_data(self) -> pd.DataFrame:
        """生成DLI示例数据"""
        years = range(2010, 2025)
        countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR']
        
        data = []
        for year in years:
            for country in countries:
                data.append({
                    'year': year,
                    'country_code': country,
                    'dli_import': np.random.normal(0.5, 0.2),
                    'dli_export': np.random.normal(0.3, 0.15),
                    'dli_composite': np.random.normal(0.4, 0.18)
                })
        
        return pd.DataFrame(data)
    
    def _generate_demo_global_metrics(self) -> pd.DataFrame:
        """生成全局指标示例数据"""
        years = range(2010, 2025)
        
        data = []
        for year in years:
            data.append({
                'year': year,
                'global_density': np.random.normal(0.3, 0.1),
                'global_transitivity': np.random.normal(0.6, 0.15),
                'global_avg_clustering': np.random.normal(0.7, 0.12),
                'global_efficiency': np.random.normal(0.8, 0.1),
                'network_size': np.random.randint(50, 100)
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_local_metrics(self) -> pd.DataFrame:
        """生成局部指标示例数据"""
        years = range(2010, 2025)
        countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'ITA', 'RUS', 'SAU', 'CAN']
        
        data = []
        for year in years:
            for country in countries:
                data.append({
                    'year': year,
                    'country_code': country,
                    'betweenness_centrality': np.random.exponential(0.1),
                    'degree_centrality': np.random.beta(2, 5),
                    'pagerank_centrality': np.random.gamma(2, 0.1),
                    'in_strength': np.random.lognormal(5, 1),
                    'out_strength': np.random.lognormal(5, 1)
                })
        
        return pd.DataFrame(data)
    
    def _generate_demo_causal_data(self) -> pd.DataFrame:
        """生成因果分析示例数据"""
        years = range(2010, 2025)
        countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR']
        
        data = []
        for year in years:
            for country in countries:
                data.append({
                    'year': year,
                    'country': country,  # 修改为country以匹配05模块的格式
                    'comprehensive_resilience': np.random.normal(0.7, 0.2),
                    'dli_composite': np.random.normal(0.1, 0.05),  # 添加dli_composite
                    'control_var1': np.random.normal(0, 1),
                    'control_var2': np.random.exponential(1)
                })
        
        return pd.DataFrame(data)


def main():
    """测试数据加载功能"""
    loader = HeterogeneityDataLoader()
    
    # 测试加载所有数据
    global_data, local_data = loader.create_analysis_dataset()
    
    print("🎯 全局分析数据集预览:")
    print(global_data.head())
    print(f"\n列名: {list(global_data.columns)}")
    
    print("\n🎯 局部分析数据集预览:")
    print(local_data.head())
    print(f"\n列名: {list(local_data.columns)}")


if __name__ == "__main__":
    main()