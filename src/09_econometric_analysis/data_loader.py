#!/usr/bin/env python3
"""
数据加载模块 (Data Loader Module)
============================================

09_econometric_analysis 模块的数据加载组件

作者：Energy Network Analysis Team
版本：v1.0 - 计量分析框架
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class DataLoader:
    """
    数据加载器 - 专门处理空数据和缺失文件的健壮加载逻辑
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            project_root: 项目根目录，如果为None则自动推断
        """
        if project_root is None:
            # 从当前文件向上追溯找到项目根目录
            self.project_root = Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        # 定义数据文件路径
        self.analytical_panel_path = self.project_root / "data" / "processed_data" / "analytical_panel.csv"
        
        logger.info(f"数据加载器初始化完成")
        logger.info(f"项目根目录: {self.project_root}")
        logger.info(f"分析面板路径: {self.analytical_panel_path}")
        
        # 定义预期的列名（基于08模块的输出规范）
        self.expected_columns = self._get_expected_columns()
    
    def _get_expected_columns(self) -> List[str]:
        """
        定义预期的数据列名
        
        Returns:
            预期的列名列表
        """
        return [
            # 基础标识变量
            'year', 'country', 'country_name',
            
            # 宏观经济控制变量
            'gdp_current_usd', 'population_total', 'trade_openness_gdp_pct',
            'log_gdp', 'log_population',
            
            # 核心研究变量（来自08模块）
            'node_dli_us',         # Node-DLI_US: 美国锚定动态锁定指数
            'vul_us',              # Vul_US: 美国锚定脆弱性指数
            'ovi',                 # OVI: 物理冗余指数
            'us_prod_shock',       # US_ProdShock: 美国产量冲击
            
            # 网络拓扑指标（来自03模块）
            'betweenness_centrality', 'eigenvector_centrality',
            'in_degree', 'out_degree', 'total_degree',
            'in_strength', 'out_strength', 'total_strength',
            'pagerank_centrality',
            
            # 辅助变量
            'import_share_from_us', 'us_import_share', 'hhi_imports',
            'lng_capacity', 'pipeline_capacity', 'energy_demand',
            'us_production_oil', 'us_production_gas'
        ]
    
    def load_analytical_panel(self) -> pd.DataFrame:
        """
        加载分析面板数据 - 核心功能，必须能处理空数据情况
        
        Returns:
            分析面板DataFrame，如果数据缺失则返回空但结构正确的DataFrame
        """
        logger.info("🔍 开始加载分析面板数据...")
        
        # 情况1: 文件不存在
        if not self.analytical_panel_path.exists():
            logger.warning(f"⚠️ 分析面板文件不存在: {self.analytical_panel_path}")
            logger.info("   创建空的DataFrame框架...")
            return self._create_empty_dataframe()
        
        try:
            # 情况2: 文件存在，尝试加载
            logger.info(f"   从文件加载: {self.analytical_panel_path}")
            df = pd.read_csv(self.analytical_panel_path)
            
            # 情况3: 文件存在但为空
            if len(df) == 0:
                logger.warning("⚠️ 分析面板文件为空")
                logger.info("   创建空的DataFrame框架...")
                return self._create_empty_dataframe()
            
            # 情况4: 文件存在有数据但所有关键变量都是NaN
            key_variables = ['node_dli_us', 'vul_us', 'ovi', 'us_prod_shock']
            all_key_vars_missing = all(
                col not in df.columns or df[col].isna().all() 
                for col in key_variables
            )
            
            if all_key_vars_missing:
                logger.warning("⚠️ 所有关键研究变量都缺失或为NaN")
                logger.info(f"   数据形状: {df.shape}，但关键变量不可用")
            else:
                logger.info(f"✅ 成功加载分析面板数据")
                logger.info(f"   数据形状: {df.shape}")
                
                # 检查关键变量的可用性
                available_vars = [col for col in key_variables if col in df.columns and not df[col].isna().all()]
                if available_vars:
                    logger.info(f"   可用关键变量: {available_vars}")
                else:
                    logger.warning("   没有可用的关键变量数据")
            
            # 确保DataFrame包含所有预期列
            df = self._ensure_expected_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 加载分析面板数据失败: {str(e)}")
            logger.info("   创建空的DataFrame框架...")
            return self._create_empty_dataframe()
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """
        创建空但结构正确的DataFrame
        
        Returns:
            包含所有预期列但无数据的DataFrame
        """
        logger.info("   构建空DataFrame框架...")
        
        # 创建空DataFrame但包含所有预期列
        df = pd.DataFrame(columns=self.expected_columns)
        
        # 设置正确的数据类型
        type_mapping = {
            'year': 'int64',
            'country': 'str',
            'country_name': 'str',
            'gdp_current_usd': 'float64',
            'population_total': 'float64',
            'trade_openness_gdp_pct': 'float64',
            'log_gdp': 'float64',
            'log_population': 'float64',
            'node_dli_us': 'float64',
            'vul_us': 'float64',
            'ovi': 'float64',
            'us_prod_shock': 'float64'
        }
        
        for col, dtype in type_mapping.items():
            if col in df.columns:
                if dtype == 'str':
                    df[col] = df[col].astype('object')
                else:
                    df[col] = df[col].astype(dtype)
        
        logger.info(f"   空DataFrame框架创建完成: {len(self.expected_columns)} 列")
        return df
    
    def _ensure_expected_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保DataFrame包含所有预期列
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含所有预期列的DataFrame
        """
        missing_cols = set(self.expected_columns) - set(df.columns)
        
        if missing_cols:
            logger.info(f"   添加缺失列: {sorted(missing_cols)}")
            for col in missing_cols:
                df[col] = np.nan
        
        # 重新排序列以匹配预期顺序
        available_expected_cols = [col for col in self.expected_columns if col in df.columns]
        other_cols = [col for col in df.columns if col not in self.expected_columns]
        df = df[available_expected_cols + other_cols]
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        获取数据摘要统计
        
        Args:
            df: 输入DataFrame
            
        Returns:
            数据摘要字典
        """
        if len(df) == 0:
            return {
                'total_rows': 0,
                'total_cols': len(df.columns),
                'year_range': 'N/A',
                'countries': 0,
                'key_variables_available': [],
                'data_status': 'empty'
            }
        
        key_variables = ['node_dli_us', 'vul_us', 'ovi', 'us_prod_shock']
        available_key_vars = []
        
        for var in key_variables:
            if var in df.columns and not df[var].isna().all():
                non_missing_count = df[var].notna().sum()
                available_key_vars.append(f"{var} ({non_missing_count}/{len(df)})")
        
        summary = {
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'year_range': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A',
            'countries': df['country'].nunique() if 'country' in df.columns else 0,
            'key_variables_available': available_key_vars,
            'data_status': 'available' if available_key_vars else 'missing_key_vars'
        }
        
        return summary
    
    def validate_data_for_analysis(self, df: pd.DataFrame) -> Dict:
        """
        验证数据是否适合进行计量分析
        
        Args:
            df: 输入DataFrame
            
        Returns:
            验证结果字典
        """
        validation_results = {
            'is_valid_for_analysis': False,
            'issues': [],
            'recommendations': []
        }
        
        # 检查1: 数据是否为空
        if len(df) == 0:
            validation_results['issues'].append("数据集为空")
            validation_results['recommendations'].append("等待08模块生成数据")
            return validation_results
        
        # 检查2: 关键变量是否可用
        key_variables = ['node_dli_us', 'vul_us', 'ovi', 'us_prod_shock']
        missing_key_vars = []
        
        for var in key_variables:
            if var not in df.columns or df[var].isna().all():
                missing_key_vars.append(var)
        
        if missing_key_vars:
            validation_results['issues'].append(f"关键变量缺失: {missing_key_vars}")
            validation_results['recommendations'].append("检查08模块数据构建状态")
        
        # 检查3: 控制变量是否可用
        control_variables = ['log_gdp', 'log_population', 'trade_openness_gdp_pct']
        missing_controls = []
        
        for var in control_variables:
            if var not in df.columns or df[var].isna().all():
                missing_controls.append(var)
        
        if missing_controls:
            validation_results['issues'].append(f"控制变量缺失: {missing_controls}")
        
        # 检查4: 面板数据结构
        if 'year' not in df.columns or 'country' not in df.columns:
            validation_results['issues'].append("缺少面板数据必需的年份或国家标识")
        elif len(df) > 0:
            year_count = df['year'].nunique()
            country_count = df['country'].nunique()
            if year_count < 2:
                validation_results['issues'].append("年份维度不足，无法进行面板分析")
            if country_count < 2:
                validation_results['issues'].append("国家维度不足，无法进行面板分析")
        
        # 最终判断
        validation_results['is_valid_for_analysis'] = len(validation_results['issues']) == 0
        
        if not validation_results['is_valid_for_analysis']:
            validation_results['recommendations'].append("当前数据不适合计量分析，建议等待数据完善")
        
        return validation_results


def load_data() -> pd.DataFrame:
    """
    便捷函数：加载分析数据
    
    Returns:
        分析面板DataFrame
    """
    loader = DataLoader()
    return loader.load_analytical_panel()


def get_data_status() -> Dict:
    """
    便捷函数：获取数据状态
    
    Returns:
        数据状态摘要
    """
    loader = DataLoader()
    df = loader.load_analytical_panel()
    summary = loader.get_data_summary(df)
    validation = loader.validate_data_for_analysis(df)
    
    return {
        'summary': summary,
        'validation': validation
    }


if __name__ == "__main__":
    # 测试数据加载功能
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🔬 09_econometric_analysis 数据加载器测试")
    print("=" * 50)
    
    # 测试数据加载
    loader = DataLoader()
    df = loader.load_analytical_panel()
    
    print(f"\n📊 数据加载结果:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 数据摘要
    summary = loader.get_data_summary(df)
    print(f"\n📈 数据摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 数据验证
    validation = loader.validate_data_for_analysis(df)
    print(f"\n✅ 数据验证:")
    print(f"  适合分析: {validation['is_valid_for_analysis']}")
    if validation['issues']:
        print(f"  问题: {validation['issues']}")
    if validation['recommendations']:
        print(f"  建议: {validation['recommendations']}")
    
    print("\n🎉 数据加载器测试完成!")