"""
出口DLI计算器 (Export DLI Calculator)
====================================

本模块专门负责计算美国对其他国家的出口锁定DLI指数。
从双向DLI分析中剥离出来，提供独立的出口锁定分析功能。

理论框架：
- 出口锁定DLI衡量美国通过能源出口对其他国家产生的"锁定"效应
- 当美国向某国出口能源时，评估该国对美国的"被锁定"程度
- 核心逻辑：目标国进口集中度 × 美国在目标国市场份额

功能：
1. 计算出口锁定力指标 (Export Locking Power)
2. 计算出口方向的持续性、基础设施、稳定性指标
3. 合成出口DLI综合指标
4. 生成独立的出口DLI数据文件

Author: Energy Network Analysis Team
Date: 2025-08-22
Version: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 导入必要的基础计算函数
from dli_calculator import calculate_continuity, calculate_infrastructure, calculate_stability

logger = logging.getLogger(__name__)

class ExportDLICalculator:
    """美国出口锁定DLI计算器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化出口DLI计算器
        
        Args:
            output_dir: 输出目录路径，默认为当前目录
        """
        self.output_dir = output_dir or Path(__file__).parent
        self.output_dir.mkdir(exist_ok=True)
        
    def calculate_export_locking_power(self, df: pd.DataFrame, global_trade_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        计算出口锁定力指标 (Export Locking Power) - 核心算法
        
        理论框架：当美国向某国出口能源时，评估该国对美国的"被锁定"程度
        
        计算逻辑：
        1. 对于美国向国家X出口产品P的每一条记录
        2. 查询全球数据，找到国家X在该年份进口产品P的所有供应商
        3. 计算国家X在产品P上的进口集中度（供应商HHI）
        4. 计算美国在国家X的产品P进口中的份额
        5. 出口锁定力 = 国家X的进口HHI × 美国在X国市场的份额
        
        Args:
            df: 包含美国贸易数据的DataFrame
            global_trade_data: 全球贸易数据字典，格式{year: DataFrame}
            
        Returns:
            添加了market_locking_power列的DataFrame（只计算出口部分）
        """
        
        logger.info("📤 开始计算出口锁定力指标（独立模块）...")
        
        df_locking = df.copy()
        
        # 只处理美国作为出口方的数据
        export_data = df_locking[df_locking['us_role'] == 'exporter'].copy()
        
        if len(export_data) == 0:
            logger.warning("没有找到美国出口数据，返回原数据")
            return df_locking
        
        if not global_trade_data:
            logger.warning("未提供全球贸易数据，出口锁定力将设为0")
            df_locking.loc[df_locking['us_role'] == 'exporter', 'market_locking_power'] = 0
            return df_locking
        
        locking_results = []
        
        # 为每个美国出口记录计算对应的出口锁定力
        for idx, row in export_data.iterrows():
            year = row['year']
            partner_country = row['us_partner']  # 美国的出口目标国
            product = row['energy_product']
            us_export_value = row['trade_value_usd']
            
            # 检查是否有该年份的全球数据
            if year not in global_trade_data:
                logger.debug(f"缺少{year}年全球数据，跳过")
                continue
            
            global_year_data = global_trade_data[year]
            
            # 查找目标国在该年份、该产品上的所有进口记录
            # 注意：在全球数据中，目标国作为reporter，流向为M(Import)
            partner_imports = global_year_data[
                (global_year_data['reporter'] == partner_country) & 
                (global_year_data['flow'] == 'M') & 
                (global_year_data['energy_product'] == product)
            ].copy()
            
            if len(partner_imports) == 0:
                # 目标国在该产品上没有进口记录，锁定力为0
                locking_results.append({
                    'year': year,
                    'us_partner': partner_country,
                    'energy_product': product,
                    'us_role': 'exporter',
                    'market_locking_power': 0,
                    'target_import_hhi': 0,
                    'us_share_in_target': 0,
                    'target_total_suppliers': 0,
                    'target_total_imports': 0
                })
                continue
            
            # 计算目标国的总进口额
            total_imports = partner_imports['trade_value_usd'].sum()
            
            if total_imports <= 0:
                locking_results.append({
                    'year': year,
                    'us_partner': partner_country,
                    'energy_product': product,
                    'us_role': 'exporter',
                    'market_locking_power': 0,
                    'target_import_hhi': 0,
                    'us_share_in_target': 0,
                    'target_total_suppliers': 0,
                    'target_total_imports': 0
                })
                continue
            
            # 计算目标国各供应商的市场份额
            supplier_shares = partner_imports.groupby('partner')['trade_value_usd'].sum() / total_imports
            
            # 计算目标国的进口集中度（供应商HHI）
            import_hhi = (supplier_shares ** 2).sum()
            
            # 计算美国在目标国市场中的份额
            us_share = supplier_shares.get('USA', 0)  # 如果美国不在供应商列表中，份额为0
            
            # 计算出口锁定力：目标国进口HHI × 美国在目标国市场的份额
            export_locking_power = import_hhi * us_share
            
            locking_results.append({
                'year': year,
                'us_partner': partner_country,
                'energy_product': product,
                'us_role': 'exporter',
                'market_locking_power': export_locking_power,
                'target_import_hhi': import_hhi,
                'us_share_in_target': us_share,
                'target_total_suppliers': len(supplier_shares),
                'target_total_imports': total_imports
            })
        
        # 转换为DataFrame
        locking_df = pd.DataFrame(locking_results)
        
        # 与原数据合并
        df_with_locking = pd.merge(
            df_locking, 
            locking_df[['year', 'us_partner', 'energy_product', 'us_role', 'market_locking_power']], 
            on=['year', 'us_partner', 'energy_product', 'us_role'], 
            how='left'
        )
        
        # 填充缺失值为0
        df_with_locking['market_locking_power'] = df_with_locking['market_locking_power'].fillna(0)
        
        # 统计摘要
        if len(locking_df) > 0:
            logger.info(f"📊 出口锁定力统计:")
            logger.info(f"  平均锁定力: {locking_df['market_locking_power'].mean():.4f}")
            logger.info(f"  最高锁定力: {locking_df['market_locking_power'].max():.4f}")
            logger.info(f"  非零锁定力记录: {(locking_df['market_locking_power'] > 0).sum()} 条")
            logger.info(f"  美国在目标市场平均份额: {locking_df['us_share_in_target'].mean():.4f}")
            logger.info(f"  目标国平均供应商数: {locking_df['target_total_suppliers'].mean():.1f}")
            
            # 按产品分析
            product_stats = locking_df.groupby('energy_product').agg({
                'market_locking_power': ['mean', 'max'],
                'target_import_hhi': 'mean',
                'us_share_in_target': 'mean'
            }).round(4)
            
            logger.info(f"  按能源产品的出口锁定力:")
            for product in product_stats.index:
                stats = product_stats.loc[product]
                logger.info(f"    {product}: 平均锁定力={stats[('market_locking_power', 'mean')]:.4f}, " +
                           f"最高锁定力={stats[('market_locking_power', 'max')]:.4f}, " +
                           f"目标国HHI={stats[('target_import_hhi', 'mean')]:.4f}")
        
        logger.info("✅ 出口锁定力指标计算完成!")
        return df_with_locking
    
    def calculate_export_dli_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算出口DLI综合指标
        
        使用PCA方法对出口方向的四个维度进行降维合成：
        - continuity: 贸易持续性
        - infrastructure: 基础设施强度  
        - stability: 贸易稳定性
        - market_locking_power: 出口锁定力
        
        Args:
            df: 包含所有维度指标的DataFrame
            
        Returns:
            添加了export_dli_score列的DataFrame
        """
        
        logger.info("🧮 开始计算出口DLI综合指标...")
        
        # 只处理出口数据
        export_data = df[df['us_role'] == 'exporter'].copy()
        
        if len(export_data) == 0:
            logger.warning("没有出口数据，跳过出口DLI计算")
            return df
        
        # 定义所需的维度
        required_dimensions = ['continuity', 'infrastructure', 'stability', 'market_locking_power']
        
        # 检查必需维度是否存在
        missing_dimensions = [dim for dim in required_dimensions if dim not in export_data.columns]
        if missing_dimensions:
            raise ValueError(f"缺少出口DLI维度: {missing_dimensions}")
        
        # 去除有缺失值的记录
        complete_data = export_data.dropna(subset=required_dimensions)
        
        if len(complete_data) == 0:
            logger.warning("没有完整的出口DLI数据，跳过计算")
            return df
        
        logger.info(f"📊 出口DLI维度数据质量检查:")
        for dim in required_dimensions:
            logger.info(f"  {dim}: 均值={complete_data[dim].mean():.4f}, 标准差={complete_data[dim].std():.4f}, " + 
                       f"范围=[{complete_data[dim].min():.4f}, {complete_data[dim].max():.4f}]")
        
        # 标准化处理
        scaler = StandardScaler()
        standardized_dimensions = scaler.fit_transform(complete_data[required_dimensions])
        
        # 执行PCA
        pca = PCA(n_components=1)
        dli_scores = pca.fit_transform(standardized_dimensions)
        
        # 计算权重信息
        weights = pca.components_[0]
        explained_variance = pca.explained_variance_ratio_[0]
        
        logger.info(f"📈 出口DLI PCA分析结果:")
        logger.info(f"  解释方差比例: {explained_variance:.3f}")
        logger.info(f"  维度权重:")
        for i, dim in enumerate(required_dimensions):
            logger.info(f"    {dim}: {weights[i]:.4f}")
        
        # 将DLI分数添加到数据中
        complete_data['export_dli_score'] = dli_scores.flatten()
        
        # 将结果合并回原始数据
        df_result = df.copy()
        df_result = df_result.merge(
            complete_data[['year', 'us_partner', 'energy_product', 'us_role', 'export_dli_score']],
            on=['year', 'us_partner', 'energy_product', 'us_role'],
            how='left'
        )
        
        # 保存权重信息
        self._save_export_weights(weights, explained_variance, required_dimensions)
        
        logger.info("✅ 出口DLI综合指标计算完成!")
        return df_result
    
    def _save_export_weights(self, weights: np.ndarray, explained_variance: float, dimensions: List[str]):
        """保存出口DLI的PCA权重信息"""
        import json
        
        weights_info = {
            'version': '1.0',
            'description': '美国出口锁定DLI权重系统',
            'export_pca_weights': {dim: float(weight) for dim, weight in zip(dimensions, weights)},
            'explained_variance_ratio': float(explained_variance),
            'dimensions': dimensions,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        weights_path = self.output_dir / "export_dli_weights.json"
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 出口DLI权重信息已保存至: {weights_path}")
    
    def _add_us_role_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为原始贸易数据添加us_role字段
        
        根据reporter和partner字段识别美国的贸易角色：
        - 当reporter为USA且flow为X时，us_role为exporter  
        - 当reporter为USA且flow为M时，us_role为importer
        - 当partner为USA且flow为X时，us_role为importer
        - 当partner为USA且flow为M时，us_role为exporter
        
        Args:
            df: 原始贸易数据DataFrame
            
        Returns:
            添加了us_role和us_partner字段的DataFrame
        """
        
        logger.info("🔄 为贸易数据添加US角色标识...")
        
        df_with_role = df.copy()
        
        # 筛选涉及美国的贸易记录
        us_trade = df_with_role[
            (df_with_role['reporter'] == 'USA') | 
            (df_with_role['partner'] == 'USA')
        ].copy()
        
        if len(us_trade) == 0:
            logger.warning("未找到涉及美国的贸易记录")
            return df_with_role
        
        # 添加us_role字段
        conditions = [
            (us_trade['reporter'] == 'USA') & (us_trade['flow'] == 'X'),  # 美国出口
            (us_trade['reporter'] == 'USA') & (us_trade['flow'] == 'M'),  # 美国进口（作为reporter）
            (us_trade['partner'] == 'USA') & (us_trade['flow'] == 'X'),   # 其他国家对美出口（美国进口）
            (us_trade['partner'] == 'USA') & (us_trade['flow'] == 'M')    # 其他国家从美进口（美国出口）
        ]
        
        choices = [
            'exporter',   # 美国出口
            'importer',   # 美国进口
            'importer',   # 美国进口  
            'exporter'    # 美国出口
        ]
        
        us_trade['us_role'] = np.select(conditions, choices, default='unknown')
        
        # 添加us_partner字段（美国的贸易伙伴）
        us_trade['us_partner'] = np.where(
            us_trade['reporter'] == 'USA',
            us_trade['partner'],
            us_trade['reporter']
        )
        
        # 只返回涉及美国的贸易数据
        valid_us_trade = us_trade[us_trade['us_role'] != 'unknown'].copy()
        
        # 添加距离信息
        valid_us_trade = self._add_distance_info(valid_us_trade)
        
        logger.info(f"✅ 美国贸易数据处理完成:")
        logger.info(f"   总记录数: {len(valid_us_trade):,}")
        logger.info(f"   出口记录: {(valid_us_trade['us_role'] == 'exporter').sum():,}")
        logger.info(f"   进口记录: {(valid_us_trade['us_role'] == 'importer').sum():,}")
        logger.info(f"   贸易伙伴数: {valid_us_trade['us_partner'].nunique()}")
        
        return valid_us_trade
    
    def _add_distance_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为贸易数据添加距离信息
        
        Args:
            df: 包含us_partner字段的贸易数据
            
        Returns:
            添加了distance_km字段的DataFrame
        """
        
        logger.info("🌍 添加地理距离信息...")
        
        try:
            # 从data_preparation模块加载距离数据
            from data_preparation import load_complete_distance_data
            distance_data = load_complete_distance_data()
            
            # 创建美国到各国的距离映射
            us_distances = {}
            for country_pair, distance in distance_data.items():
                if country_pair[0] == 'USA':
                    us_distances[country_pair[1]] = distance
                elif country_pair[1] == 'USA':
                    us_distances[country_pair[0]] = distance
            
            # 为数据添加距离信息
            df['distance_km'] = df['us_partner'].map(us_distances)
            
            # 处理缺失的距离数据
            missing_distance = df['distance_km'].isnull().sum()
            if missing_distance > 0:
                logger.warning(f"⚠️ {missing_distance} 条记录缺少距离数据，将使用平均距离填充")
                avg_distance = df['distance_km'].mean()
                df['distance_km'] = df['distance_km'].fillna(avg_distance)
            
            logger.info(f"✅ 距离信息添加完成:")
            logger.info(f"   平均距离: {df['distance_km'].mean():.0f} km")
            logger.info(f"   距离范围: {df['distance_km'].min():.0f} - {df['distance_km'].max():.0f} km")
            
        except Exception as e:
            logger.warning(f"⚠️ 无法加载距离数据: {str(e)}，使用默认距离")
            # 使用默认距离（全球平均距离约10000km）
            df['distance_km'] = 10000.0
        
        return df
    
    def generate_export_dli_data(self, trade_data: Optional[pd.DataFrame] = None, 
                                global_trade_data: Optional[Dict[int, pd.DataFrame]] = None,
                                output_filename: str = "export_dli.csv") -> pd.DataFrame:
        """
        生成完整的出口DLI数据集
        
        Args:
            trade_data: 美国贸易数据
            global_trade_data: 全球贸易数据（计算出口锁定力需要）
            output_filename: 输出文件名
            
        Returns:
            完整的出口DLI数据集
        """
        
        logger.info("🚀 开始生成出口DLI数据集...")
        
        if trade_data is None:
            # 从数据准备模块加载数据
            from data_preparation import load_global_trade_data_range
            trade_data_dict = load_global_trade_data_range()
            
            # 合并所有年份的数据
            logger.info("🔄 合并所有年份的贸易数据...")
            trade_data_list = []
            for year, yearly_data in trade_data_dict.items():
                trade_data_list.append(yearly_data)
            trade_data = pd.concat(trade_data_list, ignore_index=True)
            logger.info(f"✅ 合并完成，总计 {len(trade_data):,} 条记录")
            
            # 为原始数据添加us_role字段
            trade_data = self._add_us_role_field(trade_data)
            
            # 同时保存global_trade_data供出口锁定力计算使用
            if global_trade_data is None:
                global_trade_data = trade_data_dict
        
        # 只保留美国出口数据
        export_data = trade_data[trade_data['us_role'] == 'exporter'].copy()
        
        if len(export_data) == 0:
            raise ValueError("没有找到美国出口数据")
        
        logger.info(f"📊 出口数据基础信息:")
        logger.info(f"  记录数: {len(export_data):,}")
        logger.info(f"  年份范围: {export_data['year'].min()}-{export_data['year'].max()}")
        logger.info(f"  出口目标国数: {export_data['us_partner'].nunique()}")
        logger.info(f"  能源产品类型: {export_data['energy_product'].nunique()}")
        
        # 第1步：计算持续性指标
        logger.info("1️⃣ 计算出口贸易持续性...")
        export_data = calculate_continuity(export_data)
        
        # 第2步：计算基础设施强度
        logger.info("2️⃣ 计算出口基础设施强度...")
        export_data = calculate_infrastructure(export_data)
        
        # 第3步：计算稳定性指标
        logger.info("3️⃣ 计算出口贸易稳定性...")
        export_data = calculate_stability(export_data)
        
        # 第4步：计算出口锁定力
        logger.info("4️⃣ 计算出口锁定力...")
        if global_trade_data:
            export_data = self.calculate_export_locking_power(export_data, global_trade_data)
        else:
            logger.warning("未提供全球数据，出口锁定力设为0")
            export_data['market_locking_power'] = 0
        
        # 第5步：计算出口DLI综合指标
        logger.info("5️⃣ 计算出口DLI综合指标...")
        export_data = self.calculate_export_dli_composite(export_data)
        
        # 第6步：数据整理和输出
        logger.info("6️⃣ 整理和保存出口DLI数据...")
        
        # 选择输出列
        output_columns = [
            'year', 'us_partner', 'energy_product', 'us_role',
            'trade_value_usd', 'distance_km',
            'continuity', 'infrastructure', 'stability', 'market_locking_power',
            'export_dli_score'
        ]
        
        available_columns = [col for col in output_columns if col in export_data.columns]
        df_output = export_data[available_columns].copy()
        
        # 排序
        df_output = df_output.sort_values(['year', 'us_partner', 'energy_product'])
        df_output = df_output.reset_index(drop=True)
        
        # 数据验证
        logger.info("🔍 出口DLI数据验证:")
        logger.info(f"  总记录数: {len(df_output):,}")
        logger.info(f"  出口目标国: {df_output['us_partner'].nunique()}")
        logger.info(f"  能源产品: {df_output['energy_product'].nunique()}")
        
        # 检查缺失值
        missing_summary = df_output.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        if len(missing_cols) > 0:
            logger.warning("存在缺失值:")
            for col, count in missing_cols.items():
                logger.warning(f"  {col}: {count} ({count/len(df_output)*100:.1f}%)")
        
        # 保存数据
        output_path = self.output_dir / output_filename
        df_output.to_csv(output_path, index=False)
        
        logger.info(f"💾 出口DLI数据已保存至: {output_path}")
        logger.info("🎉 出口DLI数据生成完成!")
        
        return df_output

def main():
    """测试函数"""
    import sys
    from data_preparation import load_global_trade_data_range
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建计算器
        calculator = ExportDLICalculator()
        
        # 生成出口DLI数据
        export_dli_data = calculator.generate_export_dli_data()
        
        print(f"\n✅ 出口DLI数据生成成功！")
        print(f"📊 数据维度: {export_dli_data.shape}")
        print(f"🔗 出口DLI综合指标范围: [{export_dli_data['export_dli_score'].min():.4f}, {export_dli_data['export_dli_score'].max():.4f}]")
        print(f"🌍 出口目标国数量: {export_dli_data['us_partner'].nunique()}")
        
        # 显示前5条记录
        print(f"\n📋 数据样例:")
        print(export_dli_data.head())
        
    except Exception as e:
        logger.error(f"❌ 出口DLI计算失败: {e}")
        raise

if __name__ == "__main__":
    main()