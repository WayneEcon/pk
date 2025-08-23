"""
HHI_imports 构建器 - 独立脚本
=====================================

负责处理HHI_imports（进口多样化指数）的加载、提取和构建逻辑。
从05模块转移至08模块，保持代码结构清晰。

功能：
1. 优先从05模块加载已构建的hhi_imports.csv
2. 备用：从旧的vul_us.csv中提取hhi_imports数据  
3. 保存提取结果到outputs/hhi_imports_extracted.csv

Author: Energy Network Analysis Team
Date: 2025-08-22
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class HHIImportsBuilder:
    """HHI_imports 数据构建器"""
    
    def __init__(self, base_dir: Path, output_dir: Path, temp_data_dir: Path):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.temp_data_dir = temp_data_dir
        
    def load_hhi_imports(self) -> Optional[pd.DataFrame]:
        """
        加载或构建HHI_imports数据
        
        加载策略：
        1. 优先从05模块加载已构建的hhi_imports.csv
        2. 备用：从08data目录的hhi_imports_extracted.csv加载
        3. 最后：从旧的vul_us.csv中提取并保存
        
        Returns:
            pd.DataFrame or None: HHI_imports数据
        """
        logger.info("🔄 开始加载HHI_imports数据...")
        
        # 策略1：从05模块加载已构建的hhi_imports.csv
        hhi_imports_path = self.base_dir / "src" / "05_causal_validation" / "outputs" / "hhi_imports.csv"
        if hhi_imports_path.exists():
            try:
                hhi_imports_data = pd.read_csv(hhi_imports_path)
                logger.info(f"✅ 从05模块加载HHI_imports数据: {len(hhi_imports_data)} 行")
                logger.info("   ⚠️ 注意：已废弃vul_us构造，改用hhi_imports避免构造内生性")
                return hhi_imports_data
            except Exception as e:
                logger.warning(f"⚠️ 无法从05模块加载HHI_imports数据: {str(e)}")
        else:
            logger.info("ℹ️ 05模块hhi_imports.csv不存在，尝试备用策略...")
        
        # 策略2：从08data目录加载已提取的hhi_imports_extracted.csv
        hhi_extracted_path = self.output_dir / "hhi_imports_extracted.csv"
        if hhi_extracted_path.exists():
            try:
                hhi_imports_data = pd.read_csv(hhi_extracted_path)
                logger.info(f"✅ 加载提取的HHI_imports数据: {len(hhi_imports_data)} 行")
                return hhi_imports_data
            except Exception as e:
                logger.warning(f"⚠️ 无法加载提取的HHI_imports数据: {str(e)}")
        else:
            logger.info("ℹ️ 提取的hhi_imports_extracted.csv不存在，尝试从vul_us.csv提取...")
        
        # 策略3：从旧的vul_us.csv中提取hhi_imports
        return self._extract_from_vul_us()
    
    def _extract_from_vul_us(self) -> Optional[pd.DataFrame]:
        """
        从旧的vul_us.csv文件中提取hhi_imports数据
        
        Returns:
            pd.DataFrame or None: 提取的HHI_imports数据
        """
        vul_us_path = self.temp_data_dir / "vul_us.csv"
        
        if not vul_us_path.exists():
            logger.warning("⚠️ vul_us.csv文件不存在，无法提取HHI_imports数据")
            return None
        
        try:
            logger.info("🔄 从vul_us.csv提取HHI_imports数据...")
            vul_us_data = pd.read_csv(vul_us_path)
            
            # 验证必需的列是否存在
            required_columns = ['year', 'country', 'hhi_imports']
            missing_columns = [col for col in required_columns if col not in vul_us_data.columns]
            if missing_columns:
                logger.error(f"❌ vul_us.csv缺少必需列: {missing_columns}")
                return None
            
            # 提取hhi_imports相关列
            available_columns = ['year', 'country', 'hhi_imports']
            if 'us_import_share' in vul_us_data.columns:
                available_columns.append('us_import_share')
            
            hhi_imports_data = vul_us_data[available_columns].copy()
            
            # 数据清理
            hhi_imports_data = hhi_imports_data.dropna(subset=['hhi_imports'])
            
            # 保存提取的数据到outputs目录
            hhi_extracted_path = self.output_dir / "hhi_imports_extracted.csv"
            hhi_imports_data.to_csv(hhi_extracted_path, index=False)
            
            logger.info(f"✅ 从vul_us.csv提取HHI_imports: {len(hhi_imports_data)} 行")
            logger.info(f"   💾 已保存到: {hhi_extracted_path}")
            logger.info(f"   📊 数据概况:")
            logger.info(f"      - 国家数: {hhi_imports_data['country'].nunique()}")
            logger.info(f"      - 年份范围: {hhi_imports_data['year'].min()}-{hhi_imports_data['year'].max()}")
            logger.info(f"      - HHI范围: {hhi_imports_data['hhi_imports'].min():.4f}-{hhi_imports_data['hhi_imports'].max():.4f}")
            logger.warning("⚠️ 使用旧数据源，建议重新运行05模块生成新的hhi_imports.csv")
            
            return hhi_imports_data
            
        except Exception as e:
            logger.error(f"❌ 从vul_us.csv提取HHI_imports失败: {str(e)}", exc_info=True)
            return None
    
    def construct_from_trade_data(self, trade_data: pd.DataFrame, output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        从贸易数据直接构建HHI_imports（如果需要的话）
        
        这是从05模块移植过来的构建逻辑，作为最后的备用方案
        
        Args:
            trade_data: 贸易流数据
            output_path: 输出路径，默认为outputs/hhi_imports_constructed.csv
            
        Returns:
            pd.DataFrame or None: 构建的HHI_imports数据
        """
        logger.info("🔄 从贸易数据构建HHI_imports...")
        
        if output_path is None:
            output_path = self.output_dir / "hhi_imports_constructed.csv"
        
        try:
            # 计算各国的进口依赖度和多样化程度
            import_data = trade_data[trade_data['flow'] == 'M'].copy()
            import_data = import_data.groupby(['year', 'reporter', 'partner']).agg(
                trade_value_raw_usd=('trade_value_raw_usd', 'sum')
            ).reset_index()
            
            # 计算HHI指数（核心指标）
            total_imports = import_data.groupby(['year', 'reporter']).agg(
                total_imports=('trade_value_raw_usd', 'sum')
            ).reset_index()
            
            import_data = import_data.merge(total_imports, on=['year', 'reporter'])
            import_data['import_share'] = import_data['trade_value_raw_usd'] / import_data['total_imports']
            
            hhi_data = import_data.groupby(['year', 'reporter']).apply(
                lambda x: (x['import_share'] ** 2).sum()
            ).reset_index(name='hhi_imports')
            
            # 重命名country列以便合并
            hhi_data = hhi_data.rename(columns={'reporter': 'country'})
            
            # 计算对美依赖度（辅助变量）
            us_imports = import_data[import_data['partner'] == 'USA'].copy()
            us_imports = us_imports.rename(columns={
                'import_share': 'us_import_share',
                'reporter': 'country'
            })[['year', 'country', 'us_import_share']]
            
            # 合并数据
            final_data = hhi_data.merge(us_imports, on=['year', 'country'], how='left')
            final_data['us_import_share'] = final_data['us_import_share'].fillna(0)
            
            # 生成最终数据
            hhi_df = final_data[['year', 'country', 'hhi_imports', 'us_import_share']].copy().dropna()
            
            # 保存构建的数据
            hhi_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ HHI_imports构建完成: {len(hhi_df)} 行记录")
            logger.info(f"   💾 保存至: {output_path}")
            logger.info(f"   📊 hhi_imports范围: {hhi_df['hhi_imports'].min():.4f} - {hhi_df['hhi_imports'].max():.4f}")
            logger.info(f"   🌍 覆盖国家: {hhi_df['country'].nunique()} 个")
            logger.info(f"   📅 年份范围: {hhi_df['year'].min()}-{hhi_df['year'].max()}")
            
            return hhi_df
            
        except Exception as e:
            logger.error(f"❌ 从贸易数据构建HHI_imports失败: {str(e)}", exc_info=True)
            return None

def main():
    """独立运行时的测试函数"""
    import sys
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 设置路径
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent.parent  # project root
    output_dir = current_dir / "outputs"
    temp_data_dir = current_dir / "08data"
    
    # 创建构建器
    builder = HHIImportsBuilder(base_dir, output_dir, temp_data_dir)
    
    # 测试加载
    result = builder.load_hhi_imports()
    
    if result is not None:
        print(f"\n✅ 测试成功！加载了 {len(result)} 行HHI_imports数据")
        print(f"   列名: {list(result.columns)}")
        print(f"   数据预览:")
        print(result.head())
    else:
        print("\n❌ 测试失败：未能加载HHI_imports数据")
        
if __name__ == "__main__":
    main()