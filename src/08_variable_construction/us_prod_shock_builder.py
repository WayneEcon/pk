"""
US_ProdShock Builder - AR(2) Residual Method
===========================================

经济学标准的美国天然气产量冲击构建模块
使用AR(2)残差方法识别"非预期"的产量冲击

Author: Claude Code
Date: 2025-08-22
"""

import logging
import numpy as np
import pandas as pd
import requests
from typing import Optional
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USProdShockBuilder:
    """美国天然气产量冲击构建器 - AR(2)残差方法"""
    
    def __init__(self, eia_api_key: str = "kCKMXECZ7EZxHpYPXekyOhSdccpNc85aeOpDGIwm"):
        """
        初始化构建器
        
        Args:
            eia_api_key: EIA API密钥
        """
        self.eia_api_key = eia_api_key
        logger.info("🔧 US_ProdShock Builder 初始化完成 (AR(2)残差方法)")
    
    def fetch_us_gas_production(self, start_year: int = 2000, end_year: int = 2024) -> Optional[pd.DataFrame]:
        """
        加载正确的美国年度天然气总产量数据（硬编码本地数据）
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            包含年份和产量的DataFrame，失败时返回None
        """
        logger.info(f"📊 加载正确的美国天然气产量数据 ({start_year}-{end_year})...")
        
        try:
            # 使用正确的本地数据文件
            current_dir = Path(__file__).parent
            data_file = current_dir / "outputs" / "us_gas_production_correct_data.csv"
            
            if not data_file.exists():
                logger.error(f"❌ 正确数据文件不存在: {data_file}")
                return None
            
            # 读取正确的数据
            gas_data = pd.read_csv(data_file)
            
            # 筛选年份范围
            gas_data = gas_data[(gas_data['year'] >= start_year) & (gas_data['year'] <= end_year)]
            
            # 重命名列以保持一致性
            gas_data = gas_data.rename(columns={'us_gas_production_twh': 'us_gas_production'})
            
            # 移除缺失值并排序
            gas_data = gas_data.dropna().sort_values('year').reset_index(drop=True)
            
            logger.info(f"✅ 成功加载 {len(gas_data)} 年的天然气产量数据")
            logger.info(f"   数据范围: {gas_data['year'].min()}-{gas_data['year'].max()}")
            logger.info(f"   数据单位: TWh (太瓦时)")
            logger.info(f"   产量范围: {gas_data['us_gas_production'].min():.1f} - {gas_data['us_gas_production'].max():.1f} TWh")
            
            return gas_data
            
        except Exception as e:
            logger.error(f"❌ 加载天然气产量数据失败: {str(e)}")
            return None
    
    def build_ar2_shock(self, gas_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        使用AR(2)残差方法构建产量冲击序列
        
        Args:
            gas_data: 包含年份和产量的DataFrame
            
        Returns:
            包含冲击序列的DataFrame，失败时返回None
        """
        logger.info("🧮 正在构建AR(2)残差产量冲击...")
        
        try:
            if len(gas_data) < 5:
                logger.error("❌ 数据点过少，无法拟合AR(2)模型")
                return None
            
            # 第一步：计算产量的自然对数
            gas_data = gas_data.copy()
            gas_data['log_production'] = np.log(gas_data['us_gas_production'])
            
            logger.info(f"📈 计算对数产量序列，范围: {gas_data['log_production'].min():.3f} - {gas_data['log_production'].max():.3f}")
            
            # 第二步：拟合AR(2)模型
            log_series = gas_data['log_production'].values
            
            # 使用statsmodels的AutoReg拟合AR(2)
            ar2_model = AutoReg(log_series, lags=2, trend='c')  # 包含常数项
            ar2_fitted = ar2_model.fit()
            
            logger.info("🔍 AR(2)模型拟合完成")
            logger.info(f"   AIC: {ar2_fitted.aic:.3f}")
            logger.info(f"   BIC: {ar2_fitted.bic:.3f}")
            # 某些statsmodels版本可能没有rsquared属性
            try:
                r_squared = ar2_fitted.rsquared
                logger.info(f"   R²: {r_squared:.3f}")
            except AttributeError:
                logger.info("   R²: 不可用 (statsmodels版本限制)")
            
            # 第三步：提取残差作为非预期冲击
            residuals = ar2_fitted.resid
            
            # 第四步：构建结果DataFrame
            # 注意：AR(2)模型由于滞后特性，前2个观测值没有残差
            result_df = gas_data.copy()
            result_df['us_prod_shock'] = np.nan  # 初始化为NaN
            
            # 从第3个观测值开始填入残差（索引2开始，对应2002年之后）
            if len(residuals) > 0:
                start_idx = 2  # AR(2)需要2个滞后值
                end_idx = start_idx + len(residuals)
                result_df.loc[start_idx:end_idx-1, 'us_prod_shock'] = residuals
            
            # 添加模型统计信息到结果中（可选）
            result_df.attrs['ar2_aic'] = ar2_fitted.aic
            result_df.attrs['ar2_bic'] = ar2_fitted.bic
            try:
                result_df.attrs['ar2_rsquared'] = ar2_fitted.rsquared
            except AttributeError:
                result_df.attrs['ar2_rsquared'] = None
            
            logger.info(f"✅ AR(2)冲击序列构建完成")
            logger.info(f"   有效冲击值: {result_df['us_prod_shock'].notna().sum()} 个")
            logger.info(f"   缺失值(前2年): {result_df['us_prod_shock'].isna().sum()} 个")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ AR(2)冲击构建失败: {str(e)}")
            return None
    
    def build_us_prod_shock(self, start_year: int = 2000, end_year: int = 2024, 
                           save_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        完整的US_ProdShock构建流程
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            save_path: 保存路径（可选）
            
        Returns:
            最终的冲击数据DataFrame
        """
        logger.info("🚀 开始构建US_ProdShock (AR(2)残差方法)")
        
        # 步骤1：获取数据
        gas_data = self.fetch_us_gas_production(start_year, end_year)
        if gas_data is None:
            return None
        
        # 保存原始EIA数据（如果指定了路径）
        if save_path:
            try:
                output_dir = save_path.parent
                raw_data_path = output_dir / "us_gas_production_raw_eia.csv"
                gas_data.to_csv(raw_data_path, index=False)
                logger.info(f"💾 原始EIA数据已保存到: {raw_data_path}")
            except Exception as e:
                logger.warning(f"⚠️ 原始数据保存失败: {str(e)}")
        
        # 步骤2：构建AR(2)冲击
        shock_data = self.build_ar2_shock(gas_data)
        if shock_data is None:
            return None
        
        # 步骤3：保存结果（如果指定了路径）
        if save_path:
            try:
                # 选择关键列保存
                output_columns = ['year', 'us_gas_production', 'log_production', 'us_prod_shock']
                available_columns = [col for col in output_columns if col in shock_data.columns]
                
                shock_data[available_columns].to_csv(save_path, index=False)
                logger.info(f"💾 AR(2)冲击结果已保存到: {save_path}")
            except Exception as e:
                logger.warning(f"⚠️ 结果保存失败: {str(e)}")
        
        logger.info("🎉 US_ProdShock构建完成!")
        
        return shock_data


def main():
    """测试脚本"""
    builder = USProdShockBuilder()
    result = builder.build_us_prod_shock(
        start_year=2000, 
        end_year=2024,
        save_path=Path("test_us_prod_shock.csv")
    )
    
    if result is not None:
        print("\n📊 构建结果预览:")
        print(result[['year', 'us_gas_production', 'us_prod_shock']].head(10))
        print(f"\n📈 冲击统计: 均值={result['us_prod_shock'].mean():.4f}, 标准差={result['us_prod_shock'].std():.4f}")


if __name__ == "__main__":
    main()