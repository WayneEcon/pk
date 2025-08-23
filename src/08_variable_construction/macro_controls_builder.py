#!/usr/bin/env python3
"""
宏观控制变量构建器
===================

独立模块，负责下载、清理和生成宏观经济控制变量。
从main.py中提取的MacroDataHandler类，用于构建：
- GDP (current USD)
- Population (total)  
- Trade openness (% of GDP)
- 对数变换的GDP和人口

数据源：世界银行开放数据API
输出：macro_controls.csv (保存到outputs目录)

作者: Energy Network Analysis Team
版本: v1.0 - 独立模块版
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import requests
from typing import Optional

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MacroControlsBuilder:
    """宏观经济控制变量构建器"""
    
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        """初始化构建器"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "08data"
        else:
            self.data_dir = Path(data_dir)
            
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "outputs"
        else:
            self.output_dir = Path(output_dir)
        
        # 确保目录存在
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # 定义文件路径
        self.raw_path = self.data_dir / "macro_controls_worldbank.csv"
        self.clean_path = self.data_dir / "macro_controls_worldbank_clean.csv"
        self.final_path = self.output_dir / "macro_controls.csv"
        
        logger.info("🏗️ 宏观控制变量构建器初始化完成")
        logger.info(f"   数据目录: {self.data_dir}")
        logger.info(f"   输出目录: {self.output_dir}")

    def build_macro_controls(self, start_year: int = 2000, end_year: int = 2024) -> Optional[pd.DataFrame]:
        """构建宏观控制变量的主要方法"""
        logger.info("🌍 开始构建宏观经济控制变量...")
        
        # 获取清理后的数据
        clean_data = self._get_clean_data()
        if clean_data is None:
            logger.error("❌ 无法获取宏观数据")
            return None
        
        # 筛选年份范围
        clean_data = clean_data[
            (clean_data['year'] >= start_year) & 
            (clean_data['year'] <= end_year)
        ].copy()
        
        # 重命名列以保持简洁
        final_data = clean_data.rename(columns={
            'country_code': 'country',
            'country_name': 'country_name'
        })
        
        # 选择最终输出的列
        output_columns = [
            'country', 'country_name', 'year', 
            'gdp_current_usd', 'population_total', 'trade_openness_gdp_pct',
            'log_gdp', 'log_population'
        ]
        
        # 确保所有列都存在
        available_columns = [col for col in output_columns if col in final_data.columns]
        final_data = final_data[available_columns].copy()
        
        # 排序
        final_data = final_data.sort_values(['country', 'year']).reset_index(drop=True)
        
        # 保存到outputs目录
        final_data.to_csv(self.final_path, index=False)
        
        logger.info(f"✅ 宏观控制变量构建完成:")
        logger.info(f"   📊 数据记录: {len(final_data)} 条")
        logger.info(f"   🌍 覆盖国家: {final_data['country'].nunique()} 个")
        logger.info(f"   📅 年份范围: {final_data['year'].min()}-{final_data['year'].max()}")
        logger.info(f"   💾 保存至: {self.final_path}")
        
        return final_data

    def _get_clean_data(self) -> Optional[pd.DataFrame]:
        """获取干净的宏观数据，按需下载和清理"""
        if self.clean_path.exists():
            logger.info(f"✅ 从缓存加载已清理的宏观数据: {self.clean_path}")
            return pd.read_csv(self.clean_path)
        
        if not self.raw_path.exists():
            logger.info("⚠️ 未找到原始宏观数据缓存，开始从世界银行下载...")
            raw_data = self._download_data()
            if raw_data is None:
                logger.error("❌ 下载宏观数据失败。")
                return None
            raw_data.to_csv(self.raw_path, index=False)
            logger.info(f"💾 原始宏观数据已保存至: {self.raw_path}")
        else:
            logger.info(f"✅ 从缓存加载原始宏观数据: {self.raw_path}")
            raw_data = pd.read_csv(self.raw_path)

        logger.info("🧹 开始清理世界银行数据...")
        clean_data = self._clean_data(raw_data)
        clean_data.to_csv(self.clean_path, index=False)
        logger.info(f"💾 清理后的宏观数据已保存至: {self.clean_path}")
        
        return clean_data

    def _download_data(self) -> Optional[pd.DataFrame]:
        """直接使用世界银行REST API下载数据"""
        indicators = {
            'NY.GDP.MKTP.CD': 'gdp_current_usd', 
            'SP.POP.TOTL': 'population_total',
            'NE.TRD.GNFS.ZS': 'trade_openness_gdp_pct'
        }
        all_data = []
        
        for code, name in indicators.items():
            logger.info(f"📊 下载指标: {name} ({code})")
            url = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
            params = {
                'date': '2000:2024', 
                'format': 'json', 
                'per_page': 20000, 
                'source': '2'
            }
            
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if len(data) > 1 and data[1]:
                    for record in data[1]:
                        if record.get('value') is not None:
                            all_data.append({
                                'country_name': record['country']['value'],
                                'country_code': record['countryiso3code'],
                                'year': int(record['date']), 
                                'indicator': name,
                                'value': float(record['value'])
                            })
            except Exception as e:
                logger.error(f"   ❌ {name} 下载失败: {e}")
                continue
        
        if not all_data: 
            return None
        
        df = pd.DataFrame(all_data)
        df_pivot = df.pivot_table(
            index=['country_name', 'country_code', 'year'], 
            columns='indicator', 
            values='value'
        ).reset_index()
        df_pivot.columns.name = None
        df_pivot = df_pivot.dropna(subset=list(indicators.values()), how='all')
        
        # 计算对数变换
        with np.errstate(divide='ignore', invalid='ignore'):
            df_pivot['log_gdp'] = np.log(df_pivot['gdp_current_usd'])
            df_pivot['log_population'] = np.log(df_pivot['population_total'])
        
        return df_pivot.sort_values(['country_name', 'year']).reset_index(drop=True)

    def _clean_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """清理世界银行数据，只保留主权国家（严格过滤属地和小岛屿）"""
        # 地区和经济体分组代码
        exclude_codes = {
            'HIC', 'LIC', 'LMC', 'LMY', 'MIC', 'UMC', 'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAS', 'ECA', 'ECS',
            'EUU', 'FCS', 'HPC', 'LAC', 'LCN', 'LDC', 'MEA', 'MNA', 'NAC', 'OED', 'OSS', 'PSS', 'SAS', 'SSA', 'SSF',
            'SST', 'TEA', 'TEC', 'TLA', 'TSA', 'TSS', 'WLD', 'EMU', 'EAR', 'LTE', 'PST', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'PRE'
        }
        
        # 属地、海外领土和小岛屿（严格排除）
        exclude_territories = {
            'ABW', 'ASM', 'ATG', 'BHS', 'BRB', 'BMU', 'VGB', 'CYM', 'CUW', 'DMA', 'FRO', 'GRL', 'GRD', 'GUM', 'GIB',
            'IMN', 'KNA', 'LCA', 'LIE', 'MAC', 'MAF', 'MHL', 'FSM', 'MCO', 'NCL', 'MNP', 'PLW', 'PYF', 'KNA', 'LCA',
            'MAF', 'SXM', 'SMR', 'TCA', 'TON', 'TTO', 'TUV', 'VCT', 'VIR', 'WSM', 'STP', 'SYC', 'PLW', 'NRU',
            'MHL', 'KIR', 'FSM', 'FJI', 'VUT', 'SLB', 'TUV', 'TON', 'WSM', 'COM', 'CPV', 'MDV', 'MLT'
        }
        
        # 争议地区和特殊政治实体
        exclude_disputed = {'XKX', 'PSE', 'TWN'}  # 科索沃、巴勒斯坦、台湾
        
        exclude_keywords = ['income', 'countries', 'classification', 'indebted', 'developed', 'fragile', 'conflict', 'situations', 'area']
        manual_exclude = [
            'Arab World', 'Euro area', 'European Union', 'North America', 'Sub-Saharan Africa', 'East Asia & Pacific',
            'Europe & Central Asia', 'Latin America & Caribbean', 'Middle East & North Africa', 'South Asia', 'World'
        ]
        
        # 应用过滤
        df = raw_df[~raw_df['country_code'].isin(exclude_codes)].copy()
        df = df[~df['country_code'].isin(exclude_territories)].copy()  # 严格排除属地
        df = df[~df['country_code'].isin(exclude_disputed)].copy()    # 排除争议地区
        df = df[~df['country_name'].str.contains('&', na=False)]
        df = df[~df['country_name'].isin(manual_exclude)]
        
        def should_exclude(name):
            name_lower = name.lower()
            if 'united states' in name_lower: 
                return False
            if 'states' in name_lower: 
                return any(word in name_lower for word in ['small', 'island', 'caribbean'])
            # 排除明显的小岛屿和属地关键词
            island_keywords = ['island', 'islands', 'territory', 'overseas', 'dependency', 'crown', 'british', 'french', 'dutch']
            if any(keyword in name_lower for keyword in island_keywords):
                return True
            return any(keyword in name_lower for keyword in exclude_keywords)
            
        df = df[~df['country_name'].apply(should_exclude)]
        
        logger.info(f"🧹 国家过滤完成: 从{len(raw_df)}个实体过滤到{len(df)}个主权国家")
        logger.info(f"   排除的类型: 地区分组({len(exclude_codes)}个), 属地({len(exclude_territories)}个), 争议地区({len(exclude_disputed)}个)")
        
        return df.dropna(subset=['country_code'])

def main():
    """主函数：独立运行宏观控制变量构建"""
    print("🌍 宏观控制变量构建器")
    print("="*50)
    
    try:
        # 初始化构建器
        builder = MacroControlsBuilder()
        
        # 构建宏观控制变量
        macro_data = builder.build_macro_controls()
        
        if macro_data is not None:
            print("\n✅ 宏观控制变量构建成功！")
            print(f"📄 输出文件: {builder.final_path}")
            print(f"📊 数据概览: {len(macro_data)} 行, {macro_data['country'].nunique()} 个国家")
        else:
            print("\n❌ 宏观控制变量构建失败")
            
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()