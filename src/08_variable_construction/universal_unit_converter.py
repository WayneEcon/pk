"""
通用单位转换器 - 用于OVI计算的所有数据标准化
目标: 天然气 -> BCM/年, 石油 -> MTPA
"""

import pandas as pd
import numpy as np
from typing import Dict, Union

class UniversalUnitConverter:
    def __init__(self):
        """初始化转换系数表"""
        
        # =============================================================
        # 天然气单位转换表 (目标: BCM/年)
        # =============================================================
        self.gas_to_bcm_per_year = {
            # 已经是BCM/年的单位
            'bcm/y': 1.0,
            'bcm/year': 1.0,
            'bcm': 1.0,  # 假设年度数据
            
            # 立方米系列
            'MMSCMD': 365 * 1e6 / 1e9,  # 百万标准立方米/天 -> BCM/年
            'MMcf/d': 365 * 1e6 * 0.0283168 / 1e9,  # 百万立方英尺/天 -> BCM/年
            'mill.Sm3/day': 365 * 1e6 / 1e9,  # 百万标准立方米/天 -> BCM/年  
            'Mcf/d': 365 * 1e3 * 0.0283168 / 1e9,  # 千立方英尺/天 -> BCM/年
            'scm/yr': 1.0 / 1e9,  # 标准立方米/年 -> BCM/年
            'scm/y': 1.0 / 1e9,  # 标准立方米/年 -> BCM/年
            'm3/h': 365 * 24 / 1e9,  # 立方米/小时 -> BCM/年
            'bcf/d': 365 * 1e9 * 0.0283168 / 1e9,  # 十亿立方英尺/天 -> BCM/年
            
            # 能量单位系列 (需要转换为体积)
            'TJ/d': 365 * 1e12 / (39.3e6) / 1e9,  # 太焦耳/天 -> BCM/年 (天然气热值39.3MJ/m3)
            'PJ/y': 1e15 / (39.3e6) / 1e9,  # 拍焦耳/年 -> BCM/年
            'GWh/d': 365 * 3.6e12 / (39.3e6) / 1e9,  # 吉瓦时/天 -> BCM/年
            'MWh/d': 365 * 3.6e9 / (39.3e6) / 1e9,  # 兆瓦时/天 -> BCM/年
            'MMBtu/d': 365 * 1e6 * 1.055e6 / (39.3e6) / 1e9,  # 百万英热单位/天 -> BCM/年
            'Dth/d': 365 * 1e5 * 1.055e6 / (39.3e6) / 1e9,  # 十万英热单位/天 -> BCM/年
            
            # LNG质量单位 (液化天然气)
            'mtpa': 1.36,  # 百万吨/年 LNG -> BCM/年 (液化天然气密度转换)
            'mpta': 1.36,  # 拼写错误的mtpa
            
            # 极小单位
            'gal/day': 365 * 3.78541e-3 / 1e9,  # 加仑/天 -> BCM/年
        }
        
        # =============================================================
        # 石油单位转换表 (目标: MTPA)  
        # =============================================================
        self.oil_to_mtpa = {
            # 已经是MTPA的单位
            'mtpa': 1.0,
            'MTPA': 1.0,
            
            # 桶系列单位
            'bpd': 365 * 0.137 / 1e6,  # 桶/天 -> MTPA (1桶≈0.137吨)
            'bph': 365 * 24 * 0.137 / 1e6,  # 桶/小时 -> MTPA
            'Mb/d': 365 * 1e3 * 0.137 / 1e6,  # 千桶/天 -> MTPA  
            'mb/d': 365 * 1e3 * 0.137 / 1e6,  # 千桶/天 -> MTPA (小写)
            'thousand barrels daily': 365 * 1e3 * 0.137 / 1e6,  # 千桶/天
            'thousands of barrels per day': 365 * 1e3 * 0.137 / 1e6,  # 千桶/天
            
            # 体积单位系列 (原油密度约0.85吨/立方米)
            'm3/day': 365 * 0.85 / 1e6,  # 立方米/天 -> MTPA
            'm3/month': 12 * 0.85 / 1e6,  # 立方米/月 -> MTPA
            'm3/year': 0.85 / 1e6,  # 立方米/年 -> MTPA
            'thousand m3/year': 1e3 * 0.85 / 1e6,  # 千立方米/年 -> MTPA
            
            # 质量单位系列
            'Tn/d': 365 / 1e6,  # 吨/天 -> MTPA
            'Tn/day': 365 / 1e6,  # 吨/天 -> MTPA
            'tn/h': 365 * 24 / 1e6,  # 吨/小时 -> MTPA
            
            # 极小单位
            'lpy': 1.0 / 1e9,  # 升/年 -> MTPA (忽略密度差异)
        }
        
        # =============================================================
        # 数据清洗规则
        # =============================================================
        self.unit_aliases = {
            # 天然气单位别名
            'bcm/y': ['bcm/year', 'Bcm/y', 'BCM/y', 'BCM/year'],
            'MMcf/d': ['mmcf/d', 'MMCF/d', 'MMcf/day'],
            'MMSCMD': ['mmscmd', 'MMSCM/d', 'mmscm/d'],
            'TJ/d': ['tj/d', 'TJ/day', 'tj/day'],
            'mtpa': ['MTPA', 'Mt/a', 'mt/a'],
            
            # 石油单位别名  
            'bpd': ['BPD', 'b/d', 'B/d'],
            'mb/d': ['Mb/d', 'MB/d', 'kb/d', 'Kb/d'],
            'mtpa': ['MTPA', 'Mt/a', 'mt/a'],
        }
    
    def normalize_unit(self, unit: str) -> str:
        """标准化单位名称"""
        if pd.isna(unit) or unit == '':
            return 'unknown'
            
        unit = str(unit).strip()
        
        # 检查别名映射
        for standard, aliases in self.unit_aliases.items():
            if unit in aliases:
                return standard
                
        return unit
    
    def clean_numeric_value(self, value) -> float:
        """清理数值，处理逗号分隔符等问题"""
        if pd.isna(value):
            return 0.0
        
        # 如果已经是数字，直接返回
        if isinstance(value, (int, float)):
            return float(value)
        
        # 如果是字符串，清理格式
        if isinstance(value, str):
            # 移除逗号和其他分隔符
            clean_value = value.replace(',', '').replace(' ', '').strip()
            
            # 处理范围值（如"45-48"），取平均值
            if '-' in clean_value:
                parts = clean_value.split('-')
                if len(parts) == 2:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        return (start + end) / 2
                    except:
                        pass
            
            # 尝试转换为数字
            try:
                return float(clean_value)
            except:
                return 0.0
        
        return 0.0

    def convert_gas_to_bcm(self, value, unit: str) -> float:
        """将天然气相关数据转换为BCM/年"""
        clean_value = self.clean_numeric_value(value)
        if clean_value == 0:
            return 0.0
            
        normalized_unit = self.normalize_unit(unit)
        
        if normalized_unit in self.gas_to_bcm_per_year:
            conversion_factor = self.gas_to_bcm_per_year[normalized_unit]
            return clean_value * conversion_factor
        else:
            raise ValueError(f"未知的天然气单位: {unit} (标准化后: {normalized_unit})")
    
    def convert_oil_to_mtpa(self, value, unit: str) -> float:
        """将石油相关数据转换为MTPA"""
        clean_value = self.clean_numeric_value(value)
        if clean_value == 0:
            return 0.0
            
        normalized_unit = self.normalize_unit(unit)
        
        if normalized_unit in self.oil_to_mtpa:
            conversion_factor = self.oil_to_mtpa[normalized_unit]
            return clean_value * conversion_factor
        else:
            raise ValueError(f"未知的石油单位: {unit} (标准化后: {normalized_unit})")
    
    def get_conversion_summary(self) -> Dict:
        """获取转换系数总结"""
        return {
            'gas_units_supported': len(self.gas_to_bcm_per_year),
            'oil_units_supported': len(self.oil_to_mtpa),
            'gas_units': list(self.gas_to_bcm_per_year.keys()),
            'oil_units': list(self.oil_to_mtpa.keys())
        }
    
    def validate_conversions(self) -> Dict:
        """验证转换系数的合理性"""
        validation_results = {}
        
        # 验证一些关键转换
        test_cases = [
            ('gas', 1.0, 'bcm/y', 1.0),
            ('gas', 1000.0, 'MMSCMD', 365.0),  # 1000 MMSCMD = 365 BCM/年
            ('gas', 1.0, 'mtpa', 1.36),  # 1 MTPA LNG = 1.36 BCM
            ('oil', 1000.0, 'bpd', 0.05005),  # 1000桶/天 ≈ 0.05 MTPA
            ('oil', 1.0, 'mtpa', 1.0),
        ]
        
        for fuel_type, input_val, unit, expected_output in test_cases:
            try:
                if fuel_type == 'gas':
                    result = self.convert_gas_to_bcm(input_val, unit)
                else:
                    result = self.convert_oil_to_mtpa(input_val, unit)
                validation_results[f"{input_val} {unit}"] = {
                    'expected': expected_output,
                    'actual': result,
                    'match': abs(result - expected_output) < 0.001
                }
            except Exception as e:
                validation_results[f"{input_val} {unit}"] = {'error': str(e)}
        
        return validation_results

# 使用示例
if __name__ == "__main__":
    converter = UniversalUnitConverter()
    
    print("=== 单位转换器测试 ===")
    print(f"支持的转换: {converter.get_conversion_summary()}")
    
    print("\n=== 验证转换系数 ===")
    validation = converter.validate_conversions()
    for test, result in validation.items():
        print(f"{test}: {result}")