#!/usr/bin/env python3
"""
性能测试脚本 - 分步测试各个组件的性能
"""

import time
import pandas as pd
from pathlib import Path
import logging
from main import VariableConstructor
from timeseries_ovi_builder import TimeSeriesOVIBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def time_function(func, name):
    """计时装饰器"""
    start_time = time.time()
    try:
        result = func()
        end_time = time.time()
        logger.info(f"✅ {name} 完成，耗时: {end_time - start_time:.2f}秒")
        return result, True
    except Exception as e:
        end_time = time.time()
        logger.error(f"❌ {name} 失败，耗时: {end_time - start_time:.2f}秒，错误: {str(e)}")
        return None, False

def main():
    logger.info("🚀 开始性能测试...")
    
    # 测试1: 初始化
    logger.info("=" * 50)
    logger.info("测试1: 初始化VariableConstructor")
    constructor = VariableConstructor()
    logger.info("✅ 初始化完成")
    
    # 测试2: 宏观数据（从缓存加载）
    logger.info("=" * 50)
    logger.info("测试2: 加载宏观控制变量")
    macro_data, success = time_function(constructor.fetch_macro_controls, "宏观数据加载")
    if success and macro_data is not None:
        logger.info(f"   数据规模: {len(macro_data)}行 x {len(macro_data.columns)}列")
    
    # 测试3: 基础数据加载
    logger.info("=" * 50)
    logger.info("测试3: 加载基础数据")
    base_data, success = time_function(constructor.load_base_data, "基础数据加载")
    if success:
        logger.info(f"   数据集数量: {len(base_data)}")
    
    # 测试4: 直接测试OVI构建器
    logger.info("=" * 50)
    logger.info("测试4: 直接测试OVI构建器")
    
    def build_ovi():
        builder = TimeSeriesOVIBuilder(Path('08data'))
        return builder.build_complete_ovi_timeseries()
    
    ovi_result, success = time_function(build_ovi, "OVI时间序列构建")
    if success:
        gas_ovi, oil_ovi = ovi_result
        if gas_ovi is not None:
            logger.info(f"   天然气OVI: {len(gas_ovi)}条记录")
        
    # 测试5: Node-DLI_US构建
    logger.info("=" * 50)
    logger.info("测试5: Node-DLI_US构建")
    
    def build_node_dli():
        constructor.base_data = base_data if 'base_data' in locals() else {}
        return constructor._construct_node_dli_us()
    
    node_dli, success = time_function(build_node_dli, "Node-DLI_US构建")
    if success and node_dli is not None:
        logger.info(f"   Node-DLI数据: {len(node_dli)}条记录")
        
    # 测试6: Vul_US构建  
    logger.info("=" * 50)
    logger.info("测试6: Vul_US构建")
    
    def build_vul():
        constructor.base_data = base_data if 'base_data' in locals() else {}
        return constructor._construct_vul_us()
        
    vul_us, success = time_function(build_vul, "Vul_US构建")
    if success and vul_us is not None:
        logger.info(f"   Vul_US数据: {len(vul_us)}条记录")
        
    # 测试7: US产量冲击构建
    logger.info("=" * 50)
    logger.info("测试7: US产量冲击构建")
    
    us_shock, success = time_function(constructor._construct_us_prod_shock, "US产量冲击构建")
    if success and us_shock is not None:
        logger.info(f"   US产量冲击: {len(us_shock)}条记录")
    
    logger.info("=" * 50)  
    logger.info("🎉 性能测试完成！")

if __name__ == "__main__":
    main()