#!/usr/bin/env python3
"""
测试运行器
用于运行所有单元测试
"""

import unittest
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.CRITICAL)  # 只显示严重错误

def run_all_tests():
    """运行所有测试"""
    # 发现并运行所有测试
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    # 发现所有测试模块
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()

def run_specific_test(test_module):
    """运行特定测试模块"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 开始运行 net_02 模块单元测试...")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # 运行特定测试
        test_module = sys.argv[1]
        print(f"运行测试模块: {test_module}")
        success = run_specific_test(test_module)
    else:
        # 运行所有测试
        print("运行所有测试...")
        success = run_all_tests()
    
    print("=" * 60)
    
    if success:
        print("✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("❌ 部分测试失败!")
        sys.exit(1)