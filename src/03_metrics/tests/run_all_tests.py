#!/usr/bin/env python3
"""
03_metrics模块测试运行器

运行所有单元测试和集成测试，生成测试报告。

使用方法:
    python run_all_tests.py
    python run_all_tests.py --verbose    # 详细输出
    python run_all_tests.py --module utils  # 只测试特定模块
"""

import unittest
import sys
import time
import argparse
from pathlib import Path
from io import StringIO

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入所有测试模块
from tests.test_utils import *
from tests.test_node_metrics import *
from tests.test_global_metrics import *
from tests.test_integration import *


class ColoredTextTestResult(unittest.TextTestResult):
    """带颜色输出的测试结果类"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.writeln(f"✅ {test._testMethodName}")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.writeln(f"❌ {test._testMethodName} - ERROR")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.writeln(f"❌ {test._testMethodName} - FAIL")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.writeln(f"⏭️  {test._testMethodName} - SKIPPED")


class TestSuite:
    """测试套件管理器"""
    
    def __init__(self):
        self.modules = {
            'utils': ['test_utils'],
            'node_metrics': ['test_node_metrics'],
            'global_metrics': ['test_global_metrics'],
            'integration': ['test_integration'],
            'all': ['test_utils', 'test_node_metrics', 'test_global_metrics', 'test_integration']
        }
    
    def create_suite(self, module_name='all'):
        """创建测试套件"""
        if module_name not in self.modules:
            raise ValueError(f"未知模块: {module_name}. 可用模块: {list(self.modules.keys())}")
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        test_modules = self.modules[module_name]
        
        for module in test_modules:
            if module == 'test_utils':
                # utils模块的测试类
                suite.addTest(loader.loadTestsFromTestCase(TestSetupLogger))
                suite.addTest(loader.loadTestsFromTestCase(TestValidateGraph))
                suite.addTest(loader.loadTestsFromTestCase(TestAddDistanceAttribute))
                suite.addTest(loader.loadTestsFromTestCase(TestSafeDivide))
                suite.addTest(loader.loadTestsFromTestCase(TestGetGraphBasicStats))
                suite.addTest(loader.loadTestsFromTestCase(TestCreateEmptyDataFrames))
                suite.addTest(loader.loadTestsFromTestCase(TestTimerContext))
                suite.addTest(loader.loadTestsFromTestCase(TestValidateYear))
                suite.addTest(loader.loadTestsFromTestCase(TestLogFunctions))
                
            elif module == 'test_node_metrics':
                # node_metrics模块的测试类
                suite.addTest(loader.loadTestsFromTestCase(TestNodeMetricsCalculation))
                suite.addTest(loader.loadTestsFromTestCase(TestNodeMetricsEdgeCases))
                suite.addTest(loader.loadTestsFromTestCase(TestNodeMetricsUtility))
                suite.addTest(loader.loadTestsFromTestCase(TestNodeMetricsValidation))
                
            elif module == 'test_global_metrics':
                # global_metrics模块的测试类
                suite.addTest(loader.loadTestsFromTestCase(TestGlobalMetricsCalculation))
                suite.addTest(loader.loadTestsFromTestCase(TestGlobalMetricsEdgeCases))
                suite.addTest(loader.loadTestsFromTestCase(TestGlobalMetricsUtilities))
                suite.addTest(loader.loadTestsFromTestCase(TestGlobalMetricsValidation))
                suite.addTest(loader.loadTestsFromTestCase(TestGlobalMetricsCorrectness))
                
            elif module == 'test_integration':
                # integration测试类
                suite.addTest(loader.loadTestsFromTestCase(TestMainInterface))
                suite.addTest(loader.loadTestsFromTestCase(TestDataProcessingAndExport))
                suite.addTest(loader.loadTestsFromTestCase(TestErrorHandlingAndRobustness))
                suite.addTest(loader.loadTestsFromTestCase(TestPerformanceAndCaching))
                suite.addTest(loader.loadTestsFromTestCase(TestDataConsistencyAndIntegrity))
        
        return suite
    
    def run_tests(self, module_name='all', verbosity=1):
        """运行测试并返回结果"""
        print(f"\n🧪 运行 03_metrics 模块测试 - {module_name}")
        print("=" * 60)
        
        suite = self.create_suite(module_name)
        
        # 自定义测试运行器
        stream = StringIO() if verbosity == 0 else sys.stdout
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=verbosity,
            resultclass=ColoredTextTestResult
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # 打印测试总结
        self.print_summary(result, end_time - start_time, module_name)
        
        return result
    
    def print_summary(self, result, duration, module_name):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print(f"📊 测试总结 - {module_name} 模块")
        print("=" * 60)
        
        total_tests = result.testsRun
        success_count = getattr(result, 'success_count', total_tests - len(result.errors) - len(result.failures))
        error_count = len(result.errors)
        failure_count = len(result.failures)
        skip_count = len(result.skipped)
        
        print(f"总计测试: {total_tests}")
        print(f"✅ 通过: {success_count}")
        print(f"❌ 失败: {failure_count}")
        print(f"💥 错误: {error_count}")
        print(f"⏭️  跳过: {skip_count}")
        print(f"⏱️  用时: {duration:.2f} 秒")
        
        # 成功率
        success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
        print(f"✨ 成功率: {success_rate:.1f}%")
        
        # 详细错误信息
        if result.errors:
            print(f"\n🚫 错误详情 ({len(result.errors)} 个):")
            for i, (test, error) in enumerate(result.errors, 1):
                error_msg = error.split('\n')[-2] if error else '未知错误'
                print(f"  {i}. {test}: {error_msg}")
        
        if result.failures:
            print(f"\n🔴 失败详情 ({len(result.failures)} 个):")
            for i, (test, failure) in enumerate(result.failures, 1):
                failure_msg = failure.split('\n')[-2] if failure else '未知失败'
                print(f"  {i}. {test}: {failure_msg}")
        
        # 整体评估
        if error_count == 0 and failure_count == 0:
            print(f"\n🎉 所有测试通过！{module_name} 模块运行正常。")
        elif success_rate >= 90:
            print(f"\n⚠️  大部分测试通过，但存在少量问题需要处理。")
        else:
            print(f"\n🚨 存在较多测试问题，需要重点关注。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='03_metrics模块测试运行器')
    parser.add_argument('--module', '-m', 
                       choices=['utils', 'node_metrics', 'global_metrics', 'integration', 'all'],
                       default='all',
                       help='指定要测试的模块')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默输出（只显示总结）')
    
    args = parser.parse_args()
    
    # 设置日志级别
    import logging
    if args.quiet:
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 确定verbosity级别
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # 运行测试
    test_suite = TestSuite()
    
    try:
        result = test_suite.run_tests(args.module, verbosity)
        
        # 根据测试结果设置退出码
        if result.errors or result.failures:
            sys.exit(1)  # 有错误或失败
        else:
            sys.exit(0)  # 所有测试通过
            
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 测试运行异常: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()