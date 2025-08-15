#!/usr/bin/env python3
"""
第六章骨干网络分析模块测试运行器
============================

运行所有单元测试并生成测试报告。

使用方法：
    python run_tests.py              # 运行所有测试
    python run_tests.py -v          # 详细模式
    python run_tests.py -k pattern  # 运行匹配模式的测试

作者：Energy Network Analysis Team
"""

import unittest
import sys
import argparse
from pathlib import Path
import time

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

def discover_and_run_tests(verbosity=1, pattern=None, coverage_report=False):
    """发现并运行所有测试"""
    
    # 初始化coverage（如果需要）
    cov = None
    if coverage_report:
        if COVERAGE_AVAILABLE:
            cov = coverage.Coverage()
            cov.start()
            print("📊 启动代码覆盖率监控...")
        else:
            print("⚠️ coverage模块未安装，跳过覆盖率报告")
            coverage_report = False
    
    # 发现测试
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 如果有模式过滤，进一步过滤测试
    if pattern:
        filtered_suite = unittest.TestSuite()
        for test_group in suite:
            for test_case in test_group:
                if hasattr(test_case, '_testMethodName'):
                    if pattern.lower() in test_case._testMethodName.lower():
                        filtered_suite.addTest(test_case)
                elif hasattr(test_case, '_tests'):
                    # 处理嵌套的测试套件
                    for test in test_case._tests:
                        if hasattr(test, '_testMethodName'):
                            if pattern.lower() in test._testMethodName.lower():
                                filtered_suite.addTest(test)
        suite = filtered_suite
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    
    print("🧪 开始运行第六章骨干网络分析模块测试...")
    print("=" * 70)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    print("=" * 70)
    print(f"⏱️ 测试运行时间: {end_time - start_time:.2f} 秒")
    
    # 打印测试统计
    print(f"📈 测试统计:")
    print(f"   总测试数: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    print(f"   跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    # 打印失败和错误详情
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
    
    if result.errors:
        print("\n🔥 错误的测试:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip() if 'Error:' in traceback else 'See details above'}")
    
    # 生成覆盖率报告
    if coverage_report and cov:
        cov.stop()
        cov.save()
        
        print("\n📊 代码覆盖率报告:")
        print("-" * 50)
        
        # 控制台报告
        cov.report()
        
        # HTML报告
        try:
            html_dir = Path(__file__).parent / "coverage_html"
            cov.html_report(directory=str(html_dir))
            print(f"\n📄 详细HTML覆盖率报告已生成: {html_dir}/index.html")
        except Exception as e:
            print(f"⚠️ HTML报告生成失败: {e}")
    
    # 返回测试结果
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print(f"\n❌ 有 {len(result.failures) + len(result.errors)} 个测试失败")
    
    return success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行第六章骨干网络分析模块测试')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='详细模式输出')
    parser.add_argument('-k', '--pattern', type=str,
                       help='只运行匹配模式的测试')
    parser.add_argument('-c', '--coverage', action='store_true',
                       help='生成代码覆盖率报告')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的测试')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出所有测试
        loader = unittest.TestLoader()
        start_dir = Path(__file__).parent
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        print("📋 可用的测试:")
        print("-" * 50)
        
        def print_tests(test_suite, indent=0):
            if hasattr(test_suite, '_tests'):
                for test in test_suite._tests:
                    print_tests(test, indent)
            else:
                test_name = str(test).split(' ')[0]
                print("  " * indent + f"• {test_name}")
        
        print_tests(suite)
        return
    
    # 运行测试
    verbosity = 2 if args.verbose else 1
    success = discover_and_run_tests(
        verbosity=verbosity,
        pattern=args.pattern,
        coverage_report=args.coverage
    )
    
    # 设置退出码
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()