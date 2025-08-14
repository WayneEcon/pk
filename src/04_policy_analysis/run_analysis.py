#!/usr/bin/env python3
"""
政策影响分析运行脚本
提供简化的接口来运行政策影响分析和查看结果
"""

import logging
from pathlib import Path
import sys

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("🇺🇸 美国能源独立政策影响分析")
    print("=" * 40)
    
    while True:
        print("\n选择操作:")
        print("1. 🚀 运行完整政策分析 (数据分析+生成图表)")
        print("2. 📊 仅生成可视化图表")
        print("3. 🎨 查看和整理图表")
        print("4. 📋 查看分析结果摘要")
        print("5. 🚪 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == '1':
            print("\n🚀 开始运行完整政策分析...")
            try:
                from main import run_full_policy_analysis
                success = run_full_policy_analysis()
                if success:
                    print("✅ 政策分析完成！图表已保存到 figures/ 文件夹")
                    print("💡 建议使用选项3查看生成的图表")
                else:
                    print("❌ 政策分析失败，请检查日志")
            except Exception as e:
                print(f"❌ 运行失败: {e}")
                logger.error(f"政策分析失败: {e}")
        
        elif choice == '2':
            print("\n📊 仅生成可视化图表...")
            try:
                from main import run_visualization_only
                success = run_visualization_only()
                if success:
                    print("✅ 图表生成完成！")
                    print("💡 建议使用选项3查看生成的图表")
                else:
                    print("❌ 图表生成失败，请检查日志")
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                logger.error(f"图表生成失败: {e}")
        
        elif choice == '3':
            print("\n🎨 启动图表查看工具...")
            try:
                from view_figures import main as view_main
                view_main()
            except Exception as e:
                print(f"❌ 启动失败: {e}")
                logger.error(f"图表查看工具失败: {e}")
        
        elif choice == '4':
            print("\n📋 查看分析结果摘要...")
            try:
                # 检查分析结果文件
                summary_file = Path(__file__).parent / "policy_impact_summary.csv"
                stats_file = Path(__file__).parent / "policy_impact_statistics.json"
                
                if summary_file.exists():
                    import pandas as pd
                    summary_df = pd.read_csv(summary_file)
                    print(f"\n📊 对比分析结果 ({len(summary_df)} 个国家):")
                    print("=" * 50)
                    
                    # 显示前5个变化最大的国家
                    if 'total_strength_change' in summary_df.columns:
                        top_changes = summary_df.nlargest(5, 'total_strength_change')[
                            ['country_code', 'total_strength_change']
                        ]
                        print("🔝 总强度变化最大的5个国家:")
                        for _, row in top_changes.iterrows():
                            change = row['total_strength_change']
                            direction = "📈" if change > 0 else "📉"
                            print(f"  {direction} {row['country_code']}: {change:+.3f}")
                    
                    print(f"\n💾 详细结果文件: {summary_file.name}")
                else:
                    print("❌ 未找到分析结果文件，请先运行分析")
                
                if stats_file.exists():
                    import json
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    print(f"\n📈 统计结果概览:")
                    print("=" * 50)
                    
                    if 'summary' in stats:
                        summary = stats['summary']
                        for metric, data in summary.items():
                            if isinstance(data, dict) and 'mean_change' in data:
                                mean_change = data['mean_change']
                                direction = "📈" if mean_change > 0 else "📉"
                                print(f"  {direction} {metric}: {mean_change:+.4f}")
                    
                    print(f"\n💾 详细统计文件: {stats_file.name}")
                else:
                    print("❌ 未找到统计结果文件，请先运行分析")
                    
            except Exception as e:
                print(f"❌ 读取结果失败: {e}")
                logger.error(f"结果摘要失败: {e}")
        
        elif choice == '5':
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选项，请重新选择")

if __name__ == "__main__":
    main()