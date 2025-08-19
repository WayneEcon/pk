#!/usr/bin/env python3
"""
图表查看和整理工具
用于查看和整理06_policy_analysis模块生成的可视化图表
"""

import os
from pathlib import Path
from typing import Dict, List
import webbrowser
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_figures() -> Dict[str, List[str]]:
    """
    列出所有图表文件并按类型分类
    
    Returns:
        按类型分类的图表文件字典
    """
    figures_dir = Path(__file__).parent / "figures"
    
    if not figures_dir.exists():
        logger.error(f"图表目录不存在: {figures_dir}")
        return {}
    
    # 获取所有PNG文件
    png_files = list(figures_dir.glob("*.png"))
    
    if not png_files:
        logger.warning("未找到任何PNG图表文件")
        return {}
    
    # 按类型分类
    categorized = {
        "国家仪表盘": [],
        "期间对比图": [],
        "综合分析图": []
    }
    
    for file_path in png_files:
        filename = file_path.name
        
        if "_dashboard.png" in filename:
            categorized["国家仪表盘"].append(filename)
        elif "_period_comparison.png" in filename:
            categorized["期间对比图"].append(filename)
        else:
            categorized["综合分析图"].append(filename)
    
    # 排序
    for category in categorized:
        categorized[category].sort()
    
    return categorized

def print_figure_summary():
    """打印图表摘要信息"""
    categorized = list_figures()
    
    if not categorized:
        print("❌ 未找到任何图表文件")
        return
    
    print("📊 政策影响分析图表摘要")
    print("=" * 50)
    
    total_count = 0
    for category, files in categorized.items():
        count = len(files)
        total_count += count
        print(f"\n📁 {category} ({count} 个):")
        for filename in files[:5]:  # 只显示前5个
            print(f"  • {filename}")
        if count > 5:
            print(f"  ... 还有 {count - 5} 个文件")
    
    print(f"\n🎯 总计: {total_count} 个图表文件")

def create_html_gallery():
    """
    创建HTML图表画廊，方便在浏览器中查看所有图表
    
    Returns:
        生成的HTML文件路径
    """
    categorized = list_figures()
    
    if not categorized:
        logger.error("无图表文件，无法创建画廊")
        return None
    
    figures_dir = Path(__file__).parent / "figures"
    html_file = figures_dir / "gallery.html"
    
    # 生成HTML内容
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>政策影响分析图表画廊</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .category {
            margin-bottom: 40px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .category h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .figure-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
            text-align: center;
        }
        .figure-card img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        .figure-card img:hover {
            transform: scale(1.05);
        }
        .figure-title {
            font-weight: bold;
            margin-top: 10px;
            color: #2c3e50;
            font-size: 14px;
        }
        .stats {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }
        .stats h3 {
            margin: 0;
            color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇺🇸 美国能源独立政策影响分析图表画廊</h1>
        
        <div class="stats">
            <h3>📊 图表统计</h3>
'''
    
    # 添加统计信息
    total_count = sum(len(files) for files in categorized.values())
    html_content += f'<p>总计 <strong>{total_count}</strong> 个图表，分为 <strong>{len(categorized)}</strong> 个类别</p>\n'
    
    for category, files in categorized.items():
        html_content += f'<span style="margin: 0 10px; color: #7f8c8d;">{category}: {len(files)}个</span>\n'
    
    html_content += '</div>\n\n'
    
    # 添加各类别的图表
    for category, files in categorized.items():
        html_content += f'''
        <div class="category">
            <h2>📁 {category} ({len(files)} 个)</h2>
            <div class="gallery">
'''
        
        for filename in files:
            # 生成友好的标题
            title = filename.replace('.png', '').replace('_', ' ').title()
            html_content += f'''
                <div class="figure-card">
                    <img src="{filename}" alt="{title}" onclick="window.open('{filename}', '_blank')">
                    <div class="figure-title">{title}</div>
                </div>
'''
        
        html_content += '''
            </div>
        </div>
'''
    
    # 完成HTML
    html_content += '''
        <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
            <p>生成时间: ''' + '''</p>
            <p>点击图片可在新窗口中查看大图</p>
        </div>
    </div>

    <script>
        // 添加简单的统计和交互功能
        console.log("图表画廊加载完成");
        
        // 添加键盘导航支持
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                // ESC键关闭任何打开的大图
                console.log('ESC pressed');
            }
        });
    </script>
</body>
</html>
'''
    
    # 写入HTML文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✅ HTML图表画廊已生成: {html_file}")
    return str(html_file)

def open_gallery():
    """在默认浏览器中打开图表画廊"""
    html_file = create_html_gallery()
    
    if html_file and Path(html_file).exists():
        try:
            webbrowser.open(f"file://{html_file}")
            logger.info(f"🌐 已在浏览器中打开图表画廊")
        except Exception as e:
            logger.error(f"打开浏览器失败: {e}")
            print(f"请手动打开文件: {html_file}")
    else:
        logger.error("HTML画廊生成失败")

def main():
    """主函数"""
    print("🎨 图表查看和整理工具")
    print("=" * 40)
    
    while True:
        print("\n选择操作:")
        print("1. 📋 显示图表摘要")
        print("2. 🌐 创建并打开HTML画廊")
        print("3. 📁 打开图表文件夹")
        print("4. 🚪 退出")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            print_figure_summary()
        elif choice == '2':
            open_gallery()
        elif choice == '3':
            figures_dir = Path(__file__).parent / "figures"
            if figures_dir.exists():
                try:
                    # macOS
                    os.system(f"open '{figures_dir}'")
                    logger.info(f"📁 已打开图表文件夹: {figures_dir}")
                except:
                    print(f"请手动打开文件夹: {figures_dir}")
            else:
                print("❌ 图表文件夹不存在")
        elif choice == '4':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

if __name__ == "__main__":
    main()