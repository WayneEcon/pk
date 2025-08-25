#!/usr/bin/env python3
"""
093 LNG-only优化结果可视化生成器
专门用于生成修复后的LP-IRF图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_lng_visualization():
    """创建093 LNG-only优化的可视化图表"""
    
    # 读取结果数据
    full_sample_file = Path("outputs/final_analysis_results_full_sample.json")
    core_sample_file = Path("outputs/final_analysis_results_core_importers.json")
    
    # 确保图表目录存在
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # 创建双样本对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 全样本价格通道
    try:
        with open(full_sample_file, 'r', encoding='utf-8') as f:
            full_results = json.load(f)
        
        price_results = full_results['models']['price_channel']
        if price_results['status'] == 'success':
            horizons = []
            coefs = []
            ci_lower = []
            ci_upper = []
            
            for h in sorted(price_results['horizon_results'].keys()):
                result = price_results['horizon_results'][h]
                horizons.append(int(h))
                coefs.append(result['theta_coefficient'])
                ci_lower.append(result['theta_ci_lower'])
                ci_upper.append(result['theta_ci_upper'])
            
            # 绘制全样本价格通道
            ax1.plot(horizons, coefs, 'o-', color='#2E8B57', linewidth=3, 
                    markersize=10, label='θ_h (交互项系数)', markerfacecolor='white', 
                    markeredgewidth=3, markeredgecolor='#2E8B57')
            ax1.fill_between(horizons, ci_lower, ci_upper, 
                            alpha=0.25, color='#2E8B57', label='95%置信区间')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # 添加数值标签
            for h, coef in zip(horizons, coefs):
                ax1.annotate(f'{coef:.2f}', (h, coef), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=11, fontweight='bold', color='#2E8B57')
            
            ax1.set_title('全样本价格通道：LNG-only优化\\n(US Shock × ln(1+OVI) → log(P_lng))', 
                         fontsize=13, fontweight='bold', color='darkgreen')
            ax1.set_xlabel('预测期 h (年)', fontsize=12)
            ax1.set_ylabel('交互项系数 θ_h', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.legend(fontsize=10)
            ax1.set_xticks(horizons)
        
    except Exception as e:
        ax1.text(0.5, 0.5, f'全样本价格数据加载失败: {e}', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12, color='red')
    
    # 2. 全样本数量通道
    try:
        quantity_results = full_results['models']['quantity_channel']
        if quantity_results['status'] == 'success':
            horizons = []
            coefs = []
            ci_lower = []
            ci_upper = []
            
            for h in sorted(quantity_results['horizon_results'].keys()):
                result = quantity_results['horizon_results'][h]
                horizons.append(int(h))
                coefs.append(result['theta_coefficient'])
                ci_lower.append(result['theta_ci_lower'])
                ci_upper.append(result['theta_ci_upper'])
            
            # 绘制全样本数量通道
            ax2.plot(horizons, coefs, 'o-', color='#CD853F', linewidth=3,
                    markersize=10, label='θ_h (交互项系数)', markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor='#CD853F')
            ax2.fill_between(horizons, ci_lower, ci_upper,
                            alpha=0.25, color='#CD853F', label='95%置信区间')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # 添加数值标签
            for h, coef in zip(horizons, coefs):
                ax2.annotate(f'{coef:.2f}', (h, coef), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=11, fontweight='bold', color='#CD853F')
            
            ax2.set_title('全样本数量通道\\n(US Shock × OVI → Import Quantity)', 
                         fontsize=13, fontweight='bold', color='#B8860B')
            ax2.set_xlabel('预测期 h (年)', fontsize=12)
            ax2.set_ylabel('交互项系数 θ_h', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle=':')
            ax2.legend(fontsize=10)
            ax2.set_xticks(horizons)
        
    except Exception as e:
        ax2.text(0.5, 0.5, f'全样本数量数据加载失败: {e}', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, color='red')
    
    # 3. 核心样本价格通道
    try:
        with open(core_sample_file, 'r', encoding='utf-8') as f:
            core_results = json.load(f)
        
        price_results = core_results['models']['price_channel']
        if price_results['status'] == 'success':
            horizons = []
            coefs = []
            ci_lower = []
            ci_upper = []
            
            for h in sorted(price_results['horizon_results'].keys()):
                result = price_results['horizon_results'][h]
                horizons.append(int(h))
                coefs.append(result['theta_coefficient'])
                ci_lower.append(result['theta_ci_lower'])
                ci_upper.append(result['theta_ci_upper'])
            
            # 绘制核心样本价格通道
            ax3.plot(horizons, coefs, 'o-', color='#4169E1', linewidth=3, 
                    markersize=10, label='θ_h (交互项系数)', markerfacecolor='white', 
                    markeredgewidth=3, markeredgecolor='#4169E1')
            ax3.fill_between(horizons, ci_lower, ci_upper, 
                            alpha=0.25, color='#4169E1', label='95%置信区间')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # 添加数值标签
            for h, coef in zip(horizons, coefs):
                ax3.annotate(f'{coef:.2f}', (h, coef), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=11, fontweight='bold', color='#4169E1')
            
            ax3.set_title('核心样本价格通道：LNG-only优化\\n(30个主要LNG进口国)', 
                         fontsize=13, fontweight='bold', color='darkblue')
            ax3.set_xlabel('预测期 h (年)', fontsize=12)
            ax3.set_ylabel('交互项系数 θ_h', fontsize=12)
            ax3.grid(True, alpha=0.3, linestyle=':')
            ax3.legend(fontsize=10)
            ax3.set_xticks(horizons)
        
    except Exception as e:
        ax3.text(0.5, 0.5, f'核心样本价格数据加载失败: {e}', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12, color='red')
    
    # 4. 核心样本数量通道
    try:
        quantity_results = core_results['models']['quantity_channel']
        if quantity_results['status'] == 'success':
            horizons = []
            coefs = []
            ci_lower = []
            ci_upper = []
            
            for h in sorted(quantity_results['horizon_results'].keys()):
                result = quantity_results['horizon_results'][h]
                horizons.append(int(h))
                coefs.append(result['theta_coefficient'])
                ci_lower.append(result['theta_ci_lower'])
                ci_upper.append(result['theta_ci_upper'])
            
            # 绘制核心样本数量通道
            ax4.plot(horizons, coefs, 'o-', color='#DC143C', linewidth=3,
                    markersize=10, label='θ_h (交互项系数)', markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor='#DC143C')
            ax4.fill_between(horizons, ci_lower, ci_upper,
                            alpha=0.25, color='#DC143C', label='95%置信区间')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # 添加数值标签
            for h, coef in zip(horizons, coefs):
                ax4.annotate(f'{coef:.2f}', (h, coef), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=11, fontweight='bold', color='#DC143C')
            
            ax4.set_title('核心样本数量通道\\n(30个主要LNG进口国)', 
                         fontsize=13, fontweight='bold', color='darkred')
            ax4.set_xlabel('预测期 h (年)', fontsize=12)
            ax4.set_ylabel('交互项系数 θ_h', fontsize=12)
            ax4.grid(True, alpha=0.3, linestyle=':')
            ax4.legend(fontsize=10)
            ax4.set_xticks(horizons)
        
    except Exception as e:
        ax4.text(0.5, 0.5, f'核心样本数量数据加载失败: {e}', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12, color='red')
    
    # 设置总标题
    fig.suptitle('093 LNG-only严格优化：双样本LP-IRF对比分析\\n' + 
                 'GPT-5建议优化 | h=[0,1]预测期 | ln(1+OVI)交互项 | 平衡面板', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    output_file = figures_dir / "093_lng_only_optimization_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 093 LNG-only优化可视化图表已保存: {output_file}")
    
    # 生成摘要
    print("\n📊 093 LNG-only优化摘要:")
    print(f"   全样本价格通道: θ_0={full_results['models']['price_channel']['horizon_results']['0']['theta_coefficient']:.3f} (p={full_results['models']['price_channel']['horizon_results']['0']['theta_p_value']:.3f})")
    print(f"   全样本数量通道: θ_0={full_results['models']['quantity_channel']['horizon_results']['0']['theta_coefficient']:.3f} (p={full_results['models']['quantity_channel']['horizon_results']['0']['theta_p_value']:.3f})")
    print(f"   核心样本价格通道: θ_0={core_results['models']['price_channel']['horizon_results']['0']['theta_coefficient']:.3f} (p={core_results['models']['price_channel']['horizon_results']['0']['theta_p_value']:.3f})")
    print(f"   核心样本数量通道: θ_0={core_results['models']['quantity_channel']['horizon_results']['0']['theta_coefficient']:.3f} (p={core_results['models']['quantity_channel']['horizon_results']['0']['theta_p_value']:.3f})")
    
    plt.close()

if __name__ == "__main__":
    create_lng_visualization()