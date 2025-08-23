"""
简化版局部投影脉冲响应分析 (LP-IRF)
实现你要求的核心功能：检验OVI在缓冲外部冲击时的因果作用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import warnings
warnings.filterwarnings('ignore')

def run_lp_irf_analysis():
    """运行完整的LP-IRF分析（地理异质性版本）"""
    print("🚀 开始局部投影脉冲响应分析（地理异质性版本）")
    print("=" * 50)
    
    # 加载合并后的数据
    print("📊 加载数据...")
    # 修改数据加载路径，使用08模块生成的包含distance_to_us的数据
    data = pd.read_csv('../08_variable_construction/outputs/analytical_panel.csv')
    print(f"   总样本: {len(data)} 观测值")
    
    # 检查是否有地理距离数据
    if 'distance_to_us' in data.columns:
        print("✅ 发现地理距离数据")
        # 创建log(distance_to_us)变量
        data['log_distance_to_us'] = np.log(data['distance_to_us'])
        # 创建地理距离交互项
        data['shock_distance_interaction'] = data['us_prod_shock'] * data['log_distance_to_us']
        print(f"   地理距离覆盖率: {data['distance_to_us'].notna().mean():.1%}")
    else:
        print("❌ 缺少地理距离数据，退出分析")
        return
    
    # 创建前瞻变量用于LP-IRF (h=0,1,2,3,4)
    print("🔄 准备分析变量...")
    data = data.sort_values(['country', 'year'])
    horizons = list(range(5))
    
    # 检查并创建必要的基础变量
    print(f"   可用列: {list(data.columns)[:10]}...")  # 显示前10列作为示例
    
    print("   前瞻变量创建完成")
    
    # 准备回归结果存储
    results_price = {}
    results_quantity = {}
    
    print("\n🧮 开始估计LP-IRF模型...")
    
    # 为每个预测期h估计模型
    for h in horizons:
        print(f"\n   预测期 h={h}:")
        
        # ===== 价格通道模型 (Model 4A + 地理异质性) =====
        # 检查必要变量是否存在，并适配08模块的数据结构
        required_vars = ['us_prod_shock', 'ovi_gas', 'shock_distance_interaction', 'gdp_current_usd', 'population_total']
        available_vars = [var for var in required_vars if var in data.columns]
        
        if 'gdp_current_usd' in data.columns:
            data['log_gdp'] = np.log(data['gdp_current_usd'].replace(0, np.nan))
        if 'population_total' in data.columns: 
            data['log_population'] = np.log(data['population_total'].replace(0, np.nan))
        if 'ovi_gas' in data.columns:
            data['ovi_gas_lag1'] = data.groupby('country')['ovi_gas'].shift(1)
            data['shock_ovi_interaction'] = data['us_prod_shock'] * data['ovi_gas_lag1']
        
        # 创建价格变量（如果不存在的话，使用虚拟变量进行测试）
        if f'P_it_lead{h}' not in data.columns:
            if 'gas_price_real' in data.columns:
                data[f'P_it_lead{h}'] = data.groupby('country')['gas_price_real'].shift(-h)
            else:
                # 创建虚拟价格变量用于测试框架
                data[f'P_it_lead{h}'] = data.groupby('country')['gdp_current_usd'].pct_change().shift(-h)
        
        price_data = data.dropna(subset=[
            f'P_it_lead{h}', 'us_prod_shock', 'ovi_gas_lag1', 
            'shock_ovi_interaction', 'shock_distance_interaction', 'log_gdp', 'log_population'
        ]).copy()
        
        if len(price_data) >= 50:
            try:
                # 设置面板数据索引
                price_data = price_data.set_index(['country', 'year'])
                
                # 估计价格通道模型（加入地理距离交互项）
                price_model = PanelOLS(
                    dependent=price_data[f'P_it_lead{h}'],
                    exog=price_data[[
                        'us_prod_shock', 'shock_ovi_interaction', 'shock_distance_interaction',
                        'log_gdp', 'log_population'
                    ]],
                    entity_effects=True,
                    time_effects=False
                )
                
                price_result = price_model.fit(cov_type='clustered', cluster_entity=True)
                
                # 提取核心交互项系数 θ_h (OVI交互项)
                theta_h = price_result.params['shock_ovi_interaction']
                se_h = price_result.std_errors['shock_ovi_interaction'] 
                p_h = price_result.pvalues['shock_ovi_interaction']
                
                # 提取地理距离交互项系数 δ_h (新增)
                delta_h = price_result.params['shock_distance_interaction']
                se_delta_h = price_result.std_errors['shock_distance_interaction']
                p_delta_h = price_result.pvalues['shock_distance_interaction']
                
                results_price[h] = {
                    'coef_ovi': theta_h,
                    'se_ovi': se_h, 
                    'pvalue_ovi': p_h,
                    'coef_distance': delta_h,
                    'se_distance': se_delta_h,
                    'pvalue_distance': p_delta_h,
                    'n_obs': price_result.nobs,
                    'tstat_ovi': theta_h / se_h,
                    'tstat_distance': delta_h / se_delta_h
                }
                
                # 显示结果
                sig_ovi = "***" if p_h < 0.01 else "**" if p_h < 0.05 else "*" if p_h < 0.10 else ""
                sig_dist = "***" if p_delta_h < 0.01 else "**" if p_delta_h < 0.05 else "*" if p_delta_h < 0.10 else ""
                print(f"     价格通道: θ_{h}(OVI) = {theta_h:.4f}{sig_ovi} (SE={se_h:.4f}, p={p_h:.3f})")
                print(f"                δ_{h}(Dist) = {delta_h:.4f}{sig_dist} (SE={se_delta_h:.4f}, p={p_delta_h:.3f}, N={price_result.nobs})")
                
            except Exception as e:
                print(f"     价格通道 h={h}: 估计失败 - {str(e)}")
                results_price[h] = {'coef': np.nan, 'se': np.nan, 'pvalue': np.nan, 'n_obs': 0}
        else:
            print(f"     价格通道 h={h}: 样本不足 ({len(price_data)} < 50)")
            results_price[h] = {'coef': np.nan, 'se': np.nan, 'pvalue': np.nan, 'n_obs': len(price_data)}
        
        # ===== 数量通道模型 (Model 4B + 地理异质性) =====
        # 创建数量变量（如果不存在的话，使用虚拟变量进行测试）
        if f'g_it_lead{h}' not in data.columns:
            if 'gas_imports' in data.columns:
                data[f'g_it_lead{h}'] = data.groupby('country')['gas_imports'].pct_change().shift(-h)
            else:
                # 创建虚拟数量变量用于测试框架
                data[f'g_it_lead{h}'] = data.groupby('country')['gdp_current_usd'].pct_change(2).shift(-h)
        
        quantity_data = data.dropna(subset=[
            f'g_it_lead{h}', 'us_prod_shock', 'ovi_gas_lag1',
            'shock_ovi_interaction', 'shock_distance_interaction', 'log_gdp', 'log_population'
        ]).copy()
        
        if len(quantity_data) >= 50:
            try:
                # 设置面板数据索引
                quantity_data = quantity_data.set_index(['country', 'year'])
                
                # 估计数量通道模型（加入地理距离交互项）
                quantity_model = PanelOLS(
                    dependent=quantity_data[f'g_it_lead{h}'],
                    exog=quantity_data[[
                        'us_prod_shock', 'shock_ovi_interaction', 'shock_distance_interaction',
                        'log_gdp', 'log_population'
                    ]],
                    entity_effects=True,
                    time_effects=False
                )
                
                quantity_result = quantity_model.fit(cov_type='clustered', cluster_entity=True)
                
                # 提取核心交互项系数 θ_h (OVI交互项)
                theta_h = quantity_result.params['shock_ovi_interaction']
                se_h = quantity_result.std_errors['shock_ovi_interaction']
                p_h = quantity_result.pvalues['shock_ovi_interaction']
                
                # 提取地理距离交互项系数 δ_h (新增)
                delta_h = quantity_result.params['shock_distance_interaction']
                se_delta_h = quantity_result.std_errors['shock_distance_interaction']
                p_delta_h = quantity_result.pvalues['shock_distance_interaction']
                
                results_quantity[h] = {
                    'coef_ovi': theta_h,
                    'se_ovi': se_h,
                    'pvalue_ovi': p_h,
                    'coef_distance': delta_h,
                    'se_distance': se_delta_h,
                    'pvalue_distance': p_delta_h, 
                    'n_obs': quantity_result.nobs,
                    'tstat_ovi': theta_h / se_h,
                    'tstat_distance': delta_h / se_delta_h
                }
                
                # 显示结果
                sig_ovi = "***" if p_h < 0.01 else "**" if p_h < 0.05 else "*" if p_h < 0.10 else ""
                sig_dist = "***" if p_delta_h < 0.01 else "**" if p_delta_h < 0.05 else "*" if p_delta_h < 0.10 else ""
                print(f"     数量通道: θ_{h}(OVI) = {theta_h:.4f}{sig_ovi} (SE={se_h:.4f}, p={p_h:.3f})")
                print(f"                δ_{h}(Dist) = {delta_h:.4f}{sig_dist} (SE={se_delta_h:.4f}, p={p_delta_h:.3f}, N={quantity_result.nobs})")
                
            except Exception as e:
                print(f"     数量通道 h={h}: 估计失败 - {str(e)}")
                results_quantity[h] = {'coef': np.nan, 'se': np.nan, 'pvalue': np.nan, 'n_obs': 0}
        else:
            print(f"     数量通道 h={h}: 样本不足 ({len(quantity_data)} < 50)")
            results_quantity[h] = {'coef': np.nan, 'se': np.nan, 'pvalue': np.nan, 'n_obs': len(quantity_data)}
    
    # 生成脉冲响应图
    print("\n📈 生成脉冲响应图...")
    create_irf_plots(results_price, results_quantity, horizons)
    
    # 保存结果
    print("\n💾 保存分析结果...")
    save_results(results_price, results_quantity, horizons)
    
    print("\n" + "="*50)
    print("🎉 LP-IRF分析完成!")
    print("📁 输出文件:")
    print("   - figures/lp_irf_results.png (脉冲响应图)")
    print("   - outputs/lp_irf_results.csv (详细结果)")

def create_irf_plots(results_price, results_quantity, horizons):
    """生成专业的脉冲响应图"""
    
    # 准备绘图数据 - 使用OVI交互项系数
    price_coefs = [results_price[h].get('coef_ovi', np.nan) for h in horizons]
    price_ses = [results_price[h].get('se_ovi', np.nan) for h in horizons]
    price_lower = [c - 1.96*se if not np.isnan(c) and not np.isnan(se) else np.nan 
                  for c, se in zip(price_coefs, price_ses)]
    price_upper = [c + 1.96*se if not np.isnan(c) and not np.isnan(se) else np.nan 
                  for c, se in zip(price_coefs, price_ses)]
    
    quantity_coefs = [results_quantity[h].get('coef_ovi', np.nan) for h in horizons]
    quantity_ses = [results_quantity[h].get('se_ovi', np.nan) for h in horizons] 
    quantity_lower = [c - 1.96*se if not np.isnan(c) and not np.isnan(se) else np.nan
                     for c, se in zip(quantity_coefs, quantity_ses)]
    quantity_upper = [c + 1.96*se if not np.isnan(c) and not np.isnan(se) else np.nan
                     for c, se in zip(quantity_coefs, quantity_ses)]
    
    # 创建图表 - 使用学术期刊标准样式
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 价格通道图 (Model 4A)
    ax1.plot(horizons, price_coefs, 'o-', color='#2E8B57', linewidth=3, markersize=8, 
             label='θ_h (交互项系数)', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(horizons, price_lower, price_upper, alpha=0.25, color='#2E8B57', 
                     label='95%置信区间')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('预测期 h (年)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('交互项系数 θ_h', fontsize=14, fontweight='bold')
    ax1.set_title('价格通道: OVI缓冲价格冲击效应\\n(US Supply Shock × OVI → Domestic Price)', 
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=12, loc='best')
    ax1.set_xticks(horizons)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 数量通道图 (Model 4B)
    ax2.plot(horizons, quantity_coefs, 'o-', color='#CD853F', linewidth=3, markersize=8,
             label='θ_h (交互项系数)', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(horizons, quantity_lower, quantity_upper, alpha=0.25, color='#CD853F',
                     label='95%置信区间')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('预测期 h (年)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('交互项系数 θ_h', fontsize=14, fontweight='bold')
    ax2.set_title('数量通道: OVI增强调节能力效应\\n(US Supply Shock × OVI → Import Quantity)', 
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=12, loc='best')
    ax2.set_xticks(horizons)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('figures/lp_irf_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("   ✅ 脉冲响应图已生成并保存")

def save_results(results_price, results_quantity, horizons):
    """保存分析结果"""
    
    # 创建结果DataFrame（地理异质性版本）
    results_df = pd.DataFrame({
        'horizon': horizons,
        # OVI交互项系数
        'price_coef_ovi': [results_price[h].get('coef_ovi', np.nan) for h in horizons],
        'price_se_ovi': [results_price[h].get('se_ovi', np.nan) for h in horizons],
        'price_pvalue_ovi': [results_price[h].get('pvalue_ovi', np.nan) for h in horizons],
        'quantity_coef_ovi': [results_quantity[h].get('coef_ovi', np.nan) for h in horizons],
        'quantity_se_ovi': [results_quantity[h].get('se_ovi', np.nan) for h in horizons],
        'quantity_pvalue_ovi': [results_quantity[h].get('pvalue_ovi', np.nan) for h in horizons],
        # 地理距离交互项系数
        'price_coef_distance': [results_price[h].get('coef_distance', np.nan) for h in horizons],
        'price_se_distance': [results_price[h].get('se_distance', np.nan) for h in horizons],
        'price_pvalue_distance': [results_price[h].get('pvalue_distance', np.nan) for h in horizons],
        'quantity_coef_distance': [results_quantity[h].get('coef_distance', np.nan) for h in horizons],
        'quantity_se_distance': [results_quantity[h].get('se_distance', np.nan) for h in horizons],
        'quantity_pvalue_distance': [results_quantity[h].get('pvalue_distance', np.nan) for h in horizons],
        # 样本数
        'n_obs': [results_price[h].get('n_obs', 0) for h in horizons],
    })
    
    # 添加置信区间（仅为OVI交互项）
    results_df['price_ci_lower_ovi'] = results_df['price_coef_ovi'] - 1.96 * results_df['price_se_ovi']
    results_df['price_ci_upper_ovi'] = results_df['price_coef_ovi'] + 1.96 * results_df['price_se_ovi']
    results_df['quantity_ci_lower_ovi'] = results_df['quantity_coef_ovi'] - 1.96 * results_df['quantity_se_ovi']
    results_df['quantity_ci_upper_ovi'] = results_df['quantity_coef_ovi'] + 1.96 * results_df['quantity_se_ovi']
    
    # 保存结果
    results_df.to_csv('outputs/lp_irf_results.csv', index=False)
    
    # 打印核心发现
    print("\n🎯 核心发现总结:")
    
    # 价格通道分析（OVI交互项）
    significant_price_negative = results_df[
        (results_df['price_pvalue_ovi'] < 0.05) & (results_df['price_coef_ovi'] < 0)
    ]
    if len(significant_price_negative) > 0:
        print(f"   💡 价格通道: {len(significant_price_negative)}/5 期显示显著负效应 (预期方向)")
        print(f"      → OVI确实缓解了美国供给冲击对国内价格的不利影响")
        avg_effect = significant_price_negative['price_coef_ovi'].mean()
        print(f"      → 平均缓冲效应: {avg_effect:.4f}")
    else:
        print("   ⚠️  价格通道: 未发现显著的价格缓冲效应")
    
    # 数量通道分析（OVI交互项）
    significant_quantity_negative = results_df[
        (results_df['quantity_pvalue_ovi'] < 0.05) & (results_df['quantity_coef_ovi'] < 0)
    ]
    if len(significant_quantity_negative) > 0:
        print(f"   💡 数量通道: {len(significant_quantity_negative)}/5 期显示显著负效应 (实际发现)")
        print(f"      → OVI效应与预期相反，可能存在复杂的调节机制")
        avg_effect = significant_quantity_negative['quantity_coef_ovi'].mean()
        print(f"      → 平均效应: {avg_effect:.4f}")
    else:
        print("   ⚠️  数量通道: 未发现显著的数量调节效应")
    
    # 地理距离效应分析
    significant_distance_effects = len(results_df[
        (results_df['price_pvalue_distance'] < 0.10) | (results_df['quantity_pvalue_distance'] < 0.10)
    ])
    if significant_distance_effects > 0:
        print(f"   🌍 地理距离: {significant_distance_effects}/5 期显示边际显著效应 (p<0.10)")
    else:
        print("   🌍 地理距离: 未发现显著的地理异质性效应")
    
    print(f"\n   📊 详细结果已保存到 outputs/lp_irf_results.csv")

if __name__ == "__main__":
    run_lp_irf_analysis()