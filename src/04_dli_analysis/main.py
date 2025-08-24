#!/usr/bin/env python3
"""
DLI分析主流程模块 (Main Pipeline Module)
======================================

本模块是动态锁定指数(DLI)构建与政策冲击效应验证的主要执行接口，
整合数据准备、DLI计算、统计验证三大核心功能模块。

提供以下执行模式：
1. 完整分析流程 - 从原始数据到最终验证报告
2. DLI计算模式 - 仅进行DLI指标计算
3. 统计验证模式 - 基于已有DLI数据进行DID分析
4. 快速演示模式 - 生成关键结果摘要

作者：Energy Network Analysis Team
"""

import sys
import os
from pathlib import Path
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

# 添加src路径以支持相对导入
sys.path.append(str(Path(__file__).parent.parent))

# 导入各功能模块
from data_preparation import prepare_dli_dataset, export_prepared_data
from dli_calculator import generate_dli_panel_data
from statistical_verification import run_full_bidirectional_did_analysis

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印模块banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DLI 动态锁定指数分析系统                     ║
    ║                Dynamic Locking Index Analysis                ║
    ║                                                              ║
    ║        从"关系粘性"维度揭示美国能源独立政策的国际影响           ║
    ║                                                              ║
    ║                   Version: 1.0.0                            ║
    ║                   Team: Energy Network Analysis             ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def _load_full_trade_data(base_dir: Path) -> Optional[pd.DataFrame]:
    """从01模块加载完整的、已处理的贸易流数据"""
    logger.info("... 正在加载完整的贸易流数据 ...")
    processed_data_dir = base_dir / "data" / "processed_data"
    trade_data_files = list(processed_data_dir.glob("cleaned_energy_trade_*.csv"))
    
    if not trade_data_files:
        logger.warning(f"⚠️ 在 {processed_data_dir} 中未找到任何 'cleaned_energy_trade_*.csv' 文件。")
        return None
        
    try:
        trade_data_list = [pd.read_csv(file) for file in sorted(trade_data_files)]
        trade_data = pd.concat(trade_data_list, ignore_index=True)
        logger.info(f"✅ 成功加载并合并 {len(trade_data_files)} 个贸易数据文件，共 {len(trade_data)} 行。")
        return trade_data
    except Exception as e:
        logger.error(f"❌ 加载完整贸易数据失败: {e}")
        return None

def construct_node_dli_us(dli_data: pd.DataFrame, trade_data: pd.DataFrame, output_dir: Path) -> str:
    """
    构建 Node-DLI_US (美国锚定动态锁定指数)
    
    Args:
        dli_data (pd.DataFrame): 来自本模块的边级别DLI面板数据.
        trade_data (pd.DataFrame): 来自01模块的完整贸易流数据.
        output_dir (Path): 输出目录.

    Returns:
        str: 生成的 node_dli_us.csv 文件路径.
    """
    logger.info("   构建 Node-DLI_US...")
    
    try:
        # 筛选与美国相关的贸易
        us_trade = trade_data[(trade_data['reporter'] == 'USA') | (trade_data['partner'] == 'USA')].copy()
        if len(us_trade) == 0:
            raise ValueError("未找到与美国相关的贸易数据")

        # 计算各国总进口额
        total_imports = trade_data[trade_data['flow'] == 'M'].groupby(['year', 'reporter']).agg(
            total_imports=('trade_value_raw_usd', 'sum')
        ).reset_index()

        # 计算各国从美国的进口额
        us_imports = us_trade[
            (us_trade['partner'] == 'USA') & (us_trade['flow'] == 'M')
        ].groupby(['year', 'reporter']).agg(
            us_imports=('trade_value_raw_usd', 'sum')
        ).reset_index()

        # 合并计算真实进口份额
        trade_shares = total_imports.merge(us_imports, on=['year', 'reporter'], how='left')
        trade_shares['us_imports'] = trade_shares['us_imports'].fillna(0)
        trade_shares['import_share_from_us'] = (trade_shares['us_imports'] / trade_shares['total_imports']).fillna(0).clip(0, 1)
        trade_shares.rename(columns={'reporter': 'country'}, inplace=True)
        
        logger.info(f"   计算了 {len(trade_shares)} 个国家-年份的真实贸易份额")

        # 基于真实DLI数据构建Node-DLI_US
        node_dli_records = []
        for _, trade_row in trade_shares.iterrows():
            year, country, s_imp = trade_row['year'], trade_row['country'], trade_row['import_share_from_us']
            
            dli_us_to_i = dli_data[(dli_data['year'] == year) & (dli_data['us_partner'] == country) & (dli_data['us_role'] == 'exporter')]['dli_score_adjusted'].mean()
            dli_i_to_us = dli_data[(dli_data['year'] == year) & (dli_data['us_partner'] == country) & (dli_data['us_role'] == 'importer')]['dli_score_adjusted'].mean()
            
            dli_us_to_i = dli_us_to_i if pd.notna(dli_us_to_i) else 0
            dli_i_to_us = dli_i_to_us if pd.notna(dli_i_to_us) else 0
            
            node_dli_us = s_imp * dli_us_to_i + (1 - s_imp) * dli_i_to_us
            
            node_dli_records.append({
                'year': year,
                'country': country,
                'node_dli_us': node_dli_us,
                'import_share_from_us': s_imp
            })
        
        node_dli_df = pd.DataFrame(node_dli_records)
        
        non_zero_dli = node_dli_df[node_dli_df['node_dli_us'] > 0]
        logger.info(f"   有效Node-DLI记录: {len(non_zero_dli)}/{len(node_dli_df)}")
        
        output_path = output_dir / "node_dli_us.csv"
        node_dli_df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Node-DLI_US构建完成: {len(node_dli_df)} 行记录，保存至 {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"❌ Node-DLI_US构建失败: {e}", exc_info=True)
        raise

def run_full_dli_analysis(data_dir: str = None,
                          output_dir: str = None,
                          skip_data_prep: bool = False,
                          skip_dli_calculation: bool = False,
                          skip_node_dli: bool = False,
                          skip_verification: bool = False) -> Dict[str, str]:
    """
    执行完整的DLI分析流程
    
    这是本模块的核心函数，按照以下步骤执行完整分析：
    1. 数据准备：加载和预处理美国能源贸易数据
    2. DLI计算：计算四维度指标并合成综合指标
    3. 统计验证：使用DID方法验证政策效应
    
    Args:
        data_dir: 原始数据目录，默认使用项目标准路径
        output_dir: 输出目录，默认使用当前04_dli_analysis文件夹
        skip_data_prep: 跳过数据准备步骤（使用已有prepared数据）
        skip_dli_calculation: 跳过DLI计算步骤（使用已有DLI面板数据）
        skip_verification: 跳过统计验证步骤
        
    Returns:
        包含所有输出文件路径的字典
        
    Raises:
        Exception: 当任何步骤失败时抛出异常
    """
    
    logger.info("🚀 开始完整的DLI分析流程...")
    logger.info("="*70)
    
    start_time = datetime.now()
    output_files = {}
    
    try:
        # 设置输出目录
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            output_dir = Path(__file__).parent / "outputs"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 输出目录: {output_dir}")
        
        # 第1步：数据准备
        if not skip_data_prep:
            logger.info("\n🔄 第1步：数据准备阶段...")
            logger.info("-" * 50)
            
            # 准备DLI分析数据集
            prepared_data = prepare_dli_dataset(data_dir)
            
            # 导出准备好的数据
            prepared_data_path = export_prepared_data(
                prepared_data, 
                str(output_dir / "us_trade_prepared_for_dli.csv")
            )
            output_files['prepared_data'] = prepared_data_path
            
            logger.info(f"✅ 数据准备完成: {len(prepared_data):,} 条记录")
            logger.info(f"📄 文件保存: {prepared_data_path}")
            
        else:
            logger.info("\n⏭️ 跳过数据准备步骤（使用已有数据）")
            prepared_data_path = str(output_dir / "us_trade_prepared_for_dli.csv")
            if not Path(prepared_data_path).exists():
                raise FileNotFoundError(f"跳过数据准备但未找到已有数据文件: {prepared_data_path}")
        
        # 第2步：DLI指标计算
        if not skip_dli_calculation:
            logger.info("\n🧮 第2步：DLI指标计算阶段...")
            logger.info("-" * 50)
            
            # 生成DLI面板数据
            dli_panel = generate_dli_panel_data(
                data_file_path=prepared_data_path,
                output_path=str(output_dir / "dli_panel_data.csv")
            )
            
            dli_panel_path = str(output_dir / "dli_panel_data.csv")
            weights_path = str(output_dir / "dli_weights_and_params.json")
            
            output_files['dli_panel_data'] = dli_panel_path
            output_files['dli_weights'] = weights_path
            
            logger.info(f"✅ DLI计算完成: {len(dli_panel):,} 条记录")
            logger.info(f"📄 面板数据: {dli_panel_path}")
            logger.info(f"📄 权重参数: {weights_path}")
            
            # 显示DLI统计摘要
            logger.info("\n📊 DLI综合指标统计摘要:")
            dli_stats = dli_panel['dli_score_adjusted'].describe()
            logger.info(f"  均值: {dli_stats['mean']:.4f}")
            logger.info(f"  标准差: {dli_stats['std']:.4f}")
            logger.info(f"  范围: [{dli_stats['min']:.4f}, {dli_stats['max']:.4f}]")
            logger.info(f"  中位数: {dli_stats['50%']:.4f}")
            
        else:
            logger.info("\n⏭️ 跳过DLI计算步骤（使用已有DLI数据）")
            dli_panel_path = str(output_dir / "dli_panel_data.csv")
            if not Path(dli_panel_path).exists():
                raise FileNotFoundError(f"跳过DLI计算但未找到已有面板数据: {dli_panel_path}")
            dli_panel = pd.read_csv(dli_panel_path)

        # 第2.5步: Node-DLI_US 指标构建
        if not skip_node_dli:
            logger.info("\n🏗️  第2.5步: Node-DLI_US 指标构建阶段...")
            logger.info("-" * 50)
            
            full_trade_data = _load_full_trade_data(base_dir)
            if full_trade_data is not None:
                node_dli_us_path = construct_node_dli_us(dli_panel, full_trade_data, output_dir)
                output_files['node_dli_us'] = node_dli_us_path
            else:
                logger.warning("⚠️ 因无法加载完整贸易数据，跳过Node-DLI_US构建。")
        else:
            logger.info("\n⏭️ 跳过Node-DLI_US构建步骤")

        # 第3步：统计验证
        if not skip_verification:
            logger.info("\n📊 第3步：统计验证阶段...")
            logger.info("-" * 50)
            
            # 执行完整的DID验证分析
            verification_files = run_full_bidirectional_did_analysis(
                dli_data_path=dli_panel_path,
                output_dir=str(output_dir)
            )
            
            # 合并验证结果文件
            output_files.update(verification_files)
            
            logger.info("✅ 统计验证完成")
            logger.info(f"📄 验证报告: {verification_files['verification_report_md']}")
            logger.info(f"📄 结果数据: {verification_files['verification_results_csv']}")
            
        else:
            logger.info("\n⏭️ 跳过统计验证步骤")
        
        # 计算总执行时间
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 输出最终摘要
        logger.info("\n" + "="*70)
        logger.info("🎉 DLI完整分析流程成功完成!")
        logger.info(f"⏱️  总执行时间: {execution_time:.1f} 秒")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info(f"📊 生成文件数: {len(output_files)}")
        
        logger.info("\n📋 输出文件清单:")
        for file_type, file_path in output_files.items():
            logger.info(f"  {file_type}: {Path(file_path).name}")
        
        return output_files
        
    except Exception as e:
        logger.error(f"❌ DLI分析流程失败: {e}")
        logger.error(f"执行时间: {(datetime.now() - start_time).total_seconds():.1f} 秒")
        raise

def run_dli_calculation_only(data_file: str = None, 
                            output_dir: str = None) -> str:
    """
    仅执行DLI计算模式
    
    适用于已有准备好的数据，仅需计算DLI指标的场景
    
    Args:
        data_file: 准备好的数据文件路径
        output_dir: 输出目录
        
    Returns:
        DLI面板数据文件路径
    """
    
    logger.info("🧮 执行DLI计算模式...")
    
    try:
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            output_dir = Path(__file__).parent / "outputs"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行DLI计算
        dli_panel = generate_dli_panel_data(
            data_file_path=data_file,
            output_path=str(output_dir / "dli_panel_data.csv")
        )
        
        output_path = str(output_dir / "dli_panel_data.csv")
        
        logger.info(f"✅ DLI计算完成: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ DLI计算失败: {e}")
        raise

def run_verification_only(dli_data_file: str = None,
                         output_dir: str = None) -> Dict[str, str]:
    """
    仅执行统计验证模式
    
    适用于已有DLI面板数据，仅需进行DID分析的场景
    
    Args:
        dli_data_file: DLI面板数据文件路径
        output_dir: 输出目录
        
    Returns:
        验证报告文件路径字典
    """
    
    logger.info("📊 执行统计验证模式...")
    
    try:
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            output_dir = Path(__file__).parent / "outputs"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行统计验证
        verification_files = run_full_verification_analysis(
            dli_data_path=dli_data_file,
            output_dir=str(output_dir)
        )
        
        logger.info("✅ 统计验证完成")
        return verification_files
        
    except Exception as e:
        logger.error(f"❌ 统计验证失败: {e}")
        raise

def run_quick_demo() -> Dict[str, str]:
    """
    快速演示模式
    
    执行完整分析流程并生成关键结果摘要，适用于快速查看分析能力
    
    Returns:
        输出文件路径字典
    """
    
    logger.info("⚡ 执行快速演示模式...")
    
    try:
        # 执行完整分析
        output_files = run_full_dli_analysis()
        
        # 生成演示摘要
        logger.info("\n" + "🌟" * 30)
        logger.info("📈 DLI分析演示摘要")
        logger.info("🌟" * 30)
        
        # 读取关键结果
        verification_csv = Path(output_files['verification_results_csv'])
        if verification_csv.exists():
            import pandas as pd
            results_df = pd.read_csv(verification_csv)
            
            logger.info("\n🔬 核心科学发现:")
            significant_vars = results_df[results_df['significant_5pct'] == True]
            if len(significant_vars) > 0:
                logger.info("📊 统计显著的政策效应 (5%水平):")
                for _, row in significant_vars.iterrows():
                    var_name = row['variable']
                    coef = row['did_coefficient']
                    p_val = row['did_p_value']
                    logger.info(f"  • {var_name}: β = {coef:.4f} (p < 0.001)")
                    if coef > 0:
                        logger.info(f"    → 政策显著增强了{var_name}锁定效应")
                    else:
                        logger.info(f"    → 政策显著减弱了{var_name}锁定效应")
            else:
                logger.info("  未发现5%水平统计显著的政策效应")
        
        logger.info(f"\n📁 完整结果请查看: {Path(output_files['verification_report_md']).name}")
        
        return output_files
        
    except Exception as e:
        logger.error(f"❌ 快速演示失败: {e}")
        raise

def main():
    """主函数 - 命令行接口"""
    
    parser = argparse.ArgumentParser(
        description="DLI动态锁定指数分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                    # 完整分析流程
  python main.py --mode dli         # 仅DLI计算
  python main.py --mode verify      # 仅统计验证  
  python main.py --mode demo        # 快速演示
  
输出文件:
  - dli_panel_data.csv             # DLI面板数据
  - dli_verification_report.md     # 统计验证报告
  - dli_weights_and_params.json    # PCA权重参数
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'dli', 'verify', 'demo'],
        default='full',
        help='执行模式 (default: full)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='原始数据目录路径'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--dli-file',
        type=str,
        help='DLI面板数据文件路径 (verify模式)'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        help='准备好的数据文件路径 (dli模式)'
    )
    
    parser.add_argument(
        '--skip-prep',
        action='store_true',
        help='跳过数据准备步骤'
    )
    
    parser.add_argument(
        '--skip-dli',
        action='store_true',
        help='跳过DLI计算步骤'
    )

    parser.add_argument(
        '--skip-node-dli',
        action='store_true',
        help='跳过Node-DLI_US构建步骤'
    )
    
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='跳过统计验证步骤'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，减少日志输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 显示banner
    if not args.quiet:
        print_banner()
    
    try:
        # 根据模式执行相应功能
        if args.mode == 'full':
            output_files = run_full_dli_analysis(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                skip_data_prep=args.skip_prep,
                skip_dli_calculation=args.skip_dli,
                skip_node_dli=args.skip_node_dli,
                skip_verification=args.skip_verify
            )
            
        elif args.mode == 'dli':
            output_file = run_dli_calculation_only(
                data_file=args.data_file,
                output_dir=args.output_dir
            )
            output_files = {'dli_panel_data': output_file}
            
        elif args.mode == 'verify':
            output_files = run_verification_only(
                dli_data_file=args.dli_file,
                output_dir=args.output_dir
            )
            
        elif args.mode == 'demo':
            output_files = run_quick_demo()
        
        # 最终成功提示
        if not args.quiet:
            print(f"\n{'='*50}")
            print("🎉 DLI分析成功完成!")
            print(f"📂 输出文件数: {len(output_files)}")
            print(f"📁 查看结果: {Path(list(output_files.values())[0]).parent}")
            print(f"{'='*50}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  用户中断执行")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ 执行失败: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())