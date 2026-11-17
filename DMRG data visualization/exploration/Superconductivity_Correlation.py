# -*- coding: utf-8 -*-
"""
超导关联数据提取、SU(2) 检验、作图与衰减分析
（原 Superconductivity_Correlation.ipynb 的 .py 版本）。

流程：自动发现数据目录下全部 reference bond（super_corr_* 子目录）→
按 XC 构型键方向分类公式（见 _refbond_direction，对应测量源码
HoneycombXCCylinder::ClassifyNearestNeighbor）确定每个 refbond 的键
方向（A/B/C）→ 12 通道提取 → SU(2) 检验（a↔d、b↔c）→ 四通道合成
→ 作图 → SuperCorrDecayAnalyzer 衰减分析（摘要写入文本，不在终端打印）。

输出目录（每个 refbond 一个子文件夹，命名为 ref_{i}_{j}_{键方向}）：
    exploration/output/superconductivity_correlation/ref_49_50_A/
        fig_three_directions.png      三键方向 1×3 图
        fig_direction_{A|B|C}.png     单键方向 1×3 图
        fig_decay_{A|B|C}.png         衰减诊断 2×2 面板
        decay_summary_{A|B|C}.txt     衰减分析完整摘要

refbond 键方向计算公式（XC 圆柱，Ly=4，站点编号 i = x*Ly + y）：
    键 (i, j)，令 i<j；x0,y0 = i//Ly, i%Ly；x1,y1 = j//Ly, j%Ly
    - 同列（x0==x1，竖直键）：y_min = y0 若 (y0+Ly-y1)%Ly==1 否则 y1；
      键方向 = A 若 (y_min+x0)%2==0 否则 B
    - 同行（y0==y1，水平键）：键方向 = C
    N=200 验证：49_50→A、50_51→B、50_54→C；N=400：97_98→A、98_99→B、
    98_102→C（refbond 目录自动发现，不同 N 无需手改）。

运行方式（Calculation 环境，任意目录下运行）：
    python -X utf8 exploration/Superconductivity_Correlation.py
参数（CLI 或交互式提示，默认 N=200 / t/J=3 / doping 1_10 / D=5000）：
    --root / --N / --t-j / --doping / --bond-dim / --no-decay / --workers

衰减分析按 (refbond × 键方向) 任务并行执行（ProcessPoolExecutor 多进程，
--workers 控制进程数，默认取 CPU 核数、最多 9）。
"""
import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings  # noqa: E402
warnings.filterwarnings('ignore',
                        message='FigureCanvasAgg is non-interactive.*')
warnings.filterwarnings('ignore',
                        message='This figure includes Axes that are not '
                                'compatible with tight_layout.*')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Physical_Quantity.data_records.Physical_Data_Record import (  # noqa: E402
    SuperCorrRecord)
from Physical_Quantity.physical_analysis.Super_Correlation_Process import (  # noqa: E402
    SuperCorrProcessor, _refbond_direction)
from Physical_Quantity.physical_analysis.Super_Correlation_Analyzer import (  # noqa: E402
    SuperCorrDecayAnalyzer, AnalyzerConfig)
import Phase_Analysis_Report as par  # noqa: E402  交互与空穴率解析复用

plt.show = lambda *a, **kw: None   # Agg 后端：绘图函数末尾的 show 置空

OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'superconductivity_correlation')

BOND_LABELS = {
    'A': 'target A',
    'B': 'target B',
    'C': 'target C',
}


def save(ref_dir, name):
    """当前 figure 存盘到 refbond 子目录并关闭（静默，失败时才输出）。"""
    path = os.path.join(ref_dir, f'{name}.png')
    plt.gcf().savefig(path, dpi=200, bbox_inches='tight')
    plt.close('all')


def _decay_worker(rec_list, label, bond, key, rb, ref_dir):
    """子进程 worker：单键方向衰减分析 → 摘要 txt + 诊断图 PNG。

    在子进程内完成整条链（分析/摘要/存图），回传 (错误信息, 执行秒数)
    ——错误信息为 None 表示成功；耗时在 worker 内测量（不含排队等待），
    供主进程打印任务计时，便于定位慢任务。
    """
    t0 = time.perf_counter()
    try:
        config = AnalyzerConfig()   # 包络拟合固定全点口径
        result = SuperCorrDecayAnalyzer.analyze(rec_list, config)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SuperCorrDecayAnalyzer.print_summary(result)
        txt_path = os.path.join(ref_dir, f'decay_summary_{bond}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f'# reference {key}: {rb} 衰减分析摘要：{label}\n\n')
            f.write(buf.getvalue())
        prep = SuperCorrDecayAnalyzer.group_by_distance(rec_list)
        SuperCorrDecayAnalyzer.Decay_Visualization(
            result, prep.r, prep.C, show=False,
            visualization_mode='standard', title=label,
            save_path=os.path.join(ref_dir, f'fig_decay_{bond}.png'))
        plt.close('all')
        return (None, time.perf_counter() - t0)
    except Exception as exc:   # 失败信息回传主进程打印
        return (f'{type(exc).__name__}: {exc}', time.perf_counter() - t0)


def discover_refbonds(super_root):
    """自动发现 super_corr_{i}_{j} 目录 → [(refbond, 键方向)]。"""
    out = []
    for name in sorted(os.listdir(super_root)):
        if not name.startswith('super_corr_'):
            continue
        parts = name.split('_')[-2:]
        rb = (int(parts[0]), int(parts[1]))
        out.append((rb, _refbond_direction(*rb), name))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='超导关联提取/作图/衰减分析（notebook 的 .py 版本）')
    parser.add_argument('--root', default=os.path.join(ROOT, 'Data',
                                                       'complex', 'Data_N200'),
                        help='数据根（默认 Data/complex/Data_N200）')
    parser.add_argument('--N', type=int, default=None,
                        help='格点数（默认 200）；不提供时交互式提示输入')
    parser.add_argument('--t-j', default=None,
                        help='t/J（默认 3；写 3 或 t_J_3 均可）；'
                             '不提供时交互式提示输入')
    parser.add_argument('--doping', default=None,
                        help='空穴率（如 1_10 d1_10 或 0.1）；不提供时'
                             '交互式提示输入（默认 1_10）')
    parser.add_argument('--bond-dim', default=None,
                        help='bond dimension（如 1500 或 5000）；不提供时'
                             '交互式提示输入（默认 5000）')
    parser.add_argument('--no-decay', action='store_true',
                        help='跳过衰减分析（只提取与作图）')
    parser.add_argument('--workers', type=int, default=None,
                        help='衰减分析并行进程数（默认取 CPU 核数，最多 9）')
    args = parser.parse_args(argv)

    N, t_j, doping = par._prompt_config(args.N, args.t_j, args.doping)
    if args.t_j is not None:
        # 显式 --t-j 归一化：'3'/'3.0' → 't_J_3'；已是 't_J_3' 的保持原样
        try:
            t_j = f't_J_{float(args.t_j):g}'
        except ValueError:
            t_j = args.t_j
    if args.doping is not None:
        doping = par._fraction_to_doping(args.doping)   # 显式值归一化
        if doping is None:
            print(f'错误：无法解析空穴率输入: {args.doping!r}'
                  '（示例: 1_10、0.1、d1_10）')
            return 1
    if args.bond_dim is None:
        try:
            raw = input('请输入 bond dimension（回车使用默认 5000）: ').strip()
        except EOFError:
            raw = ''
        bond_dim = raw or '5000'
    else:
        bond_dim = str(args.bond_dim)

    super_root = os.path.join(args.root, t_j, doping,
                              f'{bond_dim}_measure_super_corr')
    if not os.path.isdir(super_root):
        print(f'错误：超导数据目录不存在: {super_root}')
        return 1
    refbonds = discover_refbonds(super_root)
    if not refbonds:
        print(f'错误：{super_root} 下没有 super_corr_* 目录')
        return 1

    jobs = []   # (ref_dir, key, rb, super_corr_dict)：提取完成后统一并行衰减
    for rb, key, dirname in refbonds:
        base_dir = os.path.join(super_root, dirname)
        ref_dir = os.path.join(OUTDIR, f'ref_{rb[0]}_{rb[1]}_{key}')
        os.makedirs(ref_dir, exist_ok=True)

        # ---- 12 通道提取 ----
        records = {}
        for bond in 'ABC':
            for channel in 'abcd':
                k = f'scs{bond}{channel}'
                fname = f'{k}_fix_refbond_{rb[0]}_{rb[1]}.json'
                records[k] = SuperCorrRecord.extract_data(
                    open(os.path.join(base_dir, fname),
                         encoding='utf-8').read(), N, 1.0)

        # ---- SU(2) 检验 + 合成 ----
        super_corr_dict = {}
        for bond, label in BOND_LABELS.items():
            recs = [records[f'scs{bond}{channel}'] for channel in 'abcd']
            SuperCorrProcessor.symmetry_check(*recs)
            super_corr_dict[label] = \
                SuperCorrProcessor.Combined_super_correlation(*recs)

        # ---- 作图 ----
        SuperCorrProcessor.Super_Corr_Visualization_three(super_corr_dict)
        save(ref_dir, 'fig_three_directions')
        for bond, label in BOND_LABELS.items():
            SuperCorrProcessor.Super_Corr_Visualization(
                super_corr_dict[label], label)
            save(ref_dir, f'fig_direction_{bond}')

        jobs.append((ref_dir, key, rb, super_corr_dict))

    # ---- 衰减分析（按 refbond × 键方向并行；摘要写 txt、图存 png） ----
    if not args.no_decay:
        n_tasks = len(jobs) * len(BOND_LABELS)
        workers = args.workers or min(os.cpu_count() or 1, n_tasks)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for ref_dir, key, rb, super_corr_dict in jobs:
                for bond, label in BOND_LABELS.items():
                    futures.append((label, key, rb, pool.submit(
                        _decay_worker, super_corr_dict[label], label,
                        bond, key, rb, ref_dir)))
            # 逐任务打印完成信息与 worker 内执行耗时（便于定位慢任务）
            print(f'衰减分析：{n_tasks} 个任务，{workers} 个并行进程')
            wall0 = time.perf_counter()
            slowest, slowest_name = 0.0, ''
            for i, (label, key, rb, fut) in enumerate(futures, 1):
                err, dt = fut.result()
                if err:
                    print(f'[{i}/{n_tasks}] reference {key}: {rb} {label} '
                          f'失败 ({dt:.1f}s): {err}')
                else:
                    print(f'[{i}/{n_tasks}] reference {key}: {rb} {label} '
                          f'完成 ({dt:.1f}s)')
                if dt > slowest:
                    slowest, slowest_name = dt, f'reference {key}: {rb} {label}'
            print(f'衰减分析完成：总耗时 {time.perf_counter() - wall0:.1f}s，'
                  f'最慢任务 {slowest_name} ({slowest:.1f}s)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
