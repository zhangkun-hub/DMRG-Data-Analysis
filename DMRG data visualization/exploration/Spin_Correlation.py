# -*- coding: utf-8 -*-
"""
UltraDMRG 自旋关联数据提取、作图与衰减分析
（原 Spin_Correlation.ipynb 中 UltraDMRG 部分的 .py 版本）。

流程：读取 A/B 子格三分量（sz/sp/sm）→ 合成 <S_i·S_j> → 2×3 统一作图 →
SpinCorrDecayAnalyzer 衰减分析（A/B 子格两任务多进程并行，摘要写文本、
诊断图存 png、任务计时打印，不在终端打印摘要正文）。

输出目录：
    exploration/output/spin_correlation/{doping}/
        fig_spin_ab.png            A/B 子格自旋关联 2×3 图
        sublattice_A/              A 子格（measure_49，源点 49）
            decay_summary.txt      衰减分析完整摘要
            fig_decay.png          衰减诊断图
        sublattice_B/              B 子格（measure_50，源点 50）
            ...

运行方式（Calculation 环境，任意目录下运行）：
    python -X utf8 exploration/Spin_Correlation.py
参数（CLI 或交互式提示，默认 N=200 / t/J=3 / doping 1_10 / D=1500）：
    --root / --N / --t-j / --doping / --bond-dim / --no-decay / --workers
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
    CorrelationRecord)
from Physical_Quantity.physical_analysis.Correlation_Process import (  # noqa: E402
    CorrelationProcessor)
from Physical_Quantity.physical_analysis.Spin_Correlation_Analyzer import (  # noqa: E402
    SpinCorrDecayAnalyzer, AnalyzerConfig)
import Phase_Analysis_Report as par  # noqa: E402  交互与空穴率解析复用

plt.show = lambda *a, **kw: None   # Agg 后端：绘图函数末尾的 show 置空

OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'spin_correlation')

# A/B 子格：名称、(measure 版本, 源点偏移, 最大场点偏移)
SUBLATTICES = [
    ('A sublattice', '49', 1, -3),   # A 子格：measure_49, 源点 49
    ('B sublattice', '50', 2, -2),   # B 子格：measure_50, 源点 50
]


def load_sub_lattice(root, t_j, doping, bond_dim, version, src_off,
                     max_off, N):
    """读取单子格自旋三分量并合成 <S_i·S_j>（list[CorrelationRecord]）。"""
    base_dir = os.path.join(root, t_j, doping,
                            f'{bond_dim}_measure_phys_qty',
                            f'measure_{version}')
    data_SuSd = open(os.path.join(base_dir, f'sm{version}sp.json'),
                     encoding='utf-8').read()
    data_SdSu = open(os.path.join(base_dir, f'sp{version}sm.json'),
                     encoding='utf-8').read()
    data_SzSz = open(os.path.join(base_dir, f'sz{version}sz.json'),
                     encoding='utf-8').read()
    source_point = int(N / 8 / 4) * 8 + src_off
    field_point_max = int(N / 4 * 3) + max_off
    sequence = range(source_point, field_point_max + 1, 4)
    SzSz = CorrelationRecord.extract_data(data_SzSz, sequence, N, 1.0)
    SuSd = CorrelationRecord.extract_data(data_SdSu, sequence, N, 1.0)  # 注意交叉对应
    SdSu = CorrelationRecord.extract_data(data_SuSd, sequence, N, 1.0)
    return CorrelationProcessor.Combined_spin_correlation(SzSz, SuSd, SdSu)


def _spin_worker(rec_list, name, out_dir, N):
    """子进程 worker：单子格自旋衰减分析 → 摘要 txt + 诊断图 PNG。

    在子进程内完成整条链（分析/摘要/存图），回传 (错误信息, 执行秒数)
    ——错误信息为 None 表示成功；耗时在 worker 内测量（不含排队等待），
    供主进程打印任务计时。
    """
    t0 = time.perf_counter()
    try:
        config = AnalyzerConfig()
        result = SpinCorrDecayAnalyzer.analyze(rec_list, config, N, 1.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SpinCorrDecayAnalyzer.print_summary(result)
        txt_path = os.path.join(out_dir, 'decay_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f'# {name} 自旋关联衰减分析摘要\n\n')
            f.write(buf.getvalue())
        prep = SpinCorrDecayAnalyzer.group_by_distance(rec_list, config, N, 1.0)
        SpinCorrDecayAnalyzer.Decay_Visualization(
            result, prep.r, prep.C, show=False,
            visualization_mode='standard', title=name,
            save_path=os.path.join(out_dir, 'fig_decay.png'))
        plt.close('all')
        return (None, time.perf_counter() - t0)
    except Exception as exc:   # 失败信息回传主进程打印
        return (f'{type(exc).__name__}: {exc}', time.perf_counter() - t0)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='UltraDMRG 自旋关联提取/作图/衰减分析'
                    '（notebook 的 .py 版本）')
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
                             '交互式提示输入（默认 1500）')
    parser.add_argument('--no-decay', action='store_true',
                        help='跳过衰减分析（只提取与作图）')
    parser.add_argument('--workers', type=int, default=None,
                        help='衰减分析并行进程数（默认取 CPU 核数，最多 2）')
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
            raw = input('请输入 bond dimension（回车使用默认 1500）: ').strip()
        except EOFError:
            raw = ''
        bond_dim = raw or '1500'
    else:
        bond_dim = str(args.bond_dim)

    base_root = os.path.join(args.root, t_j, doping,
                             f'{bond_dim}_measure_phys_qty')
    if not os.path.isdir(base_root):
        print(f'错误：自旋数据目录不存在: {base_root}')
        return 1

    out_base = os.path.join(OUTDIR, doping)
    os.makedirs(out_base, exist_ok=True)

    # ---- A/B 子格提取与合成（串行，快） ----
    sub_data = {}
    for name, version, src_off, max_off in SUBLATTICES:
        try:
            sub_data[name] = load_sub_lattice(
                args.root, t_j, doping, bond_dim, version, src_off, max_off, N)
        except Exception as exc:
            print(f'错误：{name} 数据提取失败: {exc}')
            return 1

    # ---- 自旋关联 2×3 统一作图 ----
    CorrelationProcessor.Spin_Corr_Visualization(
        {name: recs for name, recs in sub_data.items()})
    plt.gcf().savefig(os.path.join(out_base, 'fig_spin_ab.png'),
                      dpi=200, bbox_inches='tight')
    plt.close('all')

    # ---- 衰减分析（A/B 子格并行；摘要写 txt、图存 png） ----
    if not args.no_decay:
        n_tasks = len(sub_data)
        workers = args.workers or min(os.cpu_count() or 1, n_tasks)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for name, rec_list in sub_data.items():
                sub_dir = os.path.join(out_base, name.replace(' ', '_'))
                os.makedirs(sub_dir, exist_ok=True)
                futures.append((name, pool.submit(
                    _spin_worker, rec_list, name, sub_dir, N)))
            # 逐任务打印完成信息与 worker 内执行耗时（便于定位慢任务）
            print(f'衰减分析：{n_tasks} 个任务，{workers} 个并行进程')
            wall0 = time.perf_counter()
            slowest, slowest_name = 0.0, ''
            for i, (name, fut) in enumerate(futures, 1):
                err, dt = fut.result()
                if err:
                    print(f'[{i}/{n_tasks}] {name} 失败 ({dt:.1f}s): {err}')
                else:
                    print(f'[{i}/{n_tasks}] {name} 完成 ({dt:.1f}s)')
                if dt > slowest:
                    slowest, slowest_name = dt, name
            print(f'衰减分析完成：总耗时 {time.perf_counter() - wall0:.1f}s，'
                  f'最慢任务 {slowest_name} ({slowest:.1f}s)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
