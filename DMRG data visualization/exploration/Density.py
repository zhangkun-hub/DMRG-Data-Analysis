# -*- coding: utf-8 -*-
"""
UltraDMRG 电子密度数据提取与作图（原 Density.ipynb 中 UltraDMRG 部分的 .py 版本）。

流程：读取 nf.json → 封装 DensityRecord → 一维折线图 → 存图。
输出目录：exploration/output/density/fig_density_{doping}_{bond}.png

运行方式（Calculation 环境，任意目录下运行）：
    python -X utf8 exploration/Density.py
参数（CLI 或交互式提示，默认 N=200 / t/J=3 / doping 1_10 / D=1500）：
    --root / --N / --t-j / --doping / --bond-dim / --version
"""
import argparse
import os
import sys

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
    DensityRecord)
from Physical_Quantity.physical_analysis.Density_Process import (  # noqa: E402
    DensityProcessor)
import Phase_Analysis_Report as par  # noqa: E402  交互与空穴率解析复用

plt.show = lambda *a, **kw: None   # Agg 后端：绘图函数末尾的 show 置空

OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'density')


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='UltraDMRG 密度提取与作图（notebook 的 .py 版本）')
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
    parser.add_argument('--version', default=None,
                        help='measure 子目录版本（默认 49）；密度 nf.json '
                             '在 measure_49 与 measure_50 下同构')
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
    version = args.version or '49'

    base_dir = os.path.join(args.root, t_j, doping,
                            f'{bond_dim}_measure_phys_qty',
                            f'measure_{version}')
    nf_path = os.path.join(base_dir, 'nf.json')
    if not os.path.isfile(nf_path):
        print(f'错误：密度数据文件不存在: {nf_path}')
        return 1

    # ---- 提取与作图 ----
    data_n = open(nf_path, encoding='utf-8').read()
    n_records = DensityRecord.extract_data(data_n, N, 1.0)
    DensityProcessor.Line_Chart_Visualization(n_records)

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, f'fig_density_{doping}_{bond_dim}.png')
    plt.gcf().savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close('all')
    return 0


if __name__ == '__main__':
    sys.exit(main())
