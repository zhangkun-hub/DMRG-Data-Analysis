# -*- coding: utf-8 -*-
"""
d+id 配对相位分析（真实 N200 复值数据验证脚本）。

流程：自动发现 Data/complex/Data_N200/t_J_3 下全部 (doping, bond_dim)
组合 → 装配三 refbond × 12 通道数据（全量解析）→ SCPhaseAnalyzer
相位分析（含 bin 容差稳定性与 D 档一致性）→ 汇总报告写
exploration/output/sc_phase_analysis/phase_analysis_report.md。

报告明确区分：软件实现成功 / 数值分析有效 / 物理证据是否成立。
阈值为候选默认值（defaults_not_authorized，未获物理授权）。

运行方式（Calculation 环境）：
    python -X utf8 exploration/SC_Phase_Analysis.py
参数（CLI）：--root / --t-j / --dopings（逗号分隔，默认全部）
           / --bond-dims（逗号分隔，默认全部）
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Physical_Quantity.physical_analysis.SC_Phase_Analyzer import (  # noqa: E402
    SCPhaseAnalyzer, PhaseAnalyzerConfig, assemble_datasets)

OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'sc_phase_analysis')
REFS = ['49_50', '50_51', '50_54']


def discover(root, t_j, dopings=None, dims=None):
    """发现 (doping, bond_dim) 组合（默认全部有数据的组合）。"""
    combos = []
    base = os.path.join(root, 'Data_N200', t_j)
    for doping in sorted(os.listdir(base)):
        if dopings and doping not in dopings:
            continue
        dpath = os.path.join(base, doping)
        if not os.path.isdir(dpath):
            continue
        for dirname in sorted(os.listdir(dpath)):
            if not dirname.endswith('_measure_super_corr'):
                continue
            dim = dirname.replace('_measure_super_corr', '')
            if dims and dim not in dims:
                continue
            sc_dir = os.path.join(dpath, dirname)
            if all(os.path.isdir(os.path.join(sc_dir, f'super_corr_{r}'))
                   for r in REFS):
                combos.append((doping, dim))
    return combos


def analyze_combo(root, t_j, doping, dim, config):
    """单 (doping, dim) 的相位分析。返回 (datasets, result) 或 None。"""
    base = os.path.join(root, 'Data_N200', t_j, doping,
                        f'{dim}_measure_super_corr')
    files = {}
    for ref in REFS:
        rb = tuple(map(int, ref.split('_')))
        files[rb] = {}
        for bond in 'ABC':
            files[rb][bond] = {
                ch: os.path.join(base, f'super_corr_{ref}',
                                 f'scs{bond}{ch}_fix_refbond_{ref}.json')
                for ch in 'abcd'}
    datasets = assemble_datasets(files, 200)
    result = SCPhaseAnalyzer.analyze(datasets, N=200, config=config)
    return result


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='d+id 配对相位分析（N200 复值数据验证）')
    parser.add_argument('--root', default=os.path.join(ROOT, 'Data', 'complex'),
                        help='数据根（默认 Data/complex）')
    parser.add_argument('--t-j', default='t_J_3', help='t/J 目录（默认 t_J_3）')
    parser.add_argument('--dopings', default=None,
                        help='逗号分隔 doping 列表（默认全部有数据的）')
    parser.add_argument('--bond-dims', default=None,
                        help='逗号分隔 bond dimension 列表（默认全部）')
    args = parser.parse_args(argv)

    dopings = args.dopings.split(',') if args.dopings else None
    dims = args.bond_dims.split(',') if args.bond_dims else None
    combos = discover(args.root, args.t_j, dopings, dims)
    if not combos:
        print(f'错误：{args.root}/Data_N200/{args.t_j} 下无可用数据组合')
        return 1

    config = PhaseAnalyzerConfig()   # 候选默认阈值（未获物理授权）
    per_doping = {}
    for doping, dim in combos:
        try:
            result = analyze_combo(args.root, args.t_j, doping, dim, config)
        except Exception as exc:
            print(f'错误：{doping}/{dim} 分析失败: {exc}')
            continue
        per_doping.setdefault(doping, {})[dim] = result
        print(f'[{doping}/{dim}] {result.classification} '
              f'(valid {result.n_valid_bins}/{result.n_bins}, '
              f'圆周均值 {result.circular_mean_deg:.0f}°)')

    os.makedirs(OUTDIR, exist_ok=True)
    report_path = os.path.join(OUTDIR, 'phase_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# d+id 配对相位分析报告（N200 复值数据）\n\n')
        f.write('> 软件实现成功 ≠ 数值分析有效 ≠ 物理证据成立。'
                '阈值为候选默认值（defaults_not_authorized，未获物理授权）。\n\n')
        f.write(f'数据根: {args.root}/Data_N200/{args.t_j}\n\n')
        f.write('## 各 doping × bond dimension 分类总表\n\n')
        f.write('| doping | bond dim | 分类 | 有效点 | 圆周均值 | 圆周std | '
                'R | 中位\\|C22\\|/\\|C11\\| | 同bin偏差(描述性) |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        for doping, dims_res in sorted(per_doping.items()):
            for dim, r in sorted(dims_res.items()):
                conj_ok = r.conjugate_diagnostics['ok']
                conj_s = ('通过' if conj_ok is True else
                          '异常' if conj_ok is False else 'n/a')
                if np.isfinite(r.circular_mean_deg):
                    mean_s = f'{r.circular_mean_deg:.0f}°'
                else:
                    mean_s = 'n/a'
                f.write(f'| {doping} | {dim} | {r.classification} | '
                        f'{r.n_valid_bins}/{r.n_bins} | {mean_s} | '
                        f'{r.circular_std_deg:.0f}° | '
                        f'{r.circular_R:.2f} | '
                        f'{r.d2_ratio_median:.2e} | {conj_s} |\n')
        f.write('\n## 各 doping 详报\n\n')
        for doping, dims_res in sorted(per_doping.items()):
            f.write(f'### {doping}\n\n')
            # D 档一致性（单档分类 vs 跨 D 一致性 vs doping 级证据判断）
            classes = {dim: r.classification for dim, r in dims_res.items()}
            uniq = set(classes.values())
            nonchiral_family = {'d_component_suppressed',
                                'd_component_unresolved',
                                'no_chiral_signal'}
            if len(uniq) == 1:
                doping_evidence = '三档一致：' + sorted(uniq)[0]
            elif uniq <= nonchiral_family:
                doping_evidence = ('证据方向一致（各档均无手性候选，'
                                   '仅 d2 分辨能力层级不同）：'
                                   + str(classes))
            else:
                doping_evidence = ('含手性候选的档间不一致（provisional）：'
                                   + str(classes))
            f.write(f'**单档分类**: {classes}\n\n')
            f.write(f'**跨 D 一致性 / doping 级证据判断**: {doping_evidence}'
                    '（单档分类不代表跨 D 验证；若某档为手性候选而其他档'
                    '不是，doping 级标 provisional）\n\n')
            # 主档（最大 D）详报
            main_dim = max(dims_res, key=int)
            r = dims_res[main_dim]
            f.write(f'**主档 {main_dim} 详报**（bin 容差 {config.primary_tol}）:\n\n')
            f.write(f'- 分类: {r.classification}\n')
            f.write(f'- 噪声底: {r.noise_floor:.3e}（经验工程门槛；'
                    f'|C12| 序列 {r.noise_counts["total"]} 个点，'
                    f'精确零 {r.noise_counts["zero"]}、'
                    f'非零 {r.noise_counts["nonzero"]}）；'
                    f'有效距离点 {r.n_valid_bins}/{r.n_bins}\n')
            if np.isfinite(r.circular_mean_deg):
                f.write(f'- C12 相位圆周均值 {r.circular_mean_deg:.1f}°'
                        f'（std {r.circular_std_deg:.1f}°, '
                        f'R={r.circular_R:.3f}, n={r.circular_n}）\n')
            else:
                f.write(f'- C12 相位圆周均值: n/a'
                        f'（{r.circular_mean_invalid_reason}）\n')
            f.write(f'- 中位 |C22|/|C11| = {r.d2_ratio_median:.2e}'
                    f'（d2 分辨能力: {r.d2_resolution}；'
                    f'< {config.d2_unresolved_ratio:.0e} 不可分辨、'
                    f'< {config.d2_suppression_ratio:.0e} 可分辨抑制）\n')
            f.write(f'- 依据: {"；".join(r.reasons)}\n')
            f.write(f'- bin 容差稳定性: {r.tol_stability}\n')
            f.write(f'- 平台检测: +90° {"找到" if r.platform_plus.found else "未找到"}'
                    f'（{r.platform_plus.n_points} 点）；'
                    f'-90° {"找到" if r.platform_minus.found else "未找到"}'
                    f'（{r.platform_minus.n_points} 点）\n\n')
            f.write('| r | \\|C11\\| | \\|C22\\| | \\|C12\\| | Re C12 | '
                    'Im C12 | φ12 | χ_d | 门控 |\n')
            f.write('|---|---|---|---|---|---|---|---|---|\n')
            for d, c11, c22, c12, phi, chi, gate, ok in zip(
                    r.distances, r.C11, r.C22, r.C12, r.phi12_deg,
                    r.chi_d, r.amp_gate, r.valid):
                gate_s = '✓' if gate else '✗'
                chi_s = f'{chi:.2e}' if np.isfinite(chi) else 'nan'
                f.write(f'| {d:.1f} | {abs(c11):.2e} | {abs(c22):.2e} | '
                        f'{abs(c12):.2e} | {c12.real:.2e} | {c12.imag:.2e} | '
                        f'{phi:.0f}° | {chi_s} | {gate_s} |\n')
            f.write('\n')
        f.write('## 结论边界\n\n'
                '- 本报告分类为数值诊断（chiral_phase_candidate 为数值候选，'
                '非相态判定）\n'
                '- d_component_unresolved / d_component_suppressed / '
                'no_chiral_signal 仅限当前参数（N200、Ly=4 窄圆柱、'
                't/J=3）与有限 D 档\n'
                '- 不得将本结果推广为二维热力学极限下存在/不存在 d+id\n'
                '- 同 bin 跨通道偏差（描述性诊断）无同距离共轭恒等式'
                '依据，不构成手性判据或数据正确性证明\n'
                '- noise_floor 是逐组合独立估计的经验工程门槛，不同'
                'doping/D 之间不可直接比较，禁止据其差异推断物理规律\n'
                '- 同一距离 bin 可能合并相对 refbond 方向矢量不同的'
                'target 键（有限几何近似），bin 容差稳定性不能证明'
                '该近似严格成立\n')
    print(f'报告已写出: {report_path}')
    return 0


if __name__ == '__main__':
    import numpy as np  # noqa: E402  报告数值格式化用
    sys.exit(main())
