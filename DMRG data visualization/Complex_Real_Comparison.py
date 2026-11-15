# -*- coding: utf-8 -*-
"""
复值 vs 实值 DMRG 数据对比与合理性诊断（论文 d+id 数据链第一步）。

对同一参数组（N/t/J/doping/D）的实值（USE_REAL=1）与复值（USE_REAL=0）
两套测量结果做：
1. 五项合理性检查 → 三态结论（consistent / minor_deviation / suspicious）
   - 能量变分检查：复值 E0 不应显著高于实值 E0（复值空间 ⊇ 实值空间）
   - 实部一致性：无 TRSB 时复值实部应逐点 ≈ 实值结果
   - 虚部量级：虚部为机器零 = 无手性相位信号；显著 = 需相位分析
   - SU(2) 残差：复值超导 a↔d、b↔c 一致性
   - 收敛轨迹：两份优化日志的 E0 序列与最终 TruncErr
2. 对比可视化（样式与 UltraDMRG vs Itensor 对比一致）：
   密度 / 自旋（A/B 子格 2×3）/ 超导（reference A/B/C 3×3）/
   单粒子（corr1）/ 超导虚部检查图 / 能量收敛对比图
3. Markdown 报告（检查表 + 结论 + 图件索引）

对比口径（见 docs/SC_Phase_Judgment_Guide.md 或会话方案）：
- 主通道：复值实部 vs 实值结果（同一物理量同一通道的直接对比）
- 检验通道：虚部单独看（量级报告 + 超导虚部图）
- 补充通道：模对比（当前虚部≈0 时与实部对比等价）

运行方式（Calculation 环境，任意目录下均可运行）：
    python -X utf8 Complex_Real_Comparison.py            # 交互式提示
    python -X utf8 Complex_Real_Comparison.py --doping 0.12 --N 200 --t-j t_J_3

参数（CLI 或交互式决定，见参数块默认值）：
    N（默认 200）/ t/J（默认 3）/ 空穴率（默认 3_25，即 δ=0.12）
    / DIM / REAL_ROOT / COMPLEX_ROOT
"""
import os
import re
import sys
import datetime

import numpy as np

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
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Physical_Quantity.data_records.Physical_Data_Record import (  # noqa: E402
    DensityRecord, CorrelationRecord, SuperCorrRecord, SingleParticleRecord)
from Physical_Quantity.physical_analysis.Density_Process import (  # noqa: E402
    DensityProcessor)
from Physical_Quantity.physical_analysis.Correlation_Process import (  # noqa: E402
    CorrelationProcessor)
from Physical_Quantity.physical_analysis.Super_Correlation_Process import (  # noqa: E402
    SuperCorrProcessor, _refbond_direction)
import Phase_Analysis_Report as par  # noqa: E402  交互提示与空穴率解析复用

# ==========================================================================
# 参数块（N/T_J/DOPING/DELTA 是默认值，可被 CLI 参数或交互式提示覆盖，
# 见 main）
# ==========================================================================
N = 200
T_J = 't_J_3'
DOPING = 'd3_25'          # 默认：δ=0.12（3/25）
DELTA = 0.12
DIM = '1500'
OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'complex_real_comparison')


def _data_root(kind):
    """Data/{real|complex}/Data_N200/t_J_3/{doping}（doping 层为
    measure 目录与优化日志的公共父目录）。"""
    return os.path.join(SCRIPT_DIR, 'Data', kind, 'Data_N200', T_J, DOPING)


REAL_ROOT = _data_root('real')       # 模块级默认；main 内按 CLI/交互覆盖
COMPLEX_ROOT = _data_root('complex')

# 检查阈值（默认值，未获物理授权——报告内标注）
THRESH_E0_REL = 1e-6      # 能量差相对阈值：|E0_c - E0_r| / |E0_r| 超过即 flag
THRESH_REAL_REL = 1e-3    # 实部一致性相对误差阈值
THRESH_IMAG_REL = 1e-8    # 虚部/实部量级比阈值：低于 = 机器零
THRESH_SU2 = 1e-4         # SU(2) 对称性残差（经验阈值）

SUPER_LABELS = {'A': 'A key (blue bond)', 'B': 'B key (green bond)',
                'C': 'C key (orange bond)'}


# ==========================================================================
# 数据装载
# ==========================================================================
def load_density(root, a=1.0):
    """nf.json → list[DensityRecord]。"""
    phys = os.path.join(root, f'{DIM}_measure_phys_qty', 'measure_49')
    text = open(os.path.join(phys, 'nf.json'), encoding='utf-8').read()
    return DensityRecord.extract_data(text, N=N, a=a)


def load_spin(root, a=1.0):
    """自旋三分量合成 <S_i·S_j>，返回 {'A sublattice': recs, 'B sublattice': recs}。"""
    phys = os.path.join(root, f'{DIM}_measure_phys_qty')
    out = {}
    for ver, src_off, max_off, name in (('49', 1, -3, 'A sublattice'),
                                        ('50', 2, -2, 'B sublattice')):
        base = os.path.join(phys, f'measure_{ver}')
        src = int(N / 8 / 4) * 8 + src_off
        fp_max = int(N / 4 * 3) + max_off
        seq = range(src, fp_max + 1, 4)
        sz = CorrelationRecord.extract_data(
            open(os.path.join(base, f'sz{ver}sz.json'),
                 encoding='utf-8').read(), seq, N, a)
        su = CorrelationRecord.extract_data(
            open(os.path.join(base, f'sp{ver}sm.json'),
                 encoding='utf-8').read(), seq, N, a)
        sd = CorrelationRecord.extract_data(
            open(os.path.join(base, f'sm{ver}sp.json'),
                 encoding='utf-8').read(), seq, N, a)
        out[name] = CorrelationProcessor.Combined_spin_correlation(sz, su, sd)
    return out


def load_supercorr(root, a=1.0):
    """3 refbond × 12 通道 → {ref_tag: {label: 合成记录}}。"""
    sc_root = os.path.join(root, f'{DIM}_measure_super_corr')
    refdirs = sorted(d for d in os.listdir(sc_root)
                     if re.match(r'super_corr_\d+_\d+$', d))
    out = {}
    for rd in refdirs:
        rb = tuple(int(x) for x in rd.split('_')[-2:])
        ref_tag = f'reference {_refbond_direction(*rb)}: {rb}'
        channels = {}
        for key, label in SUPER_LABELS.items():
            recs = {}
            for ch in 'abcd':
                fname = f'scs{key}{ch}_fix_refbond_{rb[0]}_{rb[1]}.json'
                recs[ch] = SuperCorrRecord.extract_data(
                    open(os.path.join(sc_root, rd, fname),
                         encoding='utf-8').read(), N, a)
            channels[label] = SuperCorrProcessor.Combined_super_correlation(
                *[recs[c] for c in 'abcd'])
        out[ref_tag] = channels
    return out


def load_single_particle(root, a=1.0):
    """single_particle_corr1 → list[SingleParticleRecord]。"""
    phys = os.path.join(root, f'{DIM}_measure_phys_qty', 'measure_49')
    text = open(os.path.join(phys, 'single_particle_corr1.json'),
                encoding='utf-8').read()
    return SingleParticleRecord.extract_data(text, N=N, a=a)


_E0_RE = re.compile(r'E0 = ([-0-9.eE+]+)')
_TE_RE = re.compile(r'TruncErr = ([-0-9.eE+]+)')


def _find_log(root):
    """找含 D=1500 段的优化日志（同目录多份时取第一份）。"""
    if not os.path.isdir(root):
        return None, None
    for name in sorted(os.listdir(root)):
        if name.endswith('.log'):
            text = open(os.path.join(root, name), encoding='utf-8',
                        errors='replace').read()
            if re.search(rf'Bond dimension:\s*{DIM}/{DIM}', text):
                return name, text
    return None, None


def load_e0_series(root):
    """优化日志 → (E0 序列, 最终 TruncErr)。"""
    _, text = _find_log(root)
    if text is None:
        return [], None
    e0s = [float(v) for v in _E0_RE.findall(text)]
    tes = [float(v) for v in _TE_RE.findall(text)]
    return e0s, (max(tes[-5:]) if tes else None)


# ==========================================================================
# 五项合理性检查（返回 dict：name/passed/evidence/note）
# ==========================================================================
def check_energy(e0_c, e0_r):
    """能量变分检查：复值 E0 不应显著高于实值 E0。"""
    if e0_c is None or e0_r is None:
        return {'name': '能量变分检查', 'passed': None,
                'evidence': '日志缺失，无法检查', 'note': '日志无 E0 行'}
    diff = e0_c - e0_r
    rel = abs(diff) / abs(e0_r)
    passed = diff <= THRESH_E0_REL * abs(e0_r)   # 复值不应更高（超出阈值）
    evidence = (f'E0(complex) = {e0_c:.12f}, E0(real) = {e0_r:.12f}, '
                f'差值 = {diff:.3e}（相对 {rel:.3e}）')
    note = ('复值 ≤ 实值（变分原理），差值在收敛噪声内 → 同一基态'
            if passed else
            '复值 E0 显著高于实值 E0——复值计算可能掉 local minimum 或 '
            '程序问题，建议重算')
    return {'name': '能量变分检查', 'passed': passed,
            'evidence': evidence, 'note': note}


def _pair_stats(re_c, re_r):
    """两组实部的一致性统计，返回 (max, mean)。

    相对误差按该物理量的全局最大幅度归一（denom = max|re_r|）——逐点
    归一在近零点会爆炸（如单粒子关联的节点），不适合一致性检验。
    """
    re_c = np.asarray(re_c, dtype=float)
    re_r = np.asarray(re_r, dtype=float)
    denom = max(float(np.abs(re_r).max()), 1e-300)
    rel = np.abs(re_c - re_r) / denom
    return float(rel.max()), float(rel.mean())


def _imag_ratio(records_complex, getter):
    """虚部量级：max|Im| / max|实部|。"""
    re_max = 0.0
    im_max = 0.0
    for r in records_complex:
        z = getter(r)
        re_max = max(re_max, abs(z.real))
        im_max = max(im_max, abs(z.imag))
    return (im_max / re_max) if re_max > 0 else (1.0 if im_max > 0 else 0.0)


def run_checks(spin_r, spin_c, dens_r, dens_c, sc_r, sc_c,
               sp_r, sp_c, e0_c, e0_r, e0s_c, e0s_r, te_c, te_r):
    """执行全部检查，返回 (checks, verdict)。"""
    checks = []

    # 1. 能量变分
    checks.append(check_energy(e0_c, e0_r))

    # 2. 实部一致性（逐物理量汇总一行）
    rows = []
    for label, rc, rr, getter in (
            ('密度', dens_c, dens_r, lambda x: x._value),
            ('自旋 A 子格', spin_c['A sublattice'], spin_r['A sublattice'],
             lambda x: x._value),
            ('自旋 B 子格', spin_c['B sublattice'], spin_r['B sublattice'],
             lambda x: x._value),
            ('单粒子 corr1', sp_c, sp_r, lambda x: x.value)):
        mx, mn = _pair_stats([getter(x).real for x in rc],
                             [getter(x).real for x in rr])
        rows.append((label, mx, mn))
    for ref_tag in sc_c:
        for label in sc_c[ref_tag]:
            mx, mn = _pair_stats(
                [r.sup_corr for r in sc_c[ref_tag][label]],
                [r.sup_corr for r in sc_r[ref_tag][label]])
            rows.append((f'超导 {ref_tag} {label.split()[0]}', mx, mn))
    worst = max(mx for _, mx, _ in rows)
    checks.append({'name': '实部一致性（复值实部 vs 实值）',
                   'passed': worst < THRESH_REAL_REL,
                   'evidence': '；'.join(f'{l}: max {mx:.2e} / mean {mn:.2e}'
                                         for l, mx, mn in rows[:4])
                   + f'；… 最差 {worst:.2e}',
                   'note': (f'全部相对误差 < {THRESH_REAL_REL} → 复值实部'
                            '与实值结果一致' if worst < THRESH_REAL_REL
                            else '存在超阈值偏差（默认阈值，未授权）')})

    # 3. 虚部量级（机器零 = 无相位信号）
    im_rows = []
    for label, rc, getter in (
            ('密度', dens_c, lambda x: x._value),
            ('自旋 A 子格', spin_c['A sublattice'], lambda x: x._value),
            ('自旋 B 子格', spin_c['B sublattice'], lambda x: x._value),
            ('单粒子 corr1', sp_c, lambda x: x.value)):
        im_rows.append((label, _imag_ratio(rc, getter)))
    for ref_tag in sc_c:
        for label in sc_c[ref_tag]:
            im_rows.append(
                (f'超导 {ref_tag} {label.split()[0]}',
                 _imag_ratio(sc_c[ref_tag][label], lambda x: x._value)))
    worst_im = max(r for _, r in im_rows)
    checks.append({'name': '虚部量级（Im/Re 比）',
                   'passed': worst_im < THRESH_IMAG_REL,
                   'evidence': '；'.join(f'{l}: {r:.2e}' for l, r in
                                         im_rows[:4])
                   + f'；… 最差 {worst_im:.2e}',
                   'note': ('虚部为机器零 → 无手性相位信号（复值基态未'
                            '自发破缺 TRSB）' if worst_im < THRESH_IMAG_REL
                            else '虚部显著——存在相位信号，需进入相位分析')})

    # 4. SU(2) 残差（复值超导）
    errs = []
    for ref_tag in sc_c:
        for label in sc_c[ref_tag]:
            # 直接从通道文件做对称性检验需四通道记录；此处用合成前通道
            pass
    # SU(2) 检验需要四通道原始记录——在 load 时保留；这里用模块函数重载
    su2_note = _su2_residual_check()
    checks.append(su2_note)

    # 5. 收敛轨迹
    if e0s_c and e0s_r:
        de_c = max(abs(np.diff(e0s_c[-10:]))) if len(e0s_c) >= 11 else None
        de_r = max(abs(np.diff(e0s_r[-10:]))) if len(e0s_r) >= 11 else None
        evidence = (f'末 10 步最大 |ΔE|：complex {de_c:.2e} / real '
                    f'{de_r:.2e}；TruncErr：complex {te_c} / real {te_r}')
        passed = (de_c is not None and de_c < 1e-7
                  and (te_c is None or te_c < 1e-5))
        note = ('E0 尾部已饱和、TruncErr 小 → 单档内收敛正常' if passed
                else 'E0 尾部仍有明显变化 → 该档可能未完全收敛')
    else:
        evidence, passed, note = '日志缺失', None, '无日志可查'
    checks.append({'name': '收敛轨迹（单 D 档内）', 'passed': passed,
                   'evidence': evidence, 'note': note})

    # 三态结论
    if any(c['passed'] is False for c in checks if c['name'] != '能量变分检查'):
        verdict = 'minor_deviation'
    elif checks[0]['passed'] is False:
        verdict = 'suspicious'
    elif all(c['passed'] is True for c in checks):
        verdict = 'consistent'
    else:
        verdict = 'minor_deviation'
    return checks, verdict


def _su2_residual_check():
    """复值超导四通道 SU(2) 对称性检验（a↔d、b↔c，逐 refbond/方向）。"""
    sc_root = os.path.join(COMPLEX_ROOT, f'{DIM}_measure_super_corr')
    refdirs = sorted(d for d in os.listdir(sc_root)
                     if re.match(r'super_corr_\d+_\d+$', d))
    worst = 0.0
    detail = []
    for rd in refdirs:
        rb = tuple(int(x) for x in rd.split('_')[-2:])
        for key in 'ABC':
            recs = {}
            for ch in 'abcd':
                fname = f'scs{key}{ch}_fix_refbond_{rb[0]}_{rb[1]}.json'
                recs[ch] = SuperCorrRecord.extract_data(
                    open(os.path.join(sc_root, rd, fname),
                         encoding='utf-8').read(), N, 1.0)
            err_ad, err_bc = SuperCorrProcessor.symmetry_check(
                recs['a'], recs['b'], recs['c'], recs['d'])
            worst = max(worst, err_ad, err_bc)
            detail.append(f'{rd}/{key}: ad {err_ad:.2e}, bc {err_bc:.2e}')
    passed = bool(worst < THRESH_SU2)
    return {'name': 'SU(2) 通道一致性（复值超导）',
            'passed': passed,
            'evidence': '；'.join(detail[:3]) + f'；… 最差 {worst:.2e}',
            'note': (f'残差 < {THRESH_SU2} → 四通道自洽' if passed
                     else '残差超经验阈值——复值超导通道不自洽，合成值存疑')}


# ==========================================================================
# 对比图（复用现有 difference 系列 + 少量自绘）
# ==========================================================================
def save(name):
    path = os.path.join(OUTDIR, f'{name}.png')
    os.makedirs(OUTDIR, exist_ok=True)
    plt.gcf().savefig(path, dpi=200, bbox_inches='tight')
    plt.close('all')
    return path


def fig_density(dens_r, dens_c):
    DensityProcessor.Line_Chart_Visualization_difference(
        {'real': dens_r, 'complex': dens_c})
    return save('fig_density_diff')


def fig_spin(spin_r, spin_c):
    CorrelationProcessor.Spin_Corr_Visualization_difference(
        {'real': spin_r, 'complex': spin_c})
    return save('fig_spin_diff')


def fig_supercorr(sc_r, sc_c):
    # 行 = reference A/B/C（refbond），列 = real / complex / 绝对误差，
    # 每格内 target A/B/C 三键同图
    SuperCorrProcessor.Super_Corr_Visualization_difference_refbond(
        {'real': sc_r, 'complex': sc_c})
    return save('fig_supercorr_diff')


def fig_supercorr_imag(sc_c):
    """超导关联虚部检查图：Im⟨Δ†Δ⟩ vs 距离（含量级参考）。"""
    ref_tags = list(sc_c.keys())
    n_rows = len(ref_tags)
    fig, axs = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
    axs = axs.reshape(n_rows, 3)
    colors = ['blue', 'green', 'orange']
    for i, ref_tag in enumerate(ref_tags):
        for k, label in enumerate(SUPER_LABELS.values()):
            recs = sorted(sc_c[ref_tag][label], key=lambda r: r.distance)
            x = [r.distance for r in recs]
            y = [r.sup_corr_complex.imag for r in recs]
            c = colors[k % 3]
            axs[i, 0].scatter(x, y, color=c, s=50, zorder=3)
            axs[i, 0].plot(x, y, color=c, alpha=0.5, linewidth=1.5)
            axs[i, 1].scatter(x, np.abs(y), color=c, s=50, zorder=3)
            axs[i, 2].scatter(x, np.abs(y), color=c, s=50, zorder=3)
        axs[i, 1].set_yscale('log')
        axs[i, 2].set_xscale('log')
        axs[i, 2].set_yscale('log')
        axs[i, 0].set_ylabel(f'{ref_tag}\nIm part', fontsize=13)
        axs[i, 0].axhline(y=0, color='black', linewidth=2.5, zorder=1)
    for j, t in enumerate(('Linear', 'Log-Linear (|Im|)', 'Log-Log (|Im|)')):
        axs[0, j].set_title(t, fontsize=16)
    fig.suptitle(r'Imaginary part of $\langle \Delta^\dagger \Delta \rangle$'
                 r' (complex run) — machine-zero check', fontsize=18,
                 fontweight='bold')
    plt.tight_layout()
    return save('fig_supercorr_imag')


def fig_single_particle(sp_r, sp_c):
    """corr1 对比：real / complex 实部 / 绝对误差（1×3）。"""
    sp_r = sorted(sp_r, key=lambda x: x.distance)
    sp_c = sorted(sp_c, key=lambda x: x.distance)
    x = [r.distance for r in sp_r]
    y1 = [r.value.real for r in sp_r]
    y2 = [r.value.real for r in sp_c]
    y3 = [abs(a - b) for a, b in zip(y1, y2)]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, y, t in ((axs[0], y1, 'real'), (axs[1], y2, 'complex'),
                     (axs[2], y3, 'Absolute Error')):
        ax.scatter(x, y, color='blue', s=50, zorder=3)
        ax.plot(x, y, color='red', alpha=0.5, linewidth=1.5)
        ax.set_title(t, fontsize=16)
        ax.set_xlabel('distance r', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
    fig.suptitle(r'Single-particle correlation $G(r)$ (corr1) — real vs complex',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    return save('fig_single_particle_diff')


def fig_energy(e0s_c, e0s_r):
    """两份日志 E0 收敛轨迹同图。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for e0s, label, c in ((e0s_r, 'real', 'tab:blue'),
                          (e0s_c, 'complex', 'tab:orange')):
        if e0s:
            ax.plot(e0s, label=label, color=c, linewidth=1.5)
    ax.set_xlabel('sweep index', fontsize=14)
    ax.set_ylabel('E0', fontsize=14)
    ax.set_title('DMRG optimization trajectory: real vs complex',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return save('fig_energy_compare')


# ==========================================================================
# 报告
# ==========================================================================
def render_markdown(checks, verdict, figures) -> str:
    lines = [
        '# 复值 vs 实值 DMRG 数据对比报告',
        '',
        f'- 参数：N={N}，t/J={T_J.replace("t_J_", "")}，'
        f'doping {DOPING}（δ={DELTA}），D={DIM}',
        f'- 生成时间：{datetime.datetime.now().isoformat(timespec="seconds")}',
        '',
        '> 对比口径：主通道 = 复值实部 vs 实值结果；检验通道 = 虚部量级'
        '（机器零 = 无手性相位信号）。阈值为默认值'
        '（defaults_not_authorized，未获物理授权），仅供参考。',
        '',
        f'## 结论：**{verdict}**',
        '',
        '## 检查明细',
        '',
        '| 检查项 | 结果 | 证据 |',
        '|---|---|---|',
    ]
    for c in checks:
        mark = ('✅ 通过' if c['passed'] is True else
                ('❌ 未通过' if c['passed'] is False else '— 证据不足'))
        lines.append(f"| {c['name']} | {mark} | {c['evidence']} |")
    lines += ['', '## 图件', '']
    for f in figures:
        lines.append(f'- ![图]({f})')
    lines.append('')
    return '\n'.join(lines)


def _prompt_config(n_explicit, t_j_explicit, doping_explicit):
    """缺省参数的交互式获取（与 Convergence_Report 同款提示结构）：
    N（默认 200）、t/J（默认 3）、空穴率（默认 3_25，即当前
    Data/real|complex 对应的 δ=0.12）。非交互环境回退默认值。"""
    def _ask(prompt, default):
        try:
            raw = input(prompt).strip()
        except EOFError:
            return default
        return raw if raw else default

    n = n_explicit
    if n is None:
        n = int(_ask('请输入格点数 N（回车使用默认 200）: ', '200'))
    t_j = t_j_explicit
    if t_j is None:
        t_j_val = float(_ask('请输入 t/J（回车使用默认 3）: ', '3'))
        t_j = f't_J_{t_j_val:g}'
    doping = doping_explicit
    if doping is None:
        doping = _ask('请输入空穴率（如 3_25 d3_25 或 0.12；'
                      '回车使用默认 3_25）: ', '3_25')
    doping = par._fraction_to_doping(doping)
    if doping is None:
        raise ValueError(f'无法解析空穴率输入: {doping_explicit!r}'
                         '（示例: 3_25、0.12、d3_25）')
    return n, t_j, doping


def _doping_to_delta(doping):
    """d3_25 → 0.12（报告显示用）。"""
    m = re.match(r'^d(\d+)_(\d+)$', doping)
    if not m:
        return None
    return int(m.group(1)) / int(m.group(2))


def main(argv=None) -> int:
    global N, T_J, DOPING, DELTA, DIM, REAL_ROOT, COMPLEX_ROOT
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser(
        description='复值 vs 实值 DMRG 数据对比与合理性诊断'
                    '（不判相，详见脚本 docstring）')
    parser.add_argument('--N', type=int, default=None,
                        help='格点数（默认 200）；不提供时交互式提示输入')
    parser.add_argument('--t-j', default=None,
                        help='t/J 目录后缀（如 t_J_3）；不提供时交互式'
                             '提示输入 t/J 值（默认 3）')
    parser.add_argument('--doping', default=None,
                        help='空穴率（如 3_25 或 0.12）；不提供时交互式'
                             '提示输入（默认 3_25）')
    parser.add_argument('--dim', default=DIM,
                        help='bond dimension 档（默认 1500；'
                             '复值数据有 1500/5000 两档）')
    args = parser.parse_args(argv)

    N, T_J, DOPING = _prompt_config(args.N, args.t_j, args.doping)
    DELTA = _doping_to_delta(DOPING)
    DIM = str(args.dim)
    REAL_ROOT = _data_root('real')
    COMPLEX_ROOT = _data_root('complex')

    print(f'=== 复值 vs 实值对比（N={N}, t/J={T_J}, {DOPING}, D={DIM}）===')
    for root, tag in ((REAL_ROOT, 'real'), (COMPLEX_ROOT, 'complex')):
        if not os.path.isdir(root):
            print(f'错误：数据目录不存在: {root}')
            return 1
    os.makedirs(OUTDIR, exist_ok=True)

    # 装载
    print('装载数据…')
    dens_r, dens_c = load_density(REAL_ROOT), load_density(COMPLEX_ROOT)
    spin_r, spin_c = load_spin(REAL_ROOT), load_spin(COMPLEX_ROOT)
    sc_r, sc_c = load_supercorr(REAL_ROOT), load_supercorr(COMPLEX_ROOT)
    sp_r, sp_c = (load_single_particle(REAL_ROOT),
                  load_single_particle(COMPLEX_ROOT))
    e0s_r, te_r = load_e0_series(REAL_ROOT)
    e0s_c, te_c = load_e0_series(COMPLEX_ROOT)
    e0_r = e0s_r[-1] if e0s_r else None
    e0_c = e0s_c[-1] if e0s_c else None

    # 检查
    print('运行五项检查…')
    checks, verdict = run_checks(spin_r, spin_c, dens_r, dens_c, sc_r, sc_c,
                                 sp_r, sp_c, e0_c, e0_r, e0s_c, e0s_r,
                                 te_c, te_r)

    # 图
    print('生成对比图…')
    figures = []
    for fn, args, tag in (
            (fig_density, (dens_r, dens_c), '密度对比'),
            (fig_spin, (spin_r, spin_c), '自旋对比（A/B 子格）'),
            (fig_supercorr, (sc_r, sc_c), '超导对比（3 refbond）'),
            (fig_supercorr_imag, (sc_c,), '超导虚部检查'),
            (fig_single_particle, (sp_r, sp_c), '单粒子对比（corr1）'),
            (fig_energy, (e0s_c, e0s_r), '能量收敛轨迹')):
        try:
            figures.append(os.path.relpath(fn(*args), OUTDIR))
            print(f'  ✓ {tag}')
        except Exception as ex:  # noqa: BLE001
            print(f'  ✗ {tag}: {ex}')

    # 报告
    md = render_markdown(checks, verdict, figures)
    md_path = os.path.join(OUTDIR, 'comparison_report.md')
    with open(md_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(md)
    print()
    for c in checks:
        mark = '通过' if c['passed'] is True else ('未通过' if c['passed']
                                                  is False else '证据不足')
        print(f"  [{mark}] {c['name']}: {c['evidence']}")
    print(f'\n结论: {verdict}')
    print(f'报告: {md_path}')
    print('说明: 阈值为默认值（未授权）；本程序只诊断数值合理性，不判相。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
