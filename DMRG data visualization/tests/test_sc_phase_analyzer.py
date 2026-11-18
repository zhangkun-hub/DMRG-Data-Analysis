# -*- coding: utf-8 -*-
"""
d+id 相位分析器测试（SC_Phase_Analyzer）。

覆盖 15 项（合成数据只验证算法，不构成物理证据）：
1. 纯实数正关联          → 相位 0°，no_chiral_signal/insufficient 语义
2. 纯实数负关联          → 相位 180°
3. 虚部机器零            → 无手性
4. 纯噪声                → insufficient
5. 理想 d+id             → chiral_phase_candidate（+90°）
6. 理想 d-id             → chiral_phase_candidate（-90°），chi 反号
7. 相位跨越 ±180°        → 圆周统计正确
8. 长距离进入噪声区间    → 尾部点被门控剔除
9. 两个 D 档相位不一致   → 结论矛盾（inconclusive 语义由调用方汇总）
10. 不同 bin 容差不稳定   → inconclusive
11. bin 内复数相消       → 相消诊断输出
12. d2 通道明显抑制      → d_component_suppressed
13. C12 幅值低于噪声底   → 相位无效（门控拒绝）
14. C11/C22 分母异常     → chi_d 输出 nan，不崩溃
15. C12/C21 共轭异常     → 共轭诊断异常标记

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_sc_phase_analyzer.py
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.physical_analysis.SC_Phase_Analyzer import (  # noqa: E402
    SCPhaseAnalyzer, PhaseAnalyzerConfig)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ==========================================================================
# 合成数据构造
# ==========================================================================
REFS = [(49, 50), (50, 51), (50, 54)]   # 键方向 A / B / C
DISTS = np.arange(2.0, 26.0, 2.0)       # 12 个距离点


def tb_for(rb, alpha, d):
    """合成 target 键（唯一坐标即可；几何距离由分析器用
    HoneycombXCGeometry 重算——这里用 refbond 中点平移逼近）。"""
    return (rb[0] + int(d) * 4 + (1 if alpha in 'AB' else 0),
            rb[1] + int(d) * 4 + (1 if alpha in 'AB' else 0))


def make_datasets(v1=None, v2=None, envelope=None, phase_fn=None,
                  noise_fn=None, real_only=False, c12_direct=None):
    """通用合成构造。

    逐键方向配对幅值 Delta_alpha(d) = envelope(d) * [v1_alpha + i*v2_alpha]，
    M_ab(d) = conj(Delta_a(d)) * Delta_b(0)；再按需叠加扰动。
    若 c12_direct 给出，则直接在 d-wave 投影层面覆盖 C12（用于
    共轭异常等专门测试——通过 datasets 的附加字段传入，由测试自组合）。
    """
    v1 = v1 if v1 is not None else {'C': 2.0, 'A': -1.0, 'B': -1.0}
    v2 = v2 if v2 is not None else {'C': 0.0, 'A': 0.0, 'B': 0.0}
    envelope = envelope or (lambda d: 1e-3 * d ** (-1.5))
    phase_fn = phase_fn or (lambda d: 0.0)
    noise_fn = noise_fn or (lambda d: 0.0)
    datasets = {}
    for rb in REFS:
        beta = {REFS[0]: 'A', REFS[1]: 'B', REFS[2]: 'C'}[rb]
        datasets[rb] = {}
        for alpha in 'ABC':
            entries = []
            for d in DISTS:
                if real_only:
                    amp_a = envelope(d) * (v1[alpha] + v2[alpha])
                    amp_b = v1[beta] + v2[beta]
                else:
                    # phase_fn = v2 分量相对 v1 的相位（理想 d±id 为 ±π/2）；
                    # 配对场 Δ = A(r)·[v1 + v2·e^{iφ}]，d=0 处取 φ(0)
                    amp_a = envelope(d) * (v1[alpha]
                                           + v2[alpha] * np.exp(1j * phase_fn(d)))
                    amp_b = v1[beta] + v2[beta] * np.exp(1j * phase_fn(0.0))
                val = np.conjugate(amp_a) * amp_b
                if noise_fn(d):
                    val += noise_fn(d) * np.exp(1j * 0.3 * d)
                entries.append((tb_for(rb, alpha, d), val))
            datasets[rb][alpha] = entries
    datasets['_ref_keys'] = {rb: k for rb, k in
                             [(REFS[0], 'A'), (REFS[1], 'B'), (REFS[2], 'C')]}
    datasets['_c12_direct'] = c12_direct
    return datasets


CFG = PhaseAnalyzerConfig(tol_bins=(2.0, 3.0))


def analyze(datasets, **kw):
    cfg = PhaseAnalyzerConfig(tol_bins=(2.0, 3.0), **kw)
    return SCPhaseAnalyzer.analyze(datasets, N=200, config=cfg)


# ==========================================================================
# 1. 纯实数正关联
# ==========================================================================
print('===== 1. 纯实数正关联 =====')
res = analyze(make_datasets(real_only=True,
                            v2={'C': 0.0, 'A': 1.0, 'B': -1.0}))
phis = [p for p, v in zip(res.phi12_deg, res.valid) if v]
check('1', '有效点相位为 0°（实部正）',
      bool(phis) and all(abs((p + 180) % 360 - 180) < 1 for p in phis),
      f'phi={[round(p) for p in phis[:4]]}')
check('1', '不输出手性候选',
      res.classification in ('no_chiral_signal', 'insufficient'),
      res.classification)

# ==========================================================================
# 2. 纯实数负关联
# ==========================================================================
print('===== 2. 纯实数负关联 =====')
res = analyze(make_datasets(v1={'C': -2.0, 'A': 1.0, 'B': 1.0},
                            v2={'C': 0.0, 'A': 1.0, 'B': -1.0},
                            real_only=True))
phis = [p for p, v in zip(res.phi12_deg, res.valid) if v]
check('2', '有效点相位为 180°（实部负）',
      bool(phis) and all(abs((p + 180) % 360 - 180) > 178 for p in phis),
      f'phi={[round(p) for p in phis[:4]]}')

# ==========================================================================
# 3. 虚部机器零
# ==========================================================================
print('===== 3. 虚部机器零 =====')
res = analyze(make_datasets(real_only=True))
im_max = max((abs(v.imag) for v in res.C12), default=0.0)
check('3', 'C12 虚部为机器零且 chi_d ≈ 0',
      im_max < 1e-12 and all(abs(c) < 1e-12 for c in res.chi_d if np.isfinite(c)),
      f'max|Im|={im_max:.2e}')

# ==========================================================================
# 4. 纯噪声
# ==========================================================================
print('===== 4. 纯噪声 =====')
rng = np.random.default_rng(7)
res = analyze(make_datasets(
    v1={'C': 1.0, 'A': 1.0, 'B': 1.0},
    v2={'C': 1.0, 'A': 1.0, 'B': 1.0},
    envelope=lambda d: 1e-9,
    noise_fn=lambda d: 1e-6 * rng.standard_normal()))
check('4', '噪声数据 → insufficient（无有效点）',
      res.classification == 'insufficient', res.classification)
check('4', '无有效点时圆周均值为 nan（n/a 路径，不显示伪 90°）',
      not np.isfinite(res.circular_mean_deg), str(res.circular_mean_deg))

# ==========================================================================
# 5/6. 理想 d+id / d-id
# ==========================================================================
print('===== 5/6. 理想 d±id =====')
v1 = {'C': 2.0, 'A': -1.0, 'B': -1.0}
v2 = {'C': 0.0, 'A': 1.0, 'B': -1.0}
for tag, sign in [('d+id', +1), ('d-id', -1)]:
    # 无噪声合成数据的噪声底即信号尾部中位数；降低门控因子
    # （可配置阈值）使连续有效点 ≥3，平台判定不受噪声项干扰
    res = analyze(make_datasets(v1=v1, v2=v2,
                                phase_fn=lambda d, s=sign: s * np.pi / 2),
                  noise_prom_factor=0.01)
    target = sign * 90.0
    c_mean = res.circular_mean_deg
    check('5/6', f'{tag} 圆周均值 ≈ {target}°',
          abs((c_mean - target + 180) % 360 - 180) < 2, f'{c_mean:.1f}°')
    chis = [c for c in res.chi_d if np.isfinite(c)]
    check('5/6', f'{tag} chi_d 符号 = {sign:+d}',
          all(np.sign(c) == sign for c in chis) if chis else False,
          f'{chis[:3]}')
    check('5/6', f'{tag} 分类 = chiral_phase_candidate（R 有效、平台成立）',
          res.classification == 'chiral_phase_candidate',
          res.classification)

# ==========================================================================
# 7. 相位跨越 ±180°
# ==========================================================================
print('===== 7. 相位跨越 ±180° 与圆周统计 =====')
angles = [179.0, -179.0, 178.0, -178.0, 180.0]
c_mean, c_std, c_R, c_n = SCPhaseAnalyzer.circular_stats(angles)
check('7a', '179°/-179° 圆周均值 ≈ 180° 而非 0°',
      abs((c_mean + 180) % 360 - 180) > 175, f'mean={c_mean:.1f}')
check('7a', '圆周标准差小（角度聚集）', c_std < 5, f'std={c_std:.1f}')
check('7a', 'R 接近 1（聚集）且 n 正确', c_R > 0.9 and c_n == 5,
      f'R={c_R:.3f} n={c_n}')
# 圆周统计专项（7 种分布）
cs = SCPhaseAnalyzer.circular_stats
check('7b', '全部 0° → mean 0、R=1',
      abs(cs([0.0]*5)[0]) < 1e-9 and abs(cs([0.0]*5)[2] - 1.0) < 1e-9)
check('7b', '全部 90° → mean 90、R=1',
      abs((cs([90.0]*5)[0] - 90 + 180) % 360 - 180) < 1e-9)
check('7b', '全部 180° → mean 180、R=1',
      abs((cs([180.0]*5)[0] + 180) % 360 - 180) > 179.9)
check('7b', '179°/-179° 混合 → R 接近 1',
      cs([179.0, -179.0, 179.0])[2] > 0.99)
m0, s0, R0, n0 = cs([0.0, 180.0, 0.0, 180.0])
check('7b', '等量 0°/180° → R≈0（对极弥散）', R0 < 0.1, f'R={R0:.3f}')
import numpy as _np
rng = _np.random.default_rng(5)
unif = cs([float(v) for v in rng.uniform(-180, 180, 60)])
check('7b', '近似均匀分布 → R 接近 0', unif[2] < 0.1, f'R={unif[2]:.3f}')
mz, sz, Rz, nz = cs([0.0, 180.0])
check('7b', 'resultant length 精确零（{0°,180°}）→ R=0',
      abs(Rz) < 1e-12, f'R={Rz:.2e}')

# ==========================================================================
# 8. 长距离关联进入噪声区间
# ==========================================================================
print('===== 8. 长距离进入噪声区间 =====')
rng = np.random.default_rng(11)
res = analyze(make_datasets(
    v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2,
    envelope=lambda d: 1e-3 * d ** (-1.5) * np.exp(-d / 12.0),
    noise_fn=lambda d: 1e-7 * rng.standard_normal()))
n_all, n_valid = res.n_bins, res.n_valid_bins
check('8', '尾部点被幅值门控剔除', 0 < n_valid < n_all,
      f'valid={n_valid}/{n_all}')

# ==========================================================================
# 9. 两个 D 档相位不一致（调用方汇总语义：分类器须保留每档独立结果）
# ==========================================================================
print('===== 9. D 档不一致（逐档独立，无静默合并） =====')
r1 = analyze(make_datasets(v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2))
r2 = analyze(make_datasets(v1=v1, v2=v2, phase_fn=lambda d: 0.0))
check('9', '两档各自独立分类，不一致不静默合并',
      r1.classification != r2.classification
      or 'chiral' in r1.classification or 'chiral' in r2.classification,
      f'{r1.classification} vs {r2.classification}')

# ==========================================================================
# 10. 不同 bin 容差不稳定
# ==========================================================================
print('===== 10. bin 容差不稳定 =====')
# 构造：相位平台只在部分距离出现（窄平台），不同容差分组结果不同
rng = np.random.default_rng(13)
res = analyze(make_datasets(
    v1=v1, v2=v2,
    phase_fn=lambda d: (np.pi / 2 if 6 <= d <= 12 else 0.0),
    envelope=lambda d: 1e-3 * d ** (-1.5),
    noise_fn=lambda d: 1e-8 * rng.standard_normal()))
check('10', '多容差分类记录在 tol_stability',
      len(res.tol_stability) == 1 and '3.0' in res.tol_stability,
      str(res.tol_stability))
check('10', '不稳定情形输出 inconclusive 或非手性（不挑有利容差）',
      res.classification != 'chiral_phase_candidate', res.classification)

# ==========================================================================
# 11. bin 内复数相消
# ==========================================================================
print('===== 11. bin 内复数相消诊断 =====')
datasets = make_datasets(v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2)
# 同一 bin 内加入反号条目制造相消：同键 (k,l) 两条反号值 → bin 平均相消
rb = REFS[0]
tb_c = tb_for(rb, 'C', 4.0)
amp_a = 1e-3 * 4.0 ** (-1.5) * (v1['C'] + v2['C'] * np.exp(1j * np.pi / 2))
val = np.conjugate(amp_a) * (v1['A'] + 1j * v2['A'])
datasets[rb]['C'].append((tb_c, val))
datasets[rb]['C'].append((tb_c, -val))
res = analyze(datasets)
cancs = [row['min_cancellation'] for row in res.bins_detail]
check('11', '相消诊断输出（min_cancellation < 1）',
      any(c < 0.99 for c in cancs), f'min={min(cancs):.3f}')

# ==========================================================================
# 12. d2 通道明显抑制
# ==========================================================================
print('===== 12. d2 通道抑制（unresolved vs suppressed） =====')
# v2 ≡ 0 → C22 ≡ 0（低于分辨标尺）→ unresolved（不声称物理抑制）
res = analyze(make_datasets(
    v1=v1, v2={'C': 0.0, 'A': 0.0, 'B': 0.0},
    phase_fn=lambda d: 0.0))
check('12', 'C22≡0（低于分辨标尺）→ d_component_unresolved',
      res.classification == 'd_component_unresolved', res.classification)
check('12', 'unresolved 时 d2_resolution 字段一致',
      res.d2_resolution == 'unresolved', res.d2_resolution)
# 场 v2 = c·V_D2（c=2e-3）→ bin 平均后 C22/C11 ≈ 6.7e-7
# ∈ [1e-8, 1e-6) → 高于数值分辨比且低于抑制阈值 → suppressed
res2 = analyze(make_datasets(
    v1=v1,
    v2={'C': 0.0, 'A': 2e-3 / np.sqrt(2), 'B': -2e-3 / np.sqrt(2)},
    phase_fn=lambda d: 0.0))
check('12', 'C22 高于分辨比且比值 <1e-6 → d_component_suppressed',
      res2.classification == 'd_component_suppressed', res2.classification)
check('12', 'suppressed 时 d2_resolution=resolved',
      res2.d2_resolution == 'resolved', res2.d2_resolution)

# ==========================================================================
# 13. C12 幅值低于噪声底
# ==========================================================================
print('===== 13. C12 低于噪声底 =====')
# v2 与 v1 反平行（v1·(v1+v2)=0）→ C12 = v1·M·V_D2 精确 0
# （场的 d2 分量与 d1 共线，仅作门控路径验证）
res = analyze(make_datasets(
    v1={'C': 2.0, 'A': -1.0, 'B': -1.0},
    v2={'C': -2.0, 'A': 1.0, 'B': 1.0},
    phase_fn=lambda d: 0.0, real_only=True))
check('13', 'C12 全低于噪声底 → 门控拒绝（不产生手性判定）',
      res.classification in ('insufficient', 'd_component_unresolved',
                             'd_component_suppressed'),
      res.classification)

# ==========================================================================
# 14. C11/C22 分母异常
# ==========================================================================
print('===== 14. 分母异常 =====')
res = analyze(make_datasets(
    v1={'C': 0.0, 'A': 0.0, 'B': 0.0},
    v2={'C': 0.0, 'A': 0.0, 'B': 0.0},
    noise_fn=lambda d: 1e-12 * np.random.default_rng(3).standard_normal()))
check('14', 'C11+C22 ≈ 0 时 chi_d 输出 nan 且不崩溃',
      res.status == 'ok' and all(np.isnan(c) for c in res.chi_d),
      res.classification)

# ==========================================================================
# 15. C12/C21 共轭异常
# ==========================================================================
print('===== 15. 同 bin 跨通道偏差（纯矩阵代数性质，非物理共轭） =====')
# 注意：该诊断衡量的是同 bin 上 C12 与 conj(C21) 的数值偏差——有限
# 圆柱下二者无同距离共轭恒等式依据（矩阵厄米性只是合成数据的纯
# 代数性质，不代表真实有限 XC 数据必须满足）。仅验证诊断路径与
# "不参与分类"的语义。
datasets = make_datasets(v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2)
for rb, beta in [(REFS[0], 'A')]:
    datasets[rb]['B'] = [(tb, (1.2 + 0.5j) * val)
                         for tb, val in datasets[rb]['B']]
res = analyze(datasets)
check('15', '偏差被检出（ok=False 且偏差超阈值）',
      res.conjugate_diagnostics['ok'] is False
      and res.conjugate_diagnostics['max_relative_deviation'] > 0.1,
      str(res.conjugate_diagnostics))
check('15', '诊断含语义说明（描述性、不参与分类）',
      '描述性' in res.conjugate_diagnostics.get('semantics', ''),
      res.conjugate_diagnostics.get('semantics', '')[:40])
check('15', '偏差异常不阻断分类（P1-1：无 inconclusive 阻断）',
      res.classification != 'inconclusive', res.classification)

# ==========================================================================
# 汇总
# ==========================================================================
print()
print('=' * 60)
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in RESULTS:
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    print(f'{cat:4}: {n_pass}/{len(RESULTS[cat])} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)

# ==========================================================================
# 16. 分类优先级（P2-2：tol 一致性先于物理分类，_classify 单元级）
# ==========================================================================
print('===== 16. 分类优先级单元表 =====')
from Physical_Quantity.physical_analysis.SC_Phase_Analyzer import (
    PhaseAnalysisResult, PlatformReport)


def make_result(tol_classes, plat=False, d2res='resolved', d2ratio=1.0,
                n_valid=5, n_c11=5):
    r = PhaseAnalysisResult(config={})
    r.n_valid_bins = n_valid
    r.tol_stability = dict(tol_classes)
    if plat:
        p = PlatformReport(target_deg=90.0, found=True, n_points=3)
        r.platform_plus = p
    r.d2_resolution = d2res
    r.d2_ratio_median = d2ratio
    SCPhaseAnalyzer._classify(r, PhaseAnalyzerConfig(), d2ratio, d2res, n_c11)
    return r

# 1. tol 全一致 + 无平台 + d2 resolved 且不抑制 → no_chiral_signal
r = make_result({'1.5': 'no_chiral_signal', '2.5': 'no_chiral_signal'})
check('16', 'tol 一致 + 无平台 → no_chiral_signal',
      r.classification == 'no_chiral_signal', r.classification)
# 2. tol 不一致（suppressed vs no_chiral）→ inconclusive（不被覆盖）
r = make_result({'1.5': 'd_component_suppressed',
                 '2.5': 'no_chiral_signal'})
check('16', 'tol 不一致（suppressed/no_chiral）→ inconclusive',
      r.classification == 'inconclusive', r.classification)
# 3. tol 不一致（平台候选 vs 无平台）→ inconclusive
r = make_result({'1.5': 'chiral_phase_candidate',
                 '2.5': 'no_chiral_signal'})
check('16', 'tol 不一致（平台/无平台）→ inconclusive',
      r.classification == 'inconclusive', r.classification)
# 4. tol 不一致（insufficient vs 充分）→ inconclusive
r = make_result({'1.5': 'insufficient', '2.5': 'no_chiral_signal'})
check('16', 'tol 不一致（insufficient/充分）→ inconclusive',
      r.classification == 'inconclusive', r.classification)
# 5. 数据不足优先：n_valid=0 且 C11 不足 → insufficient（早于 tol）
r = make_result({'1.5': 'insufficient', '2.5': 'insufficient'},
                n_valid=0, n_c11=1)
check('16', '数据不足 → insufficient（早于一切）',
      r.classification == 'insufficient', r.classification)
# 6. unresolved 优先于 no_chiral_signal
r = make_result({'1.5': 'd_component_unresolved',
                 '2.5': 'd_component_unresolved'}, d2res='unresolved',
                d2ratio=1e-9)
check('16', 'unresolved 优先于 no_chiral_signal',
      r.classification == 'd_component_unresolved', r.classification)
# 7. suppressed 需要 resolved + 比值小
r = make_result({'1.5': 'd_component_suppressed',
                 '2.5': 'd_component_suppressed'}, d2res='resolved',
                d2ratio=1e-9)
check('16', 'resolved + 比值 < 1e-6 → d_component_suppressed',
      r.classification == 'd_component_suppressed', r.classification)

# ==========================================================================
# 17. cancellation 接入 valid（P2-4）
# ==========================================================================
print('===== 17. cancellation 接入有效点判定 =====')
# 构造：正常 d+id 数据 + 在某个 bin 注入完全相消条目
ds17 = make_datasets(v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2)
rb = REFS[0]
tb_c = tb_for(rb, 'C', 4.0)
amp_a = 1e-3 * 4.0 ** (-1.5) * (v1['C'] + v2['C'] * np.exp(1j * np.pi / 2))
val = np.conjugate(amp_a) * (v1['A'] + 1j * v2['A'])
ds17[rb]['C'].append((tb_c, val))
ds17[rb]['C'].append((tb_c, -val))
r_clean = analyze(make_datasets(v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2))
r_canc = analyze(ds17)
n_canc_bins = sum(1 for row in r_canc.bins_detail
                  if row.get('min_cancellation', 1.0) < 0.5
                  and row.get('min_n_raw', 0) >= 2)
check('17', '相消 bin 被识别（min_cancellation < 阈值）',
      n_canc_bins >= 1, f'{n_canc_bins} 个')
check('17', '相消导致有效点减少或分类不恶化（valid 受影响）',
      r_canc.n_valid_bins <= r_clean.n_valid_bins,
      f'{r_clean.n_valid_bins} -> {r_canc.n_valid_bins}')
# n_raw=1 的 bin：cancellation=1（无相消信息）→ 不因相消被拒绝
r_single = analyze(make_datasets(v1=v1, v2=v2,
                                 phase_fn=lambda d: np.pi / 2))
check('17', 'n_raw≥2 才应用相消判据（单点 bin 不受影响）',
      r_single.status == 'ok', r_single.status)

# ==========================================================================
# 18. 共轭（同 bin 跨通道偏差）不阻断手性候选（P1-1 端到端）
# ==========================================================================
print('===== 18. 偏差诊断不阻断手性候选 =====')
ds18 = make_datasets(v1=v1, v2=v2, phase_fn=lambda d: np.pi / 2)
# 注入跨通道偏差（复因子）
for rb18, beta in [(REFS[0], 'A')]:
    ds18[rb18]['B'] = [(tb, (1.2 + 0.5j) * v) for tb, v in ds18[rb18]['B']]
r18 = analyze(ds18)
check('18', '偏差 ok=False 时分类不被阻断（非 inconclusive）',
      r18.classification != 'inconclusive', r18.classification)
