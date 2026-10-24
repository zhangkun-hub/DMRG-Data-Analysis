# -*- coding: utf-8 -*-
"""
PairingSymmetryAnalyzer 阶段 3 定向测试（D5：公共窗口振幅重拟合 +
两级矩阵模型 + 可辨识性诊断）

覆盖：
- A. 全链路：3 参考键 × 3 方向合成文件 → 阶段 1–2 → 公共窗口重拟合
  （振幅与直调 fit_envelope_logspace 逐位一致）→ 两级矩阵拟合
  （非对称秩一真值：一般模型恢复 u/v、对称模型被拒）
- B. 两级模型纯函数：对称真值恢复、模型比较（AICc 仅参考统计量）、
  非对称拒绝对称、近零分量警告、部分矩阵/不可辨识拒绝、NaN 掩码
- C. 失败与语义：重拟合点数不足、振幅不含符号判据、未校准阈值标记、
  严格 JSON 序列化

合成输入；真实接口调用（group_by_distance / fit_envelope_logspace /
SuperCorrDecayAnalyzer.analyze）只读。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_pairing_symmetry_matrix.py
"""
import json
import math
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Physical_Data_Record import SuperCorrRecord
from Physical_Quantity.Correlation_Decay_Analyzer import (
    AnalyzerConfig, CorrelationDecayAnalyzer)
from Physical_Quantity.Super_Correlation_Analyzer import (
    SuperCorrDecayAnalyzer)
from Physical_Quantity.Pairing_Symmetry_Analyzer import (
    PairingSymmetryConfig, build_comparable_set, refit_amplitudes,
    fit_general_rank1, fit_symmetric_rank1, compare_rank1_models,
    run_stage3, LAYER_OSC, LAYER_NON_OSC)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def write_channel(ref_dir, key, ch, refbond, entries):
    path = os.path.join(
        ref_dir, f'scs{key}{ch}_fix_refbond_{refbond[0]}_{refbond[1]}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[[i, j, k, l], [v, 0]] for (i, j, k, l, v) in entries], f)
    return path


# ======================================================================
# A. 全链路：合成 3 参考键 × 3 方向 → 重拟合 → 两级矩阵模型
# ======================================================================
print('===== A. 全链路 =====')
tmp_a = tempfile.mkdtemp(prefix='d53_a_')
g200 = HoneycombXCGeometry(200, 1.0)
super_root = os.path.join(tmp_a, 'root', 't_J_3', 'd_syn',
                          '1500_measure_super_corr')
os.makedirs(super_root)
# 振幅真值：A_{αβ} = A0·c_α·d_β（一般秩一、非对称：c≠d）
A0 = 1e-4
c = {'A': 1.0, 'B': 1.2, 'C': 0.8}
d = {'A': 1.0, 'B': 0.9, 'C': 1.1}
refbonds = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
for rcls, rb in refbonds.items():
    ref_dir = os.path.join(super_root, f'super_corr_{rb[0]}_{rb[1]}')
    os.makedirs(ref_dir, exist_ok=True)
    for key in 'ABC':
        ents = []
        for m in range(1, 13):
            k, l = rb[0] + 8 * m, rb[1] + 8 * m
            r = g200.sup_corr_distance(rb[0], rb[1], k, l)
            v = A0 * c[rcls] * d[key] * r ** (-2.0)
            ents.append([rb[0], rb[1], k, l, v])
        ents_b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents]
        for ch in 'ad':
            write_channel(ref_dir, key, ch, rb, ents)
        for ch in 'bc':
            write_channel(ref_dir, key, ch, rb, ents_b)

rep = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_root}, config=PairingSymmetryConfig())
cs = rep['comparable_set']
check('A', '阶段 1–2：9 元素全部通过质量门（共同形式 K、分层归属 '
      'non_osc、层内公共窗口存在）',
      cs['status'] == 'full' and cs['common_form'] == 'K'
      and len(cs['member_keys']) == 9
      and len(cs['layers'][LAYER_NON_OSC]['member_keys']) == 9
      and cs['layers'][LAYER_NON_OSC]['common_window'] is not None,
      f"{cs['status']} {len(cs['member_keys'])} "
      f"{cs['layers'][LAYER_NON_OSC]['common_window']}")

st3 = run_stage3(rep, PairingSymmetryConfig())
refit = st3['layers'][LAYER_NON_OSC]['refit']
lw = cs['layers'][LAYER_NON_OSC]['common_window']
check('A', '公共窗口重拟合：9 元素 fit_status=ok、窗口/点数/振幅字段齐全'
      '（层内窗口）',
      len(refit['elements']) == 9
      and all(e['fit_status'] == 'ok' and e['A'] is not None
              and e['n_points'] > 0 and e['window'] == lw
              for e in refit['elements'].values()),
      refit['elements']['49_50|A']['n_points'])
# 与直调 fit_envelope_logspace 逐位一致（同源，零漂移）
recs_a = rep['records_by_element']['49_50|A']
prep = SuperCorrDecayAnalyzer.group_by_distance(recs_a)
w = lw
m_w = (prep.r >= w[0]) & (prep.r <= w[1])
fit_direct = CorrelationDecayAnalyzer.fit_envelope_logspace(
    prep.r[m_w], np.abs(prep.C[m_w]), 'power', AnalyzerConfig())
check('A', '重拟合振幅与直调 fit_envelope_logspace 逐位一致',
      refit['elements']['49_50|A']['A'] == fit_direct.params['A']
      and refit['elements']['49_50|A']['exponent'] == fit_direct.params['K'])
# 矩阵真值：c·d^T → 一般模型应恢复方向、对称模型应被拒
gfit = st3['layers'][LAYER_NON_OSC]['general_rank1']
sfit = st3['layers'][LAYER_NON_OSC]['symmetric_rank1']
c_norm = np.array([c['A'], c['B'], c['C']]); c_norm /= np.linalg.norm(c_norm)
d_norm = np.array([d['A'], d['B'], d['C']]); d_norm /= np.linalg.norm(d_norm)
u, v = np.array(gfit['u']), np.array(gfit['v'])
angle_uv = float(np.arccos(np.clip(np.dot(u, v), -1, 1)))
check('A', '一般秩一模型恢复非对称结构（u≈c、v≈d 方向，angle(u,v)>0.1）',
      gfit['ok'] and gfit['identifiable']
      and np.linalg.norm(u - c_norm) < 0.05
      and np.linalg.norm(v - d_norm) < 0.05
      and angle_uv > 0.1,
      f'{angle_uv:.3f}')
check('A', '对称模型在非对称真值上残差显著劣化（u≠v 诊断生效）',
      sfit['ok'] and sfit['residual'] > 5 * gfit['residual'],
      f"{gfit['residual']:.3e} vs {sfit['residual']:.3e}")
cmp_a = st3['layers'][LAYER_NON_OSC]['model_comparison']
check('A', '模型比较：可比、ΔAICc 报告且注明仅参考统计量',
      cmp_a['comparable'] and cmp_a['delta_aicc'] is not None
      and any('参考统计量' in n for n in cmp_a['notes']))
check('A', '诊断：覆盖度/奇异值/双残差/PSD 全部输出；严格秩一矩阵条件数'
      '发散（None=数学退化，正确诊断而非缺失）',
      st3['layers'][LAYER_NON_OSC]['diagnostics']['n_observed'] == 9
      and st3['layers'][LAYER_NON_OSC]['diagnostics']['is_partial'] is False
      and st3['layers'][LAYER_NON_OSC]['diagnostics']['numerical_stability']
      ['singular_values'] is not None
      and 'condition_number' in st3['layers'][LAYER_NON_OSC]['diagnostics']
      ['numerical_stability']
      and st3['layers'][LAYER_NON_OSC]['diagnostics']['numerical_stability']
      ['cond_threshold'] == 'uncalibrated'
      and st3['layers'][LAYER_NON_OSC]['diagnostics']['fit_quality']
      ['general_residual'] is not None
      and st3['layers'][LAYER_NON_OSC]['diagnostics']['fit_quality']
      ['general_rel_residual'] is not None,
      f"cond={st3['layers'][LAYER_NON_OSC]['diagnostics']['numerical_stability']['condition_number']}")

# ======================================================================
# B. 两级模型纯函数
# ======================================================================
print('===== B. 两级模型纯函数 =====')
rng = np.random.default_rng(42)

# B1 对称真值恢复 + 模型比较
m_true = np.array([1.0, 0.8, 0.5])
M_sym = np.outer(m_true, m_true) + 0.01 * rng.standard_normal((3, 3))
g1 = fit_general_rank1(M_sym)
s1 = fit_symmetric_rank1(M_sym)
err_m = float(np.linalg.norm(s1['m'] - m_true) / np.linalg.norm(m_true))
check('B', '对称真值：对称模型恢复 m（相对误差 <5%）、非负约束满足',
      s1['ok'] and err_m < 0.05 and np.all(s1['m'] >= 0), f'{err_m:.4f}')
c1 = compare_rank1_models(g1, s1)
check('B', '对称真值：模型比较可比，对称模型 AICc 不劣于一般模型（参数'
      '更少）且只作参考统计量',
      c1['comparable'] and c1['delta_aicc'] <= 0
      and any('参考统计量' in n for n in c1['notes']),
      f"ΔAICc={c1['delta_aicc']:.2f}")

# B2 非对称秩一：u≈v 诊断与对称拒绝（直接矩阵级）
u_true = np.array([1.0, 0.8, 0.5]); u_true /= np.linalg.norm(u_true)
v_true = np.array([1.0, 0.5, 0.8]); v_true /= np.linalg.norm(v_true)
M_asym = 2.0 * np.outer(u_true, v_true)
g2 = fit_general_rank1(M_asym)
s2 = fit_symmetric_rank1(M_asym)
check('B', '非对称秩一：一般模型精确恢复、对称模型残差显著劣化',
      g2['ok'] and np.linalg.norm(np.array(g2['u']) - u_true) < 1e-6
      and s2['residual'] > 100 * g2['residual'])

# B3 近零分量
m_nz = np.array([0.05, 1.0, 1.0])
M_nz = np.outer(m_nz, m_nz) + 0.005 * rng.standard_normal((3, 3))
s3 = fit_symmetric_rank1(M_nz)
check('B', '近零分量：定性警告（噪声主导、比值不可靠），不输出误导比值',
      s3['ok'] and s3['near_zero_ratio'] < 0.1
      and any('相对误差' in n and '发散' in n for n in s3['degeneracy_notes']))

# B4 部分矩阵与不可辨识
mask_diag = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], bool)
g_diag = fit_general_rank1(M_sym, mask_diag)
check('B', '仅对角（3 元素 < 5 参数）：一般模型拒绝拟合（元素数不足）',
      not g_diag['ok'] and '参数' in g_diag['reason'])
mask_block = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], bool)
g_block = fit_general_rank1(M_sym, mask_block)
s_block = fit_symmetric_rank1(M_sym, mask_block)
check('B', '2×2 块（4 元素）：一般模型拒绝；对称模型可拟合但 Jacobian '
      '秩亏（m0 不可辨识）并记录',
      not g_block['ok'] and s_block['ok'] and s_block['jac_rank'] < 3
      and s_block['identifiable'] is False)
s_diag = fit_symmetric_rank1(M_sym, mask_diag)
check('B', '仅对角：对称模型非负约束下可辨识（m=√diag），但无交叉验证'
      '——报告本身即诊断（可辨识性含交叉项缺失说明）',
      s_diag['ok'] and s_diag['identifiable'] and s_diag['n_obs'] == 3)

# B5 NaN 掩码
M_nan = M_sym.copy(); M_nan[0, 1] = np.nan; M_nan[2, 2] = np.nan
g_nan = fit_general_rank1(M_nan)   # mask 默认 = isfinite
check('B', 'NaN 元素被掩码处理（n_obs 自动扣除，拟合可运行）',
      g_nan['ok'] and g_nan['n_obs'] == 7, f"{g_nan['n_obs']}")

# ======================================================================
# C. 失败与语义
# ======================================================================
print('===== C. 失败与语义 =====')

# C1 重拟合点数不足（构造最小 report：2 条记录 → insufficient）
recs_few = [SuperCorrRecord((49, 50), (57, 58), 3.0, 1e-4),
            SuperCorrRecord((49, 50), (65, 66), 6.0, 1e-5)]
rep_few = {'comparable_set': {'common_window': (3.0, 36.0),
                              'common_form': 'K',
                              'member_keys': ['49_50|A'],
                              'excluded': {}},
           'records_by_element': {'49_50|A': recs_few},
           'elements': {'49_50|A': {'refbond_class': 'A',
                                    'direction': 'A'}}}
rf = refit_amplitudes(rep_few, PairingSymmetryConfig())
check('C', '重拟合点数不足：进入 excluded 且 fit_status=insufficient'
      '（不强行拟合）',
      '49_50|A' in rf['excluded']
      and rf['elements']['49_50|A']['fit_status'] == 'insufficient'
      and '点数不足' in rf['elements']['49_50|A']['reason'])

# C2 振幅不含符号判据
check('C', '对称模型声明 m 符号无物理含义（振幅矩阵语义）',
      s1['m_sign_meaningless'] is True)
check('C', '阶段 3 输出含"无对称性结论"声明（振幅不含符号/相态判定）',
      'no_symmetry_conclusions' in st3
      and '符号' in st3['no_symmetry_conclusions'])

# C3 未校准阈值显式标记
check('C', '未校准阈值清单齐全（重拟合点数/秩一残差/条件数/近零/最小'
      '元素数/过零可靠性）',
      {'rank1_residual_threshold', 'condition_number_threshold',
       'refit_min_points', 'near_zero_frac', 'min_effective_entries',
       'zero_crossing_reliability_threshold'}
      <= {u[0] for u in st3['uncalibrated']}
      and st3['layers'][LAYER_NON_OSC]['diagnostics']['numerical_stability']
      ['cond_threshold'] == 'uncalibrated')

# C4 严格 JSON 序列化（run_stage3 输出；不含 records 引用）
js = json.dumps(st3, allow_nan=False, default=str)
back = json.loads(js)
check('C', '阶段 3 输出严格 JSON 可序列化且读回一致（无 NaN/Infinity 字面量，'
      '分层结构完整）',
      'NaN' not in js and 'Infinity' not in js
      and back['layers'][LAYER_NON_OSC]['model_comparison']['comparable']
      and back['layers'][LAYER_NON_OSC]['general_rank1']['ok'])

# C5 不确定度边界声明
check('C', '不确定度边界：联合协方差 unavailable、不假定元素独立、'
      '记录级 bootstrap pending',
      st3['uncertainty']['joint_covariance'] == 'unavailable'
      and st3['uncertainty']['element_independence_assumed'] is False
      and st3['uncertainty']['record_level_bootstrap'] == 'pending')

# ======================================================================
# M. 5B-1 分层矩阵：层内独立构建、跨层拒绝、osc 层偏差警告
# ======================================================================
print('===== M. 分层矩阵 =====')
g200 = HoneycombXCGeometry(200, 1.0)


def make_tree_fn(root, refbonds, dir_values):
    """dir_values: {(refbond_class, direction): callable(r) -> v}；
    a=d 通道原值、b=c 通道 0.8 倍。"""
    for rcls, rb in refbonds.items():
        ref_dir = os.path.join(root, f'super_corr_{rb[0]}_{rb[1]}')
        os.makedirs(ref_dir, exist_ok=True)
        for key in 'ABC':
            fn = dir_values.get((rcls, key))
            if fn is None:
                continue
            ents = []
            for m in range(1, 13):
                k, l = rb[0] + 8 * m, rb[1] + 8 * m
                r = g200.sup_corr_distance(rb[0], rb[1], k, l)
                ents.append([rb[0], rb[1], k, l, float(fn(r))])
            ents_b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents]
            for ch in 'ad':
                write_channel(ref_dir, key, ch, rb, ents)
            for ch in 'bc':
                write_channel(ref_dir, key, ch, rb, ents_b)


# M1 三层各自重算窗口和矩阵（d1_10 型：osc 8 + non_osc 1）
tmp_m1 = tempfile.mkdtemp(prefix='d55_m1_')
super_m1 = os.path.join(tmp_m1, 't_J_3', 'd_syn', '1500_measure_super_corr')
refs_m = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
vals_m = {}
for rc in refs_m:
    for key in 'ABC':
        if rc == 'C' and key == 'C':
            vals_m[(rc, key)] = (lambda r: 1e-4 * r ** (-2.0))   # weak、无过零
        else:
            vals_m[(rc, key)] = (lambda r: 1e-4 * r ** (-2.0)
                                 * np.cos(0.8 * r + np.pi / 2))  # strong 振荡
make_tree_fn(super_m1, refs_m, vals_m)
rep_m1 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_m1}, config=PairingSymmetryConfig())
cs_m1 = rep_m1['comparable_set']
st3_m1 = run_stage3(rep_m1, PairingSymmetryConfig())
check('M', '分层后各层独立构建矩阵：osc 层 8 元素、non_osc 层 1 元素',
      len(st3_m1['layers'][LAYER_OSC]['refit']['elements']) == 8
      and len(st3_m1['layers'][LAYER_NON_OSC]['refit']['elements']) == 1
      and st3_m1['layers'][LAYER_OSC]['refit_matrix'] is not None
      and st3_m1['layers'][LAYER_NON_OSC]['refit_matrix'] is not None,
      f"{len(st3_m1['layers'][LAYER_OSC]['refit']['elements'])}/"
      f"{len(st3_m1['layers'][LAYER_NON_OSC]['refit']['elements'])}")
osc_lw_m1 = cs_m1['layers'][LAYER_OSC]['common_window']
nono_lw_m1 = cs_m1['layers'][LAYER_NON_OSC]['common_window']
check('M', '各层独立重算公共窗口：层窗口存在，且层内重拟合窗口与层窗口'
      '一致（不复用旧窗口）',
      osc_lw_m1 is not None and nono_lw_m1 is not None
      and all(e['window'] == osc_lw_m1
              for e in st3_m1['layers'][LAYER_OSC]['refit']['elements']
              .values()),
      f'osc={osc_lw_m1} non_osc={nono_lw_m1}')
check('M', '跨层矩阵比较被拒绝（cross_layer_comparison.allowed=False + '
      '明确原因）',
      st3_m1['cross_layer_comparison']['allowed'] is False
      and '跨层' in st3_m1['cross_layer_comparison']['reason'])
check('M', 'osc 层部分矩阵：诊断记录缺失项（is_partial + n_observed）',
      st3_m1['layers'][LAYER_OSC]['diagnostics']['is_partial'] is True
      and st3_m1['layers'][LAYER_OSC]['diagnostics']['n_observed'] == 8)
warn_m1 = st3_m1['layers'][LAYER_OSC].get('amplitude_bias_warning')
check('M', 'osc 层附振幅偏差警告（含"非普适误差界"与"未被消除"声明）',
      warn_m1 is not None and '不是普适误差界' in warn_m1
      and '未被消除' in warn_m1)
check('M', '偏差警告不改变拟合数据与拟合值（纯元数据字段，refit 数值'
      '与警告独立）',
      'amplitude_bias_warning' not in st3_m1['layers'][LAYER_OSC]['refit']
      ['elements']['49_50|A']
      and st3_m1['layers'][LAYER_OSC]['refit']['elements']['49_50|A']['A']
      is not None)
check('M', 'non_osc 层不附振幅偏差警告（警告只属 osc 层）',
      'amplitude_bias_warning' not in st3_m1['layers'][LAYER_NON_OSC])

# M2 refit_amplitudes 层选择参数（layer='osc' 只重拟合指定层）
rf_osc_only = refit_amplitudes(rep_m1, PairingSymmetryConfig(),
                               layer=LAYER_OSC)
check('M', 'refit_amplitudes(layer=...) 只重拟合指定层',
      set(rf_osc_only['layers']) == {LAYER_OSC}
      and len(rf_osc_only['layers'][LAYER_OSC]['elements']) == 8)

# ======================================================================
# 汇总
# ======================================================================
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
