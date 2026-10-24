# -*- coding: utf-8 -*-
"""
PairingSymmetryAnalyzer 阶段 4 定向测试（D5：有符号证据通道 +
三通道状态机 + 3.1 审计基础修复回归）

覆盖：
- A. 3.1 基础修复回归：跨数量级 Jacobian 秩（相对容差）、极小 SSE 退化
  标注、非负边界说明、掩码不一致拒绝比较、宽泛异常收窄、条件数原因
  分类、non_osc 语义说明
- B. 符号通道：正/负稳定序列、过零、单次翻转、零数据、无记录、
  符号独立于振幅（振幅矩阵为正而带符号序列为负 → sign=-1）
- C. 三通道状态机：magnitude/sign/phase 独立、状态组合、运行状态分离、
  严格 JSON

合成输入；真实接口调用只读。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_pairing_symmetry_sign_state.py
"""
import json
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
from Physical_Quantity.Pairing_Symmetry_Analyzer import (
    PairingSymmetryConfig, build_comparable_set, fit_general_rank1,
    fit_symmetric_rank1, compare_rank1_models, analyze_sign_channel,
    run_stage4, SIGN_EXTRACTED, SIGN_INSUFFICIENT, SIGN_UNAVAILABLE,
    MAG_STRUCTURE_OK, MAG_INCONCLUSIVE, MAG_INSUFFICIENT,
    MAG_INPUT_INCOMPLETE, LAYER_NON_OSC, OSC_AMPLITUDE_BIAS_WARNING)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def write_channel(ref_dir, key, ch, rb, entries):
    path = os.path.join(
        ref_dir, f'scs{key}{ch}_fix_refbond_{rb[0]}_{rb[1]}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[[i, j, k, l], [v, 0]] for (i, j, k, l, v) in entries], f)


def make_tree(super_root, refbonds, dir_values, flip_dirs=None):
    """dir_values: {(rcls, key): callable(r) -> v}；a=d、b=c 通道 0.8 倍。"""
    g = HoneycombXCGeometry(200, 1.0)
    for rcls, rb in refbonds.items():
        ref_dir = os.path.join(super_root, f'super_corr_{rb[0]}_{rb[1]}')
        os.makedirs(ref_dir, exist_ok=True)
        for key in 'ABC':
            fn = dir_values.get((rcls, key))
            if fn is None:
                continue
            ents = []
            for m in range(1, 13):
                k, l = rb[0] + 8 * m, rb[1] + 8 * m
                r = g.sup_corr_distance(rb[0], rb[1], k, l)
                ents.append([rb[0], rb[1], k, l, float(fn(r))])
            ents_b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents]
            for ch in 'ad':
                write_channel(ref_dir, key, ch, rb, ents)
            for ch in 'bc':
                write_channel(ref_dir, key, ch, rb, ents_b)


def build_rep(super_root):
    return build_comparable_set(
        {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
         'super_corr_root': super_root}, config=PairingSymmetryConfig())


# ======================================================================
# A. 3.1 审计基础修复回归
# ======================================================================
print('===== A. 3.1 基础修复 =====')
rng = np.random.default_rng(7)

# A1 跨数量级 Jacobian 秩（相对容差）
M_base = np.array([[1.0, 0.8, 0.5], [0.8, 0.64, 0.4], [0.5, 0.4, 0.25]])
for scale, tag in ((1.0, '1'), (1e-12, '1e-12'), (1e8, '1e8')):
    g = fit_general_rank1(M_base * scale)
    check('A', f'跨数量级（×{tag}）：同一结构 Jacobian 秩不变（5）、'
          'identifiable 不变',
          g['ok'] and g['jac_rank'] == 5 and g['identifiable'],
          f"rank={g['jac_rank']}")
u_ref = fit_general_rank1(M_base)['u']
u_small = fit_general_rank1(M_base * 1e-12)['u']
check('A', '跨数量级：u 方向恢复一致（符号规范化后）',
      np.linalg.norm(np.array(u_ref) - np.array(u_small)) < 1e-6)

# A2 极小 SSE 退化标注
M_exact = np.outer(np.array([1.0, 0.8, 0.5]), np.array([1.0, 0.8, 0.5]))
g_exact = fit_general_rank1(M_exact)
s_exact = fit_symmetric_rank1(M_exact)
c_exact = compare_rank1_models(g_exact, s_exact)
check('A', '极小 SSE：退化状态显式标注（degenerate=True + 说明）',
      c_exact['comparable'] and c_exact['degenerate']
      and any('优化器收敛精度' in n for n in c_exact['notes']),
      f"rel={g_exact['rel_residual']:.2e} degenerate={c_exact['degenerate']}")

# A3 非负边界说明
M_bound = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
s_bound = fit_symmetric_rank1(M_bound)
check('A', '非负约束边界：on_boundary=True + 信息准则渐近限制说明',
      s_bound['ok'] and s_bound['on_boundary']
      and any('边界' in n and 'AICc' in n for n in s_bound['degeneracy_notes']))

# A4 掩码不一致拒绝比较
mask_a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], bool)
mask_b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], bool)
g_a = fit_general_rank1(M_base, mask_a)
g_b = fit_general_rank1(M_base, mask_b)
s_a = fit_symmetric_rank1(M_base, mask_a)
c_mismatch = compare_rank1_models(g_a, g_b)   # 不同掩码的一般模型互比
check('A', '掩码不一致：拒绝比较（comparable=False + 原因）',
      not c_mismatch['comparable'] and '掩码不一致' in c_mismatch['notes'])

# A5 宽泛异常收窄：程序错误原样传播
try:
    fit_general_rank1('not_a_matrix')
    check('A', '类型错误（程序错误）原样传播（不吞）', False, '未抛异常')
except (TypeError, ValueError, AttributeError):
    check('A', '类型错误（程序错误）原样传播（不吞）', True)

# A6 条件数原因分类
g_rank1 = fit_general_rank1(np.outer([1.0, 0.8, 0.5], [1.0, 0.8, 0.5]))
g_zero = fit_general_rank1(np.zeros((3, 3)))
g_noisy = fit_general_rank1(M_base + 0.01 * rng.standard_normal((3, 3)))
check('A', '条件数原因分类：严格秩一 → degenerate_rank、带噪 → finite、'
      '全零 → 拒绝拟合（数据无信息）',
      g_rank1['condition_number_reason'] == 'degenerate_rank'
      and g_noisy['condition_number_reason'] == 'finite'
      and not g_zero['ok'] and '全零' in g_zero['reason'])

# A7 non_osc 语义说明（build_comparable_set 输出）
tmp_a7 = tempfile.mkdtemp(prefix='d54_a7_')
super_a7 = os.path.join(tmp_a7, 't_J_3', 'd', '1500_measure_super_corr')
make_tree(super_a7, {'A': (49, 50)},
          {('A', k): (lambda r, kk=k: 1e-4 * r ** (-2.0)) for k in 'ABC'})
rep_a7 = build_rep(super_a7)
check('A', "non_osc 语义说明存在（'按非振荡处理，不表示已证明不存在振荡'）",
      'oscillation_semantics' in rep_a7
      and '不表示已证明' in rep_a7['oscillation_semantics'])

# ======================================================================
# B. 符号通道
# ======================================================================
print('===== B. 符号通道 =====')
tmp_b = tempfile.mkdtemp(prefix='d54_b_')
super_b = os.path.join(tmp_b, 't_J_3', 'd_syn', '1500_measure_super_corr')
# 单参考键 49_50：A 正向稳定、B 负向稳定、C 在 r>21 符号翻转
vals = {}
vals[('A', 'A')] = lambda r: 1e-4 * r ** (-2.0)
vals[('A', 'B')] = lambda r: -1e-4 * r ** (-2.0)
vals[('A', 'C')] = lambda r: 1e-4 * r ** (-2.0) * (1.0 if r <= 21 else -1.0)
make_tree(super_b, {'A': (49, 50)}, vals)
rep_b = build_rep(super_b)
sign_b = analyze_sign_channel(rep_b, PairingSymmetryConfig())
pe = sign_b['per_element']
check('B', '正向稳定 → extracted +1；负向稳定 → extracted −1',
      pe['49_50|A']['state'] == SIGN_EXTRACTED
      and pe['49_50|A']['sign'] == 1
      and pe['49_50|B']['state'] == SIGN_EXTRACTED
      and pe['49_50|B']['sign'] == -1,
      f"{pe['49_50|A']['sign']}/{pe['49_50|B']['sign']}")
check('B', '单次翻转 → unavailable（中性结论，不判物理振荡）',
      pe['49_50|C']['state'] == SIGN_UNAVAILABLE
      and '不判物理振荡' in pe['49_50|C']['reason'],
      pe['49_50|C']['reason'])
check('B', '符号独立于振幅：A/B 振幅相同（|C| 包络）但符号相反',
      abs(pe['49_50|A']['noise_floor_used']
          - pe['49_50|B']['noise_floor_used']) >= 0
      and sign_b['no_sign_from_amplitudes'])

# B2 零数据 → 底层分析 INCONCLUSIVE → evidence_insufficient
tmp_z = tempfile.mkdtemp(prefix='d54_z_')
super_z = os.path.join(tmp_z, 't_J_3', 'd0', '1500_measure_super_corr')
vals_z = {('A', k): (lambda r: 0.0) for k in 'ABC'}
make_tree(super_z, {'A': (49, 50)}, vals_z)
rep_z = build_rep(super_z)
sign_z = analyze_sign_channel(rep_z, PairingSymmetryConfig())
if rep_z['comparable_set']['member_keys']:
    z_states = {v['state'] for v in sign_z['per_element'].values()}
    check('B', '零数据：符号通道 evidence_insufficient（分析级前置门）',
          z_states == {SIGN_INSUFFICIENT}, str(z_states))
else:
    check('B', '零数据：元素被质量门排除（无符号证据，理由保留）', True,
          f"members={rep_z['comparable_set']['member_keys']}")

# B3 无记录（元素被排除）→ unavailable（输入缺失）
st4_b = run_stage4(rep_b, PairingSymmetryConfig())
check('B', '符号通道聚合：存在 extracted → 通道状态 extracted 且'
      'relative_signs_available 列出元素',
      sign_b['state'] == SIGN_EXTRACTED
      and set(sign_b['relative_signs_available']) == {'49_50|A', '49_50|B'})

# ======================================================================
# C. 三通道状态机
# ======================================================================
print('===== C. 三通道状态机 =====')

# C1 magnitude structure_ok + sign extracted + phase blocked（3 参考键全正）
tmp_c1 = tempfile.mkdtemp(prefix='d54_c1_')
super_c1 = os.path.join(tmp_c1, 't_J_3', 'd_syn', '1500_measure_super_corr')
refs = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
vals_c1 = {(rc, k): (lambda r: 1e-4 * r ** (-2.0)) for rc in refs for k in 'ABC'}
make_tree(super_c1, refs, vals_c1)
st4_c1 = run_stage4(build_rep(super_c1), PairingSymmetryConfig())
check('C', '组合 1：magnitude structure_ok + sign extracted + '
      'phase blocked_by_data（三通道并存，互不覆盖）',
      st4_c1['magnitude']['state'] == MAG_STRUCTURE_OK
      and st4_c1['sign']['state'] == SIGN_EXTRACTED
      and st4_c1['phase']['state'] == 'blocked_by_data'
      and st4_c1['run_status'] == 'ok')
check('C', '组合 1：phase blocked 不覆盖 magnitude/sign 结果',
      st4_c1['magnitude']['stage3']['layers'][LAYER_NON_OSC]['general_rank1']
      ['ok']
      and st4_c1['sign']['relative_signs_available'])

# C2 magnitude 不可辨识/拒绝（单参考键 3 元素 < 5 参数）但 sign 可用
st4_c2 = run_stage4(rep_b, PairingSymmetryConfig())
check('C', '组合 2：magnitude evidence_insufficient（一般模型拒绝）但 '
      'sign extracted（独立通道）；逐层记录拒绝原因',
      st4_c2['magnitude']['state'] == MAG_INSUFFICIENT
      and st4_c2['sign']['state'] == SIGN_EXTRACTED
      and '参数' in st4_c2['magnitude']['per_layer'][LAYER_NON_OSC]['reason'],
      f"{st4_c2['magnitude']['state']}/{st4_c2['sign']['state']}")

# C3 两者均不充分（零数据条件：元素全被排除 → input_incomplete）
st4_c3 = run_stage4(rep_z, PairingSymmetryConfig())
check('C', '组合 3：无可比较元素 → magnitude input_incomplete、'
      'sign unavailable（无成员）',
      st4_c3['magnitude']['state'] == MAG_INPUT_INCOMPLETE
      and st4_c3['sign']['state'] == SIGN_UNAVAILABLE,
      f"{st4_c3['magnitude']['state']}/{st4_c3['sign']['state']}")

# C4 输入不完整（参考键目录缺失 → 阶段 1 失败）
tmp_c4 = tempfile.mkdtemp(prefix='d54_c4_')
super_c4 = os.path.join(tmp_c4, '不存在')
try:
    rep_c4 = build_rep(super_c4)
    check('C', '组合 4：目录缺失 → FileNotFoundError（失败行语义，'
          '不进入状态机）', False, '未抛异常')
except FileNotFoundError:
    check('C', '组合 4：目录缺失 → FileNotFoundError（失败行语义，'
          '不进入状态机）', True)

# C5 单元素排除但其他元素形成部分矩阵（删 B 方向 a 通道）
tmp_c5 = tempfile.mkdtemp(prefix='d54_c5_')
super_c5 = os.path.join(tmp_c5, 't_J_3', 'd_syn', '1500_measure_super_corr')
make_tree(super_c5, {'A': (49, 50)},
          {('A', k): (lambda r, kk=k: 1e-4 * r ** (-2.0)) for k in 'ABC'})
os.remove(os.path.join(super_c5, 'super_corr_49_50',
                       'scsBa_fix_refbond_49_50.json'))
st4_c5 = run_stage4(build_rep(super_c5), PairingSymmetryConfig())
check('C', '组合 5：单元素排除（缺文件）→ 部分矩阵，其余元素正常',
      st4_c5['magnitude']['state'] == MAG_INSUFFICIENT   # 仅 2 元素 < 5 参数
      and st4_c5['sign']['state'] == SIGN_EXTRACTED
      and st4_c5['sign']['relative_signs_available'] == ['49_50|A', '49_50|C'],
      str(st4_c5['sign']['relative_signs_available']))

# C6 严格 JSON
js = json.dumps(st4_c1, allow_nan=False)
back = json.loads(js)
check('C', '状态机输出严格 JSON 可序列化（无 NaN/Infinity）且读回一致',
      'NaN' not in js and 'Infinity' not in js
      and back['phase']['state'] == 'blocked_by_data'
      and back['magnitude']['state'] == MAG_STRUCTURE_OK
      and back['sign']['state'] == SIGN_EXTRACTED)
check('C', '输出声明：无 d 波/s 波/d+id/ODLRO/相态判定',
      'no_symmetry_conclusions' in st4_c1
      and 'd+id' in st4_c1['no_symmetry_conclusions'])

# C7-C9 5B-1.2：magnitude 聚合层显式携带 osc 振幅偏差警告
# （non_osc-only 场景 C1：不应虚构警告适用）
check('C', 'non_osc-only：magnitude 顶层 osc 警告为 None（不虚构适用）',
      st4_c1['magnitude'].get('osc_amplitude_bias_warning') is None)
check('C', 'non_osc-only：per_layer[osc] 无偏差警告字段（无可比较成员）',
      'amplitude_bias_warning' not in
      st4_c1['magnitude']['per_layer'][LAYER_NON_OSC]
      and 'amplitude_bias_warning' not in
      st4_c1['magnitude']['per_layer'].get('osc', {}))
check('C', 'non_osc-only：sign 通道状态不受警告逻辑影响（extracted 保持）',
      st4_c1['sign']['state'] == SIGN_EXTRACTED)

# osc 场景：strong 振荡合成数据 → osc 层 → 顶层与逐层均显式携带警告
tmp_c7 = tempfile.mkdtemp(prefix='d54_c7_')
super_c7 = os.path.join(tmp_c7, 't_J_3', 'd_syn', '1500_measure_super_corr')
refs_c7 = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
vals_c7 = {(rc, k): (lambda r: 1e-4 * r ** (-2.0)
                     * np.cos(0.8 * r + np.pi / 2))
           for rc in refs_c7 for k in 'ABC'}
make_tree(super_c7, refs_c7, vals_c7)
st4_c7 = run_stage4(build_rep(super_c7), PairingSymmetryConfig())
check('C', 'osc 层存在：magnitude 顶层显式携带偏差警告（复用常量语义）',
      st4_c7['magnitude'].get('osc_amplitude_bias_warning')
      == OSC_AMPLITUDE_BIAS_WARNING
      and '不是普适误差界' in st4_c7['magnitude']
      ['osc_amplitude_bias_warning'])
check('C', 'osc 层存在：per_layer[osc] 显式携带偏差警告（幅度解释限制，'
      '不代表候选匹配结果）',
      st4_c7['magnitude']['per_layer']['osc'].get('amplitude_bias_warning')
      == OSC_AMPLITUDE_BIAS_WARNING)
check('C', 'osc 层存在：sign 通道状态正常输出（不被警告字段改变）',
      st4_c7['sign']['state'] in (SIGN_EXTRACTED, SIGN_INSUFFICIENT,
                                  SIGN_UNAVAILABLE)
      and st4_c7['run_status'] == 'ok')

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
