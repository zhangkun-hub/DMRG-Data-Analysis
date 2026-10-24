# -*- coding: utf-8 -*-
"""
PairingSymmetryAnalyzer 阶段 5B-2 定向测试：观测模式候选匹配器

覆盖（T1-T10 + malformed + 集成）：
- T1 理想 s 幅度模板 → observational、距离 s 最小、supported None
- T2 理想 d_real 幅度模板 + 符号证据缺失 → 不产生伙伴定向结论
- T3 中间幅度 (0.4,0.3,0.3) → 不强制选择最近候选
- T4 partial/秩亏矩阵 → 拟合前提阻断
- T5 sign 全 unavailable → 仅记录缺失，不推导符号结构
- T6 SU(2) fail → 阻断
- T7 未稳定层 → 阻断
- T8 osc 层 → 偏差警告阻断
- T9 近零分量 → 分量可靠性阻断
- T10 SU(2) uncalibrated → 记录未校准，不产生支持性结论
- malformed：缺字段/非有限 m/空成员 → 保守 inconclusive
- 集成：真实 build_comparable_set + run_candidate_observations

纯函数级输入为主（精确控制每场景）；集成用真实接口全链路。
所有场景断言 supported_candidate is None。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_pairing_symmetry_candidate.py
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

from Physical_Quantity.Pairing_Symmetry_Analyzer import (
    PairingSymmetryConfig, build_comparable_set,
    fit_symmetric_rank1, fit_general_rank1,
    candidate_observation_report, run_candidate_observations,
    OSC_AMPLITUDE_BIAS_WARNING, CAND_OBSERVATIONAL, CAND_INCONCLUSIVE,
    LAYER_OSC, LAYER_NON_OSC)
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


MEMBERS = ['k1', 'k2', 'k3']


def make_stage3_layer(M, n_obs=None):
    """用真实 fit_symmetric_rank1/fit_general_rank1 构造 stage3 层结果。"""
    sym = fit_symmetric_rank1(M)
    gen = fit_general_rank1(M)
    if n_obs is None:
        n_obs = int(np.sum(np.isfinite(M)))
    return {'symmetric_rank1': sym, 'general_rank1': gen,
            'diagnostics': {'n_observed': n_obs},
            'model_comparison': {'comparable': True}}


def make_layer_info(stable=True, bias_warning=None):
    li = {'member_keys': MEMBERS,
          'layer_convergence': {'stable': stable, 'rounds_used': 1,
                                'last_round_changes': [], 'note': ''},
          'excluded': {}}
    if bias_warning is not None:
        li['amplitude_bias_warning'] = bias_warning
    return li


def make_su2(gate):
    return {k: {'gate': gate, 'max_err': 0.05, 'err_ad': 0.0,
                'err_bc': 0.0, 'detail': ''} for k in MEMBERS}


def make_sign(layer_key, state, extracted=()):
    return {'per_layer': {layer_key: {'state': state, 'note': ''}},
            'per_element': {},
            'relative_signs_available': list(extracted)}


def run(M, layer_key=LAYER_NON_OSC, stable=True, bias=None, su2_gate='pass',
        sign_state='extracted', sign_extracted=('k1', 'k2', 'k3'),
        n_obs=None):
    return candidate_observation_report(
        make_layer_info(stable, bias), make_stage3_layer(M, n_obs),
        make_su2(su2_gate), make_sign(layer_key, sign_state, sign_extracted),
        layer_key)


def sym_matrix(m):
    return np.outer(m, m)


# ======================================================================
# T1-T10 + malformed（纯函数级）
# ======================================================================
print('===== T 系列 =====')

# T1 理想 s
r1 = run(sym_matrix(np.array([1.0, 1.0, 1.0])))
check('T1', '理想 s：前提全过 → observational、supported None、'
      '距离 s 最小（<0.01）',
      r1['status'] == CAND_OBSERVATIONAL and r1['supported_candidate'] is None
      and r1['blocking_reasons'] == []
      and r1['observed_anisotropy']['candidate_distances']['s'] < 0.01
      and r1['observed_anisotropy']['component_order'] == ['A', 'B', 'C'],
      f"{r1['status']} {r1['observed_anisotropy']['candidate_distances']}")
check('T1', '理想 s：前提逐项 pass（fit/m_validity/convergence/su2/'
      'amplitude_bias/component_reliability）',
      all(r1['prerequisite_checks'][k] == 'pass' for k in
          ('fit', 'm_validity', 'convergence', 'su2',
           'component_reliability'))
      and r1['prerequisite_checks']['amplitude_bias'] == 'none',
      str(r1['prerequisite_checks']))

# T2 理想 d_real（2:1:1） + 符号证据缺失
r2 = run(sym_matrix(np.array([2.0, 1.0, 1.0])), sign_extracted=())
check('T2', '理想 d_real：距离 d_real 最小（<0.01）、supported None',
      r2['observed_anisotropy']['candidate_distances']['d_real'] < 0.01
      and r2['observed_anisotropy']['candidate_distances']['d_real']
      < r2['observed_anisotropy']['candidate_distances']['s']
      and r2['supported_candidate'] is None,
      f"{r2['observed_anisotropy']['candidate_distances']}")
check('T2', '符号证据缺失：不产生伙伴定向结论（extracted 空 + 跨元素'
      '符号结构 not_implemented）',
      r2['sign_evidence']['extracted_elements'] == []
      and r2['sign_evidence']['cross_element_sign_structure']
      == 'not_implemented'
      and '不产生伙伴定向' in r2['sign_evidence']['note'])

# T3 中间幅度 (0.4, 0.3, 0.3)：不得强制选择最近候选
r3 = run(sym_matrix(np.array([0.4, 0.3, 0.3])))
d3 = r3['observed_anisotropy']['candidate_distances']
check('T3', '中间幅度：距离非零、不强制选择最近候选（无候选结论字段，'
      'supported 恒 None）',
      d3['s'] > 0.01 and d3['d_real'] > 0.01
      and r3['supported_candidate'] is None
      and 'supported_candidate' in r3 and r3['supported_candidate'] is None,
      f"{d3}")

# T4 partial/秩亏矩阵（2×2 块 + NaN 掩码 → Jacobian 秩亏，不可辨识）
M4 = np.array([[1.0, 0.8, np.nan], [0.8, 0.64, np.nan],
               [np.nan, np.nan, np.nan]])
r4 = run(M4)
check('T4', '秩亏矩阵：拟合前提阻断（identifiable=False → inconclusive '
      '+ 原因含不可辨识）',
      r4['status'] == CAND_INCONCLUSIVE and r4['supported_candidate'] is None
      and any('不可辨识' in b for b in r4['blocking_reasons'])
      and r4['prerequisite_checks']['fit'] == 'blocked',
      str(r4['blocking_reasons']))

# T5 sign 全 unavailable
r5 = run(sym_matrix(np.array([1.0, 1.0, 1.0])), sign_state='unavailable',
         sign_extracted=())
check('T5', 'sign 全 unavailable：仅记录缺失状态，不推导符号结构；'
      '不阻断幅度观测（status observational）',
      r5['status'] == CAND_OBSERVATIONAL
      and r5['sign_evidence']['per_layer_state'] == 'unavailable'
      and r5['sign_evidence']['extracted_elements'] == []
      and r5['sign_evidence']['cross_element_sign_structure']
      == 'not_implemented',
      str(r5['sign_evidence']))

# T6 SU(2) fail
r6 = run(sym_matrix(np.array([1.0, 1.0, 1.0])), su2_gate='fail')
check('T6', 'SU(2) fail：阻断（checks.su2=fail + 原因含 SU(2)）',
      r6['status'] == CAND_INCONCLUSIVE
      and r6['prerequisite_checks']['su2'] == 'fail'
      and any('SU(2)' in b for b in r6['blocking_reasons']),
      str(r6['blocking_reasons']))

# T7 未稳定层
r7 = run(sym_matrix(np.array([1.0, 1.0, 1.0])), stable=False)
check('T7', '未稳定层：收敛前提阻断（原因含未稳定）',
      r7['status'] == CAND_INCONCLUSIVE
      and r7['prerequisite_checks']['convergence'] == 'blocked'
      and any('未稳定' in b for b in r7['blocking_reasons']),
      str(r7['blocking_reasons']))

# T8 osc 层（偏差警告）
r8 = run(sym_matrix(np.array([1.0, 1.0, 1.0])), layer_key=LAYER_OSC,
         bias=OSC_AMPLITUDE_BIAS_WARNING)
check('T8', 'osc 层：偏差警告阻断（checks.amplitude_bias=warning + '
      '原因含偏差未校正）且警告出现在 quality_warnings',
      r8['status'] == CAND_INCONCLUSIVE
      and r8['prerequisite_checks']['amplitude_bias'] == 'warning'
      and any('偏差未校正' in b for b in r8['blocking_reasons'])
      and OSC_AMPLITUDE_BIAS_WARNING in r8['quality_warnings'],
      str(r8['blocking_reasons']))
check('T8', 'osc 层：不声称已完成 q/phi/窗口/噪声校正（警告原文保留）',
      any('未被消除' in w for w in r8['quality_warnings']))

# T9 近零分量
r9 = run(sym_matrix(np.array([0.05, 1.0, 1.0])))
check('T9', '近零分量：分量可靠性阻断（原因含近零/发散，不解释为精确零）',
      r9['status'] == CAND_INCONCLUSIVE
      and r9['prerequisite_checks']['component_reliability'] == 'blocked'
      and any('近零' in b and '发散' in b for b in r9['blocking_reasons'])
      and any('不得' in b and '精确零' in b for b in r9['blocking_reasons']),
      str(r9['blocking_reasons']))

# T10 SU(2) uncalibrated
r10 = run(sym_matrix(np.array([1.0, 1.0, 1.0])), su2_gate='uncalibrated')
check('T10', 'SU(2) uncalibrated：记录未校准 + 警告，不产生支持性结论'
      '（supported 恒 None）',
      r10['prerequisite_checks']['su2'] == 'uncalibrated'
      and any('未校准' in w for w in r10['quality_warnings'])
      and r10['supported_candidate'] is None,
      str(r10['quality_warnings']))

# Malformed：m 非有限 / 缺 stage3 层 / 空成员
r_m1 = run(sym_matrix(np.array([1.0, 1.0, 1.0])))
r_m1['observed_anisotropy'] = None   # 不改变——用下面独立构造
bad_layer = make_stage3_layer(sym_matrix(np.array([1.0, 1.0, 1.0])))
bad_layer['symmetric_rank1']['m'] = [np.nan, 1.0, 1.0]
r_m2 = candidate_observation_report(
    make_layer_info(), bad_layer, make_su2('pass'),
    make_sign(LAYER_NON_OSC, 'extracted'), LAYER_NON_OSC)
check('M', '非有限 m：保守 inconclusive + 原因（不伪造证据）',
      r_m2['status'] == CAND_INCONCLUSIVE
      and any('非有限' in b for b in r_m2['blocking_reasons'])
      and r_m2['observed_anisotropy'] is None,
      str(r_m2['blocking_reasons']))
r_m3 = candidate_observation_report(
    make_layer_info(), None, make_su2('pass'),
    make_sign(LAYER_NON_OSC, 'extracted'), LAYER_NON_OSC)
check('M', '缺 stage3 层报告：保守 inconclusive + 原因（字段缺失不崩溃）',
      r_m3['status'] == CAND_INCONCLUSIVE
      and any('缺失' in b for b in r_m3['blocking_reasons']))
r_m4 = candidate_observation_report(
    {'member_keys': []}, None, {}, None, LAYER_NON_OSC)
check('M', '空成员层：保守 inconclusive + 原因（无可比较元素）',
      r_m4['status'] == CAND_INCONCLUSIVE
      and any('无可比较元素' in b for b in r_m4['blocking_reasons'])
      and r_m4['supported_candidate'] is None)

# ======================================================================
# 5B-2 语义审计修复回归：组件可靠性字段缺失（A1-A3）+ 旧式 report（B）
# ======================================================================
print('===== 修复回归 =====')
M_ok = np.outer(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))


def stage_with_fields_dropped(*names):
    sym = fit_symmetric_rank1(M_ok)
    for n in names:
        sym.pop(n, None)
    return {'symmetric_rank1': sym,
            'diagnostics': {'n_observed': 9}}


# A1 两字段同时缺失
r_a1 = candidate_observation_report(
    make_layer_info(), stage_with_fields_dropped(
        'near_zero_ratio', 'on_boundary'),
    make_su2('pass'), make_sign(LAYER_NON_OSC, 'extracted'), LAYER_NON_OSC)
check('A1', 'near_zero_ratio 与 on_boundary 同时缺失：component_reliability '
      '不可用 + inconclusive + 阻断原因含缺失字段名 + supported None',
      r_a1['prerequisite_checks']['component_reliability'] == 'unavailable'
      and r_a1['status'] == CAND_INCONCLUSIVE
      and any('缺失' in b and 'near_zero_ratio' in b and 'on_boundary' in b
              for b in r_a1['blocking_reasons'])
      and r_a1['supported_candidate'] is None,
      f"{r_a1['prerequisite_checks']['component_reliability']} "
      f"{r_a1['blocking_reasons']}")

# A2 仅 near_zero_ratio 缺失
r_a2 = candidate_observation_report(
    make_layer_info(), stage_with_fields_dropped('near_zero_ratio'),
    make_su2('pass'), make_sign(LAYER_NON_OSC, 'extracted'), LAYER_NON_OSC)
check('A2', '仅 near_zero_ratio 缺失：不能通过组件可靠性门（inconclusive '
      '+ 原因指明缺失字段）',
      r_a2['prerequisite_checks']['component_reliability'] == 'unavailable'
      and r_a2['status'] == CAND_INCONCLUSIVE
      and any('near_zero_ratio' in b for b in r_a2['blocking_reasons'])
      and r_a2['supported_candidate'] is None,
      f"{r_a2['prerequisite_checks']['component_reliability']}")

# A3 仅 on_boundary 缺失
r_a3 = candidate_observation_report(
    make_layer_info(), stage_with_fields_dropped('on_boundary'),
    make_su2('pass'), make_sign(LAYER_NON_OSC, 'extracted'), LAYER_NON_OSC)
check('A3', '仅 on_boundary 缺失：不能通过组件可靠性门（inconclusive '
      '+ 原因指明缺失字段）',
      r_a3['prerequisite_checks']['component_reliability'] == 'unavailable'
      and r_a3['status'] == CAND_INCONCLUSIVE
      and any('on_boundary' in b for b in r_a3['blocking_reasons'])
      and r_a3['supported_candidate'] is None,
      f"{r_a3['prerequisite_checks']['component_reliability']}")

# B 旧式 report 入口路径
rep_legacy = {'comparable_set': {'member_keys': ['x'],
                                 'common_window': (3.0, 36.0),
                                 'common_form': 'K', 'excluded': {}},
              'records_by_element': {}, 'elements': {}, 'su2': {}}
r_b = run_candidate_observations(rep_legacy, PairingSymmetryConfig())
check('B', '旧式 report：明确不可用 + supported_candidate 键存在且为 None '
      '+ 不伪造分层观测',
      r_b.get('status') == 'unavailable'
      and 'supported_candidate' in r_b
      and r_b['supported_candidate'] is None
      and 'layers' not in r_b
      and 'reason' in r_b,
      f"{sorted(r_b.keys())}")

# ======================================================================
# 集成：真实 build_comparable_set + run_candidate_observations
# ======================================================================
print('===== 集成 =====')
g200 = HoneycombXCGeometry(200, 1.0)


def write_channel(ref_dir, key, ch, rb, entries):
    path = os.path.join(
        ref_dir, f'scs{key}{ch}_fix_refbond_{rb[0]}_{rb[1]}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[[i, j, k, l], [v, 0]] for (i, j, k, l, v) in entries], f)


def make_tree_fn(root, refbonds, dir_values):
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


# I1 non_osc 全链路（纯幂律合成树 → non_osc 层观测报告）
tmp_i1 = tempfile.mkdtemp(prefix='d55b2_i1_')
super_i1 = os.path.join(tmp_i1, 't_J_3', 'd_syn', '1500_measure_super_corr')
refs_i = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
vals_i = {(rc, k): (lambda r: 1e-4 * r ** (-2.0)) for rc in refs_i
          for k in 'ABC'}
make_tree_fn(super_i1, refs_i, vals_i)
rep_i1 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_i1}, config=PairingSymmetryConfig())
obs_i1 = run_candidate_observations(rep_i1, PairingSymmetryConfig())
nono_i1 = obs_i1['layers'][LAYER_NON_OSC]
check('I', '集成 non_osc：观测报告存在、supported None、观测向量分量序'
      ' A/B/C、每前提检查有值',
      nono_i1['supported_candidate'] is None
      and nono_i1['observed_anisotropy'] is not None
      and nono_i1['observed_anisotropy']['component_order'] == ['A', 'B', 'C']
      and nono_i1['status'] in (CAND_OBSERVATIONAL, CAND_INCONCLUSIVE)
      and all(k in nono_i1['prerequisite_checks'] for k in
              ('fit', 'convergence', 'su2', 'amplitude_bias',
               'component_reliability')),
      f"{nono_i1['status']} {nono_i1['blocking_reasons'][:1]}")
check('I', '集成：osc 空层报告保守（无可比较元素）且 supported None',
      obs_i1['layers'][LAYER_OSC]['status'] == CAND_INCONCLUSIVE
      and any('无可比较元素' in b
              for b in obs_i1['layers'][LAYER_OSC]['blocking_reasons'])
      and obs_i1['layers'][LAYER_OSC]['supported_candidate'] is None)

# I2 osc 全链路（strong 振荡树 → osc 层偏差阻断）
tmp_i2 = tempfile.mkdtemp(prefix='d55b2_i2_')
super_i2 = os.path.join(tmp_i2, 't_J_3', 'd_syn', '1500_measure_super_corr')
vals_i2 = {(rc, k): (lambda r: 1e-4 * r ** (-2.0)
                     * np.cos(0.8 * r + np.pi / 2)) for rc in refs_i
           for k in 'ABC'}
make_tree_fn(super_i2, refs_i, vals_i2)
rep_i2 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_i2}, config=PairingSymmetryConfig())
obs_i2 = run_candidate_observations(rep_i2, PairingSymmetryConfig())
osc_i2 = obs_i2['layers'][LAYER_OSC]
check('I', '集成 osc：偏差警告进入阻断与 quality_warnings、supported '
      'None、无物理结论声明',
      osc_i2['supported_candidate'] is None
      and osc_i2['prerequisite_checks']['amplitude_bias'] == 'warning'
      and any('偏差未校正' in b for b in osc_i2['blocking_reasons'])
      and OSC_AMPLITUDE_BIAS_WARNING in osc_i2['quality_warnings']
      and 'no_physical_conclusion' in osc_i2
      and 'd+id' in osc_i2['no_physical_conclusion'],
      str(osc_i2['blocking_reasons'][:1]))
check('I', '集成：入口级声明（supported_candidate_semantics + '
      'no_physical_conclusion）与严格 JSON 可序列化',
      'no_physical_conclusion' in obs_i2
      and 'supported_candidate_semantics' in obs_i2
      and json.dumps(obs_i2, allow_nan=False, default=str))

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
