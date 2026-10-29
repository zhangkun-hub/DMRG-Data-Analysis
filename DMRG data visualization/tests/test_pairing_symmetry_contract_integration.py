# -*- coding: utf-8 -*-
"""
配对对称性 → 统一证据链 契约集成测试（P1-E.2）。

覆盖：
- sc_pairing_symmetry_report 的成功/失败/运行状态映射
- 三通道保真（payload 引用不复制；缺失通道不伪造）
- phase=blocked_by_data 映射为 capability_limitation（不是 d+id/TRSB
  反证）
- supported_candidate_status=not_evaluated 不升级为物理支持
- quality_warnings 展平保留；data_restricted=None 保留未知
- 完整链路：原生 dict → AnalysisReport → collect_evidence →
  build_evidence_items（categories 正确）

全部合成输入，不注册真实物理判据，不改变 AUTHORIZED_RULES。

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_pairing_symmetry_contract_integration.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, STATUS_FAILED, STATUS_INCONCLUSIVE)
from Physical_Quantity.Analysis_Integration import sc_pairing_symmetry_report
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, build_evidence_items,
    EVIDENCE_OBSERVATION, EVIDENCE_QUALITY_DIAGNOSTIC,
    EVIDENCE_CAPABILITY_LIMITATION, EVIDENCE_NOT_AVAILABLE)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def make_report_dict(include_sign=True, include_phase=True,
                     run_status='ok', supported_status='not_evaluated',
                     warnings=('合成质量警告',)):
    d = {'run_status': run_status,
         'magnitude': {'state': 'structure_ok', 'reason': '合成幅度结构'},
         'supported_candidate_status': supported_status}
    if include_sign:
        d['sign'] = {'state': 'extracted',
                     'per_element': {'k1': {'state': 'extracted',
                                            'sign': 1}}}
    if include_phase:
        d['phase'] = {'state': 'blocked_by_data',
                      'reason': '实值关联数据原则上不可恢复完整相对复相位'}
    d['layers'] = {'non_osc': {'quality_warnings': list(warnings),
                               'sign_evidence': {
                                   'cross_element_sign_structure':
                                       'not_implemented'}}}
    return d


# ======================================================================
# A. 报告适配器
# ======================================================================
print('===== A. 报告适配器 =====')
d_ok = make_report_dict()
rep = sc_pairing_symmetry_report(d_ok)
check('A', '成功报告：status=ok、analyzer/observable 正确',
      rep.status == STATUS_OK
      and rep.analyzer == 'PairingSymmetryAnalyzer'
      and rep.observable == 'sc_pairing_symmetry')
check('A', '三通道保真：payload 为原生 dict 引用（不复制），三通道键齐全',
      rep.payload is d_ok
      and 'magnitude' in rep.payload and 'sign' in rep.payload
      and 'phase' in rep.payload)
check('A', 'metadata 通道状态摘要：magnitude/sign/phase 状态齐全',
      rep.metadata['magnitude_state'] == 'structure_ok'
      and rep.metadata['sign_state'] == 'extracted'
      and rep.metadata['phase_state'] == 'blocked_by_data',
      str(rep.metadata))
check('A', 'supported_candidate_status=not_evaluated 原样保留（不升级）',
      rep.metadata['supported_candidate_status'] == 'not_evaluated'
      and rep.classification is None and rep.evidence is None)
check('A', 'quality_warnings 展平保留',
      '合成质量警告' in rep.warnings)
check('A', 'data_restricted=None 保留未知语义',
      rep.data_restricted is None)

# 缺失通道：不伪造
d_nosign = make_report_dict(include_sign=False)
rep_ns = sc_pairing_symmetry_report(d_nosign)
check('A', '缺失通道不伪造：sign_state 键不存在（metadata 不编造）',
      'sign_state' not in rep_ns.metadata
      and 'sign' not in rep_ns.payload,
      str(sorted(rep_ns.metadata.keys())))

# 失败报告
rep_none = sc_pairing_symmetry_report(None)
check('A', 'None 输入：status=failed（沿用契约失败约定，不伪造观测）',
      rep_none.status == STATUS_FAILED and rep_none.payload is None)

# run_status 异常值：保守 inconclusive
rep_bad = sc_pairing_symmetry_report(make_report_dict(run_status='weird'))
check('A', 'run_status 非 ok：保守 STATUS_INCONCLUSIVE（不冒充成功）',
      rep_bad.status == STATUS_INCONCLUSIVE)

# ======================================================================
# B. 证据映射（完整链路）
# ======================================================================
print('===== B. 证据映射 =====')
ph = collect_evidence([rep], labels=['pair_sym_0'])
items = build_evidence_items(ph)
it = items[0]
check('B', '完整链路：原生 dict → AnalysisReport → build_evidence_items'
      '（entry_index 关联成立）',
      len(items) == 1 and it['entry_index'] == 0
      and it['analyzer'] == 'PairingSymmetryAnalyzer'
      and it['payload_available'] is True)
check('B', 'payload 可用 → observation',
      EVIDENCE_OBSERVATION in it['categories'], str(it['categories']))
check('B', 'phase=blocked_by_data → capability_limitation + '
      'capability.phase_capability（数据能力限制，非反证）',
      EVIDENCE_CAPABILITY_LIMITATION in it['categories']
      and it['capability']['phase_capability'] == 'blocked_by_data',
      str(it['categories']))
check('B', '质量警告 → quality_diagnostic',
      EVIDENCE_QUALITY_DIAGNOSTIC in it['categories']
      and '合成质量警告' in it['warnings'])
check('B', '不产生 order_evidence/counter_evidence（phase 限制不是反证）',
      'order_evidence' not in it['categories']
      and 'counter_evidence' not in it['categories'])

# 失败报告映射
ph2 = collect_evidence([rep_none], labels=['bad'])
items2 = build_evidence_items(ph2)
check('B', '失败报告 → not_available（不伪造观测）',
      items2[0]['categories'] == [EVIDENCE_NOT_AVAILABLE],
      str(items2[0]['categories']))

# ======================================================================
# C/D/E. P1-E.3.1：两种真实输出结构适配（按 P1-E.3 实测结构构造）
# ======================================================================
print('===== C/D/E. 真实结构适配 =====')


def make_stage4_real():
    """run_stage4 真实输出结构（P1-E.3 N400 实测形态）。"""
    return {
        'magnitude': {'state': 'structure_ok', 'reason': '合成',
                      'per_layer': {'osc': {'state': 'structure_ok'}}},
        'sign': {'state': 'unavailable', 'per_element': {},
                 'per_layer': {}},
        'phase': {'state': 'blocked_by_data',
                  'reason': '实值关联数据原则上不可恢复完整相对复相位'},
        'run_status': 'ok',
        'state_semantics': {}, 'no_symmetry_conclusions': '...',
    }


def make_observations_real():
    """run_candidate_observations 真实输出结构（P1-E.3 N200 实测形态）：
    无三通道顶层字段、无 phase 通道。"""
    return {
        'no_physical_conclusion': '...',
        'supported_candidate_semantics': '...',
        'supported_candidate_status': 'not_evaluated',
        'layers': {
            'osc': {
                'layer': 'osc', 'status': 'inconclusive',
                'observed_anisotropy': None,
                'supported_candidate': None,
                'supported_candidate_status': 'not_evaluated',
                'prerequisite_checks': {},
                'blocking_reasons': ['振荡振幅偏差未校正（合成）'],
                'quality_warnings': ['合成质量警告 O'],
                'sign_evidence': {
                    'cross_element_sign_structure': 'not_implemented'},
                'no_physical_conclusion': '...'},
            'non_osc': {
                'layer': 'non_osc', 'status': 'inconclusive',
                'supported_candidate': None,
                'supported_candidate_status': 'not_evaluated',
                'blocking_reasons': ['该层无可比较元素'],
                'quality_warnings': [],
                'sign_evidence': {
                    'cross_element_sign_structure': 'not_implemented'}},
        },
    }


# C：run_stage4 结构单独输入
rep_s4 = sc_pairing_symmetry_report(make_stage4_real())
check('C', 'stage4 结构：三通道状态完整映射（magnitude/sign/phase）',
      rep_s4.metadata['magnitude_state'] == 'structure_ok'
      and rep_s4.metadata['sign_state'] == 'unavailable'
      and rep_s4.metadata['phase_state'] == 'blocked_by_data',
      str(rep_s4.metadata))
items_s4 = build_evidence_items(
    collect_evidence([rep_s4], labels=['stage4']))
check('C', 'stage4 结构：phase=blocked_by_data → capability_limitation',
      EVIDENCE_CAPABILITY_LIMITATION in items_s4[0]['categories']
      and items_s4[0]['capability']['phase_capability'] == 'blocked_by_data',
      str(items_s4[0]['categories']))

# D：run_candidate_observations 结构单独输入
rep_obs = sc_pairing_symmetry_report(make_observations_real())
check('D', 'observations 结构：层摘要提取（layer_states 含 osc/non_osc '
      'status 与阻断数），无 phase 不编造',
      'layer_states' in rep_obs.metadata
      and rep_obs.metadata['layer_states']['osc']['n_blocking'] == 1
      and rep_obs.metadata['layer_states']['non_osc']['status']
      == 'inconclusive'
      and 'phase_state' not in rep_obs.metadata
      and 'magnitude_state' not in rep_obs.metadata,
      str(rep_obs.metadata))
items_obs = build_evidence_items(
    collect_evidence([rep_obs], labels=['obs']))
check('D', 'observations 结构：无 phase 键 → 不产生虚假 capability_limitation',
      EVIDENCE_CAPABILITY_LIMITATION not in items_obs[0]['categories']
      and items_obs[0]['capability']['phase_capability'] is None,
      str(items_obs[0]['categories']))
check('D', 'observations 结构：quality_warnings 与 supported_candidate_'
      'status=not_evaluated 保真（不升级）',
      '合成质量警告 O' in rep_obs.warnings
      and rep_obs.metadata['supported_candidate_status'] == 'not_evaluated'
      and rep_obs.classification is None and rep_obs.evidence is None)

# E：复合输入（完整证据链推荐形式）
comp = {'stage4': make_stage4_real(),
        'observations': make_observations_real()}
rep_comp = sc_pairing_symmetry_report(comp)
check('E', '复合输入：三通道状态（stage4）+ 层摘要与警告（observations）'
      '同时保真',
      rep_comp.metadata['phase_state'] == 'blocked_by_data'
      and rep_comp.metadata['magnitude_state'] == 'structure_ok'
      and 'layer_states' in rep_comp.metadata
      and '合成质量警告 O' in rep_comp.warnings
      and rep_comp.payload is comp,
      str(rep_comp.metadata))
items_comp = build_evidence_items(
    collect_evidence([rep_comp], labels=['comp']))
check('E', '复合输入：capability_limitation 进入统一证据链 '
      '（N200 缺失能力限制问题经复合输入修复）',
      EVIDENCE_CAPABILITY_LIMITATION in items_comp[0]['categories']
      and items_comp[0]['capability']['phase_capability'] == 'blocked_by_data'
      and EVIDENCE_OBSERVATION in items_comp[0]['categories'],
      str(items_comp[0]['categories']))

# ======================================================================
# F. P1-E.4：标准入口（run_analysis 路由 + 同源构造性保证）
# ======================================================================
print('===== F. 标准入口 =====')
import json
import tempfile

from Physical_Quantity.Analysis_Contract import (
    OBSERVABLE_SC_PAIRING_SYMMETRY)
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.Pairing_Symmetry_Analyzer import PairingSymmetryConfig
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry

g200 = HoneycombXCGeometry(200, 1.0)


def write_channel(ref_dir, key, ch, rb, entries):
    path = os.path.join(
        ref_dir, f'scs{key}{ch}_fix_refbond_{rb[0]}_{rb[1]}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[[i, j, k, l], [v, 0]] for (i, j, k, l, v) in entries], f)


def make_synthetic_tree(root):
    refs = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
    for rcls, rb in refs.items():
        ref_dir = os.path.join(root, f'super_corr_{rb[0]}_{rb[1]}')
        os.makedirs(ref_dir, exist_ok=True)
        for key in 'ABC':
            ents = []
            for m in range(1, 13):
                k, l = rb[0] + 8 * m, rb[1] + 8 * m
                r = g200.sup_corr_distance(rb[0], rb[1], k, l)
                ents.append([rb[0], rb[1], k, l, float(1e-4 * r ** (-2.0))])
            ents_b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents]
            for ch in 'ad':
                write_channel(ref_dir, key, ch, rb, ents)
            for ch in 'bc':
                write_channel(ref_dir, key, ch, rb, ents_b)


tmp_f = tempfile.mkdtemp(prefix='p1e4_')
super_f = os.path.join(tmp_f, 't_J_3', 'd_syn', '1500_measure_super_corr')
make_synthetic_tree(super_f)
cond_f = {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
          'super_corr_root': super_f}

# F1 标准入口成功路径（operation 默认 'analyze'）
rep_f = run_analysis(OBSERVABLE_SC_PAIRING_SYMMETRY, condition=cond_f,
                     config=PairingSymmetryConfig())
check('F', '标准入口：返回 AnalysisReport、status=ok、契约常量正确',
      rep_f.status == STATUS_OK
      and rep_f.analyzer == 'PairingSymmetryAnalyzer'
      and rep_f.observable == OBSERVABLE_SC_PAIRING_SYMMETRY)
check('F', '标准入口：三通道状态齐全（stage4 部分）',
      rep_f.metadata.get('magnitude_state') is not None
      and rep_f.metadata.get('sign_state') is not None
      and rep_f.metadata.get('phase_state') == 'blocked_by_data',
      str(rep_f.metadata))
check('F', '标准入口：supported_candidate_status=not_evaluated（不升级）',
      rep_f.metadata.get('supported_candidate_status') == 'not_evaluated'
      and rep_f.classification is None and rep_f.evidence is None)
items_f = build_evidence_items(collect_evidence([rep_f], labels=['entry']))
check('F', '标准入口：blocked_by_data → capability_limitation 进入证据链',
      EVIDENCE_CAPABILITY_LIMITATION in items_f[0]['categories']
      and items_f[0]['capability']['phase_capability'] == 'blocked_by_data',
      str(items_f[0]['categories']))
# 同源构造性保证：stage4 与 observations 层键一致（同一 rep 对象生成，
# 层键 = magnitude.per_layer 与 sign.per_layer 的并集）
mag_layers_f = set(
    (rep_f.payload['stage4']['magnitude'].get('per_layer') or {}).keys())
sign_layers_f = set(
    ((rep_f.payload['stage4'].get('sign') or {}).get('per_layer')
     or {}).keys())
obs_layers_f = set(
    ((rep_f.payload['observations'] or {}).get('layers') or {}).keys())
check('F', '同源保证（结构指示）：observations 层键 ⊆ stage4 层键并集'
      '（obs 只含 osc/non_osc，sign 另含 unknown 层——子集关系；'
      '真正的同源由入口构造性保证）',
      obs_layers_f <= (mag_layers_f | sign_layers_f),
      str(sorted(obs_layers_f)))

# F2 缺失输入：目录不存在 → FileNotFoundError 原样传播（不捕获）
try:
    run_analysis(OBSERVABLE_SC_PAIRING_SYMMETRY,
                 condition={'doping': 'd_x', 'N': 200, 'bond_dim': '1500',
                            'super_corr_root': os.path.join(
                                tmp_f, 'not_exist')})
    check('F', '缺失输入：FileNotFoundError 原样传播', False, '未抛异常')
except FileNotFoundError:
    check('F', '缺失输入：FileNotFoundError 原样传播', True)

# F3 config 透传：默认（None）与显式均可用
rep_f3 = run_analysis(OBSERVABLE_SC_PAIRING_SYMMETRY, condition=cond_f)
check('F', 'config 透传：不传 config 使用默认配置且运行成功',
      rep_f3.status == STATUS_OK)

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
