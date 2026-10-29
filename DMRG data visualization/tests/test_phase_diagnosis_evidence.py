# -*- coding: utf-8 -*-
"""
Phase_Diagnosis P1-C 证据映射层定向测试（build_evidence_items 纯函数）

覆盖：
- 全类型合成报告映射：衰减 / 自旋 SF / 电荷 SF / SC 配对 SF / 峰值 /
  能量 / failed / None 报告 / 未知 analyzer
- 缺失字段（空 parameters/空 metadata/data_restricted=None）→
  None 保留，不推断
- payload 可用/不可用两种路径（payload 不复制、经 entry_index 关联）
- 类别纪律：所有 categories ⊆ 五类；不产生 order_evidence /
  counter_evidence；不修改原报告与 phase_report
- 纯函数确定性（重复调用相等、不写入传入对象）
- 严格 JSON 序列化

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_phase_diagnosis_evidence.py
"""
import json
import os
import sys
from types import SimpleNamespace

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK, STATUS_FAILED)
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, build_evidence_items,
    EVIDENCE_OBSERVATION, EVIDENCE_DERIVED_OBSERVATION,
    EVIDENCE_QUALITY_DIAGNOSTIC, EVIDENCE_CAPABILITY_LIMITATION,
    EVIDENCE_NOT_AVAILABLE, EVIDENCE_CATEGORIES)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ---- 合成报告构造 ----
def decay_rep():
    return AnalysisReport(
        analyzer='SpinCorrDecayAnalyzer', observable='spin_correlation',
        status=STATUS_OK, classification='power_law',
        parameters={'K': 1.5, 'A': 0.1}, evidence='strong',
        data_restricted=None, warnings=['w1'],
        metadata={'r_min': 2.0, 'r_max': 20.0, 'n_points': 18,
                  'verdict': 'stable', 'param_errors': {'K': 0.1}},
        payload=None)


def sf_rep(analyzer, observable, payload, metadata):
    return AnalysisReport(
        analyzer=analyzer, observable=observable, status=STATUS_OK,
        classification=None, parameters={}, evidence=None,
        data_restricted=bool(metadata.get('is_restricted')),
        warnings=[], metadata=metadata, payload=payload)


def peak_rep():
    return AnalysisReport(
        analyzer='SpinStructureFactorPeakAnalyzer',
        observable='spin_structure_factor', status=STATUS_OK,
        classification=None,
        parameters={'n_peaks': 1, 'peak_positions': [[0.0, 0.0]],
                    'peak_heights': [0.9]}, evidence=None,
        data_restricted=True, warnings=[],
        metadata={'N': 200, 'definition': 'sz_sz'}, payload=None)


def energy_rep():
    return AnalysisReport(
        analyzer='EnergyProcessor', observable='energy', status=STATUS_OK,
        classification=None,
        parameters={'E_inf': -0.0937, 'power': 1, 'threshold': 2.0,
                    'N_values': [80, 160, 240, 320, 400]},
        evidence=None, data_restricted=None, warnings=[],
        metadata={'outlier_diagnosis': {'masked_N': [160]}}, payload=None)


# ======================================================================
# A. 全类型映射
# ======================================================================
print('===== A. 全类型映射 =====')
reports = [
    decay_rep(),
    sf_rep('SpinStructureFactorEstimator', 'spin_structure_factor',
           {'S_q': np.array([1.0, 0.5]), 'is_restricted': True},
           {'N': 200, 'definition': 'sz_sz', 'is_restricted': True,
            'q_grid': {'n_qx': 12, 'n_qy': 6, 'qx_min': 0.0, 'qx_max': 1.0,
                       'qy_min': 0.0, 'qy_max': 1.0}}),
    sf_rep('ChargeStructureFactorEstimator', 'charge_structure_factor',
           {'S_q': np.array([0.8]), 'diagonal_source_sites': [0, 49, 50],
            'is_restricted': True},
           {'N': 200, 'definition': 'charge_connected', 'is_restricted': True}),
    sf_rep('SCPairStructureFactorEstimator', 'sc_pair_structure_factor',
           {'S_q': np.array([0.3]), 'is_restricted': True},
           {'N': 200, 'is_restricted': True, 'expected_pair_count': None,
            'record_source': 'full_direction_file',
            'direction': 'A', 'ref_bond': [49, 50]}),
    peak_rep(),
    energy_rep(),
    AnalysisReport(analyzer='SpinCorrDecayAnalyzer',
                   observable='spin_correlation', status=STATUS_FAILED),
    None,   # 缺失报告
]
ph = collect_evidence(reports, labels=[f'L{i}' for i in range(len(reports))])
items = build_evidence_items(ph)

check('A', '条目数与 entries 一一对应（8 条）',
      len(items) == len(ph.entries) == 8,
      f'{len(items)} vs {len(ph.entries)}')
check('A', 'evidence_id/entry_index 稳定且与 entries 索引一致',
      all(it['evidence_id'] == f'e{i}' and it['entry_index'] == i
          for i, it in enumerate(items)))
d0 = items[0]
check('A', '衰减：observation + derived_observation + quality_diagnostic '
      '（classification 仅保留，不升级）',
      set(d0['categories']) == {EVIDENCE_OBSERVATION,
                                EVIDENCE_DERIVED_OBSERVATION,
                                EVIDENCE_QUALITY_DIAGNOSTIC},
      str(d0['categories']))
check('A', '衰减：data_range 从 metadata 实际键提取（r_min/r_max/n_points），'
      '未提供的键为 None（如 N）',
      d0['data_range']['r_min'] == 2.0 and d0['data_range']['n_points'] == 18
      and d0['data_range']['N'] is None
      and d0['classification'] == 'power_law',
      str(d0['data_range']))
d1 = items[1]
check('A', '自旋 SF：payload 可用 → observation + capability_limitation'
      '（is_restricted=True）',
      d1['payload_available'] is True
      and EVIDENCE_OBSERVATION in d1['categories']
      and EVIDENCE_CAPABILITY_LIMITATION in d1['categories']
      and d1['capability']['is_restricted'] is True,
      str(d1['categories']))
d2 = items[2]
check('A', '电荷 SF：observation + capability（is_restricted=True；'
      'payload 不复制）',
      EVIDENCE_OBSERVATION in d2['categories']
      and EVIDENCE_CAPABILITY_LIMITATION in d2['categories']
      and d2['capability']['is_restricted'] is True,
      str(d2['categories']))
d3 = items[3]
check('A', 'SC 配对 SF：expected_pair_count 键存在且 None → capability '
      '记录 None；record_source 存在 → capability_limitation',
      EVIDENCE_CAPABILITY_LIMITATION in d3['categories']
      and d3['capability']['expected_pair_count'] is None
      and d3['capability']['record_source'] == 'full_direction_file',
      str(d3['capability']))
d4 = items[4]
check('A', '峰值：derived_observation（数值检测结果），不含 observation',
      d4['categories'] == [EVIDENCE_DERIVED_OBSERVATION],
      str(d4['categories']))
d5 = items[5]
check('A', '能量：observation + quality_diagnostic（outlier_diagnosis）；'
      'data_range.N_values 提取',
      set(d5['categories']) == {EVIDENCE_OBSERVATION,
                                EVIDENCE_QUALITY_DIAGNOSTIC}
      and d5['data_range']['N_values'] == [80, 160, 240, 320, 400],
      str(d5['categories']))
d6 = items[6]
check('A', 'failed 报告：not_available + 原失败信息保留（analyzer/'
      'observable 原样，不伪造来源）',
      d6['categories'] == [EVIDENCE_NOT_AVAILABLE]
      and d6['analyzer'] == 'SpinCorrDecayAnalyzer'
      and d6['observable'] == 'spin_correlation'
      and 'status=failed' in d6['payload_note'],
      str(d6['categories']))
d7 = items[7]
check('A', 'None 报告：not_available + observable/analyzer 为 None '
      '（不伪造来源）',
      d7['categories'] == [EVIDENCE_NOT_AVAILABLE]
      and d7['observable'] is None and d7['analyzer'] is None
      and d7['payload_available'] is False,
      str(d7['categories']))

# ======================================================================
# B. 缺失字段与边界
# ======================================================================
print('===== B. 缺失字段与边界 =====')

# 空 parameters/空 metadata：不推断
rep_empty = AnalysisReport(
    analyzer='SpinCorrDecayAnalyzer', observable='spin_correlation',
    status=STATUS_OK, classification=None, parameters={}, evidence=None,
    data_restricted=None, warnings=[], metadata={}, payload=None)
items_b = build_evidence_items(collect_evidence([rep_empty]))
b0 = items_b[0]
check('B', '空 parameters/空 metadata：categories 为空列表、字段全 None/空，'
      '不推断任何内容',
      b0['categories'] == [] and b0['parameters'] == {}
      and b0['data_range'] == {k: None for k in
                               ('N', 'N_values', 'n_points', 'n_distances',
                                'n_raw_points', 'r_min', 'r_max',
                                'definition', 'n_observed', 'q_grid',
                                'lattice_geometry', 'boundary',
                                'unit_cell_size')}
      and b0['data_restricted'] is None,
      str(b0['data_range']))

# payload 不可用：SF 报告 payload=None → 不归 observation
rep_nop = sf_rep('SpinStructureFactorEstimator', 'spin_structure_factor',
                 None, {'N': 200})
items_n = build_evidence_items(collect_evidence([rep_nop]))
check('B', 'payload 不可用：不归 observation + payload_note 说明不可用',
      EVIDENCE_OBSERVATION not in items_n[0]['categories']
      and items_n[0]['payload_available'] is False
      and '不可用' in (items_n[0]['payload_note'] or ''),
      str(items_n[0]['categories']))

# data_restricted=None 保留为未知
check('B', 'data_restricted=None 保留为未知（不默认 False/True）',
      d0['data_restricted'] is None)

# 未知 analyzer：保守归类 + mapping_note
rep_unk = AnalysisReport(
    analyzer='UnknownAnalyzer', observable='some_observable',
    status=STATUS_OK, classification=None,
    parameters={'x': 1.0}, evidence=None, data_restricted=None,
    warnings=['w'], metadata={}, payload=None)
items_u = build_evidence_items(collect_evidence([rep_unk]))
check('B', '未知 analyzer：仅按实际存在字段保守归类（observation + '
      'quality_diagnostic）+ mapping_note',
      set(items_u[0]['categories']) == {EVIDENCE_OBSERVATION,
                                        EVIDENCE_QUALITY_DIAGNOSTIC}
      and '未知 analyzer' in (items_u[0]['mapping_note'] or ''),
      str(items_u[0]['categories']))

# 索引越界：report_refs 短于 entries → 安全处理
holder = SimpleNamespace(
    entries=[{'observable': 'x', 'analyzer': 'y', 'status': STATUS_OK,
              'classification': None, 'evidence': None,
              'data_restricted': None, 'warnings': [], 'label': None}],
    report_refs=[])
items_o = build_evidence_items(holder)
check('B', 'report_refs 索引越界：安全归 not_available（不抛异常）',
      items_o[0]['categories'] == [EVIDENCE_NOT_AVAILABLE]
      and items_o[0]['payload_available'] is False,
      str(items_o[0]))

# ======================================================================
# C. 类别纪律与纯函数性
# ======================================================================
print('===== C. 类别纪律与纯函数性 =====')
all_cats = [c for it in items for c in it['categories']]
check('C', '所有 categories ⊆ 五类（无 order_evidence/counter_evidence）',
      all(c in EVIDENCE_CATEGORIES for c in all_cats)
      and 'order_evidence' not in all_cats
      and 'counter_evidence' not in all_cats,
      str(sorted(set(all_cats))))
check('C', '峰值/衰减 classification 不产生正式支持或反证',
      all('order_evidence' not in it['categories']
          and 'counter_evidence' not in it['categories'] for it in items))

# 纯函数：不写入传入对象；重复调用确定
attrs_before = set(vars(ph).keys())
items2 = build_evidence_items(ph)
check('C', '纯函数：不修改传入对象（无新增属性）+ 重复调用结果相等',
      set(vars(ph).keys()) == attrs_before and items == items2)
check('C', '映射不改变原报告字段（report_refs 内存对象未被修改）',
      ph.report_refs[0].parameters == {'K': 1.5, 'A': 0.1}
      and ph.report_refs[0].classification == 'power_law')

# JSON 序列化
raw = json.dumps(items, allow_nan=False)
back = json.loads(raw)
check('C', '严格 JSON 序列化：可解析且关键字段一致（NaN/Infinity 无字面量）',
      'NaN' not in raw and 'Infinity' not in raw
      and len(back) == 8 and back[0]['entry_index'] == 0
      and back[0]['categories'] == d0['categories'])

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
