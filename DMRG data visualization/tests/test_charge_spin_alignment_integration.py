# -*- coding: utf-8 -*-
"""
电荷-自旋峰位对齐观测（O4，P1-G.4）集成测试。

覆盖：
- 峰位重合：min_dq≈0；双向并集统计（一侧峰多时 max_dq 不遗漏
  无对齐对象的方向）
- 峰位全部偏离：min/max dq 与构造几何距离一致
- 无峰场景：q_grid 一致但一侧无峰 → no_peaks，不计算距离
- 可比性：q_grid 缺失 → missing_q_grid；摘要不一致 →
  inconsistent_q_grid——距离字段 None、status 仍 ok、warnings 明示
  （不可比是观测事实，不是计算失败也不是相级结论）
- 类型错误：非 AnalysisReport 输入 → TypeError 原样传播
- 证据层：derived_observation（派生数值观测），无
  order_evidence/counter_evidence，comparability 经 metadata 保留
- 序列化：严格 JSON 回读一致
- 路由契约：默认操作 align；未知 operation ValueError；
  align 不支持 config → TypeError

全部合成数据（高斯峰构造在网格点上）。

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_charge_spin_alignment_integration.py
"""
import json
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
    ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT)
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, build_evidence_items, EVIDENCE_DERIVED_OBSERVATION)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


NQX, NQY = 21, 7
QX = np.linspace(-1.0, 1.0, NQX)
QY = np.linspace(-0.6, 0.6, NQY)
Q = np.array([(x, y) for x in QX for y in QY])
QGRID = {'n_qx': NQX, 'n_qy': NQY,
         'qx_min': -1.0, 'qx_max': 1.0,
         'qy_min': -0.6, 'qy_max': 0.6}


def gauss(x, y, cx, cy, sx, sy, h):
    return h * np.exp(-((x - cx) ** 2 / (2 * sx ** 2)
                        + (y - cy) ** 2 / (2 * sy ** 2)))


def sf_of(centers, heights=()):
    """centers: [(cx, cy, sx, sy, h)]"""
    S = np.zeros(len(Q), dtype=complex)
    for cx, cy, sx, sy, h in centers:
        S += np.array([gauss(x, y, cx, cy, sx, sy, h) for x, y in Q])
    return {'q': Q, 'S_q': S,
            'definition': 'charge_density_connected', 'N': 200}


def peak_rep(sf, observable, qgrid=QGRID):
    rep = run_analysis(observable, operation='peak_analysis',
                       sf_result=sf)
    rep.metadata['q_grid'] = qgrid
    return rep


def align(rep_c, rep_s):
    return run_analysis(OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
                        charge_peak_report=rep_c, spin_peak_report=rep_s)


ONE_PEAK = [(-0.5, 0.0, 0.12, 0.2, 3.0)]
TWO_PEAK = [(-0.5, 0.0, 0.12, 0.2, 3.0), (0.5, 0.0, 0.15, 0.2, 1.5)]

# ======================================================================
# A. 峰位重合（电荷 1 峰 ⊂ 自旋 2 峰）
# ======================================================================
print('===== A. 峰位重合 =====')
rep_c = peak_rep(sf_of(ONE_PEAK), OBSERVABLE_CHARGE_STRUCTURE_FACTOR)
rep_s = peak_rep(sf_of(TWO_PEAK), OBSERVABLE_SPIN_STRUCTURE_FACTOR)
rep_a = align(rep_c, rep_s)
check('A', '路由成功：status=ok、契约常量正确',
      rep_a.status == STATUS_OK
      and rep_a.analyzer == ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT
      and rep_a.observable == OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT)
check('A', 'q_grid 摘要一致 → comparable=consistent_q_grid',
      rep_a.metadata['comparability'] == 'consistent_q_grid',
      rep_a.metadata['comparability'])
check('A', '峰数正确：电荷 1、自旋 2',
      rep_a.parameters['n_charge_peaks'] == 1
      and rep_a.parameters['n_spin_peaks'] == 2)
check('A', '重合峰位：min_dq ≈ 0',
      rep_a.parameters['min_dq'] < 1e-9, str(rep_a.parameters['min_dq']))
check('A', '双向并集统计：自旋独有峰 (0.5,0) 计入 max_dq ≈ 1.0',
      abs(rep_a.parameters['max_dq'] - 1.0) < 1e-9,
      str(rep_a.parameters['max_dq']))
check('A', '逐电荷峰最近自旋峰记录正确（q_charge/q_spin/dq）',
      rep_a.parameters['charge_peak_nearest_spin']
      == [[[-0.5, 0.0], [-0.5, 0.0], 0.0]],
      str(rep_a.parameters['charge_peak_nearest_spin']))

# ======================================================================
# B. 峰位全部偏离
# ======================================================================
print('===== B. 峰位偏离 =====')
rep_c2 = peak_rep(sf_of(ONE_PEAK), OBSERVABLE_CHARGE_STRUCTURE_FACTOR)
rep_s2 = peak_rep(sf_of([(0.5, 0.0, 0.15, 0.2, 1.5)]),
                  OBSERVABLE_SPIN_STRUCTURE_FACTOR)
rep_b = align(rep_c2, rep_s2)
check('B', '单峰偏离 1.0：min_dq = max_dq ≈ 1.0（双向一致）',
      abs(rep_b.parameters['min_dq'] - 1.0) < 1e-9
      and abs(rep_b.parameters['max_dq'] - 1.0) < 1e-9,
      str((rep_b.parameters['min_dq'], rep_b.parameters['max_dq'])))
check('B', '峰值偏离是数值观测，不产生相级结论（classification/evidence None）',
      rep_b.classification is None and rep_b.evidence is None)

# ======================================================================
# C. 无峰场景
# ======================================================================
print('===== C. 无峰场景 =====')
flat_sf = {'q': Q, 'S_q': np.ones(len(Q), dtype=complex),
           'definition': 'charge_density_connected', 'N': 200}
rep_c3 = peak_rep(sf_of(ONE_PEAK), OBSERVABLE_CHARGE_STRUCTURE_FACTOR)
rep_s3 = peak_rep(flat_sf, OBSERVABLE_SPIN_STRUCTURE_FACTOR)
rep_c3b = align(rep_c3, rep_s3)
check('C', 'q_grid 一致但自旋侧无峰 → comparability=no_peaks、距离 None',
      rep_c3b.metadata['comparability'] == 'no_peaks'
      and rep_c3b.parameters['min_dq'] is None
      and rep_c3b.parameters['charge_peak_nearest_spin'] == [],
      str((rep_c3b.metadata['comparability'],
           rep_c3b.parameters['min_dq'])))
check('C', 'no_peaks 时 status 仍 ok（观测完成，无对齐对象不是失败）',
      rep_c3b.status == STATUS_OK)

# ======================================================================
# D. q_grid 缺失 / 不一致
# ======================================================================
print('===== D. 可比性 =====')
rep_c4 = peak_rep(sf_of(ONE_PEAK), OBSERVABLE_CHARGE_STRUCTURE_FACTOR)
rep_s4 = peak_rep(sf_of(TWO_PEAK), OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                  qgrid=dict(QGRID, n_qx=41))
rep_d1 = align(rep_c4, rep_s4)
check('D', 'q_grid 摘要不一致 → inconsistent_q_grid、距离 None、'
      'warnings 明示（不静默）',
      rep_d1.metadata['comparability'] == 'inconsistent_q_grid'
      and rep_d1.parameters['min_dq'] is None
      and any('不可比' in w for w in rep_d1.warnings),
      str(rep_d1.warnings))
rep_c4b = peak_rep(sf_of(ONE_PEAK), OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                   qgrid=None)
rep_d2 = align(rep_c4b, rep_s4)
check('D', 'q_grid 缺失 → missing_q_grid、距离 None',
      rep_d2.metadata['comparability'] == 'missing_q_grid'
      and rep_d2.parameters['min_dq'] is None)
check('D', '不可比场景 status 仍 ok（观测事实，非计算失败）',
      rep_d1.status == STATUS_OK and rep_d2.status == STATUS_OK)

# ======================================================================
# E. 类型错误
# ======================================================================
print('===== E. 类型错误 =====')
try:
    align({'peaks': []}, rep_s4)
    check('E', '非 AnalysisReport 输入 → TypeError 原样传播', False, '未抛异常')
except TypeError:
    check('E', '非 AnalysisReport 输入 → TypeError 原样传播', True)

# ======================================================================
# F. 证据层与序列化
# ======================================================================
print('===== F. 证据层与序列化 =====')
ph = collect_evidence([rep_a], labels=['align'])
it = build_evidence_items(ph)[0]
check('F', '对齐条目 → derived_observation（派生数值观测）',
      EVIDENCE_DERIVED_OBSERVATION in it['categories']
      and 'order_evidence' not in it['categories']
      and 'counter_evidence' not in it['categories'],
      str(it['categories']))
check('F', 'comparability 经 metadata 保留（不扩 data_range 键集）',
      it['metadata']['comparability'] == 'consistent_q_grid'
      and 'comparability' not in it['data_range'],
      str(it['metadata'].get('comparability')))
check('F', '报告无 Stripe/CDW 判定字段',
      'stripe' not in json.dumps(rep_a.to_dict()))
tmp = os.path.join(tempfile.mkdtemp(prefix='p1g4_'), 'a.json')
rep_a.to_json(tmp)
back = json.load(open(tmp, encoding='utf-8'))
check('F', '严格 JSON 回读一致（含 parameters/metadata）',
      back['parameters'] == rep_a.to_dict()['parameters']
      and back['metadata'] == rep_a.to_dict()['metadata'])

# ======================================================================
# G. 路由契约
# ======================================================================
print('===== G. 路由契约 =====')
try:
    run_analysis(OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT, operation='estimate',
                 charge_peak_report=rep_c, spin_peak_report=rep_s)
    check('G', '未知 operation → ValueError', False, '未抛异常')
except ValueError:
    check('G', '未知 operation → ValueError', True)
try:
    run_analysis(OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
                 charge_peak_report=rep_c, spin_peak_report=rep_s,
                 config=object())
    check('G', 'align 不支持 config → TypeError', False, '未抛异常')
except TypeError:
    check('G', 'align 不支持 config → TypeError', True)

# ======================================================================
# H. qy 周期折叠（P1-G.6）
# ======================================================================
print('===== H. qy 周期折叠 =====')
HQY = np.linspace(-5.0, 5.0, 21)   # 步长 0.5，含 0 与 ±4.0
HQ = np.array([(x, y) for x in QX for y in HQY])
A1 = 1.0
Q_PERIOD = 4.0 * np.pi / (np.sqrt(3.0) * A1)   # ≈ 7.2552


def hsf(centers, a=A1):
    S = np.zeros(len(HQ), dtype=complex)
    for cx, cy, sx, sy, h in centers:
        S += np.array([gauss(x, y, cx, cy, sx, sy, h) for x, y in HQ])
    meta = {'a': float(a), 'Ly': 2.0 * np.sqrt(3.0) * float(a)}
    return {'q': HQ, 'S_q': S,
            'definition': 'charge_density_connected', 'N': 200,
            'meta': meta}


# 边界等价峰：电荷峰 qy=0、自旋峰 qy=4.0（Δ=4.0 > Q/2≈3.63）
rep_hc = run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                      operation='peak_analysis',
                      sf_result=hsf([(0.0, 0.0, 0.12, 0.5, 3.0)]))
rep_hc.metadata['q_grid'] = QGRID
rep_hs = run_analysis(OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                      operation='peak_analysis',
                      sf_result=hsf([(0.0, 4.0, 0.12, 0.5, 3.0)]))
rep_hs.metadata['q_grid'] = QGRID
rep_h = align(rep_hc, rep_hs)
check('H', '原始距离保留（min_dq ≈ 4.0，字段语义不变）',
      abs(rep_h.parameters['min_dq'] - 4.0) < 1e-6,
      str(rep_h.parameters['min_dq']))
check('H', '周期折叠距离 min_dq_periodic ≈ Q−4.0 ≈ 3.255',
      abs(rep_h.parameters['min_dq_periodic']
          - (Q_PERIOD - 4.0)) < 1e-6,
      str(rep_h.parameters['min_dq_periodic']))
check('H', 'qy_period 元数据记录 Q=4π/(√3·a) 且 periodic_available=True',
      rep_h.metadata['periodic_distance_available'] is True
      and abs(rep_h.metadata['qy_period'] - Q_PERIOD) < 1e-9,
      str(rep_h.metadata['qy_period']))

# 非边界峰（Δ 小）：周期距离与原始一致
rep_hc2 = run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                       operation='peak_analysis',
                       sf_result=hsf([(0.0, 0.0, 0.12, 0.5, 3.0)]))
rep_hc2.metadata['q_grid'] = QGRID
rep_hs2 = run_analysis(OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                       operation='peak_analysis',
                       sf_result=hsf([(0.0, 1.0, 0.12, 0.5, 3.0)]))
rep_hs2.metadata['q_grid'] = QGRID
rep_h2 = align(rep_hc2, rep_hs2)
check('H', '非边界峰（Δ=1.0 < Q/2）：periodic == 原始',
      abs(rep_h2.parameters['min_dq_periodic'] - 1.0) < 1e-6
      and abs(rep_h2.parameters['min_dq'] - 1.0) < 1e-6,
      str((rep_h2.parameters['min_dq'],
           rep_h2.parameters['min_dq_periodic'])))

# 不同 Ly（a=2）：Q=4π/(√3·2)≈3.628，Δ=4.0 → 折叠到 [−Q/2, Q/2)：
# ((4.0+Q/2) mod Q) − Q/2 ≈ 0.3724；原始距离仍 4.0
rep_hc3 = run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                       operation='peak_analysis',
                       sf_result=hsf([(0.0, 0.0, 0.12, 0.5, 3.0)], a=2.0))
rep_hc3.metadata['q_grid'] = QGRID
rep_hs3 = run_analysis(OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                       operation='peak_analysis',
                       sf_result=hsf([(0.0, 4.0, 0.12, 0.5, 3.0)], a=2.0))
rep_hs3.metadata['q_grid'] = QGRID
rep_h3 = align(rep_hc3, rep_hs3)
q3 = 4.0 * np.pi / (np.sqrt(3.0) * 2.0)
expected3 = (4.0 + q3 / 2.0) % q3 - q3 / 2.0
check('H', '不同 Ly（a=2，Q≈3.628）：periodic=fold(4.0)≈0.3724，'
      '原始仍 4.0',
      abs(rep_h3.parameters['min_dq_periodic'] - expected3) < 1e-9
      and abs(rep_h3.parameters['min_dq'] - 4.0) < 1e-6
      and rep_h3.metadata['periodic_distance_available'] is True,
      str((rep_h3.parameters['min_dq_periodic'],
           rep_h3.parameters['min_dq'])))

# 缺失 meta：periodic 字段 None + note，原始距离不受影响
rep_hc4 = run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                       operation='peak_analysis',
                       sf_result=hsf([(0.0, 0.0, 0.12, 0.5, 3.0)]))
rep_hc4.metadata['q_grid'] = QGRID
rep_hc4.payload.pop('meta', None)   # 模拟缺失几何元数据
rep_h4 = align(rep_hc4, rep_hs)
check('H', '缺失几何 meta：periodic=None + note，原始距离不受影响',
      rep_h4.metadata['periodic_distance_available'] is False
      and rep_h4.parameters['min_dq_periodic'] is None
      and rep_h4.metadata['periodic_note']
      and abs(rep_h4.parameters['min_dq'] - 4.0) < 1e-6,
      str(rep_h4.metadata['periodic_note']))

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
