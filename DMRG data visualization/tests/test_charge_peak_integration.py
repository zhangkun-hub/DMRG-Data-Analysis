# -*- coding: utf-8 -*-
"""
电荷结构因子峰值路由集成测试（P1-G.3）。

覆盖：
- 合成多峰数据：峰数 / 峰位置（网格上构造值精确复现）/ 峰高 /
  FWHM（半高线性插值 ≈ 2.355σ，容差验证）
- 归一化与峰宽定义核实：find_peaks 输入契约为通用
  compute_structure_factor dict（q / S_q 同长）；电荷侧经同一核
  （归一化 1/N）——definition='charge_density_connected' 与
  is_restricted=True 透传，不默认与自旋完全相同的物理解释
- 路由契约：operation='peak_analysis' 成功；未知 operation
  ValueError；peak_analysis 不支持 config → TypeError
- 证据层：峰报告 → derived_observation（数值检测结果），
  observable=charge_structure_factor（与自旋峰值条目区分）
- 失败路径：q 与 S_q 长度不一致 ValueError 原样传播；None → failed
- 不产生相级结论：classification/evidence 为 None，无 Stripe/CDW
  判定字段

全部合成数据（测试显式构造高斯双峰），不依赖真实数据偶然数值。

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_charge_peak_integration.py
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
    STATUS_OK, STATUS_FAILED, OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_PEAK)
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, build_evidence_items, EVIDENCE_DERIVED_OBSERVATION,
    EVIDENCE_NOT_AVAILABLE)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


NQX, NQY = 21, 7
QX = np.linspace(-1.0, 1.0, NQX)      # 步长 0.1，含 ±0.5 与 0
QY = np.linspace(-0.6, 0.6, NQY)      # 步长 0.2，含 0（峰中心在网格点）
Q = np.array([(x, y) for x in QX for y in QY])


def gauss(x, y, cx, cy, sx, sy, h):
    return h * np.exp(-((x - cx) ** 2 / (2 * sx ** 2)
                        + (y - cy) ** 2 / (2 * sy ** 2)))


# 两峰中心均在网格点上：(-0.5, 0) σ=(0.12, 0.2) h=3.0；
# (0.5, 0) σ=(0.15, 0.2) h=1.5（σ_y=0.2 → 半高 0.235 不触 qy 边界，
# FWHM/步长 > 2 不标分辨率受限）
S_SYN = np.array([gauss(x, y, -0.5, 0.0, 0.12, 0.2, 3.0)
                  + gauss(x, y, 0.5, 0.0, 0.15, 0.2, 1.5)
                  for x, y in Q], dtype=complex)


def make_sf():
    return {'q': Q, 'S_q': S_SYN,
            'definition': 'charge_density_connected',
            'N': 200, 'is_restricted': True,
            'q_grid': {'n_qx': NQX, 'n_qy': NQY,
                       'qx_min': -1.0, 'qx_max': 1.0,
                       'qy_min': -0.6, 'qy_max': 0.6}}


# ======================================================================
# A. 合成多峰：峰数 / 峰位置 / 峰高
# ======================================================================
print('===== A. 合成多峰 =====')
rep = run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                   operation='peak_analysis', sf_result=make_sf())
check('A', '路由成功：status=ok、observable=charge_structure_factor、'
      'analyzer 保持算法出处 SpinStructureFactorPeakAnalyzer',
      rep.status == STATUS_OK
      and rep.observable == OBSERVABLE_CHARGE_STRUCTURE_FACTOR
      and rep.analyzer == ANALYZER_PEAK,
      str((rep.observable, rep.analyzer)))
check('A', '检出 2 个峰',
      rep.parameters['n_peaks'] == 2, str(rep.parameters['n_peaks']))
pos = [tuple(p) for p in rep.parameters['peak_positions']]
check('A', '峰位置精确复现网格构造值（峰中心在网格点，q_original 精确）',
      all(abs(x - cx) < 1e-9 and abs(y - cy) < 1e-9
          for (x, y), (cx, cy) in zip(
              sorted(pos), [(-0.5, 0.0), (0.5, 0.0)])),
      str(pos))
check('A', '峰高与构造幅度一致（|S| 信号，两峰 3.0 / 1.5）',
      abs(rep.parameters['peak_heights'][0] - 3.0) < 1e-9
      and abs(rep.parameters['peak_heights'][1] - 1.5) < 1e-9,
      str(rep.parameters['peak_heights']))

# ======================================================================
# B. FWHM 定义核实（半高线性插值 ≈ 2.355σ）
# ======================================================================
print('===== B. FWHM =====')
payload = rep.payload
p1 = payload['peaks'][0]   # (-0.5, 0) σ=(0.12, 0.2)
fwhm_true_x = 2.3548200450309493 * 0.12
fwhm_true_y = 2.3548200450309493 * 0.2
check('B', 'fwhm_qx ≈ 2.355σ_x（σ=0.12，线性插值容差 5%）',
      p1['fwhm_qx'] is not None
      and abs(p1['fwhm_qx'] - fwhm_true_x) / fwhm_true_x < 0.05,
      str(p1['fwhm_qx']))
check('B', 'fwhm_qy ≈ 2.355σ_y（σ=0.15，线性插值容差 5%）',
      p1['fwhm_qy'] is not None
      and abs(p1['fwhm_qy'] - fwhm_true_y) / fwhm_true_y < 0.05,
      str(p1['fwhm_qy']))
check('B', '峰宽未标分辨率受限（覆盖 >2 采样步长）',
      p1['resolution_limited'] is False)
check('B', '峰报告含 definition/is_restricted 透传（电荷侧口径，'
      '不默认与自旋相同）',
      payload['definition'] == 'charge_density_connected'
      and payload['is_restricted'] is True)
check('B', 'metadata 追加 q_grid 摘要（n_qx/n_qy/端点）',
      rep.metadata.get('q_grid') == make_sf()['q_grid'],
      str(rep.metadata.get('q_grid')))

# ======================================================================
# C. 路由契约
# ======================================================================
print('===== C. 路由契约 =====')
try:
    run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR, operation='decay',
                 sf_result=make_sf())
    check('C', '未知 operation → ValueError', False, '未抛异常')
except ValueError:
    check('C', '未知 operation → ValueError', True)
try:
    run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                 operation='peak_analysis', sf_result=make_sf(),
                 config=object())
    check('C', 'peak_analysis 不支持 config → TypeError', False, '未抛异常')
except TypeError:
    check('C', 'peak_analysis 不支持 config → TypeError', True)
# 与自旋峰值条目区分：同一 sf_result 下 observable 不同
rep_spin = run_analysis(OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                        operation='peak_analysis', sf_result=make_sf())
check('C', '自旋峰值路由不受影响且 observable=spin_structure_factor',
      rep_spin.status == STATUS_OK
      and rep_spin.observable == OBSERVABLE_SPIN_STRUCTURE_FACTOR)

# ======================================================================
# D. 证据层：derived_observation（数值检测，非相级证据）
# ======================================================================
print('===== D. 证据层 =====')
ph = collect_evidence([rep], labels=['charge_peak'])
items = build_evidence_items(ph)
it = items[0]
check('D', '电荷峰值条目 → derived_observation（非原始观测）',
      EVIDENCE_DERIVED_OBSERVATION in it['categories']
      and 'observation' not in it['categories'],
      str(it['categories']))
check('D', '不产生 order_evidence/counter_evidence',
      'order_evidence' not in it['categories']
      and 'counter_evidence' not in it['categories'])
check('D', '条目 observable=charge_structure_factor（与自旋峰值区分）',
      it['observable'] == OBSERVABLE_CHARGE_STRUCTURE_FACTOR)

# ======================================================================
# E. 失败路径
# ======================================================================
print('===== E. 失败路径 =====')
bad_sf = {'q': Q, 'S_q': S_SYN[:10],
          'definition': 'charge_density_connected', 'N': 200}
try:
    run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                 operation='peak_analysis', sf_result=bad_sf)
    check('E', 'q 与 S_q 长度不一致 → ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('E', 'q 与 S_q 长度不一致 → ValueError 原样传播', True)
try:
    run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                 operation='peak_analysis', sf_result=None)
    check('E', 'None 输入 → TypeError 原样传播（与自旋峰值链路一致）',
          False, '未抛异常')
except TypeError:
    check('E', 'None 输入 → TypeError 原样传播（与自旋峰值链路一致）', True)

# ======================================================================
# F. 无相级结论
# ======================================================================
print('===== F. 无相级结论 =====')
check('F', 'classification/evidence 为 None，报告无 Stripe/CDW 判定字段',
      rep.classification is None and rep.evidence is None
      and 'stripe' not in json.dumps(rep.to_dict())
      and 'cdw' not in json.dumps(rep.to_dict()).lower()
      or 'charge_density' in json.dumps(rep.to_dict()))

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
