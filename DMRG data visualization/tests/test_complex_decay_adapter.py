# -*- coding: utf-8 -*-
"""
衰减适配层复值语义测试（模做包络、实部做 signed 通道）。

覆盖：
- A. Spin group_by_distance：复值记录 → C_abs = |复均值|（模）、
  C = 实部均值（signed 通道）；实值退化（无 *_complex 属性）→ 原行为
- B. Super group_by_distance：同款语义；虚部恒 0 的复数 → C_abs == |C|
  逐位一致（回归基线安全的前提）
- C. 散布统计与 sign_mixed 用实部（复数无排序/比较）

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_complex_decay_adapter.py
"""
import os
import sys
import types

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from Physical_Quantity.physical_analysis.Spin_Correlation_Analyzer import (  # noqa: E402
    SpinCorrDecayAnalyzer)
from Physical_Quantity.physical_analysis.Super_Correlation_Analyzer import (  # noqa: E402
    SuperCorrDecayAnalyzer)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def spin_rec(values):
    """合成 CorrelationRecord 等价对象（复数或 float 两种形态）。"""
    out = []
    for i, v in enumerate(values):
        kw = {}
        if isinstance(v, complex):
            kw['correlation_complex'] = v
        kw['correlation'] = float(v.real) if isinstance(v, complex) else float(v)
        out.append(types.SimpleNamespace(
            source_point=49, field_point=50 + i, distance=float(i + 1), **kw))
    return out


def super_rec(values):
    out = []
    for i, v in enumerate(values):
        kw = {}
        if isinstance(v, complex):
            kw['sup_corr_complex'] = v
        kw['sup_corr'] = float(v.real) if isinstance(v, complex) else float(v)
        out.append(types.SimpleNamespace(
            ref_bond=(49, 50), target_bond=(49 + i, 50 + i),
            distance=float(i + 1), **kw))
    return out


# ======================================================================
# A. Spin 复值语义
# ======================================================================
print('===== A. Spin group_by_distance =====')
# 两个复值点同距离 1.0：complex(0.3, 0.4) 与 complex(0.3, 0.4)
recs = spin_rec([complex(0.3, 0.4), complex(0.3, 0.4)])
prep = SpinCorrDecayAnalyzer.group_by_distance(recs, None, None, 1.0)
check('A', 'C_abs = |复均值| = 0.5（模口径）',
      abs(prep.C_abs[0] - 0.5) < 1e-15, f'{prep.C_abs[0]:.17g}')
check('A', 'C = 实部均值 = 0.3（signed 通道）',
      abs(prep.C[0] - 0.3) < 1e-15, f'{prep.C[0]:.17g}')
check('A', 'C_min/C_max/C_std 用实部',
      abs(prep.C_min[0] - 0.3) < 1e-15
      and abs(prep.C_max[0] - 0.3) < 1e-15)

# 实值退化：纯 float 属性（无 *_complex）
recs = spin_rec([0.3, 0.3])
prep2 = SpinCorrDecayAnalyzer.group_by_distance(recs, None, None, 1.0)
check('A', '实值退化（无 complex 属性）→ C_abs = |C|',
      abs(prep2.C_abs[0] - 0.3) < 1e-15, f'{prep2.C_abs[0]:.17g}')

# 虚部恒 0 的复数 → 与实值路径逐位一致
recs3 = spin_rec([complex(0.3, 0.0), complex(-0.1, 0.0)])
prep3 = SpinCorrDecayAnalyzer.group_by_distance(recs3, None, None, 1.0)
check('A', '虚部=0 复数 → C_abs == |C| 逐位（基线安全前提）',
      prep3.C_abs[0] == abs(prep3.C[0]), f'{prep3.C_abs[0]:.17g}')

# 实部正负混合（同距离同组）→ sign_mixed（实部判定）
recs4 = spin_rec([complex(0.3, 0.5), complex(0.3, 0.5)])  # 先占位距离
recs4[1] = types.SimpleNamespace(
    source_point=49, field_point=51, distance=1.0,
    correlation_complex=complex(-0.1, 0.5), correlation=-0.1)
prep4 = SpinCorrDecayAnalyzer.group_by_distance(recs4, None, None, 1.0)
check('A', 'sign_mixed 按实部判定（复数无法比较大小）',
      'sign_mixed' in prep4.group_flags[0])

# ======================================================================
# B. Super 复值语义
# ======================================================================
print('===== B. Super group_by_distance =====')
recs = super_rec([complex(0.3, 0.4), complex(0.3, 0.4)])
prep = SuperCorrDecayAnalyzer.group_by_distance(recs, None)
check('B', 'C_abs = 0.5（模）且 C = 0.3（实部）',
      abs(prep.C_abs[0] - 0.5) < 1e-15 and abs(prep.C[0] - 0.3) < 1e-15)

recs = super_rec([complex(0.3, 0.0), complex(-0.1, 0.0)])
prep = SuperCorrDecayAnalyzer.group_by_distance(recs, None)
check('B', '虚部=0 复数 → C_abs == |C| 逐位',
      prep.C_abs[0] == abs(prep.C[0]), f'{prep.C_abs[0]:.17g}')

recs = super_rec([0.3, 0.3])   # 纯 float（合成数据常用形态）
prep = SuperCorrDecayAnalyzer.group_by_distance(recs, None)
check('B', '实值退化（无 complex 属性）→ C_abs = |C|',
      abs(prep.C_abs[0] - 0.3) < 1e-15)

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
