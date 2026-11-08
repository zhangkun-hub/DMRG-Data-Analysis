# -*- coding: utf-8 -*-
"""
复值 vs 实值对比程序测试（Complex_Real_Comparison.py）。

覆盖：
- A. 单元：_pair_stats 全局归一（近零点不爆炸）；check_energy 三分支
  （复值更高 → 未通过 / 更低 → 通过 / 无数据 → 证据不足）；_imag_ratio；
  _find_log 无匹配 → None
- B. 门控（真实数据存在时）：四类装载成功；run_checks 返回三态之一；
  SU(2) 检查 passed 为 Python bool

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_complex_real_comparison.py
"""
import os
import sys
import tempfile
import types

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import Complex_Real_Comparison as crc  # noqa: E402

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def zrec(re, im):
    return types.SimpleNamespace(_value=complex(re, im))


# ======================================================================
# A. 单元
# ======================================================================
print('===== A. 单元 =====')
mx, mn = crc._pair_stats([1.0, 1e-10, 0.0], [1.0, 0.0, 0.0])
check('A', '_pair_stats 全局归一：近零点不爆炸',
      mx < 1e-3 and mn < 1e-3, f'max={mx:.2e}, mean={mn:.2e}')

c = crc.check_energy(-31.9, -31.9)
check('A', 'check_energy 差值极小 → 通过',
      c['passed'] is True, c['evidence'])
c = crc.check_energy(-31.8, -31.9)
check('A', 'check_energy 复值显著更高 → 未通过',
      c['passed'] is False and 'local minimum' in c['note'])
c = crc.check_energy(None, -31.9)
check('A', 'check_energy 日志缺失 → 证据不足（passed is None）',
      c['passed'] is None)

r = crc._imag_ratio([zrec(1.0, 0.0), zrec(-0.5, 0.0)], lambda x: x._value)
check('A', '_imag_ratio 纯实值 → 0', r == 0.0)
r = crc._imag_ratio([zrec(1.0, 1e-4)], lambda x: x._value)
check('A', '_imag_ratio 含虚部 → 1e-4', abs(r - 1e-4) < 1e-12, f'{r:.2e}')

check('A', '_find_log 不存在的目录 → (None, None)',
      crc._find_log(os.path.join(tempfile.gettempdir(), 'no_such_dir_xyz'))
      == (None, None))

# ======================================================================
# B. 真实数据门控
# ======================================================================
print('===== B. 真实数据门控 =====')
if os.path.isdir(crc.REAL_ROOT) and os.path.isdir(crc.COMPLEX_ROOT):
    dens_r = crc.load_density(crc.REAL_ROOT)
    dens_c = crc.load_density(crc.COMPLEX_ROOT)
    spin_r = crc.load_spin(crc.REAL_ROOT)
    spin_c = crc.load_spin(crc.COMPLEX_ROOT)
    sc_r = crc.load_supercorr(crc.REAL_ROOT)
    sc_c = crc.load_supercorr(crc.COMPLEX_ROOT)
    sp_r = crc.load_single_particle(crc.REAL_ROOT)
    sp_c = crc.load_single_particle(crc.COMPLEX_ROOT)
    check('B', '四类装载成功且两组等长',
          len(dens_r) == len(dens_c) == 200
          and len(spin_r['A sublattice']) == len(spin_c['A sublattice'])
          and len(sp_r) == len(sp_c))
    check('B', '超导 refbond 集合两组一致',
          set(sc_r) == set(sc_c) and len(sc_r) == 3)

    e0s_r, te_r = crc.load_e0_series(crc.REAL_ROOT)
    e0s_c, te_c = crc.load_e0_series(crc.COMPLEX_ROOT)
    checks, verdict = crc.run_checks(
        spin_r, spin_c, dens_r, dens_c, sc_r, sc_c, sp_r, sp_c,
        e0s_c[-1] if e0s_c else None, e0s_r[-1] if e0s_r else None,
        e0s_c, e0s_r, te_c, te_r)
    check('B', 'run_checks 返回 verdict ∈ 三态',
          verdict in ('consistent', 'minor_deviation', 'suspicious'),
          verdict)
    check('B', '五项检查 passed 均为 Python bool 或 None',
          all(isinstance(c['passed'], (bool, type(None)))
              for c in checks))
    su2 = crc._su2_residual_check()
    check('B', 'SU(2) 检查 passed 为 Python bool',
          isinstance(su2['passed'], bool) and su2['passed'] is True,
          su2['evidence'][:60])
else:
    check('B', '真实数据目录不存在（跳过）', True, '跳过')

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
