# -*- coding: utf-8 -*-
"""
Physical_Quantity 四层目录结构测试（P1 目录重组，最终形态）。

覆盖：
- A. 四层子包结构完整（4 子包 × 正确成员）；包根无任何 .py 模块
  （仅空 __init__.py）
- B. __init__.py 为空（无兼容层/别名注册）
- C. 子包路径 import 全链路可用（跨层抽查）
- D. sys.modules 无旧路径别名残留；包根无转发桩残留

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_package_layout.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PKG = os.path.join(ROOT, 'Physical_Quantity')

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


LAYOUT = {
    'data_records': ['Honeycomb_Lattice', 'Physical_Data_Record',
                     'DMRG_Log_Parser'],
    'physical_analysis': ['Density_Process', 'Correlation_Process',
                          'Super_Correlation_Process', 'Energy_Process',
                          'Correlation_Decay_Analyzer',
                          'Super_Correlation_Analyzer',
                          'Spin_Correlation_Analyzer',
                          'Charge_Correlation_Analyzer',
                          'Green_Function_Analyzer', 'Spin_Structure_Factor',
                          'Spin_Structure_Factor_Analyzer',
                          'Charge_Structure_Factor', 'SC_Pair_Structure_Factor',
                          'Pairing_Symmetry_Analyzer'],
    'evidence_integration': ['Analysis_Contract', 'Analysis_Integration',
                             'Analysis_Entry'],
    'phase_diagnosis': ['Phase_Diagnosis'],
}
ALL_MODULES = [m for mods in LAYOUT.values() for m in mods]

# ======================================================================
# A. 子包结构与包根洁净
# ======================================================================
print('===== A. 四层子包结构 =====')
for sub, mods in LAYOUT.items():
    check('A', f'{sub}/ 存在且成员完整（{len(mods)} 个）',
          all(os.path.isfile(os.path.join(PKG, sub, m + '.py'))
              for m in mods)
          and os.path.isfile(os.path.join(PKG, sub, '__init__.py')))
root_py = sorted(f for f in os.listdir(PKG)
                 if f.endswith('.py') and f != '__init__.py')
check('A', '包根无任何 .py 模块（仅 __init__.py；无转发桩残留）',
      root_py == [], root_py)
check('A', '子包目录恰为 4 个',
      sorted(d for d in os.listdir(PKG)
             if os.path.isdir(os.path.join(PKG, d))
             and not d.startswith('__')) == sorted(LAYOUT))

# ======================================================================
# B. __init__.py 为空
# ======================================================================
print('===== B. 无兼容层 =====')
init_text = open(os.path.join(PKG, '__init__.py'), encoding='utf-8').read()
check('B', 'Physical_Quantity/__init__.py 为空（无 sys.modules 别名注册）',
      init_text.strip() == '', repr(init_text[:80]))
check('B', '子包 __init__.py 均为空',
      all(open(os.path.join(PKG, sub, '__init__.py'),
               encoding='utf-8').read().strip() == '' for sub in LAYOUT))

# ======================================================================
# C. 子包路径 import 全链路
# ======================================================================
print('===== C. 子包路径 import =====')
from Physical_Quantity.data_records.Physical_Data_Record import (  # noqa: E402
    DensityRecord, CorrelationRecord, SuperCorrRecord)
from Physical_Quantity.evidence_integration.Analysis_Entry import (  # noqa: E402
    run_analysis)
from Physical_Quantity.phase_diagnosis.Phase_Diagnosis import (  # noqa: E402
    AUTHORIZED_RULES, P1F3_RULES, GAPW_RULES, run_candidate_rules)
from Physical_Quantity.physical_analysis.Charge_Structure_Factor import (  # noqa: E402
    ChargeStructureFactorEstimator)
from Physical_Quantity.physical_analysis.Pairing_Symmetry_Analyzer import (  # noqa: E402
    PairingSymmetryConfig)
from Physical_Quantity.data_records.DMRG_Log_Parser import (  # noqa: E402
    DMRGLogParser)
check('C', '公共 API 可用（DensityRecord/run_analysis/AUTHORIZED_RULES/'
      'ChargeStructureFactorEstimator/PairingSymmetryConfig）',
      DensityRecord is not None and callable(run_analysis)
      and len(AUTHORIZED_RULES) == 8
      and len(P1F3_RULES) == 6 and len(GAPW_RULES) == 2
      and callable(ChargeStructureFactorEstimator.estimate_batch)
      and PairingSymmetryConfig is not None
      and callable(DMRGLogParser))

# ======================================================================
# D. 无别名/桩残留
# ======================================================================
print('===== D. 无旧路径残留 =====')
import Physical_Quantity  # noqa: E402
for name in ALL_MODULES:
    old = f'Physical_Quantity.{name}'
    ok = old not in sys.modules
    check('D', f'sys.modules 无旧路径别名 {old}', ok,
          '' if ok else sys.modules[old].__file__)
NAME_TO_SUB = {m: s for s, mods in LAYOUT.items() for m in mods}
check('D', 'import 后模块 __file__ 指向子包真实文件（抽查 5 个）',
      all(f'{os.sep}{NAME_TO_SUB[name]}{os.sep}' in
          __import__(f'Physical_Quantity.{NAME_TO_SUB[name]}.{name}',
                     fromlist=['*']).__file__
          for name in ALL_MODULES[:5]))

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
