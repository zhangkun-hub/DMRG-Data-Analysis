# -*- coding: utf-8 -*-
"""
配对关联 vs G² 比较测试（P1 阶段 3）。

覆盖：
- A. 叶子模块：合成数据斜率比较三态判定（配对慢于 G² / 快于 G² /
  边际内 insufficient / 点数不足 insufficient / 阈值覆盖 / 来源标注）
- B. 契约：from_pairing_g2_comparison 的 failed 映射与纪律 metadata
- C. 集成：pairing_g2_report 真实数据（存在性门控）——status ok、
  verdict ∈ 三态、参数含 k_sc/k_g/k_g2
- D. 统一入口：run_analysis(pairing_g2_comparison, 'compare') 可达
- E. 纪律：classification/evidence 为 None（数值诊断不判相）

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_pairing_g2_comparison.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.physical_analysis.Pairing_G2_Comparison import (  # noqa: E402
    compare_pairing_vs_g2, VERDICT_PAIRING_DOMINANT,
    VERDICT_SINGLE_PARTICLE_DOMINANT, VERDICT_INSUFFICIENT)
from Physical_Quantity.evidence_integration.Analysis_Contract import (  # noqa: E402
    from_pairing_g2_comparison, STATUS_FAILED)
from Physical_Quantity.evidence_integration.Analysis_Integration import (  # noqa: E402
    pairing_g2_report)
from Physical_Quantity.evidence_integration.Analysis_Entry import (  # noqa: E402
    run_analysis)
from Physical_Quantity.evidence_integration.Analysis_Contract import (  # noqa: E402
    OBSERVABLE_PAIRING_G2_COMPARISON)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


import numpy as np  # noqa: E402
R = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# ======================================================================
# A. 叶子模块三态判定
# ======================================================================
print('===== A. 叶子模块 =====')
# 配对 r^-0.1（K_sc=0.1，慢）vs G r^-1（K_G=1 → 2K_G=2，快）：
# 配对慢于 G² → pairing_dominant
v = compare_pairing_vs_g2(R, [r**-0.1 for r in R],
                          [r**-1.0 for r in R])
check('A', '配对慢于 G² → pairing_dominant',
      v['verdict'] == VERDICT_PAIRING_DOMINANT, v['verdict'])
check('A', '斜率数值正确（k_sc≈-0.1、k_g2≈-2，1e-6 内）',
      abs(v['k_sc'] + 0.1) < 1e-6 and abs(v['k_g2'] + 2.0) < 1e-6,
      f"{v['k_sc']} / {v['k_g2']}")
# 配对 r^-1（快）vs G r^-0.1（G² r^-0.2 慢）：配对快于 G²
v = compare_pairing_vs_g2(R, [r**-1 for r in R], [r**-0.1 for r in R])
check('A', '配对快于 G² → single_particle_dominant',
      v['verdict'] == VERDICT_SINGLE_PARTICLE_DOMINANT, v['verdict'])
# 边际内
v = compare_pairing_vs_g2(R, [r**-1 for r in R], [r**-0.55 for r in R])
check('A', '斜率差在边际内 → insufficient（不编造区分）',
      v['verdict'] == VERDICT_INSUFFICIENT, v['detail'])
# 点数不足
v = compare_pairing_vs_g2([1.0, 2.0], [1.0, 0.5], [1.0, 0.5])
check('A', '点数不足 → insufficient', v['verdict'] == VERDICT_INSUFFICIENT)
# 长度不一致
v = compare_pairing_vs_g2([1.0, 2.0, 3.0], [1.0, 0.5],
                          [1.0, 0.5, 0.2])
check('A', '序列长度不一致 → insufficient（不抛不编造）',
      v['verdict'] == VERDICT_INSUFFICIENT)
# 阈值覆盖与来源标注
v = compare_pairing_vs_g2(R, [r**-0.1 for r in R], [r**-1 for r in R],
                          thresholds={'slope_margin': 5.0})
check('A', '显式阈值覆盖生效（大边际 → insufficient）',
      v['verdict'] == VERDICT_INSUFFICIENT)
check('A', '阈值来源恒标注 defaults_not_authorized',
      v['threshold_status'] == 'defaults_not_authorized')
# window 过滤
v = compare_pairing_vs_g2(R, [r**-0.1 for r in R], [r**-1 for r in R],
                          window=(2.0, 5.0))
check('A', 'window 过滤后仍可判定（4 点 ≥ min_points）',
      v['verdict'] == VERDICT_PAIRING_DOMINANT)

# ======================================================================
# B. 契约适配器
# ======================================================================
print('===== B. 契约 =====')
rep = from_pairing_g2_comparison(None)
check('B', 'None 输入 → failed', rep.status == STATUS_FAILED)
rep = from_pairing_g2_comparison(
    compare_pairing_vs_g2(R, [r**-1 for r in R], [r**-0.2 for r in R]))
check('B', 'classification/evidence 为 None（数值诊断不判相）',
      rep.classification is None and rep.evidence is None)
check('B', 'metadata 含不判相纪律说明与未授权阈值标注',
      '不判相' in rep.metadata['note']
      and rep.metadata['threshold_status'] == 'defaults_not_authorized')

# ======================================================================
# C/D. 集成与统一入口（真实数据门控）
# ======================================================================
print('===== C/D. 集成与入口 =====')
REAL_SUPER = os.path.join(ROOT, 'Data/real/Data_N200', 't_J_3', 'd1_10',
                          '1500_measure_super_corr', 'super_corr_49_50')
REAL_G = os.path.join(ROOT, 'Data/real/Data_N200', 't_J_3', 'd1_10',
                      '1500_measure_phys_qty', 'measure_49',
                      'single_particle_corr2.json')
if os.path.isdir(REAL_SUPER) and os.path.isfile(REAL_G):
    import Cross_Doping_Summary as cds  # noqa: E402
    from Physical_Quantity.data_records.Physical_Data_Record import (  # noqa: E402
        SingleParticleRecord)
    pairing_recs = cds.load_super_corr_records(
        REAL_SUPER, (49, 50), 'A', 200, 1.0)
    g_recs = SingleParticleRecord.extract_data(
        open(REAL_G, encoding='utf-8').read(), 200, 1.0)
    rep = pairing_g2_report(pairing_recs, g_recs)
    check('C', '真实数据：status=ok 且 verdict ∈ 三态',
          rep.status == 'ok' and rep.parameters['verdict']
          in (VERDICT_PAIRING_DOMINANT, VERDICT_SINGLE_PARTICLE_DOMINANT,
              VERDICT_INSUFFICIENT),
          rep.parameters['verdict'])
    check('C', '真实数据：k_sc/k_g2 为有限数值',
          all(isinstance(rep.parameters[k], float)
              and np.isfinite(rep.parameters[k])
              for k in ('k_sc', 'k_g2')
              if rep.parameters[k] is not None))
    rep2 = run_analysis(OBSERVABLE_PAIRING_G2_COMPARISON,
                        operation='compare',
                        pairing_recs=pairing_recs, g_recs=g_recs)
    check('D', 'run_analysis(pairing_g2_comparison, compare) 可达',
          rep2.status == 'ok'
          and rep2.observable == OBSERVABLE_PAIRING_G2_COMPARISON)
else:
    check('C', '真实数据（不存在，跳过）', True, '跳过')
    check('D', '真实数据（不存在，跳过）', True, '跳过')

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
