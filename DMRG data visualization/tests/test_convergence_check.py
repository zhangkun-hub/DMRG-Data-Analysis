# -*- coding: utf-8 -*-
"""
DMRG 收敛诊断测试（P1 阶段 1：Convergence_Check 叶子模块）。

覆盖：
- A. 五项检查各自的判定边界（合成输入：PASS/FAIL/NA）
- B. 聚合 verdict 三档逻辑（converged / needs_higher_D / suspicious）
- C. 阈值可覆盖 + 阈值来源状态标注（defaults_not_authorized）
- D. 报告脚本收敛节集成（真实数据存在性门控：--convergence-dims
  1500,5000 全流程 exit 0 且报告含收敛结论与证据图引用）

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_convergence_check.py
"""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.physical_analysis.Convergence_Check import (  # noqa: E402
    convergence_verdict, check_e0_saturation, check_truncerr, check_su2,
    check_classification_consistency, check_parameter_consistency,
    DEFAULT_THRESHOLDS, VERDICT_CONVERGED, VERDICT_NEEDS_HIGHER_D,
    VERDICT_SUSPICIOUS)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


TH = dict(DEFAULT_THRESHOLDS)

# ======================================================================
# A. 单项检查边界
# ======================================================================
print('===== A. 单项检查 =====')
r = check_e0_saturation({'1500': -40.1000, '5000': -40.100001}, TH)
check('A', 'e0 饱和：相对差 ≤ 阈值 → PASS', r['passed'] is True, r['evidence'])
r = check_e0_saturation({'1500': -40.0, '5000': -40.01}, TH)
check('A', 'e0 饱和：相对差 1e-4 量级超限 → FAIL', r['passed'] is False,
      r['evidence'])
r = check_e0_saturation({'1500': -40.0}, TH)
check('A', 'e0 饱和：单档 → NA（证据不足）', r['passed'] is None)
r = check_truncerr({'1500': 1e-9, '5000': 1e-9}, TH)
check('A', 'truncerr：两档均低 → PASS', r['passed'] is True)
r = check_truncerr({'1500': 1e-6}, TH)
check('A', 'truncerr：超限 → FAIL', r['passed'] is False, r['evidence'])
r = check_truncerr({}, TH)
check('A', 'truncerr：无记录 → NA', r['passed'] is None)
r = check_su2({'1500': 1e-5, '5000': 1e-6}, TH)
check('A', 'su2：残差低于警惕线 → PASS', r['passed'] is True)
r = check_su2({'1500': 0.05}, TH)
check('A', 'su2：残差超警惕线 → FAIL', r['passed'] is False, r['evidence'])
r = check_classification_consistency(
    {'1500': ('power_law', 'exponential'),
     '5000': ('power_law', 'exponential')}, TH)
check('A', '分类一致：两档元组相同 → PASS', r['passed'] is True)
r = check_classification_consistency(
    {'1500': ('power_law',), '5000': ('exponential',)}, TH)
check('A', '分类一致：两档不同 → FAIL（振荡信号）', r['passed'] is False)
r = check_classification_consistency({'1500': ('power_law',)}, TH)
check('A', '分类一致：单档 → NA', r['passed'] is None)
r = check_parameter_consistency(
    {'1500': {'49_50_A_K': 1.5}, '5000': {'49_50_A_K': 1.51}},
    {'1500': {'49_50_A_K': 0.01}, '5000': {'49_50_A_K': 0.01}}, TH)
check('A', '参数一致：误差内 → PASS', r['passed'] is True, r['evidence'])
r = check_parameter_consistency(
    {'1500': {'49_50_A_K': 1.5}, '5000': {'49_50_A_K': 2.5}},
    {'1500': {'49_50_A_K': 0.01}, '5000': {'49_50_A_K': 0.01}}, TH)
check('A', '参数一致：超容差 → FAIL', r['passed'] is False, r['evidence'])
r = check_parameter_consistency({'1500': {'K': 1.5}, '5000': {'K': 1.5}},
                                {}, TH)
check('A', '参数一致：误差缺失 → NA', r['passed'] is None)

# ======================================================================
# B. 聚合 verdict 三档
# ======================================================================
print('===== B. 聚合 verdict =====')
v = convergence_verdict(
    e0_by_dim={'1500': -40.1000, '5000': -40.100001},
    truncerr_by_dim={'1500': 1e-9, '5000': 1e-9},
    su2_by_dim={'1500': 1e-5, '5000': 1e-6},
    classes_by_dim={'1500': ('power_law',), '5000': ('power_law',)},
    params_by_dim={'1500': {'K': 1.5}, '5000': {'K': 1.51}},
    param_errors_by_dim={'1500': {'K': 0.01}, '5000': {'K': 0.01}})
check('B', '五项全过 → converged', v['verdict'] == VERDICT_CONVERGED,
      v['verdict'])
v = convergence_verdict(
    e0_by_dim={'1500': -40.0, '5000': -40.01})
check('B', 'E0 未饱和 → needs_higher_D', v['verdict']
      == VERDICT_NEEDS_HIGHER_D, v['reason'])
v = convergence_verdict(
    classes_by_dim={'1500': ('power_law',), '5000': ('exponential',)})
check('B', '分类不一致优先 → suspicious（即使其他项通过）',
      v['verdict'] == VERDICT_SUSPICIOUS, v['reason'])
v = convergence_verdict(e0_by_dim={'1500': -40.0})
check('B', '单档证据不足 → needs_higher_D（注明证据不足）',
      v['verdict'] == VERDICT_NEEDS_HIGHER_D and '证据不足' in v['reason'],
      v['reason'])
v = convergence_verdict()
check('B', '完全无输入 → needs_higher_D（不编造收敛）',
      v['verdict'] == VERDICT_NEEDS_HIGHER_D)

# ======================================================================
# C. 阈值与来源标注
# ======================================================================
print('===== C. 阈值纪律 =====')
v = convergence_verdict(
    e0_by_dim={'1500': -40.0, '5000': -40.01},
    thresholds={'e0_rel_tol': 1e-2})
check('C', '显式阈值覆盖生效（1e-2 下相对差 2.5e-4 → converged）',
      v['verdict'] == VERDICT_CONVERGED, v['verdict'])
v = convergence_verdict(e0_by_dim={'1500': -40.0, '5000': -40.0001})
check('C', '默认阈值的来源状态恒标注 defaults_not_authorized',
      v['threshold_status'] == 'defaults_not_authorized'
      and 'thresholds' in v and len(v['thresholds']) == 4)
check('C', 'verdict 值 ∈ 三档枚举',
      v['verdict'] in (VERDICT_CONVERGED, VERDICT_NEEDS_HIGHER_D,
                       VERDICT_SUSPICIOUS))

# ======================================================================
# D. 报告脚本集成（真实数据门控）
# ======================================================================
print('===== D. 报告脚本收敛节 =====')
REAL_ROOT = os.path.join(ROOT, 'Data/real/Data_N200', 't_J_3')
if os.path.isdir(os.path.join(REAL_ROOT, 'd1_100')):
    r = subprocess.run(
        [sys.executable, '-X', 'utf8', 'Phase_Analysis_Report.py',
         '--doping', 'd1_100', '--convergence-dims', '1500,5000',
         '--no-plots', '--outdir',
         os.path.join('tests', 'artifacts', 'conv_real')],
        cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    check('D', 'd1_100 --convergence-dims 1500,5000 全流程 exit 0',
          r.returncode == 0, (r.stdout + r.stderr)[-200:])
    md = open(os.path.join(ROOT, 'tests', 'artifacts', 'conv_real',
                           'report.md'), encoding='utf-8').read()
    check('D', '报告含收敛诊断节与结论',
          '收敛诊断' in md
          and re.search(r'\*\*结论：(converged|needs_higher_D|suspicious)'
                        r'\*\*', md) is not None)
    check('D', '报告含未授权阈值标注',
          'defaults_not_authorized' in md
          and '不回答能否判相' in md)
else:
    check('D', '真实数据组（不存在，跳过）', True, '跳过')

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
