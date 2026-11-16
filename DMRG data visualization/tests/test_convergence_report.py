# -*- coding: utf-8 -*-
"""
Convergence_Report 独立入口测试（P1 阶段 1）。

覆盖：
- A. CLI 解析与交互提示（复用 _prompt_doping）
- B. 真实数据门控：d1_10 --dims 1500,5000 --no-plots 全流程 exit 0，
  report 含三档判定结论与纪律声明；JSON 严格解析且 verdict ∈ 三档
- C. 纪律：md 明确"不回答能否判相"与未授权阈值标注

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_convergence_report.py
"""
import json
import os
import re
import subprocess
import sys
from unittest.mock import patch

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import Convergence_Report as cr  # noqa: E402
from Physical_Quantity.physical_analysis.Convergence_Check import (  # noqa: E402
    VERDICTS)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ======================================================================
# A. CLI 与交互
# ======================================================================
print('===== A. CLI 与交互提示 =====')
try:
    cr.main(['--doping', 'd1_10', '--N', 'abc'])
    check('A', '--N 非整数 → SystemExit(2)', False, '未退出')
except SystemExit as e:
    check('A', '--N 非整数 → SystemExit(2)', e.code == 2, f'code={e.code}')
check('A', '空穴率 0.1 → d1_10；0.05 → d1_20；0.08 → d2_25；'
      '目录名直通',
      cr._fraction_to_doping('0.1') == 'd1_10'
      and cr._fraction_to_doping('0.05') == 'd1_20'
      and cr._fraction_to_doping('0.08') == 'd2_25'
      and cr._fraction_to_doping('d1_10') == 'd1_10')
with patch('builtins.input', return_value=''):
    n, t_j, doping = cr._prompt_config(None, None, None)
check('A', '_prompt_config 全空输入 → 默认 N=200 / t_J_3 / d1_10',
      (n, t_j, doping) == (200, 't_J_3', 'd1_10'), str((n, t_j, doping)))
_stub_conv = {'dims': ['1500', '5000'],
              'inputs': {}, 'figures': [], 'failures': [],
              'verdict': {'verdict': 'converged', 'reason': 'stub',
                          'checks': [], 'thresholds': {},
                          'threshold_status': 'defaults_not_authorized'}}
_seen = {}


def _fake_run_convergence(config, dims):
    _seen['dims'] = dims
    _seen['doping'] = config['doping']
    _seen['N'] = config['N']
    return _stub_conv


with patch.object(cr, '_prompt_config', return_value=(200, 't_J_3',
                                                      'd1_10')), \
        patch.object(cr.par, 'run_convergence_section',
                     side_effect=_fake_run_convergence), \
        patch.object(cr, 'write_report',
                     return_value={'convergence_report.md': 'x',
                                   'convergence_report.json': 'x'}):
    rc = cr.main(['--dims', '1500,5000'])
check('A', 'main 缺参：交互配置注入 + --dims 解析 + exit 0',
      rc == 0 and _seen == {'dims': ['1500', '5000'], 'doping': 'd1_10',
                            'N': 200}, str(_seen))

# ======================================================================
# B. 真实数据门控
# ======================================================================
print('===== B. 真实数据 =====')
REAL_ROOT = os.path.join(ROOT, 'Data/real/Data_N200', 't_J_3')
if os.path.isdir(os.path.join(REAL_ROOT, 'd1_10')):
    outdir = os.path.join('tests', 'artifacts', 'conv_report_test')
    r = subprocess.run(
        [sys.executable, '-X', 'utf8', 'Convergence_Report.py',
         '--doping', 'd1_10', '--dims', '1500,5000', '--no-plots',
         '--outdir', outdir],
        cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    check('B', 'd1_10 全流程 exit 0', r.returncode == 0,
          (r.stdout + r.stderr)[-200:])
    md = open(os.path.join(ROOT, outdir, 'convergence_report.md'),
              encoding='utf-8').read()
    m = re.search(r'\*\*(converged|needs_higher_D|suspicious)\*\*', md)
    check('B', 'md 含三档判定结论', m is not None and m.group(1) in VERDICTS,
          m.group(1) if m else '缺失')
    js = json.load(open(os.path.join(ROOT, outdir,
                                     'convergence_report.json'),
                        encoding='utf-8'))
    check('B', 'JSON 严格解析且 verdict ∈ 三档',
          js['verdict']['verdict'] in VERDICTS
          and js['verdict']['threshold_status'] == 'defaults_not_authorized')
    check('B', 'd1_10 已知分类随 D 变化 → 判定 suspicious（与 G.11 审计'
          '结论一致）',
          js['verdict']['verdict'] == 'suspicious',
          js['verdict']['verdict'])
else:
    check('B', '真实数据组（不存在，跳过）', True, '跳过')

# ======================================================================
# C. 纪律声明
# ======================================================================
print('===== C. 纪律 =====')
if os.path.isdir(os.path.join(REAL_ROOT, 'd1_10')):
    md = open(os.path.join(ROOT, 'tests', 'artifacts',
                           'conv_report_test',
                           'convergence_report.md'),
              encoding='utf-8').read()
    check('C', 'md 明确"不回答能否判相"与未授权阈值标注',
          '不回答能否判相' in md and 'defaults_not_authorized' in md)
else:
    check('C', '真实数据组（不存在，跳过）', True, '跳过')

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
