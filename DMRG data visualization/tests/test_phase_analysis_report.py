# -*- coding: utf-8 -*-
"""
一键受控证据报告测试（P1-G.29 提交 2）。

覆盖：
- A. CLI 解析：--doping 必填、--N 类型校验
- B. 合成数据端到端：临时目录（能量日志 + nf.json）→ build_report
  → report.json 键结构 + 严格 JSON 解析 + report.md UTF-8；
  核心四类目录缺失时记录失败行、节级隔离不阻断
- C. 判据检查表一致性：清单 ID ⊆ 矩阵文档 ID ∪ {gap_v1, width_v1}；
  status ∈ 四态枚举；doc_quoted 条目在文档中可查
- D. 九态值域与无相级结论：candidate_verdicts ⊆ VERDICT_STATES 且
  仅 insufficient/data_missing/not_measurable；md 全文不含相级结论模式
- E. 真实数据组（存在性门控）：--no-plots 全流程 exit 0

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_phase_analysis_report.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import Phase_Analysis_Report as par  # noqa: E402
from Physical_Quantity.Phase_Diagnosis import VERDICT_STATES  # noqa: E402

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ======================================================================
# A. CLI 解析
# ======================================================================
print('===== A. CLI 解析 =====')
try:
    par.main(['--root', '.'])
    check('A', '缺少 --doping → SystemExit(2)', False, '未退出')
except SystemExit as e:
    check('A', '缺少 --doping → SystemExit(2)', e.code == 2, f'code={e.code}')
try:
    par.main(['--doping', 'd1_100', '--N', 'abc'])
    check('A', '--N 非整数 → SystemExit(2)', False, '未退出')
except SystemExit as e:
    check('A', '--N 非整数 → SystemExit(2)', e.code == 2, f'code={e.code}')

# ======================================================================
# B. 合成数据端到端（能量 + 密度；核心四类目录缺失 → 失败行不阻断）
# ======================================================================
print('===== B. 合成数据端到端 =====')
tmp = tempfile.mkdtemp(prefix='par_test_')
# 能量数据：3 个 N 的基态日志（满足外推 ≥3）
energy_root = os.path.join(tmp, 'Data_Energy')
for n, e0 in ((80, -40.0), (160, -40.1), (240, -40.15)):
    ndir = os.path.join(energy_root, 't_J_3', 'd1_10', f'N{n}')
    os.makedirs(ndir, exist_ok=True)
    with open(os.path.join(ndir, 'slurm-1.log'), 'w',
              encoding='utf-8') as f:
        f.write('Bond dimension:\t 500/500\nsweep 1\n'
                f'Site (   1,   2) E0 = {e0} TruncErr = 1.00e-09 '
                'D =  500 Iter =   2\n')
# 密度数据：N=8 的 nf.json（XC 最小构型）
root_syn = os.path.join(tmp, 'Data_N200')
mdir = os.path.join(root_syn, 't_J_3', 'd1_10',
                    '1500_measure_phys_qty', 'measure_49')
os.makedirs(mdir, exist_ok=True)
with open(os.path.join(mdir, 'nf.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps([[[i], [0.9 + 0.001 * i, 0]] for i in range(8)]))

outdir = os.path.join(tmp, 'out')
config = {'root': root_syn, 't_j': 't_J_3', 'doping': 'd1_10',
          'N': 8, 'bond_dim': '1500', 'refbonds': None,
          'n_qx': 21, 'n_qy': 4, 'a': 1.0,
          'energy_root': energy_root, 'j_value': 3.0,
          'with_pairing_symmetry': False, 'with_density_profile': True,
          'no_plots': True, 'outdir': outdir,
          'generated_at': 'test'}
result = par.build_report(config)
check('B', '核心四类目录缺失：产生失败行且不阻断整体（节级隔离）',
      any(o.get('failure') for o in result['outcomes'])
      and result['json'] is not None)
check('B', '能量外推成功（≥3 个 N 的 E∞ outcome 存在）',
      any(o['report'] is not None
          and o['context']['observable'] == 'energy'
          and o['context']['N'] == 'extrap'
          for o in result['outcomes']))
check('B', '密度 profile 成功（local_density outcome 存在）',
      any(o['report'] is not None
          and o['context']['observable'] == 'local_density'
          for o in result['outcomes']))
par.write_report(result, outdir)
check('B', 'report.md / report.json 文件生成',
      os.path.isfile(os.path.join(outdir, 'report.md'))
      and os.path.isfile(os.path.join(outdir, 'report.json')))
md_text = open(os.path.join(outdir, 'report.md'), encoding='utf-8').read()
check('B', 'report.md 为 UTF-8 且含纪律声明',
      '纪律声明' in md_text and '受控证据报告' in md_text)
check('B', 'report.json 严格解析且键结构完整',
      set(json.load(open(os.path.join(outdir, 'report.json'),
                         encoding='utf-8')).keys())
      == {'generated_by', 'config', 'reports', 'rows',
          'comparability_groups', 'phase_report', 'checklist',
          'figure_refs', 'section_failures', 'semantics'})

# ======================================================================
# C. 判据检查表与矩阵文档一致性
# ======================================================================
print('===== C. 判据检查表一致性 =====')
doc_path = os.path.join(ROOT, 'docs',
                        'Phase_Diagnosis_Physical_Criteria_Matrix.md')
doc_text = open(doc_path, encoding='utf-8').read()
doc_ids = set(re.findall(r'\|\s*(sc_lro_v1|sc_assoc_v1|did_phase_v1|'
                         r'cdw_v1|sdw_v1|pdw_v1|stripe_v1)\s*\|', doc_text))
list_ids = {row['criterion_id'] for row in par.CRITERIA_CHECKLIST}
quoted_ids = {row['criterion_id'] for row in par.CRITERIA_CHECKLIST
              if row['doc_quoted']}
check('C', '清单 ID = 矩阵 7 条 + gap/width（GAPW 规则）',
      list_ids == doc_ids | {'gap_v1', 'width_v1'}, sorted(list_ids))
check('C', 'doc_quoted 条目的 ID 均存在于矩阵文档',
      quoted_ids <= doc_ids,
      sorted(quoted_ids - doc_ids))
check('C', '全部行 status ∈ 四态枚举',
      all(row['status'] in par.CHECKLIST_STATUSES
          for row in par.CRITERIA_CHECKLIST))
check('C', '全部行含必需字段（criterion_id/candidate/status/'
      'data_availability/note）',
      all(all(k in row for k in ('criterion_id', 'candidate', 'status',
                                 'data_availability', 'note'))
          for row in par.CRITERIA_CHECKLIST))

# ======================================================================
# D. 九态值域与无相级结论
# ======================================================================
print('===== D. 值域与结论边界 =====')
verdicts = result['json']['phase_report']['candidate_verdicts']
check('D', 'candidate_verdicts 值 ⊆ 九态枚举',
      all(v['verdict'] in VERDICT_STATES for v in verdicts.values()))
check('D', 'verdict 仅出现 insufficient/data_missing/not_measurable '
      '（当前注册规则下无 supported/contradicted）',
      set(v['verdict'] for v in verdicts.values())
      <= {'insufficient', 'data_missing', 'not_measurable'},
      str(set(v['verdict'] for v in verdicts.values())))
pattern = re.compile(
    r'(是|支持|确立|发现)[^。]{0,12}'
    r'(超导|条纹|电荷密度波|自旋密度波|配对密度波|SDW|CDW|PDW|d\+id)'
    r'[^。]{0,4}(相|态)')
check('D', 'report.md 不含相级结论模式（精确正则，避免相位/相关误伤）',
      pattern.search(md_text) is None)

# ======================================================================
# E. 真实数据组（存在性门控）
# ======================================================================
print('===== E. 真实数据组 =====')
REAL_ROOT = os.path.join(ROOT, 'Data_N200', 't_J_3')
if os.path.isdir(os.path.join(REAL_ROOT, 'd1_100')):
    r = subprocess.run(
        [sys.executable, '-X', 'utf8', 'Phase_Analysis_Report.py',
         '--doping', 'd1_100', '--no-plots',
         '--outdir', os.path.join('tests', 'artifacts', 'par_real')],
        cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    check('E', 'd1_100 全流程 exit 0', r.returncode == 0,
          (r.stdout + r.stderr)[-200:])
    real_md = os.path.join(ROOT, 'tests', 'artifacts', 'par_real',
                           'report.md')
    check('E', '真实报告生成且含 8 个候选 verdict 表',
          os.path.isfile(real_md) and '候选 verdict' in
          open(real_md, encoding='utf-8').read())
else:
    check('E', '真实数据组（Data_N200/t_J_3/d1_100 不存在，跳过）',
          True, '跳过')

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
