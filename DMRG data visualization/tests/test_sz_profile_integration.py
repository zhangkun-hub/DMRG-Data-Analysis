# -*- coding: utf-8 -*-
"""
局域自旋 sz.json 分析链路测试（P1 阶段 2）。

覆盖：
- A. 契约：OBSERVABLE_LOCAL_SZ / ANALYZER_SZ_PROFILE 常量；
  from_sz_profile 的失败映射与 staggered_magnetization 计算
- B. 集成：sz_profile_report 真实数据（存在性门控）——status ok、
  参数含 sz_by_site/sub_lattice_mean/staggered_magnetization、
  data_restricted=False
- C. 统一入口：run_analysis(local_sz, 'profile') 路由可达（14 条路由
  之一）；未知路由仍抛 ValueError
- D. 边界：文件缺失 → FileNotFoundError 原样传播（不吞异常）；
  合成格式错误 → ValueError
- E. 纪律：适配器不产生 classification/evidence（数值观测，不判相）

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_sz_profile_integration.py
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.evidence_integration.Analysis_Contract import (  # noqa: E402
    OBSERVABLE_LOCAL_SZ, ANALYZER_SZ_PROFILE, from_sz_profile, STATUS_OK,
    STATUS_FAILED)
from Physical_Quantity.evidence_integration.Analysis_Integration import (  # noqa: E402
    sz_profile_report)
from Physical_Quantity.evidence_integration.Analysis_Entry import (  # noqa: E402
    run_analysis)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ======================================================================
# A. 契约与适配器
# ======================================================================
print('===== A. 契约与适配器 =====')
check('A', 'OBSERVABLE_LOCAL_SZ / ANALYZER_SZ_PROFILE 常量定义',
      OBSERVABLE_LOCAL_SZ == 'local_sz'
      and ANALYZER_SZ_PROFILE == 'SzProfileEstimator')
rep = from_sz_profile(None)
check('A', 'None 输入 → failed（映射与密度适配器一致）',
      rep.status == STATUS_FAILED
      and rep.observable == OBSERVABLE_LOCAL_SZ
      and rep.analyzer == ANALYZER_SZ_PROFILE)
synth = {'N': 8, 'a': 1.0,
         'density_by_site': [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
         'row_mean': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
         'sub_lattice_mean': {'A': 0.5, 'B': -0.5},
         'unit_cell_mean': [0.0],
         'non_finite_count': 0,
         'geometry': {'lattice': 'HoneycombXCGeometry', 'unit_cell_size': 8,
                      'boundary': 'x open / y periodic',
                      'row_partition': 'i % 4'},
         'sub_lattice_partition': {'derivation': 'BFS bipartite',
                                   'site_labels': {'A': [], 'B': []}}}
rep = from_sz_profile(synth)
check('A', 'staggered_magnetization = A - B 子格均值差',
      rep.parameters['staggered_magnetization'] == 1.0,
      str(rep.parameters['staggered_magnetization']))
check('A', 'classification/evidence 为 None（数值观测，不判相）',
      rep.classification is None and rep.evidence is None)
check('A', 'data_restricted=False（单格点全量测量）',
      rep.data_restricted is False)
check('A', 'metadata 含观测语义说明（不判 SDW/AFM 声明）',
      'staggered_magnetization' in rep.metadata.get('note', ''))

# ======================================================================
# B. 集成（真实数据门控）
# ======================================================================
print('===== B. sz_profile_report 真实数据 =====')
REAL = os.path.join(ROOT, 'Data_N200', 't_J_3', 'd1_10',
                    '1500_measure_phys_qty', 'measure_49', 'sz.json')
if os.path.isfile(REAL):
    rep = sz_profile_report(REAL, N=200, a=1.0)
    check('B', 'status=ok 且 analyzer 正确',
          rep.status == STATUS_OK and rep.analyzer == ANALYZER_SZ_PROFILE)
    check('B', 'N=200 且 sz_by_site 长度 200',
          rep.parameters['N'] == 200
          and len(rep.parameters['sz_by_site']) == 200)
    check('B', 'sub_lattice_mean 含 A/B 且 staggered 为两者差（1e-12 内）',
          set(rep.parameters['sub_lattice_mean']) == {'A', 'B'}
          and abs(rep.parameters['staggered_magnetization']
                  - (rep.parameters['sub_lattice_mean']['A']
                     - rep.parameters['sub_lattice_mean']['B'])) < 1e-12)
    check('B', 'unit_cell_mean 每胞一个（N/8=25 个）',
          len(rep.parameters['unit_cell_mean']) == 25)
else:
    check('B', '真实 sz.json（不存在，跳过）', True, '跳过')

# ======================================================================
# C. 统一入口路由
# ======================================================================
print('===== C. 统一入口 =====')
if os.path.isfile(REAL):
    rep = run_analysis(OBSERVABLE_LOCAL_SZ, operation='profile',
                       sz_path=REAL, N=200, a=1.0)
    check('C', 'run_analysis(local_sz, profile) 路由可达（14 条路由）',
          rep.status == STATUS_OK and rep.observable == OBSERVABLE_LOCAL_SZ)
try:
    run_analysis('not_a_real_observable')
    check('C', '未知 observable 仍抛 ValueError', False, '未抛异常')
except ValueError:
    check('C', '未知 observable 仍抛 ValueError', True)

# ======================================================================
# D. 边界：异常原样传播
# ======================================================================
print('===== D. 边界 =====')
try:
    sz_profile_report(os.path.join(tempfile.gettempdir(),
                                   'no_such_file.json'), N=200)
    check('D', '文件缺失 → FileNotFoundError 原样传播', False, '未抛异常')
except FileNotFoundError:
    check('D', '文件缺失 → FileNotFoundError 原样传播', True)
with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False,
                                 encoding='utf-8') as f:
    f.write('not a json')
    bad = f.name
try:
    sz_profile_report(bad, N=200)
    check('D', '格式错误 → 异常原样传播（不吞）', False, '未抛异常')
except Exception:
    check('D', '格式错误 → 异常原样传播（不吞）', True)
finally:
    os.unlink(bad)

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
