# -*- coding: utf-8 -*-
"""
Spin_Structure_Factor 测试

独立参考实现（ref_sum）与正式实现不共享求和代码，避免同源错误掩盖。
覆盖：小系统直接求和、全零、仅对角、已知矩阵、有序对/重复/对称性、
缺失对与覆盖率、定义隔离、XC 动量网格、N200 真实受限估计、异常输入。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_spin_structure_factor.py
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Spin_Structure_Factor import (
    compute_structure_factor, xc_momentum_grid, SpinStructureFactorEstimator,
    DEF_SPIN_DOT, DEF_SZ_SZ)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def ref_sum(coords, pairs, q, N):
    """独立参考实现：纯 Python 逐对求和（重复键同样取均值，规则与正式实现
    一致但代码完全独立）。"""
    # 独立去重：重复键取均值
    acc = {}
    for item in pairs:
        key = (item[0], item[1])
        v = float(item[2])
        acc[key] = 0.5 * (acc[key] + v) if key in acc else v
    S = 0.0 + 0.0j
    for (i, j), v in acc.items():
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        S += v * np.exp(1j * (q[0] * dx + q[1] * dy))
    return S / N


# ==========================================================================
# 1. N=8 人工晶格：直接求和参考
# ==========================================================================
print('===== A. 小系统直接求和验证 =====')
g8 = HoneycombXCGeometry(8, 1.0)
coords8 = g8.coords
N8 = 8
rng = np.random.default_rng(3)
pairs8 = [(i, (i * 3 + 1) % 8, float(rng.standard_normal())) for i in range(8)]
qs = [(0.0, 0.0), (0.5, 0.8), (1.2, -0.4), (2.0, 2.0)]
for q in qs:
    got = compute_structure_factor(coords8, pairs8, [q], N8, 'test')['S_q'][0]
    ref = ref_sum(coords8, pairs8, q, N8)
    check('A', f'q={q} 与独立参考一致', abs(got - ref) < 1e-12,
          f'got={got:.8f} ref={ref:.8f}')

# ==========================================================================
# 2-4. 全零 / 仅对角 / 已知矩阵
# ==========================================================================
print('===== B. 特殊关联矩阵 =====')
res = compute_structure_factor(coords8, [(i, j, 0.0) for i in range(8) for j in range(8)],
                               [(0.0, 0.0)], N8, 'test')
check('B', '全零关联 → S(q)=0', abs(res['S_q'][0]) < 1e-15, res['S_q'][0])

res = compute_structure_factor(coords8, [(i, i, 1.0) for i in range(8)],
                               [(0.5, 0.7)], N8, 'test')
check('B', '仅对角项 C_ii=1 → S(q)=1 ∀q', abs(res['S_q'][0] - 1.0) < 1e-12,
      f"S={res['S_q'][0]:.8f}")

# 已知矩阵：对称对 (0,1)+(1,0)=1 各 → S(0) = 2/N；q 非零时按相位
pairs_sym = [(0, 1, 1.0), (1, 0, 1.0)]
d01 = (coords8[0][0] - coords8[1][0], coords8[0][1] - coords8[1][1])
for q in [(0.0, 0.0), (0.3, 0.6)]:
    got = compute_structure_factor(coords8, pairs_sym, [q], N8, 'test')['S_q'][0]
    ref = (np.exp(1j * (q[0] * d01[0] + q[1] * d01[1]))
           + np.exp(-1j * (q[0] * d01[0] + q[1] * d01[1]))) / N8
    check('B', f'已知对称矩阵 q={q}', abs(got - ref) < 1e-12, f'got={got:.8f} ref={ref:.8f}')

# ==========================================================================
# 5. 有序对计数、重复输入、对称性处理
# ==========================================================================
print('===== C. 计数与对称性 =====')
res = compute_structure_factor(coords8, pairs_sym, [(0.0, 0.0)], N8, 'test')
check('C', '有序对计数 = 2（(0,1) 与 (1,0) 各算一次）',
      res['measured_pair_count'] == 2, res['measured_pair_count'])
# 重复输入：同一对两次 → 取均值 + duplicate_pairs=1
res = compute_structure_factor(coords8, [(0, 1, 1.0), (0, 1, 3.0)], [(0.0, 0.0)], N8, 'test')
check('C', '重复对取均值（(1+3)/2=2）且计数 1',
      res['duplicate_pairs'] == 1 and res['measured_pair_count'] == 1
      and abs(res['S_q'][0] - 2.0 / N8) < 1e-12,
      f"dup={res['duplicate_pairs']} n={res['measured_pair_count']} S={res['S_q'][0]:.6f}")
# 对称性不自动乘 2：只输入 (0,1) 一项 → S(0)=1/N 而非 2/N
res = compute_structure_factor(coords8, [(0, 1, 1.0)], [(0.0, 0.0)], N8, 'test')
check('C', '不自动乘 2（仅 (0,1) 一项 → S=1/N）',
      abs(res['S_q'][0] - 1.0 / N8) < 1e-12, res['S_q'][0])

# ==========================================================================
# 6. 缺失对与覆盖率
# ==========================================================================
print('===== D. 覆盖率 =====')
full8 = {(i, j) for i in range(8) for j in range(8)}
partial = [(0, 1, 1.0), (2, 3, 0.5)]
res = compute_structure_factor(coords8, partial, [(0.0, 0.0)], N8, 'test',
                               expected_pairs=full8)
check('D', '缺失对 = 62（64−2）', res['missing_pairs'] is not None
      and len(res['missing_pairs']) == 62, len(res['missing_pairs']) if res['missing_pairs'] else None)
check('D', 'pair_coverage = 2/64', abs(res['pair_coverage'] - 2 / 64) < 1e-12,
      res['pair_coverage'])
check('D', 'is_restricted=True', res['is_restricted'])
check('D', 'source_coverage = 2/8', abs(res['source_coverage'] - 2 / 8) < 1e-12,
      res['source_coverage'])
# 不提供 expected_pairs → 覆盖率不可完整评估（None），仍 is_restricted
res = compute_structure_factor(coords8, partial, [(0.0, 0.0)], N8, 'test')
check('D', '无 expected_pairs → pair_coverage=None 且 is_restricted=True',
      res['pair_coverage'] is None and res['is_restricted'])

# ==========================================================================
# 7. 定义隔离
# ==========================================================================
print('===== E. 定义隔离 =====')
res1 = compute_structure_factor(coords8, pairs8, [(0.0, 0.0)], N8, DEF_SPIN_DOT)
res2 = compute_structure_factor(coords8, pairs8, [(0.0, 0.0)], N8, DEF_SZ_SZ)
check('E', 'spin_dot 标签原样记录', res1['definition'] == 'spin_dot')
check('E', 'sz_sz 标签原样记录', res2['definition'] == 'sz_sz')
check('E', '同数据不同定义 S 值相同但标签不同',
      abs(res1['S_q'][0] - res2['S_q'][0]) < 1e-12 and res1['definition'] != res2['definition'])

# ==========================================================================
# 8. XC 动量网格
# ==========================================================================
print('===== F. XC 动量网格 =====')
qs, meta = xc_momentum_grid(200, a=1.0, n_qx=11, n_qy=4)
check('F', 'Ly = 2√3·a', abs(meta['Ly'] - 2 * np.sqrt(3.0)) < 1e-12, meta['Ly'])
check('F', 'qy 允许值 = 2πm/Ly（4 个）',
      all(abs(qs[i][1] - 2 * np.pi * (i % 4) / meta['Ly']) < 1e-12 for i in range(len(qs)))
      and len(set(np.round(qs[:, 1], 12))) == 4)
check('F', 'qx 为开放方向连续扫描（不含人工周期化标记）',
      meta['x_periodic'] is False and meta['x_note'] and len(qs) == 11 * 4)
try:
    xc_momentum_grid(10)
    check('F', 'N=10 抛 ValueError', False, '未抛异常')
except ValueError:
    check('F', 'N=10 抛 ValueError', True)

# ==========================================================================
# 9. 异常输入
# ==========================================================================
print('===== G. 异常输入 =====')
res = compute_structure_factor(coords8, [], [(0.0, 0.0)], N8, 'test')
check('G', '空对表 → S=0 且 measured=0',
      abs(res['S_q'][0]) < 1e-15 and res['measured_pair_count'] == 0)
res = compute_structure_factor(coords8, [(0, 1, np.nan), (2, 3, 1.0)], [(0.0, 0.0)], N8, 'test')
check('G', 'NaN 值不计入且 invalid_values=1',
      res['invalid_values'] == 1 and res['measured_pair_count'] == 1,
      f"invalid={res['invalid_values']} n={res['measured_pair_count']}")
res = compute_structure_factor(coords8, [(99, 1, 1.0)], [(0.0, 0.0)], N8, 'test')
check('G', '缺坐标对 → skipped_no_coords 报告且不计入',
      len(res['skipped_no_coords']) == 1 and res['measured_pair_count'] == 0,
      res['skipped_no_coords'])

# ==========================================================================
# 10. N200 真实数据受限估计
# ==========================================================================
print('===== H. N200 真实数据受限估计 =====')
sz_path = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty/measure_49/sz.json'
qs200, meta200 = xc_momentum_grid(200, a=1.0, n_qx=21, n_qy=4)
res = SpinStructureFactorEstimator.estimate_sz_sz_from_file(sz_path, qs200, 200)
check('H', 'sz_sz 定义正确且数值有限',
      res['definition'] == 'sz_sz' and np.all(np.isfinite(res['S_q'])),
      res['definition'])
check('H', 'source_coverage = 1/200（如实报告）',
      abs(res['source_coverage'] - 1 / 200) < 1e-12, res['source_coverage'])
check('H', 'is_restricted=True 且 pair_coverage=None（不可完整评估）',
      res['is_restricted'] and res['pair_coverage'] is None)
check('H', '对角项按实测计入（measured=200 条源点0数据）',
      res['measured_pair_count'] == 200, res['measured_pair_count'])
check('H', '虚部残差报告（对称性差 → 数值有限即可）',
      np.isfinite(res['imaginary_residual']), res['imaginary_residual'])
check('H', 'sz_sz 输出含 meta（几何元信息透传，来自实际几何对象）',
      res.get('meta') is not None and res['meta']['N'] == 200
      and abs(res['meta']['Ly'] - 2 * np.sqrt(3.0)) < 1e-12
      and abs(res['meta']['Lx_span'] - 74.0) < 1e-9
      and res['meta']['boundary_conditions'] == {'x': 'open', 'y': 'periodic'},
      res.get('meta'))

# 源点 49/50 三分量合成 spin_dot 受限估计
from Physical_Quantity.Physical_Data_Record import CorrelationRecord
from Physical_Quantity.Correlation_Process import CorrelationProcessor
all_recs = []
for measure, ver, seq in [('measure_49', '49', range(1, 300, 4)),
                          ('measure_50', '50', range(2, 300, 4))]:
    base = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty/{measure}'
    sz = CorrelationRecord.extract_data(
        open(f'{base}/sz{ver}sz.json', encoding='utf-8').read(), seq, 200, 1.0)
    su = CorrelationRecord.extract_data(
        open(f'{base}/sp{ver}sm.json', encoding='utf-8').read(), seq, 200, 1.0)
    sd = CorrelationRecord.extract_data(
        open(f'{base}/sm{ver}sp.json', encoding='utf-8').read(), seq, 200, 1.0)
    all_recs.extend(CorrelationProcessor.Combined_spin_correlation(sz, su, sd))
res = SpinStructureFactorEstimator.estimate_spin_dot_from_records(all_recs, qs200, 200)
check('H', 'spin_dot 受限估计：定义/受限性/数值',
      res['definition'] == 'spin_dot' and res['is_restricted']
      and np.all(np.isfinite(res['S_q'])),
      f"n={res['measured_pair_count']} src={res['source_coverage']:.4f}")
check('H', 'source_coverage = 2/200',
      abs(res['source_coverage'] - 2 / 200) < 1e-12, res['source_coverage'])
check('H', 'spin_dot 输出同样含 meta（与 sz_sz 同几何，n_cells_x=25）',
      res.get('meta') is not None and res['meta']['N'] == 200
      and res['meta']['n_cells_x'] == 25 and res['meta']['n_rows_y'] == 4
      and res['meta']['boundary_conditions'] == {'x': 'open', 'y': 'periodic'},
      res.get('meta'))

# ==========================================================================
# 11. q_grid 实际网格统计（D4.5-T1：与电荷侧语义一致）
# ==========================================================================
print('===== I. q_grid 网格统计 =====')
from Physical_Quantity.Spin_Structure_Factor import _q_grid_stats

# I1. 规则网格
qs_i1, _ = xc_momentum_grid(200, a=1.0, n_qx=11, n_qy=4)
g_i1 = _q_grid_stats(qs_i1)
check('I', '规则网格：n_qx/n_qy 为唯一坐标数量（11/4）',
      g_i1['n_qx'] == 11 and g_i1['n_qy'] == 4, g_i1)
check('I', '规则网格：端点与输入实际值一致',
      abs(g_i1['qx_min'] - qs_i1[:, 0].min()) < 1e-12
      and abs(g_i1['qx_max'] - qs_i1[:, 0].max()) < 1e-12
      and abs(g_i1['qy_min'] - qs_i1[:, 1].min()) < 1e-12
      and abs(g_i1['qy_max'] - qs_i1[:, 1].max()) < 1e-12, g_i1)

# I2. 非规则网格与单 qy
qs_i2 = np.array([(0.0, 0.0), (0.5, 0.0), (1.5, 0.0), (3.0, 0.0)])
g_i2 = _q_grid_stats(qs_i2)
check('I', '非规则/单 qy 网格：唯一坐标数与端点如实',
      g_i2['n_qx'] == 4 and g_i2['n_qy'] == 1
      and g_i2['qx_min'] == 0.0 and g_i2['qx_max'] == 3.0
      and g_i2['qy_min'] == 0.0 and g_i2['qy_max'] == 0.0, g_i2)

# I3. 空 q_list
g_i3 = _q_grid_stats(np.empty((0, 2)))
check('I', '空 q_list：n=0、四端点 None',
      g_i3['n_qx'] == 0 and g_i3['n_qy'] == 0
      and g_i3['qx_min'] is None and g_i3['qx_max'] is None
      and g_i3['qy_min'] is None and g_i3['qy_max'] is None, g_i3)

# I4. 混合有限与 NaN/Inf
qs_i4 = np.array([[0.0, 0.0], [np.nan, 0.0], [1.0, -np.inf], [2.0, 0.0]])
g_i4 = _q_grid_stats(qs_i4)
check('I', '混合 NaN/Inf：只统计两分量均有限的点',
      g_i4['n_qx'] == 2 and g_i4['n_qy'] == 1
      and g_i4['qx_min'] == 0.0 and g_i4['qx_max'] == 2.0, g_i4)

# I5. 全部非有限
qs_i5 = np.array([[np.nan, np.nan], [np.inf, np.nan]])
g_i5 = _q_grid_stats(qs_i5)
check('I', '全部非有限：n=0、四端点 None',
      g_i5['n_qx'] == 0 and g_i5['n_qy'] == 0
      and g_i5['qx_min'] is None and g_i5['qy_min'] is None, g_i5)

# I6. sz_sz 入口返回 q_grid（真实数据 + 规则网格）
res_i6 = SpinStructureFactorEstimator.estimate_sz_sz_from_file(sz_path, qs200, 200)
check('I', 'sz_sz 入口：q_grid 存在且与统计函数一致',
      res_i6.get('q_grid') == _q_grid_stats(qs200), res_i6.get('q_grid'))

# I7. spin_dot 入口返回 q_grid（合成 records）
recs_i7 = [CorrelationRecord(0, 1, 1.0, 0.5), CorrelationRecord(0, 2, 1.0, 0.3)]
res_i7 = SpinStructureFactorEstimator.estimate_spin_dot_from_records(recs_i7, qs200, 200)
check('I', 'spin_dot 入口：q_grid 存在且与统计函数一致',
      res_i7.get('q_grid') == _q_grid_stats(qs200), res_i7.get('q_grid'))

# ==========================================================================
# 汇总
# ==========================================================================
print()
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in RESULTS:
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    print(f'{cat:12}: {n_pass}/{len(RESULTS[cat])} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)
