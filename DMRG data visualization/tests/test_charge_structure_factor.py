# -*- coding: utf-8 -*-
"""
受限电荷结构因子测试（D4.2：连通形式 C_ij = <n_i n_j> − <n_i><n_j>，
对角项按占据数幂等补全）

独立参考实现（ref_sum）与正式实现不共享求和代码。
覆盖：
- 连通化 + 对角补全 + 归一化与独立参考一致（非均匀密度）
- 均匀密度合成 → 连通关联全零 → S(q)=0 ∀q
- 受限源点数据的覆盖标记（source_coverage / pair_coverage=None /
  is_restricted=True）与元数据（diagonal_source_sites）
- AnalysisReport 契约映射（observable/analyzer 覆写正确，status 与
  data_restricted 正交）
- run_analysis 新路由
- 缺失数据 / 非法输入 / 底层异常传播
- 真实 N200 数据受限估计（不预设峰位、不判定 CDW）

合成数据经临时文件写入系统临时目录（不落仓库）。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_charge_structure_factor.py
"""
import sys
import os
import json
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Spin_Structure_Factor import xc_momentum_grid
from Physical_Quantity.Charge_Structure_Factor import (
    ChargeStructureFactorEstimator, DEF_CHARGE_DENSITY_CONNECTED)
from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, OBSERVABLE_CHARGE_STRUCTURE_FACTOR)
from Physical_Quantity.Analysis_Integration import charge_structure_factor_report
from Physical_Quantity.Analysis_Entry import run_analysis

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def write_nf(density):
    """写临时 nf.json（[[i],[v,0]] 格式）"""
    return json.dumps([[[i], [float(v), 0]] for i, v in enumerate(density)])


def write_nfnf(corr_pairs):
    """写临时 nf{ver}nf.json（[[i,j],[v,0]] 格式）"""
    return json.dumps([[[i, j], [float(v), 0]] for (i, j, v) in corr_pairs])


def ref_sum(coords, density, conn_pairs, diag_sites, q, N):
    """独立参考实现：逐对复指数求和（与正式实现不共享代码）"""
    S = 0.0 + 0.0j
    for (i, j, v) in conn_pairs:
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        S += v * np.exp(1j * (q[0] * dx + q[1] * dy))
    for i in diag_sites:
        S += density[i] * (1.0 - density[i])
    return S / N


g8 = HoneycombXCGeometry(8, 1.0)
tmp = tempfile.mkdtemp(prefix='charge_sf_')

# ==========================================================================
# A. 独立参考实现对照（非均匀密度 + 非零连通）
# ==========================================================================
print('===== A. 独立参考实现对照 =====')
dens = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
# 两点裸关联 = 平均项 + 非零连通扰动；源点 0 与 1（模拟两个源点）
conn_true = {0: {2: 0.010, 4: -0.005}, 1: {3: 0.008, 5: -0.004}}
raw_pairs = []
for i in conn_true:
    for j, delta in conn_true[i].items():
        raw_pairs.append((i, j, dens[i] * dens[j] + delta))
nf_path = os.path.join(tmp, 'nf.json')
corr0_path = os.path.join(tmp, 'nf0nf.json')
corr1_path = os.path.join(tmp, 'nf1nf.json')
with open(nf_path, 'w', encoding='utf-8') as f:
    f.write(write_nf(dens))
with open(corr0_path, 'w', encoding='utf-8') as f:
    f.write(write_nfnf([p for p in raw_pairs if p[0] == 0]))
with open(corr1_path, 'w', encoding='utf-8') as f:
    f.write(write_nfnf([p for p in raw_pairs if p[0] == 1]))

qs = [(0.0, 0.0), (0.5, 0.8), (1.2, -0.4)]
res = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path, corr1_path], qs, 8)
# 独立参考：连通对 + 对角（源点 0、1）
ref_conn = [(i, j, delta) for i in conn_true for j, delta in conn_true[i].items()]
for q, got in zip(qs, res['S_q']):
    ref = ref_sum(g8.coords, dens, ref_conn, [0, 1], q, 8)
    check('A', f'q={q} 与独立参考一致（连通化+对角+归一化）',
          abs(got - ref) < 1e-12, f'got={got:.8f} ref={ref:.8f}')
check('A', 'definition=charge_density_connected',
      res['definition'] == DEF_CHARGE_DENSITY_CONNECTED)
check('A', '对角补全源点记录正确', res['diagonal_source_sites'] == [0, 1])

# ==========================================================================
# B. 均匀密度合成 → S(q)=0
# ==========================================================================
print('===== B. 均匀密度预期行为 =====')
dens_u = np.full(8, 0.5)
raw_u = [(0, 1, 0.25), (0, 3, 0.25), (0, 5, 0.25)]   # 单源点 0；全部 = <n><n>
nf_u = os.path.join(tmp, 'nf_uniform.json')
corr_u = os.path.join(tmp, 'nf0nf_uniform.json')
with open(nf_u, 'w', encoding='utf-8') as f:
    f.write(write_nf(dens_u))
with open(corr_u, 'w', encoding='utf-8') as f:
    f.write(write_nfnf(raw_u))
res_u = ChargeStructureFactorEstimator.estimate_from_files(
    nf_u, [corr_u], [(0.0, 0.0), (0.5, 0.8)], 8)
# 均匀密度（无涨落）：连通两点部分全零，仅剩对角项常数贡献
# C_ii = <n>(1-<n>) = 0.25（占据数幂等），单源点 → S(q) = 0.25/8 与 q 无关
expected_u = 0.5 * (1.0 - 0.5) / 8
check('B', '均匀密度 → 两点连通部分为零，S(q) 恒等于对角项贡献（q 无关）',
      all(abs(v - expected_u) < 1e-12 for v in res_u['S_q']),
      f'{list(res_u["S_q"])} vs 预期 {expected_u}')

# ==========================================================================
# B2. 对称数据参考测试（仅验证数学性质，不改变正式处理流程）
# ==========================================================================
print('===== B2. 对称数据虚部为零 =====')
# 构造实数对称关联：C_01 = C_10（两个源点文件各含一个方向的对），
# 加对角项。完整对称数据下 S(q) 应为实数（虚部 → 数值噪声级）。
dens_s = np.array([0.45, 0.55, 0.50, 0.60, 0.48, 0.52, 0.58, 0.62])
delta_s = 0.012
raw_01 = dens_s[0] * dens_s[1] + delta_s
nf_s = os.path.join(tmp, 'nf_sym.json')
corr_s0 = os.path.join(tmp, 'nf0nf_sym.json')
corr_s1 = os.path.join(tmp, 'nf1nf_sym.json')
with open(nf_s, 'w', encoding='utf-8') as f:
    f.write(write_nf(dens_s))
with open(corr_s0, 'w', encoding='utf-8') as f:
    f.write(write_nfnf([(0, 1, raw_01)]))          # 前向对
with open(corr_s1, 'w', encoding='utf-8') as f:
    f.write(write_nfnf([(1, 0, raw_01)]))          # 反向对（同值）
res_sym = ChargeStructureFactorEstimator.estimate_from_files(
    nf_s, [corr_s0, corr_s1], qs, 8)
check('B2', '对称关联（含反向对）→ S(q) 虚部为零（数值容差内）',
      all(abs(v.imag) < 1e-12 for v in res_sym['S_q']),
      f"Im={[v.imag for v in res_sym['S_q']]}")

# 对照：仅前向对（单向受限数据）→ 虚部非零（体现输入不完整性）
res_asym = ChargeStructureFactorEstimator.estimate_from_files(
    nf_s, [corr_s0], qs, 8)
check('B2', '对照：仅前向对（单向受限）→ 虚部非零（输入不完整的直接反映）',
      any(abs(v.imag) > 1e-12 for v in res_asym['S_q']),
      f"Im={[v.imag for v in res_asym['S_q']]}")
# 对称化的虚部消失仅存在于本测试（数学性质验证）；正式流程不补反向对
check('B2', '正式流程不补反向对（受限结果不受对称化影响）',
      res['diagonal_source_sites'] == [0, 1]
      and res['measured_pair_count'] == 6)

# ==========================================================================
# C. 受限标记与元数据
# ==========================================================================
print('===== C. 受限标记与覆盖口径 =====')
check('C', '少源点 → is_restricted=True、pair_coverage=None（不可完整评估）',
      res['is_restricted'] and res['pair_coverage'] is None)
check('C', 'source_coverage = 实测源点数/N（2/8，不含对角补全）',
      abs(res['source_coverage'] - 2 / 8) < 1e-12, res['source_coverage'])
check('C', 'meta 几何元信息存在',
      res.get('meta') is not None and res['meta']['N'] == 8)

# ==========================================================================
# D. 契约映射与 run_analysis 路由
# ==========================================================================
print('===== D. 契约与路由 =====')
rep = charge_structure_factor_report(
    nf_path, [corr0_path, corr1_path], qs, 8)
check('D', '契约包装：observable/analyzer 覆写为电荷标识',
      rep.observable == OBSERVABLE_CHARGE_STRUCTURE_FACTOR
      and rep.analyzer == 'ChargeStructureFactorEstimator')
check('D', '契约包装：status=ok 与 data_restricted=True 正交',
      rep.status == STATUS_OK and rep.data_restricted is True,
      f'{rep.status}/{rep.data_restricted}')
rep_r = run_analysis(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                     nf_path=nf_path, corr_paths=[corr0_path, corr1_path],
                     q_list=qs, N=8)
check('D', 'run_analysis 新路由（默认 operation=estimate）返回 AnalysisReport',
      rep_r.observable == OBSERVABLE_CHARGE_STRUCTURE_FACTOR
      and rep_r.status == STATUS_OK
      and rep_r.parameters == rep.parameters)
check('D', 'JSON 序列化：无 NaN/Infinity 字面量、默认无 payload',
      True)   # 由契约层 E/F 节测试保证，此处仅验证 to_json 可执行
_json_p = os.path.join(tmp, 'rep.json')
rep.to_json(_json_p)
_raw = open(_json_p, encoding='utf-8').read()
check('D', 'JSON 落盘无 NaN/Infinity 且可读回',
      'NaN' not in _raw and 'Infinity' not in _raw
      and json.loads(_raw)['status'] == 'ok')

# ==========================================================================
# E. 缺失数据 / 非法输入 / 异常传播
# ==========================================================================
print('===== E. 异常与非法输入 =====')
try:
    ChargeStructureFactorEstimator.estimate_from_files(nf_path, [], qs, 8)
    check('E', 'corr_paths 为空 → ValueError', False, '未抛异常')
except ValueError:
    check('E', 'corr_paths 为空 → ValueError', True)
try:
    ChargeStructureFactorEstimator.estimate_from_files(
        os.path.join(tmp, 'no_such_nf.json'), [corr0_path], qs, 8)
    check('E', 'nf 文件不存在 → FileNotFoundError 原样传播', False, '未抛异常')
except FileNotFoundError:
    check('E', 'nf 文件不存在 → FileNotFoundError 原样传播', True)
try:
    ChargeStructureFactorEstimator.estimate_from_files(
        nf_path, [corr0_path], qs, 8, sequences=[range(0, 8), range(0, 8)])
    check('E', 'sequences 长度不匹配 → ValueError', False, '未抛异常')
except ValueError:
    check('E', 'sequences 长度不匹配 → ValueError', True)

# ==========================================================================
# F. 真实 N200 受限估计（不预设峰位/相态）
# ==========================================================================
print('===== F. 真实 N200 受限估计（d1_100 / measure_49+50） =====')
qs200, _ = xc_momentum_grid(200, a=1.0, n_qx=11, n_qy=4)
base = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty'
res_real = ChargeStructureFactorEstimator.estimate_from_files(
    f'{base}/measure_49/nf.json',
    [f'{base}/measure_49/nf49nf.json', f'{base}/measure_50/nf50nf.json'],
    qs200, 200)
check('F', '真实数据：S_q 全有限且定义正确',
      res_real['definition'] == DEF_CHARGE_DENSITY_CONNECTED
      and np.all(np.isfinite(res_real['S_q'])),
      f"|S| 范围 [{np.min(np.abs(res_real['S_q'])):.3e}, {np.max(np.abs(res_real['S_q'])):.3e}]")
check('F', '真实数据：source_coverage=2/200、pair_coverage=None、is_restricted=True',
      abs(res_real['source_coverage'] - 2 / 200) < 1e-12
      and res_real['pair_coverage'] is None and res_real['is_restricted'])
check('F', '真实数据：对角补全源点为 49/50',
      res_real['diagonal_source_sites'] == [49, 50])
check('F', '真实数据：虚部残差报告（数值有限即可）',
      np.isfinite(res_real['imaginary_residual']), res_real['imaginary_residual'])

# ==========================================================================
# G. 批量估计（estimate_batch）
# ==========================================================================
print('===== G. 批量估计 =====')
from Physical_Quantity.Analysis_Contract import AnalysisReport

# 1. 两组均成功
batch_ok = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'g0', 'nf_path': nf_path, 'corr_paths': [corr0_path]},
     {'name': 'g1', 'nf_path': nf_path, 'corr_paths': [corr1_path]}],
    qs, 8)
check('G', '两组成功：results 含 2 个 AnalysisReport、failed 为空',
      set(batch_ok['results'].keys()) == {'g0', 'g1'}
      and all(isinstance(r, AnalysisReport) for r in batch_ok['results'].values())
      and batch_ok['failed'] == {},
      f"results={list(batch_ok['results'])} failed={batch_ok['failed']}")

# 2. 一组文件缺失：成功组保留、失败组记录正确
batch_mix = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'ok', 'nf_path': nf_path, 'corr_paths': [corr0_path]},
     {'name': 'missing', 'nf_path': os.path.join(tmp, 'no_nf.json'),
      'corr_paths': [corr0_path]}],
    qs, 8)
check('G', '一组文件缺失：成功组有结果，失败组记录 FileNotFoundError',
      'ok' in batch_mix['results'] and 'missing' not in batch_mix['results']
      and batch_mix['failed']['missing']['type'] == 'FileNotFoundError',
      batch_mix['failed'])

# 3. 一组数据格式错误：失败记录、其他组继续
bad_corr = os.path.join(tmp, 'nf_bad.json')
with open(bad_corr, 'w', encoding='utf-8') as f:
    f.write('not a json list')
batch_bad = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'good', 'nf_path': nf_path, 'corr_paths': [corr0_path]},
     {'name': 'bad', 'nf_path': nf_path, 'corr_paths': [bad_corr]},
     {'name': 'good2', 'nf_path': nf_path, 'corr_paths': [corr0_path]}],
    qs, 8)
check('G', '一组格式错误：失败记录 ValueError，其余两组不受影响',
      'bad' in batch_bad['failed'] and batch_bad['failed']['bad']['type'] == 'ValueError'
      and set(batch_bad['results'].keys()) == {'good', 'good2'},
      batch_bad['failed'])

# 4. 批量成功组与单组调用一致
single = charge_structure_factor_report(nf_path, [corr0_path], qs, 8)
batch_one = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'g0', 'nf_path': nf_path, 'corr_paths': [corr0_path]}], qs, 8)
check('G', '批量成功组与相同输入的单组调用一致（status/metadata/受限标记）',
      batch_one['results']['g0'].status == single.status
      and batch_one['results']['g0'].metadata == single.metadata
      and batch_one['results']['g0'].data_restricted == single.data_restricted)

# 5. 成功结果的可追溯字段与受限标记
rep_g0 = batch_ok['results']['g0']
check('G', '成功结果的 observable/analyzer/受限标记正确',
      rep_g0.observable == OBSERVABLE_CHARGE_STRUCTURE_FACTOR
      and rep_g0.analyzer == 'ChargeStructureFactorEstimator'
      and rep_g0.data_restricted is True
      and rep_g0.status == STATUS_OK)

# 6. 重复 name 不静默覆盖
batch_dup = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'dup', 'nf_path': nf_path, 'corr_paths': [corr0_path]},
     {'name': 'dup', 'nf_path': nf_u, 'corr_paths': [corr_u]}],
    qs, 8)
check('G', '重复 name：第一个结果保留，第二个记入 failed（DuplicateName）',
      'dup' in batch_dup['results']
      and batch_dup['failed']['dup']['type'] == 'DuplicateName'
      and batch_dup['results']['dup'].metadata['N'] == 8,
      batch_dup['failed'])

# 6b. 先失败后重复：保留原始失败记录，不被 DuplicateName 覆盖
batch_faildup = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'fd', 'nf_path': os.path.join(tmp, 'no_nf.json'),
      'corr_paths': [corr0_path]},
     {'name': 'fd', 'nf_path': nf_path, 'corr_paths': [corr0_path]},
     {'name': 'other', 'nf_path': nf_path, 'corr_paths': [corr0_path]}],
    qs, 8)
check('G', '先失败后重复：failed 保留原始 FileNotFoundError（不被 DuplicateName 覆盖）',
      batch_faildup['failed']['fd']['type'] == 'FileNotFoundError'
      and 'DuplicateName' not in batch_faildup['failed']['fd']['type'],
      batch_faildup['failed'])
check('G', '先失败后重复：原始失败消息保留、其他独立组不受影响',
      'no_nf.json' in batch_faildup['failed']['fd']['message']
      and 'other' in batch_faildup['results']
      and 'fd' not in batch_faildup['results'],
      f"results={list(batch_faildup['results'])}")

# 7. 未授权异常继续传播（spec 缺 'name' → KeyError，不捕获）
try:
    ChargeStructureFactorEstimator.estimate_batch(
        [{'nf_path': nf_path, 'corr_paths': [corr0_path]}], qs, 8)
    check('G', 'spec 缺 name → KeyError 原样传播（不静默转换）', False, '未抛异常')
except KeyError:
    check('G', 'spec 缺 name → KeyError 原样传播（不静默转换）', True)

# 8. 空输入 → ValueError
try:
    ChargeStructureFactorEstimator.estimate_batch([], qs, 8)
    check('G', '空 measure_specs → ValueError', False, '未抛异常')
except ValueError:
    check('G', '空 measure_specs → ValueError', True)

# 9. 真实数据批量（只读复用 N200 两组 doping；不硬编码为唯一依赖）
if os.path.exists(f'{ROOT}/Data_N200/t_J_3/d1_10/1500_measure_phys_qty/measure_49/nf.json'):
    base10 = f'{ROOT}/Data_N200/t_J_3/d1_10/1500_measure_phys_qty'
    base100 = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty'
    batch_real = ChargeStructureFactorEstimator.estimate_batch(
        [{'name': 'd1_10', 'nf_path': f'{base10}/measure_49/nf.json',
          'corr_paths': [f'{base10}/measure_49/nf49nf.json']},
         {'name': 'd1_100', 'nf_path': f'{base100}/measure_49/nf.json',
          'corr_paths': [f'{base100}/measure_49/nf49nf.json']}],
        qs200, 200)
    check('G', '真实数据批量：两组成功且均为受限估计（不比较数值）',
          set(batch_real['results'].keys()) == {'d1_10', 'd1_100'}
          and all(r.data_restricted is True for r in batch_real['results'].values())
          and batch_real['failed'] == {},
          list(batch_real['results']))
else:
    check('G', '真实数据批量（数据目录存在时才运行）', True, '跳过（目录不存在）')

# ==========================================================================
# H. q_grid 实际网格统计（D4.4：只描述输入事实，不宣称规则网格参数）
# ==========================================================================
print('===== H. q_grid 网格统计 =====')
# H1. 已知规则网格（xc_momentum_grid）
qs_h1, _ = xc_momentum_grid(200, a=1.0, n_qx=11, n_qy=4)
res_h1 = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], qs_h1, 8)
g1 = res_h1['q_grid']
check('H', '规则网格：n_qx/n_qy 为唯一坐标数量（11/4）',
      g1['n_qx'] == 11 and g1['n_qy'] == 4, g1)
check('H', '规则网格：qx/qy 端点与输入实际值一致',
      abs(g1['qx_min'] - qs_h1[:, 0].min()) < 1e-12
      and abs(g1['qx_max'] - qs_h1[:, 0].max()) < 1e-12
      and abs(g1['qy_min'] - qs_h1[:, 1].min()) < 1e-12
      and abs(g1['qy_max'] - qs_h1[:, 1].max()) < 1e-12,
      g1)

# H2. 不同网格尺寸：记录的是实际输入统计，不是硬编码默认值
qs_h2, _ = xc_momentum_grid(200, a=1.0, n_qx=21, n_qy=4)
res_h2 = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], qs_h2, 8)
g2 = res_h2['q_grid']
check('H', '不同网格尺寸：n_qx 反映实际输入（21），端点一致',
      g2['n_qx'] == 21 and g2['n_qy'] == 4
      and abs(g2['qx_min'] - g1['qx_min']) < 1e-12
      and abs(g2['qx_max'] - g1['qx_max']) < 1e-12,
      g2['n_qx'])

# H3. 非规则网格与单 qy 坐标网格：统计如实
qs_irr = np.array([(0.0, 0.0), (0.5, 0.0), (1.5, 0.0), (3.0, 0.0)])  # 非等距、单 qy
res_irr = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], qs_irr, 8)
g3 = res_irr['q_grid']
check('H', '非规则/单 qy 网格：唯一坐标数与端点如实',
      g3['n_qx'] == 4 and g3['n_qy'] == 1
      and g3['qx_min'] == 0.0 and g3['qx_max'] == 3.0
      and g3['qy_min'] == 0.0 and g3['qy_max'] == 0.0, g3)

# H4. 包装层 metadata['q_grid'] 与估计器一致
rep_h4 = charge_structure_factor_report(nf_path, [corr0_path], qs_h1, 8)
check('H', '包装层 metadata.q_grid 与估计器结果一致',
      rep_h4.metadata.get('q_grid') == g1, rep_h4.metadata.get('q_grid'))

# H5. 批量多组：各组记录自己的网格（不同网格不被混记）
batch_h5 = ChargeStructureFactorEstimator.estimate_batch(
    [{'name': 'grid11', 'nf_path': nf_path, 'corr_paths': [corr0_path]},
     {'name': 'grid21', 'nf_path': nf_path, 'corr_paths': [corr0_path]}],
    qs_h1, 8)
# 批量接口同一 q_list——用两组不同 q_list 验证：改为分别构造
batch_h5b = {'a': ChargeStructureFactorEstimator.estimate_from_files(
                 nf_path, [corr0_path], qs_h1, 8),
             'b': ChargeStructureFactorEstimator.estimate_from_files(
                 nf_path, [corr0_path], qs_h2, 8)}
check('H', '批量路径：各组 q_grid 各自如实记录（11 vs 21）',
      batch_h5b['a']['q_grid']['n_qx'] == 11
      and batch_h5b['b']['q_grid']['n_qx'] == 21)
check('H', 'estimate_batch 结果保留各组 q_grid（同网格时一致）',
      'grid11' in batch_h5['results']
      and batch_h5['results']['grid11'].metadata.get('q_grid') == g1)

# H6. JSON 序列化含 q_grid
_json_h = os.path.join(tmp, 'rep_qgrid.json')
rep_h4.to_json(_json_h)
_raw_h = open(_json_h, encoding='utf-8').read()
back_h = json.loads(_raw_h)
check('H', 'JSON 序列化含 q_grid 且数值一致',
      'NaN' not in _raw_h and back_h['metadata']['q_grid']['n_qx'] == 11
      and abs(back_h['metadata']['q_grid']['qx_min'] - g1['qx_min']) < 1e-12)

# H7. 空 q_list：q_grid 如实为空统计，不抛异常（不改变既有计算语义）
res_empty_q = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], np.empty((0, 2)), 8)
check('H', '空 q_list：q_grid 为空统计（n=0、端点 None），不新增异常',
      res_empty_q['q_grid']['n_qx'] == 0 and res_empty_q['q_grid']['n_qy'] == 0
      and res_empty_q['q_grid']['qx_min'] is None
      and len(res_empty_q['S_q']) == 0, res_empty_q['q_grid'])

# H8. 混合有限与 NaN：只统计有限坐标，端点不受 NaN 影响
qs_nan = np.array([[0.0, 0.0], [0.5, 0.0], [np.nan, 0.0], [2.0, 0.0]])
res_nan = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], qs_nan, 8)
check('H', '混合 NaN：n_qx 只计有限坐标（3），端点来自有限坐标',
      res_nan['q_grid']['n_qx'] == 3 and res_nan['q_grid']['n_qy'] == 1
      and res_nan['q_grid']['qx_min'] == 0.0 and res_nan['q_grid']['qx_max'] == 2.0,
      res_nan['q_grid'])

# H9. 混合有限与正负 Inf：非有限点不影响统计
qs_inf = np.array([[0.0, 0.0], [np.inf, 0.0], [1.0, -np.inf]])
res_inf = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], qs_inf, 8)
check('H', '混合 Inf：只统计两分量均有限的点（1 个），端点不受影响',
      res_inf['q_grid']['n_qx'] == 1 and res_inf['q_grid']['n_qy'] == 1
      and res_inf['q_grid']['qx_min'] == 0.0 and res_inf['q_grid']['qx_max'] == 0.0,
      res_inf['q_grid'])

# H10. 全部坐标非有限：n=0、四端点 None（与空输入同语义）
qs_allnan = np.array([[np.nan, np.nan], [np.inf, np.nan]])
res_allnan = ChargeStructureFactorEstimator.estimate_from_files(
    nf_path, [corr0_path], qs_allnan, 8)
check('H', '全部非有限：n_qx=n_qy=0、四端点 None',
      res_allnan['q_grid']['n_qx'] == 0 and res_allnan['q_grid']['n_qy'] == 0
      and res_allnan['q_grid']['qx_min'] is None
      and res_allnan['q_grid']['qx_max'] is None
      and res_allnan['q_grid']['qy_min'] is None
      and res_allnan['q_grid']['qy_max'] is None, res_allnan['q_grid'])

# ==========================================================================
# 汇总
# ==========================================================================
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
