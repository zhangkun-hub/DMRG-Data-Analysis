# -*- coding: utf-8 -*-
"""
超导配对结构因子测试（Phase C 第一阶段：SCPairStructureFactorEstimator）

独立参考实现（ref_sum）与正式实现不共享求和代码（坐标差、复指数、
归一化均独立重写）。覆盖：
- A. 已知动量峰位的合成数据 + 独立参考实现逐点对照
- B. 零关联 → S(q)=0 全网格
- C. 实数关联 → P(−q)=conj(P(q))（作为需验证的性质，不作假设）；
     复数关联 → 明确拒绝（诊断信息）
- D. 相位/符号约定：已知中点坐标的期望相位精确值
- E. 归一化 1/N 缩放与求和范围
- F. 键方向区分（不同目标键集 + direction 字段）
- G. 通道目标键集合不一致 → 禁止合成（ValueError），逐通道仍可用
- H. 无效输入与边界：空记录 / N 非 8 倍数 / 非法方向 / 非法通道 /
     空与含 NaN 的动量网格 / 复数记录值 / 参考键不一致 / 索引越界 /
     目录名非法 / 目录缺失 / 文件缺失 / 文件内参考键与目录名不一致
- I. 与公共核数值一致性（同一整数 ID 编码输入直调 compute_structure_factor）
- J. 确定性（两次调用逐位一致）
- K. 两条计算路径等价（合成后算 SF ≡ 通道 SF 线性组合，目标键集合
     一致条件下）
- L. 合成公式语义（a=d、b=c → <Δ†Δ> = a−b）
- M. 真实 N200 d1_100 数据受限估计（不预设峰位、不判定对称性/相态）

注：xc_momentum_grid 的 q 顺序为 qx 外层 × qy 内层，qy=0 行位于
索引 j*n_qy（本测试按此索引取 qy=0 行的数值）。

合成数据经临时文件写入系统临时目录（不落仓库）。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_sc_pair_structure_factor.py
"""
import json
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Physical_Data_Record import SuperCorrRecord
from Physical_Quantity.Spin_Structure_Factor import (
    compute_structure_factor, xc_momentum_grid)
from Physical_Quantity.Super_Correlation_Process import SuperCorrProcessor
from Physical_Quantity.SC_Pair_Structure_Factor import (
    SCPairStructureFactorEstimator, DEF_SC_PAIR, VALID_DIRECTIONS)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def make_recs(ref_bond, target_values):
    """[(k, l, v)] → list[SuperCorrRecord]（合成测试用，distance 不参与计算）"""
    return [SuperCorrRecord(tuple(ref_bond), (k, l), 0.0, float(v))
            for (k, l, v) in target_values]


def ref_sum(geometry, ref_bond, pairs, q, N):
    """独立参考实现：P(q) = (1/N) Σ exp[i q·(R_ref − R_t)]·C_t，
    中点坐标与求和顺序独立重写（不共享正式实现代码）"""
    def mid(b):
        (x1, y1) = geometry.coords[b[0]]
        (x2, y2) = geometry.coords[b[1]]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    Rr = mid(ref_bond)
    S = 0.0 + 0.0j
    for (k, l, v) in pairs:
        Rt = mid((k, l))
        S += v * np.exp(1j * (q[0] * (Rr[0] - Rt[0])
                              + q[1] * (Rr[1] - Rt[1])))
    return S / N


g16 = HoneycombXCGeometry(16, 1.0)
g64 = HoneycombXCGeometry(64, 1.0)
tmp = tempfile.mkdtemp(prefix='sc_pair_sf_')

# ==========================================================================
# A. 已知峰位合成数据 + 独立参考实现
# ==========================================================================
# refbond (0,4) 中点 (0.5,0)；目标键 (8+8m,12+8m) 中点位移 3(m+1)
# C_t = cos(q0·3(m+1)) → |P(q)| 在 qx=±q0 成峰（q0=0.6 在 [−π/3,π/3] 内）
q0 = 0.6
qs, meta = xc_momentum_grid(64, a=1.0, n_qx=41, n_qy=4)
targets = [((8 + 8 * m) % 64, (12 + 8 * m) % 64,
            float(np.cos(q0 * 3.0 * (m + 1)))) for m in range(7)]
recs_a = make_recs((0, 4), targets)
res_a = SCPairStructureFactorEstimator.estimate_from_records(
    recs_a, 'A', qs, 64)
qx_axis = np.linspace(-np.pi / 3, np.pi / 3, 41)
for n, q in enumerate(qs):
    expect = ref_sum(g64, (0, 4), targets, q, 64)
    if abs(res_a['S_q'][n] - expect) > 1e-12:
        check('A', '独立参考实现逐点对照',
              False, f'q={q} 模块={res_a["S_q"][n]:.15g} 参考={expect:.15g}')
        break
else:
    check('A', '独立参考实现逐点对照（164 点全部 ≤1e-12）', True)
S_abs = np.abs(res_a['S_q'])
peak_idx = int(np.argmax(S_abs))
peak_qx = res_a['q'][peak_idx][0]
# 期望峰在 ±q0（对称双峰，argmax 按网格顺序取先出现者——不限定符号）
dq = qx_axis[1] - qx_axis[0]
check('A', '峰位在 qx=±q0 附近（±2 网格）',
      min(abs(peak_qx - q0), abs(peak_qx + q0)) <= 2 * dq,
      f'peak_qx={peak_qx:.4f} 期望 ±{q0}')
i_pos = int(np.argmin(np.abs(qx_axis - q0)))
i_neg = int(np.argmin(np.abs(qx_axis + q0)))
s_pos = S_abs[i_pos * 4]   # qy=0 行（q 顺序：qx 外层 × qy 内层）
s_neg = S_abs[i_neg * 4]
check('A', '±q0 峰高对称（|S(+q0)|≈|S(−q0)|，≤1e-10）',
      abs(s_pos - s_neg) <= 1e-10 * max(s_pos, s_neg),
      f'{s_pos:.12g} vs {s_neg:.12g}')
check('A', 'direction 字段与受限语义',
      res_a['direction'] == 'A' and res_a['channel'] == 'combined'
      and res_a['definition'] == DEF_SC_PAIR
      and res_a['is_restricted'] is True
      and res_a['expected_pair_count'] is None
      and res_a['pair_coverage'] is None
      and res_a['source_coverage'] is None
      and res_a['ref_bond'] == [0, 4]
      and res_a['target_bond_count'] == 7
      and len(res_a['target_bonds']) == 7
      and res_a['meta']['position_basis'] == 'bond_midpoint')

# ==========================================================================
# B. 零关联
# ==========================================================================
recs_zero = make_recs((0, 4), [(8, 12, 0.0), (9, 13, 0.0)])
res_b = SCPairStructureFactorEstimator.estimate_from_records(
    recs_zero, 'B', qs, 16)
check('B', '零关联 → S(q)=0 全网格',
      np.all(np.abs(res_b['S_q']) == 0.0))

# ==========================================================================
# C. 共轭性质（验证而非假设）与复数拒绝
# ==========================================================================
q_pairs = np.array([[0.5, 0.0], [-0.5, 0.0], [0.3, 0.2], [-0.3, -0.2],
                    [0.0, 0.5], [0.0, -0.5]])
recs_c = make_recs((0, 4), [(8, 12, 1.5), (9, 13, -0.8), (10, 14, 0.3)])
res_c = SCPairStructureFactorEstimator.estimate_from_records(
    recs_c, 'C', q_pairs, 16)
c_ok = True
for i in range(0, len(q_pairs), 2):
    if abs(res_c['S_q'][i] - np.conj(res_c['S_q'][i + 1])) > 1e-14:
        c_ok = False
check('C', '实数关联 → P(−q)=conj(P(q))（全部 3 对，≤1e-14）', c_ok)
recs_cplx = make_recs((0, 4), [(8, 12, 1.0)])
recs_cplx[0].sup_corr = 1.0 + 2.0j
try:
    SCPairStructureFactorEstimator.estimate_from_records(
        recs_cplx, 'A', np.array([[0.5, 0.0]]), 16)
    check('C', '复数关联值被明确拒绝', False)
except ValueError as e:
    check('C', '复数关联值被明确拒绝（ValueError 诊断）',
          '复数' in str(e), str(e)[:60])

# ==========================================================================
# D. 相位/符号约定精确值
# ==========================================================================
# ref (0,4) 中点 (0.5,0)，target (8,12) 中点 (3.5,0)，v=2
# 期望 S(q=(0.7,0)) = (2/16)·exp[i·0.7·(0.5−3.5)] = (2/16)·e^{−2.1i}
res_d = SCPairStructureFactorEstimator.estimate_from_records(
    make_recs((0, 4), [(8, 12, 2.0)]), 'A',
    np.array([[0.7, 0.0]]), 16)
expect_d = 2.0 / 16.0 * np.exp(-1j * 2.1)
check('D', '相位 = exp[i q·(R_ref − R_t)]（实部与虚部 ≤1e-14）',
      abs(res_d['S_q'][0] - expect_d) < 1e-14,
      f'{res_d["S_q"][0]:.15g} 期望 {expect_d:.15g}')

# ==========================================================================
# E. 归一化与求和范围
# ==========================================================================
recs_e = make_recs((0, 4), [(8, 12, 2.0), (9, 13, 1.0)])
res_e16 = SCPairStructureFactorEstimator.estimate_from_records(
    recs_e, 'A', np.array([[0.7, 0.0]]), 16)
res_e32 = SCPairStructureFactorEstimator.estimate_from_records(
    recs_e, 'A', np.array([[0.7, 0.0]]), 32)
check('E', '归一化 1/N（N=32 为 N=16 的一半，≤1e-15）',
      abs(res_e32['S_q'][0] - res_e16['S_q'][0] / 2) < 1e-15)
expect_e = ref_sum(g16, (0, 4),
                   [(8, 12, 2.0), (9, 13, 1.0)], (0.7, 0.0), 16)
check('E', '求和范围 = 仅提供的目标键（与独立参考一致）',
      abs(res_e16['S_q'][0] - expect_e) < 1e-13)

# ==========================================================================
# F. 键方向区分
# ==========================================================================
recs_fa = make_recs((0, 4), [(8, 12, 1.0)])
recs_fb = make_recs((0, 4), [(5, 1, 1.0)])   # 不同目标键（不同中点）
res_fa = SCPairStructureFactorEstimator.estimate_from_records(
    recs_fa, 'A', np.array([[0.7, 0.0]]), 16)
res_fb = SCPairStructureFactorEstimator.estimate_from_records(
    recs_fb, 'B', np.array([[0.7, 0.0]]), 16)
check('F', '不同目标键集 + direction 字段独立标注',
      res_fa['direction'] == 'A' and res_fb['direction'] == 'B'
      and res_fa['target_bonds'] != res_fb['target_bonds']
      and abs(res_fa['S_q'][0] - res_fb['S_q'][0]) > 1e-12)

# ==========================================================================
# G. 通道目标键集合不一致 → 禁止合成；逐通道仍可用
# ==========================================================================
recs_ga = make_recs((0, 4), [(8, 12, 1.0), (9, 13, 2.0)])
recs_gb = make_recs((0, 4), [(8, 12, 3.0), (9, 13, 4.0)])
recs_gc = make_recs((0, 4), [(8, 12, 5.0), (9, 13, 6.0)])
recs_gd = make_recs((0, 4), [(8, 12, 7.0)])   # 缺 (9,13)
try:
    SCPairStructureFactorEstimator.estimate_from_channels(
        recs_ga, recs_gb, recs_gc, recs_gd, 'A', np.array([[0.7, 0.0]]), 16)
    check('G', '通道目标键集合不一致 → 禁止合成（ValueError）', False)
except ValueError as e:
    check('G', '通道目标键集合不一致 → 禁止合成（ValueError 含逐通道计数）',
          'a' in str(e) and '2' in str(e) and '1' in str(e), str(e)[:80])
res_gd_alone = SCPairStructureFactorEstimator.estimate_channel_from_records(
    recs_gd, 'd', 'A', np.array([[0.7, 0.0]]), 16)
check('G', '逐通道结果仍可单独取得（channel="d"）',
      res_gd_alone['channel'] == 'd'
      and res_gd_alone['definition'] == 'sc_pair_channel_d'
      and res_gd_alone['target_bond_count'] == 1)

# ==========================================================================
# H. 无效输入与边界
# ==========================================================================
try:
    SCPairStructureFactorEstimator.estimate_from_records(
        [], 'A', np.array([[0.7, 0.0]]), 16)
    check('H', '空记录列表 → ValueError', False)
except ValueError:
    check('H', '空记录列表 → ValueError', True)
try:
    SCPairStructureFactorEstimator.estimate_from_records(
        make_recs((0, 4), [(8, 12, 1.0)]), 'A',
        np.array([[0.7, 0.0]]), 10)
    check('H', 'N 非 8 倍数 → ValueError（几何校验）', False)
except ValueError:
    check('H', 'N 非 8 倍数 → ValueError（几何校验）', True)
try:
    SCPairStructureFactorEstimator.estimate_from_records(
        make_recs((0, 4), [(8, 12, 1.0)]), 'X',
        np.array([[0.7, 0.0]]), 16)
    check('H', '非法 direction → ValueError', False)
except ValueError as e:
    check('H', '非法 direction → ValueError', VALID_DIRECTIONS[0] in str(e))
try:
    SCPairStructureFactorEstimator.estimate_channel_from_records(
        make_recs((0, 4), [(8, 12, 1.0)]), 'z', 'A',
        np.array([[0.7, 0.0]]), 16)
    check('H', '非法 channel → ValueError', False)
except ValueError:
    check('H', '非法 channel → ValueError', True)
res_hq = SCPairStructureFactorEstimator.estimate_from_records(
    make_recs((0, 4), [(8, 12, 1.0)]), 'A', np.empty((0, 2)), 16)
check('H', '空 q_list → 空 S_q + q_grid 空统计（不抛异常）',
      len(res_hq['S_q']) == 0 and res_hq['q_grid']['n_qx'] == 0
      and res_hq['q_grid']['qx_min'] is None)
qs_nan = np.array([[0.0, 0.0], [0.5, 0.0], [np.nan, 0.0], [2.0, 0.0]])
res_hn = SCPairStructureFactorEstimator.estimate_from_records(
    make_recs((0, 4), [(8, 12, 1.0)]), 'A', qs_nan, 16)
check('H', '混合 NaN 动量网格 → q_grid 只计有限坐标（n_qx=3）',
      res_hn['q_grid']['n_qx'] == 3 and res_hn['q_grid']['n_qy'] == 1)
recs_h1 = make_recs((0, 4), [(8, 12, 1.0)])
recs_h1.append(SuperCorrRecord((4, 6), (8, 12), 0.0, 1.0))   # 参考键不一致
try:
    SCPairStructureFactorEstimator.estimate_from_records(
        recs_h1, 'A', np.array([[0.7, 0.0]]), 16)
    check('H', 'record 参考键不一致 → ValueError', False)
except ValueError:
    check('H', 'record 参考键不一致 → ValueError', True)
recs_h2 = make_recs((0, 4), [(16, 12, 1.0)])   # 目标键越界
try:
    SCPairStructureFactorEstimator.estimate_from_records(
        recs_h2, 'A', np.array([[0.7, 0.0]]), 16)
    check('H', '目标键索引越界 → ValueError', False)
except ValueError:
    check('H', '目标键索引越界 → ValueError', True)
recs_nan = make_recs((0, 4), [(8, 12, np.nan)])
res_hnan = SCPairStructureFactorEstimator.estimate_from_records(
    recs_nan, 'A', np.array([[0.7, 0.0]]), 16)
check('H', 'NaN 关联值 → 公共核跳过并计数 invalid_values=1',
      res_hnan['invalid_values'] == 1)

# estimate_from_files 目录/文件边界（合成临时文件）
d_dir = os.path.join(tmp, 'super_corr_1_2')
os.makedirs(d_dir)
for ch in 'abcd':
    with open(os.path.join(d_dir, f'scsA{ch}_fix_refbond_1_2.json'),
              'w', encoding='utf-8') as f:
        f.write(json.dumps([[[1, 2, 8, 12], [float(1.0), 0]]]))
res_hf = SCPairStructureFactorEstimator.estimate_from_files(
    'A', d_dir, np.array([[0.7, 0.0]]), 16)
check('H', 'estimate_from_files 合成文件端到端（含目录名交叉校验）',
      res_hf['record_source'] == 'full_direction_file'
      and res_hf['ref_bond'] == [1, 2]
      and res_hf['target_bond_count'] == 1
      and res_hf['channel_target_consistency'] is True)
try:
    SCPairStructureFactorEstimator.estimate_from_files(
        'A', os.path.join(tmp, 'no_such_dir'),
        np.array([[0.7, 0.0]]), 16)
    check('H', '参考键目录不存在 → FileNotFoundError', False)
except FileNotFoundError:
    check('H', '参考键目录不存在 → FileNotFoundError', True)
bad_dir = os.path.join(tmp, 'wrong_name')
os.makedirs(bad_dir)
try:
    SCPairStructureFactorEstimator.estimate_from_files(
        'A', bad_dir, np.array([[0.7, 0.0]]), 16)
    check('H', '目录名不符合 super_corr_{i}_{j} → ValueError', False)
except ValueError:
    check('H', '目录名不符合 super_corr_{i}_{j} → ValueError', True)
d_dir2 = os.path.join(tmp, 'super_corr_1_3')
os.makedirs(d_dir2)
with open(os.path.join(d_dir2, 'scsAa_fix_refbond_1_3.json'),
          'w', encoding='utf-8') as f:
    f.write(json.dumps([[[1, 2, 8, 12], [1.0, 0]]]))   # 文件内参考键与目录名不符
try:
    SCPairStructureFactorEstimator.estimate_from_files(
        'A', d_dir2, np.array([[0.7, 0.0]]), 16)
    check('H', '文件内参考键与目录名不一致 → ValueError', False)
except ValueError:
    check('H', '文件内参考键与目录名不一致 → ValueError', True)
d_dir3 = os.path.join(tmp, 'super_corr_1_4')
os.makedirs(d_dir3)
try:
    SCPairStructureFactorEstimator.estimate_from_files(
        'A', d_dir3, np.array([[0.7, 0.0]]), 16)
    check('H', '通道文件缺失 → FileNotFoundError 传播', False)
except FileNotFoundError:
    check('H', '通道文件缺失 → FileNotFoundError 传播', True)

# parse_direction_file 边界
bad_fmt = os.path.join(tmp, 'bad_fmt.json')
with open(bad_fmt, 'w', encoding='utf-8') as f:
    f.write(json.dumps([[[1, 2, 8], [1.0, 0]]]))   # 三元组而非四元组
try:
    SCPairStructureFactorEstimator.parse_direction_file(bad_fmt, 16)
    check('H', 'parse 格式错误 → ValueError', False)
except ValueError:
    check('H', 'parse 格式错误 → ValueError', True)
bad_idx = os.path.join(tmp, 'bad_idx.json')
with open(bad_idx, 'w', encoding='utf-8') as f:
    f.write(json.dumps([[[1, 2, 8, 16], [1.0, 0]]]))   # l=16 越界
try:
    SCPairStructureFactorEstimator.parse_direction_file(bad_idx, 16)
    check('H', 'parse 索引越界 → ValueError', False)
except ValueError:
    check('H', 'parse 索引越界 → ValueError', True)

# ==========================================================================
# I. 与公共核数值一致性（同一整数 ID 编码输入，直调公共核）
# ==========================================================================
recs_i = make_recs((0, 4), [(8, 12, 1.5), (9, 13, -0.8), (10, 14, 0.3)])
res_i = SCPairStructureFactorEstimator.estimate_from_records(
    recs_i, 'A', np.array([[0.7, 0.0], [0.2, 0.5]]), 16)
# 手工复现模块的整数 ID 编码后直调公共核
N = 16
bonds = [(0, 4), (8, 12), (9, 13), (10, 14)]
coords_i = {}
for b in bonds:
    (x1, y1) = g16.coords[b[0]]
    (x2, y2) = g16.coords[b[1]]
    coords_i[b[0] * N + b[1]] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
C_i = [((0 * N + 4), (8 * N + 12), 1.5), ((0 * N + 4), (9 * N + 13), -0.8),
       ((0 * N + 4), (10 * N + 14), 0.3)]
res_core = compute_structure_factor(
    coords_i, C_i, np.array([[0.7, 0.0], [0.2, 0.5]]), 16, DEF_SC_PAIR)
check('I', '与直调公共核逐位一致（S_q 完全相等）',
      np.array_equal(res_i['S_q'], res_core['S_q']))

# ==========================================================================
# J. 确定性
# ==========================================================================
res_j1 = SCPairStructureFactorEstimator.estimate_from_records(
    recs_i, 'A', np.array([[0.7, 0.0], [0.2, 0.5]]), 16)
res_j2 = SCPairStructureFactorEstimator.estimate_from_records(
    recs_i, 'A', np.array([[0.7, 0.0], [0.2, 0.5]]), 16)
check('J', '两次调用 S_q 逐位一致',
      np.array_equal(res_j1['S_q'], res_j2['S_q'])
      and res_j1['target_bonds'] == res_j2['target_bonds'])

# ==========================================================================
# K. 两条计算路径等价（目标键集合一致条件下）
# ==========================================================================
recs_ka = make_recs((0, 4), [(8, 12, 1.0), (9, 13, 2.0)])
recs_kb = make_recs((0, 4), [(8, 12, -3.0), (9, 13, 0.5)])
recs_kc = make_recs((0, 4), [(8, 12, 0.2), (9, 13, -1.0)])
recs_kd = make_recs((0, 4), [(8, 12, 4.0), (9, 13, 1.5)])
qk = np.array([[0.7, 0.0], [0.2, 0.5]])
res_k = SCPairStructureFactorEstimator.estimate_from_channels(
    recs_ka, recs_kb, recs_kc, recs_kd, 'A', qk, 16, include_channels=True)
combined_k = SuperCorrProcessor.Combined_super_correlation(
    recs_ka, recs_kb, recs_kc, recs_kd)
res_k2 = SCPairStructureFactorEstimator.estimate_from_records(
    combined_k, 'A', qk, 16)
check('K', '路径 1（合成后算 SF）≡ 路径 2（手工 Combined 后算 SF）逐位一致',
      np.array_equal(res_k['S_q'], res_k2['S_q']))
linear = (0.5 * (res_k['channel_sf']['a']['S_q']
                 - res_k['channel_sf']['b']['S_q']
                 - res_k['channel_sf']['c']['S_q']
                 + res_k['channel_sf']['d']['S_q']))
# 线性性成立但浮点结合顺序不同（合成值先线性组合再乘相位 vs 各通道
# 求和后线性组合）→ 机器精度级一致，不要求逐位相等
check('K', '合成 SF ≡ 0.5(P_a−P_b−P_c+P_d)（线性性，≤1e-13）',
      np.max(np.abs(res_k['S_q'] - linear)) <= 1e-13,
      f"max|Δ|={np.max(np.abs(res_k['S_q'] - linear)):.3e}")
check('K', 'channel_sf 四通道齐全且 channel_target_consistency=True',
      set(res_k['channel_sf']) == set('abcd')
      and res_k['channel_target_consistency'] is True)

# ==========================================================================
# L. 合成公式语义（a=d、b=c → <Δ†Δ> = a−b）
# ==========================================================================
recs_la = make_recs((0, 4), [(8, 12, 5.0), (9, 13, 3.0)])
recs_lb = make_recs((0, 4), [(8, 12, 1.0), (9, 13, -2.0)])
recs_lc = list(recs_lb)   # b = c
recs_ld = list(recs_la)   # a = d
combined_l = SuperCorrProcessor.Combined_super_correlation(
    recs_la, recs_lb, recs_lc, recs_ld)
expect_l = [0.5 * (5.0 - 1.0 - 1.0 + 5.0), 0.5 * (3.0 + 2.0 + 2.0 + 3.0)]
check('L', 'a=d、b=c 时合成值 = a−b（逐元素）',
      all(abs(r.sup_corr - e) < 1e-14
          for r, e in zip(combined_l, expect_l)),
      [round(r.sup_corr, 12) for r in combined_l])

# ==========================================================================
# M. 真实 N200 d1_100 数据受限估计（49_50/A）
# ==========================================================================
real_dir = os.path.join(ROOT, 'Data_N200', 't_J_3', 'd1_100',
                        '1500_measure_super_corr', 'super_corr_49_50')
if os.path.isdir(real_dir):
    qs_real, _ = xc_momentum_grid(200, a=1.0, n_qx=41, n_qy=4)
    res_m = SCPairStructureFactorEstimator.estimate_from_files(
        'A', real_dir, qs_real, 200)
    check('M', '真实 d1_100/49_50/A：全文件 74 条目标键、受限语义',
          res_m['target_bond_count'] == 74
          and res_m['is_restricted'] is True
          and res_m['expected_pair_count'] is None
          and res_m['record_source'] == 'full_direction_file'
          and res_m['ref_bond'] == [49, 50]
          and res_m['channel_target_consistency'] is True
          and len(res_m['S_q']) == len(qs_real)
          and res_m['q_grid']['n_qx'] == 41
          and res_m['q_grid']['n_qy'] == 4
          and len(res_m['channel_sf']) == 4
          and np.isfinite(res_m['imaginary_residual'])
          and res_m['invalid_values'] == 0,
          f"n={res_m['target_bond_count']} restricted={res_m['is_restricted']}")
else:
    check('M', '真实 d1_100/49_50 数据目录存在', False, real_dir)

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
