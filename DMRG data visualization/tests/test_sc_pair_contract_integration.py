# -*- coding: utf-8 -*-
"""
SC 配对结构因子契约集成测试（A1：SCPairStructureFactorEstimator →
run_analysis 路由 → AnalysisReport）

覆盖：
- 合成 4 通道方向文件（临时目录）→ run_analysis 路由返回 AnalysisReport；
  与直接调用 estimate_from_files 的 S_q 逐位一致（集成不改变物理计算）
- 契约字段：observable/analyzer 标识、data_restricted=True（受限语义）、
  metadata 的 SC 专属身份字段（direction/ref_bond/target_bond_count/
  channel_target_consistency/q_grid）
- include_channels=False 路径（payload 无 channel_sf）
- 错误路径：未知 observable / 不支持的 operation → ValueError（含支持
  列表）；缺参 → TypeError；config → TypeError；文件缺失
  FileNotFoundError 原样传播；通道目标键集合不一致 ValueError 原样传播；
  非法方向 ValueError 原样传播
- JSON 严格序列化（无 NaN/Infinity 字面量、默认无 payload）
- 不判定配对对称性或物理相态（报告无 classification/evidence）

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_sc_pair_contract_integration.py
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

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR, ANALYZER_SC_PAIR_SF,
)
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.SC_Pair_Structure_Factor import (
    SCPairStructureFactorEstimator)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ---- 合成数据：N=16、参考键 (0,1)、目标键 3 个，四通道目标键集合一致 ----
N = 16
A = 1.0
_Q = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5)]


def _write_channel(tmp_dir, key, ch, targets, values):
    path = os.path.join(
        tmp_dir, f'scs{key}{ch}_fix_refbond_0_1.json')
    data = [[[0, 1, k, l], [v, 0]] for (k, l), v in zip(targets, values)]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return path


def make_refbond_dir(tmp_root, targets=((2, 3), (4, 5), (6, 7)),
                     values_a=(4.0, 4.0, 4.0), values_b=(5.0, 5.0, 5.0),
                     values_c=(5.0, 5.0, 5.0), values_d=(4.0, 4.0, 4.0),
                     channels='abcd'):
    """创建 super_corr_0_1 目录与指定通道文件（a=d、b=c → <Δ†Δ> = a−b）。"""
    ref_dir = os.path.join(tmp_root, 'super_corr_0_1')
    os.makedirs(ref_dir, exist_ok=True)
    vals = {'a': values_a, 'b': values_b, 'c': values_c, 'd': values_d}
    for ch in channels:
        _write_channel(ref_dir, 'A', ch, targets, vals[ch])
    return ref_dir


# ==========================================================================
# A. 路由返回 AnalysisReport + 与直接调用逐位一致
# ==========================================================================
print('===== A. 路由与直调一致性 =====')
tmp_a = tempfile.mkdtemp(prefix='scpair_route_')
ref_dir = make_refbond_dir(tmp_a)
rep = run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                   direction='A', refbond_dir=ref_dir,
                   q_list=_Q, N=N, a=A)
direct = SCPairStructureFactorEstimator.estimate_from_files(
    'A', ref_dir, _Q, N, A)
check('A', '路由返回 AnalysisReport', isinstance(rep, AnalysisReport))
check('A', 'observable/analyzer 标识正确',
      rep.observable == OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR
      and rep.analyzer == ANALYZER_SC_PAIR_SF)
check('A', 'status=ok 且 classification/evidence 为空（不判物理相态）',
      rep.status == STATUS_OK and rep.classification is None
      and rep.evidence is None)
check('A', 'S_q 与直接调用逐位一致（集成不改变物理计算）',
      np.array_equal(np.asarray(rep.payload['S_q'], dtype=complex),
                     np.asarray(direct['S_q'], dtype=complex)),
      str(rep.payload['S_q'][:2]))
check('A', 'payload 为 estimator 原始 dict（含 q/definition 等）',
      isinstance(rep.payload, dict) and rep.payload.get('definition')
      == 'sc_pair' and 'q' in rep.payload)

# ==========================================================================
# B. 契约字段（受限语义 / metadata 身份字段）
# ==========================================================================
print('===== B. 契约字段 =====')
check('B', 'data_restricted=True（受限估计语义直通）',
      rep.data_restricted is True)
md = rep.metadata
check('B', 'metadata 含 direction/ref_bond/target_bond_count（SC 专属身份）',
      md.get('direction') == 'A' and md.get('ref_bond') == [0, 1]
      and md.get('target_bond_count') == 3, {k: md.get(k) for k in
                                             ('direction', 'ref_bond',
                                              'target_bond_count')})
check('B', 'metadata 含 channel_target_consistency=True 与 q_grid 摘要',
      md.get('channel_target_consistency') is True
      and md.get('q_grid', {}).get('n_qx') == 3)
check('B', 'payload 含 channel_sf 四通道（include_channels 默认 True）',
      isinstance(rep.payload.get('channel_sf'), dict)
      and set(rep.payload['channel_sf']) == {'a', 'b', 'c', 'd'})
check('B', 'payload 含 target_bonds 全量目标键（可追溯）',
      rep.payload.get('target_bonds') == [[2, 3], [4, 5], [6, 7]])

rep_noc = run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                       direction='A', refbond_dir=ref_dir,
                       q_list=_Q, N=N, a=A, include_channels=False)
check('B', 'include_channels=False：payload 无 channel_sf，一致性标记仍为 True'
      '（合成本身通过校验）',
      'channel_sf' not in rep_noc.payload
      and rep_noc.metadata.get('channel_target_consistency') is True)

# ==========================================================================
# C. 错误路径（清晰报错 / 原样传播，不伪装成功）
# ==========================================================================
print('===== C. 错误路径 =====')
try:
    run_analysis('unknown_observable_xyz')
    check('C', '未知 observable → ValueError（含支持列表）', False, '未抛异常')
except ValueError as e:
    check('C', '未知 observable → ValueError（含支持列表）',
          '支持' in str(e), str(e)[:60])
try:
    run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR, operation='decay',
                 direction='A', refbond_dir=ref_dir, q_list=_Q, N=N)
    check('C', '不支持的 operation → ValueError', False, '未抛异常')
except ValueError as e:
    check('C', '不支持的 operation → ValueError', '支持' in str(e), str(e)[:60])
try:
    run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR)   # 缺 direction 等
    check('C', '缺必需参数 → TypeError（不静默忽略）', False, '未抛异常')
except TypeError:
    check('C', '缺必需参数 → TypeError（不静默忽略）', True)
try:
    run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                 direction='A', refbond_dir=ref_dir, q_list=_Q, N=N,
                 config={'x': 1})
    check('C', '不支持 config 的链路收到 config → TypeError', False, '未抛异常')
except TypeError as e:
    check('C', '不支持 config 的链路收到 config → TypeError',
          '不支持 config' in str(e))
try:
    run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR, direction='D',
                 refbond_dir=ref_dir, q_list=_Q, N=N)
    check('C', '非法方向 → ValueError 原样传播', False, '未抛异常')
except ValueError as e:
    check('C', '非法方向 → ValueError 原样传播', 'direction' in str(e),
          str(e)[:60])

# 文件缺失：只写 3 个通道文件
tmp_miss = tempfile.mkdtemp(prefix='scpair_miss_')
ref_dir_miss = make_refbond_dir(tmp_miss, channels='abd')
try:
    run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                 direction='A', refbond_dir=ref_dir_miss,
                 q_list=_Q, N=N)
    check('C', '通道文件缺失 → FileNotFoundError 原样传播', False, '未抛异常')
except FileNotFoundError:
    check('C', '通道文件缺失 → FileNotFoundError 原样传播', True)

# 通道目标键集合不一致（c 通道目标键不同）
tmp_bad = tempfile.mkdtemp(prefix='scpair_bad_')
ref_dir_bad = os.path.join(tmp_bad, 'super_corr_0_1')
os.makedirs(ref_dir_bad)
_write_channel(ref_dir_bad, 'A', 'a', [(2, 3), (4, 5)], [4.0, 4.0])
_write_channel(ref_dir_bad, 'A', 'b', [(2, 3), (4, 5)], [5.0, 5.0])
_write_channel(ref_dir_bad, 'A', 'c', [(2, 3), (6, 7)], [5.0, 5.0])
_write_channel(ref_dir_bad, 'A', 'd', [(2, 3), (4, 5)], [4.0, 4.0])
try:
    run_analysis(OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                 direction='A', refbond_dir=ref_dir_bad,
                 q_list=_Q, N=N)
    check('C', '通道目标键集合不一致 → ValueError（禁止未经支持的合成）',
          False, '未抛异常')
except ValueError as e:
    check('C', '通道目标键集合不一致 → ValueError（禁止未经支持的合成）',
          '不一致' in str(e), str(e)[:70])

# ==========================================================================
# D. JSON 严格序列化
# ==========================================================================
print('===== D. JSON 序列化 =====')
tmp_d = tempfile.mkdtemp(prefix='scpair_json_')
json_path = os.path.join(tmp_d, 'report.json')
rep.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
back = json.loads(raw)
check('D', 'JSON：无 NaN/Infinity 字面量、默认无 payload、可解析读回',
      'NaN' not in raw and 'Infinity' not in raw
      and '"payload"' not in raw and back['status'] == 'ok'
      and back['observable'] == OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR
      and back['data_restricted'] is True)
check('D', 'JSON：metadata 身份字段读回一致',
      back['metadata']['ref_bond'] == [0, 1]
      and back['metadata']['direction'] == 'A')

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
