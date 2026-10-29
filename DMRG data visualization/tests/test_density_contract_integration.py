# -*- coding: utf-8 -*-
"""
局域密度报告契约集成测试（P1-G.1）。

覆盖：
- 子晶格划分：近邻表二部染色推导正确（A={0,2,5,7} mod 8）、全部
  近邻边异色、完整覆盖；非二部/不连通输入不伪造划分
- 均匀密度：逐格点向量与行/子晶格/晶胞分组摘要正确
- 已知子晶格调制：A/B 平均与预期一致
- 已知空间调制：逐格点向量完整保留（分组平均不掩盖细节）
- 边界隆起合成数据：boundary 元数据保留，报告不输出 Stripe 判定
- 契约序列化/回读（严格 JSON，payload 默认不序列化）
- 标准入口路由：run_analysis(local_density, 'profile') 成功；
  未知 operation 抛 ValueError；config 不支持抛 TypeError
- 失败路径：缺失文件 FileNotFoundError、畸形 JSON ValueError、
  格点数不匹配 ValueError、N 非 8 倍数 ValueError、索引缺失
  ValueError——均原样传播，不静默补默认值
- 证据层：密度报告 → observation（不是相级支持/反证），
  data_range 保留 N/lattice_geometry/boundary/unit_cell_size

全部合成数据（测试显式构造），不依赖真实数据偶然数值。

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_density_contract_integration.py
"""
import json
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, STATUS_FAILED, OBSERVABLE_LOCAL_DENSITY, ANALYZER_DENSITY)
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.Density_Process import DensityProcessor
from Physical_Quantity.Physical_Data_Record import DensityRecord
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, build_evidence_items,
    EVIDENCE_OBSERVATION, EVIDENCE_NOT_AVAILABLE)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


N, A = 200, 1.0
TMP = tempfile.mkdtemp(prefix='p1g1_test_')


def write_nf(name, vals):
    path = os.path.join(TMP, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[[i], [float(v), 0]] for i, v in enumerate(vals)], f)
    return path


# ======================================================================
# A. 子晶格划分（几何推导与验证）
# ======================================================================
print('===== A. 子晶格划分 =====')
g = HoneycombXCGeometry(N, A)
part = g.sub_lattice_partition()
lab = part['site_labels']
check('A', 'derivation 注明近邻表二部染色（可追溯，不硬编码）',
      part['derivation'] == 'bipartite_coloring_from_nearest_neighbor_table')
check('A', '染色完整覆盖 0..N-1',
      sorted(set(lab)) == ['A', 'B'] and len(lab) == N)
a8 = sorted({i % 8 for i in range(N) if lab[i] == 'A'})
b8 = sorted({i % 8 for i in range(N) if lab[i] == 'B'})
check('A', '二部染色与基元模板一致：A={0,2,5,7} mod 8、B={1,3,4,6} mod 8',
      a8 == [0, 2, 5, 7] and b8 == [1, 3, 4, 6], str((a8, b8)))
check('A', '全部最近邻边两端异色（二部性验证）',
      all(lab[i] != lab[j] for i, j in g._nearest_neighbor_grid_index()))

# ======================================================================
# B. 均匀密度：逐格点向量与全部分组摘要
# ======================================================================
print('===== B. 均匀密度 =====')
nf_uni = write_nf('nf_uni.json', [0.75] * N)
rep_u = run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=nf_uni, N=N, a=A)
check('B', '标准入口：status=ok、analyzer/observable 契约常量正确',
      rep_u.status == STATUS_OK and rep_u.analyzer == ANALYZER_DENSITY
      and rep_u.observable == OBSERVABLE_LOCAL_DENSITY)
check('B', '逐格点向量完整保留且值正确（N 个元素）',
      len(rep_u.parameters['density_by_site']) == N
      and all(abs(v - 0.75) < 1e-12 for v in rep_u.parameters['density_by_site']))
check('B', '行平均 4 组均等于均匀值',
      all(abs(v - 0.75) < 1e-12 for v in rep_u.parameters['row_mean'].values()))
check('B', 'A/B 子晶格平均均等于均匀值',
      all(abs(v - 0.75) < 1e-12
          for v in rep_u.parameters['sub_lattice_mean'].values()))
check('B', '晶胞分组平均为 N/8 个晶胞且均等于均匀值',
      len(rep_u.parameters['unit_cell_mean']) == N // 8
      and all(abs(v - 0.75) < 1e-12
              for v in rep_u.parameters['unit_cell_mean']))
check('B', '几何与边界元数据齐全（lattice/unit_cell_size/boundary/划分推导）',
      rep_u.metadata['lattice_geometry'] == 'honeycomb_xc'
      and rep_u.metadata['unit_cell_size'] == 8
      and rep_u.metadata['boundary'] == {'x': 'open', 'y': 'periodic'}
      and rep_u.metadata['sub_lattice_derivation']
      == 'bipartite_coloring_from_nearest_neighbor_table'
      and rep_u.metadata['sub_lattice_labels'] == lab,
      str(rep_u.metadata['boundary']))
check('B', 'data_restricted=False（单格点全量测量）且无 warnings',
      rep_u.data_restricted is False and rep_u.warnings == [])

# ======================================================================
# C. 已知子晶格调制：A/B 平均与预期一致
# ======================================================================
print('===== C. 子晶格调制 =====')
vals_sub = [1.0 if lab[i] == 'A' else 0.5 for i in range(N)]
rep_s = run_analysis(OBSERVABLE_LOCAL_DENSITY,
                     nf_path=write_nf('nf_sub.json', vals_sub), N=N, a=A)
sm = rep_s.parameters['sub_lattice_mean']
check('C', '子晶格平均正确区分 A=1.0 / B=0.5',
      abs(sm['A'] - 1.0) < 1e-12 and abs(sm['B'] - 0.5) < 1e-12,
      str(sm))
check('C', '逐格点向量完整保留调制（分组平均不掩盖细节）',
      rep_s.parameters['density_by_site'] == [float(v) for v in vals_sub])

# ======================================================================
# D. 已知空间调制：逐格点向量完整保留
# ======================================================================
print('===== D. 空间调制（逐格点向量保真） =====')
vals_mod = [0.9 + 0.05 * ((i // 8) % 2) for i in range(N)]  # 晶胞交替
rep_m = run_analysis(OBSERVABLE_LOCAL_DENSITY,
                     nf_path=write_nf('nf_mod.json', vals_mod), N=N, a=A)
ucm = rep_m.parameters['unit_cell_mean']
check('D', '晶胞分组平均呈现奇偶晶胞交替（0.9/0.95）',
      abs(ucm[0] - 0.9) < 1e-12 and abs(ucm[1] - 0.95) < 1e-12
      and abs(ucm[2] - 0.9) < 1e-12, str(ucm[:4]))
check('D', '逐格点向量与输入逐一相等（不能仅靠分组平均掩盖）',
      rep_m.parameters['density_by_site'] == [float(v) for v in vals_mod])

# ======================================================================
# E. 边界隆起合成数据：保留边界信息，不输出 Stripe 判定
# ======================================================================
print('===== E. 边界隆起 =====')
vals_edge = [1.2 if i // 8 in (0, 1) else 0.8 for i in range(N)]
rep_e = run_analysis(OBSERVABLE_LOCAL_DENSITY,
                     nf_path=write_nf('nf_edge.json', vals_edge), N=N, a=A)
check('E', '边界隆起数据完整保留（前两晶胞高密度）',
      abs(rep_e.parameters['unit_cell_mean'][0] - 1.2) < 1e-12
      and abs(rep_e.parameters['unit_cell_mean'][2] - 0.8) < 1e-12,
      str(rep_e.parameters['unit_cell_mean'][:3]))
check('E', 'boundary 元数据保留开边界信息（x open / y periodic）',
      rep_e.metadata['boundary'] == {'x': 'open', 'y': 'periodic'})
check('E', '报告无相级判定：classification/evidence 为 None、'
      '无 Stripe 结论字段',
      rep_e.classification is None and rep_e.evidence is None
      and 'stripe' not in json.dumps(rep_e.to_dict()))

# ======================================================================
# F. 契约序列化与反序列化
# ======================================================================
print('===== F. 序列化 =====')
rep_f = run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=nf_uni, N=N, a=A)
d_f = rep_f.to_dict()
check('F', 'to_dict 含逐格点向量/分组摘要/元数据且不含 payload',
      'density_by_site' in d_f['parameters']
      and 'boundary' in d_f['metadata']
      and 'payload' not in d_f)
p_f = os.path.join(TMP, 'rep.json')
rep_f.to_json(p_f)
back_f = json.load(open(p_f, encoding='utf-8'))
check('F', 'to_json 严格落盘可回读且参数一致',
      back_f['parameters'] == d_f['parameters']
      and back_f['metadata'] == d_f['metadata'])
check('F', 'include_payload=True 时 payload 可序列化',
      'density_by_site' in rep_f.to_dict(include_payload=True)['payload'])

# ======================================================================
# G. 失败路径（原样传播，不静默补默认值）
# ======================================================================
print('===== G. 失败路径 =====')
try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY,
                 nf_path=os.path.join(TMP, 'not_exist.json'), N=N, a=A)
    check('G', '缺失文件 → FileNotFoundError 原样传播', False, '未抛异常')
except FileNotFoundError:
    check('G', '缺失文件 → FileNotFoundError 原样传播', True)

bad_path = os.path.join(TMP, 'bad.json')
with open(bad_path, 'w', encoding='utf-8') as f:
    f.write('{not valid json')
try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=bad_path, N=N, a=A)
    check('G', '畸形 JSON → ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('G', '畸形 JSON → ValueError 原样传播', True)

short_path = write_nf('short.json', [0.7] * (N - 1))
try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=short_path, N=N, a=A)
    check('G', '格点数少于 N → ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('G', '格点数少于 N → ValueError 原样传播', True)

dup_path = os.path.join(TMP, 'dup.json')
with open(dup_path, 'w', encoding='utf-8') as f:
    json.dump([[[i if i < 199 else 0], [0.7, 0]]
               for i in range(N)], f)   # 格点 199 缺失、格点 0 重复
try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=dup_path, N=N, a=A)
    check('G', '格点索引未覆盖 0..N-1（重复/缺失）→ ValueError', False,
          '未抛异常')
except ValueError:
    check('G', '格点索引未覆盖 0..N-1（重复/缺失）→ ValueError', True)

try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=nf_uni, N=100, a=A)
    check('G', 'N 非 8 倍数 → ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('G', 'N 非 8 倍数 → ValueError 原样传播', True)

# ======================================================================
# H. 路由契约（未知 operation / 不支持 config）
# ======================================================================
print('===== H. 路由契约 =====')
try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY, operation='decay',
                 nf_path=nf_uni, N=N, a=A)
    check('H', '未知 operation → ValueError', False, '未抛异常')
except ValueError:
    check('H', '未知 operation → ValueError', True)
try:
    run_analysis(OBSERVABLE_LOCAL_DENSITY, nf_path=nf_uni, N=N, a=A,
                 config=object())
    check('H', 'profile 链路不支持 config → TypeError', False, '未抛异常')
except TypeError:
    check('H', 'profile 链路不支持 config → TypeError', True)

# ======================================================================
# I. 证据层映射：observation，不是相级支持/反证
# ======================================================================
print('===== I. 证据层映射 =====')
ph = collect_evidence([rep_u, rep_e], labels=['uni', 'edge'])
items = build_evidence_items(ph)
for it in items:
    check('I', f'条目 e{it["entry_index"]} 归类 observation（非相级证据）',
          EVIDENCE_OBSERVATION in it['categories']
          and 'order_evidence' not in it['categories']
          and 'counter_evidence' not in it['categories'],
          str(it['categories']))
    check('I', f'条目 e{it["entry_index"]} data_range 保留 N/几何/边界/晶胞',
          it['data_range']['N'] == N
          and it['data_range']['lattice_geometry'] == 'honeycomb_xc'
          and it['data_range']['boundary'] == {'x': 'open', 'y': 'periodic'}
          and it['data_range']['unit_cell_size'] == 8)
    check('I', f'条目 e{it["entry_index"]} payload 可用（原始 profile dict）',
          it['payload_available'] is True)
# 失败报告 → not_available（不伪造观测）
rep_none = run_analysis.__wrapped__ if False else None
from Physical_Quantity.Analysis_Contract import from_density_profile
ph2 = collect_evidence([from_density_profile(None)], labels=['bad'])
check('I', 'None 输入 → status=failed → not_available',
      ph2.entries[0]['status'] == STATUS_FAILED
      and build_evidence_items(ph2)[0]['categories'] == [EVIDENCE_NOT_AVAILABLE])

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
