# -*- coding: utf-8 -*-
"""
能量外推契约集成测试（D3.3b：extrapolate → AnalysisReport）

使用合成 EnergyRecord（直接 dataclass 构造），不依赖真实集群数据。
覆盖：
- 包装函数的 E_inf 与同一输入下直接调用底层 extrapolate() 一致
- N_values / power / threshold 映射正确
- status='ok'，classification/evidence/data_restricted 均为 None（语义正交）
- warnings 按契约保持空列表（信息缺失不作为失败）
- 记录数不足时底层 ValueError 原样传播
- to_json 可被标准 JSON 解析，无 NaN/Infinity 非标准字面量
- 直接调用旧接口仍返回 float，数值行为未被包装改变

注意：不假定某个异常点必然被剔除（由既有算法决定），
只验证包装与底层调用的一致性。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_energy_contract_integration.py
"""
import sys
import os
import json
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')   # extrapolate 含 plt.show 副作用，Agg 下为 no-op
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Energy_Process import EnergyRecord, EnergyProcessor
from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, OBSERVABLE_ENERGY, ANALYZER_ENERGY)
from Physical_Quantity.Analysis_Integration import energy_extrapolation_report

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


# 合成记录：理论 Es ≈ -1.5，含一个明显偏离的异常点（是否被剔除由
# 既有算法决定，测试不预设）
RECORDS = [
    EnergyRecord(N=10, E_min=-14.2, E_avg=-14.2, Es=-1.42, Es_J=-1.42),
    EnergyRecord(N=20, E_min=-29.6, E_avg=-29.6, Es=-1.48, Es_J=-1.48),
    EnergyRecord(N=30, E_min=-44.7, E_avg=-44.7, Es=-1.49, Es_J=-1.49),
    EnergyRecord(N=40, E_min=-40.0, E_avg=-40.0, Es=-1.00, Es_J=-1.00),  # 明显异常点
    EnergyRecord(N=60, E_min=-89.7, E_avg=-89.7, Es=-1.495, Es_J=-1.495),
    EnergyRecord(N=100, E_min=-149.8, E_avg=-149.8, Es=-1.498, Es_J=-1.498),
]

# ==========================================================================
# A. 包装与底层一致性 + 字段映射
# ==========================================================================
print('===== A. 包装与底层一致性 =====')
rep = energy_extrapolation_report(RECORDS, power=1, threshold=2.0)
direct = EnergyProcessor.extrapolate(RECORDS, power=1, threshold=2.0)
check('A', 'E_inf 与直接调用底层 extrapolate 结果精确一致',
      rep.parameters['E_inf'] == direct,
      f"report={rep.parameters['E_inf']:.8f} direct={direct:.8f}")
check('A', 'N_values 映射与输入 N 列表一致',
      rep.parameters['N_values'] == [10, 20, 30, 40, 60, 100],
      rep.parameters['N_values'])
check('A', 'power / threshold 直通',
      rep.parameters['power'] == 1 and rep.parameters['threshold'] == 2.0)
check('A', 'status=ok 且 classification/evidence/data_restricted 均为 None（语义正交）',
      rep.status == STATUS_OK and rep.classification is None
      and rep.evidence is None and rep.data_restricted is None,
      f"{rep.status}/{rep.classification}/{rep.evidence}/{rep.data_restricted}")
check('A', 'warnings 按契约保持空列表（信息缺失不误判为失败）',
      rep.warnings == [], rep.warnings)
check('A', 'analyzer/observable 标识正确',
      rep.analyzer == ANALYZER_ENERGY and rep.observable == OBSERVABLE_ENERGY)

# ==========================================================================
# B. 失败传播
# ==========================================================================
print('===== B. 失败场景 =====')
try:
    energy_extrapolation_report(RECORDS[:2], power=1, threshold=2.0)
    check('B', '记录数不足（<3）→ ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('B', '记录数不足（<3）→ ValueError 原样传播', True)

# ==========================================================================
# C. JSON 序列化
# ==========================================================================
print('===== C. JSON 序列化 =====')
tmp = tempfile.mkdtemp(prefix='energy_contract_')
json_path = os.path.join(tmp, 'report.json')
rep.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
back = json.loads(raw)
check('C', 'to_json 可被标准 JSON 解析且无 NaN/Infinity 非标准字面量',
      'NaN' not in raw and 'Infinity' not in raw
      and back['parameters']['E_inf'] == direct
      and back['parameters']['N_values'] == [10, 20, 30, 40, 60, 100])

# ==========================================================================
# D. 旧接口不变
# ==========================================================================
print('===== D. 旧接口 =====')
check('D', '直接调用 extrapolate 仍返回 float',
      isinstance(direct, float), type(direct).__name__)
# 包装调用前后，底层对同一输入的返回数值一致（未被包装改变）
direct_again = EnergyProcessor.extrapolate(RECORDS, power=1, threshold=2.0)
check('D', '底层数值行为未被包装改变（重复调用一致）',
      direct == direct_again, f"{direct:.8f} vs {direct_again:.8f}")

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
