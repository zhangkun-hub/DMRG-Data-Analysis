# -*- coding: utf-8 -*-
"""
电荷关联契约集成测试（D3.5：ChargeCorrDecayAnalyzer → AnalysisReport）

覆盖：
- 合成幂律数据与近常数数据的契约映射（status/classification 直通，
  data_restricted=None，param_errors/fit_quality 独立存放）
- 真实 N200 nf49nf.json 全 field 解析（两族均衡 75/75）→ 包装结果与
  直接分析结果一致；不预设衰减分类、不推断物理相态
- 空输入 ValueError 原样传播
- 旧接口：直接 analyze 返回 DecayAnalysisResult，行为不变
- 新增 OBSERVABLE_CHARGE_CORRELATION 常量正确
- JSON 序列化无 NaN/Infinity 非标准字面量

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_charge_contract_integration.py
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

from Physical_Quantity.Physical_Data_Record import CorrelationRecord
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Charge_Correlation_Analyzer import ChargeCorrDecayAnalyzer
from Physical_Quantity.Correlation_Decay_Analyzer import DecayAnalysisResult
from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, OBSERVABLE_CHARGE_CORRELATION)
from Physical_Quantity.Analysis_Integration import analyze_with_report

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def load_charge(ver, N=200):
    """全 field 解析 nf{ver}nf.json（field 连续，用 range(0,N,1) 保留全部条目；
    与 sz 系列 step-4 sequence 不同——后者会丢弃 3/4 数据）"""
    base = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty/measure_{ver}'
    return CorrelationRecord.extract_data(
        open(f'{base}/nf{ver}nf.json', encoding='utf-8').read(),
        range(0, N, 1), N, 1.0)


# ==========================================================================
# A. 合成数据（幂律 + 近常数）
# ==========================================================================
print('===== A. 合成数据契约映射 =====')
g = HoneycombXCGeometry(200, 1.0)
source = 49
fields = [j for j in range(50, 200) if j != source]
dists = [g.get_distance(source, j) for j in fields]
rng = np.random.default_rng(11)

# 幂律合成（近程主导衰减形态，仅验证包装与底层一致性，不预设分类）
base_vals = np.array([0.25 * d ** (-1.5) for d in dists])
C_pow = base_vals + 0.05 * np.abs(base_vals) * rng.standard_normal(len(base_vals))
recs_pow = [CorrelationRecord(source, j, d, v) for j, d, v in zip(fields, dists, C_pow)]

rep_pow = analyze_with_report(ChargeCorrDecayAnalyzer, recs_pow,
                              OBSERVABLE_CHARGE_CORRELATION, N=200)
direct_pow = ChargeCorrDecayAnalyzer.analyze(recs_pow, N=200)
check('A', '幂律合成：status=ok 且 classification 与直接调用一致',
      rep_pow.status == STATUS_OK
      and rep_pow.classification == direct_pow.classification,
      rep_pow.classification)
check('A', '幂律合成：parameters 与直接调用包络参数一致',
      rep_pow.parameters == dict(direct_pow.envelope_fit.params))
check('A', '幂律合成：param_errors/fit_quality 独立存放于 metadata',
      'param_errors' in rep_pow.metadata and 'fit_quality' in rep_pow.metadata
      and 'param_errors' not in rep_pow.parameters)
check('A', '幂律合成：data_restricted=None（衰减输出无受限性标记，不编造）',
      rep_pow.data_restricted is None)

# 近常数合成（模拟密度关联 ~0.99 形态；只验证包装与底层一致性）
C_const = 0.99 + 0.005 * rng.standard_normal(len(dists))
recs_const = [CorrelationRecord(source, j, d, v) for j, d, v in zip(fields, dists, C_const)]
rep_const = analyze_with_report(ChargeCorrDecayAnalyzer, recs_const,
                                OBSERVABLE_CHARGE_CORRELATION, N=200)
direct_const = ChargeCorrDecayAnalyzer.analyze(recs_const, N=200)
check('A', '近常数合成：包装与直接调用分类一致（不预设具体分类）',
      rep_const.status == STATUS_OK
      and rep_const.classification == direct_const.classification,
      rep_const.classification)
check('A', '近常数合成：parameters 与直接调用一致',
      rep_const.parameters == dict(direct_const.envelope_fit.params))

# ==========================================================================
# B. 真实 N200 数据（nf49nf 全 field 解析）
# ==========================================================================
print('===== B. 真实 N200 电荷关联（d1_100 / measure_49） =====')
recs_real = load_charge('49')
rep_real = analyze_with_report(ChargeCorrDecayAnalyzer, recs_real,
                               OBSERVABLE_CHARGE_CORRELATION, N=200)
direct_real = ChargeCorrDecayAnalyzer.analyze(recs_real, N=200)
check('B', '真实数据：全 field 解析 150 条（两族均衡数据可用）',
      len(recs_real) == 150, f'{len(recs_real)} 条')
check('B', '真实数据：包装与直接调用 classification/evidence 一致（不预设分类）',
      rep_real.classification == direct_real.classification
      and rep_real.evidence == direct_real.evidence_level,
      f"{rep_real.classification}/{rep_real.evidence}")
check('B', '真实数据：parameters 与直接调用一致',
      rep_real.parameters == dict(direct_real.envelope_fit.params))
check('B', '真实数据：analyzer/observable 标识正确',
      rep_real.analyzer == 'ChargeCorrDecayAnalyzer'
      and rep_real.observable == OBSERVABLE_CHARGE_CORRELATION)
check('B', '真实数据：族分解诊断存在（两族均有数据）',
      rep_real.payload is not None
      and direct_real.family_diagnostics is not None
      and set(direct_real.family_diagnostics.family_r.keys()) == {'0', '1'})

# ==========================================================================
# C. 旧接口 / 常量 / 失败 / JSON
# ==========================================================================
print('===== C. 旧接口与边界 =====')
check('C', 'OBSERVABLE_CHARGE_CORRELATION 常量值为 charge_correlation',
      OBSERVABLE_CHARGE_CORRELATION == 'charge_correlation')
check('C', '直接 analyze 返回 DecayAnalysisResult（旧接口类型不变）',
      isinstance(direct_real, DecayAnalysisResult))
try:
    analyze_with_report(ChargeCorrDecayAnalyzer, [], OBSERVABLE_CHARGE_CORRELATION)
    check('C', '空输入 ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('C', '空输入 ValueError 原样传播', True)
tmp = tempfile.mkdtemp(prefix='charge_contract_')
json_path = os.path.join(tmp, 'report.json')
rep_real.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
back = json.loads(raw)
check('C', 'JSON：无 NaN/Infinity 字面量、默认无 payload、可解析读回',
      'NaN' not in raw and 'Infinity' not in raw
      and '"payload"' not in raw and back['status'] == 'ok'
      and back['classification'] == rep_real.classification)

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
