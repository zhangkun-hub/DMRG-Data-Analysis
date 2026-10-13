# -*- coding: utf-8 -*-
"""
统一分析入口测试（D3.7：run_analysis 路由分派）

覆盖：
- 六条路由均返回 AnalysisReport
- 结构因子 estimate 与 peak_analysis 明确区分（operation=None → ValueError）
- 衰减链路的 config 与必要参数正确透传
- 能量外推的 power/threshold 正确透传
- 未知 observable / 不支持的 operation → 清晰 ValueError
- 缺参/多参/不支持 config 的链路收到 config → TypeError（不静默忽略）
- 底层异常原样传播（空输入 ValueError）
- 路由结果与对应现有包装函数的结果一致

优先合成数据/mock；衰减三条链路用合成记录（确定性），
与直接调用包装函数的关键字段逐项对比。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_analysis_entry.py
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Physical_Data_Record import CorrelationRecord, SuperCorrRecord
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Spin_Structure_Factor import SpinStructureFactorEstimator
from Physical_Quantity.Energy_Process import EnergyRecord
from Physical_Quantity.Correlation_Decay_Analyzer import AnalyzerConfig
from Physical_Quantity.Analysis_Contract import (
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_CORRELATION,
    OBSERVABLE_CHARGE_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_ENERGY,
)
from Physical_Quantity.Analysis_Integration import (
    analyze_with_report, structure_factor_report,
    peak_analysis_report, energy_extrapolation_report)
from Physical_Quantity.Super_Correlation_Analyzer import SuperCorrDecayAnalyzer
from Physical_Quantity.Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer
from Physical_Quantity.Charge_Correlation_Analyzer import ChargeCorrDecayAnalyzer
from Physical_Quantity.Analysis_Entry import run_analysis

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


# ---- 合成数据（确定性） ----
g = HoneycombXCGeometry(200, 1.0)
rng = np.random.default_rng(5)
_dists = np.arange(3.0, 37.0, 3.0)   # 12 个距离点
_base = 0.25 * _dists ** (-1.5)
_C = _base + 0.05 * np.abs(_base) * rng.standard_normal(len(_dists))

recs_spin = [CorrelationRecord(50, 50 + 4 + 4 * i, d, v)
             for i, (d, v) in enumerate(zip(_dists, _C))]
recs_charge = [CorrelationRecord(49, 53 + 4 * i, d, v)
               for i, (d, v) in enumerate(zip(_dists, _C))]
# SC 合成：bond 对（ref bond 固定，target bond 水平平移），族由 r mod 3 fallback 或真实几何判定
recs_sc = [SuperCorrRecord((49, 50), (52 + 4 * i, 53 + 4 * i), d, v)
           for i, (d, v) in enumerate(zip(_dists, _C))]

recs_energy = [
    EnergyRecord(N=10, E_min=-14.2, E_avg=-14.2, Es=-1.42, Es_J=-1.42),
    EnergyRecord(N=20, E_min=-29.6, E_avg=-29.6, Es=-1.48, Es_J=-1.48),
    EnergyRecord(N=40, E_min=-59.8, E_avg=-59.8, Es=-1.495, Es_J=-1.495),
    EnergyRecord(N=80, E_min=-119.8, E_avg=-119.8, Es=-1.4975, Es_J=-1.4975),
]

_qx = np.linspace(-1.5, 1.5, 15)
_qy = np.linspace(-1.5, 1.5, 13)
_XX, _YY = np.meshgrid(_qx, _qy, indexing='ij')
_sig = np.exp(-(_XX ** 2 + _YY ** 2) / 0.3)
_q = np.array([(x, y) for x in _qx for y in _qy])
SF_SYNTH = {'q': _q, 'S_q': _sig.reshape(-1).astype(complex), 'definition': 'test',
            'N': 8, 'is_restricted': False, 'measured_pair_count': 195,
            'expected_pair_count': None, 'pair_coverage': None,
            'source_coverage': 0.0, 'missing_pairs': None,
            'imaginary_residual': 0.0, 'duplicate_pairs': 0,
            'invalid_values': 0, 'skipped_no_coords': []}


def _est_synth(arg, q_list, N, a=1.0, **kwargs):
    return SF_SYNTH


# ==========================================================================
# A. 六条路由返回 AnalysisReport + 与包装函数一致
# ==========================================================================
print('===== A. 六条路由 =====')
from Physical_Quantity.Analysis_Contract import AnalysisReport

pairs = [
    ('SC 衰减', OBSERVABLE_SUPERCONDUCTING_CORRELATION, None,
     dict(record_list=recs_sc),
     lambda kw: analyze_with_report(SuperCorrDecayAnalyzer, recs_sc,
                                    OBSERVABLE_SUPERCONDUCTING_CORRELATION)),
    ('Spin 衰减', OBSERVABLE_SPIN_CORRELATION, None,
     dict(record_list=recs_spin, N=200),
     lambda kw: analyze_with_report(SpinCorrDecayAnalyzer, recs_spin,
                                    OBSERVABLE_SPIN_CORRELATION, N=200)),
    ('Charge 衰减', OBSERVABLE_CHARGE_CORRELATION, None,
     dict(record_list=recs_charge, N=200),
     lambda kw: analyze_with_report(ChargeCorrDecayAnalyzer, recs_charge,
                                    OBSERVABLE_CHARGE_CORRELATION, N=200)),
    ('SF 估计', OBSERVABLE_SPIN_STRUCTURE_FACTOR, 'estimate',
     dict(estimator_fn=_est_synth, estimator_arg=None,
          q_list=_q, N=8),
     lambda kw: structure_factor_report(_est_synth, None, _q, 8)),
    ('SF 峰值', OBSERVABLE_SPIN_STRUCTURE_FACTOR, 'peak_analysis',
     dict(sf_result=SF_SYNTH),
     lambda kw: peak_analysis_report(SF_SYNTH)),
    ('能量外推', OBSERVABLE_ENERGY, None,
     dict(records=recs_energy, power=1, threshold=2.0),
     lambda kw: energy_extrapolation_report(recs_energy, 1, 2.0)),
]
for name, obs, op, kw, direct_fn in pairs:
    rep = run_analysis(obs, operation=op, **kw)
    direct = direct_fn(kw)
    check('A', f'{name}：路由返回 AnalysisReport', isinstance(rep, AnalysisReport))
    check('A', f'{name}：classification 与直接包装一致',
          rep.classification == direct.classification,
          rep.classification)
    check('A', f'{name}：status 与直接包装一致',
          rep.status == direct.status, rep.status)

# ==========================================================================
# B. operation 区分 / config 与参数透传
# ==========================================================================
print('===== B. 区分与透传 =====')
try:
    run_analysis(OBSERVABLE_SPIN_STRUCTURE_FACTOR)
    check('B', 'SF 未指定 operation → ValueError（不猜测操作）', False, '未抛异常')
except ValueError:
    check('B', 'SF 未指定 operation → ValueError（不猜测操作）', True)

cfg = AnalyzerConfig(envelope_mode='extrema', r_min_override=6.0)
rep_cfg = run_analysis(OBSERVABLE_SPIN_CORRELATION, config=cfg,
                       record_list=recs_spin, N=200)
check('B', '衰减链路 config 透传（envelope_mode 生效）',
      rep_cfg.metadata['config']['envelope_mode'] == 'extrema'
      and rep_cfg.metadata['config']['r_min_override'] == 6.0)
rep_pow = run_analysis(OBSERVABLE_ENERGY, records=recs_energy,
                       power=2, threshold=1.5)
check('B', '能量外推 power/threshold 透传',
      rep_pow.parameters['power'] == 2 and rep_pow.parameters['threshold'] == 1.5)

# ==========================================================================
# C. 错误路径（清晰报错，不静默忽略）
# ==========================================================================
print('===== C. 错误路径 =====')
try:
    run_analysis('unknown_observable')
    check('C', '未知 observable → ValueError', False, '未抛异常')
except ValueError as e:
    check('C', '未知 observable → ValueError', '支持' in str(e), str(e)[:60])
try:
    run_analysis(OBSERVABLE_ENERGY, operation='decay', records=recs_energy)
    check('C', '不支持的 operation → ValueError', False, '未抛异常')
except ValueError as e:
    check('C', '不支持的 operation → ValueError', '支持' in str(e), str(e)[:60])
try:
    run_analysis(OBSERVABLE_SPIN_CORRELATION)   # 缺 record_list
    check('C', '缺必需参数 → TypeError（不静默忽略）', False, '未抛异常')
except TypeError:
    check('C', '缺必需参数 → TypeError（不静默忽略）', True)
try:
    run_analysis(OBSERVABLE_ENERGY, records=recs_energy, config=cfg)
    check('C', '不支持 config 的链路收到 config → TypeError', False, '未抛异常')
except TypeError as e:
    check('C', '不支持 config 的链路收到 config → TypeError', '不支持 config' in str(e))
try:
    run_analysis(OBSERVABLE_SUPERCONDUCTING_CORRELATION, record_list=[])
    check('C', '底层异常原样传播（空输入 ValueError）', False, '未抛异常')
except ValueError:
    check('C', '底层异常原样传播（空输入 ValueError）', True)

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
