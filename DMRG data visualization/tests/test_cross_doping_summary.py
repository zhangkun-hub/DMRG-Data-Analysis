# -*- coding: utf-8 -*-
"""
跨 doping 汇总脚本测试（PhaseDiagnosis-P4.3）

覆盖（mock / 小型合成数据为主，避免依赖完整科研数据目录）：
- A. 行构建保真：doping/N/bond_dim/观测量/分析器/参考键/键方向、status/
  classification/data_restricted 独立语义、拟合窗口/拟合模型/数值质量
  信息、failed 行、INCONCLUSIVE 与 status=ok 并存、缺失字段留空、
  NaN → JSON null
- B. 可比性分组：direct（条件一致）/ cautious（参考键或源点、键方向或
  定义不同）/ not_direct（N 或 bond_dim 不一致、q_grid 摘要不一致、
  分析失败、同组全失败）/ sole（无同类条目）；q_grid 仅摘要级标注
- C. 输出：严格 JSON（无 NaN/Infinity 字面量）、CSV 固定列序与空值
  保守处理、输出目录自动创建、重复执行覆盖不追加且字节一致
- D. 无物理结论：输出不含趋势/相态字段；conclusion 仅
  evidence_collected；sufficiency overall=unknown（rules=None）
- E. mock 集成：fake run_analysis + fake loaders + 临时目录树；数据缺失
  与 DependencyFailed 显式失败行；参考键过滤；CLI main() 端到端
- F. 真实数据 smoke（数据目录存在时运行）：d1_100 + 参考键 49_50，
  真跑 SC 衰减 / 自旋 / 电荷分析，验证真实端到端

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_cross_doping_summary.py
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 根目录独立脚本经 importlib 载入（非包内模块，避免路径耦合）
_SPEC = importlib.util.spec_from_file_location(
    'cross_doping_summary', os.path.join(ROOT, 'Cross_Doping_Summary.py'))
cds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cds)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK, STATUS_FAILED,
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR)

SC_OBS = OBSERVABLE_SUPERCONDUCTING_CORRELATION
SPIN_OBS = OBSERVABLE_SPIN_STRUCTURE_FACTOR
CHARGE_OBS = OBSERVABLE_CHARGE_STRUCTURE_FACTOR

ART = os.path.join(HERE, 'artifacts')
RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ---- 合成输入 ----
Q_SMALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
QGRID = {'n_qx': 2, 'n_qy': 2, 'qx_min': 0.0, 'qx_max': 1.0,
         'qy_min': 0.0, 'qy_max': 1.0}
QGRID_OTHER = {'n_qx': 2, 'n_qy': 2, 'qx_min': 0.0, 'qx_max': 2.0,
               'qy_min': 0.0, 'qy_max': 1.0}


def outcome(doping, observable, analyzer, ref, key, report=None,
            failure=None, N=200, bond_dim='1500'):
    return {'context': {'doping': doping, 'N': N, 'bond_dim': bond_dim,
                        'observable': observable, 'analyzer': analyzer,
                        'refbond_or_source': ref, 'key_or_definition': key},
            'report': report, 'failure': failure}


def sc_report(classification='exponential', **meta):
    base = {'r_min': 1.5, 'r_max': 30.0, 'verdict': 'moderate',
            'n_points': 20, 'n_distances': 20}
    base.update(meta)
    return AnalysisReport(
        analyzer='SuperCorrDecayAnalyzer', observable=SC_OBS,
        status=STATUS_OK, classification=classification,
        parameters={'xi': 5.0}, evidence='moderate',
        data_restricted=None, warnings=[],
        metadata=base,
        payload=SimpleNamespace(recommended_model='exp_Cinf',
                                envelope_mode_used='all_points'))


def spin_sf_report(qgrid=QGRID, definition='sz_sz', q=None):
    payload = {'q_grid': qgrid, 'definition': definition}
    if q is not None:
        payload['q'] = q
    return AnalysisReport(
        analyzer='SpinStructureFactorEstimator', observable=SPIN_OBS,
        status=STATUS_OK, classification=None, parameters={}, evidence=None,
        data_restricted=True, warnings=[],
        metadata={'definition': definition, 'measured_pair_count': 74,
                  'source_coverage': 0.01, 'imaginary_residual': 0.0},
        payload=payload)


def peak_report(definition='sz_sz'):
    return AnalysisReport(
        analyzer='SpinStructureFactorPeakAnalyzer', observable=SPIN_OBS,
        status=STATUS_OK, classification=None, parameters={'n_peaks': 1},
        evidence=None, data_restricted=True, warnings=[],
        metadata={'definition': definition, 'N': 200, 'signal_used': 'abs'},
        payload={'status': 'ok', 'peaks': []})


def charge_report(qgrid=QGRID, sources=(49, 50)):
    return AnalysisReport(
        analyzer='ChargeStructureFactorEstimator', observable=CHARGE_OBS,
        status=STATUS_OK, classification=None, parameters={}, evidence=None,
        data_restricted=True, warnings=[],
        metadata={'measured_pair_count': 500, 'source_coverage': 0.01,
                  'imaginary_residual': 0.0},
        payload={'q_grid': qgrid,
                 'diagonal_source_sites': list(sources)})


def charge_report_no_sources():
    """电荷报告但 payload 缺 diagonal_source_sites（源点未知场景）。"""
    return AnalysisReport(
        analyzer='ChargeStructureFactorEstimator', observable=CHARGE_OBS,
        status=STATUS_OK, classification=None, parameters={}, evidence=None,
        data_restricted=True, warnings=[],
        metadata={'measured_pair_count': 500, 'source_coverage': 0.01,
                  'imaginary_residual': 0.0},
        payload={'q_grid': QGRID})


# ==========================================================================
# A. 行构建保真（缺失字段保守处理 / 四态独立语义 / NaN 处理）
# ==========================================================================
print('===== A. 行构建保真 =====')
outcomes_a = [
    outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
            sc_report(fit_quality={'r2': 0.95, 'aicc': -20.0},
                      param_errors={'xi': 0.5})),
    outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'B',
            sc_report(classification='INCONCLUSIVE')),
    outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'C',
            None, {'type': 'FileNotFoundError', 'message': 'scsCa 缺失'}),
    outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0', 'sz_sz',
            spin_sf_report(q=Q_SMALL)),
    outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0', 'sz_sz',
            peak_report()),
    outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
            'charge_density_connected', charge_report()),
    outcome('d1_20', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
            sc_report(r_min=np.nan, fit_quality={'r2': float('nan')})),
    # 无有效前置的峰值行（前置是衰减行，非 SF estimate）→ q_grid 保守留空
    outcome('d1_20', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'B',
            sc_report()),
    outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0', 'sz_sz',
            peak_report()),
]
summ_a = cds.summarize(outcomes_a, Q_SMALL)
rows_a = summ_a['rows']
check('A', '行数与 outcome 数一致', len(rows_a) == 9, f'{len(rows_a)}')
r_decay = rows_a[0]
check('A', '衰减行：拟合窗口/verdict/点数/模型名/模式保真',
      r_decay['fit_model'] == 'exp_Cinf'
      and r_decay['envelope_mode'] == 'all_points'
      and r_decay['fit_r_min'] == 1.5 and r_decay['verdict'] == 'moderate'
      and r_decay['n_points'] == 20,
      f"{r_decay['fit_model']}/{r_decay['fit_r_min']}")
check('A', '衰减行：params/fit_quality/param_errors 三个 JSON 列可解析且保真',
      json.loads(r_decay['params_json']) == {'xi': 5.0}
      and json.loads(r_decay['fit_quality_json'])['r2'] == 0.95
      and json.loads(r_decay['param_errors_json']) == {'xi': 0.5})
check('A', '衰减行：data_restricted=None → unknown（不默认 unrestricted）',
      r_decay['data_restricted'] == 'unknown')
r_incon = rows_a[1]
check('A', 'INCONCLUSIVE 分类保留且 status=ok 并存（两字段独立）',
      r_incon['classification'] == 'INCONCLUSIVE'
      and r_incon['status'] == 'ok')
r_failed = rows_a[2]
check('A', '失败行：status=failed、failed_type/message 保留、无伪造分类',
      r_failed['status'] == 'failed'
      and r_failed['failed_type'] == 'FileNotFoundError'
      and 'scsCa' in r_failed['failed_message']
      and r_failed['classification'] == ''
      and r_failed['data_restricted'] == 'unknown')
r_spin = rows_a[3]
check('A', '自旋 SF 行：restricted 标签与覆盖计数保真、q_grid 可解析',
      r_spin['data_restricted'] == 'restricted'
      and r_spin['measured_pair_count'] == 74
      and r_spin['source_coverage'] == 0.01
      and json.loads(r_spin['q_grid_json']) == QGRID)
check('A', '峰值行：n_peaks 保真、q_grid 按唯一匹配的 estimate 的 q 坐标计算',
      rows_a[4]['n_peaks'] == 1
      and rows_a[4]['peak_estimate_entry_index'] == 3
      and json.loads(rows_a[4]['q_grid_json']) == QGRID)
check('A', '峰值行无有效前置：q_grid 与关联索引保守留空（unknown，不补默认值）',
      rows_a[8]['q_grid_json'] == ''
      and rows_a[8]['peak_estimate_entry_index'] == '')
r_chg = rows_a[5]
check('A', '电荷行：源点身份以 payload 实际 diagonal_source_sites 为准',
      r_chg['refbond_or_source'] == '49_50'
      and json.loads(r_chg['diagonal_source_sites_json']) == [49, 50])
r_nan = rows_a[6]
check('A', 'NaN：拟合窗口 NaN 保留为 NaN（CSV 层转空），fit_quality 转 null',
      math.isnan(r_nan['fit_r_min'])
      and json.loads(r_nan['fit_quality_json']) == {'r2': None})
check('A', '缺失字段：无 fit_quality 的行该列留空（不编造默认值）',
      r_incon['fit_quality_json'] == '' and r_failed['fit_model'] == '')

# ==========================================================================
# B. 可比性分组（机械条件检查）
# ==========================================================================
print('===== B. 可比性分组 =====')


def bucket_of(doping, ref, key, outcomes, analyzer=None):
    summ = cds.summarize(outcomes, Q_SMALL)
    for row in summ['rows']:
        if (row['doping'] == doping and row['refbond_or_source'] == ref
                and row['key_or_definition'] == key
                and (analyzer is None or row['analyzer'] == analyzer)):
            return row['comparability_bucket'], row['comparability_reason']
    raise AssertionError(f'未找到行 {doping}/{ref}/{key}/{analyzer}')


# B1. 7 doping 同参考键同键方向 → direct
o_b1 = [outcome(d, SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A', sc_report())
        for d in cds.DEFAULT_DOPINGS]
summ_b1 = cds.summarize(o_b1, Q_SMALL)
check('B', '7 doping 同条件 SC：全部 direct，组 n_ok=7',
      all(r['comparability_bucket'] == 'direct' for r in summ_b1['rows'])
      and summ_b1['comparability_groups'][0]['bucket'] == 'direct'
      and summ_b1['comparability_groups'][0]['n_ok'] == 7,
      summ_b1['comparability_groups'][0]['reason'][:60])

# B2. 参考键不同 → cautious
o_b2 = [outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                sc_report()),
        outcome('d1_20', SC_OBS, 'SuperCorrDecayAnalyzer', '50_51', 'A',
                sc_report())]
b2, r2 = bucket_of('d1_10', '49_50', 'A', o_b2)
check('B', '参考键不同：cautious（已知差异，只能谨慎比较）',
      b2 == 'cautious' and '参考键' in r2 and '源点' not in r2, r2[:60])

# B3. N 不一致 → not_direct
o_b3 = [outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                sc_report(), N=200),
        outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                sc_report(), N=400)]
b3, r3 = bucket_of('d1_10', '49_50', 'A', o_b3)
check('B', 'N 不一致：not_direct（关键条件不一致）',
      b3 == 'not_direct' and 'N 或 bond dimension' in r3, r3[:60])

# B4. 失败行 → not_direct
o_b4 = [outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                sc_report()),
        outcome('d1_20', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                None, {'type': 'ValueError', 'message': '空输入'})]
b4, r4 = bucket_of('d1_20', '49_50', 'A', o_b4)
check('B', '失败行：not_direct（无数值参与并列比较）',
      b4 == 'not_direct' and '分析失败' in r4, r4[:60])

# B5. 单条电荷行 → sole
o_b5 = [outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                'charge_density_connected', charge_report())]
summ_b5 = cds.summarize(o_b5, Q_SMALL)
check('B', '无同类条目：sole（无并列比较对象）',
      summ_b5['rows'][0]['comparability_bucket'] == 'sole')

# B6. 源点集合不同 → not_direct（P1 修复：不得默认可比，不落 cautious）
o_b6 = [outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                'charge_density_connected', charge_report(sources=(49, 50))),
        outcome('d1_20', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                'charge_density_connected', charge_report(sources=(49,)))]
b6, r6 = bucket_of('d1_10', '49_50', 'charge_density_connected', o_b6)
check('B', '源点集合不同：not_direct（不得默认可比）',
      b6 == 'not_direct' and '不得默认可比' in r6, r6[:70])

# B7. q_grid 摘要不一致 → not_direct
o_b7 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                'sz_sz', spin_sf_report(QGRID)),
        outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                'sz_sz', spin_sf_report(QGRID_OTHER))]
b7, r7 = bucket_of('d1_10', '0', 'sz_sz', o_b7)
check('B', 'q_grid 摘要不一致：not_direct',
      b7 == 'not_direct' and 'q_grid 摘要不一致' in r7, r7[:60])

# B8. 自旋 SF 摘要一致 → direct（明确摘要级，完整坐标未验证）
o_b8 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                'sz_sz', spin_sf_report(QGRID)),
        outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                'sz_sz', spin_sf_report(QGRID))]
b8, r8 = bucket_of('d1_10', '0', 'sz_sz', o_b8)
check('B', '自旋 SF 摘要一致：direct 且明确标注完整坐标未验证',
      b8 == 'direct' and 'q_grid 摘要一致' in r8 and '完整坐标未验证' in r8,
      r8[:80])

# B9. 电荷同源点同 q_grid → direct（含源点集合说明）
o_b9 = [outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                'charge_density_connected', charge_report()),
        outcome('d1_20', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                'charge_density_connected', charge_report())]
b9, r9 = bucket_of('d1_10', '49_50', 'charge_density_connected', o_b9)
check('B', '电荷同源点同 q_grid：direct，注明 diagonal_source_sites 语义',
      b9 == 'direct' and 'diagonal_source_sites 集合相等' in r9, r9[:80])

# B10. 峰值行（P2 不适用）→ 行内 q_grid 摘要（P5：前置 estimate 的 q 坐标）
o_b10 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=Q_SMALL)),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report()),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=Q_SMALL)),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report())]
b10, r10 = bucket_of('d1_10', '0', 'sz_sz', o_b10,
                     analyzer='SpinStructureFactorPeakAnalyzer')
check('B', '峰值行：direct（q 摘要按前置 estimate 的 payload q 计算，'
      '注明来源且完整坐标未验证）',
      b10 == 'direct' and '对应 SF 结果' in r10
      and '完整坐标未验证' in r10, r10[:80])

# B11. 键方向不同 → cautious
o_b11 = [outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                 sc_report()),
         outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'B',
                 sc_report())]
b11, r11 = bucket_of('d1_10', '49_50', 'A', o_b11)
check('B', '键方向不同：cautious（已知差异）',
      b11 == 'cautious' and '键方向或定义' in r11, r11[:60])

# B12. 同组仅一条成功其余失败 → 成功行 not_direct（U1 锁定：同身份组
# 全部失败时优先 not_direct，不降级到 cautious）
o_b12 = [outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                 sc_report()),
         outcome('d1_20', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                 None, {'type': 'ValueError', 'message': '空输入'})]
b12, r12 = bucket_of('d1_10', '49_50', 'A', o_b12)
check('B', '同组仅一条成功：not_direct（其余失败无数值可并列）',
      b12 == 'not_direct' and '无成功条目可并列' in r12, r12[:60])

# B13. 源点差异与定义差异并存 → 源点差异优先 not_direct（P1）
o_b13 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report()),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '49_50',
                 'spin_dot', spin_sf_report(definition='spin_dot'))]
b13, r13 = bucket_of('d1_10', '0', 'sz_sz', o_b13)
check('B', '源点差异与定义差异并存：源点差异优先 not_direct（不落 cautious）',
      b13 == 'not_direct' and '不得默认可比' in r13, r13[:70])

# B14. 同源点、定义不同但 P2 q_grid 不一致 → not_direct（P2 不被覆盖）
o_b14 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(QGRID)),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'spin_dot', spin_sf_report(QGRID_OTHER, definition='spin_dot'))]
b14, r14 = bucket_of('d1_10', '0', 'sz_sz', o_b14)
check('B', '定义不同且 P2 q_grid 不一致：not_direct（不被 cautious 覆盖）',
      b14 == 'not_direct' and 'q_grid 摘要不一致或未知' in r14, r14[:70])

# B15. source set 未知（一行缺 diagonal_source_sites / 两行都缺）
o_b15a = [outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                  'charge_density_connected', charge_report()),
          outcome('d1_20', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                  'charge_density_connected', charge_report_no_sources())]
b15a, r15a = bucket_of('d1_10', '49_50', 'charge_density_connected', o_b15a)
check('B', '源点未知（一行缺 diagonal_source_sites）：not_direct，'
      '不被宽松分支覆盖',
      b15a == 'not_direct' and '不得默认可比' in r15a, r15a[:70])
o_b15b = [outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                  'charge_density_connected', charge_report_no_sources()),
          outcome('d1_20', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                  'charge_density_connected', charge_report_no_sources())]
summ_b15b = cds.summarize(o_b15b, Q_SMALL)
check('B', '两行均缺 diagonal_source_sites：P2 unknown → not_direct',
      summ_b15b['rows'][0]['comparability_bucket'] == 'not_direct'
      and 'diagonal_source_sites 未知或未审计'
      in summ_b15b['rows'][0]['comparability_reason'])

# B16. 3 个同身份 SF 行 → direct 且消息去重（P4）
o_b16 = [outcome(d, SPIN_OBS, 'SpinStructureFactorEstimator', '0', 'sz_sz',
                 spin_sf_report()) for d in ('d1_10', 'd1_20', 'd1_25')]
summ_b16 = cds.summarize(o_b16, Q_SMALL)
r16 = summ_b16['rows'][0]
check('B', '3 个同身份 SF 行 direct：q_grid 摘要一致短语只出现一次（去重）',
      r16['comparability_bucket'] == 'direct'
      and r16['comparability_reason'].count('q_grid 摘要一致') == 1,
      r16['comparability_reason'][:80])

# B17. 峰值行前置 q 坐标不同 → not_direct（P5：不用全局 q_list 假设）
QA = np.array([[0.0, 0.0], [1.0, 1.0]])
QB = np.array([[0.0, 0.0], [2.0, 1.0]])
o_b17 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=QA)),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report()),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=QB)),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report())]
summ_b17 = cds.summarize(o_b17, Q_SMALL)
peaks17 = [r for r in summ_b17['rows']
           if r['analyzer'] == 'SpinStructureFactorPeakAnalyzer']
check('B', '峰值行前置 q 坐标不同：not_direct（不使用全局 q_list）',
      len(peaks17) == 2
      and all(r['comparability_bucket'] == 'not_direct'
              and 'q 坐标摘要不一致' in r['comparability_reason']
              for r in peaks17))

# B18. 身份来源标记（P3）
o_b18 = [outcome('d1_10', SC_OBS, 'SuperCorrDecayAnalyzer', '49_50', 'A',
                 sc_report()),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report()),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report()),
         outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                 'charge_density_connected', charge_report()),
         outcome('d1_10', CHARGE_OBS, 'ChargeStructureFactorEstimator', '',
                 'charge_density_connected', charge_report_no_sources())]
summ_b18 = cds.summarize(o_b18, Q_SMALL)
rows_b18 = summ_b18['rows']
check('B', '身份来源标记：SC=refbond_dir / 电荷=payload_field / '
      '自旋=script_convention / 字段缺失=空',
      rows_b18[0]['source_identity_kind'] == 'refbond_dir'
      and rows_b18[1]['source_identity_kind'] == 'script_convention'
      and rows_b18[2]['source_identity_kind'] == 'script_convention'
      and rows_b18[3]['source_identity_kind'] == 'payload_field'
      and rows_b18[4]['source_identity_kind'] == '',
      [r['source_identity_kind'] for r in rows_b18])

# B19. 同源点、定义不同且一行 q_grid 缺失 → P2 unknown → not_direct
o_b19 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(QGRID)),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'spin_dot', spin_sf_report(qgrid=None, definition='spin_dot'))]
b19, r19 = bucket_of('d1_10', '0', 'sz_sz', o_b19)
check('B', '同行 q_grid 缺失（P2 unknown）：not_direct（未知不被覆盖）',
      b19 == 'not_direct' and 'unknown' in r19, r19[:70])

# B20. U2：不同约定源点的 Peak 之间 / estimate 与不同源点 Peak 之间，
# 均不得因其他条件相同判为 direct
o_b20a = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                  'sz_sz', peak_report()),
          outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer',
                  '49_50', 'spin_dot', peak_report(definition='spin_dot'))]
b20a, r20a = bucket_of('d1_10', '0', 'sz_sz', o_b20a,
                       analyzer='SpinStructureFactorPeakAnalyzer')
check('B', '不同约定源点的 Peak：not_direct（不因其他条件相同判 direct）',
      b20a == 'not_direct' and '不得默认可比' in r20a, r20a[:70])
o_b20b = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                  'sz_sz', spin_sf_report()),
          outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer',
                  '49_50', 'spin_dot', peak_report(definition='spin_dot'))]
summ_b20b = cds.summarize(o_b20b, Q_SMALL)
check('B', 'estimate 与不同约定源点的 Peak：均不判 direct（无同类条目 → sole）',
      all(r['comparability_bucket'] != 'direct' for r in summ_b20b['rows'])
      and all(r['comparability_bucket'] == 'sole' for r in summ_b20b['rows']))

# B21. 对应 estimate 的 P2 状态 unknown → Peak 不得 direct（方案 b）
o_b21 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(qgrid=None, q=QA)),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report()),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(qgrid=None, q=QA)),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report())]
summ_b21 = cds.summarize(o_b21, Q_SMALL)
peaks21 = [r for r in summ_b21['rows']
           if r['analyzer'] == 'SpinStructureFactorPeakAnalyzer']
check('B', '对应 estimate 的 P2 unknown：Peak 不得 direct，reason 注明'
      '限制来自前置 estimate 的 P2 审计',
      len(peaks21) == 2
      and all(r['comparability_bucket'] == 'not_direct'
              and '对应 estimate 的 P2' in r['comparability_reason']
              and 'unknown' in r['comparability_reason'] for r in peaks21),
      peaks21[0]['comparability_reason'][:60] if peaks21 else '')

# B22. 对应 estimate 的 P2 inconsistent → Peak 不得 direct（q 摘要相同
# 也不得通过）
o_b22 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(QGRID, q=QA)),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report()),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(QGRID_OTHER, q=QA)),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report())]
summ_b22 = cds.summarize(o_b22, Q_SMALL)
peaks22 = [r for r in summ_b22['rows']
           if r['analyzer'] == 'SpinStructureFactorPeakAnalyzer']
check('B', '对应 estimate 的 P2 inconsistent：Peak 不得 direct',
      len(peaks22) == 2
      and all(r['comparability_bucket'] == 'not_direct'
              and '对应 estimate 的 P2' in r['comparability_reason']
              and 'inconsistent' in r['comparability_reason']
              for r in peaks22),
      peaks22[0]['comparability_reason'][:60] if peaks22 else '')

# B23. 多个候选 estimate → 不静默选择第一个，q_grid 与关联索引保守留空
o_b23 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=QA)),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=QA)),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report())]
summ_b23 = cds.summarize(o_b23, Q_SMALL)
peak23 = [r for r in summ_b23['rows']
          if r['analyzer'] == 'SpinStructureFactorPeakAnalyzer'][0]
check('B', '多个候选 estimate：不静默选择第一个，保守留空',
      peak23['q_grid_json'] == ''
      and peak23['peak_estimate_entry_index'] == '')

# B24. N 不同：Peak 不复用前置 estimate 的 q 摘要，不得 direct
o_b24 = [outcome('d1_10', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=QA), N=200),
         outcome('d1_10', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report(), N=400),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorEstimator', '0',
                 'sz_sz', spin_sf_report(q=QA), N=200),
         outcome('d1_20', SPIN_OBS, 'SpinStructureFactorPeakAnalyzer', '0',
                 'sz_sz', peak_report(), N=400)]
summ_b24 = cds.summarize(o_b24, Q_SMALL)
peaks24 = [r for r in summ_b24['rows']
           if r['analyzer'] == 'SpinStructureFactorPeakAnalyzer']
check('B', 'N 不同：Peak 关联键校验生效，q 摘要留空且 not_direct',
      len(peaks24) == 2
      and all(r['q_grid_json'] == ''
              and r['peak_estimate_entry_index'] == ''
              and r['comparability_bucket'] == 'not_direct' for r in peaks24))

# ==========================================================================
# C. 输出：严格 JSON / CSV 保守空值 / 确定性覆盖
# ==========================================================================
print('===== C. 输出 =====')
summ_c = cds.summarize(outcomes_a, Q_SMALL)
outdir_c = os.path.join(tempfile.mkdtemp(prefix='p43_out_'), 'nested', 'out')
paths_c = cds.write_outputs(summ_c, outdir_c)
check('C', '输出目录自动创建（含嵌套）且两文件生成',
      os.path.isfile(paths_c['csv']) and os.path.isfile(paths_c['json']))
raw_json = open(paths_c['json'], encoding='utf-8').read()
check('C', '严格 JSON：无 NaN/Infinity 字面量且可解析（NaN 转 null）',
      'NaN' not in raw_json and 'Infinity' not in raw_json
      and json.loads(raw_json)['rows'][6]['fit_quality_json']
      == '{"r2": null}')
with open(paths_c['csv'], encoding='utf-8-sig', newline='') as f:
    import csv
    lines = list(csv.DictReader(f))
check('C', 'CSV：固定列序与行数（不追加旧记录）',
      list(lines[0].keys()) == list(cds.CSV_COLUMNS)
      and len(lines) == len(outcomes_a), f'{len(lines)} 行')
check('C', 'CSV：NaN/None 保守空值、unknown 标签显式',
      lines[6]['fit_r_min'] == '' and lines[1]['data_restricted'] == 'unknown'
      and lines[2]['failed_message'] != '')
paths_c2 = cds.write_outputs(cds.summarize(outcomes_a, Q_SMALL), outdir_c)
b1 = open(paths_c['csv'], 'rb').read() == open(paths_c2['csv'], 'rb').read()
b2 = open(paths_c['json'], 'rb').read() == open(paths_c2['json'], 'rb').read()
check('C', '重复执行：输出字节一致（确定性覆盖，不追加）',
      b1 and b2 and len(json.load(open(paths_c2['json'], encoding='utf-8'))['rows'])
      == len(outcomes_a))

# ==========================================================================
# D. 无物理结论（不判趋势/单调性/相态）
# ==========================================================================
print('===== D. 无物理结论 =====')
summ_d = cds.summarize(outcomes_a, Q_SMALL)
paths_d = cds.write_outputs(summ_d, tempfile.mkdtemp(prefix='p43_nophys_'))
doc_d = json.load(open(paths_d['json'], encoding='utf-8'))
check('D', 'JSON 顶层不含趋势/单调性/相态结论字段',
      not any(k in doc_d for k in ('trend', 'monotonic', 'phase_conclusion',
                                   'phase_diagram', 'physics_conclusion')))
check('D', '语义说明块存在且明确不判物理结论',
      'no_physical_conclusions' in doc_d.get('semantics', {}))
check('D', '所有行 comparability_bucket 均在四档枚举内',
      all(r['comparability_bucket'] in cds._BUCKETS
          for r in doc_d['rows']))
check('D', '行字段集恰为固定列集（无隐藏趋势字段）',
      set(doc_d['rows'][0].keys()) == set(cds.CSV_COLUMNS))
pr = doc_d['phase_report']
check('D', '结论仅 evidence_collected（P1 语义，非物理结论）',
      pr['conclusion'] == 'evidence_collected')
check('D', 'sufficiency overall=unknown（rules=None 如实记录）',
      pr['sufficiency']['overall'] == 'unknown'
      and 'rules=None' in pr['sufficiency']['overall_note'])
import io
import contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cds._print_console(summ_d, paths_d, {'N': 200, 'bond_dim': '1500'})
check('D', '控制台摘要可运行且不含趋势结论/相态词汇（仅边界声明中出现'
      '"趋势/相态"字样）',
      all(w not in buf.getvalue() for w in ('单调', '递增', '递减', 'AFM',
                                            'CDW', '超导相', '相变', 'PDW')))

# ==========================================================================
# E. mock 集成（fake run_analysis + fake loaders + 临时目录树）
# ==========================================================================
print('===== E. mock 集成 =====')


def fake_run(observable, *, operation=None, config=None, **kwargs):
    blob = str(kwargs)
    if 'd_miss' in blob:
        raise FileNotFoundError(f'模拟缺失文件（路径含 d_miss）: {blob[:50]}')
    if observable == SC_OBS:
        return sc_report()
    if observable == SPIN_OBS and operation == 'estimate':
        return spin_sf_report()
    if observable == SPIN_OBS and operation == 'peak_analysis':
        return peak_report()
    if observable == CHARGE_OBS:
        return charge_report()
    raise AssertionError(f'意外调用: {observable}/{operation}')


def fake_sc_loader(_dir, _rb, _key, _N, _a):
    if 'd_miss' in str(_dir):
        raise FileNotFoundError('模拟 super_corr 文件缺失')
    return ['fake_sc_recs']


def fake_spin_loader(_dir, _N, _a):
    if 'd_miss' in str(_dir):
        raise FileNotFoundError('模拟 spin 关联文件缺失')
    return ['fake_spin_recs']


fake_loaders = cds.Loaders(super_corr=fake_sc_loader,
                           spin_dot=fake_spin_loader)

tmp_root = os.path.join(tempfile.mkdtemp(prefix='p43_tree_'), 'root')
for d in ('d1', 'd_miss'):
    os.makedirs(os.path.join(
        tmp_root, 't_J_3', d, '1500_measure_phys_qty', 'measure_49'))
    os.makedirs(os.path.join(
        tmp_root, 't_J_3', d, '1500_measure_phys_qty', 'measure_50'))
os.makedirs(os.path.join(
    tmp_root, 't_J_3', 'd1', '1500_measure_super_corr', 'super_corr_0_1'))

orig_run = cds.run_analysis
cds.run_analysis = fake_run
try:
    outs_d1 = cds.analyze_doping(
        'd1', root=tmp_root, t_j='t_J_3', bond_dim='1500', N=200, a=1.0,
        q_list=Q_SMALL, loaders=fake_loaders)
    outs_miss = cds.analyze_doping(
        'd_miss', root=tmp_root, t_j='t_J_3', bond_dim='1500', N=200, a=1.0,
        q_list=Q_SMALL, loaders=fake_loaders)
    # 参考键过滤：目录只有 super_corr_0_1，过滤 9_9 → 无 SC 条目
    outs_filt = cds.analyze_doping(
        'd1', root=tmp_root, t_j='t_J_3', bond_dim='1500', N=200, a=1.0,
        q_list=Q_SMALL, loaders=fake_loaders, refbonds={'9_9'})
finally:
    cds.run_analysis = orig_run

check('E', 'd1：8 条全部成功（3 SC + 2 spin SF + 2 peak + 1 charge）',
      len(outs_d1) == 8 and all(o['failure'] is None for o in outs_d1),
      f'{len(outs_d1)} 条')
check('E', 'd_miss：6 条显式失败（目录缺失 + 文件缺失 + 依赖失败）',
      len(outs_miss) == 6
      and all(o['report'] is None and o['failure'] is not None
              for o in outs_miss),
      [o['failure']['type'] for o in outs_miss])
types_miss = [o['failure']['type'] for o in outs_miss]
check('E', 'd_miss：失败类型包含 FileNotFoundError 与 DependencyFailed',
      'FileNotFoundError' in types_miss and 'DependencyFailed' in types_miss)
check('E', '参考键过滤生效：过滤后 SC 条目为 0、其余分析照常',
      len(outs_filt) == 5
      and all(o['context']['observable'] != SC_OBS for o in outs_filt))

summ_e = cds.summarize(outs_d1 + outs_miss, Q_SMALL)
rows_e = summ_e['rows']
check('E', '汇总行数 14、标签唯一（无重复记录）',
      len(rows_e) == 14 and len({(r['doping'], r['observable'], r['analyzer'],
                                  r['refbond_or_source'],
                                  r['key_or_definition']) for r in rows_e}) == 14)
miss_rows = [r for r in rows_e if r['doping'] == 'd_miss']
check('E', 'd_miss 行全部 status=failed 且带失败原因（不静默跳过）',
      all(r['status'] == 'failed' and r['failed_message']
          for r in miss_rows), f'{len(miss_rows)} 行')
d1_sc = [r for r in rows_e
         if r['doping'] == 'd1' and r['observable'] == SC_OBS]
check('E', '单 doping 三键方向：cautious（已知差异：键方向不同）',
      len(d1_sc) == 3
      and all(r['comparability_bucket'] == 'cautious' for r in d1_sc))
check('E', 'd_miss 行：not_direct（无数值参与并列比较）',
      all(r['comparability_bucket'] == 'not_direct' for r in miss_rows))
chg_e = [r for r in rows_e
         if r['observable'] == CHARGE_OBS and r['doping'] == 'd1']
chg_miss = [r for r in miss_rows if r['observable'] == CHARGE_OBS]
check('E', 'mock 集成：身份来源标记（SC=refbond_dir、成功电荷=payload_field、'
      '失败电荷=空）',
      all(r['source_identity_kind'] == 'refbond_dir' for r in d1_sc)
      and chg_e and chg_e[0]['source_identity_kind'] == 'payload_field'
      and chg_miss and chg_miss[0]['source_identity_kind'] == '')

outdir_e = tempfile.mkdtemp(prefix='p43_mock_out_')
paths_e1 = cds.write_outputs(summ_e, outdir_e)
paths_e2 = cds.write_outputs(cds.summarize(outs_d1 + outs_miss, Q_SMALL),
                             outdir_e)
same = (open(paths_e1['csv'], 'rb').read() == open(paths_e2['csv'], 'rb').read()
        and open(paths_e1['json'], 'rb').read()
        == open(paths_e2['json'], 'rb').read())
check('E', 'mock 集成：重复执行字节一致（不追加旧记录）', same)

# CLI main() 端到端（fake run 仍生效）
cds.run_analysis = fake_run
try:
    rc = cds.main(['--root', tmp_root, '--dopings', 'd1,d_miss',
                   '--N', '200', '--n-qx', '5', '--n-qy', '2',
                   '--outdir', os.path.join(outdir_e, 'cli')])
finally:
    cds.run_analysis = orig_run
cli_json = os.path.join(outdir_e, 'cli', 'cross_doping_summary.json')
check('E', 'CLI main() 返回 0 且生成输出文件',
      rc == 0 and os.path.isfile(cli_json))

# ==========================================================================
# F. 真实数据 smoke（数据目录存在时运行；d1_100 + 参考键 49_50）
# ==========================================================================
print('===== F. 真实数据 smoke =====')
REAL_ROOT = os.path.join(ROOT, 'Data_N200')
if os.path.isdir(os.path.join(REAL_ROOT, 't_J_3', 'd1_100')):
    qs200, _ = cds.xc_momentum_grid(200, a=1.0, n_qx=11, n_qy=4)
    outs_real = cds.analyze_doping(
        'd1_100', root=REAL_ROOT, t_j='t_J_3', bond_dim='1500',
        N=200, a=1.0, q_list=qs200, refbonds={'49_50'})
    check('F', '真实 d1_100：8 条分析全部运行（3 SC + 2+2 自旋 + 1 电荷）',
          len(outs_real) == 8, f'{len(outs_real)} 条')
    ok_real = [o for o in outs_real if o['failure'] is None]
    check('F', '真实 d1_100：无失败条目（status 为 ok/inconclusive 合法）',
          len(ok_real) == 8
          and all(o['report'].status in ('ok', 'inconclusive')
                  for o in ok_real),
          [o['failure']['type'] if o['failure'] else o['report'].status
           for o in outs_real])
    summ_real = cds.summarize(
        outs_real, qs200, config={'root': REAL_ROOT, 't_j': 't_J_3',
                                  'dopings': ['d1_100'], 'bond_dim': '1500',
                                  'N': 200})
    paths_real = cds.write_outputs(
        summ_real, os.path.join(ART, 'p43_smoke'))
    rows_real = summ_real['rows']
    r_sc = [r for r in rows_real
            if r['observable'] == OBSERVABLE_SUPERCONDUCTING_CORRELATION]
    check('F', '真实 SC 三键方向行：status=ok、拟合窗口与模型名存在',
          len(r_sc) == 3 and all(r['status'] == 'ok'
                                 and r['fit_model'] for r in r_sc),
          [r['fit_model'] for r in r_sc])
    r_chg = [r for r in rows_real
             if r['observable'] == OBSERVABLE_CHARGE_STRUCTURE_FACTOR]
    check('F', '真实电荷行：源点身份 49_50、restricted、q_grid 摘要存在',
          len(r_chg) == 1 and r_chg[0]['refbond_or_source'] == '49_50'
          and r_chg[0]['data_restricted'] == 'restricted'
          and r_chg[0]['q_grid_json'])
    raw = open(paths_real['json'], encoding='utf-8').read()
    check('F', '真实输出：严格 JSON 可解析、无 NaN 字面量',
          'NaN' not in raw and json.loads(raw)['rows'])
    check('F', '真实输出：单 doping 组内 SC 为 cautious（键方向差异，不做'
          '跨 doping 结论）',
          all(r['comparability_bucket'] == 'cautious' for r in r_sc))
    r_peak_real = [r for r in rows_real
                   if r['analyzer'] == 'SpinStructureFactorPeakAnalyzer']
    check('F', '真实峰值行：q_grid 摘要来自前置 estimate 的 q 坐标'
          '（非全局假设）',
          len(r_peak_real) == 2 and all(r['q_grid_json']
                                        for r in r_peak_real))
else:
    check('F', '真实数据 smoke（数据目录不存在，跳过）', True,
          '跳过（Data_N200 不存在）')

# ==========================================================================
# 汇总
# ==========================================================================
print()
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in RESULTS:
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    print(f'{cat:6}: {n_pass}/{len(RESULTS[cat])} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)
