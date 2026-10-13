# -*- coding: utf-8 -*-
"""
Analysis_Integration 测试（D3.2 契约集成试点：SC / Spin 衰减分析器包装）

覆盖：
- 合成数据：包装后状态、分类与拟合信息（误差棒/拟合优度独立存放）
- 真实 N200 SC 数据：集成映射字段与直接调用 analyze 的结果一致
- 真实 N200 Spin 数据：N 等专属参数经 **kwargs 正确透传
- 旧接口兼容：直接 analyze 仍返回 DecayAnalysisResult，
  且 report.payload is 原结果对象（同一引用）
- JSON 序列化：临时目录写入/读回，无 NaN/Infinity 字面量，默认无 payload
- 失败场景：空输入 ValueError 原样传播；from_decay_result(None) 行为不变
- 元数据：参数误差与拟合质量指标独立存放，不污染 parameters

不依赖绝对路径（HERE/ROOT 动态推导），不修改数据文件与回归基线。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_analysis_integration.py
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

from Physical_Quantity.Physical_Data_Record import CorrelationRecord, SuperCorrRecord
from Physical_Quantity.Correlation_Process import CorrelationProcessor
from Physical_Quantity.Super_Correlation_Process import SuperCorrProcessor
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Super_Correlation_Analyzer import SuperCorrDecayAnalyzer
from Physical_Quantity.Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer
from Physical_Quantity.Correlation_Decay_Analyzer import DecayAnalysisResult
from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, STATUS_FAILED,
    OBSERVABLE_SUPERCONDUCTING_CORRELATION, OBSERVABLE_SPIN_CORRELATION,
    from_decay_result,
)
from Physical_Quantity.Analysis_Integration import analyze_with_report

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def load_sc_n200(doping, ref, bond, N=200):
    """最小 loader（与既有测试同款路径约定）"""
    base = f'{ROOT}/Data_N200/t_J_3/{doping}/1500_measure_super_corr/super_corr_{ref}'
    recs4 = {}
    for ch in 'abcd':
        recs4[ch] = SuperCorrRecord.extract_data(
            open(f'{base}/scs{bond}{ch}_fix_refbond_{ref}.json', encoding='utf-8').read(),
            N, 1.0)
    return SuperCorrProcessor.Combined_super_correlation(
        recs4['a'], recs4['b'], recs4['c'], recs4['d'])


def load_spin_n200(measure, ver, N=200):
    """最小 loader（与既有测试同款路径约定）"""
    base = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty/{measure}'
    seq = range(1, N + 100, 4) if ver == '49' else range(2, N + 100, 4)
    sz = CorrelationRecord.extract_data(
        open(f'{base}/sz{ver}sz.json', encoding='utf-8').read(), seq, N, 1.0)
    su = CorrelationRecord.extract_data(
        open(f'{base}/sp{ver}sm.json', encoding='utf-8').read(), seq, N, 1.0)
    sd = CorrelationRecord.extract_data(
        open(f'{base}/sm{ver}sp.json', encoding='utf-8').read(), seq, N, 1.0)
    return CorrelationProcessor.Combined_spin_correlation(sz, su, sd)


# ==========================================================================
# A. 合成数据（Spin 幂律合成：快，覆盖包装核心行为）
# ==========================================================================
print('===== A. 合成数据 =====')
g = HoneycombXCGeometry(200, 1.0)
source = 50
fields = [j for j in range(2, 200, 4) if j != source]
dists = [g.get_distance(source, j) for j in fields]
rng = np.random.default_rng(7)
base_vals = np.array([0.25 * d ** (-1.5) for d in dists])
C_pow = base_vals + 0.05 * np.abs(base_vals) * rng.standard_normal(len(base_vals))
recs_pow = [CorrelationRecord(source, j, d, v) for j, d, v in zip(fields, dists, C_pow)]

rep = analyze_with_report(SpinCorrDecayAnalyzer, recs_pow,
                          OBSERVABLE_SPIN_CORRELATION, N=200)
direct = SpinCorrDecayAnalyzer.analyze(recs_pow, N=200)
check('A', '合成：status=ok 且分类合法',
      rep.status == STATUS_OK
      and rep.classification in ('power_law', 'exponential', 'non_monotonic',
                                 'oscillatory_power_law', 'oscillatory_exponential',
                                 'complex_decay', 'INCONCLUSIVE'),
      rep.classification)
check('A', '合成：parameters 与直接调用包络参数一致',
      rep.parameters == dict(direct.envelope_fit.params), rep.parameters)
check('A', '合成：metadata 含 param_errors（独立存放，不在 parameters 内）',
      'param_errors' in rep.metadata and 'param_errors' not in rep.parameters
      and rep.metadata['param_errors'] == dict(direct.envelope_fit.param_errors))
check('A', '合成：metadata.fit_quality 含实际存在指标且数值一致',
      set(rep.metadata['fit_quality']) == {'r2', 'rss', 'rmse', 'aic', 'aicc', 'bic'}
      and abs(rep.metadata['fit_quality']['r2'] - direct.envelope_fit.r2) < 1e-12)
check('A', '合成：evidence/verdict 与直接调用一致',
      rep.evidence == direct.evidence_level
      and rep.metadata['verdict'] == direct.stability.verdict)

# ==========================================================================
# B. 真实 N200 SC 数据：集成映射与直接调用一致
# ==========================================================================
print('===== B. 真实 N200 SC 数据（d1_100 / 49_50 / A 键） =====')
comb_sc = load_sc_n200('d1_100', '49_50', 'A')
rep_sc = analyze_with_report(SuperCorrDecayAnalyzer, comb_sc,
                             OBSERVABLE_SUPERCONDUCTING_CORRELATION)
direct_sc = SuperCorrDecayAnalyzer.analyze(comb_sc)
check('B', 'SC：analyzer/observable 标识正确',
      rep_sc.analyzer == 'SuperCorrDecayAnalyzer'
      and rep_sc.observable == OBSERVABLE_SUPERCONDUCTING_CORRELATION)
check('B', 'SC：classification/evidence 与直接调用一致',
      rep_sc.classification == direct_sc.classification
      and rep_sc.evidence == direct_sc.evidence_level,
      f"{rep_sc.classification}/{rep_sc.evidence}")
check('B', 'SC：直接 analyze 仍返回 DecayAnalysisResult（旧接口不变）',
      isinstance(direct_sc, DecayAnalysisResult))

# payload 引用语义：mock 分析器返回已知对象 → report.payload is 该对象
# （集成函数内部一次调用与测试另一次调用是不同的对象，必须用 mock 验证引用）
from Physical_Quantity.Correlation_Decay_Analyzer import FitResult
_mock_result = DecayAnalysisResult(
    classification='power_law', evidence_level='moderate', n_points=10,
    envelope_fit=FitResult(model='envelope_power', fit_space='log_absC',
                           r2_space='log',
                           params={'A': 1.0, 'K': 2.0},
                           param_errors={'A': 0.1}, r2=0.9, aicc=-50.0),
    stability=None, warnings=[], analysis_version='1.0', config=None)


class _MockAnalyzer:
    """mock 分析器：记录 kwargs 并返回已知结果对象"""
    received = None

    @classmethod
    def analyze(cls, record_list, config=None, **kwargs):
        cls.received = dict(kwargs)
        return _mock_result


rep_mock = analyze_with_report(_MockAnalyzer, [1, 2],
                               OBSERVABLE_SPIN_CORRELATION, N=200)
check('B', 'payload 为集成函数内部 analyze 返回的同一对象（引用一致）',
      rep_mock.payload is _mock_result and rep_mock.analyzer == '_MockAnalyzer')
check('B', '专属参数经 **kwargs 透传到分析器（N=200）',
      _MockAnalyzer.received == {'N': 200}, _MockAnalyzer.received)

# ==========================================================================
# C. 真实 N200 Spin 数据：专属参数透传
# ==========================================================================
print('===== C. 真实 N200 Spin 数据（measure_49，N=200 透传） =====')
comb_sp = load_spin_n200('measure_49', '49')
rep_sp = analyze_with_report(SpinCorrDecayAnalyzer, comb_sp,
                             OBSERVABLE_SPIN_CORRELATION, N=200)
direct_sp = SpinCorrDecayAnalyzer.analyze(comb_sp, N=200)
check('C', 'Spin：N 经 **kwargs 透传成功且结果一致',
      rep_sp.status == STATUS_OK
      and rep_sp.classification == direct_sp.classification
      and rep_sp.parameters == dict(direct_sp.envelope_fit.params),
      rep_sp.classification)

# ==========================================================================
# D. JSON 序列化（临时目录，不落仓库）
# ==========================================================================
print('===== D. JSON 序列化 =====')
tmp = tempfile.mkdtemp(prefix='integration_test_')
json_path = os.path.join(tmp, 'report.json')
rep_sc.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
check('D', 'JSON 无 NaN/Infinity 非标准字面量',
      'NaN' not in raw and 'Infinity' not in raw)
back = json.loads(raw)
check('D', '默认输出不含 payload 键，读回结构完整',
      'payload' not in back and back['status'] == 'ok'
      and back['classification'] == rep_sc.classification
      and back['metadata']['fit_quality']['aicc'] is not None)

# ==========================================================================
# E. 失败场景
# ==========================================================================
print('===== E. 失败场景 =====')
try:
    analyze_with_report(SuperCorrDecayAnalyzer, [], OBSERVABLE_SUPERCONDUCTING_CORRELATION)
    check('E', '空输入 ValueError 原样传播', False, '未抛异常')
except ValueError:
    check('E', '空输入 ValueError 原样传播', True)
check('E', 'from_decay_result(None) → status=failed（既有行为不变）',
      from_decay_result(None, 'X', 'y').status == STATUS_FAILED)

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
