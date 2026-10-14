# -*- coding: utf-8 -*-
"""
Structure Factor 契约集成测试（D3.3a：estimator / find_peaks → AnalysisReport）

覆盖：
- 合成结构因子结果的包装与 payload 引用
- N200 真实 sz_sz 数据集成（estimator 文件入口）
- N200 真实 spin_dot records 入口集成
- 结构因子报告与直接 estimator 结果的关键字段一致
- 峰值报告与直接 find_peaks 结果的关键字段一致
- ok / no_peak_above_threshold / flat / all_zero / all_invalid 状态映射
  （集成路径，用真实 find_peaks 对合成信号产生各状态）
- missing_pairs > 0 时生成警告；None（未评估）与空（无缺失）时不生成
- status='ok' 与 data_restricted=True 可同时成立（受限数据计算正常）
- 缺 is_restricted → data_restricted=None（契约规则）
- JSON 默认输出不含 payload 且可解析读回
- 失败场景异常原样传播（空文件路径）
- 旧接口：estimator / find_peaks 返回类型与字段集保持不变

区分说明：状态映射的适配器级行为已在 test_analysis_contract 锁定；
本文件的断言全部经过集成函数（包装层）的自动化验证，非代码阅读推断。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_structure_contract_integration.py
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
from Physical_Quantity.Physical_Data_Record import CorrelationRecord
from Physical_Quantity.Correlation_Process import CorrelationProcessor
from Physical_Quantity.Spin_Structure_Factor import (
    compute_structure_factor, xc_momentum_grid, SpinStructureFactorEstimator)
from Physical_Quantity.Spin_Structure_Factor_Analyzer import (
    SpinStructureFactorPeakAnalyzer)
from Physical_Quantity.Analysis_Contract import (
    STATUS_OK, STATUS_INCONCLUSIVE, STATUS_FAILED,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR)
from Physical_Quantity.Analysis_Integration import (
    structure_factor_report, peak_analysis_report)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def load_spin_records(measure, ver, N=200):
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


def make_signal_sf(S2d, qx_axis, qy_axis, definition='test', restricted=False):
    """构造规则网格结构因子 dict（字段与 compute_structure_factor 一致）"""
    nx, ny = S2d.shape
    q = np.array([(x, y) for x in qx_axis for y in qy_axis])
    return {'q': q, 'S_q': S2d.reshape(-1).astype(complex), 'definition': definition,
            'N': 8, 'measured_pair_count': nx * ny,
            'expected_pair_count': None, 'pair_coverage': None,
            'source_coverage': 0.0, 'missing_pairs': None,
            'is_restricted': restricted, 'imaginary_residual': 0.0,
            'duplicate_pairs': 0, 'invalid_values': 0, 'skipped_no_coords': []}


# ==========================================================================
# A. 合成：包装与 payload 引用 + missing_pairs 警告规则
# ==========================================================================
print('===== A. 合成与 missing_pairs 警告 =====')
g8 = HoneycombXCGeometry(8, 1.0)
qs8 = [(0.0, 0.0)]
full8 = {(i, j) for i in range(8) for j in range(8)}
partial = [(0, 1, 1.0), (2, 3, 0.5)]

# missing_pairs > 0：包装后必须追加数据质量警告
sf_missing = compute_structure_factor(g8.coords, partial, qs8, 8, 'test',
                                      expected_pairs=full8)
rep_missing = structure_factor_report(
    lambda arg, ql, N, a=1.0, **kw: sf_missing, None, qs8, 8)
check('A', 'missing_pairs>0 时追加数据质量警告',
      rep_missing.status == STATUS_OK
      and any('缺失' in w for w in rep_missing.warnings),
      rep_missing.warnings)
check('A', 'payload 为 estimator 返回的同一 dict（引用一致）',
      rep_missing.payload is sf_missing)

# missing_pairs=None（未评估）：不追加该警告
sf_noeval = compute_structure_factor(g8.coords, partial, qs8, 8, 'test')
rep_noeval = structure_factor_report(
    lambda arg, ql, N, a=1.0, **kw: sf_noeval, None, qs8, 8)
check('A', 'missing_pairs=None（未评估）不追加缺失警告',
      not any('缺失' in w for w in rep_noeval.warnings), rep_noeval.warnings)

# missing_pairs 为空列表（无缺失）：不追加该警告
sf_nomiss = compute_structure_factor(g8.coords, partial, qs8, 8, 'test',
                                     expected_pairs={(0, 1), (2, 3)})
rep_nomiss = structure_factor_report(
    lambda arg, ql, N, a=1.0, **kw: sf_nomiss, None, qs8, 8)
check('A', 'missing_pairs 为空（无缺失）不追加缺失警告',
      not any('缺失' in w for w in rep_nomiss.warnings), rep_nomiss.warnings)

# ==========================================================================
# B. N200 真实 sz_sz（文件入口）+ spin_dot（records 入口）
# ==========================================================================
print('===== B. N200 真实数据集成 =====')
sz_path = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty/measure_49/sz.json'
qs200, _ = xc_momentum_grid(200, a=1.0, n_qx=21, n_qy=4)

rep_sz = structure_factor_report(
    SpinStructureFactorEstimator.estimate_sz_sz_from_file, sz_path, qs200, 200)
direct_sz = SpinStructureFactorEstimator.estimate_sz_sz_from_file(sz_path, qs200, 200)
check('B', 'sz_sz：报告与直接 estimator 的关键字段一致',
      rep_sz.metadata['definition'] == direct_sz['definition']
      and rep_sz.metadata['measured_pair_count'] == direct_sz['measured_pair_count']
      and rep_sz.data_restricted is direct_sz['is_restricted'],
      f"n={rep_sz.metadata['measured_pair_count']}")
check('B', 'sz_sz：status=ok 与 data_restricted=True 同时成立（两字段正交）',
      rep_sz.status == STATUS_OK and rep_sz.data_restricted is True,
      f"{rep_sz.status}/{rep_sz.data_restricted}")
check('B', 'sz_sz：payload 引用 estimator 原始 dict',
      rep_sz.payload is not None and isinstance(rep_sz.payload, dict)
      and 'S_q' in rep_sz.payload)
check('B', 'sz_sz：observable 标识正确',
      rep_sz.observable == OBSERVABLE_SPIN_STRUCTURE_FACTOR
      and rep_sz.analyzer == 'SpinStructureFactorEstimator')

recs = load_spin_records('measure_49', '49')
rep_dot = structure_factor_report(
    SpinStructureFactorEstimator.estimate_spin_dot_from_records, recs, qs200, 200)
direct_dot = SpinStructureFactorEstimator.estimate_spin_dot_from_records(recs, qs200, 200)
check('B', 'spin_dot：records 入口集成成功且定义正确',
      rep_dot.status == STATUS_OK
      and rep_dot.metadata['definition'] == direct_dot['definition']
      and rep_dot.data_restricted is True,
      rep_dot.metadata['definition'])
check('B', 'spin_dot：与直接 estimator 的覆盖计数一致',
      rep_dot.metadata['measured_pair_count'] == direct_dot['measured_pair_count']
      and rep_dot.metadata['source_coverage'] == direct_dot['source_coverage'])

# ==========================================================================
# C. 峰值报告：真实数据 + 五种状态（集成路径）
# ==========================================================================
print('===== C. 峰值报告集成 =====')
rep_peak = peak_analysis_report(direct_sz)
direct_peak = SpinStructureFactorPeakAnalyzer.find_peaks(direct_sz)
check('C', '真实 sz_sz 峰值报告与直接 find_peaks 关键字段一致',
      rep_peak.status == STATUS_OK
      and rep_peak.parameters['n_peaks'] == direct_peak['n_peaks']
      and rep_peak.metadata['grid_uniform'] == direct_peak['resolution']['grid_uniform'],
      f"n_peaks={rep_peak.parameters['n_peaks']}")
check('C', '峰值报告 payload 引用 find_peaks 原始 dict',
      rep_peak.payload is not None and isinstance(rep_peak.payload, dict)
      and 'peaks' in rep_peak.payload)
check('C', '峰值细节（q_refined）仍经 payload 访问',
      all('q_refined' in p for p in rep_peak.payload['peaks']))

qx = np.linspace(-1.5, 1.5, 15)
qy = np.linspace(-1.5, 1.5, 13)
XX, YY = np.meshgrid(qx, qy, indexing='ij')
# ok：高斯峰（8 邻域局部极大 + 高度筛选通过）
sf_ok = make_signal_sf(np.exp(-(XX ** 2 + YY ** 2) / 0.3), qx, qy)
rep_ok = peak_analysis_report(sf_ok)
check('C', '合成高斯：状态 ok 且 n_peaks≥1',
      rep_ok.status == STATUS_OK and rep_ok.parameters['n_peaks'] >= 1,
      f"n_peaks={rep_ok.parameters['n_peaks']}")
# no_peak_above_threshold：|S| 平顶信号（signal 默认 'abs'，故须在模长
# 空间平顶：中心多点同为全局最大且互为等值邻居 → 无任何严格局部极大；
# 外围非负单调到 0，避免 abs 反弹出边界峰）
_R = np.sqrt(XX ** 2 + YY ** 2)
_slope = np.minimum(1.0, np.maximum(0.0, 1.3 - _R))
sf_slope = make_signal_sf(_slope, qx, qy)
rep_slope = peak_analysis_report(sf_slope)
check('C', '平顶信号：无峰但计算正常 → status=ok，原因入 warnings',
      rep_slope.status == STATUS_OK
      and rep_slope.parameters['n_peaks'] == 0
      and len(rep_slope.warnings) >= 1, rep_slope.warnings[:1])
# flat：常数信号
sf_flat = make_signal_sf(np.ones((len(qx), len(qy))), qx, qy)
rep_flat = peak_analysis_report(sf_flat)
check('C', '常数信号：flat → status=ok',
      rep_flat.status == STATUS_OK and rep_flat.parameters['n_peaks'] == 0)
# all_zero：全零信号
sf_zero = make_signal_sf(np.zeros((len(qx), len(qy))), qx, qy)
rep_zero = peak_analysis_report(sf_zero)
check('C', '全零信号：all_zero → status=inconclusive',
      rep_zero.status == STATUS_INCONCLUSIVE)
# all_invalid：全 NaN 信号
sf_nan = make_signal_sf(np.full((len(qx), len(qy)), np.nan), qx, qy)
rep_nan = peak_analysis_report(sf_nan)
check('C', '全 NaN 信号：all_invalid → status=inconclusive',
      rep_nan.status == STATUS_INCONCLUSIVE)

# ==========================================================================
# D. data_restricted 缺失回退 + JSON + 失败传播 + 旧接口
# ==========================================================================
print('===== D. 缺失字段 / JSON / 失败 / 旧接口 =====')
sf_norest = make_signal_sf(np.exp(-(XX ** 2 + YY ** 2) / 0.3), qx, qy)
del sf_norest['is_restricted']
rep_norest = peak_analysis_report(sf_norest)
check('D', '缺 is_restricted（且 sf_dict 亦缺）→ data_restricted=None',
      rep_norest.data_restricted is None)

tmp = tempfile.mkdtemp(prefix='sf_contract_')
json_path = os.path.join(tmp, 'report.json')
rep_sz.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
back = json.loads(raw)
check('D', 'JSON：无 NaN/Infinity 字面量、默认无 payload、可解析读回',
      'NaN' not in raw and 'Infinity' not in raw
      and '"payload"' not in raw and back['status'] == 'ok'
      and back['data_restricted'] is True)

try:
    structure_factor_report(
        SpinStructureFactorEstimator.estimate_sz_sz_from_file,
        f'{ROOT}/Data_N200/does_not_exist.json', qs200, 200)
    check('D', '空/不存在文件路径异常原样传播', False, '未抛异常')
except FileNotFoundError:
    check('D', '空/不存在文件路径异常原样传播', True)

# 旧接口：直接调用返回类型与字段集不变
direct_types = (isinstance(direct_sz, dict)
                and set(direct_sz.keys()) == {
                    'q', 'S_q', 'definition', 'N', 'measured_pair_count',
                    'expected_pair_count', 'pair_coverage', 'source_coverage',
                    'missing_pairs', 'is_restricted', 'imaginary_residual',
                    'duplicate_pairs', 'invalid_values', 'skipped_no_coords',
                    'meta', 'q_grid'})
check('D', '旧接口：estimator 返回 dict 且字段集不变', direct_types)
check('D', '旧接口：find_peaks 返回 dict 且含 peaks/resolution',
      isinstance(direct_peak, dict) and 'peaks' in direct_peak
      and 'resolution' in direct_peak)

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
