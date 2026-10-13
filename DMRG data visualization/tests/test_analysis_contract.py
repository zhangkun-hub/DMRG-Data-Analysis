# -*- coding: utf-8 -*-
"""
Analysis_Contract 测试（D3.1 统一结果契约，纯增量层）

合成数据覆盖：
- 三个适配器的字段映射（decay / structure_factor / peak_report）
- INCONCLUSIVE 分类与计算状态 status 的区分
- data_restricted 的 True / False / None 三种情况
- 适配器不修改原始结果对象
- numpy 标量/数组/嵌套结构与非有限浮点的严格 JSON 序列化（无 NaN 字面量）
- 默认排除 payload，显式 include_payload=True 才序列化
- 输出中不存在绝对路径
- 不同分析器专有字段仍可从 payload（原始结果）中获取

不依赖真实数据与既有基线；临时 JSON 写入系统临时目录，不落仓库。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_analysis_contract.py
"""
import sys
import os
import json
import copy
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, _convert,
    STATUS_OK, STATUS_INCONCLUSIVE, STATUS_FAILED,
    ANALYZER_DECAY_SUPERCONDUCTING, ANALYZER_DECAY_SPIN,
    ANALYZER_STRUCTURE_FACTOR, ANALYZER_PEAK, ANALYZER_ENERGY,
    OBSERVABLE_SUPERCONDUCTING_CORRELATION, OBSERVABLE_SPIN_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR, OBSERVABLE_ENERGY,
    from_decay_result, from_structure_factor, from_peak_report,
    from_energy_extrapolation,
)
from Physical_Quantity.Correlation_Decay_Analyzer import (
    DecayAnalysisResult, FitResult, OscillationInfo, StabilityReport,
    AnalyzerConfig)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def make_decay_result(classification='power_law', evidence='moderate'):
    """合成 DecayAnalysisResult（真实 dataclass，字段取自实际输出结构）"""
    return DecayAnalysisResult(
        classification=classification,
        recommended_model=classification,
        evidence_level=evidence,
        n_points=12, n_distances=12, n_raw_points=24,
        r_min=3.0, r_max=36.0,
        envelope_fit=FitResult(model='envelope_power', fit_space='log_absC',
                               r2_space='log', params={'A': 1.0, 'K': np.float64(2.5)},
                               param_errors={'A': 0.1, 'K': 0.2}, r2=0.95),
        oscillation=OscillationInfo(Q=0.9, phi=0.3, evidence='moderate'),
        stability=StabilityReport(verdict='stable', recommended_r_min=3.0),
        analysis_version='1.0',
        warnings=['测试警告 w1', '测试警告 w2'],
        config=AnalyzerConfig(),
    )


def make_sf_dict(restricted=True):
    """合成结构因子 dict（字段取自 compute_structure_factor L132-147）"""
    q = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    S = np.array([0.5 + 0.01j, 0.2 + 0j, 0.1 + 0j], dtype=complex)
    return {'q': q, 'S_q': S, 'definition': 'sz_sz', 'N': 8,
            'measured_pair_count': 3, 'expected_pair_count': None,
            'pair_coverage': None, 'source_coverage': 3 / 8,
            'missing_pairs': None, 'is_restricted': restricted,
            'imaginary_residual': 0.02, 'duplicate_pairs': 1,
            'invalid_values': 0, 'skipped_no_coords': [(99, 1)],
            'meta': {'geometry': 'HoneycombXCGeometry (XC cylinder, 8-site basis)',
                     'N': 8, 'Ly': 3.4641016151377544,
                     'boundary_conditions': {'x': 'open', 'y': 'periodic'}}}


def make_peak_dict(status='ok', n_peaks=1, restricted=True):
    """合成峰值报告 dict（字段取自 find_peaks L213-230）"""
    return {'status': status, 'status_reason': '',
            'signal_used': 'abs', 'n_peaks': n_peaks,
            'peaks': [{'q_original': (0.0, 0.0), 'q_refined': (0.01, 0.0),
                       'height': np.float64(0.9), 'refine_status': 'refined',
                       'boundary': False, 'fwhm_qx': 0.3, 'fwhm_qy': 0.4,
                       'resolution_limited': False}] if n_peaks else [],
            'resolution': {'dq_x': 0.1, 'dq_y': 0.2, 'grid_uniform': True},
            'max_signal': 0.9, 'invalid_signal_points': 0,
            'is_restricted': restricted}


# ==========================================================================
# A. 适配器字段映射
# ==========================================================================
print('===== A. 适配器字段映射 =====')
res = make_decay_result()
rep = from_decay_result(res, ANALYZER_DECAY_SUPERCONDUCTING,
                        OBSERVABLE_SUPERCONDUCTING_CORRELATION)
check('A', 'decay 适配器：analyzer/observable/status 正确',
      rep.analyzer == ANALYZER_DECAY_SUPERCONDUCTING
      and rep.observable == OBSERVABLE_SUPERCONDUCTING_CORRELATION
      and rep.status == STATUS_OK)
check('A', 'decay 适配器：classification/evidence 直通',
      rep.classification == 'power_law' and rep.evidence == 'moderate')
check('A', 'decay 适配器：parameters 为包络参数副本且含 K',
      rep.parameters == {'A': 1.0, 'K': 2.5} and isinstance(rep.parameters, dict))
check('A', 'decay 适配器：warnings/metadata（含 config 快照与 verdict）',
      rep.warnings == ['测试警告 w1', '测试警告 w2']
      and rep.metadata['analysis_version'] == '1.0'
      and rep.metadata['verdict'] == 'stable'
      and isinstance(rep.metadata['config'], dict))

sf = make_sf_dict()
rep_sf = from_structure_factor(sf)
check('A', 'sf 适配器：analyzer/observable/status',
      rep_sf.analyzer == ANALYZER_STRUCTURE_FACTOR
      and rep_sf.observable == OBSERVABLE_SPIN_STRUCTURE_FACTOR
      and rep_sf.status == STATUS_OK)
check('A', 'sf 适配器：classification=None（不编造模型分类）',
      rep_sf.classification is None and rep_sf.evidence is None)
check('A', 'sf 适配器：warnings 由实际计数构造（duplicate/skipped）',
      len(rep_sf.warnings) == 2 and any('重复对' in w for w in rep_sf.warnings)
      and any('缺坐标' in w for w in rep_sf.warnings))
check('A', 'sf 适配器：metadata 含 definition/N/覆盖计数/几何 meta',
      rep_sf.metadata['definition'] == 'sz_sz' and rep_sf.metadata['N'] == 8
      and rep_sf.metadata['source_coverage'] == 3 / 8
      and rep_sf.metadata['meta']['boundary_conditions'] == {'x': 'open', 'y': 'periodic'})

pk = make_peak_dict()
rep_pk = from_peak_report(pk, sf)
check('A', 'peak 适配器：analyzer/status/observable',
      rep_pk.analyzer == ANALYZER_PEAK and rep_pk.status == STATUS_OK
      and rep_pk.observable == OBSERVABLE_SPIN_STRUCTURE_FACTOR)
check('A', 'peak 适配器：parameters 峰摘要（n_peaks/峰位/峰高）',
      rep_pk.parameters['n_peaks'] == 1
      and rep_pk.parameters['peak_positions'] == [[0.0, 0.0]]
      and rep_pk.parameters['peak_heights'] == [0.9])
check('A', 'peak 适配器：metadata 含 signal_used/grid_uniform',
      rep_pk.metadata['signal_used'] == 'abs'
      and rep_pk.metadata['grid_uniform'] is True)

rep_e = from_energy_extrapolation(-1.499, [24, 48, 96], power=1, threshold=2.0)
check('A', 'energy 包装：E_inf/N 列表入 parameters，classification/evidence=None',
      rep_e.parameters['E_inf'] == -1.499
      and rep_e.parameters['N_values'] == [24, 48, 96]
      and rep_e.classification is None and rep_e.evidence is None)

# ==========================================================================
# B. INCONCLUSIVE 分类与 status 的区分
# ==========================================================================
print('===== B. INCONCLUSIVE 与 status 区分 =====')
res_inc = make_decay_result(classification='INCONCLUSIVE', evidence='insufficient')
rep_inc = from_decay_result(res_inc, ANALYZER_DECAY_SPIN, OBSERVABLE_SPIN_CORRELATION)
check('B', 'classification=INCONCLUSIVE 时 status 仍为 ok（计算完成，模型不可判定）',
      rep_inc.status == STATUS_OK and rep_inc.classification == 'INCONCLUSIVE')
check('B', 'None 输入 → status=failed（不抛异常）',
      from_decay_result(None, ANALYZER_DECAY_SPIN, OBSERVABLE_SPIN_CORRELATION).status == STATUS_FAILED
      and from_structure_factor(None).status == STATUS_FAILED
      and from_peak_report(None).status == STATUS_FAILED)
pk_flat = make_peak_dict(status='no_peak_above_threshold', n_peaks=0)
pk_flat['status_reason'] = '无高于阈值的局部极大'
rep_flat = from_peak_report(pk_flat)
check('B', '无峰（no_peak_above_threshold）→ status=ok，原因入 warnings',
      rep_flat.status == STATUS_OK and len(rep_flat.warnings) == 1
      and rep_flat.parameters['n_peaks'] == 0)
pk_bad = make_peak_dict(status='all_invalid', n_peaks=0)
pk_bad['status_reason'] = '全部信号点为 NaN/Inf'
rep_bad = from_peak_report(pk_bad)
check('B', '信号全无效（all_invalid）→ status=inconclusive',
      rep_bad.status == STATUS_INCONCLUSIVE)

# 五种已知状态行为逐一对照（防止加固改动误伤既有映射）
for st, exp_status in [('ok', STATUS_OK),
                       ('no_peak_above_threshold', STATUS_OK),
                       ('flat', STATUS_OK),
                       ('all_zero', STATUS_INCONCLUSIVE),
                       ('all_invalid', STATUS_INCONCLUSIVE)]:
    rep_st = from_peak_report(make_peak_dict(status=st,
                                             n_peaks=1 if st == 'ok' else 0))
    check('B', f'已知峰值状态 {st} → {exp_status}（映射不变）',
          rep_st.status == exp_status, rep_st.status)

# 未知状态：保守映射 inconclusive + warnings 记录原始值（不静默吞掉）
pk_unk = make_peak_dict(status='future_new_status', n_peaks=0)
rep_unk = from_peak_report(pk_unk)
check('B', '未知峰值状态保守映射为 inconclusive',
      rep_unk.status == STATUS_INCONCLUSIVE, rep_unk.status)
check('B', '未知状态的原始值记录进 warnings',
      any('future_new_status' in w for w in rep_unk.warnings),
      rep_unk.warnings)

# ==========================================================================
# C. data_restricted 的 True / False / None
# ==========================================================================
print('===== C. data_restricted 三种情况 =====')
check('C', 'sf 受限 → data_restricted=True',
      from_structure_factor(make_sf_dict(True)).data_restricted is True)
check('C', 'sf 不受限 → data_restricted=False',
      from_structure_factor(make_sf_dict(False)).data_restricted is False)
check('C', 'decay 输出无受限性标记 → None（未知不默认 False）',
      from_decay_result(make_decay_result(), ANALYZER_DECAY_SPIN,
                        OBSERVABLE_SPIN_CORRELATION).data_restricted is None)
pk_norest = make_peak_dict()
del pk_norest['is_restricted']
check('C', 'peak 报告缺 is_restricted 且 sf_dict 亦缺 → None',
      from_peak_report(pk_norest, {}).data_restricted is None)
check('C', 'peak 报告缺 is_restricted 时回退 sf_dict 的值',
      from_peak_report(pk_norest, make_sf_dict(True)).data_restricted is True)

# ==========================================================================
# D. 原始结果不被修改
# ==========================================================================
print('===== D. 适配器不修改输入 =====')
res0 = make_decay_result()
snap0 = copy.deepcopy(res0.__dict__)
from_decay_result(res0, ANALYZER_DECAY_SPIN, OBSERVABLE_SPIN_CORRELATION)
diffs = []
def _walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            diffs.append(f'{path} 键集变')
        for k in a:
            if k in b:
                _walk(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            diffs.append(f'{path} 数组变')
    elif str(a) != str(b):
        diffs.append(f'{path}: {a!r} vs {b!r}')
_walk(snap0, copy.deepcopy(res0.__dict__), '')
check('D', 'decay 适配器调用后原对象深比较不变', not diffs, diffs[:2] or '不变')

sf0 = make_sf_dict()
snap_sf = copy.deepcopy(sf0)
from_structure_factor(sf0)
check('D', 'sf 适配器调用后原 dict 不变（数组逐元素）',
      np.array_equal(sf0['q'], snap_sf['q'])
      and np.array_equal(sf0['S_q'], snap_sf['S_q'])
      and sf0['is_restricted'] == snap_sf['is_restricted'])

# ==========================================================================
# E. 序列化：numpy/嵌套/非有限/严格 JSON
# ==========================================================================
print('===== E. 序列化 =====')
rep_ser = from_structure_factor(make_sf_dict())
rep_ser.parameters['sample_np'] = np.float64(0.125)          # numpy 标量
rep_ser.parameters['sample_arr'] = np.array([1.0, 2.0])      # numpy 数组
rep_ser.parameters['sample_nan'] = np.float64('nan')         # 非有限
rep_ser.parameters['sample_inf'] = float('inf')
rep_ser.parameters['nested'] = {'inner': [np.float32(0.5), None]}
d_ser = rep_ser.to_dict()
check('E', 'numpy 标量/数组转为 python 原生类型',
      d_ser['parameters']['sample_np'] == 0.125
      and d_ser['parameters']['sample_arr'] == [1.0, 2.0]
      and d_ser['parameters']['nested'] == {'inner': [0.5, None]})
check('E', '非有限浮点（NaN/Inf，numpy 与 python）统一转 None',
      d_ser['parameters']['sample_nan'] is None
      and d_ser['parameters']['sample_inf'] is None)
tmp = tempfile.mkdtemp(prefix='contract_test_')
json_path = os.path.join(tmp, 'report.json')
rep_ser.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
check('E', 'JSON 文件无 NaN/Infinity 字面量',
      'NaN' not in raw and 'Infinity' not in raw)
back = json.loads(raw)
check('E', 'JSON 读回：非有限为 null、嵌套结构一致',
      back['parameters']['sample_nan'] is None
      and back['parameters']['sample_arr'] == [1.0, 2.0])

# ==========================================================================
# F. payload 默认排除 / 显式包含
# ==========================================================================
print('===== F. payload 序列化控制 =====')
rep_p = from_decay_result(make_decay_result(), ANALYZER_DECAY_SPIN,
                          OBSERVABLE_SPIN_CORRELATION)
check('F', 'to_dict 默认不含 payload 键',
      'payload' not in rep_p.to_dict())
check('F', 'include_payload=True 时包含 payload',
      'payload' in rep_p.to_dict(include_payload=True))
tmp2 = tempfile.mkdtemp(prefix='contract_test_')
rep_p.to_json(os.path.join(tmp2, 'no_payload.json'))
raw2 = open(os.path.join(tmp2, 'no_payload.json'), encoding='utf-8').read()
check('F', 'to_json 默认文件不含 payload', '"payload"' not in raw2)

# ==========================================================================
# G. 无绝对路径 / 专有字段可获取
# ==========================================================================
print('===== G. 无绝对路径与 payload 访问 =====')
d_all = rep_p.to_dict(include_payload=True)
s_all = json.dumps(d_all, ensure_ascii=False, default=str)
check('G', 'to_dict(include_payload) 无绝对路径',
      'C:' not in s_all and 'Users' not in s_all and 'Desktop' not in s_all)
check('G', 'payload 保留原始对象引用（专有字段可访问）',
      rep_p.payload is not None
      and rep_p.payload.oscillation.Q == 0.9
      and rep_p.payload.envelope_fit.params['K'] == 2.5)
check('G', 'sf 专有字段从 payload 获取（S_q 复数数组）',
      rep_sf.payload['S_q'][0] == 0.5 + 0.01j)
check('G', 'peak 专有字段从 payload 获取（refine_status/fwhm）',
      rep_pk.payload['peaks'][0]['refine_status'] == 'refined'
      and rep_pk.payload['peaks'][0]['fwhm_qx'] == 0.3)

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
