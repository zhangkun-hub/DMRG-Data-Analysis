# -*- coding: utf-8 -*-
"""
PairingSymmetryAnalyzer（D5 阶段 1–4：数据装配 + 可比性质量门 +
公共窗口振幅重拟合与两级矩阵模型 + 有符号证据通道与三通道状态机）

依据设计文档 docs/PairingSymmetryAnalyzer_Design.md（v2）、D3 接口审计、
D4 合成验证与 D5 3.1 审计。本模块当前实现：

- 阶段 1：SC 数据装配（参考键目录发现、四通道加载与合成、输入完整性
  检查与逐问题记录）
- 阶段 2：可比性质量门（K/xi 模型分流、振荡证据分层 osc/non_osc/
  unknown——5B-1 落实设计文档 §5.2"分别处理"、公共窗口内带符号过零
  证据、层内公共拟合窗口与有效点数、SU(2) 检查记录）与可比较元素
  集合构建（各层独立集合，跨层比较默认拒绝）
- 阶段 3：层内公共窗口包络振幅重拟合（|C| 空间，复用
  fit_envelope_logspace）、一般秩一模型（σ·u·v^T，球面参数化）、
  对称秩一模型（m·m^T，m≥0 非负约束——振幅矩阵语义）与可辨识性诊断
  （数据覆盖度、尺度自适应 Jacobian 秩、奇异值/条件数原因分类、绝对
  与相对残差、模型比较 AICc/BIC 参考统计量与退化状态）；osc 层附
  振幅偏差警告（5B-0 合成实验证据，非普适误差界）
- 阶段 4：有符号证据通道（analyze_sign_channel，输入=带符号原始记录，
  与振幅矩阵完全独立）+ 三通道状态机（run_stage4：magnitude 结构证据
  / sign 逐元素符号证据 / phase blocked_by_data，互不覆盖）

明确不实现（后续阶段）：记录级 bootstrap、联合协方差、幅度候选支持
判定（supported_candidate 恒 None 且 supported_candidate_status 恒
'not_evaluated'，P0 定稿）、跨元素符号结构推断（逐对积已由 5B-3 提供，
模式级解释未实现）、d 波/s 波/d+id/ODLRO 判定、Analysis_Entry 路由集成。

P0 语义收口（三通道定稿）：
- magnitude：振幅矩阵观测 + 秩一结构诊断 + 模板距离描述（数值事实），
  不含统计支持结论——candidate_distances 不是支持度/概率/显著性
- sign：逐元素符号为原始观测；层内逐对积 r_αβ=s_α·s_β 为派生观测
  （5B-3，仅供描述、可作未来候选匹配的输入）；跨元素模式到配对表示
  的规则化解释未实现（cross_element_sign_structure='not_implemented'）
- phase：blocked_by_data = 数据能力限制（实值数据无复相位信息），
  不是算法失败、不是拟合失败、不是候选匹配失败
- PhaseDiagnosis 消费约定（最小证据类别，不新增 AnalysisReport 字段）：
  magnitude_observation（振幅矩阵/秩一/距离）、sign_observation
  （逐元素符号/逐对积）、phase_capability（blocked_by_data）、
  quality_diagnostic（checks/blocking_reasons/warnings）、
  analysis_limitation（supported_candidate_status='not_evaluated'、
  cross_element_sign_structure='not_implemented'）——由未来接入层
  映射，本模块不改契约

关键语义：
- 包络振幅 A 取自 |C(r)| 对数空间拟合（非负）：只能用于振幅矩阵结构
  分析，**不携带配对符号信息**；符号通道只消费带符号记录
- 对称模型采用 m ≥ 0 参数化（振幅非负语义）；m 的符号无物理含义
- 条件数、秩一残差阈值等未校准量只报告数值与定性警告，不作生产判据
- 模型比较 AICc/BIC 只作参考统计量；残差达数值精度量级时标注退化
- Jacobian 数值秩用相对容差（1e-6 × 最大奇异值），跨数量级缩放不变

设计约束（与设计文档一致）：
- 复用既有提取与合成原语（SuperCorrRecord.extract_data /
  SuperCorrProcessor.Combined_super_correlation / symmetry_check /
  fit_envelope_logspace），不修改既有模块
- 四类状态严格区分：文件缺失（missing）/ 文件存在但记录不完整
  （incomplete）/ 输入完整但质量门拒绝（quality gate fail）/
  分析执行失败（analysis_failed，捕获 FileNotFoundError/ValueError，
  其余异常原样传播——与 Cross_Doping_Summary._try 语义一致）
- 未经校准的阈值（公共窗口最小点数、SU(2) 误差上限、秩一残差阈值、
  符号通道最小点数等）以 None 表示"待校准"，输出中显式标记
- 参考键类别由键向量线方向角导出（60°→A、120°→B、0°→C），不硬编码
  具体参考键索引
"""

import ast
import math
import os

import numpy as np
from scipy.optimize import least_squares

from ..data_records.Honeycomb_Lattice import HoneycombXCGeometry
from ..data_records.Physical_Data_Record import SuperCorrRecord
from .Super_Correlation_Process import SuperCorrProcessor
from .Super_Correlation_Analyzer import SuperCorrDecayAnalyzer
from .Correlation_Decay_Analyzer import AnalyzerConfig, CorrelationDecayAnalyzer

# ---- 完整性/装配状态（与设计文档 6.1 语义对齐，不含判定类状态） ----
INTEGRITY_OK = 'ok'
INTEGRITY_MISSING = 'missing'          # 文件/目录缺失（FileNotFoundError 级）
INTEGRITY_INCOMPLETE = 'incomplete'    # 文件存在但记录不完整/不合法
ANALYSIS_OK = 'ok'
ANALYSIS_FAILED = 'analysis_failed'    # 数据级异常（FileNotFoundError/ValueError）

# ---- 完整性问题类别（逐项记录，含位置信息） ----
ISSUE_MISSING_FILE = 'missing_file'
ISSUE_MISSING_CHANNEL = 'missing_channel'
ISSUE_REFBOND_MISMATCH = 'refbond_mismatch'
ISSUE_DUPLICATE_KEY = 'duplicate_key'
ISSUE_NON_FINITE = 'non_finite'
ISSUE_EMPTY_RECORDS = 'empty_records'
ISSUE_FORMAT = 'format_error'

# 合法键方向（与测量程序 A/B/C 一致）
VALID_DIRECTIONS = ('A', 'B', 'C')
CHANNELS = ('a', 'b', 'c', 'd')


class PairingSymmetryConfig:
    """阶段 1–3 配置。None = 待校准（输出显式标记，不伪装成已验证）。"""

    def __init__(self, min_common_points=None, su2_error_max=None,
                 analyzer_config=None):
        # 公共窗口内最小有效点数。None = 待校准；默认阈值来源见
        # gate_common_window docstring（n_min_fit 依据）
        self.min_common_points = min_common_points
        # SU(2) 相对误差上限（a↔d、b↔c 中较大者）。
        # None = 待校准：只记录误差，不判通过/拒绝
        self.su2_error_max = su2_error_max
        # 透传给衰减分析器的配置（None → 分析器默认）
        self.analyzer_config = analyzer_config
        # 公共窗口振幅重拟合的额外最小点数（在数学下限 4 之上）。
        # None = 待校准（只使用数学下限 n > k+1，即 ≥4 点）
        self.refit_min_points = None
        # 符号通道窗口内高于噪声底的最小点数（在数学下限 2 之上）。
        # None = 待校准（只使用数学下限 ≥2 点）
        self.sign_min_points = None


# ==========================================================================
# 阶段 1：数据装配与完整性检查
# ==========================================================================
def classify_refbond(refbond, geometry):
    """参考键类别：由键向量未定向线方向角导出（60°→A、120°→B、0°→C）。
    与 DMRGtJModel XC ClassifyNearestNeighbor 的分类一致（已实测核对）。"""
    i, j = int(refbond[0]), int(refbond[1])
    if not (0 <= i < geometry.N and 0 <= j < geometry.N):
        raise ValueError(f"参考键 {refbond} 超出 [0,{geometry.N})")
    v = np.array(geometry.coords[j]) - np.array(geometry.coords[i])
    ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0   # 未定向线角 ∈ [0,180)
    for cls, ref in (('A', 60.0), ('B', 120.0), ('C', 0.0)):
        if abs(ang - ref) < 1.0 or abs(ang - ref + 180.0) < 1.0:
            return cls
    raise ValueError(f"参考键 {refbond} 线方向角 {ang:.1f}° 不属于"
                     f"A(60°)/B(120°)/C(0°) 三类")


def discover_refbond_dirs(super_corr_root):
    """发现 super_corr_{i}_{j} 目录（按现有命名约定）。目录缺失抛
    FileNotFoundError（与既有失败语义一致）。"""
    if not os.path.isdir(super_corr_root):
        raise FileNotFoundError(f'super_corr 目录缺失或不可读: {super_corr_root}')
    dirs = []
    for name in sorted(os.listdir(super_corr_root)):
        parts = name.split('_')
        if len(parts) == 4 and parts[0] == 'super' and parts[1] == 'corr':
            try:
                dirs.append((name, (int(parts[2]), int(parts[3]))))
            except ValueError:
                continue
    return dirs


def raw_parse_channel_file(path):
    """结构级解析：返回 (entries, issues)。不做 allowed_cond 过滤。

    entries = [[i, j, k, l], [value, 0]] 原样列表；issues 逐项记录：
    - non_finite：非有限关联值（含条目索引与值）
    - duplicate_key：重复 (i,j,k,l) 键（计数，不覆盖）
    - refbond_mismatch：参考键与首条不一致
    - empty_records：文件为空
    - format_error：格式不符合 [[i,j,k,l],[v,0]]
    - missing_file：文件不存在
    不静默替换、不自动去重——问题逐项暴露给上层。
    """
    issues = []
    if not os.path.isfile(path):
        return None, [{'category': ISSUE_MISSING_FILE,
                       'location': path,
                       'detail': '文件不存在'}]
    with open(path, encoding='utf-8') as f:
        try:
            data = ast.literal_eval(f.read())
        except (ValueError, SyntaxError) as e:
            return None, [{'category': ISSUE_FORMAT, 'location': path,
                           'detail': f'无法解析: {e}'}]
    if not data:
        return None, [{'category': ISSUE_EMPTY_RECORDS, 'location': path,
                       'detail': '文件内容为空'}]
    entries = []
    first_bond = None
    seen = {}
    for m, element in enumerate(data):
        try:
            [i, j, k, l], [value, *_] = element
        except (TypeError, ValueError):
            issues.append({'category': ISSUE_FORMAT, 'location': path,
                           'detail': f'第 {m} 个元素格式不符合 '
                                     f'[[i,j,k,l],[v,0]]: {element}'})
            continue
        bond = (int(i), int(j))
        if first_bond is None:
            first_bond = bond
        elif bond != first_bond:
            issues.append({'category': ISSUE_REFBOND_MISMATCH,
                           'location': path,
                           'detail': f'第 {m} 个元素参考键 {bond}，'
                                     f'预期 {first_bond}'})
        key = (int(i), int(j), int(k), int(l))
        if key in seen:
            issues.append({'category': ISSUE_DUPLICATE_KEY, 'location': path,
                           'detail': f'重复键 {key}（第 {m} 个元素），'
                                     f'重复计数 {seen[key] + 1}'})
        seen[key] = seen.get(key, 0) + 1
        try:
            v = float(value)
        except (TypeError, ValueError):
            issues.append({'category': ISSUE_FORMAT, 'location': path,
                           'detail': f'第 {m} 个元素值无法转 float: {value}'})
            v = float('nan')
        if not np.isfinite(v):
            issues.append({'category': ISSUE_NON_FINITE, 'location': path,
                           'detail': f'第 {m} 个元素值非有限: {value}'})
        entries.append([int(i), int(j), int(k), int(l), v])
    return entries, issues


def load_element(element, N, a=1.0):
    """加载单个 (参考键, 方向) 元素：四通道结构检查 + extract_data 合成。

    返回 (channels, combined_records, integrity)：
    - channels: {ch: list[SuperCorrRecord]}（结构检查通过且可解析的通道）
    - combined_records: list[SuperCorrRecord]（四通道齐全时合成；否则 None）
    - integrity: {'status': ok|missing|incomplete, 'issues': [...]}
    extract_data 的数据级异常（ValueError/FileNotFoundError）按既有语义
    捕获并记为 issue；其余异常原样传播。
    """
    issues = []
    channels = {}
    for ch in CHANNELS:
        path = os.path.join(
            element['refbond_dir'],
            f"scs{element['direction']}{ch}_fix_refbond_"
            f"{element['refbond'][0]}_{element['refbond'][1]}.json")
        entries, ch_issues = raw_parse_channel_file(path)
        issues.extend(ch_issues)
        if entries is None:
            continue
        if any(iss['category'] in (ISSUE_FORMAT, ISSUE_EMPTY_RECORDS)
               for iss in ch_issues):
            continue   # 结构级失败，不进入 extract_data
        try:
            text = json_fmt(entries)
        except ValueError as e:
            issues.append({'category': ISSUE_FORMAT, 'location': path,
                           'detail': f'非有限值无法序列化为标准 JSON: {e}'})
            continue
        try:
            recs = SuperCorrRecord.extract_data(text, N, a)
        except (FileNotFoundError, ValueError) as e:
            issues.append({'category': ISSUE_FORMAT, 'location': path,
                           'detail': f'extract_data 数据级异常: {e}'})
            continue
        channels[ch] = recs
    combined = None
    if set(channels) == set(CHANNELS):
        combined = SuperCorrProcessor.Combined_super_correlation(
            channels['a'], channels['b'], channels['c'], channels['d'])
    if not channels:
        status = INTEGRITY_MISSING if any(
            iss['category'] == ISSUE_MISSING_FILE for iss in issues) \
            else INTEGRITY_INCOMPLETE
    elif set(channels) != set(CHANNELS):
        status = INTEGRITY_MISSING if any(
            iss['category'] == ISSUE_MISSING_FILE for iss in issues) \
            else INTEGRITY_INCOMPLETE
    else:
        status = INTEGRITY_OK
    return channels, combined, {'status': status, 'issues': issues}


def json_fmt(entries):
    """entries([[i,j,k,l,v],...]) → extract_data 所需 JSON 文本（[[i,j,k,l],[v,0]]）。

    allow_nan=False：非有限值无法以标准 JSON 表达，直接抛 ValueError
    （由调用方记为 format_error 类问题，不静默输出 NaN/Infinity 字面量）。"""
    import json
    return json.dumps([[[e[0], e[1], e[2], e[3]], [e[4], 0]] for e in entries],
                      allow_nan=False)


# ==========================================================================
# 阶段 2：可比性质量门
# ==========================================================================
def analyze_element(combined_records, analyzer_config=None):
    """对元素运行真实 SC 衰减分析（不重新实现拟合）。

    返回 {'status': ok|analysis_failed, 'result': DecayAnalysisResult|None,
          'failure': {...}|None, 'meta': {...}}。
    meta 含包络形式（K/xi，取自 envelope_fit.params 真实键）、指数与误差、
    窗口、有效点数、振荡检测与证据、分类。
    捕获 FileNotFoundError/ValueError（与 _try 语义一致），其余传播。
    """
    try:
        res = SuperCorrDecayAnalyzer.analyze(combined_records, analyzer_config)
    except (FileNotFoundError, ValueError) as e:
        return {'status': ANALYSIS_FAILED, 'result': None,
                'failure': {'type': type(e).__name__, 'message': str(e)},
                'meta': {}}
    env = res.envelope_fit
    form = None
    exponent = exponent_error = None
    if env is not None and getattr(env, 'params', None):
        if 'K' in env.params:
            form = 'K'
            exponent = float(env.params['K'])
            exponent_error = float(env.param_errors.get('K', np.nan))
        elif 'xi' in env.params:
            form = 'xi'
            exponent = float(env.params['xi'])
            exponent_error = float(env.param_errors.get('xi', np.nan))
    meta = {
        'classification': res.classification,
        'envelope_form': form,
        'exponent': exponent,
        'exponent_error': exponent_error,
        'r_min': float(res.r_min),
        'r_max': float(res.r_max),
        'n_points': int(res.n_points),
        'osc_detected': bool(getattr(res.oscillation, 'detected', False)),
        'osc_evidence': getattr(res.oscillation, 'evidence', None),
        'verdict': getattr(getattr(res, 'stability', None), 'verdict', None),
        'final_noise_floor': float(res.final_noise_floor),
    }
    return {'status': ANALYSIS_OK, 'result': res, 'failure': None,
            'meta': meta}


def element_distance_points(combined_records):
    """元素的距离分组点（供公共窗口内有效点数计数）。
    复用 SuperCorrDecayAnalyzer.group_by_distance（真实分组逻辑）。"""
    prep = SuperCorrDecayAnalyzer.group_by_distance(combined_records)
    return np.asarray(prep.r, dtype=float)


def gate_form_consistency(elem_meta):
    """K/xi 分流：返回 (common_form, incompatible)。

    elem_meta: {element_key: meta}。包络形式缺失/冲突 → 不可比较并保留
    原因。K 与 xi 是不同物理量，绝不混列（设计文档 §5.2 硬前置）。
    """
    forms = {k: v.get('envelope_form') for k, v in elem_meta.items()}
    if not forms:
        return None, {}
    common = next(iter(forms.values()))
    if common is None:
        return None, {k: '包络形式缺失（无 envelope_fit 参数）'
                      for k, v in forms.items() if v is None}
    bad = {}
    for k, v in forms.items():
        if v is None:
            bad[k] = '包络形式缺失（无 envelope_fit 参数）'
        elif v != common:
            bad[k] = f'包络形式 {v} 与共同形式 {common} 不一致（K/xi 不混用）'
    return common, bad


# ---- 振荡证据分层（5B-1：设计文档 §5.2"过零组合与无过零组合分别
# 处理"的落实；替代 5B-0 审计的硬排除混合规则） ----
LAYER_OSC = 'osc'
LAYER_NON_OSC = 'non_osc'
LAYER_UNKNOWN = 'unknown'
LAYER_KEYS = (LAYER_OSC, LAYER_NON_OSC, LAYER_UNKNOWN)

# 振荡层振幅偏差警告（5B-0 合成实验观察；特定实验条件下的结果，
# 不是普适误差界，不得作为校正系数或置信区间）
OSC_AMPLITUDE_BIAS_WARNING = (
    '振荡元素的 |C| 包络振幅拟合存在与节点位置、q/phi、拟合窗口及'
    '噪声底相关的系统偏差（5B-0 合成实验观察：振幅偏差约 −13%~−69%、'
    '跨元素相对极差可达 ~114%；该范围为特定实验条件下的结果，不是'
    '普适误差界，不得作为真实数据的校正系数或置信区间）；层内拟合'
    '成功不等于振幅可直接用于可靠候选匹配；当前未完成 q/phi 逐元素'
    '校正，振荡偏差未被消除')


def compute_window_zero_crossings(records_by_element, window,
                                  floors_by_element):
    """公共窗口内带符号序列的过零证据。只消费带符号原始记录，绝不由
    |C|、拟合振幅 A 或非负向量 m 推断（设计文档 §4.5 硬规则）。

    翻转点对 (C[i], C[i+1])（i 与 i+1 均在窗口内）：
    - 两侧 |C| 均 > noise_floor → 可靠翻转（数据点高于噪声底）
    - 任一侧 ≤ noise_floor → 不可靠翻转（噪声底附近，不足以判定真实
      翻转；可靠性倍率阈值未校准，噪声底本身作为标尺）
    - noise_floor 缺失 → reliability='unavailable'（数据不足，不判可靠）
    返回 {element_key: report}。
    """
    lo, hi = float(window[0]), float(window[1])
    out = {}
    for key, recs in records_by_element.items():
        floor = floors_by_element.get(key)
        prep = SuperCorrDecayAnalyzer.group_by_distance(recs)
        r = np.asarray(prep.r, dtype=float)
        C = np.asarray(prep.C, dtype=float)
        wm = (r >= lo) & (r <= hi)
        C_w = C[wm]
        n = int(np.sum(wm))
        n_cross = n_rel = n_unrel = 0
        if n >= 2:
            signs = np.sign(C_w)
            idx = np.nonzero(signs[1:] * signs[:-1] < 0)[0]
            n_cross = int(len(idx))
            if n_cross and floor is not None:
                for i in idx:
                    if min(abs(C_w[i]), abs(C_w[i + 1])) > floor:
                        n_rel += 1
                    else:
                        n_unrel += 1
        if n_cross == 0:
            rel = 'none'
            note = '窗口内无符号翻转'
        elif floor is None:
            rel = 'unavailable'
            note = ('final_noise_floor 缺失，过零可靠性无法判定'
                    '（数据不足，不判可靠）')
        elif n_rel >= 1:
            rel = 'reliable'
            note = (f'{n_rel} 次可靠翻转（翻转点对两侧 |C| 均高于噪声底）'
                    + (f'；另有 {n_unrel} 次不可靠翻转' if n_unrel else ''))
        else:
            rel = 'unreliable_only'
            note = (f'{n_unrel} 次翻转全部发生在噪声底附近（两侧 |C| 未'
                    '全部高于噪声底），不足以判定为真实振荡')
        out[key] = {
            'window': [lo, hi],
            'n_points_in_window': n,
            'zero_crossing_observed': n_cross,
            'reliable_crossings': n_rel,
            'unreliable_crossings': n_unrel,
            'reliability': rel,
            'reliability_note': note,
            'noise_floor_used': (float(floor) if floor is not None else None),
        }
    return out


def layer_oscillation_classification(elem_meta, crossings):
    """振荡分层判定（5B-0 实证的层判据）。返回
    (layers, layer_reports)。

    规则（按序）：
    1. osc_evidence ∈ {strong, moderate} → osc（保留证据等级）
    2. 证据缺失（None）→ unknown
    3. 公共窗口内存在可靠过零 → osc（即使证据级别为 weak/possible/none）
    4. 过零全部不可靠（噪声底附近）→ unknown（不足以判定真实振荡）
    5. 过零可靠性不可判定（噪声底缺失）→ unknown（数据不足）
    6. 无过零且证据可用 → non_osc（按非振荡处理，不表示已证明不存在
       振荡）
    不能明确分类 → unknown，不强行归类。
    """
    layers = {}
    reports = {}
    for k, meta in elem_meta.items():
        ev = meta.get('osc_evidence')
        cr = crossings.get(k)
        if ev in ('strong', 'moderate'):
            layer = LAYER_OSC
            reason = (f'振荡证据级别 {ev}（证据等级保留；公共窗口过零'
                      '证据同步记录）')
        elif ev is None:
            layer = LAYER_UNKNOWN
            reason = '振荡证据缺失（分类缺失），不假定兼容'
        elif cr is None:
            layer = LAYER_UNKNOWN
            reason = '无公共窗口过零证据（数据不足），不强行归类'
        elif cr['reliability'] == 'reliable':
            layer = LAYER_OSC
            reason = (f'公共窗口内存在可靠过零（{cr["reliable_crossings"]}'
                      f' 次；证据级别 {ev} 不足但过零证据充分）')
        elif cr['reliability'] == 'none':
            layer = LAYER_NON_OSC
            reason = (f'公共窗口内无过零（证据级别 {ev}，按非振荡处理；'
                      '不表示已证明不存在振荡）')
        else:   # unreliable_only / unavailable
            layer = LAYER_UNKNOWN
            reason = (cr.get('reliability_note') or '过零可靠性不可判定') \
                + '；保留 unknown，不强行归类'
        layers[k] = layer
        reports[k] = {
            'layer': layer,
            'osc_evidence': ev,
            'reason': reason,
            'zero_crossing_observed': (cr or {}).get('zero_crossing_observed'),
            'reliable_crossings': (cr or {}).get('reliable_crossings'),
            'unreliable_crossings': (cr or {}).get('unreliable_crossings'),
            'crossing_reliability': (cr or {}).get('reliability'),
        }
    return layers, reports


def _intersect_private_windows(meta, keys):
    """元素私有窗口 (r_min, r_max) 的交。keys 为空或任一窗口非有限 →
    None；不相交 → None。"""
    if not keys:
        return None
    lo = max(meta[k]['r_min'] for k in keys)
    hi = min(meta[k]['r_max'] for k in keys)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        return None
    return (float(lo), float(hi))


def gate_common_window(elem_meta, point_counts, min_common_points=None):
    """公共窗口：取各元素 (r_min, r_max) 的交；再按窗口内实际保留点数
    检查（不是只比较端点）。

    min_common_points=None → 待校准：不按点数排除，但记录每个元素的
    窗口内点数与 'uncalibrated' 标记。默认阈值来源建议：衰减分析器
    n_min_fit（窗口内需足够点数才能重拟合包络），实现为显式参数。

    返回 (common_window, excluded, point_counts_in_window, uncalibrated)。
    """
    lo = max(v['r_min'] for v in elem_meta.values())
    hi = min(v['r_max'] for v in elem_meta.values())
    if lo > hi:
        return None, {k: '公共窗口不存在（r_min/r_max 区间不相交）'
                      for k in elem_meta}, {}, False
    excluded = {}
    counts = {}
    for k in elem_meta:
        counts[k] = int(np.sum((point_counts[k] >= lo)
                               & (point_counts[k] <= hi)))
        if min_common_points is not None and counts[k] < min_common_points:
            excluded[k] = (f'公共窗口 [{lo:.3f}, {hi:.3f}] 内点数 '
                           f'{counts[k]} < {min_common_points}')
    return (lo, hi), excluded, counts, (min_common_points is None)


def gate_su2(channels_by_element, su2_error_max=None):
    """SU(2) 检查：显式调用 SuperCorrProcessor.symmetry_check（不重实现
    公式）。返回 {element_key: {'err_ad', 'err_bc', 'max_err', 'gate'}}。

    gate ∈ {'pass', 'fail', 'uncalibrated', 'skipped'}：
    - su2_error_max 给定 → 超过上限 fail，否则 pass
    - None（待校准）→ uncalibrated：只记录误差，不判通过/拒绝
    - 通道不全 → skipped（无法检查，原因注明）
    检查通过不解释为超导对称性成立（设计文档 §5.4）。
    """
    out = {}
    for k, channels in channels_by_element.items():
        if set(channels) != set(CHANNELS):
            out[k] = {'err_ad': None, 'err_bc': None, 'max_err': None,
                      'gate': 'skipped',
                      'detail': f'通道不全（{sorted(channels)}），无法检查'}
            continue
        try:
            with np.errstate(all='ignore'):
                err_ad, err_bc = SuperCorrProcessor.symmetry_check(
                    channels['a'], channels['b'], channels['c'],
                    channels['d'])
        except (ValueError, ZeroDivisionError) as e:
            out[k] = {'err_ad': None, 'err_bc': None, 'max_err': None,
                      'gate': 'skipped', 'detail': f'检查异常: {e}'}
            continue
        if not (np.isfinite(err_ad) and np.isfinite(err_bc)):
            # 全零等退化数据：symmetry_check 除零 → 非有限误差
            out[k] = {'err_ad': None, 'err_bc': None, 'max_err': None,
                      'gate': 'uncomputable',
                      'detail': '关联值全零等退化数据，SU(2) 相对误差'
                                '无定义'}
            continue
        max_err = float(max(err_ad, err_bc))
        if su2_error_max is None:
            gate = 'uncalibrated'
        else:
            gate = 'fail' if max_err > su2_error_max else 'pass'
        out[k] = {'err_ad': float(err_ad), 'err_bc': float(err_bc),
                  'max_err': max_err, 'gate': gate, 'detail': ''}
    return out


# ==========================================================================
# 可比较元素集合构建（阶段 1+2 串联）
# ==========================================================================
def build_comparable_set(condition, config=None):
    """对一个条件执行阶段 1（装配+完整性）+ 阶段 2（质量门 + 振荡
    证据分层），输出可比较元素集合报告（不拟合矩阵、不产生对称性
    判定）。

    分层流程（5B-1）：形式分流（K/xi 硬前置）→ 形式组公共窗口 →
    组窗口内带符号过零证据（只消费带符号原始记录）→ osc/non_osc/
    unknown 分层（公共窗口为准，不用元素私有窗口过零）→ 层内公共
    窗口重算 + 过零复核（至稳定，上限 3 轮）→ 层内点数/SU(2) 门。
    各层独立构建集合与窗口，跨层比较默认拒绝。

    condition: {'doping', 'N', 'bond_dim', 'super_corr_root',
                'refbonds': 可选 {tag}}
    返回 dict：comparable_set 含 layers（osc/non_osc/unknown 三层）、
    cross_layer_comparison（拒绝声明）、聚合视图（status/common_form/
    member_keys/excluded 等）。
    """
    config = config or PairingSymmetryConfig()
    N = condition['N']
    geometry = HoneycombXCGeometry(N, 1.0)
    refbonds = condition.get('refbonds')

    elements = {}
    for dir_name, refbond in discover_refbond_dirs(
            condition['super_corr_root']):
        tag = f'{refbond[0]}_{refbond[1]}'
        if refbonds is not None and tag not in refbonds:
            continue
        try:
            rclass = classify_refbond(refbond, geometry)
        except ValueError as e:
            rclass = None
        for direction in VALID_DIRECTIONS:
            key = f'{tag}|{direction}'
            elements[key] = {
                'element_key': key,
                'refbond': list(refbond),
                'refbond_class': rclass,
                'direction': direction,
                'refbond_dir': os.path.join(condition['super_corr_root'],
                                            dir_name),
            }

    channels_by_element = {}
    records_by_element = {}
    for key, el in elements.items():
        channels, combined, integrity = load_element(el, N, 1.0)
        el['integrity'] = integrity
        el['channels'] = {c: True for c in channels}
        if combined is not None:
            channels_by_element[key] = channels
            records_by_element[key] = combined

    # ---- 阶段 2：对完整性 ok 的元素做真实分析 ----
    meta = {}
    for key, recs in records_by_element.items():
        ares = analyze_element(recs, config.analyzer_config)
        elements[key]['analysis'] = {
            'status': ares['status'],
            'failure': ares['failure'],
            'meta': ares['meta'],
        }
        if ares['status'] == ANALYSIS_OK:
            meta[key] = ares['meta']

    # ---- 质量门（顺序：形式 → 振荡分层 → 层内公共窗口 → SU(2)） ----
    common_form, form_bad = gate_form_consistency(meta)
    point_counts = {k: element_distance_points(recs)
                    for k, recs in records_by_element.items()}
    su2 = gate_su2(channels_by_element, config.su2_error_max)
    floors = {k: v.get('final_noise_floor') for k, v in meta.items()}

    # 形式组成员（分析成功、窗口有限、形式一致——分层的基础集合）
    form_members = [k for k in meta
                    if k not in form_bad
                    and np.isfinite(meta[k].get('r_min', np.nan))
                    and np.isfinite(meta[k].get('r_max', np.nan))]
    form_meta = {k: meta[k] for k in form_members}
    form_recs = {k: records_by_element[k] for k in form_members}

    layers_of = {}
    layer_reports = {}
    layer_window_note = ''
    win_form_was_none = False
    rounds_used = 0
    per_layer_changes = {LAYER_OSC: [], LAYER_NON_OSC: []}
    unstable = False
    if common_form is None or not form_members:
        win_form_was_none = True
        layer_window_note = '形式组为空（K/xi 分流后无可用元素），无法分层'
    else:
        # 第 1 轮：形式组公共窗口内过零证据 → 分层（公共窗口为准，
        # 不用元素私有窗口的过零计数）
        win_form = _intersect_private_windows(meta, form_members)
        if win_form is None:
            win_form_was_none = True
            layer_window_note = ('形式组公共窗口不存在（私有窗口不相交），'
                                 '无分层窗口可依')
        else:
            crossings = compute_window_zero_crossings(form_recs, win_form,
                                                      floors)
            layers_of, layer_reports = layer_oscillation_classification(
                form_meta, crossings)
            # 复核轮（上限 2 轮）：层窗口相对初始形式组窗口可能变宽
            # 也可能变窄（层成员是形式组子集，lo 取层内最大值可更大、
            # hi 取层内最小值可更小）→ 层内窗口的过零数可增可减，元素
            # 分类可能双向变化（non_osc→osc/unknown，或 osc→non_osc）。
            # 每轮对每个非空层用当前层窗口重算过零并重新分类，直到
            # 一轮内无任何归属变化（固定点）或轮数耗尽；不得把未稳定
            # 结果标注为已稳定（unstable + 逐层变化记录）。
            for _round in range(2):
                rounds_used = _round + 1
                changed = False
                round_changes = {LAYER_OSC: [], LAYER_NON_OSC: []}
                for L in (LAYER_OSC, LAYER_NON_OSC):
                    ms = [k for k in form_members if layers_of.get(k) == L]
                    if not ms:
                        continue
                    wL = _intersect_private_windows(meta, ms)
                    crossL = compute_window_zero_crossings(
                        {k: records_by_element[k] for k in ms}, wL, floors)
                    _, repL = layer_oscillation_classification(
                        {k: meta[k] for k in ms}, crossL)
                    for k, rep in repL.items():
                        layer_reports[k] = rep
                        if layers_of.get(k) != rep['layer']:
                            round_changes[L].append({
                                'element': k,
                                'from': layers_of.get(k),
                                'to': rep['layer']})
                            layers_of[k] = rep['layer']
                            changed = True
                per_layer_changes = round_changes
                if not changed:
                    break
            else:
                unstable = True
                layer_window_note = ('分层复核达到最大轮数仍未稳定，'
                                     '以最后一轮归属为准（保守标注）')

    # 层内公共窗口（层成员 ⊆ 形式组 → 组窗口存在则层窗口必存在）
    layer_windows = {}
    for L in LAYER_KEYS:
        ms = [k for k in form_members if layers_of.get(k) == L]
        layer_windows[L] = (_intersect_private_windows(meta, ms)
                            if ms else None)

    # 层内剩余质量门（点数 / SU(2)）与排除汇总
    layer_structs = {}
    global_excluded = {}
    for k, el in elements.items():
        reasons = []
        if el['integrity']['status'] != INTEGRITY_OK:
            reasons.append(f"完整性 {el['integrity']['status']}: "
                           + '; '.join(sorted({i['category'] for i in
                                               el['integrity']['issues']})))
        else:
            ar = el['analysis']
            if ar['status'] == ANALYSIS_FAILED:
                reasons.append(f"分析执行失败（{ar['failure']['type']}）")
            else:
                if k in form_bad:
                    reasons.append(form_bad[k])
                if k in meta and k not in form_members and not reasons:
                    reasons.append('元素私有窗口非有限，无法参与振荡分层')
                if win_form_was_none:
                    reasons.append(layer_window_note)
        if reasons:
            global_excluded[k] = reasons
        el['quality_gate'] = {
            'passed': not reasons,
            'reasons': reasons,
            'layer': layers_of.get(k),
        }

    for L in LAYER_KEYS:
        ms = [k for k in form_members if layers_of.get(k) == L]
        if L == LAYER_UNKNOWN:
            # unknown 层不构建矩阵（保守，不强行归类）
            layer_structs[L] = {
                'status': 'none',
                'common_form': common_form,
                'common_window': None,
                'common_window_points': {},
                'member_keys': [],
                'elements': sorted(ms),
                'excluded': {k: [layer_reports.get(k, {}).get(
                    'reason', 'unknown 分类（不强行归类）')] for k in ms},
                'zero_crossing_report': {k: (layer_reports.get(k, {}) | {
                    'layer_window': None}) for k in ms},
            }
            continue
        wL = layer_windows[L]
        excluded_L = {}
        counts_L = {}
        uncal_L = False
        if wL is None:
            for k in ms:
                excluded_L[k] = ['层内公共窗口不存在（私有窗口不相交）']
        else:
            _, ex_p, counts_L, uncal_L = gate_common_window(
                {k: meta[k] for k in ms}, point_counts,
                config.min_common_points)
            excluded_L.update({k: [r] for k, r in ex_p.items()})
        for k in ms:
            sg = su2.get(k, {})
            if sg.get('gate') == 'fail':
                excluded_L.setdefault(k, []).append(
                    f"SU(2) 超标（max_err={sg['max_err']:.3g}"
                    f" > {config.su2_error_max}）")
        member_L = [k for k in ms if not excluded_L.get(k)]
        layer_structs[L] = {
            'status': ('full' if ms and not excluded_L
                       else ('partial' if member_L else 'none')),
            'is_partial': bool(member_L and excluded_L),
            'common_form': common_form,
            'common_window': (list(wL) if wL is not None else None),
            'common_window_points': counts_L if wL is not None else {},
            'member_keys': member_L,
            'excluded': excluded_L,
            'osc_evidence_levels': {k: layer_reports.get(k, {}).get(
                'osc_evidence') for k in ms},
            'zero_crossing_report': {k: (layer_reports.get(k, {}) | {
                'layer_window': (list(wL) if wL is not None else None)})
                for k in ms},
            'window_points_uncalibrated': uncal_L,
            'layer_classification_rounds_note': layer_window_note,
            'layer_convergence': {
                'stable': not unstable,
                'rounds_used': rounds_used,
                'last_round_changes': per_layer_changes.get(L, []),
                'note': layer_window_note,
            },
        }
        if L == LAYER_OSC:
            layer_structs[L]['amplitude_bias_warning'] = \
                OSC_AMPLITUDE_BIAS_WARNING

    member_keys_all = sorted(
        k for L in (LAYER_OSC, LAYER_NON_OSC)
        for k in layer_structs[L]['member_keys'])
    excluded_all = dict(global_excluded)
    for L in LAYER_KEYS:
        for k, rs in layer_structs[L]['excluded'].items():
            excluded_all.setdefault(k, []).extend(rs)
    # 层内排除回写 quality_gate（全局门与层内门统一可见）
    for k, el in elements.items():
        extra = [r for L in LAYER_KEYS
                 for r in layer_structs[L]['excluded'].get(k, [])]
        if extra:
            el['quality_gate']['reasons'].extend(extra)
            el['quality_gate']['passed'] = False
    # records_by_element 仅内存引用（供阶段 3 层内公共窗口重拟合）；
    # 含 SuperCorrRecord 对象，不可直接 JSON 序列化——不写入任何落盘产物
    report = {
        'condition': {'doping': condition['doping'],
                      'N': condition['N'],
                      'bond_dim': condition['bond_dim']},
        'elements': elements,
        'su2': su2,
        'comparable_set': {
            'layers': layer_structs,
            'cross_layer_comparison': {
                'allowed': False,
                'reason': ('振荡证据层间振幅可比性未建立（5B-0：|C| '
                           '包络振幅偏差元素相关，跨层混排污染矩阵结构）；'
                           '跨层比较默认拒绝'),
            },
            'status': ('full' if member_keys_all and not excluded_all
                       else ('partial' if member_keys_all else 'none')),
            'is_partial': bool(member_keys_all and excluded_all),
            'common_form': common_form,
            'common_window': {L: layer_structs[L]['common_window']
                              for L in LAYER_KEYS},
            'common_window_points': {L: layer_structs[L][
                'common_window_points'] for L in LAYER_KEYS},
            'member_keys': member_keys_all,
            'excluded': excluded_all,
            'uncalibrated_thresholds': [
                'min_common_points' if config.min_common_points is None
                else None,
                'su2_error_max' if config.su2_error_max is None else None,
                'zero_crossing_reliability_threshold',
            ],
            'uncalibrated_notes': [
                ('min_common_points=None：公共窗口内最小点数未校准；'
                 '仅记录各元素窗口内点数，不按点数排除')
                if config.min_common_points is None else None,
                ('su2_error_max=None：SU(2) 误差上限未校准；仅记录 '
                 'a↔d/b↔c 误差，不判通过/拒绝')
                if config.su2_error_max is None else None,
                ('zero_crossing_reliability_threshold：过零可靠性以'
                 ' final_noise_floor 本身为标尺（翻转点对两侧 |C| 均'
                 '高于噪声底才算可靠），任何倍率因子均未校准'),
            ],
        },
        'no_symmetry_conclusions': (
            '本报告只完成数据装配、可比性质量门与振荡证据分层；不含'
            '矩阵拟合、配对对称性或任何物理相态判定'),
        'records_by_element': records_by_element,
        'oscillation_semantics': (
            '振荡证据按层分类（5B-1）：strong/moderate → osc；公共窗口'
            '内存在可靠过零 → osc（即使证据级别 weak/possible/none）；'
            '无过零且证据可用 → non_osc（按非振荡处理，不表示已证明'
            '不存在振荡）；证据缺失/过零可靠性不可判定 → unknown'
            '（不强行归类）。各层独立构建公共窗口与矩阵，跨层比较'
            '默认拒绝'),
    }
    return report


# ==========================================================================
# 阶段 3：公共窗口振幅重拟合 + 两级矩阵模型 + 可辨识性诊断
# ==========================================================================
# 公共窗口重拟合的数学下限：2 参数包络（A,K 或 A,ξ）需 n > k+1 → n ≥ 4
_REFIT_HARD_MIN_POINTS = 4


def _jac_rank(jac):
    """尺度自适应的数值秩：容差 = 1e-6 × 最大奇异值（相对判据）。

    绝对容差会随数据幅度缩放而失效（1e-12 量级矩阵被误判为秩 0，
    D5 3.1 审计实测）；相对判据保证同一结构仅整体缩放时秩不变。
    返回 (秩, 奇异值列表)。"""
    jac = np.asarray(jac, dtype=float)
    s = np.linalg.svd(jac, compute_uv=False)
    if len(s) == 0:
        return 0, []
    tol = 1e-6 * s[0] if s[0] > 0 else 1e-12
    return int(np.sum(s > tol)), [float(x) for x in s]


def _refit_unavailable(reason):
    return {'elements': {}, 'excluded': {}, 'matrix': None, 'mask': None,
            'row_labels': [], 'col_labels': [], 'status': 'unavailable',
            'reason': reason}


def _refit_single(report, config, window, form, member_keys,
                  excluded_extra=None):
    """单层（或单集合）公共窗口振幅重拟合核心（|C| 空间，复用
    fit_envelope_logspace；不复用元素原推荐窗口的拟合参数）。

    元素结果字段：window/n_points/A/exponent/exponent_error/r2/aicc/bic/
    fit_status(ok|insufficient|failed)/reason。拟合失败不静默丢弃。
    振幅 A 为 |C| 包络（非负），不携带符号信息（符号通道另属后续阶段）。
    """
    config = config or PairingSymmetryConfig()
    excluded_extra = excluded_extra or {}
    records = report.get('records_by_element', {})
    lo, hi = window
    min_pts = (config.refit_min_points
               if config.refit_min_points is not None
               else _REFIT_HARD_MIN_POINTS)
    # common_form（'K'|'xi'）→ fit_envelope_logspace 的 form（'power'|'exp'）
    fit_form = {'K': 'power', 'xi': 'exp'}.get(form)
    if fit_form is None:
        return _refit_unavailable(f'共同形式 {form} 无法映射到包络拟合形式')
    elements = {}
    excluded = {}
    # 行/列标签（按参考键类别 × 方向，排序确定）
    row_labels = sorted({report['elements'][k].get('refbond_class') or '?'
                         for k in member_keys})
    col_labels = list(VALID_DIRECTIONS)
    row_idx = {r: i for i, r in enumerate(row_labels)}
    col_idx = {c: i for i, c in enumerate(col_labels)}
    M = np.full((len(row_labels), len(col_labels)), np.nan)
    mask = np.zeros((len(row_labels), len(col_labels)), bool)

    for key in member_keys:
        recs = records.get(key)
        el = report['elements'].get(key, {})
        rclass = el.get('refbond_class') or '?'
        direction = el.get('direction', '')
        if recs is None or rclass not in row_idx or direction not in col_idx:
            excluded[key] = '缺记录或身份无法定位'
            continue
        if key in excluded_extra:
            excluded[key] = '; '.join(excluded_extra[key])
            continue
        # 距离分组（真实逻辑）+ 公共窗口裁剪 + 有效正 |C| 点计数
        prep = SuperCorrDecayAnalyzer.group_by_distance(recs)
        r_all = np.asarray(prep.r, dtype=float)
        C_all = np.asarray(prep.C, dtype=float)
        wmask = (r_all >= lo) & (r_all <= hi)
        r_w = r_all[wmask]
        C_w = C_all[wmask]
        n_pts = int(np.sum(wmask))
        n_pos = int(np.sum(C_w > 0.0))
        if n_pts < min_pts or n_pos < _REFIT_HARD_MIN_POINTS:
            excluded[key] = (f'公共窗口内点数不足（总 {n_pts}、正 |C| '
                             f'{n_pos}，下限 {min_pts}/{_REFIT_HARD_MIN_POINTS}）')
            elements[key] = {'element_key': key, 'window': [lo, hi],
                             'n_points': n_pts, 'A': None, 'exponent': None,
                             'exponent_error': None, 'r2': None,
                             'aicc': None, 'bic': None,
                             'fit_status': 'insufficient',
                             'reason': excluded[key]}
            continue
        try:
            fit = CorrelationDecayAnalyzer.fit_envelope_logspace(
                r_w, np.abs(C_w), fit_form, config.analyzer_config)
        except (FileNotFoundError, ValueError) as e:
            excluded[key] = f'重拟合数据级异常: {e}'
            elements[key] = {'element_key': key, 'window': [lo, hi],
                             'n_points': n_pts, 'A': None, 'exponent': None,
                             'exponent_error': None, 'r2': None,
                             'aicc': None, 'bic': None,
                             'fit_status': 'failed', 'reason': excluded[key]}
            continue
        if not fit.valid or not fit.params:
            excluded[key] = f'重拟合无效: {fit.message}'
            elements[key] = {'element_key': key, 'window': [lo, hi],
                             'n_points': n_pts, 'A': None, 'exponent': None,
                             'exponent_error': None, 'r2': None,
                             'aicc': None, 'bic': None,
                             'fit_status': 'failed', 'reason': excluded[key]}
            continue
        exp_key = 'K' if 'K' in fit.params else ('xi' if 'xi' in fit.params
                                                 else None)
        res = {'element_key': key, 'window': [lo, hi], 'n_points': n_pts,
               'A': float(fit.params.get('A', np.nan)),
               'exponent': (float(fit.params[exp_key]) if exp_key else None),
               'exponent_error': (float(fit.param_errors[exp_key])
                                  if exp_key and exp_key in fit.param_errors
                                  else None),
               'r2': float(fit.r2), 'aicc': float(fit.aicc),
               'bic': float(fit.bic), 'fit_status': 'ok', 'reason': ''}
        elements[key] = res
        M[row_idx[rclass], col_idx[direction]] = res['A']
        mask[row_idx[rclass], col_idx[direction]] = True
    return {'elements': elements, 'excluded': excluded, 'matrix': M,
            'mask': mask, 'row_labels': row_labels,
            'col_labels': col_labels, 'status': 'ok'}


def refit_amplitudes(report, config=None, layer=None):
    """公共窗口振幅重拟合（5B-1 分层版）。

    report['comparable_set'] 含 'layers'（build_comparable_set 5B-1
    结构）→ 对非空层分别重拟合：每层使用层内公共窗口与层内排除，
    不复用旧窗口/旧拟合（层内窗口由层成员私有窗口交重新计算）；
    返回 {'status', 'layers': {layer: 单层结果},
          'cross_layer_comparison'}；osc 层结果附振幅偏差警告
    （纯元数据，不改变拟合数据与拟合值）。
    旧式单集合 report（无 'layers'）→ 保持单集合行为（兼容手工构造
    的最小 report，测试 C1 场景）。
    """
    config = config or PairingSymmetryConfig()
    cs = report['comparable_set']
    if 'layers' not in cs:
        window = cs['common_window']
        form = cs['common_form']
        if window is None or form is None or not cs['member_keys']:
            return _refit_unavailable(
                '可比集合为空或无公共窗口/共同形式，无可重拟合元素')
        return _refit_single(report, config, tuple(window), form,
                             cs['member_keys'], cs['excluded'])
    layers_sel = ([layer] if layer is not None else [
        L for L in (LAYER_OSC, LAYER_NON_OSC)
        if cs['layers'][L]['member_keys']])
    out = {}
    for L in layers_sel:
        linfo = cs['layers'][L]
        w = linfo['common_window']
        if w is None or linfo['common_form'] is None:
            out[L] = _refit_unavailable(
                '层内无公共窗口或共同形式，无可重拟合元素')
            continue
        single = _refit_single(report, config, tuple(w),
                               linfo['common_form'], linfo['member_keys'],
                               linfo['excluded'])
        if L == LAYER_OSC:
            single['amplitude_bias_warning'] = OSC_AMPLITUDE_BIAS_WARNING
        out[L] = single
    return {'status': 'ok', 'layers': out,
            'cross_layer_comparison': cs['cross_layer_comparison']}


# ----------------------------------------------------------------------
def _sphere(a1, a2):
    return np.array([np.cos(a1), np.sin(a1) * np.cos(a2),
                     np.sin(a1) * np.sin(a2)])


def _matrix_obs(M, mask):
    M = np.asarray(M, float)
    n = M.shape[0]
    if mask is None:
        mask = np.isfinite(M)
    return [(i, j) for i in range(n) for j in range(n) if mask[i, j]]


def fit_general_rank1(M, mask=None, W=None):
    """模型一：M ≈ σ·u·v^T（一般秩一）。

    球面参数化 [log σ, θ_u1, θ_u2, θ_v1, θ_v2]（5 参数）：消除尺度与
    范数规范自由度，参数计数与 Jacobian 秩才有意义。
    缺失元素经 mask 跳过；W 为逐项权重（默认 1）。

    返回 ok/sigma/u/v/residual(SSE)/rel_residual(SSE/Σw·M²)/n_obs/
    n_eff_params/jac_rank/singular_values/condition_number/psd_min_eig/
    near_zero_ratio/degeneracy_notes。
    """
    M = np.asarray(M, float)
    n = M.shape[0]
    obs = _matrix_obs(M, mask)
    if W is None:
        W = np.ones((n, n))
    W = np.asarray(W, float)
    if len(obs) < 5:
        return {'ok': False, 'reason': f'有效元素 {len(obs)} < 参数 5，'
                                       '拒绝拟合（可辨识性不足）',
                'n_obs': len(obs)}
    mask_fp = tuple(sorted((int(i), int(j)) for i, j in obs))
    sst = float(np.sum([W[i, j] * M[i, j] ** 2 for i, j in obs]))
    if sst == 0.0:
        return {'ok': False, 'reason': '全零矩阵（数据无信息），拒绝拟合',
                'n_obs': len(obs), 'mask_fingerprint': mask_fp}
    M0 = np.where(np.isfinite(M), M, 0.0)
    U, s, Vt = np.linalg.svd(M0)
    u0, v0 = U[:, 0], Vt[0, :]
    a1u = float(np.arccos(np.clip(u0[0], -1, 1)))
    a2u = float(np.arctan2(u0[2], u0[1]))
    a1v = float(np.arccos(np.clip(v0[0], -1, 1)))
    a2v = float(np.arctan2(v0[2], v0[1]))
    p0 = [float(np.log(max(s[0], 1e-12))), a1u, a2u, a1v, a2v]

    def resid(p):
        sig = np.exp(p[0])
        u = _sphere(p[1], p[2])
        v = _sphere(p[3], p[4])
        return np.array([np.sqrt(W[i, j]) * (M[i, j] - sig * u[i] * v[j])
                         for i, j in obs])
    # 只捕获数据级异常（与 _try 语义一致）；程序错误原样传播
    try:
        res = least_squares(resid, p0, method='lm', max_nfev=40000)
    except (ValueError, FileNotFoundError) as e:
        return {'ok': False, 'reason': f'拟合数据级异常: {e}',
                'n_obs': len(obs), 'mask_fingerprint': mask_fp}
    sig = np.exp(res.x[0])
    u = _sphere(res.x[1], res.x[2])
    v = _sphere(res.x[3], res.x[4])
    # 全局符号规范：u[0] < 0 时 u、v 同时取反（σ·u·v^T 不变）；
    # 消除 SVD 初值带来的双反号，使输出可直接比较
    if u[0] < 0:
        u, v = -u, -v
    sse = float(np.sum(res.fun ** 2))
    jac = np.asarray(res.jac)
    rank, jac_svals = _jac_rank(jac)
    s_vals = np.linalg.svd(M0, compute_uv=False)
    # 条件数原因分类（D5 3.1 审计 m4）：数学退化 vs 数值失败
    cond = None
    cond_reason = 'finite'
    if s_vals[0] > 0 and s_vals[-1] > 0:
        cond = float(s_vals[0] / s_vals[-1])
        if not np.isfinite(cond) or s_vals[-1] <= 1e-14 * s_vals[0]:
            cond, cond_reason = None, 'degenerate_rank'
    elif s_vals[0] > 0 and s_vals[-1] == 0.0:
        cond, cond_reason = None, 'degenerate_rank'
    out = {'ok': True, 'sigma': float(sig), 'u': u, 'v': v,
           'residual': sse,
           'rel_residual': float(sse / sst) if sst > 0 else None,
           'n_obs': len(obs), 'n_eff_params': 5,
           'jac_rank': rank,
           'jac_singular_values': jac_svals,
           'singular_values': [float(x) for x in s_vals],
           'condition_number': cond,
           'condition_number_reason': cond_reason,
           'mask_fingerprint': mask_fp,
           'identifiable': bool(len(obs) >= 5 and rank == 5),
           'degeneracy_notes': []}
    if rank < 5:
        out['degeneracy_notes'].append(
            f'Jacobian 秩 {rank} < 5：存在连续不可辨识方向（含球面参数化'
            '坐标奇点等保守误报情形，需结合数据检查）')
    return out


def fit_symmetric_rank1(M, mask=None, W=None):
    """模型二：M ≈ m·m^T（对称秩一，m ≥ 0 非负约束）。

    振幅矩阵语义：包络振幅非负，m 的符号不携带物理信息；非负约束
    保证参数化与振幅定义一致（不强行允许负振幅）。连续参数 3 个。
    返回 ok/m/residual/rel_residual/n_obs/n_eff_params/jac_rank/
    identifiable/near_zero_ratio/degeneracy_notes。
    """
    M = np.asarray(M, float)
    n = M.shape[0]
    obs = _matrix_obs(M, mask)
    if W is None:
        W = np.ones((n, n))
    W = np.asarray(W, float)
    if len(obs) < 3:
        return {'ok': False, 'reason': f'有效元素 {len(obs)} < 参数 3，'
                                       '拒绝拟合（可辨识性不足）',
                'n_obs': len(obs)}
    mask_fp = tuple(sorted((int(i), int(j)) for i, j in obs))
    sst = float(np.sum([W[i, j] * M[i, j] ** 2 for i, j in obs]))
    if sst == 0.0:
        return {'ok': False, 'reason': '全零矩阵（数据无信息），拒绝拟合',
                'n_obs': len(obs), 'mask_fingerprint': mask_fp}
    g = fit_general_rank1(M, mask, W)
    if g['ok']:
        # 非负初值：|u+v| 方向 × sqrt(σ) 尺度（振幅矩阵语义）
        m0 = np.abs(0.5 * (g['u'] + g['v']))
        scale = math.sqrt(max(g['sigma'], 1e-6))
    else:
        m0 = np.sqrt(np.maximum(np.diag(np.where(np.isfinite(M), M, 0.0)),
                                1e-12))
        scale = 1.0
    m0 = m0 / (np.linalg.norm(m0) or 1.0) * scale

    def resid(p):
        return np.array([np.sqrt(W[i, j]) * (M[i, j] - p[i] * p[j])
                         for i, j in obs])
    # 只捕获数据级异常（与 _try 语义一致）；程序错误原样传播
    try:
        res = least_squares(resid, np.maximum(m0, 1e-9),
                            bounds=(np.zeros(3), np.full(3, np.inf)),
                            method='trf', max_nfev=40000)
    except (ValueError, FileNotFoundError) as e:
        return {'ok': False, 'reason': f'拟合数据级异常: {e}',
                'n_obs': len(obs), 'mask_fingerprint': mask_fp}
    m = np.asarray(res.x)
    sse = float(np.sum(res.fun ** 2))
    jac = np.asarray(res.jac)
    rank, _ = _jac_rank(jac)
    mmax = float(np.max(m)) if len(m) else 0.0
    near_zero = (float(np.min(m) / mmax) if mmax > 1e-12 else 1.0)
    # 边界检测用优化器数值边界带（trf 收敛到 ~1e-9 而非精确 0）
    on_boundary = bool(np.min(m) <= 1e-6 * mmax)
    out = {'ok': True, 'm': m, 'residual': sse,
           'rel_residual': float(sse / sst) if sst > 0 else None,
           'n_obs': len(obs), 'n_eff_params': 3,
           'jac_rank': rank,
           'identifiable': bool(rank == 3),
           'near_zero_ratio': near_zero,
           'on_boundary': on_boundary,
           'm_sign_meaningless': True,   # 振幅语义：m 符号无物理含义
           'mask_fingerprint': mask_fp,
           'degeneracy_notes': []}
    if rank < 3:
        out['degeneracy_notes'].append(
            f'Jacobian 秩 {rank} < 3：存在连续不可辨识方向')
    if near_zero < 0.1:
        # D4 实证：接近零分量估计为噪声主导，相对误差发散——定性警告，
        # 不输出误导性分量比值（阈值非生产判据，仅定性提示）
        out['degeneracy_notes'].append(
            f'接近零分量（min/max={near_zero:.3g}）：该分量估计噪声主导，'
            '相对误差可能发散，分量比值不可靠（定性警告，非校准阈值）')
    if on_boundary:
        # 3.1 审计 m6：参数位于非负约束边界 → 信息准则渐近规律性不成立
        out['degeneracy_notes'].append(
            '参数位于非负约束边界（min(m)=0）：AICc/BIC 渐近规律性条件'
            '不成立，信息准则仅为启发式')
    return out


def aicc_bic_matrix(residual, n_obs, n_params):
    """矩阵拟合的 AICc/BIC（高斯噪声近似，仅参考统计量）。"""
    if n_obs <= n_params + 1 or residual <= 0:
        return float('inf'), float('inf')
    aicc = (n_obs * math.log(residual / n_obs) + 2 * n_params
            + 2 * n_params * (n_params + 1) / (n_obs - n_params - 1))
    bic = n_obs * math.log(residual / n_obs) + n_params * math.log(n_obs)
    return float(aicc), float(bic)


def compare_rank1_models(gen, sym):
    """两级模型比较（同一观测集、同一掩码、同一权重口径下才可比较）。

    返回 {'comparable', 'degenerate', 'delta_aicc', 'delta_bic', 'notes'}。
    - 掩码指纹不一致 → 拒绝比较（不可核实同一观测集）
    - 双残差达数值精度量级（rel_residual < 1e-10，数值精度判定而非统计
      阈值）→ degenerate=True：信息准则无统计意义，ΔAICc 仅反映优化器
      收敛精度
    - 对称模型参数在非负约束边界 → 渐近规律性不成立的说明
    AICc/BIC 仅为参考统计量，不据其认定对称模型成立（设计文档 §4.3）。
    """
    if not (gen.get('ok') and sym.get('ok')):
        return {'comparable': False, 'degenerate': None,
                'delta_aicc': None, 'delta_bic': None,
                'notes': '至少一个模型拟合失败，不可比较'}
    if gen.get('mask_fingerprint') != sym.get('mask_fingerprint'):
        return {'comparable': False, 'degenerate': None,
                'delta_aicc': None, 'delta_bic': None,
                'notes': '两模型观测掩码不一致（无法核实同一观测集），'
                         '拒绝比较'}
    if gen['n_obs'] != sym['n_obs']:
        return {'comparable': False, 'degenerate': None,
                'delta_aicc': None, 'delta_bic': None,
                'notes': '观测数不一致，不可比较'}
    ga, gb = aicc_bic_matrix(gen['residual'], gen['n_obs'], 5)
    sa, sb = aicc_bic_matrix(sym['residual'], sym['n_obs'], 3)

    def _f(x):
        return float(x) if np.isfinite(x) else None
    notes = ['AICc/BIC 仅为参考统计量：模型选择需结合残差、可辨识性与'
             '物理前提，不得仅凭 AICc 认定对称模型成立']
    degenerate = False
    if (gen.get('rel_residual') is not None
            and sym.get('rel_residual') is not None
            and gen['rel_residual'] < 1e-10
            and sym['rel_residual'] < 1e-10):
        degenerate = True
        notes.append('双残差达数值精度量级（rel_residual<1e-10，数值精度'
                     '判定）：信息准则无统计意义，ΔAICc/BIC 仅反映优化器'
                     '收敛精度，不得作为统计证据')
    if sym.get('on_boundary'):
        notes.append('对称模型参数位于非负约束边界：AICc/BIC 渐近规律性'
                     '条件不成立，仅为启发式')
    return {'comparable': True, 'degenerate': degenerate,
            'delta_aicc': _f(sa - ga) if np.isfinite(sa) and np.isfinite(ga)
            else None,
            'delta_bic': _f(sb - gb) if np.isfinite(sb) and np.isfinite(gb)
            else None,
            'aicc_general': _f(ga), 'aicc_symmetric': _f(sa),
            'bic_general': _f(gb), 'bic_symmetric': _f(sb),
            'notes': notes}


def _stage3_single(M, mask, row_labels, col_labels):
    """单层两级矩阵模型 + 可辨识性诊断（原 run_stage3 主体）。"""
    n_obs = int(np.sum(mask))
    missing = [[row_labels[i], col_labels[j]]
               for i in range(M.shape[0]) for j in range(M.shape[1])
               if not mask[i, j]]
    # PSD 诊断（仅完整矩阵；部分矩阵 not_applicable）
    psd = None
    if np.all(mask):
        psd = float(np.linalg.eigvalsh(0.5 * (M + M.T))[0])
    gen = fit_general_rank1(M, mask)
    sym = fit_symmetric_rank1(M, mask)
    cmp_ = compare_rank1_models(gen, sym)
    return {
        'general_rank1': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in gen.items()},
        'symmetric_rank1': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in sym.items()},
        'model_comparison': cmp_,
        'diagnostics': {
            'matrix_shape': list(M.shape),
            'n_observed': n_obs,
            'missing_entries': missing,
            'is_partial': bool(missing),
            'psd_min_eig': psd,
            'psd_note': ('完整矩阵的最小特征值（对称部分）为负 → 对称模型'
                         '与数据不相容的迹象（需显著性，噪声下不单凭点估计'
                         '判断）；部分矩阵不适用'
                         if psd is not None else '部分矩阵：PSD 检查不适用'),
            'identifiability': {
                'general_jac_rank': gen.get('jac_rank'),
                'symmetric_jac_rank': sym.get('jac_rank'),
                'general_identifiable': gen.get('identifiable'),
                'symmetric_identifiable': sym.get('identifiable'),
                'note': ('可辨识性由 Jacobian 秩 + 有效元素数 ≥ 参数数'
                         '判断；元素数达标不是可辨识的充分条件'),
            },
            'numerical_stability': {
                'singular_values': (gen.get('singular_values')
                                    if gen.get('ok') else None),
                'condition_number': gen.get('condition_number'),
                'cond_threshold': 'uncalibrated',
                'note': ('条件数仅作数值报告（D4 的 cond>30 不采用为生产'
                         '判据）；退化迹象见 degeneracy_notes'),
            },
            'fit_quality': {
                'general_residual': gen.get('residual'),
                'general_rel_residual': gen.get('rel_residual'),
                'symmetric_residual': sym.get('residual'),
                'symmetric_rel_residual': sym.get('rel_residual'),
                'note': '绝对残差（SSE）与相对残差（SSE/ΣwM²）同时报告',
            },
            'degeneracy_notes': (gen.get('degeneracy_notes') or [])
            + (sym.get('degeneracy_notes') or []),
        },
    }


def run_stage3(report, config=None):
    """阶段 3 主流程（5B-1 分层版）：各层独立公共窗口振幅重拟合 →
    两级矩阵拟合 → 诊断。跨层比较默认拒绝。

    输入：build_comparable_set 的输出（含 records_by_element）。
    输出 dict：
    - layers：{layer: {'refit', 'refit_matrix', 'refit_mask',
      'row_labels', 'col_labels', 'general_rank1', 'symmetric_rank1',
      'model_comparison', 'diagnostics', ...}}；osc 层附
      amplitude_bias_warning（纯元数据，不改变拟合值）
    - cross_layer_comparison：跨层比较拒绝声明
    - uncertainty / uncalibrated / no_symmetry_conclusions
    旧式单集合 report（无 'layers'）→ 保持旧输出结构（兼容）。
    """
    config = config or PairingSymmetryConfig()
    refit = refit_amplitudes(report, config)
    common = {
        'uncertainty': {
            'joint_covariance': 'unavailable',
            'element_independence_assumed': False,
            'record_level_bootstrap': 'pending',
            'per_element_errors': 'diagonal_param_errors_only',
            'note': ('未实现联合矩阵 bootstrap；九个矩阵元素共享同一 MPS'
                     '基态，不得视为独立样本；单元素拟合误差不代表矩阵'
                     '参数的完整统计不确定度'),
        },
        'uncalibrated': [
            ('refit_min_points', '公共窗口重拟合额外最小点数（仅使用数学'
                                 '下限 n≥4）'),
            ('rank1_residual_threshold', '秩一残差拒绝阈值 t_res（本轮只'
                                         '报告残差数值，不作生产判据）'),
            ('condition_number_threshold', '条件数未设生产判据（D4 的 '
                                           'cond>30 不采用，只作定性提示）'),
            ('near_zero_frac', '近零分量警告比例（本轮只报告比值与定性'
                               '警告，不作排除判据）'),
            ('min_effective_entries', '部分矩阵最小有效元素数（本轮只按'
                                      '参数数硬下限拒绝）'),
            ('zero_crossing_reliability_threshold', '过零可靠性以'
                                                    ' final_noise_floor '
                                                    '本身为标尺，任何倍率'
                                                    '因子均未校准'),
        ],
        'no_symmetry_conclusions': (
            '本阶段只回答各振荡证据层内振幅矩阵的秩一结构与数值稳定性'
            '问题；包络振幅为 |C| 空间非负量，不含符号信息；不产生任何'
            '配对对称性、超导序参量或物理相态判定'),
    }
    if 'layers' not in refit:
        # 旧式单集合 report：保持旧输出结构（兼容手工构造的最小 report）
        out = {
            'refit': {k: v for k, v in refit.items()
                      if k not in ('matrix', 'mask')},
            'refit_matrix': (refit['matrix'].tolist()
                             if refit['matrix'] is not None else None),
            'refit_mask': (refit['mask'].tolist()
                           if refit['mask'] is not None else None),
            'row_labels': refit['row_labels'],
            'col_labels': refit['col_labels'],
            'general_rank1': None, 'symmetric_rank1': None,
            'model_comparison': None, 'diagnostics': None,
        }
        out.update(common)
        if refit['matrix'] is None:
            out['diagnostics'] = {'status': 'unavailable',
                                  'reason': refit['reason']}
            return out
        out.update(_stage3_single(refit['matrix'], refit['mask'],
                                  refit['row_labels'], refit['col_labels']))
        return out
    out = dict(common)
    out['cross_layer_comparison'] = refit['cross_layer_comparison']
    out['layers'] = {}
    for L, single_refit in refit['layers'].items():
        if single_refit['matrix'] is None:
            layer_out = {
                'refit': {k: v for k, v in single_refit.items()
                          if k not in ('matrix', 'mask')},
                'refit_matrix': None, 'refit_mask': None,
                'row_labels': [], 'col_labels': [],
                'general_rank1': None, 'symmetric_rank1': None,
                'model_comparison': None,
                'diagnostics': {'status': 'unavailable',
                                'reason': single_refit['reason']},
            }
        else:
            layer_out = {
                'refit': {k: v for k, v in single_refit.items()
                          if k not in ('matrix', 'mask')},
                'refit_matrix': single_refit['matrix'].tolist(),
                'refit_mask': single_refit['mask'].tolist(),
                'row_labels': single_refit['row_labels'],
                'col_labels': single_refit['col_labels'],
            }
            layer_out.update(_stage3_single(
                single_refit['matrix'], single_refit['mask'],
                single_refit['row_labels'], single_refit['col_labels']))
        if L == LAYER_OSC:
            layer_out['amplitude_bias_warning'] = OSC_AMPLITUDE_BIAS_WARNING
        out['layers'][L] = layer_out
    return out


# ==========================================================================
# 阶段 4：有符号证据通道 + 三通道状态机
# ==========================================================================
# 符号通道硬数学下限：≥2 个同号点才构成"稳定符号"的最小证据
_SIGN_HARD_MIN_POINTS = 2

# 符号通道状态枚举（与设计文档 §6.1 一致）
SIGN_EXTRACTED = 'extracted'
SIGN_INSUFFICIENT = 'evidence_insufficient'
SIGN_UNAVAILABLE = 'unavailable'

# 幅度通道（阶段 3 结构证据）状态枚举。设计文档 §6.1 的
# supported_candidate 属阶段 5 候选匹配，本阶段不实现；
# 本阶段 magnitude 只报告结构证据等级
MAG_STRUCTURE_OK = 'structure_ok'
MAG_INCONCLUSIVE = 'inconclusive'
MAG_INSUFFICIENT = 'evidence_insufficient'
MAG_INPUT_INCOMPLETE = 'input_incomplete'


def analyze_sign_channel(report, config=None):
    """有符号证据通道（设计文档 §4.5）。

    输入：records_by_element（阶段 1–2 保留的**带符号原始记录**），
    **不使用**阶段 3 的非负振幅矩阵；符号只来自带符号数据，绝不由
    A / |C(r)| / m 的正负推断。

    每元素规则（D4 实证的 v3 规则链）：
    1. 无记录 → unavailable（输入缺失）
    2. 底层分析不可用（INCONCLUSIVE/分析失败）→ evidence_insufficient
       （分析级前置门）
    3. 公共窗口内保留 |C| > noise_floor 的点（噪声底取元素真实分析
       final_noise_floor；噪声底以下点不提供符号证据，排除而非否决）
    4. 保留点 < min_pts（config.sign_min_points，None=待校准，仅用数学
       下限 2）→ evidence_insufficient（噪声主导）
    5. 保留点过零 → unavailable（中性结论，不判物理振荡；单次翻转与
       多次翻转同等处理——不得把单个翻转直接认定为真实振荡）
    6. 保留点同号 → extracted ±1

    返回 {'state', 'per_element', 'relative_signs_available',
          'no_sign_from_amplitudes'}；跨元素符号结构（s_α·s_β）属后续
    候选判定阶段，本阶段不推断。
    """
    config = config or PairingSymmetryConfig()
    cs = report['comparable_set']
    records = report.get('records_by_element', {})
    # 5B-1 分层：符号证据使用元素所属层的层内公共窗口（跨层不混用）
    layers = cs.get('layers')

    def layer_of(key):
        return ((report['elements'].get(key, {}).get('quality_gate')
                 or {}).get('layer'))

    def window_for(key):
        if layers is not None:
            L = layer_of(key)
            if L is None or L not in layers:
                return None
            return layers[L]['common_window']
        return cs['common_window']

    min_pts = (config.sign_min_points
               if config.sign_min_points is not None
               else _SIGN_HARD_MIN_POINTS)
    per_element = {}
    for key in cs['member_keys']:
        window = window_for(key)
        recs = records.get(key)
        if recs is None:
            per_element[key] = {'state': SIGN_UNAVAILABLE, 'sign': None,
                                'reason': '无有符号记录（输入缺失）',
                                'n_points_above_floor': 0, 'crossings': 0,
                                'noise_floor_used': None}
            continue
        el_analysis = report['elements'].get(key, {}).get('analysis')
        meta = (el_analysis or {}).get('meta') or {}
        # 前置门只拦"分析执行失败"；INCONCLUSIVE 不硬拦截——全负稳定
        # 序列会被分析器保守判 INCONCLUSIVE，但符号证据质量由下方的
        # 噪声底/点数/过零规则判定（噪声主导数据在规则层自然落入
        # unavailable/insufficient，无需依赖 classification 字面值）
        if not el_analysis or el_analysis.get('status') != ANALYSIS_OK:
            per_element[key] = {
                'state': SIGN_INSUFFICIENT, 'sign': None,
                'reason': '底层分析执行失败，符号证据前置门不通过',
                'n_points_above_floor': 0, 'crossings': 0,
                'noise_floor_used': None}
            continue
        prep = SuperCorrDecayAnalyzer.group_by_distance(recs)
        r_all = np.asarray(prep.r, dtype=float)
        C_all = np.asarray(prep.C, dtype=float)
        if window is not None:
            m_w = (r_all >= window[0]) & (r_all <= window[1])
            r_w, C_w = r_all[m_w], C_all[m_w]
        else:
            r_w, C_w = r_all, C_all
        floor = meta.get('final_noise_floor')
        keep = np.abs(C_w) > (floor if floor is not None else 0.0)
        rk, Ck = r_w[keep], C_w[keep]
        n_cross = (CorrelationDecayAnalyzer.detect_zero_crossings(rk, Ck)[1]
                   if len(rk) else 0)
        if len(Ck) < min_pts:
            state, sign = SIGN_INSUFFICIENT, None
            reason = (f'窗口内高于噪声底的点仅 {len(Ck)} 个'
                      f'（<{min_pts}），噪声主导')
        elif n_cross > 0:
            state, sign = SIGN_UNAVAILABLE, None
            reason = (f'窗口内高于噪声底的点过零 {n_cross} 次'
                      '（中性结论，不判物理振荡）')
        elif np.all(Ck > 0):
            state, sign = SIGN_EXTRACTED, 1
            reason = f'同号稳定（{len(Ck)} 点高于噪声底）'
        elif np.all(Ck < 0):
            state, sign = SIGN_EXTRACTED, -1
            reason = f'同号稳定（{len(Ck)} 点高于噪声底）'
        else:
            state, sign = SIGN_UNAVAILABLE, None
            reason = '符号不稳定'
        per_element[key] = {'state': state, 'sign': sign, 'reason': reason,
                            'n_points_above_floor': int(len(Ck)),
                            'crossings': int(n_cross),
                            'noise_floor_used': floor}
    states = [v['state'] for v in per_element.values()]
    if any(s == SIGN_EXTRACTED for s in states):
        agg = SIGN_EXTRACTED
    elif states and all(s == SIGN_UNAVAILABLE for s in states):
        agg = SIGN_UNAVAILABLE
    elif states:
        agg = SIGN_INSUFFICIENT
    else:
        agg = SIGN_UNAVAILABLE   # 无任何成员元素（输入不完整）
    # 5B-1：逐层聚合（层内符号证据；跨层符号结构不做）
    per_layer = {}
    if layers is not None:
        for L in LAYER_KEYS:
            ks = [k for k in per_element if layer_of(k) == L]
            if not ks:
                per_layer[L] = {'state': SIGN_UNAVAILABLE,
                                'note': '该层无符号通道成员'}
                continue
            st = [per_element[k]['state'] for k in ks]
            if any(s == SIGN_EXTRACTED for s in st):
                ls = SIGN_EXTRACTED
            elif all(s == SIGN_UNAVAILABLE for s in st):
                ls = SIGN_UNAVAILABLE
            else:
                ls = SIGN_INSUFFICIENT
            per_layer[L] = {'state': ls,
                            'note': '层内符号证据聚合（跨层不合并）'}
    return {
        'state': agg,
        'per_element': per_element,
        'per_layer': per_layer,
        'relative_signs_available': sorted(
            k for k, v in per_element.items() if v['state'] == SIGN_EXTRACTED),
        'note': ('仅逐元素符号证据；跨元素符号结构（s_α·s_β）属后续'
                 '候选判定阶段，本阶段不推断；分层后各层独立聚合'),
        'no_sign_from_amplitudes': (
            '符号只来自带符号记录（records_by_element）；绝不由 A / '
            '|C(r)| / m 的正负推断'),
        'sign_min_points_uncalibrated': (config.sign_min_points is None),
    }


# ---- 5B-3 跨元素相对符号证据（r_αβ = s_α·s_β） ----
RELPAIR_AVAILABLE = 'available'
RELPAIR_UNAVAILABLE = 'unavailable'


def _sign_is_numeric_unit(s):
    """符号值严格校验：数值 ±1（排除 bool——True == 1 的陷阱；
    排除非有限/None/其他类型）。"""
    if isinstance(s, (bool, np.bool_)):
        return False
    if not isinstance(s, (int, float, np.integer, np.floating)):
        return False
    try:
        f = float(s)
    except (TypeError, ValueError):
        return False
    return np.isfinite(f) and f in (1.0, -1.0)


def _finite_or_none(x):
    """数值有限性转换：非有限（NaN/±inf）或不可解释 → None；
    合法有限值保留。用于 provenance 字段（保证严格 JSON 可序列化）。"""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def relative_sign_pair_reports(sign_channel, report):
    """跨元素相对符号证据（5B-3，纯函数）：r_αβ = s_α·s_β，
    s ∈ {−1, +1}。

    只消费 analyze_sign_channel 的逐元素符号状态（**原始相关函数
    符号**，不是拟合振幅符号——本函数不接收任何振幅输入）与
    build_comparable_set 的层归属（elements[k].quality_gate.layer）；
    不重复符号提取与拟合。

    配对规则：
    - 只处理不同元素的无序配对（键排序规范顺序，无自配对、
      无 (i,j)/(j,i) 重复）
    - 双方层归属明确且相同，且 ∈ {osc, non_osc}；unknown、缺失层、
      跨层 → unavailable（即使双方符号值都有效）
    - 双方 state=extracted 且 sign 为数值 ±1（严格校验，bool 排除）
    - 任一不满足 → product=None + 逐条原因；绝不由 None/未知/零/
      非有限推断 ±1
    - 不新增符号可靠性阈值（sign_min_points/noise-floor 判据不变）

    输入缺失（sign_channel=None、report=None、无 per_element）→
    空结果 + 说明，不伪造关系；类型错误（非 dict 非 None）→
    ValueError（数据级异常语义，与项目 _try 捕获契约一致，不伪装成
    空配对结果）。
    """
    if sign_channel is not None and not isinstance(sign_channel, dict):
        raise ValueError(
            'relative_sign_pair_reports: sign_channel 必须是 dict 或 None，'
            f'实际类型 {type(sign_channel).__name__}')
    if report is not None and not isinstance(report, dict):
        raise ValueError(
            'relative_sign_pair_reports: report 必须是 dict 或 None，'
            f'实际类型 {type(report).__name__}')
    per_element = (sign_channel or {}).get('per_element') or {}
    if not isinstance(per_element, dict):
        raise ValueError(
            'relative_sign_pair_reports: sign_channel.per_element 必须是 '
            f'dict，实际类型 {type(per_element).__name__}')
    elements_raw = (report or {}).get('elements') or {}
    if not isinstance(elements_raw, dict):
        raise ValueError(
            'relative_sign_pair_reports: report.elements 必须是 dict，'
            f'实际类型 {type(elements_raw).__name__}')
    layer_of = {}
    for k, el in elements_raw.items():
        qg = el.get('quality_gate') if isinstance(el, dict) else None
        layer_of[k] = (qg or {}).get('layer')

    keys = sorted(per_element.keys())
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            vi = per_element.get(ki) if isinstance(
                per_element.get(ki), dict) else {}
            vj = per_element.get(kj) if isinstance(
                per_element.get(kj), dict) else {}
            Li = layer_of.get(ki)
            Lj = layer_of.get(kj)
            reasons = []
            if Li not in (LAYER_OSC, LAYER_NON_OSC) \
                    or Lj not in (LAYER_OSC, LAYER_NON_OSC):
                reasons.append('层归属缺失或 unknown（' + f'{ki}:{Li}, '
                               f'{kj}:{Lj}），保守拒绝配对')
            elif Li != Lj:
                reasons.append(f'跨层配对被拒绝（{ki}:{Li} vs {kj}:{Lj}），'
                               '不同振荡证据层不得计算相对符号')
            for k, v in ((ki, vi), (kj, vj)):
                st = v.get('state')
                s = v.get('sign')
                if st != SIGN_EXTRACTED:
                    reasons.append(
                        f'{k} 符号状态 {st}（不可用）：'
                        + (str(v.get('reason')) if v.get('reason') else
                           '无原因记录'))
                elif not _sign_is_numeric_unit(s):
                    reasons.append(f'{k} 符号值缺失/非法/非有限'
                                   f'（{s!r}），不得推断 ±1')
            prov = {}
            for k, v in ((ki, vi), (kj, vj)):
                prov[k] = {
                    'sign': (int(v['sign'])
                             if _sign_is_numeric_unit(v.get('sign'))
                             else None),
                    'state': v.get('state'),
                    'layer': layer_of.get(k),
                    # 数值字段经有限性转换（NaN/±inf → None），
                    # 保证 provenance 可通过 json.dumps(allow_nan=False)
                    'n_points_above_floor': _finite_or_none(
                        v.get('n_points_above_floor')),
                    'noise_floor_used': _finite_or_none(
                        v.get('noise_floor_used')),
                    'reason': v.get('reason'),
                }
            if reasons:
                pairs.append({
                    'pair': [ki, kj],
                    'layer': None,
                    'product': None,
                    'state': RELPAIR_UNAVAILABLE,
                    'reason': reasons,
                    'provenance': prov,
                })
            else:
                si = int(vi['sign'])
                sj = int(vj['sign'])
                pairs.append({
                    'pair': [ki, kj],
                    'layer': Li,
                    'product': si * sj,
                    'state': RELPAIR_AVAILABLE,
                    'reason': [],
                    'provenance': prov,
                })
    n_avail = sum(1 for p in pairs if p['state'] == RELPAIR_AVAILABLE)
    return {
        'pairs': pairs,
        'n_pairs': len(pairs),
        'n_available': n_avail,
        'n_unavailable': len(pairs) - n_avail,
        'semantics': (
            '相对符号 r_αβ = s_α·s_β 只消费逐元素原始相关函数符号'
            '（analyze_sign_channel 输出）；不含拟合振幅符号、不推断'
            '缺失符号、不跨层比较；product=+1/−1 仅为符号关系观测'),
        'no_physical_conclusion': (
            '本模块不判断任何配对表示、d 波方向、d+id、TRSB 或 '
            'ODLRO；相对符号关系即使可靠，也只是后续物理判定的'
            '证据之一'),
    }


def run_stage4(report, config=None):
    """三通道状态机：magnitude（阶段 3 结构证据）+ sign（有符号证据）
    + phase（blocked_by_data）。

    三通道独立记录状态与依据，互不覆盖；运行状态（run_status）与
    证据状态分离。任何通道失败不得转成物理阴性结论。

    magnitude 状态映射（结构证据等级，非候选对称性判定）：
    - 无可比较元素 → input_incomplete
    - 一般秩一拟合拒绝/失败 → evidence_insufficient
    - 拟合成功但局部不可辨识 → inconclusive
    - 拟合成功且局部可辨识 → structure_ok（候选匹配属阶段 5）
    phase 恒 blocked_by_data（实值数据不可恢复完整相对复相位）。
    """
    config = config or PairingSymmetryConfig()
    st3 = run_stage3(report, config)
    sign = analyze_sign_channel(report, config)
    cs = report['comparable_set']
    st3_layers = st3.get('layers')
    if st3_layers is None:
        # 旧式单集合 report：保持旧状态映射
        g = st3['general_rank1']
        if not cs['member_keys']:
            mag_state, mag_reason = MAG_INPUT_INCOMPLETE, (
                '无可比较元素（输入不完整或质量门全部拒绝）')
        elif not (g and g.get('ok')):
            mag_state, mag_reason = MAG_INSUFFICIENT, (
                g.get('reason') if g else '矩阵拟合不可用')
        elif not g.get('identifiable'):
            mag_state, mag_reason = MAG_INCONCLUSIVE, '; '.join(
                g.get('degeneracy_notes') or ['局部可辨识性不足'])
        else:
            mag_state, mag_reason = MAG_STRUCTURE_OK, (
                '一般秩一模型拟合成功且局部可辨识（结构证据可用；'
                '候选对称性匹配属后续阶段，本阶段不判定）')
        mag_per_layer = None
    else:
        # 5B-1：逐层状态映射 + 聚合（各层证据独立，跨层不互证）
        mag_per_layer = {}
        for L in (LAYER_OSC, LAYER_NON_OSC):
            linfo = cs['layers'].get(L, {})
            li = st3_layers.get(L) or {}
            g = li.get('general_rank1')
            if not linfo.get('member_keys'):
                mag_per_layer[L] = {'state': MAG_INPUT_INCOMPLETE,
                                    'reason': '该层无可比较元素'}
            elif not (g and g.get('ok')):
                mag_per_layer[L] = {
                    'state': MAG_INSUFFICIENT,
                    'reason': g.get('reason') if g else '矩阵拟合不可用'}
            elif not g.get('identifiable'):
                mag_per_layer[L] = {
                    'state': MAG_INCONCLUSIVE,
                    'reason': '; '.join(g.get('degeneracy_notes')
                                       or ['局部可辨识性不足'])}
            else:
                extra = ('；振幅偏差未校正（见 amplitude_bias_warning，'
                         '不得直接用于可靠候选匹配）'
                         if L == LAYER_OSC else '')
                mag_per_layer[L] = {
                    'state': MAG_STRUCTURE_OK,
                    'reason': ('一般秩一模型拟合成功且局部可辨识'
                               '（结构证据可用；候选对称性匹配属后续'
                               '阶段，本阶段不判定）' + extra)}
            if L == LAYER_OSC and linfo.get('member_keys'):
                # 5B-1.2：偏差警告随层显式传递到聚合结果（幅度解释
                # 限制，不代表候选匹配结果）
                mag_per_layer[L]['amplitude_bias_warning'] = \
                    OSC_AMPLITUDE_BIAS_WARNING
        has_members = any(cs['layers'][L]['member_keys']
                          for L in (LAYER_OSC, LAYER_NON_OSC))
        any_ok = any(v['state'] == MAG_STRUCTURE_OK
                     for v in mag_per_layer.values())
        any_inc = any(v['state'] == MAG_INCONCLUSIVE
                      for v in mag_per_layer.values())
        if not has_members:
            mag_state, mag_reason = MAG_INPUT_INCOMPLETE, (
                '无可比较元素（输入不完整或质量门全部拒绝）')
        elif any_ok:
            mag_state, mag_reason = MAG_STRUCTURE_OK, (
                '至少一个振荡证据层的一般秩一模型拟合成功且局部可辨识'
                '（各层证据独立，跨层不互证；osc 层振幅偏差未校正）')
        elif any_inc:
            mag_state, mag_reason = MAG_INCONCLUSIVE, (
                '各层拟合成功但均局部不可辨识（逐层状态见 '
                'magnitude.per_layer）')
        else:
            mag_state, mag_reason = MAG_INSUFFICIENT, (
                '各层矩阵拟合均被拒绝/失败（逐层状态见 '
                'magnitude.per_layer）')
    phase = {'state': 'blocked_by_data',
             'reason': ('实值关联数据原则上不可恢复完整相对复相位；'
                        'd+id/TRSB 类判定不可产生（设计文档 §4.6）')}
    mag_out = {'state': mag_state, 'reason': mag_reason, 'stage3': st3}
    if mag_per_layer is not None:
        mag_out['per_layer'] = mag_per_layer
        # 5B-1.2：osc 层有成员时在 magnitude 聚合顶层显式携带偏差警告；
        # 无 osc 成员（non_osc-only）→ None，不虚构警告适用
        mag_out['osc_amplitude_bias_warning'] = (
            OSC_AMPLITUDE_BIAS_WARNING
            if cs['layers'][LAYER_OSC]['member_keys'] else None)
    return {
        'magnitude': mag_out,
        'sign': sign,
        'phase': phase,
        'run_status': 'ok',
        'state_semantics': {
            'magnitude': (f'{MAG_STRUCTURE_OK}=结构证据可用（候选匹配属'
                          '阶段 5）；{MAG_INCONCLUSIVE}=拟合成功但局部不可'
                          '辨识；{MAG_INSUFFICIENT}=拟合拒绝/失败；'
                          f'{MAG_INPUT_INCOMPLETE}=无可比较元素；'
                          '分层后各振荡证据层独立记录（per_layer），'
                          '聚合取最强层，跨层不互证；'
                          'osc_amplitude_bias_warning 为幅度解释限制'
                          '（|C| 包络振幅未校正系统偏差），不代表候选'
                          '匹配结果'),
            'sign': (f'{SIGN_EXTRACTED}=逐元素稳定符号证据；'
                     f'{SIGN_INSUFFICIENT}=噪声主导/前置门不通过；'
                     f'{SIGN_UNAVAILABLE}=过零/无记录/符号不稳定'),
            'phase': 'blocked_by_data（实值数据不可恢复复相位）',
            'channel_independence': ('三通道独立记录状态与依据；任何'
                                     '通道失败不得被转成物理阴性结论'),
            'run_status': '运行执行状态（ok）；与各证据通道状态分离',
            'layer_semantics': ('振荡证据分层：osc/non_osc/unknown；'
                                '各层独立构建公共窗口与矩阵；跨层比较'
                                '默认拒绝；osc 层振幅偏差未校正'),
        },
        'no_symmetry_conclusions': (
            '本阶段不输出 d 波、s 波、d+id、ODLRO 或最终相态判定；'
            '幅度候选匹配与符号结构推断均属后续阶段'),
    }


# ==========================================================================
# 阶段 5B-2：观测模式候选匹配器（仅结构化观测，无支持性结论）
# ==========================================================================
# 候选幅度模板（归一化、分量顺序 A/B/C）。仅用于几何观测比较
# （欧氏距离），不证明真实配对对称性；d_real 含 6 个置换模板
# （2:1:1 × 3 + 0:1:1 × 3），距离取最小值
CANDIDATE_TEMPLATES = {
    's': [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)],
    'd_real': [(0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5),
               (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
    'magnitude_degenerate': [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)],
}

# 观测报告状态（非支持性）：前提全过但不确定度不可用 → observational；
# 存在阻断原因 → inconclusive（沿用状态机语义，不逆行）
CAND_OBSERVATIONAL = 'observational'
CAND_INCONCLUSIVE = 'inconclusive'

# 候选支持判定执行状态（P0 语义收口；与 supported_candidate 的值解耦，
# None 不再单独承载"未执行/无候选/不确定"多义）：
# - not_evaluated：支持判定未执行（record-level bootstrap 与联合协方差
#   未实现，禁止用点估计加任意容差冒充统计支持结论）——当前实现恒此值
# - evaluated_none / evaluated_candidate：未来语义预留（bootstrap 实现后
#   才可达），当前任何路径不可产生
CAND_STATUS_NOT_EVALUATED = 'not_evaluated'
CAND_STATUS_EVALUATED_NONE = 'evaluated_none'
CAND_STATUS_EVALUATED_CANDIDATE = 'evaluated_candidate'


def candidate_observation_report(layer_info, stage3_layer, su2_by_element,
                                 sign_channel, layer_key, config=None,
                                 rel_pairs=None):
    """单个振荡证据层的观测模式候选报告（5B-2，纯函数）。

    输入：
    - layer_info: comparable_set['layers'][L]（member_keys、收敛状态、
      偏差警告、排除）
    - stage3_layer: run_stage3 输出的 layers[L]（symmetric_rank1、
      general_rank1、diagnostics、amplitude_bias_warning）
    - su2_by_element: build_comparable_set 的 su2 报告
    - sign_channel: analyze_sign_channel 输出（或 None：sign 证据缺失）
    - layer_key: 'osc' | 'non_osc'
    - rel_pairs: relative_sign_pair_reports 输出（或 None：未注入）。
      非 None 时提取本层的逐对积观测（available）与 unavailable 计数
      注入 sign_evidence.relative_sign_pairs——派生观测仅供描述

    语义（任务书 5B-2 §2/§3 + P0 定稿）：
    - observed_anisotropy：归一化非负幅度（分量顺序 A/B/C），仅观测
    - candidate_distances：到理想幅度模板的欧氏距离，仅描述性诊断，
      不是概率/置信度/支持程度
    - supported_candidate 恒 None + supported_candidate_status 恒
      'not_evaluated'（P0：判定未执行与"判定后无候选"语义分离；
      joint covariance 不可用、bootstrap 未实现；禁止点估计加任意
      容差冒充置信区间）
    - status：observational（无阻断）/ inconclusive（存在阻断）
    - 保守：缺字段、非有限值、状态矛盾 → inconclusive + 原因，
      不伪造证据；不新增生产阈值
    """
    checks = {}
    warnings = []
    blocking = []
    members = list(layer_info.get('member_keys') or [])
    su2_entries = {k: su2_by_element.get(k) or {} for k in members}
    su2_gates = [v.get('gate') for v in su2_entries.values()]

    # ---- P1 拟合与可辨识性 + m 有效性 ----
    sym = (stage3_layer or {}).get('symmetric_rank1') or {}
    m = None
    mn = None
    if not members:
        # 空层短路：无可比较元素即可概括，不得虚构 SU(2)/偏差等
        # 检查适用（osc 空层同样不适用振幅偏差警告）
        blocking.append('该层无可比较元素（输入不完整或质量门全部拒绝）')
        checks['fit'] = 'blocked'
    elif not sym:
        blocking.append('对称模型结果缺失（stage3 层报告不可用）')
        checks['fit'] = 'blocked'
    elif sym.get('ok') is not True:
        blocking.append(f"对称模型拟合失败/不可用："
                        f"{sym.get('reason') or '无原因记录'}")
        checks['fit'] = 'blocked'
    elif sym.get('identifiable') is not True:
        blocking.append('对称模型局部不可辨识（Jacobian 秩亏）')
        checks['fit'] = 'blocked'
    else:
        checks['fit'] = 'pass'
        m_arr = np.asarray(sym.get('m'), dtype=float)
        if (m_arr.size != 3 or not np.all(np.isfinite(m_arr))
                or np.any(m_arr < 0) or not np.any(m_arr > 0)):
            blocking.append('对称模型 m 缺失/非有限/非负约束违反/全零'
                            '（保守拒绝，不伪造证据）')
            checks['m_validity'] = 'blocked'
        else:
            checks['m_validity'] = 'pass'
            m = m_arr
            mn = m_arr / np.sum(m_arr)

    if members:
        # ---- P2 收敛 ----
        conv = layer_info.get('layer_convergence') or {}
        if conv.get('stable') is not True:
            blocking.append('层分类未稳定（layer_convergence.stable=False，'
                            '以最后一轮归属为准，不参与候选判定）')
            checks['convergence'] = 'blocked'
        else:
            checks['convergence'] = 'pass'

        # ---- P3 SU(2) ----
        if not su2_gates:
            blocking.append('SU(2) 证据缺失（层成员无 SU(2) 报告）')
            checks['su2'] = 'unavailable'
        elif any(g == 'fail' for g in su2_gates):
            su2_max = max((v.get('max_err') for v in su2_entries.values()
                           if v.get('max_err') is not None), default=None)
            blocking.append(
                f"SU(2) 超标（层内最大 max_err={su2_max:.3g}），"
                '不得用于对称性分类' if su2_max is not None
                else 'SU(2) 超标，不得用于对称性分类')
            checks['su2'] = 'fail'
        elif any(g in ('skipped', 'uncomputable', None) for g in su2_gates):
            blocking.append('SU(2) 证据不可用（skipped/uncomputable/缺失），'
                            '阻断候选判定')
            checks['su2'] = 'unavailable'
        elif all(g == 'pass' for g in su2_gates):
            checks['su2'] = 'pass'
        else:   # 含 uncalibrated（可能混有 pass）
            checks['su2'] = 'uncalibrated'
            su2_max_unc = max(
                (v.get('max_err') for v in su2_entries.values()
                 if v.get('max_err') is not None), default=None)
            warnings.append(
                'SU(2) 阈值未校准：误差已记录但不判通过/拒绝'
                + (f'（层内最大 max_err={su2_max_unc:.3g}）'
                   if su2_max_unc is not None else '')
                + '；候选观测仅作数据描述，不得作支持性结论')

        # ---- P4 振荡偏差 ----
        bias = bool(layer_key == LAYER_OSC) \
            or bool(layer_info.get('amplitude_bias_warning')) \
            or bool((stage3_layer or {}).get('amplitude_bias_warning'))
        if bias:
            checks['amplitude_bias'] = 'warning'
            blocking.append('振荡振幅偏差未校正（|C| 包络振幅与节点位置、'
                            'q/phi、窗口及噪声底相关的系统偏差）：该层幅度'
                            '不得用于候选支持，仅作观测')
            warnings.append(OSC_AMPLITUDE_BIAS_WARNING)
        else:
            checks['amplitude_bias'] = 'none'

        # ---- P5 近零/边界 ----
        # 分别验证 near_zero_ratio 与 on_boundary 的存在性与有效性：
        # 任一缺失/类型不可解释/非有限 → 保守拒绝，不得默认为可靠
        # （5B-2 语义审计缺陷 1 修复）
        nz = sym.get('near_zero_ratio')
        ob = sym.get('on_boundary')
        if nz is None or ob is None:
            missing = [n for n, v in (('near_zero_ratio', nz),
                                      ('on_boundary', ob)) if v is None]
            blocking.append(
                '组件可靠性字段缺失（' + '、'.join(missing) + '）：'
                '无法验证幅度分量可靠性，保守拒绝，不得默认为可靠')
            checks['component_reliability'] = 'unavailable'
        elif (not isinstance(nz, (int, float, np.floating, np.integer))
              or not np.isfinite(nz)
              or not isinstance(ob, (bool, np.bool_))):
            blocking.append('组件可靠性字段类型/取值不可解释（保守拒绝，'
                            '不得默认为可靠）')
            checks['component_reliability'] = 'blocked'
        elif bool(ob) or nz < 0.1:
            blocking.append('幅度分量可靠性不足（近零分量/非负边界：比值'
                            '相对误差发散；不得把近零拟合分量解释为精确零，'
                            '相关候选判据不可判定）')
            checks['component_reliability'] = 'blocked'
        else:
            checks['component_reliability'] = 'pass'

        # ---- P6 有效点数记录（阈值未校准，不阻断，仅记录与警告） ----
        n_obs = ((stage3_layer or {}).get('diagnostics') or {}).get(
            'n_observed')
        if n_obs is None or not np.isfinite(n_obs):
            checks['n_observed'] = 'unavailable'
            warnings.append('有效矩阵元素数不可用')
        else:
            checks['n_observed'] = int(n_obs)
            if int(n_obs) < 9:
                warnings.append(f'有效矩阵元素数少（n_observed={int(n_obs)}），'
                                '统计可靠性风险（阈值未校准，仅记录）')
    else:
        # 空层：其余前提不适用（skipped，不虚构检查适用性）
        for key in ('convergence', 'su2', 'amplitude_bias',
                    'component_reliability', 'n_observed'):
            checks[key] = 'skipped'

    # ---- 观测值（m 可用时输出，独立于前提门——观测是数据事实） ----
    obs = None
    if mn is not None:
        dists = {}
        for name, tpls in CANDIDATE_TEMPLATES.items():
            dists[name] = float(min(
                np.linalg.norm(mn - np.asarray(t, dtype=float))
                for t in tpls))
        obs = {
            'values': [float(x) for x in mn],
            'component_order': ['A', 'B', 'C'],
            'candidate_distances': dists,
            'distance_semantics': ('到理想幅度模板的欧氏距离；仅描述性'
                                   '诊断，不是概率/置信度/支持程度'),
            'template_semantics': ('模板仅用于几何观测比较；d_real 距离'
                                   '取 6 个置换模板的最小值；不证明真实'
                                   '配对对称性'),
        }

    # ---- sign_evidence（只记录现有证据与缺失状态，不推导跨元素符号） ----
    per_layer_state = None
    extracted = []
    if isinstance(sign_channel, dict):
        per_layer_state = (sign_channel.get('per_layer') or {}).get(
            layer_key, {}).get('state')
        available = set(sign_channel.get('relative_signs_available') or [])
        extracted = sorted(set(members) & available)
    # 层内逐对积（派生观测，5B-3；rel_pairs=None 表示未注入——
    # 与"本层无 available 配对"区分：后者是注入后得到空列表）
    rel_avail = None
    rel_unavail = None
    if rel_pairs is not None:
        layer_pairs = [p for p in rel_pairs.get('pairs', [])
                       if p.get('layer') == layer_key]
        rel_avail = [p for p in layer_pairs
                     if p.get('state') == RELPAIR_AVAILABLE]
        rel_unavail = len(layer_pairs) - len(rel_avail)
    sign_evidence = {
        'per_layer_state': per_layer_state,
        'extracted_elements': extracted,
        'relative_sign_pairs': rel_avail,
        'n_relative_sign_pairs_unavailable': rel_unavail,
        'cross_element_sign_structure': 'not_implemented',
        'note': ('逐元素符号为原始观测；relative_sign_pairs 为派生观测'
                 '（r_αβ=s_α·s_β，仅供描述，可作未来候选匹配的输入，'
                 '不产生伙伴定向结论）；跨元素符号模式到配对表示的'
                 '规则化解释未实现（cross_element_sign_structure='
                 "'not_implemented'），不自行推导"),
    }

    status = CAND_INCONCLUSIVE if blocking else CAND_OBSERVATIONAL
    return {
        'layer': layer_key,
        'status': status,
        'observed_anisotropy': obs,
        'supported_candidate': None,
        'supported_candidate_status': CAND_STATUS_NOT_EVALUATED,
        'supported_candidate_semantics': (
            'supported_candidate 恒为 None（未执行统计支持判定）；'
            'supported_candidate_status 恒为 not_evaluated：'
            'joint covariance 不可用、record-level bootstrap 未实现，'
            '禁止用点估计加任意容差冒充置信区间；'
            'evaluated_none / evaluated_candidate 为未来语义预留'
            '（bootstrap 实现后才可达），当前任何路径不可产生'),
        'prerequisite_checks': checks,
        'blocking_reasons': blocking,
        'quality_warnings': warnings,
        'sign_evidence': sign_evidence,
        'no_physical_conclusion': (
            '本报告不判定 s 波、d 波、d+id、TRSB 或 ODLRO；'
            'candidate_distances 仅为几何观测诊断'),
    }


def run_candidate_observations(report, config=None):
    """5B-2 观测模式候选匹配入口：对每个非空振荡证据层输出观测报告。

    复用 run_stage3 与 analyze_sign_channel 的真实结果（不重复拟合、
    不重复符号提取）；跨层不合并。旧式单集合 report（无 layers）→
    状态 unavailable（不伪造分层观测）。

    P0 定稿：额外计算 relative_sign_pair_reports（5B-3）并把各层的
    逐对积观测注入对应层的 sign_evidence.relative_sign_pairs
    （派生观测仅供描述；模式级解释仍 not_implemented）。
    """
    config = config or PairingSymmetryConfig()
    cs = report['comparable_set']
    base = {
        'no_physical_conclusion': (
            '观测模式候选报告不判定 s 波、d 波、d+id、TRSB 或 ODLRO；'
            'candidate_distances 仅为几何观测诊断'),
        'supported_candidate_semantics': (
            'supported_candidate 在所有路径下恒为 None：joint covariance '
            '不可用且 record-level bootstrap 未实现，禁止用点估计加'
            '任意容差冒充置信区间；supported_candidate_status 恒为 '
            'not_evaluated（判定未执行），evaluated_none/'
            'evaluated_candidate 为未来语义预留'),
        'supported_candidate_status': CAND_STATUS_NOT_EVALUATED,
    }
    if 'layers' not in cs:
        base['status'] = 'unavailable'
        base['reason'] = '旧式单集合 report 无分层结构，无法分层观测'
        # 5B-2 语义审计缺陷 2 修复：与正常报告统一结构
        # （supported_candidate 恒 None；不伪造分层观测）
        base['supported_candidate'] = None
        return base
    st3 = run_stage3(report, config)
    sign = analyze_sign_channel(report, config)
    rel = relative_sign_pair_reports(sign, report)
    out = dict(base)
    out['layers'] = {}
    for L in (LAYER_OSC, LAYER_NON_OSC):
        out['layers'][L] = candidate_observation_report(
            cs['layers'][L], st3['layers'].get(L), report.get('su2') or {},
            sign, L, config, rel_pairs=rel)
    return out
