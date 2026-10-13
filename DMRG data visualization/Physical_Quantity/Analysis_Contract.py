# -*- coding: utf-8 -*-
"""
统一分析结果契约（D3.1，纯增量实现）

为各分析器输出提供统一外壳 AnalysisReport，供未来证据融合
（PhaseDiagnosis）与总控（PhysicalCorrelationAnalyzer）消费。

设计约束（与 D3 盘点结论一致）：
- 不修改任何现有分析器的输出结构与 to_dict()——N200/N400 回归基线
  锁死现有键集，本模块是纯增量层
- status / classification / evidence / data_restricted 语义正交，互不推导：
  status           = 计算状态（是否正常完成计算）
  classification   = 数值拟合模型分类（8 类，不是物理相态）
  evidence         = 证据强度等级
  data_restricted  = 数据受限性（True 受限 / False 不受限 / None 未知）
- classification 不得映射为超导、反铁磁等物理相态名称
- 缺失信息保持 None / 空集合语义，不编造物理结论
- 适配器为纯函数，不修改输入对象（payload 只保留引用）
- 不自动记录绝对文件路径
- 序列化严格：非有限浮点统一转 JSON null，禁止 NaN/Infinity 字面量，
  不使用 default=str 掩盖无法正确序列化的类型
- to_dict()/to_json() 默认不序列化 payload；include_payload=True 时
  payload 必须能被 _convert 与严格 JSON 序列化处理——例如未经转换的
  DecayAnalysisResult dataclass 对象会触发 TypeError（严格拒绝而非
  静默兜底），并非所有 payload 都能自动序列化
"""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---- 计算状态（正交于 classification/evidence/data_restricted） ----
STATUS_OK = 'ok'                       # 计算正常完成（含 classification=INCONCLUSIVE）
STATUS_INCONCLUSIVE = 'inconclusive'   # 计算完成但无可用的有意义结果（如信号全无效）
STATUS_FAILED = 'failed'               # 输入为空/结构损坏（适配器对 None 输入的映射）

# ---- 分析器标识（类名，非路径） ----
ANALYZER_DECAY_SUPERCONDUCTING = 'SuperCorrDecayAnalyzer'
ANALYZER_DECAY_SPIN = 'SpinCorrDecayAnalyzer'
ANALYZER_STRUCTURE_FACTOR = 'SpinStructureFactorEstimator'
ANALYZER_PEAK = 'SpinStructureFactorPeakAnalyzer'
ANALYZER_ENERGY = 'EnergyProcessor'

# ---- 物理量标识 ----
OBSERVABLE_SUPERCONDUCTING_CORRELATION = 'superconducting_correlation'
OBSERVABLE_SPIN_CORRELATION = 'spin_correlation'
OBSERVABLE_SPIN_STRUCTURE_FACTOR = 'spin_structure_factor'
OBSERVABLE_ENERGY = 'energy'


def _convert(obj):
    """递归转换：numpy 标量/数组 → python 原生；非有限浮点 → None；
    未知类型原样返回（由 json.dumps(allow_nan=False) 严格拒绝，不掩盖）。"""
    if isinstance(obj, np.ndarray):
        return [_convert(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


@dataclass
class AnalysisReport:
    """统一分析结果外壳。payload 保留原始分析器结果引用（默认不序列化）。"""
    analyzer: str                        # 分析器类名（非路径）
    observable: str                      # 物理量标识
    status: str                          # STATUS_OK / STATUS_INCONCLUSIVE / STATUS_FAILED
    classification: str = None           # 数值拟合分类（仅衰减分析填写；可为 None）
    parameters: dict = field(default_factory=dict)   # 主要参数摘要
    evidence: str = None                 # 证据强度（仅衰减分析有；可为 None）
    data_restricted: bool = None         # 数据受限性：True/False/None（未知不默认 False）
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)     # 可确认元数据，缺失不补默认值
    payload: Any = None                  # 原始分析器结果（引用；默认不序列化）

    def to_dict(self, include_payload: bool = False) -> dict:
        """紧凑报告 dict。默认不含 payload（大型原始结果）；显式
        include_payload=True 时才递归转换并包含。"""
        out = {
            'analyzer': self.analyzer,
            'observable': self.observable,
            'status': self.status,
            'classification': self.classification,
            'parameters': _convert(self.parameters),
            'evidence': self.evidence,
            'data_restricted': self.data_restricted,
            'warnings': _convert(self.warnings),
            'metadata': _convert(self.metadata),
        }
        if include_payload:
            out['payload'] = _convert(self.payload)
        return out

    def to_json(self, path: str, include_payload: bool = False) -> None:
        """严格 JSON 落盘：allow_nan=False 双保险（非有限值已在 _convert 中
        转 null，遗漏时抛 ValueError 而非静默输出 NaN/Infinity 字面量）。"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(include_payload), f,
                      ensure_ascii=False, indent=2, allow_nan=False)


# ==========================================================================
# 适配器（纯函数：不修改输入对象）
# ==========================================================================
def from_decay_result(result, analyzer: str, observable: str) -> AnalysisReport:
    """DecayAnalysisResult（含 SuperCorrDecayAnalyzer / SpinCorrDecayAnalyzer
    的输出，两者共享该数据结构）→ AnalysisReport。

    映射规则（依据 Correlation_Decay_Analyzer.py 实际字段）：
    - status：result 为 None → 'failed'；否则 'ok'（计算正常完成，
      classification=='INCONCLUSIVE' 是模型不可判定，不是计算失败）
    - classification：result.classification 直通（8 类数值分类，不映射相态）
    - parameters：envelope_fit.params 的副本（不修改原对象）
    - evidence：result.evidence_level 直通
    - data_restricted：衰减分析输出无受限性标记 → None（未知不默认 False）
    - warnings：result.warnings 副本
    - metadata：analysis_version / 规模字段 / r 窗口 / config 阈值快照"""
    if result is None:
        return AnalysisReport(analyzer=analyzer, observable=observable,
                              status=STATUS_FAILED)
    params = {}
    if getattr(result, 'envelope_fit', None) is not None:
        params = dict(result.envelope_fit.params)
    metadata = {
        'analysis_version': getattr(result, 'analysis_version', None),
        'n_points': getattr(result, 'n_points', None),
        'n_distances': getattr(result, 'n_distances', None),
        'n_raw_points': getattr(result, 'n_raw_points', None),
        'r_min': getattr(result, 'r_min', None),
        'r_max': getattr(result, 'r_max', None),
        'verdict': (getattr(result.stability, 'verdict', None)
                    if getattr(result, 'stability', None) is not None else None),
        'config': None,
    }
    if getattr(result, 'config', None) is not None:
        metadata['config'] = dict(vars(result.config))
    return AnalysisReport(
        analyzer=analyzer,
        observable=observable,
        status=STATUS_OK,
        classification=getattr(result, 'classification', None),
        parameters=params,
        evidence=getattr(result, 'evidence_level', None),
        data_restricted=None,
        warnings=list(getattr(result, 'warnings', [])),
        metadata=metadata,
        payload=result,
    )


def from_structure_factor(sf_dict) -> AnalysisReport:
    """compute_structure_factor / estimator 的 dict 输出 → AnalysisReport。

    映射规则（依据 Spin_Structure_Factor.py L132-147 实际字段）：
    - status：sf_dict 为 None → 'failed'；否则 'ok'
    - classification：结构因子无模型分类 → None（不编造）
    - data_restricted：sf_dict['is_restricted'] 直通
    - warnings：由实际计数构造文本（invalid_values / duplicate_pairs /
      skipped_no_coords）；无计数异常时保持空列表
    - metadata：definition / N / 覆盖计数 / 几何 meta（缺失键不补默认值）；
      metadata['meta'] 采用引用语义（非深拷贝）——调用方后续修改
      sf_dict['meta'] 会影响报告内容，适配器自身不修改输入"""
    if sf_dict is None:
        return AnalysisReport(analyzer=ANALYZER_STRUCTURE_FACTOR,
                              observable=OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                              status=STATUS_FAILED)
    warnings = []
    if sf_dict.get('invalid_values', 0):
        warnings.append(f"{sf_dict['invalid_values']} 个非有限关联值被跳过")
    if sf_dict.get('duplicate_pairs', 0):
        warnings.append(f"{sf_dict['duplicate_pairs']} 个重复对按组内平均处理")
    if sf_dict.get('skipped_no_coords'):
        warnings.append(f"{len(sf_dict['skipped_no_coords'])} 个缺坐标对被跳过")
    metadata = {}
    for k in ('definition', 'N', 'measured_pair_count', 'expected_pair_count',
              'pair_coverage', 'source_coverage', 'is_restricted',
              'imaginary_residual'):
        if k in sf_dict:
            metadata[k] = sf_dict[k]
    if sf_dict.get('meta') is not None:
        metadata['meta'] = sf_dict['meta']
    return AnalysisReport(
        analyzer=ANALYZER_STRUCTURE_FACTOR,
        observable=OBSERVABLE_SPIN_STRUCTURE_FACTOR,
        status=STATUS_OK,
        classification=None,
        parameters={},
        evidence=None,
        data_restricted=bool(sf_dict['is_restricted']) if 'is_restricted' in sf_dict else None,
        warnings=warnings,
        metadata=metadata,
        payload=sf_dict,
    )


def from_peak_report(peak_dict, sf_dict=None) -> AnalysisReport:
    """find_peaks 的 dict 输出 → AnalysisReport。

    映射规则（依据 Spin_Structure_Factor_Analyzer.py L213-230 实际字段）：
    - status：peak_dict 为 None → 'failed'；
      find_peaks.status=='ok' → 'ok'；
      'no_peak_above_threshold'/'flat' → 'ok'（计算完成，无峰是合法结果，
      原因入 warnings）；'all_zero'/'all_invalid' → 'inconclusive'
      （信号无效，无有意义的峰结果）；未知状态 → 保守映射 'inconclusive'
      且原始值记入 warnings（不静默吞掉）
    - classification：峰值分析不是模型分类 → None
    - parameters：峰摘要（n_peaks + 原始网格峰位/峰高），无峰为空
    - evidence：无证据等级概念 → None
    - data_restricted：透传字段 'is_restricted' 直通；缺失时回退 sf_dict；
      两者均无 → None（未知不默认 False）
    - warnings：status_reason 非空时记入
    - metadata：definition / N / signal_used / grid_uniform（缺失不补）"""
    if peak_dict is None:
        return AnalysisReport(analyzer=ANALYZER_PEAK,
                              observable=OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                              status=STATUS_FAILED)
    status = peak_dict.get('status')
    unknown_status = False
    if status == 'ok':
        rep_status = STATUS_OK
    elif status in ('no_peak_above_threshold', 'flat'):
        rep_status = STATUS_OK
    elif status in ('all_zero', 'all_invalid'):
        rep_status = STATUS_INCONCLUSIVE
    else:
        # 未知状态（未来 find_peaks 新增状态时可达）：保守映射为
        # inconclusive，原始值记入 warnings，不静默吞掉
        rep_status = STATUS_INCONCLUSIVE
        unknown_status = True
    peaks = peak_dict.get('peaks', []) or []
    parameters = {
        'n_peaks': int(peak_dict.get('n_peaks', len(peaks))),
        'peak_positions': [list(p['q_original']) for p in peaks
                           if p.get('q_original') is not None],
        'peak_heights': [p['height'] for p in peaks
                         if p.get('height') is not None],
    }
    warnings = []
    if unknown_status:
        warnings.append(f"未知峰值分析状态（原值: {status!r}），"
                        f"保守映射为 {STATUS_INCONCLUSIVE}")
    reason = peak_dict.get('status_reason')
    if reason:
        warnings.append(reason)
    restricted = None
    if 'is_restricted' in peak_dict:
        restricted = bool(peak_dict['is_restricted'])
    elif sf_dict is not None and 'is_restricted' in sf_dict:
        restricted = bool(sf_dict['is_restricted'])
    metadata = {}
    for k in ('definition', 'N', 'signal_used'):
        if k in peak_dict:
            metadata[k] = peak_dict[k]
    if peak_dict.get('resolution', {}).get('grid_uniform') is not None:
        metadata['grid_uniform'] = peak_dict['resolution']['grid_uniform']
    return AnalysisReport(
        analyzer=ANALYZER_PEAK,
        observable=OBSERVABLE_SPIN_STRUCTURE_FACTOR,
        status=rep_status,
        classification=None,
        parameters=parameters,
        evidence=None,
        data_restricted=restricted,
        warnings=warnings,
        metadata=metadata,
        payload=peak_dict,
    )


def from_energy_extrapolation(e_inf, ns, power: int = 1, threshold: float = 2.0) -> AnalysisReport:
    """EnergyProcessor.extrapolate 的裸 float 返回值的独立包装函数。

    注意：extrapolate 原函数签名/返回/行为未修改；其异常点剔除信息只经
    print 输出、不随返回值返回，因此本包装的 warnings 保持空列表
    （无法可靠映射，不猜测）。"""
    return AnalysisReport(
        analyzer=ANALYZER_ENERGY,
        observable=OBSERVABLE_ENERGY,
        status=STATUS_OK,
        classification=None,
        parameters={'E_inf': e_inf, 'power': power, 'threshold': threshold,
                    'N_values': list(ns)},
        evidence=None,
        data_restricted=None,
        warnings=[],
        metadata={},
        payload=None,
    )
