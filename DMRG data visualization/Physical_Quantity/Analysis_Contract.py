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
ANALYZER_SC_PAIRING_SYMMETRY = 'PairingSymmetryAnalyzer'
ANALYZER_SC_PAIR_SF = 'SCPairStructureFactorEstimator'
ANALYZER_DENSITY = 'DensityEstimator'
ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT = 'ChargeSpinPeakAlignmentEstimator'

# ---- 物理量标识 ----
OBSERVABLE_SUPERCONDUCTING_CORRELATION = 'superconducting_correlation'
OBSERVABLE_SPIN_CORRELATION = 'spin_correlation'
OBSERVABLE_CHARGE_CORRELATION = 'charge_correlation'
OBSERVABLE_SINGLE_PARTICLE_CORRELATION = 'single_particle_correlation'
OBSERVABLE_SPIN_STRUCTURE_FACTOR = 'spin_structure_factor'
OBSERVABLE_CHARGE_STRUCTURE_FACTOR = 'charge_structure_factor'
OBSERVABLE_ENERGY = 'energy'
OBSERVABLE_SC_PAIRING_SYMMETRY = 'sc_pairing_symmetry'
OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR = 'sc_pair_structure_factor'
OBSERVABLE_LOCAL_DENSITY = 'local_density'
OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT = 'charge_spin_peak_alignment'


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


def from_density_profile(profile_dict) -> AnalysisReport:
    """DensityProcessor.profile_summary 的 dict 输出 → AnalysisReport。

    映射规则：
    - status：profile_dict 为 None → 'failed'；否则 'ok'
    - classification / evidence：密度是数值观测，无模型分类与证据
      等级 → None
    - parameters：N / 逐格点密度向量 / 行平均 / A-B 子晶格平均 /
      晶胞分组平均（数值摘要；不复制可视化）
    - data_restricted：False（单格点全量测量，无源点子集概念）
    - warnings：非有限密度值计数（non_finite_count > 0 时构造；
      向量原样保留，序列化时按契约转 null）
    - metadata：几何与边界元数据（lattice/unit_cell_size/boundary/
      子晶格划分及其推导方式/行分组语义说明）——行平均与子晶格
      平均是空间观测，不构成任何相级支持或反证
    - payload：profile_dict 原对象引用（默认不序列化）
    """
    if profile_dict is None:
        return AnalysisReport(analyzer=ANALYZER_DENSITY,
                              observable=OBSERVABLE_LOCAL_DENSITY,
                              status=STATUS_FAILED)
    params = {
        'N': profile_dict['N'],
        'density_by_site': list(profile_dict['density_by_site']),
        'row_mean': dict(profile_dict['row_mean']),
        'sub_lattice_mean': dict(profile_dict['sub_lattice_mean']),
        'unit_cell_mean': list(profile_dict['unit_cell_mean']),
    }
    geo = profile_dict.get('geometry') or {}
    partition = profile_dict.get('sub_lattice_partition') or {}
    metadata = {
        'N': profile_dict['N'],
        'a': profile_dict.get('a'),
        'lattice_geometry': geo.get('lattice'),
        'unit_cell_size': geo.get('unit_cell_size'),
        'boundary': geo.get('boundary'),
        'row_partition': geo.get('row_partition'),
        'sub_lattice_derivation': partition.get('derivation'),
        'sub_lattice_labels': partition.get('site_labels'),
    }
    warnings = []
    if profile_dict.get('non_finite_count'):
        warnings.append(f"{profile_dict['non_finite_count']} 个非有限"
                        "密度值（向量原样保留，序列化转 null）")
    return AnalysisReport(
        analyzer=ANALYZER_DENSITY,
        observable=OBSERVABLE_LOCAL_DENSITY,
        status=STATUS_OK,
        classification=None,
        parameters=params,
        evidence=None,
        data_restricted=False,
        warnings=warnings,
        metadata=metadata,
        payload=profile_dict,
    )


def from_charge_spin_peak_alignment(align_dict) -> AnalysisReport:
    """电荷-自旋峰位对齐观测（O4）的 dict 输出 → AnalysisReport。

    映射规则：
    - status：align_dict 为 None → 'failed'；否则 'ok'（对齐计算完成，
      可比性不成立 / 无峰均不是计算失败，经 metadata/warnings 表达）
    - classification / evidence：数值对齐观测，无模型分类与证据等级
      → None
    - parameters：n_charge_peaks / n_spin_peaks / charge_peak_nearest_spin
      （逐电荷峰最近自旋峰及 q 距离，可为空列表）/ spin_peak_nearest_charge
      / min_dq / median_dq / max_dq（无可比对时 None）
    - data_restricted：None（对齐观测不引入新的受限性概念；两输入
      各自的受限性经其自身报告表达）
    - warnings：q_grid 不一致（comparability 不成立）时记录；不抛错、
      不静默（comparability 字段显式标记）
    - metadata：comparability（consistent_q_grid /
      inconsistent_q_grid / missing_q_grid / no_peaks 之一）、
      q_grid 摘要（两输入各自的，缺失 None）
    - payload：align_dict 原对象引用（默认不序列化）

    语义边界：峰位距离是数值观测，不构成任何相级支持或反证
    （峰位重合 ≠ 耦合序存在；峰位偏离 ≠ 不存在）。
    """
    if align_dict is None:
        return AnalysisReport(analyzer=ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT,
                              observable=OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
                              status=STATUS_FAILED)
    params = {
        'n_charge_peaks': align_dict['n_charge_peaks'],
        'n_spin_peaks': align_dict['n_spin_peaks'],
        'charge_peak_nearest_spin': align_dict['charge_peak_nearest_spin'],
        'spin_peak_nearest_charge': align_dict['spin_peak_nearest_charge'],
        'min_dq': align_dict.get('min_dq'),
        'median_dq': align_dict.get('median_dq'),
        'max_dq': align_dict.get('max_dq'),
        # qy 周期折叠距离（P1-G.6）：与原始距离分列，不混淆
        'charge_peak_nearest_spin_periodic':
            align_dict.get('charge_peak_nearest_spin_periodic', []),
        'spin_peak_nearest_charge_periodic':
            align_dict.get('spin_peak_nearest_charge_periodic', []),
        'min_dq_periodic': align_dict.get('min_dq_periodic'),
        'median_dq_periodic': align_dict.get('median_dq_periodic'),
        'max_dq_periodic': align_dict.get('max_dq_periodic'),
    }
    warnings = []
    if align_dict.get('comparability') != 'consistent_q_grid':
        warnings.append(
            f"峰位对齐不可比：{align_dict.get('comparability')}"
            f"（{align_dict.get('comparability_note', '')}）")
    return AnalysisReport(
        analyzer=ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT,
        observable=OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
        status=STATUS_OK,
        classification=None,
        parameters=params,
        evidence=None,
        data_restricted=None,
        warnings=warnings,
        metadata={
            'comparability': align_dict.get('comparability'),
            'comparability_note': align_dict.get('comparability_note'),
            'charge_q_grid': align_dict.get('charge_q_grid'),
            'spin_q_grid': align_dict.get('spin_q_grid'),
            'qy_period': align_dict.get('qy_period'),
            'periodic_distance_available':
                align_dict.get('periodic_distance_available'),
            'periodic_note': align_dict.get('periodic_note'),
        },
        payload=align_dict,
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

    注意：正式 e_inf 为全点拟合结果（extrapolate 主结果，B3-A7）；MAD
    候选异常点仅为诊断与敏感性分析，不改变正式结果、不进入 warnings。
    结构化诊断经 EnergyProcessor.diagnose_outliers 获取，由
    energy_extrapolation_report 接入 report.metadata
    ['outlier_diagnosis']（B3-A9）——本适配器不复制该信息。"""
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
