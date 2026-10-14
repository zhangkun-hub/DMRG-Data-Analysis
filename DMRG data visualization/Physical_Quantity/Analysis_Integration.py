# -*- coding: utf-8 -*-
"""
统一契约集成层（SC / Spin 衰减分析器包装 + Spin Structure Factor 包装）

将现有分析器的输出包装为 AnalysisReport，供后续证据融合
（PhaseDiagnosis）消费。纯增量层：

- 不修改任何现有分析器的返回类型、数值算法或输出结构
- 数值路径与直接调用 analyze()/estimator/find_peaks() 完全一致
  （回归基线不受影响）
- 不捕获并吞掉分析器异常——原始异常（如空输入 ValueError）原样传播
- 通过 **kwargs 兼容 SC 与 Spin 适配层的 analyze 签名差异
  （SuperCorrDecayAnalyzer.analyze(record_list, config)；
   SpinCorrDecayAnalyzer.analyze(record_list, config, N, a)）
- 完整候选模型、模型比较、振荡分析、去趋势结果、稳定性扫描、
  结构因子全量 S(q) 与峰值细节等均保留在 report.payload
- 不把 SC 的 SU(2) 对称性检查结果（SuperCorrProcessor.symmetry_check
  的独立输出）伪装成衰减分析器自身的输出
- 不把峰值位置、峰数或峰值状态映射为物理相态结论
"""

from .Analysis_Contract import (
    AnalysisReport, from_decay_result, from_structure_factor, from_peak_report,
    from_energy_extrapolation, OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
)
from .Spin_Structure_Factor_Analyzer import SpinStructureFactorPeakAnalyzer
from .Energy_Process import EnergyProcessor
from .Charge_Structure_Factor import ChargeStructureFactorEstimator

# 拟合质量指标：按 FitResult 实际字段名（r2/rss/rmse/aic/aicc/bic）提取，
# 存在才收录；不编造缺失数值
_FIT_QUALITY_FIELDS = ('r2', 'rss', 'rmse', 'aic', 'aicc', 'bic')


def analyze_with_report(analyzer_cls, record_list, observable,
                        config=None, **kwargs) -> AnalysisReport:
    """调用指定衰减分析器并包装为 AnalysisReport。

    参数
    ----
    analyzer_cls : SuperCorrDecayAnalyzer 或 SpinCorrDecayAnalyzer 类
    record_list : 已合成的关联记录列表（与分析器 analyze 的输入一致）
    observable  : Analysis_Contract 中的物理量常量
    config      : AnalyzerConfig（None 用分析器默认）
    **kwargs    : 透传给 analyze 的专属参数（如 Spin 的 N、a）

    返回
    ----
    AnalysisReport：payload 为原始 DecayAnalysisResult 对象引用。
    metadata 在契约映射之外补充（不修改 AnalysisReport 字段定义）：
    - metadata['param_errors']：envelope_fit.param_errors 的副本
      （仅拟合存在且非空时收录）
    - metadata['fit_quality']：envelope_fit 上实际存在的质量指标
      （r2/rss/rmse/aic/aicc/bic），存在才收录，值含 NaN 时如实保留
      （序列化时按契约转 null）
    拟合参数值仍只在 parameters；误差与质量指标不混入 parameters。
    """
    result = analyzer_cls.analyze(record_list, config, **kwargs)
    report = from_decay_result(result, analyzer_cls.__name__, observable)

    # 集成层补充：误差棒与拟合优度（独立存放，不污染 parameters）
    env = getattr(result, 'envelope_fit', None)
    if env is not None:
        param_errors = getattr(env, 'param_errors', None)
        if param_errors:
            report.metadata['param_errors'] = dict(param_errors)
        fit_quality = {}
        for name in _FIT_QUALITY_FIELDS:
            value = getattr(env, name, None)
            if value is not None:
                fit_quality[name] = value
        if fit_quality:
            report.metadata['fit_quality'] = fit_quality
    return report


def structure_factor_report(estimator_fn, estimator_arg, q_list, N, a=1.0,
                            **kwargs) -> AnalysisReport:
    """调用 SpinStructureFactorEstimator 入口并包装为 AnalysisReport。

    参数
    ----
    estimator_fn  : SpinStructureFactorEstimator.estimate_sz_sz_from_file
                    或 estimate_spin_dot_from_records（调用方传入实际入口，
                    不修改 estimator 签名）
    estimator_arg : 入口的第一个位置参数（文件路径 / record_list）
    q_list, N, a  : 与 estimator 入口一致的参数
    **kwargs      : 其余入口专属参数（如 estimate_sz_sz_from_file 的 source）

    语义约定：
    - status 与 data_restricted 独立：受限数据（is_restricted=True）仍可
      status='ok'（计算正常完成），两字段不互相推导
    - missing_pairs 非空（>0）时在 from_structure_factor 之外追加一条
      数据质量警告（None=未评估、空=无缺失时均不追加）；不修改契约模块
    - 不捕获底层异常（空文件路径 FileNotFoundError 等原样传播）
    - payload 保留 estimator 原始 dict 引用；complex 数组 S_q 在 payload
      中，默认序列化不展开；显式 include_payload=True 序列化含 complex
      的 payload 仍按契约严格拒绝（TypeError），不添加宽松兜底
    """
    sf_result = estimator_fn(estimator_arg, q_list, N, a, **kwargs)
    report = from_structure_factor(sf_result)
    missing = sf_result.get('missing_pairs')
    if missing:
        report.warnings.append(
            f"{len(missing)} 个预期关联对缺失（受限估计）")
    return report


def peak_analysis_report(sf_result, signal='abs', min_height_frac=0.2,
                         refine=True, q_shape=None) -> AnalysisReport:
    """调用 SpinStructureFactorPeakAnalyzer.find_peaks 并包装为 AnalysisReport。

    sf_result 同时传给 from_peak_report 作为 data_restricted 回退源
    （峰值报告缺 is_restricted 透传字段时按契约规则回退）。

    语义约定：
    - parameters 保持契约既有结构（n_peaks / 原始网格峰位 / 峰高）；
      q_refined 与 fwhm 等细节不并入 parameters，经 payload 访问
    - 峰值位置、峰数与峰值状态是数值层面的检测结果，不映射为
      AFM 或其他物理相态结论
    - 不捕获底层异常；不修改 find_peaks 签名
    """
    peak_dict = SpinStructureFactorPeakAnalyzer.find_peaks(
        sf_result, signal=signal, min_height_frac=min_height_frac,
        refine=refine, q_shape=q_shape)
    return from_peak_report(peak_dict, sf_result)


def energy_extrapolation_report(records, power: int = 1,
                                threshold: float = 2.0) -> AnalysisReport:
    """调用 EnergyProcessor.extrapolate 并包装为 AnalysisReport。

    语义约定（与契约 from_energy_extrapolation 一致）：
    - 复用契约适配器，不重复实现拟合、异常点剔除或结果映射
    - 不捕获底层异常（records < 3 的 ValueError 等原样传播）
    - 不修改绘图/打印行为（extrapolate 的 plt.show 与异常点 print
      副作用保持不变）
    - 不通过解析 stdout 获取异常点信息；误差棒与拟合优度在
      extrapolate 返回值中不存在，按契约保持空值语义，不编造
    - 单一外推结果不映射任何物理相态结论
    """
    e_inf = EnergyProcessor.extrapolate(records, power, threshold)
    return from_energy_extrapolation(
        e_inf, [r.N for r in records], power, threshold)


def charge_structure_factor_report(nf_path, corr_paths, q_list, N,
                                   a=1.0, sequences=None) -> AnalysisReport:
    """调用 ChargeStructureFactorEstimator.estimate_from_files 并包装为
    AnalysisReport。

    契约适配器 from_structure_factor 的 analyzer/observable 按自旋结构因子
    硬编码；本包装在适配后覆写为电荷标识（不改契约适配器签名）。
    其余语义与 from_structure_factor 一致：status 与 data_restricted 正交
    （受限数据仍 status='ok'）、missing_pairs 空、warnings 由实际计数构造、
    payload 保留原始 dict（含复数 S_q，默认不序列化）。
    """
    sf_result = ChargeStructureFactorEstimator.estimate_from_files(
        nf_path, corr_paths, q_list, N, a, sequences)
    report = from_structure_factor(sf_result)
    report.observable = OBSERVABLE_CHARGE_STRUCTURE_FACTOR
    report.analyzer = 'ChargeStructureFactorEstimator'
    # 实际 q 网格统计入 metadata（与 fit_quality 同模式，不改契约适配器；
    # 缺失时保持原有行为，不伪造默认值）
    if sf_result.get('q_grid') is not None:
        report.metadata['q_grid'] = sf_result['q_grid']
    return report
