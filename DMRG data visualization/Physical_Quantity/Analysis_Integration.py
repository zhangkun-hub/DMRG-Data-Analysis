# -*- coding: utf-8 -*-
"""
统一契约集成层（D3.2：SC / Spin 衰减分析器包装试点）

将现有关联函数衰减分析器的输出包装为 AnalysisReport，供后续
证据融合（PhaseDiagnosis）消费。纯增量层：

- 不修改任何现有分析器的返回类型、数值算法或输出结构
- 数值路径与直接调用 analyze() 完全一致（回归基线不受影响）
- 不捕获并吞掉分析器异常——原始异常（如空输入 ValueError）原样传播
- 通过 **kwargs 兼容 SC 与 Spin 适配层的 analyze 签名差异
  （SuperCorrDecayAnalyzer.analyze(record_list, config)；
   SpinCorrDecayAnalyzer.analyze(record_list, config, N, a)）
- 完整候选模型、模型比较、振荡分析、去趋势结果、稳定性扫描等
  详细数据保留在 report.payload（原始结果对象引用）
- 不把 SC 的 SU(2) 对称性检查结果（SuperCorrProcessor.symmetry_check
  的独立输出）伪装成衰减分析器自身的输出
"""

from .Analysis_Contract import (
    AnalysisReport, from_decay_result,
)

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
