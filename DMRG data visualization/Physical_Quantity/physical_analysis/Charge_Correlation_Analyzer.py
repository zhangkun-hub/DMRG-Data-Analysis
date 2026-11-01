# -*- coding: utf-8 -*-
"""
电荷关联函数衰减分析模块（电荷薄适配层）

对密度-密度关联 <n_i n_j>（list[CorrelationRecord]，如 nf{ver}nf.json
经 CorrelationRecord.extract_data 解析）做长距离衰减分析。

复用策略（零复制、零新算法）：
- 管线主体复用 Correlation_Decay_Analyzer 通用核心（analyze_generic）
- 分组与预处理直接复用 SpinCorrDecayAnalyzer._prepare / group_by_distance
  （其分组逻辑与格点子格族划分是纯几何规则，与物理量无关；
  自旋适配层 docstring 已注明该复用边界）
- 文案经类属性注入本模块专属标题

重要说明：
- 本模块只输出数学拟合结果（幂律 / 指数 / 常数 / 振荡衰减等衰减分类），
  不据此判定 CDW、条纹序或其他物理相——物理相判定留待 PhaseDiagnosis。
- 密度关联数据接近常数正背景（N200 实测 ~0.97-0.99），近常数形态的
  分类行为由通用核心的平坦度判据决定，本模块不预设分类结果。
- 蜂窝 A/B 子格族划分对密度关联适用（全 field 解析下两族数据均衡，
  实测 nf49nf 150 条族分布 75/75）。
"""
import numpy as np

from .Correlation_Decay_Analyzer import (
    AnalyzerConfig, DecayAnalysisResult, CorrelationDecayAnalyzer,
)
from .Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer


class ChargeCorrDecayAnalyzer(CorrelationDecayAnalyzer):
    """电荷关联函数衰减分析器。公开入口：
    analyze(record_list, config=None, N=None, a=1.0) → DecayAnalysisResult
    N 提供时启用蜂窝格点子格族划分（复用自旋适配层的几何规则）；
    N=None 时族信息如实标记不可用。"""

    # ---- 电荷专属文案（覆盖通用类的默认文案） ----
    _REPORT_TITLE = "电荷关联函数衰减分析报告"
    _SUPTITLE_MAIN = "Charge Correlation Decay Analysis"
    _COMPARE_SUPTITLE = 'Charge Decay Analysis Comparison (no automatic cross-dataset conclusion)'
    _BG_PHYSICS_NOTE = "可能为常数值背景"
    _RAW_UNIT_DESC = "数据点"

    # ----------------------------------------------------------------------
    @classmethod
    def analyze(cls, charge_corr: list, config: AnalyzerConfig = None,
                N: int = None, a: float = 1.0) -> DecayAnalysisResult:
        """完整分析入口：输入密度-密度关联记录（list[CorrelationRecord]），
        输出 DecayAnalysisResult（含完整 config 快照，可追溯）。
        分组与预处理复用 SpinCorrDecayAnalyzer._prepare（几何族规则通用）；
        管线主体在通用核心 analyze_generic。"""
        config = config or AnalyzerConfig()
        prep = SpinCorrDecayAnalyzer._prepare(charge_corr, config, N, a)
        return cls.analyze_generic(prep, config)

    # ----------------------------------------------------------------------
    @classmethod
    def analyze_batch(cls, datasets: dict, config: AnalyzerConfig = None,
                      N: int = None, a: float = 1.0) -> dict:
        """批量分析：逐数据集独立跑 analyze 并返回
        dict[str, DecayAnalysisResult]。第一版不自动做跨数据集物理结论。"""
        config = config or AnalyzerConfig()
        return {name: cls.analyze(recs, config, N, a)
                for name, recs in datasets.items()}
