# -*- coding: utf-8 -*-
"""
GreenFunctionDecayAnalyzer（Phase B2）：单粒子关联衰减分析薄适配层。

依据 B1/B1.1 审计（测量源码 CONFIRMED）：
- 四通道算符：ch1 = <c↓(ref)·[弦]·c↓†(j)>（空穴式，自旋下）、
  ch2 = <c↓†(ref)·[弦]·c↓(j)>（粒子式，自旋下）、ch3/ch4 同理自旋上
  （DMRGtJModel/src/measure1.cpp:79-95 + tJ_operators.h:60-63）
- j > i 单向采样（finite_mps_measu_disk.h:232-237）；实值基态
  （数据虚部恒 0，SUPPORTED）
- ch1==ch2、ch3==ch4 逐记录严格相等为**当前数据集的经验事实**
  （B1.1 全量 72/72 组实测），不编码为普适物理恒等式
- 首版四通道独立分析：不自动自旋求和、不粒子/空穴合并、不生成
  总关联；静态关联衰减指数/关联长度**不解释为单粒子谱能隙**
  （B1 §9：需额外假设/动力学数据）
- 不因振荡模型拟合胜出而宣称物理序；不产生 d+id/相态结论

适配方式：距离分组与预处理直接复用 SpinCorrDecayAnalyzer 的同款
实现（格点子格族对单粒子同子格/异子格关联同样适用——同一晶格同一
族定义），算法主体继承通用核心 analyze_generic，零复制。
"""
from .Correlation_Decay_Analyzer import (AnalyzerConfig, CorrelationDecayAnalyzer,
                                         DecayAnalysisResult, PreparedData)
from .Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer


class GreenFunctionDecayAnalyzer(CorrelationDecayAnalyzer):
    """单粒子关联函数衰减分析器（B2 薄适配层）。

    公开入口：analyze(record_list, config=None, N=None, a=1.0)
    → DecayAnalysisResult。N 提供时启用蜂窝格点子格族划分
    （复用 Spin 适配层的族函数——同一定义）。"""

    # ---- 单粒子专属文案（覆盖通用类默认文案） ----
    _REPORT_TITLE = "单粒子关联函数衰减分析报告"
    _SUPTITLE_MAIN = "Single-Particle Correlation Decay Analysis"
    _COMPARE_SUPTITLE = ('Green Function Decay Analysis Comparison '
                         '(no automatic cross-channel conclusion)')
    _BG_PHYSICS_NOTE = ("可能为常数值背景；静态关联衰减不解释为单粒子"
                        "谱能隙")
    _RAW_UNIT_DESC = "格点数据点"

    # ----------------------------------------------------------------------
    # 1. 数据预处理：直接复用 Spin 适配层的分组与准备（族函数为格点
    #    子格族，同一晶格同一族定义；SingleParticleRecord 经
    #    correlation 属性提供分析实部，字段约定一致）
    # ----------------------------------------------------------------------
    @staticmethod
    def group_by_distance(record_list: list,
                          config: AnalyzerConfig = None,
                          N: int = None, a: float = 1.0) -> PreparedData:
        return SpinCorrDecayAnalyzer.group_by_distance(
            record_list, config, N, a)

    @staticmethod
    def _prepare(record_list: list, config: AnalyzerConfig,
                 N: int = None, a: float = 1.0) -> PreparedData:
        return SpinCorrDecayAnalyzer._prepare(record_list, config, N, a)

    # ----------------------------------------------------------------------
    # 2. 综合接口
    # ----------------------------------------------------------------------
    @classmethod
    def analyze(cls, record_list: list, config: AnalyzerConfig = None,
                N: int = None, a: float = 1.0) -> DecayAnalysisResult:
        """完整分析入口：输入单粒子关联记录列表（list[SingleParticleRecord]），
        输出 DecayAnalysisResult（含完整 config 快照，可追溯）。

        管线主体在通用核心 analyze_generic（距离分组/噪声底/窗口扫描/
        包络拟合/振荡诊断/候选模型/分类），与 SC/Spin/Charge 同源。
        N 提供时启用蜂窝格点子格族划分。

        Real-part semantics (B2.3): the pipeline consumes the real part
        of each correlation value via record.correlation; the original
        complex value is preserved in record.value. This analyzer does
        not fit or analyze the imaginary part, and results must not be
        interpreted as an analysis of the full complex correlation.
        """
        config = config or AnalyzerConfig()
        prep = cls._prepare(record_list, config, N, a)
        return cls.analyze_generic(prep, config)

    @classmethod
    def analyze_batch(cls, datasets: dict, config: AnalyzerConfig = None,
                      N: int = None, a: float = 1.0) -> dict:
        """批量分析：逐通道独立跑 analyze；不自动做跨通道物理结论
        （首版四通道独立，不做自旋求和/粒子空穴合并）。"""
        config = config or AnalyzerConfig()
        return {name: cls.analyze(recs, config, N, a)
                for name, recs in datasets.items()}
