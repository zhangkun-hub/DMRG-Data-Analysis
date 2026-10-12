# -*- coding: utf-8 -*-
"""
自旋关联函数衰减分析模块（自旋专用适配层）

对 CorrelationProcessor.Combined_spin_correlation 合成的 <S_i·S_j>
（list[CorrelationRecord]，其中 S_i·S_j = SzSz + 1/2(SuSd + SdSu)）
做长距离衰减分析。管线主体复用 Correlation_Decay_Analyzer 通用核心，
本模块只负责：
- CorrelationRecord → (r, C) 输入适配与格点族提取
- 自旋专属文案（报告标题 / 图标题）
- 输入校验与族信息不足的如实标记

重要说明：
- 本模块只输出数学拟合结果（幂律 / 指数 / 振荡衰减等衰减分类），
  不据此判定 AFM 长程序、SDW、CDW、PDW 或超导相——物理相判定
  留待后续独立模块。
- 族划分依据蜂窝格点 y 坐标：dy 为 0.5√3·a 的偶数倍 = 同子格对（族 0）、
  奇数倍 = 异子格对（族 1）。现有单子格 sequence 提取的数据只含同子格对
  （族恒 0），族分解诊断由通用核心自动 insufficient 降级，不猜测族标签。
"""

import numpy as np

from .Physical_Data_Record import CorrelationRecord
from .Honeycomb_Lattice import HoneycombXCGeometry
from .Correlation_Decay_Analyzer import (
    ANALYSIS_VERSION, AnalyzerConfig, PreparedData, DecayAnalysisResult,
    CorrelationDecayAnalyzer, _window_mask,
)


class SpinCorrDecayAnalyzer(CorrelationDecayAnalyzer):
    """自旋关联函数衰减分析器。公开入口：
    analyze(record_list, config=None, N=None, a=1.0) → DecayAnalysisResult
    N 提供时启用蜂窝格点族划分；N=None 时族信息如实标记不可用。"""

    # ---- 自旋专属文案（覆盖通用类的默认文案） ----
    _REPORT_TITLE = "自旋关联函数衰减分析报告"
    _SUPTITLE_MAIN = "Spin Correlation Decay Analysis"
    _COMPARE_SUPTITLE = 'Spin Decay Analysis Comparison (no automatic cross-dataset conclusion)'
    _BG_PHYSICS_NOTE = "可能为常数值背景"
    _RAW_UNIT_DESC = "数据点"

    # ----------------------------------------------------------------------
    # 1. 格点族划分（蜂窝几何；与超导的 bond 中点族不同，不复制其逻辑）
    # ----------------------------------------------------------------------
    @staticmethod
    def _spin_family(source_point: int, field_point: int,
                     geometry: HoneycombXCGeometry) -> int:
        """蜂窝格点族：dy = |y_field − y_source| 为 0.5√3·a 的偶数倍 → 族 0
        （同子格对）；奇数倍 → 族 1（异子格对）；无法判定 → -1。"""
        try:
            dy = (geometry.coords[field_point][1]
                  - geometry.coords[source_point][1])
            k = int(round(dy / (np.sqrt(3.0) * geometry.a / 2.0)))
            return abs(k) % 2
        except Exception:
            return -1   # 无法判定（无索引/无几何时由族诊断降级处理）

    # ----------------------------------------------------------------------
    # 2. 数据预处理（分组 / 噪声底 / 窗口；框架与超导适配层同款，只换
    #    值字段与族函数——这是自旋专属适配，非通用核心逻辑的复制）
    # ----------------------------------------------------------------------
    @staticmethod
    def group_by_distance(record_list: list, config: AnalyzerConfig = None,
                          N: int = None, a: float = 1.0) -> PreparedData:
        """容差分组 + 每组完整保留原始值/统计量/几何族编号。
        主分析用组内平均 C，原始点永不丢失（group_raw）。"""
        config = config or AnalyzerConfig()
        if not record_list:
            raise ValueError("输入 record_list 为空，无法分析")
        # 输入校验：记录必须携带格点索引与关联值字段（CorrelationRecord 约定）
        rec0 = record_list[0]
        if not all(hasattr(rec0, f) for f in
                   ('source_point', 'field_point', 'distance', 'correlation')):
            raise ValueError(
                "输入记录缺少 source_point/field_point/distance/correlation 字段"
                "（应为 CorrelationRecord 或其等价结构）")

        geometry = None
        if N is not None:
            geometry = HoneycombXCGeometry(N, a)

        # 逐点提取：距离 / 值 / 格点族
        points = []
        for rec in record_list:
            d = float(rec.distance)
            v = float(rec.correlation)
            fam = (SpinCorrDecayAnalyzer._spin_family(
                rec.source_point, rec.field_point, geometry)
                if geometry is not None else -1)
            points.append((d, v, fam))
        points.sort(key=lambda t: t[0])

        # 容差聚类（rtol=1e-9：浮点孪生距离合并；相邻物理距离差远大于容差，安全）
        groups = []
        for d, v, fam in points:
            if groups and np.isclose(groups[-1][0], d,
                                     rtol=config.rtol_group, atol=config.atol_group):
                groups[-1][1].append((d, v, fam))
            else:
                groups.append([d, [(d, v, fam)]])

        n_g = len(groups)
        r = np.empty(n_g)
        C = np.empty(n_g)
        C_abs = np.empty(n_g)
        C_min = np.empty(n_g)
        C_max = np.empty(n_g)
        C_std = np.empty(n_g)
        n_per = np.empty(n_g, dtype=int)
        group_raw = []
        group_family = np.empty(n_g, dtype=int)
        group_flags = []

        for gi, (_, items) in enumerate(groups):
            ds = np.array([it[0] for it in items])
            vs = np.array([it[1] for it in items])
            fs = np.array([it[2] for it in items])
            r[gi] = float(np.mean(ds))
            C[gi] = float(np.mean(vs))
            C_abs[gi] = abs(C[gi])
            C_min[gi] = float(np.min(vs))
            C_max[gi] = float(np.max(vs))
            C_std[gi] = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
            n_per[gi] = len(vs)
            group_raw.append(vs.copy())
            # 组内几何族：全部可判定时取一致值；否则用众数；-1 表示不可判定
            known = fs[fs >= 0]
            group_family[gi] = int(np.bincount(known, minlength=2).argmax()) if len(known) else -1
            # 组内符号抵消候选标记（是否记 warning 待噪声底算完后在 prepare 中决定）
            flags = []
            if np.min(vs) < 0 < np.max(vs):
                flags.append('sign_mixed')
            group_flags.append(flags)

        # 族编号不可判定（N=None 或索引异常）→ 保持 -1，族诊断自动 insufficient 降级
        prep = PreparedData(
            r=r, C=C, C_abs=C_abs, n_per_distance=n_per,
            group_raw=group_raw, C_min=C_min, C_max=C_max, C_std=C_std,
            group_family=group_family, group_flags=group_flags,
            n_raw=len(points), n_distance=n_g,
        )
        prep.warnings.append(
            f"主分析使用同距离平均 C(r)（distance_aggregation='mean_by_distance'），"
            f"原始 {prep.n_raw} 个数据点保留在 group_raw")
        if N is None:
            prep.warnings.append("未提供 N，蜂窝格点族划分不可用，族分解诊断降级")
        elif (group_family < 0).any():
            prep.warnings.append("部分记录无法判定格点族，族分解诊断降级")
        return prep

    # ----------------------------------------------------------------------
    @staticmethod
    def _prepare(record_list: list, config: AnalyzerConfig,
                 N: int = None, a: float = 1.0) -> PreparedData:
        """完整预处理编排：分组 → initial 噪声底 → r_max_auto 截断 → final 噪声底。
        计算顺序固定，无循环依赖（与超导适配层同款编排）。"""
        config = config or AnalyzerConfig()
        prep = SpinCorrDecayAnalyzer.group_by_distance(record_list, config, N, a)

        # 1. initial noise floor（全距离范围）
        init_floor, init_comp = SpinCorrDecayAnalyzer.estimate_noise_floor(
            prep.r, prep.C_abs, prep.C_std, config)
        prep.initial_noise_floor = init_floor
        prep.initial_noise_floor_components = init_comp

        # 2. r_max_auto 截断：连续 r_max_noise_run 个点低于阈值才截断（防单点尖峰）
        r_max = float(np.max(prep.r))
        if config.r_max_auto and prep.n_distance > 0:
            below = prep.C_abs < init_floor * config.r_max_noise_cut
            run = 0
            cut_at = prep.n_distance
            for i in range(prep.n_distance - 1, -1, -1):
                if below[i]:
                    run += 1
                    if run >= config.r_max_noise_run:
                        cut_at = i + run   # 连续低值段起点
                        break
                else:
                    run = 0
            if cut_at < prep.n_distance:
                # r_max = 连续低值段之前最后一个点
                if cut_at - config.r_max_noise_run >= 0:
                    r_max = float(prep.r[cut_at - config.r_max_noise_run])
                    prep.warnings.append(
                        f"远端连续 {config.r_max_noise_run} 个点低于 effective noise floor×{config.r_max_noise_cut}，"
                        f"自动截断 r_max → {r_max:.6g}")
        prep.r_max_used = min(r_max, config.r_max_frac * float(np.max(prep.r)))

        # 3. final noise floor（截断后重算）
        mask = _window_mask(prep.r, 0.0, prep.r_max_used)
        fin_floor, fin_comp = SpinCorrDecayAnalyzer.estimate_noise_floor(
            prep.r[mask], prep.C_abs[mask], prep.C_std[mask], config)
        prep.final_noise_floor = fin_floor
        prep.final_noise_floor_components = fin_comp

        # 4. 组内符号抵消正式判定（用 initial noise floor 作为保守标尺）
        for gi, flags in enumerate(prep.group_flags):
            if 'sign_mixed' in flags:
                if max(abs(v) for v in prep.group_raw[gi]) > init_floor:
                    flags.append('sign_cancel')
                    prep.warnings.append(
                        f"距离 r={prep.r[gi]:.6g} 处组内两点符号相反且幅度高于噪声标尺，"
                        f"平均后信息被抵消（原始值保留在 group_raw）")
        return prep

    # ----------------------------------------------------------------------
    # 3. 综合接口
    # ----------------------------------------------------------------------
    @classmethod
    def analyze(cls, spin_corr_A: list, config: AnalyzerConfig = None,
                N: int = None, a: float = 1.0) -> DecayAnalysisResult:
        """完整分析入口：输入已合成的 <S_i·S_j>（list[CorrelationRecord]），
        输出 DecayAnalysisResult（含完整 config 快照，可追溯）。
        N 提供时启用蜂窝格点族划分；管线主体在通用核心 analyze_generic。"""
        config = config or AnalyzerConfig()
        prep = cls._prepare(spin_corr_A, config, N, a)
        return cls.analyze_generic(prep, config)

    # ----------------------------------------------------------------------
    @classmethod
    def analyze_batch(cls, datasets: dict, config: AnalyzerConfig = None,
                      N: int = None, a: float = 1.0) -> dict:
        """批量分析：逐数据集独立跑 analyze 并返回 dict[str, DecayAnalysisResult]。
        保留每个数据集的 Q、K/ξ、φ、classification、evidence 供人工对比；
        第一版不自动做跨数据集物理结论。"""
        config = config or AnalyzerConfig()
        return {name: cls.analyze(recs, config, N, a)
                for name, recs in datasets.items()}
