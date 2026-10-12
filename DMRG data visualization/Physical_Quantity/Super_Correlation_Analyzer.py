# -*- coding: utf-8 -*-
"""
超导关联函数衰减类型分析与拟合模块（超导专用适配层）

对已合成的 <Δ†Δ>（list[SuperCorrRecord]）做长距离衰减分析。
输入为 SuperCorrProcessor.Combined_super_correlation 的输出（单键方向）。

本模块是 Correlation_Decay_Analyzer.py（通用衰减分析核心）的薄适配层：
- SuperCorrRecord → (r, C) 输入适配与几何族提取（group_by_distance）
- 预处理编排与完整分析入口（_prepare / analyze）
- 蜂窝键族几何 helper 与超导物理文案（标题 / ylabel / 背景物理解释）

通用算法（噪声底、包络、振荡、候选模型、AICc、稳定性、分类、可视化）
全部继承自 CorrelationDecayAnalyzer，算法与阈值与抽取前逐位一致。
"""

import numpy as np

from .Physical_Data_Record import SuperCorrRecord
from .Correlation_Decay_Analyzer import (
    # 数据结构与通用核心（re-export 保持旧 import 路径兼容）
    ANALYSIS_VERSION, AnalyzerConfig, FitResult, FamilyDiagnostics,
    OscillationInfo, StabilityReport, PreparedData, DecayAnalysisResult,
    CorrelationDecayAnalyzer,
    # 模块级工具（本文件保留的代码仍需使用 + re-export 保持旧 import 路径兼容）
    _fmt_sci, _rel_mad, _mad_abs, _mad_circular, _max_circular,
    _r2_in_space, _scaled_fit_problem, _aicc_bic_core, _logspace_lstsq,
    _local_slopes, _spearman, _window_mask,
)


# ==========================================================================
# 蜂窝键族几何 helper（超导专用：bond 中点两族划分）
# ==========================================================================
def _basis_coords() -> list:
    """HoneycombXCGeometry 的 8 格点基元坐标（与 Honeycomb_Lattice.py 一致，a=1）"""
    hy = np.sqrt(3.0)
    return [
        (0.0, 0.0), (-0.5, 0.5 * hy), (0.0, hy), (-0.5, 1.5 * hy),
        (1.0, 0.0), (1.5, 0.5 * hy), (1.0, hy), (1.5, 1.5 * hy),
    ]


def _bond_midpoint(p: int, q: int) -> tuple:
    """格点 p、q 所成 bond 的中点坐标（只依赖基元坐标与胞编号，不依赖 N）"""
    hx = 3.0
    b = _basis_coords()
    xp = b[p % 8][0] + (p // 8) * hx
    yp = b[p % 8][1]
    xq = b[q % 8][0] + (q // 8) * hx
    yq = b[q % 8][1]
    return ((xp + xq) / 2.0, (yp + yq) / 2.0)


def _family_of_record(record: SuperCorrRecord, a: float = 1.0) -> int:
    """距离族几何定义（已对照 Honeycomb_Lattice.py 基元坐标核对）：
    family 0：bond 中点位移 dy = 0（整基元平移，r = 3m·a 族）
    family 1：bond 中点位移 |dy| = 0.5√3·a（半基元偏移，r = √((1.5+3m)²+0.75)·a 族）"""
    try:
        i, j = record.ref_bond
        k, l = record.target_bond
        xr, yr = _bond_midpoint(i, j)
        xt, yt = _bond_midpoint(k, l)
        dy = yt - yr
        return 0 if abs(dy) < 1e-9 * a else 1
    except Exception:
        return -1   # 无法判定（无 bond 索引时由调用方走 r mod 3 fallback）


def _family_from_r_fallback(r: np.ndarray) -> np.ndarray:
    """无 bond 索引数据（合成测试）的 r mod 3 diagnostic fallback（非正式几何定义）"""
    return (np.abs(np.mod(r, 3.0) - 1.5) < 0.5).astype(int)


# ==========================================================================
# SuperCorrDecayAnalyzer（超导适配层；算法继承自 CorrelationDecayAnalyzer）
# ==========================================================================
class SuperCorrDecayAnalyzer(CorrelationDecayAnalyzer):
    """超导关联函数衰减分析器。公开 API 与抽取前完全一致；
    超导物理文案经类属性注入，算法行为继承自通用核心。"""

    # ---- 超导物理文案（覆盖通用类的默认文案） ----
    _REPORT_TITLE = "超导关联函数衰减分析报告"
    _SUPTITLE_MAIN = "Superconducting Correlation Decay Analysis"
    _COMPARE_SUPTITLE = 'Decay Analysis Comparison (no automatic cross-refbond conclusion)'
    _BG_PHYSICS_NOTE = "可能为凝聚体背景 |<Δ>|²"
    _RAW_UNIT_DESC = "bond 数据点"

    # ----------------------------------------------------------------------
    # 1. 数据预处理（分组 / 噪声底 / 窗口）
    # ----------------------------------------------------------------------
    @staticmethod
    def group_by_distance(record_list: list, config: AnalyzerConfig = None) -> PreparedData:
        """容差分组 + 每组完整保留原始值/统计量/几何族编号。
        主分析用组内平均 C，原始点永不丢失（group_raw）。"""
        config = config or AnalyzerConfig()
        if not record_list:
            raise ValueError("输入 record_list 为空，无法分析")

        # 逐点提取：距离 / 值 / 几何族 / 键方向
        points = []
        for rec in record_list:
            d = float(rec.distance)
            v = float(rec.sup_corr)
            fam = _family_of_record(rec)
            points.append((d, v, fam))
        points.sort(key=lambda t: t[0])
        dists = np.array([p[0] for p in points])
        vals = np.array([p[1] for p in points])
        fams = np.array([p[2] for p in points])

        # 容差聚类（rtol=1e-9：√3 浮点孪生对合并；相邻物理距离差远大于容差，安全）
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
            # 组内几何族：全部可判定时取一致值；否则用众数；-1 表示待 fallback
            known = fs[fs >= 0]
            group_family[gi] = int(np.bincount(known, minlength=2).argmax()) if len(known) else -1
            # 组内符号抵消候选标记（是否记 warning 待噪声底算完后在 prepare 中决定）
            flags = []
            if np.min(vs) < 0 < np.max(vs):
                flags.append('sign_mixed')
            group_flags.append(flags)

        # 族编号 fallback：无法用几何判定(合成数据) → r mod 3
        if (group_family < 0).any():
            fb = _family_from_r_fallback(r)
            group_family[group_family < 0] = fb[group_family < 0]

        prep = PreparedData(
            r=r, C=C, C_abs=C_abs, n_per_distance=n_per,
            group_raw=group_raw, C_min=C_min, C_max=C_max, C_std=C_std,
            group_family=group_family, group_flags=group_flags,
            n_raw=len(points), n_distance=n_g,
        )
        prep.warnings.append(
            f"主分析使用同距离平均 C(r)（distance_aggregation='mean_by_distance'），"
            f"原始 {prep.n_raw} 个 bond 数据点保留在 group_raw")
        return prep

    # ----------------------------------------------------------------------
    @staticmethod
    def _prepare(record_list: list, config: AnalyzerConfig) -> PreparedData:
        """完整预处理编排：分组 → initial 噪声底 → r_max_auto 截断 → final 噪声底。
        计算顺序固定，无循环依赖。"""
        config = config or AnalyzerConfig()
        prep = SuperCorrDecayAnalyzer.group_by_distance(record_list, config)

        # 1. initial noise floor（全距离范围）
        init_floor, init_comp = SuperCorrDecayAnalyzer.estimate_noise_floor(
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
        fin_floor, fin_comp = SuperCorrDecayAnalyzer.estimate_noise_floor(
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
    # 2. 综合接口
    # ----------------------------------------------------------------------
    @staticmethod
    def analyze(super_corr_A: list, config: AnalyzerConfig = None) -> DecayAnalysisResult:
        """完整分析入口：输入已合成的 <Δ†Δ>（list[SuperCorrRecord]），
        输出 DecayAnalysisResult（含完整 config 快照，可追溯）。
        管线主体在通用核心 CorrelationDecayAnalyzer.analyze_generic。"""
        config = config or AnalyzerConfig()
        prep = SuperCorrDecayAnalyzer._prepare(super_corr_A, config)
        return SuperCorrDecayAnalyzer.analyze_generic(prep, config)

    # ----------------------------------------------------------------------
    @staticmethod
    def analyze_batch(datasets: dict, config: AnalyzerConfig = None) -> dict:
        """批量分析：逐数据集独立跑 analyze 并返回 dict[str, DecayAnalysisResult]。
        保留每个 refbond 的 Q、K/ξ、φ、classification、evidence 供人工对比；
        第一版不自动做跨 refbond 的最终物理结论。"""
        config = config or AnalyzerConfig()
        return {name: SuperCorrDecayAnalyzer.analyze(recs, config)
                for name, recs in datasets.items()}

    # ----------------------------------------------------------------------
    # 3. 超导专属可视化覆盖（通用面板 + 超导 ylabel）
    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_signed(ax, r, C, result):
        ax = CorrelationDecayAnalyzer._draw_panel_signed(ax, r, C, result)
        ax.set_ylabel(r'$\langle \Delta^\dagger \Delta \rangle$', fontsize=12)
        return ax

    @staticmethod
    def _draw_panel_loglin(ax, r, C, result):
        ax = CorrelationDecayAnalyzer._draw_panel_loglin(ax, r, C, result)
        ax.set_ylabel(r'$|\langle \Delta^\dagger \Delta \rangle|$', fontsize=12)
        return ax

    @staticmethod
    def _draw_panel_loglog(ax, r, C, result):
        ax = CorrelationDecayAnalyzer._draw_panel_loglog(ax, r, C, result)
        ax.set_ylabel(r'$|\langle \Delta^\dagger \Delta \rangle|$', fontsize=12)
        return ax


# ==========================================================================
# 独立面板保存接口（plot_panel_*：ax=None 自建，save_path 非 None 单独保存）
# ==========================================================================
def _plot_panel(name: str, result: DecayAnalysisResult, r=None, C=None,
                ax=None, save_path: str = None):
    """面板绘制公共入口：drawer 名 → 面板函数"""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(6, 5))
    r = np.asarray(r, dtype=float) if r is not None else np.array([])
    C = np.asarray(C, dtype=float) if C is not None else np.array([])
    drawer = getattr(SuperCorrDecayAnalyzer, f'_draw_panel_{name}')
    if name in ('scan', 'summary'):
        drawer(ax, result)
    else:
        drawer(ax, r, C, result)
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
    return ax


def plot_panel_signed(result: DecayAnalysisResult, r=None, C=None, ax=None,
                      save_path: str = None):
    return _plot_panel('signed', result, r, C, ax, save_path)


def plot_panel_loglin(result: DecayAnalysisResult, r=None, C=None, ax=None,
                      save_path: str = None):
    return _plot_panel('loglin', result, r, C, ax, save_path)


def plot_panel_loglog(result: DecayAnalysisResult, r=None, C=None, ax=None,
                      save_path: str = None):
    return _plot_panel('loglog', result, r, C, ax, save_path)


def plot_panel_detrended(result: DecayAnalysisResult, r=None, C=None, ax=None,
                         save_path: str = None):
    return _plot_panel('detrended', result, r, C, ax, save_path)


def plot_panel_models(result: DecayAnalysisResult, r=None, C=None, ax=None,
                      save_path: str = None):
    return _plot_panel('models', result, r, C, ax, save_path)


def plot_panel_residuals(result: DecayAnalysisResult, r=None, C=None, ax=None,
                         save_path: str = None):
    return _plot_panel('residuals', result, r, C, ax, save_path)


def plot_panel_scan(result: DecayAnalysisResult, ax=None, save_path: str = None):
    return _plot_panel('scan', result, None, None, ax, save_path)


def plot_panel_summary(result: DecayAnalysisResult, ax=None, save_path: str = None):
    return _plot_panel('summary', result, None, None, ax, save_path)
