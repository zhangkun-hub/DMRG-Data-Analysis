# -*- coding: utf-8 -*-
"""
超导关联函数衰减类型分析与拟合模块

对已合成的 <Δ†Δ>（list[SuperCorrRecord]）做长距离衰减分析：
  原始 signed C(r) → 长距离窗口选择(r_min 扫描) → 符号变化/极值检测
  → 包络提取(log 空间线性拟合) → 去包络 C(r)/E(r) → cosine 振荡诊断(证据评分制)
  → 5 个候选模型拟合(传统比较层) → 同空间同窗口组内 AICc/BIC 比较
  → 冲突仲裁 → 8 类分类 + evidence level。

输入为 SuperCorrProcessor.Combined_super_correlation 的输出（单键方向）。
本模块不 import、不修改 SuperCorrProcessor 与 SuperCorrRecord。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy import optimize, signal

from .Physical_Data_Record import SuperCorrRecord

# 分析器版本号：配合 config 快照保证历史结果可追溯
ANALYSIS_VERSION = "1.0"


# ==========================================================================
# 1. 数据结构（dataclass）
# ==========================================================================
@dataclass
class AnalyzerConfig:
    """分析器全部可调阈值（默认值经合成数据校准，可按需调整单个阈值而不改流程）"""
    # ---- 距离分组 ----
    rtol_group: float = 1e-9           # 浮点孪生距离合并相对容差
    atol_group: float = 1e-12          # 绝对容差
    # ---- 点数门槛 ----
    n_min_fit: int = 8                 # 包络(2 参数)拟合最小点数
    n_min_osc_fit: int = 10            # 4 参数模型最小点数(<10 warning, <6 拒绝)
    n_min_osc: int = 12                # 振荡分析的最小点数
    n_min_abs: int = 6                 # 全局最小距离数，不足 → INCONCLUSIVE
    # ---- 窗口 ----
    r_max_frac: float = 1.0
    r_max_auto: bool = True            # 自动截断噪声尾
    r_max_noise_cut: float = 3.0       # |C| < noise_floor*cut 视为噪声点
    r_max_noise_run: int = 3           # 连续低于阈值的点数要求(防单点尖峰)
    n_rmin_candidates: int = 20        # r_min 候选数上限
    # ---- 噪声底(effective noise floor / conservative noise scale) ----
    tail_frac_noise: float = 0.25      # 远距尾部分数
    noise_floor_abs_min: float = 1e-15 # 绝对下限
    prom_factor: float = 3.0           # 极值噪声筛选倍数
    # ---- 振荡诊断检查(diagnostic checks，非硬门槛) ----
    n_cross_min: int = 4
    n_extrema_min: int = 4
    spacing_regularity_max: float = 0.4  # 过零/极值间距 MAD/中位数 上限
    r2_cos_min: float = 0.6              # 去包络 cosine 线性空间 R²
    r2_env_log_min: float = 0.85         # 包络 log 空间 R²
    q_stab_max: float = 0.20             # Q rel_mad 上限
    phi_stab_max: float = 0.5            # φ MAD 上限(rad)
    q_grid_points: int = 400             # Lomb-Scargle 粗扫网格
    q_refine_points: int = 200           # 线性化精化网格
    family_sign_purity_warn: float = 0.8 # 符号-距离族耦合警告线
    # ---- 模型比较 ----
    delta_aicc_tie: float = 2.0          # ΔAICc 小于此值视为不可区分
    delta_aicc_osc_min: float = 6.0      # 振荡模型须胜出的幅度(同空间比较时)
    # ---- 稳定性 ----
    k_stab_max: float = 0.30             # 稳定/中等分界
    k_stab_moderate: float = 0.60        # 中等/不稳定分界
    model_consistency_min: float = 0.6   # 模型一致性门槛
    # ---- 不确定度兜底 ----
    bootstrap_n: int = 200               # 残差 bootstrap 次数(仅 pcov 失效兜底)
    pcov_max_cond: float = 1e12          # pcov 条件数上限


@dataclass
class FitResult:
    """单次拟合结果（R² 永远与 RSS 同空间计算，r2_space 强制标注）"""
    model: str                           # 模型标识
    fit_space: str                       # 'linear_C' | 'linear_absC' | 'log_absC'
    r2_space: str                        # 'linear' | 'log'
    params: dict = field(default_factory=dict)
    param_errors: dict = field(default_factory=dict)
    n: int = 0
    k: int = 0
    rss: float = np.nan
    rmse: float = np.nan
    r2: float = np.nan
    aic: float = np.nan
    aicc: float = np.nan                 # n ≤ k+1 时无定义 → NaN
    bic: float = np.nan
    window: tuple = (np.nan, np.nan)     # (r_min, r_max)，比较分组的依据之一
    valid: bool = True
    error_source: str = 'pcov'           # 'pcov' | 'bootstrap' | 'lstsq_cov'
    message: str = ''


@dataclass
class FamilyDiagnostics:
    """距离族分解诊断（几何定义：dy=0 族 / |dy|=0.5√3·a 族）"""
    family_r: dict = field(default_factory=dict)      # {'0': array, '1': array}
    family_C: dict = field(default_factory=dict)
    per_family: dict = field(default_factory=dict)    # 每族诊断结果
    consistency: dict = field(default_factory=dict)   # 每族核心检查是否通过
    structural_artifact_suspected: bool = False
    insufficient: bool = False                        # 族内点数不足(含仅一族有数据)


@dataclass
class OscillationInfo:
    """振荡诊断结果（8 项 diagnostic checks 的证据聚合）"""
    detected: bool = False
    n_cross: int = 0
    cross_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    cross_spacing_regularity: float = np.nan
    n_ext: int = 0
    ext_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    ext_spacing_regularity: float = np.nan
    Q: float = np.nan
    Q_rel_mad: float = np.nan
    phi: float = np.nan
    phi_mad: float = np.nan
    amplitude: float = np.nan
    r2_cos: float = np.nan                 # 去包络 cosine 线性空间 R²
    envelope_form: str = ''                # 'power' | 'exp'
    r2_env_log: float = np.nan
    subwindow_Qs: np.ndarray = field(default_factory=lambda: np.array([]))
    criterion_checks: dict = field(default_factory=dict)   # 8 项逐项结果
    evidence: str = 'none'                 # 'strong'|'moderate'|'weak'|'possible'|'none'|'inconclusive'
    structural_artifact_suspected: bool = False
    structural_warning: str = ''


@dataclass
class StabilityReport:
    """r_min 扫描与稳定性报告（完整保存全部候选，含跳过原因）"""
    scan_table: list = field(default_factory=list)
    recommended_r_min: float = np.nan     # 语义：推荐分析窗口起点(非物理 crossover)
    rel_mad_K: float = np.nan
    rel_mad_xi: float = np.nan
    rel_mad_Q: float = np.nan
    model_consistency: float = np.nan
    spearman_rho: float = np.nan
    verdict: str = 'unstable'             # 'stable' | 'moderate' | 'unstable'
    messages: list = field(default_factory=list)


@dataclass
class PreparedData:
    """预处理后的数据（主分析用同距离平均 C，原始点永不丢失）"""
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    C: np.ndarray = field(default_factory=lambda: np.array([]))
    C_abs: np.ndarray = field(default_factory=lambda: np.array([]))
    n_per_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    group_raw: list = field(default_factory=list)          # 每组原始值(未平均)
    C_min: np.ndarray = field(default_factory=lambda: np.array([]))
    C_max: np.ndarray = field(default_factory=lambda: np.array([]))
    C_std: np.ndarray = field(default_factory=lambda: np.array([]))
    group_family: np.ndarray = field(default_factory=lambda: np.array([]))
    group_flags: list = field(default_factory=list)        # 如 'sign_cancel'
    n_raw: int = 0
    n_distance: int = 0
    initial_noise_floor: float = np.nan
    initial_noise_floor_components: dict = field(default_factory=dict)
    final_noise_floor: float = np.nan
    final_noise_floor_components: dict = field(default_factory=dict)
    r_max_used: float = np.nan
    warnings: list = field(default_factory=list)


@dataclass
class DecayAnalysisResult:
    """衰减分析最终结果（含完整 config 快照，修改阈值后历史结果仍可追溯）"""
    classification: str = ''
    recommended_model: str = ''
    evidence_level: str = 'insufficient'
    distance_aggregation: str = 'mean_by_distance'  # 主分析使用同距离平均值
    nonmonotonic_reason: str = ''     # 'envelope_hump'|'sign_change'|'both'（仅 non_monotonic 非空）
    full_range: tuple = (np.nan, np.nan)
    r_min: float = np.nan
    r_max: float = np.nan
    n_points: int = 0
    n_raw_points: int = 0
    n_distances: int = 0
    fit_space: str = ''
    envelope_fit: FitResult = None
    detrended_fit: FitResult = None
    candidate_fits: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    oscillation: OscillationInfo = None
    family_diagnostics: FamilyDiagnostics = None
    stability: StabilityReport = None
    initial_noise_floor: float = np.nan
    initial_noise_floor_components: dict = field(default_factory=dict)
    final_noise_floor: float = np.nan
    final_noise_floor_components: dict = field(default_factory=dict)
    analysis_version: str = ANALYSIS_VERSION
    physical_form: str = ''
    warnings: list = field(default_factory=list)
    config: AnalyzerConfig = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """嵌套递归转 dict（numpy 数组转 list，含完整 config 与 analysis_version）"""
        def conv(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(v) for v in obj]
            if isinstance(obj, (FitResult, FamilyDiagnostics, OscillationInfo,
                                StabilityReport, PreparedData, AnalyzerConfig)):
                return conv(obj.__dict__)
            if isinstance(obj, float):
                return None if np.isnan(obj) else obj
            return obj
        return conv(self.__dict__)

    def to_json(self, path: str) -> None:
        """保存为 JSON 文件（ensure_ascii=False，NaN 转 null）"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# ==========================================================================
# 10. 工具（模块级私有小函数）
# ==========================================================================
def _rel_mad(x: np.ndarray) -> float:
    """鲁棒相对离散度 rel_mad = 1.4826·median(|x−median|)/|median|"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    if abs(med) < 1e-300:
        return np.nan
    return 1.4826 * np.median(np.abs(x - med)) / abs(med)


def _mad_abs(x: np.ndarray) -> float:
    """绝对 MAD（用于 φ 等无标度量）"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return np.median(np.abs(x - np.median(x)))


def _mad_circular(phis) -> float:
    """圆周 MAD（相位差按圆上距离计算，避免 2π 环绕抵消真实漂移）"""
    phis = np.asarray(phis, dtype=float)
    phis = phis[np.isfinite(phis)]
    if len(phis) == 0:
        return np.nan
    med = np.median(phis)
    dists = np.abs(np.angle(np.exp(1j * (phis - med))))
    return float(np.median(dists))


def _max_circular(phis) -> float:
    """圆周最大距离（子窗口数少时 MAD 会被中位数掩盖单个漂移窗口，用 max 做防护判据）"""
    phis = np.asarray(phis, dtype=float)
    phis = phis[np.isfinite(phis)]
    if len(phis) < 2:
        return np.nan
    dists = np.abs(np.angle(np.exp(1j * (phis[:, None] - phis[None, :]))))
    return float(np.max(dists))


def _r2_in_space(y: np.ndarray, yhat: np.ndarray) -> float:
    """R² 在 (y, yhat) 所在空间内计算；R² 为负如实报告，不截断"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _aicc_bic_core(rss: float, n: int, k: int) -> tuple:
    """手写 AIC / AICc / BIC。n ≤ k+1 时 AICc 无定义 → NaN"""
    if n <= 0 or rss <= 0:
        return (np.nan, np.nan, np.nan)
    base = n * np.log(rss / n)
    aic = base + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.nan
    bic = base + k * np.log(n)
    return (aic, aicc, bic)


def _logspace_lstsq(x: np.ndarray, y: np.ndarray) -> tuple:
    """log 空间线性最小二乘。y > 0 必须保证（调用方负责过滤）。
    返回 (coef, cov, residuals)，cov = (RSS/(n-2))·inv(XᵀX)"""
    X = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, p = X.shape
    rss = float(np.sum(resid ** 2))
    if n > p:
        cov = (rss / (n - p)) * np.linalg.inv(X.T @ X)
    else:
        cov = np.full((p, p), np.nan)
    return coef, cov, resid


def _local_slopes(r: np.ndarray, C: np.ndarray) -> np.ndarray:
    """相邻区间局部斜率 s_i = (C[i+1]−C[i])/(r[i+1]−r[i])（适配不规则间距）"""
    dr = np.diff(r)
    dc = np.diff(C)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = dc / dr
    return s


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关系数"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[m]))
    rb = np.argsort(np.argsort(b[m]))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


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


def _window_mask(r: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    return (r >= r_min - 1e-12) & (r <= r_max + 1e-12)


# ==========================================================================
# SuperCorrDecayAnalyzer（全 @staticmethod，无隐式状态；config 显式传入）
# ==========================================================================
class SuperCorrDecayAnalyzer:

    # ----------------------------------------------------------------------
    # 2. 数据预处理（分组 / 噪声底 / 窗口）
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
    def estimate_noise_floor(r: np.ndarray, C_abs: np.ndarray, C_std: np.ndarray,
                             config: AnalyzerConfig = None) -> tuple:
        """effective noise floor（conservative noise scale）：三来源取 max。
        返回 (noise_floor, components)。"""
        config = config or AnalyzerConfig()
        n = len(r)
        if n == 0:
            return (config.noise_floor_abs_min,
                    {'tail_median': np.nan, 'intra_scatter': np.nan,
                     'absolute_min': config.noise_floor_abs_min})
        n_tail = max(1, int(np.ceil(n * config.tail_frac_noise)))
        tail_median = float(np.median(C_abs[-n_tail:]))
        intra_scatter = float(np.median(C_std)) if len(C_std) else 0.0
        components = {
            'tail_median': tail_median,        # 远距尾部中位数（截断误差主导区）
            'intra_scatter': intra_scatter,    # 组内离散度（within-distance scatter）
            'absolute_min': config.noise_floor_abs_min,
        }
        floor = max(components.values())
        return (floor, components)

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
    # 3. 包络提取与对数空间拟合
    # ----------------------------------------------------------------------
    @staticmethod
    def fit_envelope_logspace(r: np.ndarray, C_abs: np.ndarray, form: str,
                              config: AnalyzerConfig = None) -> FitResult:
        """包络对数空间线性拟合。form ∈ {'power', 'exp'}。
        power: ln|C| = lnA − K·ln r；exp: ln|C| = lnA − r/ξ。
        R² 在 log 空间计算（r2_space='log'），AIC/AICc/BIC 基于 log 空间 RSS。"""
        config = config or AnalyzerConfig()
        r = np.asarray(r, dtype=float)
        C_abs = np.asarray(C_abs, dtype=float)
        pos = C_abs > 0.0
        msg = ''
        if not pos.all():
            n_drop = int((~pos).sum())
            msg = f"剔除 {n_drop} 个非正 |C| 点（log 空间无定义）"
            r, C_abs = r[pos], C_abs[pos]

        form = form.lower()
        if form not in ('power', 'exp'):
            raise ValueError(f"未知包络形式 form={form}，可选 'power' | 'exp'")
        x = np.log(r) if form == 'power' else r
        y = np.log(C_abs)
        coef, cov, _ = _logspace_lstsq(x, y)
        lnA, slope = float(coef[0]), float(coef[1])
        A = float(np.exp(lnA))
        if form == 'power':
            K = float(-slope)
            params = {'A': A, 'K': K}
            param_errors = {'A': A * float(np.sqrt(cov[0, 0])),
                            'K': float(np.sqrt(cov[1, 1]))}
        else:
            xi = float(-1.0 / slope) if abs(slope) > 1e-300 else np.nan
            params = {'A': A, 'xi': xi}
            # σ_ξ = ξ²·σ_{1/ξ}（误差传递）
            param_errors = {'A': A * float(np.sqrt(cov[0, 0])),
                            'xi': (xi ** 2) * float(np.sqrt(cov[1, 1]))}
        yhat = lnA + slope * x
        rss = float(np.sum((y - yhat) ** 2))
        r2_log = _r2_in_space(y, yhat)
        aic, aicc, bic = _aicc_bic_core(rss, len(y), 2)
        return FitResult(
            model=f'envelope_{form}', fit_space='log_absC', r2_space='log',
            params=params, param_errors=param_errors,
            n=len(y), k=2, rss=rss, rmse=float(np.sqrt(rss / len(y))),
            r2=r2_log, aic=aic, aicc=aicc, bic=bic,
            window=(float(np.min(r)), float(np.max(r))),
            error_source='lstsq_cov', message=msg)

    # ----------------------------------------------------------------------
    @staticmethod
    def _envelope_value(fit: FitResult, r: np.ndarray) -> np.ndarray:
        """参数化包络 E(r) 求值（单调平滑，去包络用，不做分段插值）"""
        if fit is None or not fit.params:
            return np.ones_like(r, dtype=float) * np.nan
        p = fit.params
        if 'K' in p:
            return p['A'] * np.asarray(r, dtype=float) ** (-p['K'])
        if 'xi' in p:
            return p['A'] * np.exp(-np.asarray(r, dtype=float) / p['xi'])
        return p.get('A', 1.0) * np.ones_like(r, dtype=float)

    # ----------------------------------------------------------------------
    @staticmethod
    def detect_zero_crossings(r: np.ndarray, C: np.ndarray) -> tuple:
        """相邻点 C_i·C_{i+1}<0 → 线性插值过零位置。返回 (positions, n_cross)"""
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        positions = []
        for i in range(len(C) - 1):
            if C[i] * C[i + 1] < 0:
                t = -C[i] / (C[i + 1] - C[i])
                positions.append(float(r[i] + t * (r[i + 1] - r[i])))
        return (np.array(positions), len(positions))

    # ----------------------------------------------------------------------
    @staticmethod
    def detect_extrema(r: np.ndarray, C: np.ndarray, noise_floor: float,
                       config: AnalyzerConfig = None) -> tuple:
        """斜率符号法检测局部极值（适配不规则间距 r）：
        s_i = ΔC/Δr，s[i−1]>0 且 s[i]<0 → 极大；反向 → 极小。
        筛选：|C_ext| > noise_floor×prom_factor、相邻异类极值落差 > noise_floor×3。"""
        config = config or AnalyzerConfig()
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        s = _local_slopes(r, C)
        n = len(C)
        ext_r, ext_v = [], []
        for i in range(1, n - 1):
            if s[i - 1] > 0 and s[i] < 0:      # 局部极大
                ext_r.append(r[i])
                ext_v.append(C[i])
            elif s[i - 1] < 0 and s[i] > 0:    # 局部极小
                ext_r.append(r[i])
                ext_v.append(C[i])
        ext_r = np.array(ext_r)
        ext_v = np.array(ext_v)
        if len(ext_v) == 0:
            return (ext_r, ext_v)
        # 幅度筛选（噪声标尺）
        keep = np.abs(ext_v) > noise_floor * config.prom_factor
        ext_r, ext_v = ext_r[keep], ext_v[keep]
        # 相邻异类极值落差筛选
        if len(ext_v) >= 2:
            keep2 = np.ones(len(ext_v), dtype=bool)
            for i in range(len(ext_v) - 1):
                if abs(ext_v[i] - ext_v[i + 1]) <= noise_floor * config.prom_factor:
                    keep2[i] = keep2[i + 1] = False
            ext_r, ext_v = ext_r[keep2], ext_v[keep2]
        return (ext_r, ext_v)

    # ----------------------------------------------------------------------
    # 4. 振荡诊断（过零 / 极值 / Q / 稳定性 / 距离族分解）
    # ----------------------------------------------------------------------
    @staticmethod
    def estimate_q_lomb_scargle(r: np.ndarray, y_det: np.ndarray,
                                config: AnalyzerConfig = None) -> tuple:
        """Lomb-Scargle 粗扫。
        Q 范围为启发式区间 [π/span, π/min_Δr]——irregular sampling 下不存在严格
        Nyquist 频率，min_Δr 仅作为搜索上限（存在 alias 风险，由过零间距交叉
        校验与子窗口稳定性检验兜底）。返回 (Q_peak, freqs, power)"""
        config = config or AnalyzerConfig()
        r = np.asarray(r, dtype=float)
        y_det = np.asarray(y_det, dtype=float)
        if len(r) < 4:
            return (np.nan, np.array([]), np.array([]))
        dr = np.diff(r)
        dr_pos = dr[dr > 0]
        if len(dr_pos) == 0:
            return (np.nan, np.array([]), np.array([]))
        span = float(np.max(r) - np.min(r))
        q_min = np.pi / span if span > 0 else np.nan
        q_max = np.pi / float(np.min(dr_pos))
        if not np.isfinite(q_min) or q_max <= q_min:
            return (np.nan, np.array([]), np.array([]))
        freqs = np.linspace(q_min, q_max, config.q_grid_points)
        power = signal.lombscargle(r, y_det - np.mean(y_det), freqs, precenter=True)
        q_peak = float(freqs[int(np.argmax(power))])
        return (q_peak, freqs, power)

    # ----------------------------------------------------------------------
    @staticmethod
    def refine_q_cosine(r: np.ndarray, y_det: np.ndarray, q_guess: float,
                        config: AnalyzerConfig = None) -> tuple:
        """固定 Q 线性化精扫：Q ∈ [0.8,1.2]×q_guess 细网格，
        每次固定 Q 解 y ≈ a·cos(Qr)+b·sin(Qr) 闭式 lstsq，取 R² 最大的 Q。
        规避 4 参数非线性拟合的周期简并（Q→2Q 局部极小）。
        返回 (Q, phi, amp, r2_cos)"""
        config = config or AnalyzerConfig()
        r = np.asarray(r, dtype=float)
        y_det = np.asarray(y_det, dtype=float)
        if not np.isfinite(q_guess) or len(r) < 5:
            return (np.nan, np.nan, np.nan, np.nan)
        qs = np.linspace(0.8 * q_guess, 1.2 * q_guess, config.q_refine_points)
        best_q, best_r2 = np.nan, -np.inf
        best_ab = None
        for q in qs:
            X = np.column_stack([np.cos(q * r), np.sin(q * r)])
            coef, _, _, _ = np.linalg.lstsq(X, y_det, rcond=None)
            yhat = X @ coef
            r2 = _r2_in_space(y_det, yhat)
            if r2 > best_r2:
                best_r2, best_q, best_ab = r2, float(q), coef
        if best_ab is None:
            return (np.nan, np.nan, np.nan, np.nan)
        a, b = float(best_ab[0]), float(best_ab[1])
        phi = float(np.arctan2(-b, a))
        amp = float(np.sqrt(a * a + b * b))
        return (best_q, phi, amp, best_r2)

    # ----------------------------------------------------------------------
    @staticmethod
    def _subwindow_q_phi(r: np.ndarray, y_det: np.ndarray,
                         config: AnalyzerConfig) -> tuple:
        """三等分重叠子窗口（重叠约 30%）分别精化 Q、φ → 稳定性序列"""
        n = len(r)
        if n < 8:
            return ([], [])
        size = max(int(0.4 * n), 6)
        starts = [0, int(0.3 * n), int(0.6 * n)]
        qs, phis = [], []
        for s in starts:
            seg = slice(s, min(s + size, n))
            rr, yy = r[seg], y_det[seg]
            if len(rr) < 5:
                continue
            q0, _, _ = SuperCorrDecayAnalyzer.estimate_q_lomb_scargle(rr, yy, config)
            q, phi, _, _ = SuperCorrDecayAnalyzer.refine_q_cosine(rr, yy, q0, config)
            if np.isfinite(q):
                qs.append(q)
                phis.append(phi)
        return (qs, phis)

    # ----------------------------------------------------------------------
    @staticmethod
    def _oscillation_evidence(checks: dict, n: int, n_cross: int,
                              config: AnalyzerConfig) -> str:
        """证据聚合：核心检查(cosine R²/Q 稳定/包络 R²)各 2 分，
        支持检查(过零数/极值数/过零间距/极值间距/φ 稳定)各 1 分，满分 11。
        单项不通过只扣分，不直接否决（无任何单项硬门槛）。"""
        core = ('r2_cos', 'q_stable', 'r2_env_log')
        score = sum(2 if checks.get(k, False) else 0 for k in core)
        support = ('n_cross', 'n_ext', 'cross_spacing', 'ext_spacing', 'phi_stable')
        score += sum(1 if checks.get(k, False) else 0 for k in support)
        core_pass = sum(1 for k in core if checks.get(k, False))
        if score >= 10 and n >= config.n_min_osc:
            return 'strong'
        if score >= 7 and core_pass >= 2:
            return 'moderate'
        if score >= 4 and core_pass >= 1:
            return 'weak'
        if score >= 3:
            return 'possible'
        if n_cross == 0:
            return 'none'
        return 'possible'

    # ----------------------------------------------------------------------
    @staticmethod
    def _quick_oscillation(r: np.ndarray, C: np.ndarray, env_fit: FitResult,
                           config: AnalyzerConfig) -> dict:
        """在给定窗口上做一次完整振荡诊断（scan_r_min 复用）。
        返回 {osc_flag, evidence, Q, phi, r2_cos, n_cross, n_ext}"""
        n = len(r)
        res = {'osc_flag': False, 'evidence': 'none', 'Q': np.nan, 'phi': np.nan,
               'r2_cos': np.nan, 'n_cross': 0, 'n_ext': 0}
        if n < 5 or env_fit is None:
            return res
        cross_pos, n_cross = SuperCorrDecayAnalyzer.detect_zero_crossings(r, C)
        E = SuperCorrDecayAnalyzer._envelope_value(env_fit, r)
        y_det = C / E
        q0, _, _ = SuperCorrDecayAnalyzer.estimate_q_lomb_scargle(r, y_det, config)
        Q, phi, amp, r2_cos = SuperCorrDecayAnalyzer.refine_q_cosine(r, y_det, q0, config)
        qs, phis = SuperCorrDecayAnalyzer._subwindow_q_phi(r, y_det, config)
        q_rel_mad = _rel_mad(np.array(qs)) if qs else np.nan
        phi_mad = _mad_abs(np.array(phis)) if phis else np.nan
        # 过零/极值间距一致性
        d_cross = np.diff(cross_pos) if len(cross_pos) > 1 else np.array([])
        cross_reg = _rel_mad(d_cross) if len(d_cross) > 1 else np.nan
        noise = 0.0
        ext_r, ext_v = SuperCorrDecayAnalyzer.detect_extrema(r, C, noise, config)
        d_ext = np.diff(ext_r) if len(ext_r) > 1 else np.array([])
        ext_reg = _rel_mad(d_ext) if len(d_ext) > 1 else np.nan
        # 过零间距与 LS Q 交叉校验：|π/Q − Δ̄|/Δ̄ > 30% → 记入过零间距检查失败
        cross_ok = False
        if len(cross_pos) > 1 and np.isfinite(Q):
            delta_bar = float(np.median(d_cross))
            if delta_bar > 0:
                mismatch = abs(np.pi / Q - delta_bar) / delta_bar
                cross_ok = (cross_reg is not None and not np.isnan(cross_reg)
                            and cross_reg <= config.spacing_regularity_max
                            and mismatch <= 0.3)
        checks = {
            'n_cross': n_cross >= config.n_cross_min,
            'n_ext': len(ext_r) >= config.n_extrema_min,
            'cross_spacing': cross_ok,
            'ext_spacing': (ext_reg is not None and not np.isnan(ext_reg)
                            and ext_reg <= config.spacing_regularity_max),
            'r2_cos': np.isfinite(r2_cos) and r2_cos >= config.r2_cos_min,
            'r2_env_log': np.isfinite(env_fit.r2) and env_fit.r2 >= config.r2_env_log_min,
            'q_stable': np.isfinite(q_rel_mad) and q_rel_mad <= config.q_stab_max,
            'phi_stable': np.isfinite(phi_mad) and phi_mad <= config.phi_stab_max,
        }
        evidence = SuperCorrDecayAnalyzer._oscillation_evidence(checks, n, n_cross, config)
        res.update({
            'osc_flag': evidence in ('strong', 'moderate', 'weak'),
            'evidence': evidence,
            'Q': Q, 'phi': phi, 'r2_cos': r2_cos,
            'n_cross': n_cross, 'n_ext': len(ext_r),
            'cross_spacing_regularity': cross_reg,
            'ext_spacing_regularity': ext_reg,
            'q_rel_mad': q_rel_mad, 'phi_mad': phi_mad,
            'checks': checks,
        })
        return res

    # ----------------------------------------------------------------------
    @staticmethod
    def family_separated_diagnostic(prep: PreparedData, env_fit: FitResult,
                                    config: AnalyzerConfig = None) -> FamilyDiagnostics:
        """距离族分解诊断：对两个几何族分别做符号纯度、过零、包络、去包络 cosine。
        族内点数 < n_min_fit/2 → 该族标记 insufficient。"""
        config = config or AnalyzerConfig()
        diag = FamilyDiagnostics()
        fams = prep.group_family
        for f in sorted(set(fams)):
            mask = fams == f
            r_f, C_f = prep.r[mask], prep.C[mask]
            diag.family_r[str(f)] = r_f
            diag.family_C[str(f)] = C_f
            if len(r_f) < config.n_min_fit // 2:
                diag.per_family[str(f)] = {'insufficient': True, 'n': len(r_f)}
                diag.consistency[str(f)] = False
                diag.insufficient = True
                continue
            entry = {'n': len(r_f)}
            pos = C_f > 0
            entry['sign_purity'] = float(max(pos.mean(), 1.0 - pos.mean()))
            entry['n_cross'] = SuperCorrDecayAnalyzer.detect_zero_crossings(r_f, C_f)[1]
            # 族内包络（|C| log 空间 power/exp 选 AICc 优者）
            absC = np.abs(C_f)
            f_p = SuperCorrDecayAnalyzer.fit_envelope_logspace(r_f, absC, 'power', config)
            f_e = SuperCorrDecayAnalyzer.fit_envelope_logspace(r_f, absC, 'exp', config)
            pick = f_p if (np.isfinite(f_p.aicc) and
                           (not np.isfinite(f_e.aicc) or f_p.aicc <= f_e.aicc)) else f_e
            entry['envelope_form'] = pick.model.replace('envelope_', '')
            entry['envelope_param'] = pick.params.get('K', pick.params.get('xi', np.nan))
            entry['r2_env_log'] = pick.r2
            # 族内去包络 cosine
            E_f = SuperCorrDecayAnalyzer._envelope_value(pick, r_f)
            y_det = C_f / E_f
            q0, _, _ = SuperCorrDecayAnalyzer.estimate_q_lomb_scargle(r_f, y_det, config)
            Q_f, phi_f, _, r2_cos = SuperCorrDecayAnalyzer.refine_q_cosine(
                r_f, y_det, q0, config)
            entry['Q'] = Q_f
            entry['phi'] = phi_f
            entry['r2_cos'] = r2_cos
            # 核心检查：包络 R² 与 cosine R²
            core_ok = (np.isfinite(pick.r2) and pick.r2 >= config.r2_env_log_min and
                       np.isfinite(r2_cos) and r2_cos >= config.r2_cos_min)
            diag.per_family[str(f)] = entry
            diag.consistency[str(f)] = bool(core_ok)
        # 结构赝振荡初步判定线索（最终判定在 analyze_oscillation 结合混合证据）
        if set(diag.family_r.keys()) != {'0', '1'}:
            diag.insufficient = True
            diag.per_family.setdefault('note', '仅一个距离族有数据，族分解诊断降级')
        return diag

    # ----------------------------------------------------------------------
    @staticmethod
    def analyze_oscillation(r: np.ndarray, C: np.ndarray, envelope_fit: FitResult,
                            family_diag: FamilyDiagnostics = None,
                            config: AnalyzerConfig = None) -> OscillationInfo:
        """振荡诊断（证据评分制）：8 项 diagnostic checks 逐项记录，
        通过证据聚合得到 evidence level；结合族分解做结构赝振荡判定。"""
        config = config or AnalyzerConfig()
        info = OscillationInfo(envelope_form=envelope_fit.model.replace('envelope_', ''),
                               r2_env_log=envelope_fit.r2)
        n = len(r)
        noise = 0.0
        cross_pos, n_cross = SuperCorrDecayAnalyzer.detect_zero_crossings(r, C)
        info.n_cross = n_cross
        info.cross_positions = cross_pos
        ext_r, ext_v = SuperCorrDecayAnalyzer.detect_extrema(r, C, noise, config)
        info.n_ext = len(ext_r)
        info.ext_positions = ext_r
        d_cross = np.diff(cross_pos) if len(cross_pos) > 1 else np.array([])
        d_ext = np.diff(ext_r) if len(ext_r) > 1 else np.array([])
        info.cross_spacing_regularity = _rel_mad(d_cross) if len(d_cross) > 1 else np.nan
        info.ext_spacing_regularity = _rel_mad(d_ext) if len(d_ext) > 1 else np.nan

        # 去包络 + Q 估计
        E = SuperCorrDecayAnalyzer._envelope_value(envelope_fit, r)
        y_det = C / E
        q0, freqs, power = SuperCorrDecayAnalyzer.estimate_q_lomb_scargle(r, y_det, config)
        Q, phi, amp, r2_cos = SuperCorrDecayAnalyzer.refine_q_cosine(r, y_det, q0, config)
        info.Q, info.phi, info.amplitude, info.r2_cos = Q, phi, amp, r2_cos
        qs, phis = SuperCorrDecayAnalyzer._subwindow_q_phi(r, y_det, config)
        info.subwindow_Qs = np.array(qs)
        info.Q_rel_mad = _rel_mad(np.array(qs)) if qs else np.nan
        info.phi_mad = _mad_circular(np.array(phis)) if phis else np.nan
        # Q 搜索区间分辨率检查（启发式区间，见 estimate_q_lomb_scargle 说明）
        if len(r) > 1:
            span = float(np.max(r) - np.min(r))
            dr_pos = np.diff(r)[np.diff(r) > 0]
            if span > 0 and len(dr_pos) > 0 and (np.pi / float(np.min(dr_pos))) / (np.pi / span) < 2.0:
                info.structural_warning = ("Q 搜索区间过窄（窗口内不足 2 个可分辨周期），"
                                           "Q 分辨率受限，证据从严")

        # 过零间距与 LS Q 交叉校验
        cross_ok = False
        if len(d_cross) > 1 and np.isfinite(Q):
            delta_bar = float(np.median(d_cross))
            if delta_bar > 0:
                mismatch = abs(np.pi / Q - delta_bar) / delta_bar
                cross_ok = (not np.isnan(info.cross_spacing_regularity)
                            and info.cross_spacing_regularity <= config.spacing_regularity_max
                            and mismatch <= 0.3)
        info.criterion_checks = {
            'n_cross': n_cross >= config.n_cross_min,
            'n_ext': len(ext_r) >= config.n_extrema_min,
            'cross_spacing': cross_ok,
            'ext_spacing': (not np.isnan(info.ext_spacing_regularity)
                            and info.ext_spacing_regularity <= config.spacing_regularity_max),
            'r2_cos': np.isfinite(r2_cos) and r2_cos >= config.r2_cos_min,
            'r2_env_log': np.isfinite(info.r2_env_log) and info.r2_env_log >= config.r2_env_log_min,
            'q_stable': np.isfinite(info.Q_rel_mad) and info.Q_rel_mad <= config.q_stab_max,
            'phi_stable': np.isfinite(info.phi_mad) and info.phi_mad <= config.phi_stab_max,
        }
        info.evidence = SuperCorrDecayAnalyzer._oscillation_evidence(
            info.criterion_checks, n, n_cross, config)
        # 远距相位紊乱防护：φ 子窗口最大圆周距离显著（>2×phi_stab_max）
        # 说明振荡模型在远端失效。用 max 而非 MAD：3 个子窗口中 2 个一致时
        # 中位数会掩盖单个漂移窗口。
        phi_max = _max_circular(np.array(phis)) if phis else np.nan
        if (np.isfinite(phi_max) and phi_max > 2.0 * config.phi_stab_max
                and info.evidence in ('strong', 'moderate')):
            info.evidence = 'weak'
            info.structural_warning = (info.structural_warning + "；"
                                       f"φ 子窗口最大漂移 {phi_max:.2f} rad，"
                                       "振荡模型在远端不稳定，证据降为 weak")
        info.detected = info.evidence in ('strong', 'moderate', 'weak')

        # 结构赝振荡判定（混合 evidence ≥ moderate 且族分解不稳定 → artifact）
        if family_diag is not None:
            if family_diag.insufficient:
                info.structural_warning = (info.structural_warning + "；"
                                           "距离族分解数据不足（仅一族或无足够点），结构赝振荡无法评估")
            else:
                # 符号-距离族耦合列联表检查：某族符号纯度 > 80% → warning
                for f, entry in family_diag.per_family.items():
                    if entry.get('sign_purity', 0.0) > config.family_sign_purity_warn:
                        info.structural_warning = (info.structural_warning +
                                                   f"；距离族 {f} 符号纯度 {entry['sign_purity']:.2f}，"
                                                   "符号模式与该族强耦合，可能是键方向/子格结构赝振荡")
                # 混合稳定 + 分族不稳定 → artifact 显式标记（专用字段，非仅 warning）
                if info.evidence in ('strong', 'moderate'):
                    fam_keys = list(family_diag.consistency.keys())
                    fam_ok = [family_diag.consistency.get(k, False) for k in fam_keys]
                    Qs = [family_diag.per_family.get(k, {}).get('Q', np.nan) for k in fam_keys]
                    phis = [family_diag.per_family.get(k, {}).get('phi', np.nan) for k in fam_keys]
                    q_mismatch = False
                    phi_mismatch = False
                    if len([q for q in Qs if np.isfinite(q)]) == 2:
                        q_mismatch = abs(Qs[0] - Qs[1]) / max(abs(Qs[0]), abs(Qs[1])) > 0.3
                        phi_mismatch = abs(np.angle(np.exp(1j * (phis[0] - phis[1])))) > 0.5
                    if (not all(fam_ok)) or q_mismatch or phi_mismatch:
                        info.structural_artifact_suspected = True
                        info.evidence = 'weak'   # artifact 时 evidence 上限 weak
                        info.structural_warning = (info.structural_warning + "；"
                                                   "混合数据振荡证据与距离族分解诊断冲突："
                                                   "possible structural/geometric artifact")
        info.structural_warning = info.structural_warning.strip('；')
        return info

    # ----------------------------------------------------------------------
    # 5. 候选模型拟合（传统比较层）
    # ----------------------------------------------------------------------
    @staticmethod
    def bootstrap_errors(f, x: np.ndarray, y: np.ndarray, p_opt: np.ndarray,
                         config: AnalyzerConfig = None) -> np.ndarray:
        """残差 bootstrap 参数不确定度。仅作 pcov 失效时的兜底，
        不作为主要物理证据，不参与模型选择与分类决策。"""
        config = config or AnalyzerConfig()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(f(x, *p_opt), dtype=float)
        resid = y - yhat
        rng = np.random.default_rng(42)
        params = []
        for _ in range(config.bootstrap_n):
            ys = yhat + rng.choice(resid, size=len(y), replace=True)
            try:
                popt, _ = optimize.curve_fit(f, x, ys, p0=p_opt, maxfev=20000)
                params.append(popt)
            except Exception:
                continue
        if not params:
            return np.full(len(p_opt), np.nan)
        return np.std(np.array(params), axis=0)

    # ----------------------------------------------------------------------
    @staticmethod
    def _fit_one(model: str, f, r_win: np.ndarray, y_win: np.ndarray,
                 p0: list, bounds: tuple, k: int, window: tuple,
                 param_names: list, fit_space: str,
                 config: AnalyzerConfig) -> FitResult:
        """单个候选模型拟合的统一封装（curve_fit + pcov/bootstrapping 两级不确定度）。
        fit_space 由调用方显式传入（'linear_C' 拟 signed C；'linear_absC' 拟 |C|）。"""
        r_win = np.asarray(r_win, dtype=float)
        y_win = np.asarray(y_win, dtype=float)
        n = len(y_win)
        res = FitResult(model=model, fit_space=fit_space, r2_space='linear',
                        n=n, k=k, window=tuple(window))
        if n < k + 3:
            res.valid = False
            res.message = f"点数 n={n} 不足以约束 {k} 参数模型"
            return res
        if k == 4 and n < config.n_min_osc_fit:
            res.message += f"；n={n} < n_min_osc_fit({config.n_min_osc_fit})，4 参数模型自由度紧张"
        lo, hi = bounds
        p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)
        try:
            popt, pcov = optimize.curve_fit(f, r_win, y_win, p0=p0,
                                            bounds=(lo, hi), max_nfev=20000)
        except Exception as e:
            res.valid = False
            res.message += f"；拟合失败: {type(e).__name__}"
            return res
        yhat = np.asarray(f(r_win, *popt), dtype=float)
        rss = float(np.sum((y_win - yhat) ** 2))
        r2 = _r2_in_space(y_win, yhat)
        aic, aicc, bic = _aicc_bic_core(rss, n, k)
        res.rss, res.rmse, res.r2 = rss, float(np.sqrt(rss / n)), r2
        res.aic, res.aicc, res.bic = aic, aicc, bic
        # 参数不确定度：主来源 pcov；异常 → bootstrap 兜底
        err = np.full(k, np.nan)
        err_src = 'pcov'
        if pcov is not None and np.all(np.isfinite(pcov)):
            diag = np.diag(pcov)
            try:
                cond = float(np.linalg.cond(pcov))
            except Exception:
                cond = np.inf
            if np.all(diag > 0) and cond < config.pcov_max_cond:
                err = np.sqrt(diag)
            else:
                err = SuperCorrDecayAnalyzer.bootstrap_errors(f, r_win, y_win, popt, config)
                err_src = 'bootstrap'
        else:
            err = SuperCorrDecayAnalyzer.bootstrap_errors(f, r_win, y_win, popt, config)
            err_src = 'bootstrap'
        res.params = {name: float(v) for name, v in zip(param_names, popt)}
        res.param_errors = {name: float(e) for name, e in zip(param_names, err)}
        res.error_source = err_src
        return res

    # ----------------------------------------------------------------------
    @staticmethod
    def fit_candidate_models(r: np.ndarray, C: np.ndarray, envelope_fit: FitResult,
                             osc_info: OscillationInfo, config: AnalyzerConfig = None) -> dict:
        """5 个候选模型拟合。拟合空间策略：
        无过零 → 5 模型全部拟 signed C(r)；有过零 → 非振荡 3 模型拟 |C|、
        振荡 2 模型拟 signed C。AICc/BIC 只在同 fit_space 且同窗口组内比较。"""
        config = config or AnalyzerConfig()
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        span = float(np.max(r) - np.min(r))
        max_abs = float(np.max(np.abs(C)))
        y_abs = np.abs(C)
        n_cross = SuperCorrDecayAnalyzer.detect_zero_crossings(r, C)[1]
        window = (float(np.min(r)), float(np.max(r)))
        fits = {}

        # ---- 非振荡组：无过零拟 signed，有过零拟 |C| ----
        y_non = C if n_cross == 0 else y_abs
        fit_space_non = 'linear_C' if n_cross == 0 else 'linear_absC'
        A_lo, A_hi = 0.1 * max_abs, 10.0 * max_abs
        if A_lo <= 0 or A_hi <= A_lo:
            A_lo, A_hi = 1e-300, max(1.0, 10 * max_abs)

        # constant
        fits['constant'] = SuperCorrDecayAnalyzer._fit_one(
            'constant', lambda rr, A: np.full_like(rr, A, dtype=float),
            r, y_non, [float(np.mean(y_non))], ([-A_hi], [A_hi]), 1, window,
            ['A'], fit_space_non, config)

        # power-law（log-log lstsq 初值）
        pos = y_non > 0
        if pos.sum() >= 2:
            x_ll = np.log(r[pos])
            coef_ll, _, _ = _logspace_lstsq(x_ll, np.log(y_non[pos]))
            A0_p = float(np.exp(coef_ll[0]))
            K0 = float(-coef_ll[1])
        else:
            A0_p, K0 = max_abs, 1.0
        fits['power_law'] = SuperCorrDecayAnalyzer._fit_one(
            'power_law', lambda rr, A, K: A * rr ** (-K),
            r, y_non, [A0_p, np.clip(K0, 0.1, 8.0)],
            ([A_lo, 0.0], [A_hi, 8.0]), 2, window,
            ['A', 'K'], fit_space_non, config)

        # exponential（log-lin lstsq 初值）
        if pos.sum() >= 2:
            coef_le, _, _ = _logspace_lstsq(r[pos], np.log(y_non[pos]))
            A0_e = float(np.exp(coef_le[0]))
            xi0 = float(-1.0 / coef_le[1]) if abs(coef_le[1]) > 1e-300 else 10 * span
        else:
            A0_e, xi0 = max_abs, span
        fits['exponential'] = SuperCorrDecayAnalyzer._fit_one(
            'exponential', lambda rr, A, xi: A * np.exp(-rr / xi),
            r, y_non, [A0_e, np.clip(xi0, 0.1, 10 * span)],
            ([A_lo, 0.1], [A_hi, 10.0 * span]), 2, window,
            ['A', 'xi'], fit_space_non, config)

        # ---- 振荡组：拟 signed C(r) ----
        env = envelope_fit if envelope_fit is not None else fits['power_law']
        A0 = float(env.params.get('A', max_abs))
        Q0 = osc_info.Q if (osc_info is not None and np.isfinite(osc_info.Q)) else 1.0
        phi0 = osc_info.phi if (osc_info is not None and np.isfinite(osc_info.phi)) else 0.0
        Q_lo, Q_hi = 0.8 * Q0, 1.2 * Q0

        K_env = float(env.params.get('K', 1.0))
        fits['osc_power'] = SuperCorrDecayAnalyzer._fit_one(
            'osc_power', lambda rr, A, K, Q, phi: A * rr ** (-K) * np.cos(Q * rr + phi),
            r, C, [A0, np.clip(K_env, 0.0, 8.0), Q0, phi0],
            ([A_lo, 0.0, Q_lo, -np.pi], [A_hi, 8.0, Q_hi, np.pi]), 4, window,
            ['A', 'K', 'Q', 'phi'], 'linear_C', config)

        xi_env = float(env.params.get('xi', span))
        fits['osc_exp'] = SuperCorrDecayAnalyzer._fit_one(
            'osc_exp', lambda rr, A, xi, Q, phi: A * np.exp(-rr / xi) * np.cos(Q * rr + phi),
            r, C, [A0, np.clip(xi_env, 0.1, 10 * span), Q0, phi0],
            ([A_lo, 0.1, Q_lo, -np.pi], [A_hi, 10.0 * span, Q_hi, np.pi]), 4, window,
            ['A', 'xi', 'Q', 'phi'], 'linear_C', config)
        return fits

    # ----------------------------------------------------------------------
    # 6. 模型比较（手写信息准则，同空间同窗口组内比较）
    # ----------------------------------------------------------------------
    @staticmethod
    def compute_information_criteria(rss: float, n: int, k: int) -> tuple:
        """AIC = n·ln(RSS/n)+2k；AICc = AIC+2k(k+1)/(n−k−1)（n≤k+1 → NaN）；
        BIC = n·ln(RSS/n)+k·ln(n)"""
        return _aicc_bic_core(rss, n, k)

    # ----------------------------------------------------------------------
    @staticmethod
    def compare_models(fit_dict: dict, config: AnalyzerConfig = None) -> dict:
        """同 (fit_space, window) 组内排序。跨 fit_space / 跨窗口的 AIC/BIC
        直接比较一律禁止——不同组不进同一排序表。返回组内 ΔAICc/ΔBIC 证据表。"""
        config = config or AnalyzerConfig()
        groups = {}
        for name, fit in fit_dict.items():
            if fit is None or not fit.valid:
                continue
            key = (fit.fit_space, tuple(fit.window))
            groups.setdefault(key, []).append((name, fit))
        result = {'groups': {}, 'best_per_group': {}}
        for key, items in groups.items():
            usable = [(n, f) for n, f in items if np.isfinite(f.aicc)]
            if not usable:
                continue
            usable.sort(key=lambda t: t[1].aicc)
            best_name, best_fit = usable[0]
            rows = []
            for name, fit in usable:
                daicc = fit.aicc - best_fit.aicc
                dbic = fit.bic - best_fit.bic
                # ΔAICc 证据：0-2 弱 / 2-4 正 / 4-7 强 / >7 极强；ΔBIC 用 Kass&Raftery 2/6/10
                if dbic <= 2:
                    bic_ev = '弱（无显著差异）'
                elif dbic <= 6:
                    bic_ev = '正证据'
                elif dbic <= 10:
                    bic_ev = '强证据'
                else:
                    bic_ev = '极强证据'
                rows.append({'model': name, 'aicc': fit.aicc, 'bic': fit.bic,
                             'delta_aicc': daicc, 'delta_bic': dbic,
                             'bic_evidence': bic_ev,
                             'n': fit.n, 'k': fit.k, 'r2': fit.r2})
            result['groups'][f'{key[0]} @ {key[1]}'] = {
                'best_model': best_name, 'rows': rows}
            result['best_per_group'][key] = best_name
        return result

    # ----------------------------------------------------------------------
    # 7. r_min 扫描与稳定性
    # ----------------------------------------------------------------------
    @staticmethod
    def scan_r_min(r: np.ndarray, C: np.ndarray,
                   config: AnalyzerConfig = None) -> StabilityReport:
        """r_min 全网格自动扫描。每个有效候选：包络 power/exp 拟合 +
        该窗口内重新执行振荡诊断（evidence/Q/φ/R2_cos/osc_flag），
        完整保存全部候选行（含跳过原因），推荐值只从扫描表自动产生。"""
        config = config or AnalyzerConfig()
        rep = StabilityReport()
        M = len(r)
        if M < config.n_min_fit:
            rep.messages.append(f"总距离数 M={M} < n_min_fit({config.n_min_fit})，无法扫描")
            return rep

        n_short = min(10, M - config.n_min_fit)
        idx_short = list(range(n_short))
        if M - config.n_min_fit > 10:
            idx_spread = np.unique(np.linspace(10, M - config.n_min_fit, 10)
                                   .round().astype(int)).tolist()
        else:
            idx_spread = []
        candidates = sorted(set(idx_short) | set(idx_spread))

        rows = []
        for idx in candidates:
            r_min = float(r[idx])
            r_win = r[idx:]
            C_win = C[idx:]
            n = len(r_win)
            if n < config.n_min_fit:
                rows.append({'r_min': r_min, 'n_pts': n, 'skipped': True,
                             'reason': f'n_pts={n} < n_min_fit({config.n_min_fit})'})
                continue
            # 包络拟合（|C| log 空间）
            absC = np.abs(C_win)
            f_p = SuperCorrDecayAnalyzer.fit_envelope_logspace(r_win, absC, 'power', config)
            f_e = SuperCorrDecayAnalyzer.fit_envelope_logspace(r_win, absC, 'exp', config)
            if np.isfinite(f_p.aicc) and (not np.isfinite(f_e.aicc) or f_p.aicc <= f_e.aicc):
                env = f_p
            else:
                env = f_e
            param_key = 'K' if 'K' in env.params else 'xi'
            # 该候选窗口内重新执行振荡诊断
            osc = SuperCorrDecayAnalyzer._quick_oscillation(r_win, C_win, env, config)
            rows.append({
                'r_min': r_min, 'n_pts': n, 'skipped': False,
                'envelope_model': env.model.replace('envelope_', ''),
                'param': float(env.params.get(param_key, np.nan)),
                'param_err': float(env.param_errors.get(param_key, np.nan)),
                'r2_env_log': env.r2, 'aicc': env.aicc, 'bic': env.bic,
                'osc_flag': osc['osc_flag'], 'osc_evidence': osc['evidence'],
                'Q': osc['Q'], 'phi': osc['phi'], 'r2_cos': osc['r2_cos'],
            })
        rep.scan_table = rows

        valid_rows = [row for row in rows if not row['skipped']]
        if len(valid_rows) < 2:
            rep.messages.append("有效候选不足 2 个，无法评估稳定性")
            return rep

        # 全局多数包络模型 M* 与参数序列
        from collections import Counter
        model_counts = Counter(row['envelope_model'] for row in valid_rows)
        best_model, _ = model_counts.most_common(1)[0]
        params = np.array([row['param'] for row in valid_rows
                           if row['envelope_model'] == best_model])
        rep.model_consistency = float(len(params) / len(valid_rows))
        if len(params) < 2:
            rep.messages.append("多数模型样本不足，稳定性降级")
        med = float(np.median(params))
        mad = float(np.median(np.abs(params - med)))
        rel_mad = 1.4826 * mad / abs(med) if abs(med) > 1e-300 else np.nan
        if best_model == 'power':
            rep.rel_mad_K = rel_mad
        else:
            rep.rel_mad_xi = rel_mad

        # Q 稳定性（跨候选）
        qs = np.array([row['Q'] for row in valid_rows if np.isfinite(row['Q'])])
        rep.rel_mad_Q = _rel_mad(qs) if len(qs) > 1 else np.nan

        # Spearman：参数随 r_min 单调漂移（只用多数模型行，K/ξ 不混算）
        r_mins = np.array([row['r_min'] for row in valid_rows
                           if row['envelope_model'] == best_model])
        rep.spearman_rho = _spearman(r_mins, params)

        # verdict
        if np.isfinite(rel_mad):
            if rel_mad <= config.k_stab_max:
                rep.verdict = 'stable'
            elif rel_mad <= config.k_stab_moderate:
                rep.verdict = 'moderate'
            else:
                rep.verdict = 'unstable'
        else:
            rep.verdict = 'unstable'

        # 推荐 r_min（自动产生；语义 = 当前稳定性准则下的推荐分析窗口起点）
        in_band = [row for row in valid_rows
                   if row['envelope_model'] == best_model
                   and abs(row['param'] - med) <= 1.5 * mad]
        if in_band:
            rep.recommended_r_min = float(min(row['r_min'] for row in in_band))
        else:
            rep.recommended_r_min = float(min(valid_rows,
                                              key=lambda row: row['aicc'])['r_min'])
            rep.messages.append("无候选满足参数带与模型一致性要求，推荐值取 AICc 最优候选，证据降档")
        if np.isfinite(rep.spearman_rho) and abs(rep.spearman_rho) > 0.8:
            rep.messages.append(
                f"参数随 r_min 单调漂移（Spearman ρ={rep.spearman_rho:.2f}），"
                "可能存在未被模型描述的近程贡献/边界效应，证据降档")
        return rep

    # ----------------------------------------------------------------------
    # 8. 综合判定（分类决策树）
    # ----------------------------------------------------------------------
    @staticmethod
    def _envelope_hump(r: np.ndarray, C_abs: np.ndarray, noise_floor: float) -> bool:
        """包络斜率变号检测：ln|C| vs r 的局域斜率由负转正 ≥1 次（|C| 先降后升），
        仅在 |C| > noise_floor×3 的可靠区检测。"""
        r = np.asarray(r, dtype=float)
        C_abs = np.asarray(C_abs, dtype=float)
        ok = (C_abs > noise_floor * 3.0) & (C_abs > 0)
        if ok.sum() < 4:
            return False
        s = _local_slopes(r[ok], np.log(C_abs[ok]))
        return bool(np.any((s[:-1] < 0) & (s[1:] > 0)))

    # ----------------------------------------------------------------------
    @staticmethod
    def _form_switch(stability: StabilityReport) -> bool:
        """包络形式随 r_min 窗口在 power↔exp 间切换（各自有稳定支持）→ 冲突线索"""
        rows = [row for row in stability.scan_table if not row.get('skipped', True)]
        forms = [row['envelope_model'] for row in rows]
        n_power = forms.count('power')
        n_exp = forms.count('exp')
        return n_power >= 3 and n_exp >= 3

    # ----------------------------------------------------------------------
    @staticmethod
    def _dual_plateau(stability: StabilityReport, config: AnalyzerConfig) -> bool:
        """r_min 扫描出现 ≥2 个互不相交参数平台（各平台内部 rel_mad ≤ k_stab_max、
        合并后 rel_mad > k_stab_moderate）→ 分析路径冲突线索之一"""
        rows = [row for row in stability.scan_table if not row.get('skipped', True)]
        if len(rows) < 6:
            return False
        rows = sorted(rows, key=lambda row: row['r_min'])
        params = np.array([row['param'] for row in rows])
        overall = _rel_mad(params)
        if not np.isfinite(overall) or overall <= config.k_stab_moderate:
            return False
        # 按最大跳变贪心切分为两段，检查各段内部稳定性
        jumps = np.abs(np.diff(params))
        split = int(np.argmax(jumps)) + 1
        segs = [params[:split], params[split:]]
        if min(len(seg) for seg in segs) < 3:
            return False
        return all(_rel_mad(seg) <= config.k_stab_max for seg in segs)

    # ----------------------------------------------------------------------
    @staticmethod
    def classify(components: dict, config: AnalyzerConfig = None) -> dict:
        """K 节决策树。返回 {'classification', 'evidence_level', 'nonmonotonic_reason'}。
        最终物理 classification 不由 AIC/BIC 单独决定。"""
        config = config or AnalyzerConfig()
        prep = components['prep']
        osc = components['osc']
        stability = components['stability']
        fits = components['candidate_fits']
        envelope_fit = components['envelope_fit']
        hump = components['hump']
        dual = components['dual_plateau']
        warnings = components.get('warnings', [])

        # ---- 门槛 1：数据充分性 ----
        if prep.n_distance < config.n_min_abs:
            warnings.append(f"距离数 {prep.n_distance} < n_min_abs({config.n_min_abs})，数据不足")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}
        mask = _window_mask(prep.r, 0.0, prep.r_max_used)
        c_win = prep.C_abs[mask]
        # 噪声主导门槛只对有符号变化的数据适用：
        # 无过零的恒定信号（constant 数据）其 |C| 天然位于自身尾部中位数附近，
        # 不能用"全部低于 noise floor×cut"误杀。
        n_cross_full = SuperCorrDecayAnalyzer.detect_zero_crossings(
            prep.r[mask], prep.C[mask])[1]
        if n_cross_full > 0 and np.all(c_win < prep.final_noise_floor * config.r_max_noise_cut):
            warnings.append("全部数据低于 effective noise floor×cut，噪声主导")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}
        # 近噪声主导补充门槛：多数点低于噪声标尺且信号峰值无显著余量 → 噪声主导
        frac_above = float(np.mean(c_win > prep.final_noise_floor * config.r_max_noise_cut))
        if (n_cross_full > 0 and frac_above < 0.5
                and float(np.max(c_win)) < prep.final_noise_floor * 10.0):
            warnings.append(f"仅 {frac_above:.0%} 的点高于 effective noise floor×"
                            f"{config.r_max_noise_cut} 且峰值无显著余量，噪声主导")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}

        phys_osc = osc.evidence in ('strong', 'moderate')
        art = osc.structural_artifact_suspected
        span = float(np.max(prep.r) - np.min(prep.r))

        valid_non = {m: fits[m] for m in ('constant', 'power_law', 'exponential')
                     if m in fits and fits[m] is not None and fits[m].valid}
        valid_osc = {m: fits[m] for m in ('osc_power', 'osc_exp')
                     if m in fits and fits[m] is not None and fits[m].valid}

        # ---- 物理管线：振荡证据 ----
        if phys_osc:
            if art:
                warnings.append("混合振荡证据与距离族分解冲突（possible structural/geometric "
                                "artifact）→ 分析路径冲突")
                return {'classification': 'complex_decay', 'evidence_level': 'weak',
                        'nonmonotonic_reason': ''}
            env_form = envelope_fit.model.replace('envelope_', '')
            match = {'power': 'osc_power', 'exp': 'osc_exp'}[env_form]
            if match in valid_osc:
                winner = min(valid_osc, key=lambda m:
                             valid_osc[m].aicc if np.isfinite(valid_osc[m].aicc) else np.inf)
                if winner == match:
                    classification = ('oscillatory_power_law' if match == 'osc_power'
                                      else 'oscillatory_exponential')
                    evidence = osc.evidence
                    if stability.verdict == 'unstable':
                        evidence = 'weak'
                        warnings.append("稳定性 verdict=unstable，evidence 降为 weak")
                    return {'classification': classification, 'evidence_level': evidence,
                            'nonmonotonic_reason': ''}
                warnings.append("振荡证据 ≥ moderate 但候选模型组内首选与包络形式矛盾 → 冲突")
                return {'classification': 'complex_decay', 'evidence_level': 'weak',
                        'nonmonotonic_reason': ''}
            warnings.append("振荡证据 ≥ moderate 但无有效振荡候选模型，采纳物理管线并降档")
            classification = ('oscillatory_power_law' if env_form == 'power'
                              else 'oscillatory_exponential')
            return {'classification': classification, 'evidence_level': 'weak',
                    'nonmonotonic_reason': ''}

        # ---- 平坦度预判（在冲突检查之前）：衰减斜率实为零 → constant 路径 ----
        flat = False
        if 'power_law' in valid_non:
            flat = abs(valid_non['power_law'].params.get('K', 1.0)) < 0.1
        elif 'exponential' in valid_non:
            flat = valid_non['exponential'].params.get('xi', 0.0) > 10 * span
        if flat and 'constant' in valid_non:
            d_const = valid_non['constant'].aicc - min(
                valid_non[m].aicc for m in valid_non)
            if d_const < config.delta_aicc_tie:
                return {'classification': 'constant',
                        'evidence_level': 'moderate' if d_const >= 4 else 'weak',
                        'nonmonotonic_reason': ''}

        # ---- 非振荡分支 ----
        if dual:
            warnings.append("r_min 扫描出现 ≥2 个参数平台（互不相交），分析路径冲突")
            return {'classification': 'complex_decay', 'evidence_level': 'weak',
                    'nonmonotonic_reason': ''}
        # 形式切换只对非平坦数据有意义（平坦数据下 power/exp 的 AICc 差是噪声驱动）
        form_switch = SuperCorrDecayAnalyzer._form_switch(stability) and not flat
        if form_switch:
            warnings.append("包络形式随 r_min 窗口在 power↔exp 间切换（各自有稳定支持），"
                            "分析路径冲突")
            return {'classification': 'complex_decay', 'evidence_level': 'weak',
                    'nonmonotonic_reason': ''}
        if hump or osc.n_cross > 0:
            reason = ('both' if (hump and osc.n_cross > 0)
                      else ('envelope_hump' if hump else 'sign_change'))
            if osc.evidence in ('weak', 'possible'):
                warnings.append("存在振荡迹象但证据不足，summary 保留 possible oscillation 信息")
            return {'classification': 'non_monotonic', 'evidence_level': 'weak',
                    'nonmonotonic_reason': reason}
        if not valid_non:
            warnings.append("无有效非振荡候选模型")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}

        # ---- 非振荡组内模型选择（同空间同窗口，AICc 辅助证据） ----
        ranked = sorted(valid_non, key=lambda m:
                        valid_non[m].aicc if np.isfinite(valid_non[m].aicc) else np.inf)
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        daicc = ((valid_non[second].aicc - valid_non[best].aicc) if second is not None
                 else np.inf)
        # 平坦度判据：衰减斜率实为零（|K|<0.1 或 ξ>10×span）→ constant 候选
        flat = False
        if 'power_law' in valid_non:
            flat = abs(valid_non['power_law'].params.get('K', 1.0)) < 0.1
        elif 'exponential' in valid_non:
            flat = valid_non['exponential'].params.get('xi', 0.0) > 10 * span
        if flat and 'constant' in valid_non:
            d_const = valid_non['constant'].aicc - valid_non[best].aicc
            if d_const < config.delta_aicc_tie:
                return {'classification': 'constant',
                        'evidence_level': 'moderate' if d_const >= 4 else 'weak',
                        'nonmonotonic_reason': ''}
        if best in ('power_law', 'exponential'):
            if stability.verdict == 'unstable':
                warnings.append("分类依赖漂移参数且稳定性 verdict=unstable")
                return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                        'nonmonotonic_reason': ''}
            if daicc >= config.delta_aicc_tie:
                evidence = 'moderate' if daicc >= 4 else 'weak'
                return {'classification': best, 'evidence_level': evidence,
                        'nonmonotonic_reason': ''}
            warnings.append(f"power_law 与 exponential 统计并列（ΔAICc={daicc:.3f} < "
                            f"{config.delta_aicc_tie}），不可区分")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}
        return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                'nonmonotonic_reason': ''}

    # ----------------------------------------------------------------------
    # 综合接口
    # ----------------------------------------------------------------------
    @staticmethod
    def analyze(super_corr_A: list, config: AnalyzerConfig = None) -> DecayAnalysisResult:
        """完整分析入口：输入已合成的 <Δ†Δ>（list[SuperCorrRecord]），
        输出 DecayAnalysisResult（含完整 config 快照，可追溯）。"""
        config = config or AnalyzerConfig()
        warnings = []
        prep = SuperCorrDecayAnalyzer._prepare(super_corr_A, config)
        warnings.extend(prep.warnings)
        full_range = (float(np.min(prep.r)), float(np.max(prep.r)))

        result = DecayAnalysisResult(
            full_range=full_range, n_raw_points=prep.n_raw, n_distances=prep.n_distance,
            initial_noise_floor=prep.initial_noise_floor,
            initial_noise_floor_components=prep.initial_noise_floor_components,
            final_noise_floor=prep.final_noise_floor,
            final_noise_floor_components=prep.final_noise_floor_components,
            warnings=warnings, config=config)

        # 数据充分性早退
        if prep.n_distance < config.n_min_fit:
            warnings.append(f"距离数 {prep.n_distance} < n_min_fit({config.n_min_fit})，"
                            "无法进行包络拟合")
            result.classification = 'INCONCLUSIVE'
            result.evidence_level = 'insufficient'
            result.oscillation = OscillationInfo(evidence='none')
            result.stability = StabilityReport()
            result.family_diagnostics = None
            result.warnings = warnings
            return result

        # r_min 扫描
        stability = SuperCorrDecayAnalyzer.scan_r_min(prep.r, prep.C, config)
        warnings.extend(stability.messages)
        result.stability = stability
        r_min = (stability.recommended_r_min
                 if np.isfinite(stability.recommended_r_min) else float(np.min(prep.r)))
        mask = _window_mask(prep.r, r_min, prep.r_max_used)
        r_win, C_win = prep.r[mask], prep.C[mask]
        result.r_min, result.r_max = r_min, prep.r_max_used
        result.n_points = len(r_win)

        # 包络拟合（推荐窗口，|C| log 空间）
        absC = np.abs(C_win)
        f_p = SuperCorrDecayAnalyzer.fit_envelope_logspace(r_win, absC, 'power', config)
        f_e = SuperCorrDecayAnalyzer.fit_envelope_logspace(r_win, absC, 'exp', config)
        envelope_fit = (f_p if np.isfinite(f_p.aicc) and
                        (not np.isfinite(f_e.aicc) or f_p.aicc <= f_e.aicc) else f_e)
        result.envelope_fit = envelope_fit
        if envelope_fit.message:
            warnings.append(f"包络拟合: {envelope_fit.message}")

        # 距离族分解
        family_diag = SuperCorrDecayAnalyzer.family_separated_diagnostic(
            prep, envelope_fit, config)
        result.family_diagnostics = family_diag

        # 振荡诊断
        osc = SuperCorrDecayAnalyzer.analyze_oscillation(
            r_win, C_win, envelope_fit, family_diag, config)
        result.oscillation = osc
        if osc.structural_warning:
            warnings.append(osc.structural_warning)

        # 候选模型 + 组内比较
        candidate_fits = SuperCorrDecayAnalyzer.fit_candidate_models(
            r_win, C_win, envelope_fit, osc, config)
        result.candidate_fits = candidate_fits
        result.comparison = SuperCorrDecayAnalyzer.compare_models(candidate_fits, config)
        for name, fit in candidate_fits.items():
            if fit is not None and (not fit.valid or fit.message):
                warnings.append(f"候选模型 {name}: {'无效' if not fit.valid else ''}"
                                f"{fit.message}")

        # 去包络 cosine 拟合结果（detrended_fit）
        E = SuperCorrDecayAnalyzer._envelope_value(envelope_fit, r_win)
        y_det = C_win / E
        det = FitResult(model='cosine_detrended', fit_space='linear_detrended',
                        r2_space='linear', params={'Q': osc.Q, 'phi': osc.phi,
                                                   'amplitude': osc.amplitude},
                        n=len(y_det), k=2, r2=osc.r2_cos,
                        window=(r_min, prep.r_max_used))
        result.detrended_fit = det

        # 分类决策树
        hump = SuperCorrDecayAnalyzer._envelope_hump(r_win, absC, prep.final_noise_floor)
        dual = SuperCorrDecayAnalyzer._dual_plateau(stability, config)
        verdict = SuperCorrDecayAnalyzer.classify(
            {'prep': prep, 'osc': osc, 'stability': stability, 'candidate_fits': candidate_fits,
             'envelope_fit': envelope_fit, 'hump': hump, 'dual_plateau': dual,
             'warnings': warnings}, config)
        result.classification = verdict['classification']
        result.evidence_level = verdict['evidence_level']
        result.nonmonotonic_reason = verdict['nonmonotonic_reason']

        # 推荐模型与物理形式
        result.recommended_model = verdict['classification']
        p = envelope_fit.params
        if 'K' in p:
            env_str = f"A·r^(−{p['K']:.3g})"
        elif 'xi' in p:
            env_str = f"A·exp(−r/{p['xi']:.3g})"
        else:
            env_str = "A"
        if np.isfinite(osc.Q) and osc.evidence in ('strong', 'moderate', 'weak'):
            result.physical_form = f"C(r) ≈ {env_str}·cos({osc.Q:.3g}·r + {osc.phi:.3g})"
        else:
            result.physical_form = f"C(r) ≈ {env_str}"
        result.warnings = warnings
        return result

    # ----------------------------------------------------------------------
    @staticmethod
    def print_summary(result: DecayAnalysisResult) -> None:
        """打印分析摘要（含噪声底来源、8 项诊断检查逐项表、扫描表、artifact 标记、
        推荐窗口措辞；possible oscillation 信息保留）"""
        print("=" * 72)
        print(f"超导关联函数衰减分析报告 (analysis_version={result.analysis_version})")
        print("=" * 72)
        print(f"最终分类            : {result.classification}")
        print(f"推荐模型            : {result.recommended_model}")
        print(f"evidence level      : {result.evidence_level}")
        if result.nonmonotonic_reason:
            print(f"nonmonotonic_reason : {result.nonmonotonic_reason}")
        print(f"完整距离范围        : r ∈ [{result.full_range[0]:.6g}, {result.full_range[1]:.6g}]")
        print(f"推荐分析窗口(起点)  : r_min = {result.r_min:.6g}（当前稳定性准则下的推荐分析窗口起点，"
              f"非物理 crossover）")
        print(f"实际拟合范围        : r ∈ [{result.r_min:.6g}, {result.r_max:.6g}]")
        print(f"拟合点数            : {result.n_points}（原始 {result.n_raw_points} 个 bond 数据点，"
              f"{result.n_distances} 个距离；{result.distance_aggregation}）")
        print(f"拟合空间            : 包络 log 空间 ln|C(r)|（R² 与 RSS 同空间计算）")
        # 噪声底
        ic = result.initial_noise_floor_components
        fc = result.final_noise_floor_components
        print(f"effective noise floor: initial = {result.initial_noise_floor:.3e}"
              f"（远距尾部中位数 {ic.get('tail_median', float('nan')):.3e} / "
              f"组内离散度 {ic.get('intra_scatter', float('nan')):.3e} / "
              f"绝对下限 {ic.get('absolute_min', float('nan')):.3e}，取最大）")
        if not np.isclose(result.initial_noise_floor, result.final_noise_floor):
            print(f"                       final   = {result.final_noise_floor:.3e}"
                  f"（远距尾部中位数 {fc.get('tail_median', float('nan')):.3e} / "
                  f"组内离散度 {fc.get('intra_scatter', float('nan')):.3e} / "
                  f"绝对下限 {fc.get('absolute_min', float('nan')):.3e}，取最大）")
        # 包络
        env = result.envelope_fit
        if env is not None:
            print("-" * 72)
            print("包络拟合（log 空间）:")
            print(f"  模型: {env.model}   R2_log = {env.r2:.6f}  "
                  f"RSS(log) = {env.rss:.3e}  AICc = {env.aicc:.3f}  BIC = {env.bic:.3f}")
            for name, val in env.params.items():
                print(f"  {name} = {val:.6e} ± {env.param_errors.get(name, float('nan')):.3e}")
        # 振荡
        osc = result.oscillation
        if osc is not None:
            print("-" * 72)
            print(f"振荡诊断（8 项 diagnostic checks，证据聚合）: evidence = {osc.evidence}")
            print(f"  过零数 = {osc.n_cross}  极值数 = {osc.n_ext}")
            print(f"  Q = {osc.Q:.6g}  φ = {osc.phi:.6g}  amplitude = {osc.amplitude:.3e}  "
                  f"R2_cos = {osc.r2_cos:.6f}")
            print(f"  过零间距一致性 = {osc.cross_spacing_regularity:.3f}   "
                  f"极值间距一致性 = {osc.ext_spacing_regularity:.3f}")
            print(f"  Q_rel_mad = {osc.Q_rel_mad:.3f}  φ_MAD = {osc.phi_mad:.3f} rad")
            print("  逐项检查:", "  ".join(
                f"{k}:{'过' if v else '未过'}" for k, v in osc.criterion_checks.items()))
            if osc.structural_artifact_suspected:
                print("  *** possible structural/geometric artifact（专用标记）***")
            if osc.evidence == 'possible':
                print("  *** 保留 possible oscillation 信息（证据不足，未宣称振荡）***")
        # 候选模型比较
        print("-" * 72)
        print("候选模型组内比较（同 fit_space 且同窗口；跨空间不比较）:")
        for gkey, g in result.comparison.get('groups', {}).items():
            print(f"  [{gkey}] 组内最佳: {g['best_model']}")
            for row in g['rows']:
                print(f"    {row['model']:<12} AICc={row['aicc']:.3f}  "
                      f"ΔAICc={row['delta_aicc']:.3f}  ΔBIC={row['delta_bic']:.3f}"
                      f"（{row['bic_evidence']}）  R2={row['r2']:.6f}")
        # 稳定性
        stab = result.stability
        if stab is not None and stab.scan_table:
            print("-" * 72)
            print(f"r_min 扫描（{len(stab.scan_table)} 个候选，完整保留）: "
                  f"verdict = {stab.verdict}，推荐 r_min = {stab.recommended_r_min:.6g}")
            print(f"  rel_mad_K = {stab.rel_mad_K:.3f}  rel_mad_xi = {stab.rel_mad_xi:.3f}  "
                  f"rel_mad_Q = {stab.rel_mad_Q:.3f}")
            print(f"  模型一致性 = {stab.model_consistency:.2f}  "
                  f"Spearman ρ = {stab.spearman_rho:.3f}")
            for row in stab.scan_table:
                if row.get('skipped'):
                    print(f"  r_min={row['r_min']:.4g}  [跳过: {row['reason']}]")
                else:
                    print(f"  r_min={row['r_min']:.4g}  n={row['n_pts']:>3}  "
                          f"model={row['envelope_model']:<5} param={row['param']:.4g}±"
                          f"{row['param_err']:.2g}  R2_log={row['r2_env_log']:.4f}  "
                          f"AICc={row['aicc']:.2f}  osc={row['osc_evidence']:<8} "
                          f"Q={row['Q']:.4g}")
        # warnings 与物理形式
        if result.warnings:
            print("-" * 72)
            print("warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        print("-" * 72)
        print(f"物理形式: {result.physical_form}")
        print("=" * 72)

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
    # 9. 可视化（新增独立方法，不复用/不改动旧 SuperCorrProcessor 可视化）
    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_signed(ax, r, C, result):
        """面板 1：原始 signed C(r) 线性图（拟合区间底纹、过零点、极值标记、零线）"""
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        in_win = _window_mask(r, result.r_min, result.r_max)
        short = ~in_win & (r < result.r_min)
        ax.axvspan(result.r_min, result.r_max, color='gray', alpha=0.15,
                   label='fitting window')
        ax.axhline(0, color='black', linewidth=1.5, alpha=0.8)
        if short.any():
            ax.scatter(r[short], C[short], color='tab:gray', s=45, zorder=3,
                       label='short-distance (excluded)')
        ax.scatter(r[in_win], C[in_win], color='tab:blue', s=55, zorder=3, label='C(r)')
        osc = result.oscillation
        if osc is not None:
            if len(osc.cross_positions) > 0:
                ax.plot(osc.cross_positions, np.zeros_like(osc.cross_positions),
                        'gx', markersize=9, mew=2, label='zero crossings')
            if len(osc.ext_positions) > 0:
                ev = [C[np.argmin(np.abs(r - e))] for e in osc.ext_positions]
                ax.plot(osc.ext_positions, ev, 'r^', markersize=8, label='extrema')
        ax.set_title('Signed C(r)', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel(r'$\langle \Delta^\dagger \Delta \rangle$', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_loglin(ax, r, C, result):
        """面板 2：|C| 半对数 + 指数包络"""
        r = np.asarray(r, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        in_win = _window_mask(r, result.r_min, result.r_max)
        ax.semilogy(r, np.abs(C), 'o', color='tab:blue', ms=5, label='|C(r)|')
        env = result.envelope_fit
        if env is not None:
            rr = np.linspace(result.r_min, result.r_max, 200)
            ax.semilogy(rr, SuperCorrDecayAnalyzer._envelope_value(env, rr),
                        'r-', linewidth=2, label=f"exp envelope (ξ={env.params.get('xi', float('nan')):.3g})")
        ax.axvspan(result.r_min, result.r_max, color='gray', alpha=0.12)
        ax.set_title(r'$|C(r)|$ Log-Linear (exponential check)', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel(r'$|\langle \Delta^\dagger \Delta \rangle|$', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_loglog(ax, r, C, result):
        """面板 3：|C| 双对数 + 幂律包络"""
        r = np.asarray(r, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        ax.loglog(r, np.abs(C), 'o', color='tab:blue', ms=5, label='|C(r)|')
        env = result.envelope_fit
        if env is not None:
            rr = np.linspace(result.r_min, result.r_max, 200)
            ax.loglog(rr, SuperCorrDecayAnalyzer._envelope_value(env, rr),
                      'r-', linewidth=2, label=f"power envelope (K={env.params.get('K', float('nan')):.3g})")
        ax.axvspan(result.r_min, result.r_max, color='gray', alpha=0.12)
        ax.set_title(r'$|C(r)|$ Log-Log (power-law check)', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel(r'$|\langle \Delta^\dagger \Delta \rangle|$', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_detrended(ax, r, C, result):
        """面板 4：去包络 C(r)/E(r) + cosine 拟合"""
        r = np.asarray(r, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        in_win = _window_mask(r, result.r_min, result.r_max)
        env = result.envelope_fit
        osc = result.oscillation
        if env is not None and osc is not None:
            E = SuperCorrDecayAnalyzer._envelope_value(env, r[in_win])
            y_det = C[in_win] / E
            ax.plot(r[in_win], y_det, 'o', color='tab:blue', ms=5, label='C(r)/E(r)')
            if np.isfinite(osc.Q):
                rr = np.linspace(result.r_min, result.r_max, 300)
                yhat = osc.amplitude * np.cos(osc.Q * rr + osc.phi)
                ax.plot(rr, yhat, 'r-', linewidth=2,
                        label=fr'cos fit: Q={osc.Q:.3g}, φ={osc.phi:.3g}, R²={osc.r2_cos:.3f}')
        ax.axhline(0, color='black', linewidth=1.2, alpha=0.7)
        ax.set_title('Detrended C(r)/E(r) + cosine fit', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel(r'$C(r)/E(r)$', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_models(ax, r, C, result):
        """面板 5：候选模型对比（同 fit_space 组内最佳高亮，跨空间模型不混比）"""
        r = np.asarray(r, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        rr = np.linspace(result.r_min, result.r_max, 300)
        colors = {'constant': 'tab:gray', 'power_law': 'tab:green',
                  'exponential': 'tab:orange', 'osc_power': 'tab:purple',
                  'osc_exp': 'tab:brown'}
        n_cross = result.oscillation.n_cross if result.oscillation is not None else 0
        y_plot = C if n_cross == 0 else np.abs(C)
        best_names = set(result.comparison.get('best_per_group', {}).values())
        for name, fit in result.candidate_fits.items():
            if fit is None or not fit.valid:
                continue
            p = fit.params
            lw = 2.6 if name in best_names else 1.2
            ls = '-' if name in best_names else '--'
            try:
                if name == 'constant':
                    yhat = np.full_like(rr, p['A'])
                elif name == 'power_law':
                    yhat = p['A'] * rr ** (-p['K'])
                elif name == 'exponential':
                    yhat = p['A'] * np.exp(-rr / p['xi'])
                elif name == 'osc_power':
                    yhat = p['A'] * rr ** (-p['K']) * np.cos(p['Q'] * rr + p['phi'])
                elif name == 'osc_exp':
                    yhat = p['A'] * np.exp(-rr / p['xi']) * np.cos(p['Q'] * rr + p['phi'])
                else:
                    continue
            except Exception:
                continue
            ax.plot(rr, yhat, color=colors.get(name, 'black'), lw=lw, ls=ls,
                    label=f"{name}{' (best)' if name in best_names else ''}")
        ax.plot(r, y_plot, 'o', color='tab:blue', ms=4, alpha=0.7, label='data')
        ax.set_title('Candidate models (same-space best highlighted)', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel('C(r) or |C(r)|', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='best', ncol=2)
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_residuals(ax, r, C, result):
        """面板 6：最佳模型残差 ±2σ 带"""
        r = np.asarray(r, dtype=float)
        fits = result.candidate_fits
        best = None
        for gname, g in result.comparison.get('groups', {}).items():
            for row in g['rows']:
                if row['model'] in fits and fits[row['model']] is not None:
                    best = fits[row['model']]
                    break
            if best is not None:
                break
        if best is None or not best.valid:
            ax.text(0.5, 0.5, 'no valid candidate model', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        n_cross = result.oscillation.n_cross if result.oscillation is not None else 0
        in_win = _window_mask(r, result.r_min, result.r_max)
        r = r[in_win]
        C = C[in_win]
        y_win = C if n_cross == 0 else np.abs(C)
        p = best.params
        if best.model == 'constant':
            yhat = np.full_like(r, p['A'])
        elif best.model == 'power_law':
            yhat = p['A'] * r ** (-p['K'])
        elif best.model == 'exponential':
            yhat = p['A'] * np.exp(-r / p['xi'])
        elif best.model == 'osc_power':
            yhat = p['A'] * r ** (-p['K']) * np.cos(p['Q'] * r + p['phi'])
        else:
            yhat = p['A'] * np.exp(-r / p['xi']) * np.cos(p['Q'] * r + p['phi'])
        resid = y_win - yhat
        sigma = np.std(resid)
        ax.plot(r, resid, 'o', color='tab:blue', ms=5)
        ax.axhline(0, color='black', linewidth=1.2)
        ax.axhspan(-2 * sigma, 2 * sigma, color='gray', alpha=0.2, label='±2σ')
        ax.set_title(f'Residuals of best model ({best.model})', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel('residual', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_scan(ax, result):
        """面板 7：r_min 扫描（参数 vs r_min 左轴 + AICc vs r_min 右轴，含推荐线）"""
        stab = result.stability
        if stab is None or not stab.scan_table:
            ax.text(0.5, 0.5, 'no scan data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        rows = [row for row in stab.scan_table if not row.get('skipped')]
        r_mins = [row['r_min'] for row in rows]
        params = [row['param'] for row in rows]
        aiccs = [row['aicc'] for row in rows]
        ax.plot(r_mins, params, 'o-', color='tab:blue', label='K or ξ')
        ax.axvline(stab.recommended_r_min, color='red', linestyle='--', linewidth=1.5,
                   label='recommended r_min')
        ax.set_xlabel('r_min (recommended window start)', fontsize=12)
        ax.set_ylabel('envelope parameter', fontsize=12)
        ax2 = ax.twinx()
        ax2.plot(r_mins, aiccs, 's-', color='tab:orange', alpha=0.8, label='AICc')
        ax2.set_ylabel('AICc', fontsize=12)
        ax.set_title(f'r_min scan (verdict: {stab.verdict})', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_summary(ax, result):
        """面板 8：文字摘要面板"""
        ax.axis('off')
        env = result.envelope_fit
        osc = result.oscillation
        lines = [
            f"classification: {result.classification}",
            f"model: {result.recommended_model}",
            f"evidence: {result.evidence_level}",
            f"window: r ∈ [{result.r_min:.4g}, {result.r_max:.4g}], n = {result.n_points}",
            f"aggregation: {result.distance_aggregation}",
        ]
        if env is not None:
            for k, v in env.params.items():
                lines.append(f"{k} = {v:.4e} ± {env.param_errors.get(k, float('nan')):.2e}")
            lines.append(f"R2({env.r2_space}) = {env.r2:.6f}  AICc = {env.aicc:.3f}  "
                         f"BIC = {env.bic:.3f}")
        if osc is not None:
            lines.append(f"osc evidence: {osc.evidence}  Q = {osc.Q:.4g}  φ = {osc.phi:.4g}")
            if osc.structural_artifact_suspected:
                lines.append("*** possible structural/geometric artifact ***")
        lines.append(f"noise floor: {result.initial_noise_floor:.3e} → "
                     f"{result.final_noise_floor:.3e}")
        if result.nonmonotonic_reason:
            lines.append(f"nonmonotonic_reason: {result.nonmonotonic_reason}")
        lines.append("warnings:")
        for w in result.warnings[:3]:
            lines.append(f"  - {w[:60]}")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, va='top', family='monospace',
                fontname='Microsoft YaHei')   # Windows 中文字体，保证 warnings 正常显示
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def Decay_Visualization(result: DecayAnalysisResult, r: np.ndarray = None,
                            C: np.ndarray = None, show: bool = True,
                            save_path: str = None) -> plt.Figure:
        """综合 8 面板大图。r/C 为预处理后的数据（同距离平均序列），
        可由 group_by_distance 获得；不传时部分面板显示提示。"""
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)
        fig.suptitle(
            f'Superconducting Correlation Decay Analysis — '
            f'classification: {result.classification} (evidence: {result.evidence_level})',
            fontsize=20, fontweight='bold')
        if r is None or C is None:
            r = np.array([])
            C = np.array([])
        SuperCorrDecayAnalyzer._draw_panel_signed(fig.add_subplot(gs[0, 0]), r, C, result)
        SuperCorrDecayAnalyzer._draw_panel_loglin(fig.add_subplot(gs[0, 1]), r, C, result)
        SuperCorrDecayAnalyzer._draw_panel_loglog(fig.add_subplot(gs[0, 2]), r, C, result)
        SuperCorrDecayAnalyzer._draw_panel_detrended(fig.add_subplot(gs[0, 3]), r, C, result)
        SuperCorrDecayAnalyzer._draw_panel_models(fig.add_subplot(gs[1, 0]), r, C, result)
        SuperCorrDecayAnalyzer._draw_panel_residuals(fig.add_subplot(gs[1, 1]), r, C, result)
        SuperCorrDecayAnalyzer._draw_panel_scan(fig.add_subplot(gs[1, 2]), result)
        SuperCorrDecayAnalyzer._draw_panel_summary(fig.add_subplot(gs[1, 3]), result)
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    # ----------------------------------------------------------------------
    @staticmethod
    def Decay_Visualization_compare(results: dict, show: bool = True,
                                    save_path: str = None) -> plt.Figure:
        """批量对比图：左面板叠加各组 |C|(r)+包络，右面板 K/ξ 与分类对比表。
        仅并排展示供人工比较，不自动做跨 refbond 物理结论。"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Decay Analysis Comparison (no automatic cross-refbond conclusion)',
                     fontsize=18, fontweight='bold')
        colors = plt.cm.tab10.colors
        labels, params, classes = [], [], []
        for i, (name, res) in enumerate(results.items()):
            c = colors[i % len(colors)]
            labels.append(name)
            env = res.envelope_fit
            pval = np.nan
            if env is not None:
                pval = env.params.get('K', env.params.get('xi', np.nan))
            params.append(pval)
            classes.append(f"{res.classification} ({res.evidence_level})")
            ax1.annotate(0, (0, 0), alpha=0)   # 占位（数据由调用方传入则叠加；此处给出图例文本）
        ax1.axis('off')
        ax1.text(0.5, 0.5,
                 "将各数据集的 |C(r)| 与包络叠加绘图需传入 (r, C) 序列；\n"
                 "请使用 Decay_Visualization 单图查看，或在此传入数据后扩展。",
                 transform=ax1.transAxes, ha='center', va='center', fontsize=12,
                 fontname='Microsoft YaHei')
        ax2.axis('off')
        rows = [[labels[i], f"{params[i]:.4g}", classes[i]]
                for i in range(len(results))]
        table = ax2.table(cellText=rows, colLabels=['dataset', 'K or ξ', 'classification'],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig


# ==========================================================================
# 独立面板保存接口（plot_panel_*：ax=None 自建，save_path 非 None 单独保存）
# ==========================================================================
def _plot_panel(name: str, result: DecayAnalysisResult, r=None, C=None,
                ax=None, save_path: str = None):
    """面板绘制公共入口：drawer 名 → 面板函数"""
    if ax is None:
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
