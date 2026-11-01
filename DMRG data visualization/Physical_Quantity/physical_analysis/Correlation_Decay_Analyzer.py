# -*- coding: utf-8 -*-
"""
通用空间关联函数衰减类型分析与拟合核心（与具体物理量无关）

对任意 (r, C) 关联数据做长距离衰减分析：
  原始 signed C(r) → 长距离窗口选择(r_min 扫描) → 符号变化/极值检测
  → 包络提取(log 空间线性拟合) → 去包络 C(r)/E(r) → cosine 振荡诊断(证据评分制)
  → 9 个候选模型拟合(传统比较层) → 同空间同窗口组内 AICc/BIC 比较
  → 冲突仲裁 → 8 类分类 + evidence level。

只依赖 numpy 数组 (r, C) 与结果 dataclass，不 import 任何
Physical_Quantity 模块。距离族（几何族）语义由数据提供方决定
（PreparedData.group_family），族分解诊断对族编号本身无假设。

超导专用适配见 Super_Correlation_Analyzer.py（SuperCorrDecayAnalyzer
继承本类，覆盖 record 输入适配、蜂窝键族提取与物理文案）。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy import optimize, signal

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
    n_min_hump_window: int = 6         # hump 后衰减段窗口最小点数（2 参数拟合 n>k+3 即可，略低于 n_min_fit）
    n_min_osc_fit: int = 10            # 4 参数模型最小点数(<10 warning, <6 拒绝)
    n_min_osc: int = 12                # 振荡分析的最小点数
    n_min_abs: int = 6                 # 全局最小距离数，不足 → INCONCLUSIVE
    # ---- 窗口 ----
    r_min_override: float = None       # 手动指定拟合窗口起点（覆盖自动推荐；None=自动）
    r_max_frac: float = 1.0
    r_max_auto: bool = True            # 自动截断噪声尾
    r_max_noise_cut: float = 3.0       # |C| < noise_floor*cut 视为噪声点
    r_max_noise_run: int = 3           # 连续低于阈值的点数要求(防单点尖峰)
    n_rmin_candidates: int = 20        # r_min 候选数上限
    # ---- 噪声底(effective noise floor / conservative noise scale) ----
    envelope_mode: str = 'all_points'  # 包络模式：'all_points'(全点) | 'extrema'(峰值法) | 'both'(两者都算并对比)
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
    # ---- 常数项 C_inf 约束（带常数项模型） ----
    constrain_cinf_abs: bool = True      # |C| 空间：C_inf ≥ 0（默认启用；|C| 空间常数项物理上非负）
    constrain_cinf_signed: bool = False  # signed C(r) 空间：默认自由（可为负）；True 时施加 C_inf ≥ 0
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
    bound_hits: list = field(default_factory=list)   # 触及 bounds 的参数名列表


@dataclass
class FamilyDiagnostics:
    """距离族分解诊断（族编号语义由数据提供方定义；
    超导场景下为几何族：dy=0 族 / |dy|=0.5√3·a 族）"""
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
    post_hump_windows: int = 0            # 峰后合法窗口起点数（无 hump 时为 0）
    verdict: str = 'unstable'             # 'stable' | 'moderate' | 'unstable' | 'insufficient'
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
    envelope_fit_alt: FitResult = None   # envelope_mode='both' 时的对比包络（另一种模式）
    envelope_extrema_r: np.ndarray = field(default_factory=lambda: np.array([]))   # 峰值法参与拟合的极值点位置
    envelope_extrema_abs: np.ndarray = field(default_factory=lambda: np.array([])) # 对应 |C| 值（可视化高亮用）
    envelope_extrema_excluded: list = field(default_factory=list)   # 峰值法按噪声底筛选剔除的极值点 [(r, C, 原因)]
    envelope_mode_used: str = ''         # 实际生效的包络模式：'all_points' | 'extrema'
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
# 2. 工具（模块级私有小函数）
# ==========================================================================
def _fmt_sci(x: float) -> str:
    """物理参数打印格式：|指数| > 3 时用科学计数法（如 4.079838×10^(−6)），
    普通数量级保持常规小数（6 位）；nan/inf 原样输出。"""
    x = float(x)
    if not np.isfinite(x):
        return f"{x}"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    if abs(exp) > 3:
        return f"{x / 10.0 ** exp:.6f}×10^({exp:+d})"
    return f"{x:.6f}"


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


def _scaled_fit_problem(f, param_names: list, scale: float):
    """构造幅度缩放后的模型函数：缩放空间拟合 f_s = f/scale，
    幅度类参数（A / C_inf）在缩放空间为原值的 1/scale（非线性参数不变）。
    用于消除小量级数据下 trf 优化器的数值病态（浮点相消使收敛判据失效）。"""
    amp_idx = [i for i, n in enumerate(param_names) if n in ('A', 'C_inf')]

    def f_s(x, *p):
        p = list(p)
        for i in amp_idx:
            p[i] = p[i] * scale      # 缩放空间幅度参数 → 原空间
        return f(x, *p) / scale
    return f_s, amp_idx


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


def _window_mask(r: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    return (r >= r_min - 1e-12) & (r <= r_max + 1e-12)


# ==========================================================================
# 3. 通用衰减分析核心（CorrelationDecayAnalyzer）
#    文案类属性由子类覆盖实现物理量专属表述；算法无任何物理量假设
# ==========================================================================
class CorrelationDecayAnalyzer:
    """通用关联函数衰减分析核心（全 staticmethod/classmethod，无隐式状态；
    config 显式传入）。输入仅 (r, C) 数组与结果 dataclass。

    子类适配点（文案，不影响算法行为）：
      _REPORT_TITLE      完整报告标题
      _SUPTITLE_MAIN     可视化主标题（standard/full 模式）
      _COMPARE_SUPTITLE  批量对比图标题
      _BG_PHYSICS_NOTE   常数项背景显著性 warning 的物理解释语
      _RAW_UNIT_DESC     原始点单位描述（如 "bond 数据点" / "数据点"）
    """

    # ---- 文案类属性（超导子类覆盖） ----
    _REPORT_TITLE = "关联函数衰减分析报告"
    _SUPTITLE_MAIN = "Correlation Decay Analysis"
    _COMPARE_SUPTITLE = 'Decay Analysis Comparison (no automatic cross-dataset conclusion)'
    _BG_PHYSICS_NOTE = "可能为常数值背景"
    _RAW_UNIT_DESC = "数据点"

    # ----------------------------------------------------------------------
    # 3.1 噪声底与包络拟合
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
        if len(r) == 0:
            # 无有效正 |C| 点（全零/全非正/全 NaN）→ 无效 FitResult，不除零
            return FitResult(
                model=f'envelope_{form}', fit_space='log_absC', r2_space='log',
                valid=False, message="无有效 |C| 点（log 空间无定义），无法拟合包络")
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
    def _fit_envelope_with_mode(r: np.ndarray, C: np.ndarray, noise_floor: float,
                                config: AnalyzerConfig) -> tuple:
        """按 config.envelope_mode 拟合包络，
        返回 (envelope_fit, alt_fit, warnings, extrema_data)：
        extrema_data = (ext_r, |ext_C|, excluded)——峰值法参与拟合的极值点
        （供可视化高亮）与按噪声底筛选剔除的极值点 [(r, C, 原因)]，
        all_points 模式时 ext_r 为空数组。
        - 'all_points'：全点 |C(r)| 对数空间拟合（谷底点会系统性拉低包络）
        - 'extrema'：峰值法——斜率符号法找到的局部极值点做对数空间拟合（不受谷底拉低，但样本少）
        - 'both'：两种都算，AICc 优者为主包络，另一个作为对比结果返回
        """
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        warnings = []
        absC = np.abs(C)
        empty = (np.array([]), np.array([]), [])
        f_p_all = CorrelationDecayAnalyzer.fit_envelope_logspace(r, absC, 'power', config)
        f_e_all = CorrelationDecayAnalyzer.fit_envelope_logspace(r, absC, 'exp', config)
        env_all = (f_p_all if np.isfinite(f_p_all.aicc) and
                   (not np.isfinite(f_e_all.aicc) or f_p_all.aicc <= f_e_all.aicc)
                   else f_e_all)
        tag_all = f"全点包络（all_points，n={len(r)} 个点）"
        env_all.message = (tag_all if not env_all.message
                           else env_all.message + "；" + tag_all)
        if config.envelope_mode == 'all_points':
            return (env_all, None, warnings, empty)

        ext_r, ext_v = CorrelationDecayAnalyzer.detect_extrema(r, C, noise_floor, config)
        # 原始（未筛选）极值 vs 筛选后极值：记录剔除的极值点及原因
        # （振荡诊断的"极值数"是未筛选原始计数，与峰值法拟合样本口径不同）
        raw_r, raw_v = CorrelationDecayAnalyzer.detect_extrema(r, C, 0.0, config)
        excluded = []
        for rr, vv in zip(raw_r, raw_v):
            if not np.any(np.isclose(ext_r, rr)):
                reason = ('幅度不足' if abs(vv) <= noise_floor * config.prom_factor
                          else '与相邻异类极值落差不足')
                excluded.append((float(rr), float(vv), reason))
        if len(ext_r) < 4:
            # 退化分级：原始极值充足但被噪声标尺筛掉 = 极值幅度不确定（小动态
            # 范围数据下噪声底与信号同量级，绝对阈值失效），全点拟合仅作对照；
            # 原始极值本身不足 = 噪声主导，全点拟合为主结果
            if len(raw_r) >= 4:
                warnings.append(
                    f"envelope_mode='{config.envelope_mode}' 极值幅度与噪声底同量级："
                    f"原始极值 {len(raw_r)} 个，通过幅度筛选的仅 {len(ext_r)} 个（<4），"
                    f"峰值法不可用，全点拟合仅作对照")
            else:
                warnings.append(f"envelope_mode='{config.envelope_mode}' 但极值点仅 "
                                f"{len(ext_r)} 个（<4），退化回全点包络")
            return (env_all, None, warnings, (np.array([]), np.array([]), excluded))
        ext_abs = np.abs(ext_v)
        f_p_ext = CorrelationDecayAnalyzer.fit_envelope_logspace(
            ext_r, ext_abs, 'power', config)
        f_e_ext = CorrelationDecayAnalyzer.fit_envelope_logspace(
            ext_r, ext_abs, 'exp', config)
        env_ext = (f_p_ext if np.isfinite(f_p_ext.aicc) and
                   (not np.isfinite(f_e_ext.aicc) or f_p_ext.aicc <= f_e_ext.aicc)
                   else f_e_ext)
        tag = f"极值点包络（峰值法，n={len(ext_r)} 个极值点）"
        env_ext.message = tag if not env_ext.message else env_ext.message + "；" + tag
        if config.envelope_mode == 'extrema':
            return (env_ext, None, warnings, (ext_r, ext_abs, excluded))

        # both：AICc 优者为主（并列时按规则峰值法胜出）
        if np.isfinite(env_ext.aicc) and (not np.isfinite(env_all.aicc)
                                          or env_ext.aicc <= env_all.aicc):
            return (env_ext, env_all, warnings, (ext_r, ext_abs, excluded))
        return (env_all, env_ext, warnings, (ext_r, ext_abs, excluded))

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
    # 3.2 振荡诊断（过零 / 极值 / Q / 稳定性 / 距离族分解）
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
        # NaN/Inf 守卫：y_det 含非有限值时 lstsq 因 NaN 矩阵抛 LinAlgError，
        # 优雅返回 NaN（调用方将其视为振荡证据缺失）
        if not np.all(np.isfinite(y_det)):
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
            q0, _, _ = CorrelationDecayAnalyzer.estimate_q_lomb_scargle(rr, yy, config)
            q, phi, _, _ = CorrelationDecayAnalyzer.refine_q_cosine(rr, yy, q0, config)
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
        cross_pos, n_cross = CorrelationDecayAnalyzer.detect_zero_crossings(r, C)
        E = CorrelationDecayAnalyzer._envelope_value(env_fit, r)
        y_det = C / E
        # NaN/Inf 守卫：去包络信号必须逐点有限（含单点 NaN 时振荡
        # 拟合无定义，lstsq 会因 NaN 矩阵抛 LinAlgError）——优雅跳过，
        # osc_flag=False，不崩溃
        if not np.all(np.isfinite(y_det)):
            return res
        q0, _, _ = CorrelationDecayAnalyzer.estimate_q_lomb_scargle(r, y_det, config)
        Q, phi, amp, r2_cos = CorrelationDecayAnalyzer.refine_q_cosine(r, y_det, q0, config)
        qs, phis = CorrelationDecayAnalyzer._subwindow_q_phi(r, y_det, config)
        q_rel_mad = _rel_mad(np.array(qs)) if qs else np.nan
        phi_mad = _mad_abs(np.array(phis)) if phis else np.nan
        # 过零/极值间距一致性
        d_cross = np.diff(cross_pos) if len(cross_pos) > 1 else np.array([])
        cross_reg = _rel_mad(d_cross) if len(d_cross) > 1 else np.nan
        noise = 0.0
        ext_r, ext_v = CorrelationDecayAnalyzer.detect_extrema(r, C, noise, config)
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
        evidence = CorrelationDecayAnalyzer._oscillation_evidence(checks, n, n_cross, config)
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
        """距离族分解诊断：对各几何族分别做符号纯度、过零、包络、去包络 cosine。
        族内点数 < n_min_fit/2 → 该族标记 insufficient。
        （族编号语义由 PreparedData.group_family 的提供方定义）"""
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
            entry['n_cross'] = CorrelationDecayAnalyzer.detect_zero_crossings(r_f, C_f)[1]
            # 族内包络（|C| log 空间 power/exp 选 AICc 优者）
            absC = np.abs(C_f)
            f_p = CorrelationDecayAnalyzer.fit_envelope_logspace(r_f, absC, 'power', config)
            f_e = CorrelationDecayAnalyzer.fit_envelope_logspace(r_f, absC, 'exp', config)
            pick = f_p if (np.isfinite(f_p.aicc) and
                           (not np.isfinite(f_e.aicc) or f_p.aicc <= f_e.aicc)) else f_e
            entry['envelope_form'] = pick.model.replace('envelope_', '')
            entry['envelope_param'] = pick.params.get('K', pick.params.get('xi', np.nan))
            entry['r2_env_log'] = pick.r2
            # 族内去包络 cosine
            E_f = CorrelationDecayAnalyzer._envelope_value(pick, r_f)
            y_det = C_f / E_f
            q0, _, _ = CorrelationDecayAnalyzer.estimate_q_lomb_scargle(r_f, y_det, config)
            Q_f, phi_f, _, r2_cos = CorrelationDecayAnalyzer.refine_q_cosine(
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
        cross_pos, n_cross = CorrelationDecayAnalyzer.detect_zero_crossings(r, C)
        info.n_cross = n_cross
        info.cross_positions = cross_pos
        ext_r, ext_v = CorrelationDecayAnalyzer.detect_extrema(r, C, noise, config)
        info.n_ext = len(ext_r)
        info.ext_positions = ext_r
        d_cross = np.diff(cross_pos) if len(cross_pos) > 1 else np.array([])
        d_ext = np.diff(ext_r) if len(ext_r) > 1 else np.array([])
        info.cross_spacing_regularity = _rel_mad(d_cross) if len(d_cross) > 1 else np.nan
        info.ext_spacing_regularity = _rel_mad(d_ext) if len(d_ext) > 1 else np.nan

        # 去包络 + Q 估计
        E = CorrelationDecayAnalyzer._envelope_value(envelope_fit, r)
        y_det = C / E
        q0, freqs, power = CorrelationDecayAnalyzer.estimate_q_lomb_scargle(r, y_det, config)
        Q, phi, amp, r2_cos = CorrelationDecayAnalyzer.refine_q_cosine(r, y_det, q0, config)
        info.Q, info.phi, info.amplitude, info.r2_cos = Q, phi, amp, r2_cos
        qs, phis = CorrelationDecayAnalyzer._subwindow_q_phi(r, y_det, config)
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
        info.evidence = CorrelationDecayAnalyzer._oscillation_evidence(
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
    # 3.3 候选模型拟合（传统比较层）
    # ----------------------------------------------------------------------
    @staticmethod
    def bootstrap_errors(f, x: np.ndarray, y: np.ndarray, p_opt: np.ndarray,
                         bounds: tuple = None, param_names: list = None,
                         config: AnalyzerConfig = None) -> np.ndarray:
        """残差 bootstrap 参数不确定度。仅作 pcov 失效时的兜底，
        不作为主要物理证据，不参与模型选择与分类决策。
        与主拟合使用同一参数边界（避免参数试探越界，如 ξ<0 导致 exp 溢出），
        并做同样的幅度缩放（避免小量级数据的数值病态）。"""
        import warnings as _warnings
        config = config or AnalyzerConfig()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(f(x, *p_opt), dtype=float)
        resid = y - yhat
        scale = float(np.max(np.abs(y))) or 1.0
        f_s, amp_idx = _scaled_fit_problem(f, param_names or [], scale)
        lo = np.asarray(bounds[0], dtype=float) if bounds else None
        hi = np.asarray(bounds[1], dtype=float) if bounds else None
        rng = np.random.default_rng(42)
        params = []
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')   # 越界试探产生的数值警告属预期，静默
            for _ in range(config.bootstrap_n):
                ys = (yhat + rng.choice(resid, size=len(y), replace=True)) / scale
                try:
                    p0_s = np.array(p_opt, dtype=float)
                    p0_s[amp_idx] /= scale
                    if bounds is not None:
                        lo_s = np.array(bounds[0], dtype=float)
                        hi_s = np.array(bounds[1], dtype=float)
                        lo_s[amp_idx] /= scale
                        hi_s[amp_idx] /= scale
                        b = (lo_s, hi_s)
                    else:
                        b = (-np.inf, np.inf)
                    popt, _ = optimize.curve_fit(f_s, x, ys, p0=p0_s,
                                                 bounds=b, maxfev=20000)
                    popt = np.array(popt)
                    popt[amp_idx] *= scale
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
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)
        # 幅度缩放：小量级数据下 trf 优化器的收敛判据会被浮点相消淹没
        # （表现为 popt 完全不动、停在 p0）。缩放空间拟合后，
        # 幅度类参数 (A / C_inf) 线性反变换，非线性参数 (K/xi/Q/phi) 不变。
        scale = float(np.max(np.abs(y_win)))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        y_s = y_win / scale
        f_s, amp_idx = _scaled_fit_problem(f, param_names, scale)
        p0_s, lo_s, hi_s = p0.copy(), lo.copy(), hi.copy()
        p0_s[amp_idx] /= scale
        lo_s[amp_idx] /= scale
        hi_s[amp_idx] /= scale
        import warnings as _warnings
        try:
            # pcov 估计失败/数值溢出属预期情形（参数顶边界），已由后续兜底处理，静默警告
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                popt, pcov = optimize.curve_fit(f_s, r_win, y_s, p0=p0_s,
                                                bounds=(lo_s, hi_s), max_nfev=20000)
        except Exception as e:
            res.valid = False
            res.message += f"；拟合失败: {type(e).__name__}"
            return res
        # 幅度参数反变换 + 协方差反变换（D_ii = scale 或 1 的对角变换）
        popt = np.asarray(popt, dtype=float)
        popt[amp_idx] *= scale
        if pcov is not None and np.all(np.isfinite(pcov)):
            D = np.ones(len(popt))
            D[amp_idx] = scale
            pcov = pcov * np.outer(D, D)
        yhat = np.asarray(f(r_win, *popt), dtype=float)
        rss = float(np.sum((y_win - yhat) ** 2))
        r2 = _r2_in_space(y_win, yhat)
        aic, aicc, bic = _aicc_bic_core(rss, n, k)
        res.rss, res.rmse, res.r2 = rss, float(np.sqrt(rss / n)), r2
        res.aic, res.aicc, res.bic = aic, aicc, bic
        # 触及边界报告（相对容差 1%）：以参数自身最优值尺度为准，
        # 边界为 0 时改用对侧边界尺度（机器残差如 1e-26 相对自身无法判定）。
        # 若把 bounds 跨度混入尺度，跨度巨大的参数（如放宽后的 A 上界）
        # 会把所有拟合都误报为"触及边界"
        hit_names = []
        for pname, v, l, h in zip(param_names, popt, lo, hi):
            if np.isfinite(l):
                denom_l = max(abs(v), abs(l), 1e-300) if l != 0 else max(abs(h), 1e-300)
                if abs(v - l) <= 0.01 * denom_l:
                    hit_names.append(f"参数 {pname} 触及下界 {l:.4g}")
                    res.bound_hits.append(pname)
                    continue
            if np.isfinite(h):
                denom_h = max(abs(v), abs(h), 1e-300) if h != 0 else max(abs(l), 1e-300)
                if abs(v - h) <= 0.01 * denom_h:
                    hit_names.append(f"参数 {pname} 触及上界 {h:.4g}")
                    res.bound_hits.append(pname)
        if hit_names:
            res.message = ((res.message + "；") if res.message else "") + "；".join(hit_names)
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
                err = CorrelationDecayAnalyzer.bootstrap_errors(
                    f, r_win, y_win, popt, (lo, hi), param_names, config)
                err_src = 'bootstrap'
        else:
            err = CorrelationDecayAnalyzer.bootstrap_errors(
                f, r_win, y_win, popt, (lo, hi), param_names, config)
            err_src = 'bootstrap'
        res.params = {name: float(v) for name, v in zip(param_names, popt)}
        res.param_errors = {name: float(e) for name, e in zip(param_names, err)}
        res.error_source = err_src
        return res

    # ----------------------------------------------------------------------
    @staticmethod
    def fit_candidate_models(r: np.ndarray, C: np.ndarray, envelope_fit: FitResult,
                             osc_info: OscillationInfo, config: AnalyzerConfig = None) -> dict:
        """9 个候选模型拟合（5 基础 + 4 带常数项 _C 变体）。拟合空间策略：
        无过零 → 非振荡模型拟 signed C(r)；有过零 → 非振荡模型拟 |C|、
        振荡模型拟 signed C。AICc/BIC 只在同 fit_space 且同窗口组内比较。"""
        config = config or AnalyzerConfig()
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        span = float(np.max(r) - np.min(r))
        max_abs = float(np.max(np.abs(C)))
        y_abs = np.abs(C)
        n_cross = CorrelationDecayAnalyzer.detect_zero_crossings(r, C)[1]
        window = (float(np.min(r)), float(np.max(r)))
        fits = {}

        # ---- 非振荡组：无过零拟 signed，有过零拟 |C| ----
        y_non = C if n_cross == 0 else y_abs
        fit_space_non = 'linear_C' if n_cross == 0 else 'linear_absC'
        A_lo, A_hi = 0.1 * max_abs, 10.0 * max_abs
        if A_lo <= 0 or A_hi <= A_lo:
            A_lo, A_hi = 1e-300, max(1.0, 10 * max_abs)
        # A 上界放宽：窗口起点 r_min 较大时真实幅度可远超窗口内 max|C|
        # （A·r_min^(-K) = max|C| → A = max|C|·r_min^K）。取 max(r)^8 覆盖 K ≤ 8 的任意窗口，
        # 避免幅度被边界卡住后用其它参数（如负 C_inf）错误补偿。
        A_hi = max(A_hi, max_abs * float(np.max(r)) ** 8)
        # ---- 常数项 C_inf 约束规则（带常数项模型） ----
        # |C| 空间：constrain_cinf_abs=True 时 C_inf ∈ [0, max]；否则 [-max, max]
        # signed 空间：默认自由 [-max, max]；constrain_cinf_signed=True 时 [0, max]
        # 振荡模型只约束非振荡背景 C_inf，不约束整个振荡函数值
        c_hi = max_abs
        c_lo_abs = 0.0 if config.constrain_cinf_abs else -max_abs
        c_lo_signed = 0.0 if config.constrain_cinf_signed else -max_abs
        c_lo_non = c_lo_abs if n_cross > 0 else c_lo_signed

        # constant
        fits['constant'] = CorrelationDecayAnalyzer._fit_one(
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
        fits['power_law'] = CorrelationDecayAnalyzer._fit_one(
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
        fits['exponential'] = CorrelationDecayAnalyzer._fit_one(
            'exponential', lambda rr, A, xi: A * np.exp(-rr / xi),
            r, y_non, [A0_e, np.clip(xi0, 0.1, 10 * span)],
            ([A_lo, 0.1], [A_hi, 10.0 * span]), 2, window,
            ['A', 'xi'], fit_space_non, config)

        # ---- 带常数项的非振荡模型（C_inf + 衰减；C_inf 约束随拟合空间） ----
        fits['power_law_C'] = CorrelationDecayAnalyzer._fit_one(
            'power_law_C', lambda rr, C0, A, K: C0 + A * rr ** (-K),
            r, y_non, [0.0, A0_p, np.clip(K0, 0.1, 8.0)],
            ([c_lo_non, A_lo, 0.0], [c_hi, A_hi, 8.0]), 3, window,
            ['C_inf', 'A', 'K'], fit_space_non, config)
        fits['exponential_C'] = CorrelationDecayAnalyzer._fit_one(
            'exponential_C', lambda rr, C0, A, xi: C0 + A * np.exp(-rr / xi),
            r, y_non, [0.0, A0_e, np.clip(xi0, 0.1, 10 * span)],
            ([c_lo_non, A_lo, 0.1], [c_hi, A_hi, 10.0 * span]), 3, window,
            ['C_inf', 'A', 'xi'], fit_space_non, config)

        # ---- 振荡组：拟 signed C(r) ----
        env = envelope_fit if envelope_fit is not None else fits['power_law']
        A0 = float(env.params.get('A', max_abs))
        Q0 = osc_info.Q if (osc_info is not None and np.isfinite(osc_info.Q)) else 1.0
        phi0 = osc_info.phi if (osc_info is not None and np.isfinite(osc_info.phi)) else 0.0
        Q_lo, Q_hi = 0.8 * Q0, 1.2 * Q0

        K_env = float(env.params.get('K', 1.0))
        fits['osc_power'] = CorrelationDecayAnalyzer._fit_one(
            'osc_power', lambda rr, A, K, Q, phi: A * rr ** (-K) * np.cos(Q * rr + phi),
            r, C, [A0, np.clip(K_env, 0.0, 8.0), Q0, phi0],
            ([A_lo, 0.0, Q_lo, -np.pi], [A_hi, 8.0, Q_hi, np.pi]), 4, window,
            ['A', 'K', 'Q', 'phi'], 'linear_C', config)

        xi_env = float(env.params.get('xi', span))
        fits['osc_exp'] = CorrelationDecayAnalyzer._fit_one(
            'osc_exp', lambda rr, A, xi, Q, phi: A * np.exp(-rr / xi) * np.cos(Q * rr + phi),
            r, C, [A0, np.clip(xi_env, 0.1, 10 * span), Q0, phi0],
            ([A_lo, 0.1, Q_lo, -np.pi], [A_hi, 10.0 * span, Q_hi, np.pi]), 4, window,
            ['A', 'xi', 'Q', 'phi'], 'linear_C', config)

        # ---- 带常数项的振荡模型（只约束非振荡背景 C_inf，不约束函数值） ----
        fits['osc_power_C'] = CorrelationDecayAnalyzer._fit_one(
            'osc_power_C',
            lambda rr, C0, A, K, Q, phi: C0 + A * rr ** (-K) * np.cos(Q * rr + phi),
            r, C, [0.0, A0, np.clip(K_env, 0.0, 8.0), Q0, phi0],
            ([c_lo_signed, A_lo, 0.0, Q_lo, -np.pi],
             [c_hi, A_hi, 8.0, Q_hi, np.pi]), 5, window,
            ['C_inf', 'A', 'K', 'Q', 'phi'], 'linear_C', config)
        fits['osc_exp_C'] = CorrelationDecayAnalyzer._fit_one(
            'osc_exp_C',
            lambda rr, C0, A, xi, Q, phi: C0 + A * np.exp(-rr / xi) * np.cos(Q * rr + phi),
            r, C, [0.0, A0, np.clip(xi_env, 0.1, 10 * span), Q0, phi0],
            ([c_lo_signed, A_lo, 0.1, Q_lo, -np.pi],
             [c_hi, A_hi, 10.0 * span, Q_hi, np.pi]), 5, window,
            ['C_inf', 'A', 'xi', 'Q', 'phi'], 'linear_C', config)

        # 约束启用状态写入 message（报告约束是否启用；print_summary 自动透出）
        for name in ('power_law_C', 'exponential_C', 'osc_power_C', 'osc_exp_C'):
            f = fits[name]
            clo = c_lo_non if name in ('power_law_C', 'exponential_C') else c_lo_signed
            tag = (f"C_inf ∈ [{clo:.4g}, {c_hi:.4g}]"
                   f"（{'非负约束启用' if clo >= 0 else '无符号约束'}）")
            f.message = ((f.message + "；") if f.message else "") + tag
        return fits

    # ----------------------------------------------------------------------
    # 3.4 模型比较（手写信息准则，同空间同窗口组内比较）
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
    # 3.5 r_min 扫描与稳定性
    # ----------------------------------------------------------------------
    @staticmethod
    def scan_r_min(r: np.ndarray, C: np.ndarray,
                   config: AnalyzerConfig = None,
                   noise_floor: float = 0.0,
                   hump_peak: float = None) -> StabilityReport:
        """r_min 全网格自动扫描。每个有效候选：包络 power/exp 拟合 +
        该窗口内重新执行振荡诊断（evidence/Q/φ/R2_cos/osc_flag），
        完整保存全部候选行（含跳过原因），推荐值只从扫描表自动产生。
        hump_peak 非 None 时（数据存在包络 hump），推荐窗口起点设在 hump 之后，
        使单调包络只拟合衰减段。"""
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
        # hump 修正：峰后满足最小窗口的起点补入网格，保证推荐窗口参与稳定性评估
        # （否则 hump 把推荐起点推到网格之外，主分析窗口从未被扫描）
        hump_present = hump_peak is not None and np.isfinite(hump_peak)
        if hump_present:
            after_idx = [i for i in range(M)
                         if r[i] > hump_peak + 1e-12
                         and M - i >= config.n_min_hump_window]
            candidates = sorted(set(candidates) | set(after_idx))

        rows = []
        for idx in candidates:
            r_min = float(r[idx])
            r_win = r[idx:]
            C_win = C[idx:]
            n = len(r_win)
            # 峰后候选按 hump 窗口门槛（略低于 n_min_fit），近程候选维持 n_min_fit
            n_min_w = (config.n_min_hump_window
                       if hump_present and r_min > hump_peak + 1e-12
                       else config.n_min_fit)
            if n < n_min_w:
                rows.append({'r_min': r_min, 'n_pts': n, 'skipped': True,
                             'reason': f'n_pts={n} < n_min_w({n_min_w})'})
                continue
            # 包络拟合（按 envelope_mode；'both' 时以主包络为准）
            env, _, _, _ = CorrelationDecayAnalyzer._fit_envelope_with_mode(
                r_win, C_win, noise_floor, config)
            param_key = 'K' if 'K' in env.params else 'xi'
            # 该候选窗口内重新执行振荡诊断
            osc = CorrelationDecayAnalyzer._quick_oscillation(r_win, C_win, env, config)
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

        # verdict（基础档：只由参数相对离散度决定）
        if np.isfinite(rel_mad):
            if rel_mad <= config.k_stab_max:
                rep.verdict = 'stable'
            elif rel_mad <= config.k_stab_moderate:
                rep.verdict = 'moderate'
            else:
                rep.verdict = 'unstable'
        else:
            rep.verdict = 'unstable'
        # Spearman 单调漂移降档（与 insufficient 分开：这是窗口依赖问题）
        if (np.isfinite(rep.spearman_rho) and abs(rep.spearman_rho) >= 0.8
                and len(valid_rows) >= 4):
            if rep.verdict == 'stable':
                rep.verdict = 'moderate'
            rep.messages.append(
                f"参数随 r_min 单调漂移（Spearman ρ={rep.spearman_rho:.2f}），"
                f"verdict 降为 moderate（可能存在未被模型描述的近程贡献/边界效应）")

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
        # hump 修正：峰后合法窗口数分级（与稳定性降档语义分开）——
        # 峰后窗口可拟合 ≠ 峰后拟合稳定；峰后不足 ≠ 静默退回全区间
        if hump_peak is not None and np.isfinite(hump_peak):
            after_idx = [i for i in range(M)
                         if r[i] > hump_peak + 1e-12
                         and M - i >= config.n_min_hump_window]
            rep.post_hump_windows = len(after_idx)
            if rep.post_hump_windows == 0:
                # 峰后点数不足：不给出推荐起点（不静默退回全区间当衰减窗口），
                # verdict 独立记 insufficient（数据不足，非稳定性降档）
                rep.recommended_r_min = np.nan
                rep.verdict = 'insufficient'
                rep.messages.append(
                    f"检测到包络 hump（峰值 r≈{hump_peak:.6g}）但峰后点数不足 "
                    f"n_min_hump_window({config.n_min_hump_window})，衰减段证据不足："
                    f"不推荐全区间作为衰减窗口，包络参数仅作对照")
            elif rep.post_hump_windows == 1:
                rep.recommended_r_min = float(r[after_idx[0]])
                if rep.verdict == 'stable':
                    rep.verdict = 'moderate'
                rep.messages.append(
                    f"数据存在包络 hump（峰值 r≈{hump_peak:.6g}），推荐窗口起点设在 "
                    f"hump 之后；峰后仅 1 个合法窗口起点：峰后窗口可拟合，"
                    f"但稳定性无法评估（verdict 最多 moderate）")
            else:
                rep.recommended_r_min = float(r[after_idx[0]])
                rep.messages.append(
                    f"数据存在包络 hump（峰值 r≈{hump_peak:.6g}），"
                    f"推荐窗口起点设在 hump 之后以拟合衰减段"
                    f"（峰后 {len(after_idx)} 个合法窗口起点）")
        # 推荐窗口覆盖检查（安全网）：推荐起点必须在扫描网格内
        if (np.isfinite(rep.recommended_r_min)
                and not any(abs(row['r_min'] - rep.recommended_r_min) < 1e-9
                            for row in valid_rows)):
            rep.verdict = 'unstable'
            rep.messages.append("推荐窗口起点不在扫描网格内，稳定性无法评估")
        return rep

    # ----------------------------------------------------------------------
    # 3.6 综合判定（分类决策树）
    # ----------------------------------------------------------------------
    @staticmethod
    def _envelope_hump(r: np.ndarray, C_abs: np.ndarray) -> tuple:
        """包络 hump 检测（相对判据，不依赖噪声标尺——小动态范围数据下
        noise floor 可能与信号同量级，绝对阈值会失效）。
        判据：ln|C| 的局域斜率由正转负（峰值）＋ 峰后 ≥3 个点单调衰减
        ＋ 峰后 |斜率| 严格单调递增（近峰缓降 → 远端加速衰减的系统特征，
        噪声起伏的峰后斜率不具备此规律性）
        ＋ 峰相对**此前最近谷点**的上升 ≥ 5%（弱幅度下限，主判别依赖
        斜率形态而非单一幅度阈值——不同数据集的 hump 幅度 9%~16% 波动）
        ＋ 峰前上升段为 ≥3 段连续正斜率（系统趋势，防单段噪声毛刺）
        ＋ 峰不低于此前全局最大值的 50%（"主要结构"要求——拦截远端噪声
        小起伏被误判为 hump）。
        振荡数据的峰后斜率不会全为负，不会被误判。
        返回 (hump_detected, hump_peak_r)。"""
        r = np.asarray(r, dtype=float)
        C_abs = np.asarray(C_abs, dtype=float)
        ok = C_abs > 0
        if ok.sum() < 6:
            return (False, np.nan)
        rr = r[ok]
        yy = np.log(C_abs[ok])
        s = _local_slopes(rr, yy)
        n = len(rr)
        for i in range(1, n - 1):
            if s[i - 1] > 0 and s[i] < 0:          # 斜率由正转负 → 候选峰
                tail = s[i:]                        # 峰后所有斜率
                if len(tail) < 3 or np.any(tail > 0):
                    continue                        # 峰后不是单调衰减（振荡/噪声）
                if yy[i] < np.log(1.05) + np.min(yy[:i]):
                    continue                        # 峰相对此前最近谷点上升 < 5%
                if yy[i] < np.log(0.5) + np.max(yy[:i]):
                    continue                        # 峰低于此前全局最大值一半（非主要结构）
                # 峰后衰减加速：|斜率| 严格单调递增（系统衰减特征）
                if np.any(np.diff(np.abs(tail)) <= 0):
                    continue
                # 上升段长度：峰前连续正斜率段数 ≥3（系统趋势，防单段噪声毛刺）
                k = i - 1
                rise_seg = 0
                while k >= 0 and s[k] > 0:
                    rise_seg += 1
                    k -= 1
                if rise_seg < 3:
                    continue
                return (True, float(rr[i]))
        return (False, np.nan)

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
    @classmethod
    def classify(cls, components: dict, config: AnalyzerConfig = None) -> dict:
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
        # 窗口内无任何有效 |C| 数据（全零 / 全非正 / 全非有限）：log 空间与
        # 候选模型均无可拟合数据，直接判数据不足（避免下游对空样本拟合）
        if not np.any(np.isfinite(c_win)) or np.all(c_win <= 0):
            warnings.append("窗口内无有效 |C| 数据（全零/非正/非有限），无法分析")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}
        # 噪声主导门槛只对有符号变化的数据适用：
        # 无过零的恒定信号（constant 数据）其 |C| 天然位于自身尾部中位数附近，
        # 不能用"全部低于 noise floor×cut"误杀。
        n_cross_full = cls.detect_zero_crossings(prep.r[mask], prep.C[mask])[1]
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

        valid_non = {m: fits[m] for m in ('constant', 'power_law', 'exponential',
                                          'power_law_C', 'exponential_C')
                     if m in fits and fits[m] is not None and fits[m].valid}
        valid_osc = {m: fits[m] for m in ('osc_power', 'osc_exp',
                                          'osc_power_C', 'osc_exp_C')
                     if m in fits and fits[m] is not None and fits[m].valid}

        def _bg_note(fit, fit_dict=None):
            """带常数项模型胜出时的背景说明（仅胜出者含 C_inf 时调用）。
            显著性判据：|C_inf| 与该模型衰减部分在最远距离处的值比较
            （用噪声底判据会自相矛盾——平台数据本身会把噪声底抬高）。
            fit_dict 给非振荡分支用：两族 _C 模型互相竞争，真实的常数背景
            应被两族同号恢复；异号是噪声/模型误指定的伪背景标志。"""
            c = fit.params.get('C_inf', 0.0)
            ce = fit.param_errors.get('C_inf', np.nan)
            rmax = fit.window[1] if fit.window is not None else np.nan
            if 'K' in fit.params and np.isfinite(rmax):
                tail_model = fit.params['A'] * rmax ** (-fit.params['K'])
            elif 'xi' in fit.params and np.isfinite(rmax):
                tail_model = fit.params['A'] * np.exp(-rmax / fit.params['xi'])
            else:
                tail_model = np.nan
            # 退化守卫：衰减部分在窗口内实际为零（如 xi 触下界 → tail_model≈0）时，
            # "|C_inf| > 0.3×tail_model" 恒真，会把纯噪声常数项误报为显著背景
            A_fit = fit.params.get('A', np.nan)
            decay_alive = (np.isfinite(tail_model) and np.isfinite(A_fit)
                           and tail_model > 1e-6 * A_fit)
            # 跨形式符号一致性（仅非振荡分支）
            same_sign = True
            if fit_dict is not None:
                sib_name = ('exponential_C' if 'power' in fit.model
                            else 'power_law_C')
                sib = fit_dict.get(sib_name)
                same_sign = (sib is None or not sib.valid
                             or sib.params.get('C_inf', 0.0) * c >= 0)
            if decay_alive and same_sign and abs(c) > 0.3 * tail_model:
                warnings.append(f"带常数项模型胜出：C_inf = {_fmt_sci(c)} ± {_fmt_sci(ce)}，"
                                f"为最远距离衰减值的 {abs(c) / tail_model:.2f} 倍"
                                f"（{cls._BG_PHYSICS_NOTE}）")
            elif decay_alive and not same_sign:
                warnings.append(f"带常数项模型胜出但常数项与零一致"
                                f"（C_inf = {_fmt_sci(c)}，另一形式模型给出异号，符号冲突）")
            else:
                warnings.append(f"带常数项模型胜出但常数项与零一致（C_inf = {_fmt_sci(c)}）")

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
                # 振幅 A 或相位 phi 触及边界的振荡候选是边界退化解
                # （A 触界→振幅不可辨识；phi 钉在 ±π 配超大 A→A·cos 简并，
                # 模型靠相位补偿拟合，AICc 优势是数值巧合），不参与组内竞争
                contenders = {m: f for m, f in valid_osc.items()
                              if not ({'A', 'phi'} & set(f.bound_hits))}
                if not contenders:
                    warnings.append("振荡候选模型振幅全部触及边界（振幅不可辨识），"
                                    "采纳物理管线并降档")
                    classification = ('oscillatory_power_law' if env_form == 'power'
                                      else 'oscillatory_exponential')
                    return {'classification': classification, 'evidence_level': 'weak',
                            'nonmonotonic_reason': ''}
                winner = min(contenders, key=lambda m:
                             contenders[m].aicc if np.isfinite(contenders[m].aicc) else np.inf)
                if winner.replace('_C', '') == match:
                    classification = ('oscillatory_power_law' if match == 'osc_power'
                                      else 'oscillatory_exponential')
                    evidence = osc.evidence
                    if stability.verdict in ('unstable', 'insufficient'):
                        evidence = 'weak'
                        if stability.verdict == 'insufficient':
                            warnings.append("峰后衰减段数据不足（verdict=insufficient），"
                                            "evidence 降为 weak，包络参数仅作对照")
                        else:
                            warnings.append("稳定性 verdict=unstable，evidence 降为 weak")
                    if winner.endswith('_C'):
                        _bg_note(valid_osc[winner])
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
        form_switch = cls._form_switch(stability) and not flat
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
            if stability.verdict == 'insufficient':
                warnings.append("峰后衰减段点数不足（< n_min_hump_window），衰减参数证据不足，"
                                "全点包络仅作对照")
            return {'classification': 'non_monotonic', 'evidence_level': 'weak',
                    'nonmonotonic_reason': reason}
        if not valid_non:
            warnings.append("无有效非振荡候选模型")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}

        # ---- 非振荡组内模型选择（同空间同窗口，AICc 辅助证据） ----
        ranked = sorted(valid_non, key=lambda m:
                        valid_non[m].aicc if np.isfinite(valid_non[m].aicc) else np.inf)
        best = ranked[0]
        # 并列判定只在不同形式族之间进行：嵌套对（X vs X_C）在零背景下
        # ΔAICc 天然≈0，是同一模型族的多余参数而非"模型不可区分"，AICc 罚已足够
        _family = lambda m: m.replace('_C', '')
        second = next((m for m in ranked[1:] if _family(m) != _family(best)), None)
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
        if best in ('power_law', 'exponential', 'power_law_C', 'exponential_C'):
            best_base = best.replace('_C', '')
            if stability.verdict == 'unstable':
                warnings.append("分类依赖漂移参数且稳定性 verdict=unstable")
                return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                        'nonmonotonic_reason': ''}
            if daicc >= config.delta_aicc_tie:
                evidence = 'moderate' if daicc >= 4 else 'weak'
                if best.endswith('_C'):
                    _bg_note(valid_non[best], valid_non)
                return {'classification': best_base, 'evidence_level': evidence,
                        'nonmonotonic_reason': ''}
            if best.endswith('_C'):
                _bg_note(valid_non[best], valid_non)
            warnings.append(f"power_law 与 exponential 统计并列（ΔAICc={daicc:.3f} < "
                            f"{config.delta_aicc_tie}），不可区分")
            return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                    'nonmonotonic_reason': ''}
        return {'classification': 'INCONCLUSIVE', 'evidence_level': 'insufficient',
                'nonmonotonic_reason': ''}

    # ----------------------------------------------------------------------
    @staticmethod
    def print_compact_summary(result: DecayAnalysisResult) -> None:
        """简洁文本摘要（summary 模式）。只输出当前模型实际计算得到且有意义的
        参数，不适用字段直接省略，不填虚构值；纯消费已有 DecayAnalysisResult，
        不做任何重复拟合。"""
        env = result.envelope_fit
        osc = result.oscillation
        stab = result.stability
        print("=" * 60)
        print(f"衰减分析摘要 (analysis_version={result.analysis_version})")
        print("=" * 60)
        print(f"classification : {result.classification}")
        print(f"evidence       : {result.evidence_level}")
        if result.nonmonotonic_reason:
            print(f"nonmono_reason : {result.nonmonotonic_reason}")
        if env is not None and env.params:
            mode_used = (result.envelope_mode_used
                         or ('extrema' if '极值点' in env.message else 'all_points'))
            mode_tag = '[峰值法]' if mode_used == 'extrema' else '[全点]'
            if 'K' in env.params:
                print(f"envelope       : {mode_tag} power-law, K = "
                      f"{_fmt_sci(env.params['K'])} ± "
                      f"{_fmt_sci(env.param_errors.get('K', np.nan))} "
                      f"(R2_log = {env.r2:.4f})")
            elif 'xi' in env.params:
                print(f"envelope       : {mode_tag} exponential, ξ = "
                      f"{_fmt_sci(env.params['xi'])} ± "
                      f"{_fmt_sci(env.param_errors.get('xi', np.nan))} "
                      f"(R2_log = {env.r2:.4f})")
        # 仅在振荡有实际证据（非 none/inconclusive）时才输出振荡参数：
        # 无振荡数据上 Q 也会有数值解，输出会误导
        if (osc is not None and np.isfinite(osc.Q)
                and osc.evidence in ('strong', 'moderate', 'weak', 'possible')):
            print(f"oscillation    : Q = {_fmt_sci(osc.Q)}, phi = {_fmt_sci(osc.phi)}, "
                  f"R2_cos = {osc.r2_cos:.4f} (evidence: {osc.evidence})")
            if osc.structural_artifact_suspected:
                print("artifact       : possible structural/geometric artifact")
        if stab is not None and stab.verdict == 'insufficient':
            print("note           : 峰后衰减段数据不足，包络参数仅作对照（证据不足）")
        if np.isfinite(result.r_min):
            print(f"fit window     : r ∈ [{result.r_min:.6g}, {result.r_max:.6g}], "
                  f"n = {result.n_points} ({result.distance_aggregation})")
        if stab is not None and np.isfinite(stab.recommended_r_min):
            print(f"r_min stability: verdict = {stab.verdict}, "
                  f"recommended r_min = {stab.recommended_r_min:.6g}")
        if result.warnings:
            print("warnings       :")
            for w in result.warnings[:5]:
                print(f"  - {w}")
        print("=" * 60)

    # ----------------------------------------------------------------------
    @classmethod
    def print_summary(cls, result: DecayAnalysisResult) -> None:
        """打印分析摘要（含噪声底来源、8 项诊断检查逐项表、扫描表、artifact 标记、
        推荐窗口措辞；possible oscillation 信息保留）"""
        print("=" * 72)
        print(f"{cls._REPORT_TITLE} (analysis_version={result.analysis_version})")
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
        print(f"拟合点数            : {result.n_points}（原始 {result.n_raw_points} 个 "
              f"{cls._RAW_UNIT_DESC}，{result.n_distances} 个距离；{result.distance_aggregation}）")
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
            # 实际生效模式：优先用 result.envelope_mode_used（analyze 记录），
            # 兼容旧结果用 message 标记反推；退化 = 配置非 all_points 但无峰值法极值点
            mode_used = (result.envelope_mode_used
                         or ('extrema' if '极值点' in env.message else 'all_points'))
            degraded = (result.config.envelope_mode != 'all_points'
                        and len(result.envelope_extrema_r) == 0)
            mode_name = '峰值法 extrema' if mode_used == 'extrema' else '全点拟合 all_points'
            degrade_note = '——峰值法因极值点不足退化' if degraded else ''
            print(f"包络拟合（log 空间，envelope_mode={result.config.envelope_mode}"
                  f"，实际生效：{mode_name}{degrade_note}）:")
            if (result.stability is not None
                    and result.stability.verdict == 'insufficient'):
                print("  （峰后衰减段数据不足，上述包络参数为全区间拟合值，仅作对照）")
            print(f"  模型: {env.model}   R2_log = {env.r2:.6f}  "
                  f"RSS(log) = {env.rss:.3e}  AICc = {env.aicc:.3f}  BIC = {env.bic:.3f}")
            for name, val in env.params.items():
                # 参数显示名：关联长度用希腊字母 ξ（内部键名仍为 'xi'）
                disp = 'ξ' if name == 'xi' else name
                print(f"  {disp} = {_fmt_sci(val)} ± "
                      f"{_fmt_sci(env.param_errors.get(name, float('nan')))}")
            # envelope_mode='both' 的对比包络
            alt = result.envelope_fit_alt
            if alt is not None:
                kind_alt = '极值点(峰值法)' if '极值点' in alt.message else '全点'
                kind_env = '极值点(峰值法)' if '极值点' in env.message else '全点'
                print(f"  包络模式对比: 主包络[{kind_env}] vs 对比包络[{kind_alt}]")
                print(f"    {kind_alt}: {alt.model}  R2_log = {alt.r2:.6f}  "
                      f"AICc = {alt.aicc:.3f}  "
                      f"params = " + ", ".join(
                          f"{('ξ' if k == 'xi' else k)}={_fmt_sci(v)}"
                          for k, v in alt.params.items()))
                # 主包络获胜原因（both 规则：AICc 优者为主，并列时峰值法胜出）
                if np.isfinite(alt.aicc) and np.isfinite(env.aicc):
                    daicc = alt.aicc - env.aicc
                    if daicc > 1e-9:
                        print(f"    获胜原因: {kind_env} AICc 更优（ΔAICc = 对比 − 主 = {daicc:.3f}）")
                    else:
                        print(f"    获胜原因: AICc 并列（ΔAICc = {daicc:.3f}），"
                              f"按规则峰值法胜出")
                else:
                    print(f"    获胜原因: 对比包络 AICc 不可用，主包络为唯一有效结果")
        # 振荡
        osc = result.oscillation
        if osc is not None:
            print("-" * 72)
            print(f"振荡诊断（8 项 diagnostic checks，证据聚合）: evidence = {osc.evidence}")
            # 极值数口径：诊断计数为未筛选原始检出数；峰值法参与拟合的
            # 极值点经噪声底筛选（|C| > nf×prom_factor 及相邻落差），口径不同如实注明
            ext_line = f"  过零数 = {osc.n_cross}  极值数 = {osc.n_ext}"
            n_peak = len(result.envelope_extrema_r)
            excl = result.envelope_extrema_excluded
            if result.config.envelope_mode != 'all_points' and (n_peak > 0 or excl):
                if n_peak > 0:
                    ext_line += f"（峰值法参与拟合的极值点 n={n_peak}"
                    if excl:
                        ext_line += "，剔除的极值点：" + "、".join(
                            f"r={rr:.4g}（|C|={_fmt_sci(abs(vv))}，{reason}）"
                            for rr, vv, reason in excl)
                    ext_line += "）"
                else:
                    ext_line += "（峰值法退化：筛选后极值点 < 4"
                    if excl:
                        ext_line += "，剔除的极值点：" + "、".join(
                            f"r={rr:.4g}（|C|={_fmt_sci(abs(vv))}，{reason}）"
                            for rr, vv, reason in excl)
                    ext_line += "）"
            print(ext_line)
            print(f"  Q = {_fmt_sci(osc.Q)}  φ = {_fmt_sci(osc.phi)}  "
                  f"amplitude = {_fmt_sci(osc.amplitude)}  R2_cos = {osc.r2_cos:.6f}")
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
        # 物理形式（结构化多行；result.physical_form 字段保持单行以兼容 to_dict）
        form_str = (result.physical_form.split("≈ ", 1)[1]
                    if "≈ " in result.physical_form else "A·r^(−K)")
        p_env = result.envelope_fit.params
        err_env = result.envelope_fit.param_errors
        print(f"拟合形式：C(r) ≈ {form_str}")
        if 'A' in p_env:
            print(f"振幅因子：A = {_fmt_sci(p_env['A'])} ± "
                  f"{_fmt_sci(err_env.get('A', np.nan))}")
        if 'xi' in p_env:
            print(f"关联长度：ξ = {_fmt_sci(p_env['xi'])} ± "
                  f"{_fmt_sci(err_env.get('xi', np.nan))}")
        elif 'K' in p_env:
            print(f"衰减指数：K = {_fmt_sci(p_env['K'])} ± "
                  f"{_fmt_sci(err_env.get('K', np.nan))}")
        print("=" * 72)

    # ----------------------------------------------------------------------
    # 3.7 可视化（面板与组合图）
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
        ax.set_ylabel(r'$C(r)$', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_loglin(ax, r, C, result):
        """面板 2：|C| 半对数 + 指数包络（本面板固定检验 exp 形式：
        指数在半对数下为直线，数据偏离直线的程度即 exp 假设的失效程度）"""
        r = np.asarray(r, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        in_win = _window_mask(r, result.r_min, result.r_max)
        ax.semilogy(r, np.abs(C), 'o', color='tab:blue', ms=5, label='|C(r)|')
        # 固定对该窗口做 exp 形式拟合（无论最终选中的是哪种形式）
        if in_win.sum() >= 3:
            f_e = CorrelationDecayAnalyzer.fit_envelope_logspace(
                r[in_win], np.abs(C[in_win]), 'exp', result.config)
            rr = np.linspace(result.r_min, result.r_max, 200)
            ax.semilogy(rr, CorrelationDecayAnalyzer._envelope_value(f_e, rr),
                        'r-', linewidth=2,
                        label=f"exp fit: ξ={f_e.params.get('xi', float('nan')):.3g}, "
                              f"R2_log={f_e.r2:.3f}")
        # 峰值法参与拟合的极值点高亮（envelope_mode='extrema'/'both' 时非空）
        # 白底品红粗描边空心圆：与蓝色数据点、红色包络线均有强对比，且会覆盖住底下的线
        if len(result.envelope_extrema_r) > 0:
            ax.semilogy(result.envelope_extrema_r, result.envelope_extrema_abs,
                        'o', ms=9, mfc='white', mec='magenta', mew=2.0, zorder=6,
                        label=f'extrema in envelope fit (n={len(result.envelope_extrema_r)})')
        ax.axvspan(result.r_min, result.r_max, color='gray', alpha=0.12)
        ax.set_title(r'$|C(r)|$ Log-Linear (exponential check)', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel(r'$|C(r)|$', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(fontsize=9, loc='best')
        return ax

    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_panel_loglog(ax, r, C, result):
        """面板 3：|C| 双对数 + 幂律/指数两种包络同时比较。
        两种拟合共用同一有效点掩码与窗口（公平比较：相同样本、相同 log 空间
        目标函数、k=2 复杂度惩罚已含于 AICc）；比较结论复用
        fit_envelope_logspace 已计算的 FitResult.aicc，不引入新比较指标。"""
        r = np.asarray(r, dtype=float)
        if len(r) == 0:
            ax.text(0.5, 0.5, 'no data provided', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)
            return ax
        in_win = _window_mask(r, result.r_min, result.r_max)
        C_abs = np.abs(np.asarray(C, dtype=float))
        ax.loglog(r, C_abs, 'o', color='tab:blue', ms=5, label='|C(r)|')
        # 统一有效点筛选（同一掩码用于两种形式，保证公平比较）
        valid = in_win & (C_abs > 0) & np.isfinite(C_abs)
        n_drop = int(in_win.sum() - valid.sum())
        if valid.sum() >= 3:
            rv, cv = r[valid], C_abs[valid]
            f_p = CorrelationDecayAnalyzer.fit_envelope_logspace(
                rv, cv, 'power', result.config)
            f_e = CorrelationDecayAnalyzer.fit_envelope_logspace(
                rv, cv, 'exp', result.config)
            rr = np.linspace(result.r_min, result.r_max, 200)
            # 两种拟合曲线：power 红色实线、exp 蓝色虚线（颜色+线型双重区分）
            K = f_p.params.get('K', np.nan)
            K_ok = bool(np.isfinite(K) and K >= 0)
            if K_ok:
                ax.loglog(rr, CorrelationDecayAnalyzer._envelope_value(f_p, rr),
                          'r-', linewidth=2, label='power fit')
            xi = f_e.params.get('xi', np.nan)
            xi_ok = bool(np.isfinite(xi) and xi > 0)
            if xi_ok:
                ax.loglog(rr, CorrelationDecayAnalyzer._envelope_value(f_e, rr),
                          'b--', linewidth=2, label='exp fit')
            # 标注框（全部来自该次拟合的实际结果；无效参数如实标注，不伪造曲线）
            # 图内文字统一用英文：默认字体 DejaVu Sans 不含中文字形
            lines = []
            if K_ok:
                lines.append(f"K = {K:.4g} ± {f_p.param_errors.get('K', np.nan):.4g}"
                             f"   $R^2_{{log}}$ = {f_p.r2:.4f}")
            else:
                lines.append("power fit invalid (unidentifiable)")
            if xi_ok:
                lines.append(f"ξ = {xi:.4g} ± {f_e.param_errors.get('xi', np.nan):.4g}"
                             f"   $R^2_{{log}}$ = {f_e.r2:.4f}")
            else:
                lines.append("exp fit invalid (unidentifiable or ξ<=0)")
            # 模型比较（复用既有 AICc 口径；仅当两种形式都有效时才比较，
            # ΔAICc < delta_aicc_tie 视为统计并列；单侧无效时如实标注不参与比较）
            if K_ok and xi_ok and np.isfinite(f_p.aicc) and np.isfinite(f_e.aicc):
                d = f_p.aicc - f_e.aicc
                if abs(d) < result.config.delta_aicc_tie:
                    lines.append(f"AICc: power {f_p.aicc:.1f} vs exp {f_e.aicc:.1f}"
                                 " -> statistical tie")
                elif d < 0:
                    lines.append(f"AICc: power {f_p.aicc:.1f} vs exp {f_e.aicc:.1f}"
                                 f" -> power preferred (dAICc={-d:.2f})")
                else:
                    lines.append(f"AICc: power {f_p.aicc:.1f} vs exp {f_e.aicc:.1f}"
                                 f" -> exp preferred (dAICc={d:.2f})")
            elif K_ok and np.isfinite(f_p.aicc):
                lines.append(f"AICc: power {f_p.aicc:.1f} (exp invalid, excluded)")
            elif xi_ok and np.isfinite(f_e.aicc):
                lines.append(f"AICc: exp {f_e.aicc:.1f} (power invalid, excluded)")
            # 两种形式均无效时不输出比较行（避免对无效拟合给出误导性结论）
            lines.append(f"window: [{result.r_min:.6g}, {result.r_max:.6g}], "
                         f"n = {int(valid.sum())}")
            if n_drop > 0:
                lines.append(f"dropped {n_drop} non-positive/non-finite points")
            ax.text(0.03, 0.03, "\n".join(lines), transform=ax.transAxes,
                    fontsize=9.5, va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray',
                              alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'no valid positive data in window',
                    transform=ax.transAxes, ha='center', va='center', fontsize=13)
        # 峰值法参与拟合的极值点高亮（envelope_mode='extrema'/'both' 时非空）
        # 白底品红粗描边空心圆：与蓝色数据点、红色包络线均有强对比，且会覆盖住底下的线
        if len(result.envelope_extrema_r) > 0:
            ax.loglog(result.envelope_extrema_r, result.envelope_extrema_abs,
                      'o', ms=9, mfc='white', mec='magenta', mew=2.0, zorder=6,
                      label=f'extrema in envelope fit (n={len(result.envelope_extrema_r)})')
        ax.axvspan(result.r_min, result.r_max, color='gray', alpha=0.12)
        ax.set_title(r'$|C(r)|$ Log-Log (power vs exponential)', fontsize=13)
        ax.set_xlabel('distance r', fontsize=12)
        ax.set_ylabel(r'$|C(r)|$', fontsize=12)
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
            E = CorrelationDecayAnalyzer._envelope_value(env, r[in_win])
            y_det = C[in_win] / E
            ax.plot(r[in_win], y_det, 'o', color='tab:blue', ms=5, label='C(r)/E(r)')
            if np.isfinite(osc.Q):
                rr = np.linspace(result.r_min, result.r_max, 300)
                yhat = osc.amplitude * np.cos(osc.Q * rr + osc.phi)
                ax.plot(rr, yhat, 'r-', linewidth=2, label='cosine fit')
                # 振荡参数标注（来自 result.oscillation 的实际结果，无硬编码）
                txt = (f"Q = {osc.Q:.4g}\nphi = {osc.phi:.4g}\n"
                       f"$R^2$ = {osc.r2_cos:.4f}")
                ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=10,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray',
                                  alpha=0.85))
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
                  'osc_exp': 'tab:brown',
                  # _C 族用浅色变体区分（带常数项版本）
                  'power_law_C': 'lightgreen', 'exponential_C': 'moccasin',
                  'osc_power_C': 'plum', 'osc_exp_C': 'lightcoral'}
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
    @classmethod
    def Decay_Visualization(cls, result: DecayAnalysisResult, r: np.ndarray = None,
                            C: np.ndarray = None, show: bool = True,
                            save_path: str = None,
                            visualization_mode: str = 'standard',
                            title: str = ''):
        """按展示级别生成综合分析图。r/C 为预处理后的数据（同距离平均序列，
        可由 group_by_distance 获得）。visualization_mode 可选：
        - 'standard'（默认）：2×2 四核心面板（signed C(r) / |C| log-log / detrended / r_min scan）
        - 'diagnostic'：1×3 辅助诊断面板（log-lin 指数检验 / 候选模型比较 / 残差）
        - 'summary'：不画图，打印简洁文本摘要并返回 None
        - 'full'：原有 8 面板完整图（兼容旧调用）
        title：主标题前缀标注（如键方向 "A key (blue bond)"），空串则不显示
        """
        if visualization_mode not in ('summary', 'standard', 'diagnostic', 'full'):
            raise ValueError(
                f"不支持的 visualization_mode='{visualization_mode}'，"
                "可选 'summary' | 'standard' | 'diagnostic' | 'full'")
        if visualization_mode == 'summary':
            cls.print_compact_summary(result)
            return None
        if r is None or C is None:
            r = np.array([])
            C = np.array([])
        title_prefix = f"{title} — " if title else ""
        if visualization_mode == 'standard':
            fig = plt.figure(figsize=(16, 11))
            gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
            fig.suptitle(
                f'{title_prefix}{cls._SUPTITLE_MAIN} — '
                f'classification: {result.classification} (evidence: {result.evidence_level})',
                fontsize=18, fontweight='bold')
            cls._draw_panel_signed(fig.add_subplot(gs[0, 0]), r, C, result)
            cls._draw_panel_loglog(fig.add_subplot(gs[0, 1]), r, C, result)
            cls._draw_panel_detrended(fig.add_subplot(gs[1, 0]), r, C, result)
            cls._draw_panel_scan(fig.add_subplot(gs[1, 1]), result)
        elif visualization_mode == 'diagnostic':
            fig = plt.figure(figsize=(20, 7))
            gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.28)
            fig.suptitle(
                f'{title_prefix}Decay Analysis Diagnostics — '
                f'classification: {result.classification} '
                f'(evidence: {result.evidence_level})',
                fontsize=16, fontweight='bold')
            cls._draw_panel_loglin(fig.add_subplot(gs[0, 0]), r, C, result)
            cls._draw_panel_models(fig.add_subplot(gs[0, 1]), r, C, result)
            cls._draw_panel_residuals(fig.add_subplot(gs[0, 2]), r, C, result)
        else:   # 'full'：保留原 8 面板完整图
            fig = plt.figure(figsize=(28, 16))
            gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.32)
            fig.suptitle(
                f'{title_prefix}{cls._SUPTITLE_MAIN} — '
                f'classification: {result.classification} (evidence: {result.evidence_level})',
                fontsize=24, fontweight='bold')
            cls._draw_panel_signed(fig.add_subplot(gs[0, 0]), r, C, result)
            cls._draw_panel_loglin(fig.add_subplot(gs[0, 1]), r, C, result)
            cls._draw_panel_loglog(fig.add_subplot(gs[0, 2]), r, C, result)
            cls._draw_panel_detrended(fig.add_subplot(gs[0, 3]), r, C, result)
            cls._draw_panel_models(fig.add_subplot(gs[1, 0]), r, C, result)
            cls._draw_panel_residuals(fig.add_subplot(gs[1, 1]), r, C, result)
            cls._draw_panel_scan(fig.add_subplot(gs[1, 2]), result)
            cls._draw_panel_summary(fig.add_subplot(gs[1, 3]), result)
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    # ----------------------------------------------------------------------
    @classmethod
    def Decay_Visualization_compare(cls, results: dict, show: bool = True,
                                    save_path: str = None) -> plt.Figure:
        """批量对比图：左面板叠加各组 |C|(r)+包络，右面板 K/ξ 与分类对比表。
        仅并排展示供人工比较，不自动做跨数据集物理结论。"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(cls._COMPARE_SUPTITLE, fontsize=18, fontweight='bold')
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

    # ----------------------------------------------------------------------
    # 3.8 完整分析管线（从预处理结果出发；record 输入适配由子类完成）
    # ----------------------------------------------------------------------
    @classmethod
    def analyze_generic(cls, prep: PreparedData,
                        config: AnalyzerConfig = None) -> DecayAnalysisResult:
        """通用完整分析管线：从 PreparedData 出发执行 hump 检测 → r_min 扫描
        → 窗口选择 → 包络拟合 → 族分解 → 振荡诊断 → 候选模型 → 组内比较
        → 分类 → 物理形式，输出 DecayAnalysisResult。
        数据来源适配（record → PreparedData）由子类的 analyze 负责。"""
        config = config or AnalyzerConfig()
        warnings = list(prep.warnings)
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

        # 包络 hump 检测（全距离范围，先于窗口选择；相对判据，不依赖噪声标尺）
        mask_all = _window_mask(prep.r, 0.0, prep.r_max_used)
        hump, hump_peak = cls._envelope_hump(prep.r[mask_all], prep.C_abs[mask_all])

        # r_min 扫描（窗口选择标准始终基于全点包络，最终包络按 envelope_mode 计算）
        stability = cls.scan_r_min(
            prep.r, prep.C, config, noise_floor=prep.final_noise_floor,
            hump_peak=hump_peak if hump else None)
        warnings.extend(stability.messages)
        result.stability = stability
        r_min = (config.r_min_override
                 if config.r_min_override is not None
                 else (stability.recommended_r_min
                       if np.isfinite(stability.recommended_r_min)
                       else float(np.min(prep.r))))
        if config.r_min_override is not None:
            warnings.append(f"使用手动指定拟合窗口起点 r_min={config.r_min_override}（覆盖自动推荐）")
        mask = _window_mask(prep.r, r_min, prep.r_max_used)
        r_win, C_win = prep.r[mask], prep.C[mask]
        result.r_min, result.r_max = r_min, prep.r_max_used
        result.n_points = len(r_win)

        # 包络拟合（按 envelope_mode：全点 / 峰值法 / 两者对比）
        envelope_fit, envelope_fit_alt, env_warnings, ext_data = \
            cls._fit_envelope_with_mode(r_win, C_win, prep.final_noise_floor, config)
        result.envelope_fit = envelope_fit
        result.envelope_fit_alt = envelope_fit_alt
        (result.envelope_extrema_r, result.envelope_extrema_abs,
         result.envelope_extrema_excluded) = ext_data
        result.envelope_mode_used = ('extrema' if '极值点' in envelope_fit.message
                                     else 'all_points')
        warnings.extend(env_warnings)
        if envelope_fit.message:
            warnings.append(f"包络拟合: {envelope_fit.message}")
        if (envelope_fit_alt is not None and envelope_fit_alt.valid
                and envelope_fit_alt.model != envelope_fit.model):
            warnings.append("两种包络模式的形式不一致（全点 vs 峰值法），"
                            "包络形式选择存在方法学依赖，结论需谨慎")

        # 距离族分解
        family_diag = cls.family_separated_diagnostic(prep, envelope_fit, config)
        result.family_diagnostics = family_diag

        # 振荡诊断
        osc = cls.analyze_oscillation(r_win, C_win, envelope_fit, family_diag, config)
        result.oscillation = osc
        if osc.structural_warning:
            warnings.append(osc.structural_warning)

        # 候选模型 + 组内比较
        candidate_fits = cls.fit_candidate_models(r_win, C_win, envelope_fit, osc, config)
        result.candidate_fits = candidate_fits
        result.comparison = cls.compare_models(candidate_fits, config)
        for name, fit in candidate_fits.items():
            if fit is not None and (not fit.valid or fit.message):
                warnings.append(f"候选模型 {name}: {'无效' if not fit.valid else ''}"
                                f"{fit.message}")

        # 去包络 cosine 拟合结果（detrended_fit）
        E = cls._envelope_value(envelope_fit, r_win)
        y_det = C_win / E
        det = FitResult(model='cosine_detrended', fit_space='linear_detrended',
                        r2_space='linear', params={'Q': osc.Q, 'phi': osc.phi,
                                                   'amplitude': osc.amplitude},
                        n=len(y_det), k=2, r2=osc.r2_cos,
                        window=(r_min, prep.r_max_used))
        result.detrended_fit = det

        # 分类决策树（hump 已在全距离范围上检测）
        dual = cls._dual_plateau(stability, config)
        verdict = cls.classify(
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
