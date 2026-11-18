# -*- coding: utf-8 -*-
"""
d+id 配对相位分析器（叶子模块，只依赖 numpy 与 Honeycomb_Lattice 几何）。

物理口径（依据 docs/DPlusID_PhaseAnalyzer_Physics_Audit.md，2026-11-18）：
- 键方向 (A, B, C) 由 HoneycombXCCylinder::ClassifyNearestNeighbor 定义；
  XC 圆柱坐标下 C 键沿圆柱轴（x 轴，θ=0°），A/B 键与 x 轴夹角 ±120°。
- d-wave 基函数（键序 (C, A, B)）：
    d1 = (2Δ_C - Δ_A - Δ_B)/√6      (d_{x²-y²} 取向)
    d2 = (Δ_A - Δ_B)/√2             (d_{xy} 取向)
- 判据：理想 d+id 下 arg C_12(r) ≈ ±90° 且 C_11 ≈ C_22；
  实值基态（或虚部机器零）下 arg C_12 ∈ {0°, 180°}。
- 手性指标 chi_d = 2 Im C_12 / (C_11 + C_22)（辅助指标，分母有效性检查）；
  R_12 = C_12/√(C_11 C_22) 在 d2 通道受抑制（C_22≈0）时数值发散，不作为主判据。

数据要求：三个 refbond（键方向 A/B/C 各一）× 三个 target 键方向 ×
四通道（a/b/c/d）的**全量**解析数据（不经衰减分析的纵向通道过滤）。
每条目为 (target_bond=(k,l), value=complex)，四通道合成
<Δ†Δ> = 0.5(a - b - c + d) 由调用方或 assemble_* 完成。

分类语义（五态）：
- chiral_phase_candidate：存在可靠的 ±90° 相位平台（幅值门控、
  距离稳定性、bin 容差稳定性、D 档一致性全部通过）
- no_chiral_signal：有效关联数据充足但未观察到稳定的 ±90° 平台
- d_component_suppressed：一个 d-wave 通道（实测为 d2）明显受抑制，
  无法支持两个分量共同形成 d+id（优先级高于 no_chiral_signal）
- insufficient：有效关联幅值不足或有效距离点不足
- inconclusive：不同 D 档 / bin 容差 / 关联诊断结果相互矛盾

阈值均为候选默认值（defaults_not_authorized），未获物理授权；
只作数值分析工具，不输出物理相态判定。
"""
import ast
from dataclasses import dataclass, field

import numpy as np

from ..data_records.Honeycomb_Lattice import HoneycombXCGeometry

# 键方向顺序（基函数定义序）：d1=(2C-A-B)/√6、d2=(A-B)/√2
KEY_ORDER = ('C', 'A', 'B')
V_D1 = np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0)
V_D2 = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)


# ==========================================================================
# 配置与结果契约
# ==========================================================================
@dataclass
class PhaseAnalyzerConfig:
    """相位分析配置。全部阈值为候选默认值（未获物理授权）。"""
    tol_bins: tuple = (1.5, 2.0, 2.5)   # 多容差稳定性比较；主容差用 primary_tol
    primary_tol: float = 2.0            # 主距离 bin 容差
    phase_tolerance_deg: float = 15.0   # 平台相位偏离目标（±90°）容差
    circ_std_max_deg: float = 15.0      # 平台窗口内圆周标准差上限
    min_platform_points: int = 3        # 平台最小连续有效距离点数
    noise_prom_factor: float = 3.0      # 幅值门控：|C12| > noise_floor × 此因子
    tail_frac_noise: float = 0.25       # 噪声底：远距尾部中位数（|C12| 序列）
    noise_abs_min: float = 1e-15        # 噪声底绝对下限
    d2_suppression_ratio: float = 1e-6  # 中位 |C22|/|C11| 低于此 → d2 显著弱于 d1
    d2_unresolved_ratio: float = 1e-8   # 比值低于此 → 低于数值分辨能力
                                        # （经验分辨比，相对双精度有安全边际；
                                        # 未获物理授权）
    d2_suppression_n: int = 3           # 抑制判定所需的有效点数
    conjugate_tol: float = 1e-6         # 同 bin 跨通道偏差描述性阈值
                                        # （无物理共轭恒等式依据，不参与分类）
    # 相消诊断：bin 内复均值模长与 |均值| 之比（1=无相消，0=完全相消）
    cancellation_min_ratio: float = 0.5
    # 圆周均值有效性：resultant length R 低于此值 → 均值显示 n/a
    circ_mean_min_R: float = 0.5


@dataclass
class BinEntry:
    """单个 (α,β) 组合在某个距离 bin 上的统计。"""
    alpha: str
    beta: str
    r_mean: float
    r_std: float
    n_raw: int
    value: complex                # bin 内复数平均
    mean_abs: float               # bin 内 |value| 的平均
    scatter: float                # bin 内复数离散度（标准差，按实部+虚部）
    cancellation: float           # |复均值| / mean_abs（<1 表示相消）


@dataclass
class PlatformReport:
    """单个目标相位（+90° / -90°）的候选平台报告。"""
    target_deg: float = 90.0
    found: bool = False
    n_points: int = 0
    r_range: tuple = (np.nan, np.nan)
    circ_mean_deg: float = np.nan
    circ_std_deg: float = np.nan
    max_deviation_deg: float = np.nan
    points: list = field(default_factory=list)   # [(r, phi_deg)]


@dataclass
class PhaseAnalysisResult:
    """相位分析结果（可 to_dict，含 config 快照）。"""
    classification: str = ''
    status: str = 'ok'            # ok / error
    reasons: list = field(default_factory=list)
    # 主容差结果
    n_bins: int = 0
    n_valid_bins: int = 0
    distances: list = field(default_factory=list)
    C11: list = field(default_factory=list)
    C22: list = field(default_factory=list)
    C12: list = field(default_factory=list)     # 复数
    C21: list = field(default_factory=list)
    phi12_deg: list = field(default_factory=list)
    chi_d: list = field(default_factory=list)
    amp_gate: list = field(default_factory=list)      # bool：幅值门控通过
    valid: list = field(default_factory=list)          # bool：有效点（门控+非退化）
    noise_floor: float = np.nan
    noise_counts: dict = field(default_factory=dict)   # total/zero/nonzero
    circular_mean_deg: float = np.nan   # R 低于阈值的弥散分布下为 nan
    circular_std_deg: float = np.nan
    circular_R: float = np.nan
    circular_n: int = 0
    circular_mean_invalid_reason: str = ''
    conjugate_diagnostics: dict = field(default_factory=dict)
    platform_plus: PlatformReport = field(default_factory=PlatformReport)
    platform_minus: PlatformReport = field(default_factory=PlatformReport)
    d2_ratio_median: float = np.nan
    d2_resolution: str = ''    # 'resolved' | 'unresolved' | 'insufficient'
    # 稳定性检查
    tol_stability: dict = field(default_factory=dict)   # {tol: classification}
    bins_detail: list = field(default_factory=list)     # 主容差逐 bin 明细
    config: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'classification': self.classification,
            'status': self.status,
            'reasons': list(self.reasons),
            'n_bins': self.n_bins,
            'n_valid_bins': self.n_valid_bins,
            'distances': [float(x) for x in self.distances],
            'C11': [complex(v) for v in self.C11],
            'C22': [complex(v) for v in self.C22],
            'C12': [complex(v) for v in self.C12],
            'C21': [complex(v) for v in self.C21],
            'phi12_deg': [float(x) for x in self.phi12_deg],
            'chi_d': [float(x) for x in self.chi_d],
            'amp_gate': [bool(x) for x in self.amp_gate],
            'valid': [bool(x) for x in self.valid],
            'noise_floor': float(self.noise_floor),
            'noise_counts': dict(self.noise_counts),
            'circular_mean_deg': float(self.circular_mean_deg),
            'circular_std_deg': float(self.circular_std_deg),
            'circular_R': float(self.circular_R),
            'circular_n': int(self.circular_n),
            'circular_mean_invalid_reason': self.circular_mean_invalid_reason,
            'conjugate_diagnostics': dict(self.conjugate_diagnostics),
            'platform_plus': self.platform_plus.__dict__,
            'platform_minus': self.platform_minus.__dict__,
            'd2_ratio_median': float(self.d2_ratio_median),
            'd2_resolution': self.d2_resolution,
            'tol_stability': {str(k): v for k, v in self.tol_stability.items()},
            'bins_detail': [dict(b) for b in self.bins_detail],
            'config': dict(self.config),
        }


# ==========================================================================
# 几何与解析
# ==========================================================================
def refbond_direction(i: int, j: int, ly: int = 4) -> str:
    """XC 圆柱最近邻键方向分类（对应 HoneycombXCCylinder::
    ClassifyNearestNeighbor；与 Super_Correlation_Process._refbond_direction
    一致，此处内嵌避免跨模块依赖）。"""
    if i > j:
        i, j = j, i
    x0, y0 = divmod(i, ly)
    x1, y1 = divmod(j, ly)
    if x0 == x1:
        y_min = y0 if (y0 + ly - y1) % ly == 1 else y1
        return 'A' if (y_min + x0) % 2 == 0 else 'B'
    return 'C'


def parse_full_sc(data_text: str) -> list:
    """全量解析超导关联 JSON 文本（不过滤 target 键），
    返回 [(target_bond=(k,l), value=complex)]。"""
    data_list = ast.literal_eval(data_text)
    out = []
    for element in data_list:
        [i, j, k, l], [re_part, im_part] = element
        out.append(((k, l), complex(float(re_part), float(im_part))))
    return out


def combine_channels(a: list, b: list, c: list, d: list) -> list:
    """四通道合成 <Δ†Δ> = 0.5(a - b - c + d)（复数，逐键）。
    输入均为 parse_full_sc 输出（键集合一致）。"""
    ad = dict(a)
    return [((k, l), 0.5 * (ad[(k, l)] - dict(b)[(k, l)]
                           - dict(c)[(k, l)] + dict(d)[(k, l)]))
            for (k, l), _ in a]


def assemble_datasets(files_by_refbond: dict, N: int) -> dict:
    """从原始文件路径装配三 refbond 数据。

    files_by_refbond = {refbond_tuple: {bond_dir: {channel: file_path}}}
    （12 文件/refbond）。返回
    {refbond_tuple: {alpha: [(target_bond, complex_combined)]}}，
    并记录 refbond 键方向到 results['_ref_keys']。
    """
    out = {}
    ref_keys = {}
    for rb, by_dir in files_by_refbond.items():
        rk = refbond_direction(*rb)
        ref_keys[rb] = rk
        out[rb] = {}
        for alpha in 'ABC':
            ch = {k: open(v, encoding='utf-8').read()
                  for k, v in by_dir[alpha].items()}
            parsed = {k: parse_full_sc(txt) for k, txt in ch.items()}
            out[rb][alpha] = combine_channels(
                parsed['a'], parsed['b'], parsed['c'], parsed['d'])
    out['_ref_keys'] = ref_keys
    return out


# ==========================================================================
# 核心管线
# ==========================================================================
class SCPhaseAnalyzer:
    """d+id 相位分析核心（全静态方法，可独立测试）。"""

    # ------------------------------------------------------------------
    @staticmethod
    def _distance(rb, tb, N, geo):
        return geo.sup_corr_distance(rb[0], rb[1], tb[0], tb[1])

    @staticmethod
    def _bin_entries(datasets, rb, alpha, N, geo, tol) -> dict:
        """单 (rb, alpha) 组合按距离容差分 bin → {bin_id: BinEntry(部分)}。"""
        entries = datasets[rb][alpha]
        bins = {}
        for tb, value in entries:
            d = SCPhaseAnalyzer._distance(rb, tb, N, geo)
            bid = round(d / tol)
            b = bins.setdefault(bid, {'ds': [], 'vals': []})
            b['ds'].append(d)
            b['vals'].append(value)
        out = {}
        for bid, b in bins.items():
            vals = np.array(b['vals'])
            abses = np.abs(vals)
            mean = float(np.mean(vals.real)) + 1j * float(np.mean(vals.imag))
            mean_abs = float(np.mean(abses))
            scatter = float(np.std(vals.real) + np.std(vals.imag))
            out[bid] = BinEntry(
                alpha=alpha, beta=None, r_mean=float(np.mean(b['ds'])),
                r_std=float(np.std(b['ds'])), n_raw=len(b['ds']),
                value=mean, mean_abs=mean_abs, scatter=scatter,
                cancellation=(abs(mean) / mean_abs if mean_abs > 0 else 0.0))
        return out

    @staticmethod
    def estimate_noise_floor(mod_series, config: PhaseAnalyzerConfig) -> float:
        """幅值序列的噪声底：远距尾部中位数与绝对下限取最大
        （与衰减分析同款口径）。零值点不进尾部统计——序列含大量
        精确零（正交组合）时用非零值的尾部量级，避免噪声底坍缩到 0
        使门控形同虚设。"""
        mods = np.asarray([m for m in mod_series if np.isfinite(m) and m > 0],
                          dtype=float)
        if mods.size == 0:
            return config.noise_abs_min
        n_tail = max(1, int(np.ceil(mods.size * config.tail_frac_noise)))
        tail_median = float(np.median(mods[-n_tail:]))
        return max(tail_median, config.noise_abs_min)

    @staticmethod
    def circular_stats(angles_deg) -> tuple:
        """圆周统计：返回 (mean_deg, std_deg, R, n)。
        R = resultant length（合成向量模长/点数，1=完全聚集、0=完全
        弥散）；空输入返回 (nan, nan, nan, 0)。"""
        ang = np.asarray(angles_deg, dtype=float) * np.pi / 180.0
        ang = ang[np.isfinite(ang)]
        if ang.size == 0:
            return (np.nan, np.nan, np.nan, 0)
        s, c = np.sum(np.sin(ang)), np.sum(np.cos(ang))
        R = np.hypot(s, c) / ang.size
        mean = float(np.degrees(np.arctan2(s, c)))
        std = float(np.degrees(np.sqrt(-2.0 * np.log(R)))) if R > 0 else np.inf
        return (mean, std, float(R), int(ang.size))

    @staticmethod
    def detect_platform(distances, phi_deg, valid, target_deg,
                        config: PhaseAnalyzerConfig) -> PlatformReport:
        """在有效点上检测接近 target_deg 的连续平台。"""
        rep = PlatformReport(target_deg=target_deg)
        runs = []
        cur = []
        for d, p, v in zip(distances, phi_deg, valid):
            if not v:
                cur = []
                continue
            dev = abs((p - target_deg + 180.0) % 360.0 - 180.0)
            if dev <= config.phase_tolerance_deg:
                cur.append((d, p))
            else:
                if len(cur) >= config.min_platform_points:
                    runs.append(cur)
                cur = []
        if len(cur) >= config.min_platform_points:
            runs.append(cur)
        if not runs:
            return rep
        best = max(runs, key=len)
        c_mean, c_std, _, _ = SCPhaseAnalyzer.circular_stats([p for _, p in best])
        devs = [abs((p - target_deg + 180.0) % 360.0 - 180.0) for _, p in best]
        ok = (c_std <= config.circ_std_max_deg
              and max(devs) <= config.phase_tolerance_deg)
        rep.found = ok
        rep.n_points = len(best)
        rep.r_range = (best[0][0], best[-1][0])
        rep.circ_mean_deg = c_mean
        rep.circ_std_deg = c_std
        rep.max_deviation_deg = max(devs)
        rep.points = [(float(d), float(p)) for d, p in best]
        return rep

    # ------------------------------------------------------------------
    @staticmethod
    def analyze(datasets: dict, N: int = 200, config: PhaseAnalyzerConfig = None
                ) -> PhaseAnalysisResult:
        """完整相位分析管线。

        datasets：assemble_datasets 的输出（含 '_ref_keys'）；
        或等价结构 {refbond: {alpha: [(target_bond, complex)]}}，
        refbond 的键方向必须覆盖 A/B/C 各一（由 refbond_direction 判定）。
        """
        config = config or PhaseAnalyzerConfig()
        result = PhaseAnalysisResult(config=config.__dict__)
        ref_keys = datasets.pop('_ref_keys', {})
        geo = HoneycombXCGeometry(N, 1.0)

        # ---- 1. 距离 bin（主容差）----
        binned = {}    # {(alpha, beta): {bin_id: BinEntry}}
        common = None
        for rb, by_alpha in datasets.items():
            if isinstance(rb, str) and rb.startswith('_'):
                continue   # 跳过元数据键
            beta = ref_keys.get(rb) or refbond_direction(*rb)
            for alpha in 'ABC':
                key = (alpha, beta)
                binned[key] = SCPhaseAnalyzer._bin_entries(
                    datasets, rb, alpha, N, geo, config.primary_tol)
                for bid in binned[key]:
                    binned[key][bid].beta = beta
                common = (set(binned[key]) if common is None
                          else common & set(binned[key]))
        if not common:
            result.status = 'error'
            result.reasons.append('9 个 (α,β) 组合无公共距离 bin（数据对齐失败）')
            result.classification = 'inconclusive'
            return result

        # ---- 2. 逐 bin 3×3 矩阵与 d-wave 投影 ----
        distances = sorted(bin_center
                           for bin_center in [config.primary_tol * b for b in sorted(common)])
        C = {'11': [], '22': [], '12': [], '21': []}
        detail = []
        for b in sorted(common):
            mat = np.zeros((3, 3), dtype=complex)
            for ia, a in enumerate(KEY_ORDER):
                for ib, bb in enumerate(KEY_ORDER):
                    mat[ia, ib] = binned[(a, bb)][b].value
            c11 = V_D1 @ mat @ V_D1
            c22 = V_D2 @ mat @ V_D2
            c12 = V_D1 @ mat @ V_D2
            c21 = V_D2 @ mat @ V_D1
            C['11'].append(c11)
            C['22'].append(c22)
            C['12'].append(c12)
            C['21'].append(c21)
            # bin 明细（含相消与离散度诊断）
            row = {'r': config.primary_tol * b}
            for key, e in binned.items():
                row[f'{key[0]}{key[1]}_n'] = e[b].n_raw
                row[f'{key[0]}{key[1]}_canc'] = e[b].cancellation
                row[f'{key[0]}{key[1]}_scat'] = e[b].scatter
            row['r_std'] = float(np.mean(
                [binned[k][b].r_std for k in binned]))
            row['min_cancellation'] = float(min(
                binned[k][b].cancellation for k in binned))
            row['min_n_raw'] = int(min(
                binned[k][b].n_raw for k in binned))
            detail.append(row)

        c11 = np.array(C['11']); c22 = np.array(C['22'])
        c12 = np.array(C['12']); c21 = np.array(C['21'])

        # ---- 4. 幅值门控（|C12| 对噪声底）----
        noise_floor = SCPhaseAnalyzer.estimate_noise_floor(
            np.abs(c12), config)
        amp_gate = [abs(v) > noise_floor * config.noise_prom_factor
                    for v in c12]

        # ---- 3. 同 bin 跨通道偏差诊断（描述性，不参与分类）----
        # 无物理共轭恒等式依据：有限 XC 圆柱（开放边界、refbond 错位）
        # 下同一距离 bin 的 C12 与 conj(C21) 不构成共轭对应坐标；
        # 该指标仅描述两个通道组合在同一 bin 上的数值偏差。
        both_ok = [(abs(a) > noise_floor * config.noise_prom_factor)
                   and (abs(b) > noise_floor * config.noise_prom_factor)
                   for a, b in zip(c12, c21)]
        conj_dev = np.nan
        if any(both_ok):
            devs = [abs(a - np.conjugate(b)) / max(abs(a), abs(b))
                    for a, b, ok in zip(c12, c21, both_ok) if ok]
            conj_dev = float(np.max(devs)) if devs else np.nan
        result.conjugate_diagnostics = {
            'max_relative_deviation': conj_dev,
            'ok': (bool(conj_dev <= config.conjugate_tol)
                   if np.isfinite(conj_dev) else None),
            'semantics': ('same-bin cross-channel deviation（描述性）；'
                          '有限圆柱上无同距离共轭恒等式依据，'
                          '不参与手性判据与数据正确性证明'),
        }
        # 分母有效性（chi_d 用）
        denom_ok = [abs(v) > 1e-30 for v in (c11 + c22)]
        # bin 相消有效性：9 个组合中最严重的相消低于阈值 → bin 无效
        cancel_ok = []
        for row in detail:
            canc = row.get('min_cancellation', 1.0)
            n_min = row.get('min_n_raw', 0)
            if n_min >= 2 and canc < config.cancellation_min_ratio:
                cancel_ok.append(False)
            else:
                cancel_ok.append(True)
        valid = [g and d and c
                 for g, d, c in zip(amp_gate, denom_ok, cancel_ok)]

        phi12 = [float(np.degrees(np.angle(v))) for v in c12]
        chi = [float(2 * v.imag / w.real) if abs(w) > 1e-30 else np.nan
               for v, w in zip(c12, c11 + c22)]

        # ---- 5. 圆周统计（有效点）----
        valid_phi = [p for p, v in zip(phi12, valid) if v]
        c_mean, c_std, c_R, c_n = SCPhaseAnalyzer.circular_stats(valid_phi)
        mean_invalid = ''
        if np.isfinite(c_R) and c_R < config.circ_mean_min_R:
            mean_invalid = (f'resultant length R={c_R:.3f} < '
                            f'{config.circ_mean_min_R}，相位弥散，'
                            '圆周均值无统计意义')
            c_mean = np.nan

        # ---- 6. 平台检测 ----
        plat_p = SCPhaseAnalyzer.detect_platform(
            distances, phi12, valid, +90.0, config)
        plat_m = SCPhaseAnalyzer.detect_platform(
            distances, phi12, valid, -90.0, config)

        # ---- 7. d2 通道诊断（基于 C11 可靠性，不依赖 C12 门控）----
        # 分辨能力三层区分（双阈值比值口径）：
        #   resolved：比值 ≥ d2_unresolved_ratio（高于经验分辨比）
        #   unresolved：比值 < d2_unresolved_ratio → 低于数值分辨能力，
        #               物理抑制与数值零无法区分
        #   insufficient：C11 有效点不足，无法比较
        nf11 = SCPhaseAnalyzer.estimate_noise_floor(np.abs(c11), config)
        c11_valid = [abs(v) > nf11 * config.noise_prom_factor for v in c11]
        r22 = [abs(x) / abs(y) for x, y, v in zip(c22, c11, c11_valid) if v]
        d2_ratio = float(np.median(r22)) if r22 else np.nan
        if int(np.sum(c11_valid)) < config.d2_suppression_n:
            d2_resolution = 'insufficient'
        elif np.isfinite(d2_ratio) and d2_ratio < config.d2_unresolved_ratio:
            d2_resolution = 'unresolved'
        else:
            d2_resolution = 'resolved'

        # ---- 8. 多容差稳定性 ----
        tol_stability = {}
        for tol in config.tol_bins:
            if tol == config.primary_tol:
                continue
            tol_stability[str(tol)] = SCPhaseAnalyzer._analyze_single_tol(
                datasets, ref_keys, N, geo, tol, config)

        # ---- 9. 填充结果 ----
        result.n_bins = len(distances)
        result.n_valid_bins = int(np.sum(valid))
        result.distances = [float(d) for d in distances]
        result.C11 = [complex(v) for v in c11]
        result.C22 = [complex(v) for v in c22]
        result.C12 = [complex(v) for v in c12]
        result.C21 = [complex(v) for v in c21]
        result.phi12_deg = phi12
        result.chi_d = chi
        result.amp_gate = list(amp_gate)
        result.valid = list(valid)
        result.noise_floor = float(noise_floor)
        abs_c12 = np.abs(c12)
        result.noise_counts = {
            'total': int(abs_c12.size),
            'zero': int(np.sum(abs_c12 == 0.0)),
            'nonzero': int(np.sum(abs_c12 > 0.0)),
        }
        result.circular_mean_deg = c_mean
        result.circular_std_deg = c_std
        result.circular_R = c_R
        result.circular_n = c_n
        result.circular_mean_invalid_reason = mean_invalid
        result.platform_plus = plat_p
        result.platform_minus = plat_m
        result.d2_ratio_median = d2_ratio
        result.d2_resolution = d2_resolution
        result.tol_stability = tol_stability
        result.bins_detail = detail

        # ---- 10. 分类 ----
        SCPhaseAnalyzer._classify(result, config, d2_ratio,
                                  d2_resolution, int(np.sum(c11_valid)))
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_single_tol(datasets, ref_keys, N, geo, tol,
                            config: PhaseAnalyzerConfig) -> str:
        """单容差的简化分类（多容差稳定性比较用）。"""
        binned = {}
        common = None
        for rb, by_alpha in datasets.items():
            if isinstance(rb, str) and rb.startswith('_'):
                continue   # 跳过元数据键
            beta = ref_keys.get(rb) or refbond_direction(*rb)
            for alpha in 'ABC':
                binned[(alpha, beta)] = SCPhaseAnalyzer._bin_entries(
                    datasets, rb, alpha, N, geo, tol)
                common = (set(binned[(alpha, beta)]) if common is None
                          else common & set(binned[(alpha, beta)]))
        if not common:
            return 'inconclusive'
        c12_list, c11_list, c22_list = [], [], []
        for b in sorted(common):
            mat = np.zeros((3, 3), dtype=complex)
            for ia, a in enumerate(KEY_ORDER):
                for ib, bb in enumerate(KEY_ORDER):
                    mat[ia, ib] = binned[(a, bb)][b].value
            c12_list.append(V_D1 @ mat @ V_D2)
            c11_list.append(V_D1 @ mat @ V_D1)
            c22_list.append(V_D2 @ mat @ V_D2)
        c12 = np.array(c12_list)
        nf = SCPhaseAnalyzer.estimate_noise_floor(np.abs(c12), config)
        valid = [abs(v) > nf * config.noise_prom_factor for v in c12]
        phi = [float(np.degrees(np.angle(v))) for v in c12]
        dists = [tol * b for b in sorted(common)]
        plat_p = SCPhaseAnalyzer.detect_platform(
            dists, phi, valid, +90.0, config)
        plat_m = SCPhaseAnalyzer.detect_platform(
            dists, phi, valid, -90.0, config)
        if plat_p.found or plat_m.found:
            return 'chiral_phase_candidate'
        # d2 分辨能力三层（与主管线一致：双阈值比值口径）
        nf11 = SCPhaseAnalyzer.estimate_noise_floor(
            np.abs(np.array(c11_list)), config)
        c11_valid = [abs(v) > nf11 * config.noise_prom_factor
                     for v in c11_list]
        r22 = [abs(x) / abs(y) for x, y, v in zip(c22_list, c11_list, c11_valid)
               if v]
        d2r = float(np.median(r22)) if r22 else np.nan
        if int(np.sum(c11_valid)) < config.d2_suppression_n:
            if not np.any(valid):
                return 'insufficient'
            return 'no_chiral_signal'
        if np.isfinite(d2r) and d2r < config.d2_unresolved_ratio:
            return 'd_component_unresolved'
        if np.isfinite(d2r) and d2r < config.d2_suppression_ratio:
            return 'd_component_suppressed'
        if not np.any(valid):
            return 'insufficient'
        return 'no_chiral_signal'

    @staticmethod
    def _classify(result: PhaseAnalysisResult, config: PhaseAnalyzerConfig,
                  d2_ratio: float, d2_resolution: str,
                  n_valid_c11: int) -> None:
        """分类优先级（审计修正版）：

        1. insufficient：C12 无有效点且 C11 可靠点不足
        2. inconclusive：多容差分类不一致（先于物理分类，不被覆盖）
        3. d_component_unresolved：d2 低于数值分辨能力（物理抑制与
           数值零无法区分；不能声称物理抑制）
        4. d_component_suppressed：d2 高于分辨标尺且比值 < 阈值
           （物理可分辨的抑制）
        5. chiral_phase_candidate：稳定 ±90° 平台
        6. no_chiral_signal：数据充足但无手性平台

        同 bin 跨通道偏差诊断（conjugate_diagnostics）为描述性指标，
        不参与任何分类分支。
        """
        plat_ok = result.platform_plus.found or result.platform_minus.found
        n_valid = result.n_valid_bins
        tol_classes = set(result.tol_stability.values())

        if n_valid == 0 and n_valid_c11 < config.d2_suppression_n:
            result.classification = 'insufficient'
            result.reasons.append('无有效距离点（C12 幅值门控全部拒绝）'
                                  '且 C11 可靠点不足，无法进行可靠判断')
            return
        if len(tol_classes) > 1:
            result.classification = 'inconclusive'
            result.reasons.append(f'多容差分类不一致: {sorted(tol_classes)}')
            return
        if d2_resolution == 'unresolved':
            result.classification = 'd_component_unresolved'
            result.reasons.append(
                f'd2 通道低于当前数据的数值分辨能力'
                f'（中位 |C22|/|C11| = {d2_ratio:.2e} < '
                f'{config.d2_unresolved_ratio:.0e}），'
                '无法区分物理抑制与数值零；'
                '无论何种解释，均无法支持 d1/d2 双分量共存形成 d+id'
                + ('；同时未观察到稳定 ±90° 平台' if not plat_ok else ''))
            return
        if d2_resolution == 'resolved' and np.isfinite(d2_ratio) \
                and d2_ratio < config.d2_suppression_ratio:
            result.classification = 'd_component_suppressed'
            result.reasons.append(
                f'd2 通道高于数值分辨比且显著弱于 d1'
                f'（C11 有效点上中位 |C22|/|C11| = {d2_ratio:.2e} '
                f'< {config.d2_suppression_ratio:.0e}），'
                '物理可分辨的通道抑制，无法支持 d1/d2 双分量共存形成 d+id'
                + ('；同时未观察到稳定 ±90° 平台' if not plat_ok else ''))
            return
        if plat_ok:
            result.classification = 'chiral_phase_candidate'
            sign = '+' if result.platform_plus.found else '-'
            result.reasons.append(f'{sign}90° 相位平台候选，'
                                  '多容差稳定性通过')
            return
        result.classification = 'no_chiral_signal'
        result.reasons.append('有效关联数据充足但未观察到稳定的 ±90° 相位平台')

    # ------------------------------------------------------------------
    @staticmethod
    def print_summary(result: PhaseAnalysisResult) -> None:
        """终端摘要打印（圆周统计 + 平台 + 门控）。"""
        print('=' * 66)
        print('d+id 配对相位分析摘要')
        print('=' * 66)
        print(f'classification      : {result.classification}')
        print(f'有效距离点数        : {result.n_valid_bins}/{result.n_bins}'
              f'（bin 容差 {result.config.get("primary_tol")}）')
        print(f'噪声底              : {result.noise_floor:.3e}'
              f'（非零值 {result.noise_counts.get("nonzero")}/'
              f'{result.noise_counts.get("total")}，'
              f'精确零 {result.noise_counts.get("zero")}）')
        if np.isfinite(result.circular_mean_deg):
            print(f'C12 相位圆周均值    : {result.circular_mean_deg:.1f}°'
                  f'（std {result.circular_std_deg:.1f}°, '
                  f'R={result.circular_R:.3f}, n={result.circular_n}）')
        else:
            print(f'C12 相位圆周均值    : n/a（{result.circular_mean_invalid_reason}）')
        print(f'中位 |C22|/|C11|    : {result.d2_ratio_median:.3e}'
              f'（分辨能力: {result.d2_resolution}）')
        conj_ok = result.conjugate_diagnostics['ok']
        conj_dev = result.conjugate_diagnostics['max_relative_deviation']
        conj_str = ('通过' if conj_ok is True else
                    '异常' if conj_ok is False else '无可比点(not_applicable)')
        print(f'同 bin 跨通道偏差   : '
              f'{conj_str}'
              + (f'（最大相对偏差 {conj_dev:.3e}）'
                 if np.isfinite(conj_dev) else '') + '（描述性，不参与分类）')
        for tag, p in [('+90°', result.platform_plus),
                       ('-90°', result.platform_minus)]:
            if p.found:
                print(f'{tag} 平台            : 找到 {p.n_points} 个连续点，'
                      f'r ∈ [{p.r_range[0]:.1f}, {p.r_range[1]:.1f}]，'
                      f'圆周均值 {p.circ_mean_deg:.1f}°')
        for tol, cls in result.tol_stability.items():
            print(f'容差 {tol} 分类        : {cls}')
        for r in result.reasons:
            print(f'  依据: {r}')
        print('=' * 66)
