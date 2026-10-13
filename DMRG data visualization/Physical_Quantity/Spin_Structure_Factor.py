# -*- coding: utf-8 -*-
"""
自旋结构因子计算模块（蜂窝 XC 圆柱 t-J 模型）

定义（归一化 1/N，复指数直接求和）：
    S(q) = (1/N) Σ_{(i,j)} exp[i q·(r_i − r_j)] · C_ij

其中 N 为格点总数，r_i 为格点笛卡尔坐标，C_ij 为**实测**关联值。
对称对 (i,j)/(j,i) 各自按其复指数贡献累加，不把对称推导的对
当作实测数据（有序对求和天然处理共轭成对，虚部应为数值噪声级）。

重要限制（基于数据完整性审计）：
- N200 数据仅含源点 0（仅 SzSz 纵向）、49、50 的部分关联，
  完整 200×200 关联矩阵不存在。本模块对现有数据的输出一律标注
  is_restricted=True 的**受限估计**，不补零、不插值、不冒充完整结构因子。
- 本模块不根据结构因子峰值判定 AFM 序、SDW 或其他物理相。

动量空间（XC 圆柱几何，代码核实）：
- 周向 y：周期（最近邻表含 (k+3,k) wrap 对），周期平移矢量长度
  Ly = 4 × (0.5√3·a) = 2√3·a → 允许动量 qy = 2πm/Ly（m 整数，独立值 4 个）。
- 轴向 x：开放（无 x 周期 wrap），不人工周期化——qx 为连续傅里叶
  扫描（有限窗口 25 胞的解释限制见文档，q 分辨率 ~2π/(75a)）。
- 基元内格点相位通过实际坐标 (x, y) 直接进入复指数，不重复计入。
"""

import ast
import numpy as np

from .Honeycomb_Lattice import HoneycombXCGeometry

# 关联量定义标签（严格区分，不混用）
DEF_SPIN_DOT = 'spin_dot'   # <S_i·S_j> = SzSz + 1/2(SuSd + SdSu)（三分量合成）
DEF_SZ_SZ = 'sz_sz'         # <S_i^z S_j^z>（仅纵向分量）


# ==========================================================================
# 1. 通用结构因子核（物理量无关）
# ==========================================================================
def compute_structure_factor(coords, C_ordered, q_list, N, definition,
                             expected_pairs=None):
    """通用结构因子核：S(q) = (1/N) Σ_{(i,j)} exp[i q·(r_i−r_j)]·C_ij。

    参数
    ----
    coords : dict，格点编号 -> (x, y) 笛卡尔坐标
    C_ordered : dict (i, j) -> value，或 list[(i, j, value)]
        实测有序格点对关联值。list 输入时重复键取均值
        （duplicate_pairs 计数，规则明确：同对多次测量按组内平均处理）。
        值为 NaN/Inf 的对不计入求和，计入 invalid_values 并报告。
    q_list : ndarray (M, 2) 或 list[(qx, qy)]
    N : int，物理系统总格点数（归一化 1/N 与 source_coverage 分母）
    definition : str，关联量定义标签（'spin_dot' / 'sz_sz'，原样记录到输出）
    expected_pairs : 可选 set[(i, j)]，本次计算预期覆盖的有序对集合。
        不提供时 expected_pair_count / pair_coverage / missing_pairs 为 None，
        **不得声称得到完整覆盖率**。

    返回
    ----
    dict：q / S_q / definition / N / measured_pair_count / expected_pair_count /
    pair_coverage / source_coverage / missing_pairs / is_restricted /
    imaginary_residual / duplicate_pairs / invalid_values / skipped_no_coords

    覆盖率口径：pair_coverage = measured / expected（仅当 expected_pairs
    提供时定义）；source_coverage = 实测涉及的源点数 / N。
    """
    # ---- 输入规范化：list → dict（重复键取均值） ----
    duplicate_pairs = 0
    if isinstance(C_ordered, dict):
        C = dict(C_ordered)
    else:
        C = {}
        for item in C_ordered:
            key = (int(item[0]), int(item[1]))
            v = float(item[2])
            if key in C:
                C[key] = 0.5 * (C[key] + v)   # 重复对：取均值
                duplicate_pairs += 1
            else:
                C[key] = v

    q_arr = np.asarray(q_list, dtype=float)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(1, 2)

    # ---- 有效对筛选（NaN/Inf、缺坐标 → 显式报告，不静默） ----
    valid_pairs = []
    invalid_values = 0
    skipped_no_coords = []
    for (i, j), v in C.items():
        if not np.isfinite(v):
            invalid_values += 1
            continue
        if i not in coords or j not in coords:
            skipped_no_coords.append((i, j))
            continue
        valid_pairs.append((i, j, v, coords[i], coords[j]))
    n_measured = len(valid_pairs)

    # ---- 求和：S(q) = (1/N) Σ C_ij exp[i q·(r_i − r_j)] ----
    S = np.zeros(len(q_arr), dtype=complex)
    for (i, j, v, ri, rj) in valid_pairs:
        dx = ri[0] - rj[0]
        dy = ri[1] - rj[1]
        phase = q_arr[:, 0] * dx + q_arr[:, 1] * dy
        S += v * (np.cos(phase) + 1j * np.sin(phase))
    S = S / N

    # ---- 覆盖率与受限性 ----
    # expected_pairs 不提供时覆盖率无法完整评估：如实返回 None，不伪造数值
    expected_pair_count = len(expected_pairs) if expected_pairs is not None else None
    if expected_pairs is not None:
        measured_keys = {(p[0], p[1]) for p in valid_pairs}
        missing = sorted(expected_pairs - measured_keys)
        pair_coverage = n_measured / expected_pair_count if expected_pair_count else 0.0
    else:
        missing = None
        pair_coverage = None
    source_coverage = len({p[0] for p in valid_pairs}) / N if N > 0 else np.nan

    is_restricted = (expected_pairs is None
                     or pair_coverage is None
                     or pair_coverage < 1.0
                     or source_coverage < 1.0)

    # 虚部残差：对称对共轭成对时虚部应为数值噪声级；
    # imaginary_residual = max|Im S| / max|S|（max|S|=0 时 NaN）
    abs_S = np.abs(S)
    denom = float(np.max(abs_S)) if len(abs_S) else np.nan
    imaginary_residual = (float(np.max(np.abs(S.imag))) / denom
                          if np.isfinite(denom) and denom > 0 else np.nan)

    return {
        'q': q_arr,
        'S_q': S,
        'definition': definition,
        'N': N,
        'measured_pair_count': n_measured,
        'expected_pair_count': expected_pair_count,
        'pair_coverage': pair_coverage,
        'source_coverage': float(source_coverage),
        'missing_pairs': missing,
        'is_restricted': bool(is_restricted),
        'imaginary_residual': imaginary_residual,
        'duplicate_pairs': duplicate_pairs,
        'invalid_values': invalid_values,
        'skipped_no_coords': skipped_no_coords,
    }


# ==========================================================================
# 2. XC 圆柱动量网格
# ==========================================================================
def xc_momentum_grid(N, a=1.0, n_qx=41, qx_lim=None, n_qy=4):
    """XC 圆柱动量网格（依据 HoneycombXCGeometry 实际几何）。

    周向 y（周期）：周期平移矢量长度 Ly = 4 × 0.5√3·a = 2√3·a，
    允许动量 qy = 2πm/Ly，m = 0..n_qy−1（4 行 → 独立 qy 至多 4 个）。
    轴向 x（开放）：不人工周期化，qx 为等距连续傅里叶扫描，
    默认区间 [−π/(3a), π/(3a)]（3a 为胞间距参考尺度；有限窗口
    25 胞的解释限制：q 分辨率 ~2π/(75a)，非周期方向的峰宽受窗口卷积）。

    返回 (q_list, meta)，meta 含 Ly、周期性说明。
    """
    if N % 8 != 0:
        raise ValueError(f"N={N} 不是 8 的倍数，不符合 XC 几何约束")
    # 用几何对象核对周期平移矢量（y 行距与 wrap 结构由 HoneycombXCGeometry 保证）
    HoneycombXCGeometry(N, a)   # 仅校验 N 合法性
    Ly = 2.0 * np.sqrt(3.0) * a
    if qx_lim is None:
        qx_lim = (-np.pi / (3.0 * a), np.pi / (3.0 * a))
    qx = np.linspace(qx_lim[0], qx_lim[1], n_qx)
    qy = np.arange(n_qy) * (2.0 * np.pi / Ly)
    qs = np.array([(qx_i, qy_j) for qx_i in qx for qy_j in qy])
    meta = {
        'Ly': Ly,
        'y_periodic': True,
        'y_allowed_q': 'qy = 2πm/Ly, m integer (independent values ≤ 4)',
        'x_periodic': False,
        'x_note': '开放方向：连续傅里叶扫描，不做人工周期化；'
                  '有限窗口(25 胞)的 q 分辨率 ~2π/(75a)，结果解释受限',
        'qx_lim': qx_lim,
        'n_qx': n_qx, 'n_qy': n_qy,
    }
    return qs, meta


# ==========================================================================
# 3. 自旋结构因子受限估计入口
# ==========================================================================
def _xc_geometry_meta(geometry: HoneycombXCGeometry) -> dict:
    """从实际几何对象组装 XC 圆柱元信息（供有限尺寸标度接口追溯）。

    边界条件由最近邻索引表动态判定（y 方向 (k+3,k) wrap 对 = 周期；
    x 方向无跨胞远端对 = 开放），不是目录名推断。Lx 字段明确为坐标
    跨度 Lx_span（开放方向无周期平移矢量，不冒充物理长度）。
    """
    nn = geometry._nearest_neighbor_grid_index()
    y_periodic = any((a - b == 3 and a % 4 == 3) for a, b in nn)
    x_wrap = any(abs(a - b) > 10 for a, b in nn)
    xs = [v[0] for v in geometry.coords.values()]
    ys = [v[1] for v in geometry.coords.values()]
    return {
        'geometry': 'HoneycombXCGeometry (XC cylinder, 8-site basis)',
        'N': geometry.N,
        'a': geometry.a,
        'n_cells_x': geometry.N // 8,
        'Lx_span': float(max(xs) - min(xs)),       # 坐标跨度，非周期长度
        'Ly': 2.0 * np.sqrt(3.0) * geometry.a,      # 周向周期（4 行 × 0.5√3·a，wrap 结构实测）
        'n_rows_y': len(set(round(y, 9) for y in ys)),
        'boundary_conditions': {'x': 'open' if not x_wrap else 'periodic',
                                'y': 'periodic' if y_periodic else 'open'},
        'q_units': 'inverse of lattice coordinate units (a)',
        'q_coordinate_system': 'cartesian, same axes as HoneycombXCGeometry.coords',
    }


class SpinStructureFactorEstimator:
    """自旋结构因子受限估计（N200 数据仅 3 个源点，覆盖率不可完整评估）。

    输出一律 is_restricted=True：现有数据无法构成完整 200×200 关联矩阵，
    不补零、不插值、不声称完整结构因子。输出附带 'meta' 键（几何元信息，
    来自实际几何对象），供有限尺寸标度等下游接口追溯。
    """

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_spin_dot_from_records(record_list, q_list, N, a=1.0):
        """从三分量合成的 <S_i·S_j>（list[CorrelationRecord]，
        CorrelationProcessor.Combined_spin_correlation 的输出）估计受限的
        完整自旋点积关联结构因子。

        每个记录贡献有序对 (source_point, field_point)；对称推导的反向
        对不补入（不把推导数据算作实测）。expected_pairs 无法确定
        （完整格点对集未知）→ 覆盖率标记为不可完整评估。
        """
        geometry = HoneycombXCGeometry(N, a)
        pairs = []
        for rec in record_list:
            pairs.append((int(rec.source_point), int(rec.field_point),
                          float(rec.correlation)))
        if not pairs:
            raise ValueError("输入 record_list 为空，无法计算结构因子")
        result = compute_structure_factor(
            geometry.coords, pairs, q_list, N,
            definition=DEF_SPIN_DOT, expected_pairs=None)
        result['meta'] = _xc_geometry_meta(geometry)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_sz_sz_from_file(path, q_list, N, a=1.0, source=0):
        """从源点 0 的旧版 sz.json（格式 [[j], [value, 0]]，单索引 = 场点，
        源点隐含为 0）独立解析并估计受限的纵向结构因子 <S^z S^z>。

        仅纵向分量，**不是**完整自旋点积结构因子（definition='sz_sz'）。
        对角项 j=0 在文件中实测存在，按实测数据计入。
        """
        with open(path, encoding='utf-8') as f:
            data = ast.literal_eval(f.read())
        if not data:
            raise ValueError(f"输入文件 {path} 内容为空，无法计算结构因子")
        geometry = HoneycombXCGeometry(N, a)
        pairs = []
        for element in data:
            try:
                j = int(element[0][0])
                v = float(element[1][0])
            except (IndexError, TypeError, ValueError):
                raise ValueError(f"sz.json 元素格式不符合 [[j], [value, 0]]: {element}")
            pairs.append((source, j, v))
        result = compute_structure_factor(
            geometry.coords, pairs, q_list, N,
            definition=DEF_SZ_SZ, expected_pairs=None)
        result['meta'] = _xc_geometry_meta(geometry)
        return result
