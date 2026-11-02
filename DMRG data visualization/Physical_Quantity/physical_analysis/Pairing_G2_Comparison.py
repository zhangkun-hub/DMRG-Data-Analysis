# -*- coding: utf-8 -*-
"""
配对关联 vs 单粒子关联平方比较（P1 阶段 3：叶子模块，只依赖 numpy）。

数值诊断：窄圆柱/准一维语境下，配对关联 ⟨Δ†Δ⟩(r) 若衰减得**慢于**
单粒子关联的平方 G(r)²，则配对关联主导（非平庸配对信号，配对由
Cooper 对传播而非单粒子传播子平庸给出）；反之若衰减得**快于** G²，
则无配对优势——这是窄圆柱 DMRG 文献常用的参照基准检验（文献语境：
arXiv:2303.16487 的准 1D 圆柱适用性讨论；配对 vs 单粒子基准）。

结论三态：
- pairing_dominant：配对关联慢于 G²（K_sc < 2·K_G − slope_margin）
- single_particle_dominant：配对关联快于 G²（K_sc > 2·K_G + slope_margin）
- insufficient：数据不足（点数/正值不足）或斜率不确定

纪律：本诊断只输出数值比较与三态结论，**不判相**——pairing_dominant
不等于超导相；single_particle_dominant 不等于无超导。DEFAULT_THRESHOLDS
为默认值（未获物理授权），调用方可覆盖。
"""
import numpy as np

DEFAULT_THRESHOLDS = {
    'slope_margin': 0.2,   # 斜率差边际（未获物理授权）
    'min_points': 3,       # log 拟合最少点数
}

VERDICT_PAIRING_DOMINANT = 'pairing_dominant'
VERDICT_SINGLE_PARTICLE_DOMINANT = 'single_particle_dominant'
VERDICT_INSUFFICIENT = 'insufficient'
VERDICTS = (VERDICT_PAIRING_DOMINANT, VERDICT_SINGLE_PARTICLE_DOMINANT,
            VERDICT_INSUFFICIENT)


def _log_slope(r, y):
    """log(y) vs log(r) 的线性斜率（最小二乘）。y 必须全为正。"""
    mask = (np.asarray(r, dtype=float) > 0) & (np.asarray(y, dtype=float) > 0)
    r = np.asarray(r, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if len(r) < DEFAULT_THRESHOLDS['min_points']:
        return None, len(r)
    lr, ly = np.log(r), np.log(y)
    slope, _intercept = np.polyfit(lr, ly, 1)
    return float(slope), len(r)


def compare_pairing_vs_g2(r, pairing_abs, g_abs, window=None,
                          thresholds=None) -> dict:
    """核心比较：|ΔΔ|(r) 与 G(r)² 的衰减斜率对比。

    参数：
    - r / pairing_abs / g_abs：等长平行序列（distance / |ΔΔ| / |G|）
    - window：(r_min, r_max) 可选距离窗口过滤
    - thresholds：覆盖默认阈值（slope_margin / min_points）

    返回 dict：verdict / k_sc / k_g / k_g2（=2·k_g）/ slope_diff /
    n_points / r_window / thresholds / threshold_status / detail。
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    r = np.asarray(r, dtype=float)
    pairing_abs = np.asarray(pairing_abs, dtype=float)
    g_abs = np.asarray(g_abs, dtype=float)
    if not (len(r) == len(pairing_abs) == len(g_abs)) or len(r) == 0:
        return _result(VERDICT_INSUFFICIENT, th, None, None, None,
                       '输入序列为空或长度不一致')
    if window is not None:
        lo, hi = window
        mask = (r >= lo) & (r <= hi)
        r, pairing_abs, g_abs = r[mask], pairing_abs[mask], g_abs[mask]
    k_sc, n_sc = _log_slope(r, pairing_abs)
    k_g, n_g = _log_slope(r, g_abs)
    n = min(n_sc, n_g)
    if k_sc is None or k_g is None:
        return _result(VERDICT_INSUFFICIENT, th, k_sc, k_g, None,
                       f'有效点数不足（需 ≥{th["min_points"]} 个正距离/'
                       '正值点）')
    k_g2 = 2.0 * k_g
    # 斜率均为负值（衰减）：数值越大衰减越慢。配对慢于 G² ⟺ k_sc > k_g2
    diff = k_sc - k_g2   # >0：配对比 G² 慢（配对主导方向）
    if diff > th['slope_margin']:
        verdict = VERDICT_PAIRING_DOMINANT
        detail = (f'配对关联比 G² 衰减慢（K_sc={-k_sc:.3g} < '
                  f'2·K_G={-k_g2:.3g}）')
    elif diff < -th['slope_margin']:
        verdict = VERDICT_SINGLE_PARTICLE_DOMINANT
        detail = (f'配对关联比 G² 衰减快（K_sc={-k_sc:.3g} > '
                  f'2·K_G={-k_g2:.3g}）')
    else:
        verdict = VERDICT_INSUFFICIENT
        detail = (f'斜率差 {diff:.3g} 在边际 ±{th["slope_margin"]} 内，'
                  '无法区分（数值不确定）')
    return _result(verdict, th, k_sc, k_g, k_g2, detail)


def _result(verdict, th, k_sc, k_g, k_g2, detail):
    return {'verdict': verdict, 'k_sc': k_sc, 'k_g': k_g, 'k_g2': k_g2,
            'slope_diff': (None if k_g2 is None or k_sc is None
                           else k_sc - k_g2),
            'detail': detail, 'thresholds': dict(th),
            'threshold_status': 'defaults_not_authorized'}
