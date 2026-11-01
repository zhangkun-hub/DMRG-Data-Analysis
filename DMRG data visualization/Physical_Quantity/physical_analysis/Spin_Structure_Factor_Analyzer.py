# -*- coding: utf-8 -*-
"""
自旋结构因子峰值分析模块（数值层面，非物理相判定）

对 Spin_Structure_Factor.compute_structure_factor 的输出做数值峰值识别：
- 原始 2D q 网格上的 8 邻域严格局部极大 + 高度筛选
- 亚网格细化峰位（2D 抛物面拟合，仅在拟合有效、Hessian 负定、
  顶点位于合理邻域内时接受，否则回退原始峰位并记录原因）
- 沿 qx/qy 方向线性插值 FWHM 峰宽估计与分辨率限制标记
- 边界峰 / 近邻峰 / 平坦与无效信号的明确状态报告

重要说明：
- 本模块只输出数值峰值信息，不据此判定磁有序、Néel 相、SDW 或其他物理相。
- 峰值检测成功不改变输入结果的 is_restricted 标记；受限估计不会因
  检测到峰而被升级为完整结构因子。
- 默认信号量取 |S(q)| 模长：受限数据（单源点等）关联矩阵不对称，
  S(q) 为复数，模长是相位无关的强度量；模长不等于受限数据的物理
  结构因子强度，仅用于峰位与相对强度的数值分析。'real' 模式对
  对称完整数据可用。
"""

import numpy as np

# 峰值报告透传的元数据字段（字段缺失时保留缺失状态，不补默认值）
_PASS_THROUGH_FIELDS = ('definition', 'is_restricted', 'source_coverage',
                        'pair_coverage', 'expected_pair_count', 'N', 'meta')


class SpinStructureFactorPeakAnalyzer:
    """自旋结构因子峰值分析器（数值层面）。"""

    # ------------------------------------------------------------------
    @staticmethod
    def _grid_from_q(q, q_shape=None):
        """从 q 数组构造规则 2D 网格并验证。

        q 要求为嵌套排序的规则矩形网格（如 xc_momentum_grid 的输出：
        qx 外层 × qy 内层）。无法无歧义构造规则网格时抛 ValueError，
        不静默 reshape。等距性单独校验（"矩形网格" ≠ "等距网格"）：
        索引坐标抛物面细化仅在等距网格上合法。
        返回 (grid_q, qx_axis, qy_axis, dq_x, dq_y, uniform_x, uniform_y)。
        """
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != 2 or len(q) == 0:
            raise ValueError("q 必须为非空 (M, 2) 数组")
        if q_shape is not None:
            nx, ny = int(q_shape[0]), int(q_shape[1])
            if nx * ny != len(q):
                raise ValueError(f"q_shape={q_shape} 与 q 点数 {len(q)} 不匹配")
        else:
            nx = len(np.unique(np.round(q[:, 0], 12)))
            ny = len(np.unique(np.round(q[:, 1], 12)))
            if nx * ny != len(q):
                raise ValueError(
                    f"q 网格不规则：{len(q)} 个点无法构成 "
                    f"{nx}×{ny} 规则网格（重复或缺失采样）")
        grid_q = q.reshape(nx, ny, 2)
        # 验证规则性：同一行的 qx 一致、同一列的 qy 一致
        qx_row = grid_q[:, 0, 0][:, None] * np.ones((1, ny))
        qy_col = np.ones((nx, 1)) * grid_q[0, :, 1][None, :]
        if (not np.allclose(grid_q[:, :, 0], qx_row, atol=1e-9)
                or not np.allclose(grid_q[:, :, 1], qy_col, atol=1e-9)):
            raise ValueError("q 网格不是规则矩形网格（行列坐标不一致）")
        qx_axis = grid_q[:, 0, 0]
        qy_axis = grid_q[0, :, 1]
        dq_x = float(np.median(np.diff(qx_axis))) if nx > 1 else np.nan
        dq_y = float(np.median(np.diff(qy_axis))) if ny > 1 else np.nan
        # 等距性校验：各方向步长与中位数一致（相对容差 1e-6 吸收浮点误差）
        def _uniform(axis):
            if len(axis) <= 1:
                return True
            step = np.diff(axis)
            med = float(np.median(step))
            return bool(np.allclose(step, med, rtol=1e-6, atol=1e-12))
        uniform_x = _uniform(qx_axis)
        uniform_y = _uniform(qy_axis)
        return grid_q, qx_axis, qy_axis, dq_x, dq_y, uniform_x, uniform_y

    # ------------------------------------------------------------------
    @staticmethod
    def find_peaks(sf_result: dict, signal: str = 'abs',
                   min_height_frac: float = 0.2, refine: bool = True,
                   q_shape: tuple = None) -> dict:
        """数值峰值识别。

        参数
        ----
        sf_result : compute_structure_factor 的输出 dict
        signal : 'abs'（默认，|S(q)| 模长）| 'real'（S(q) 实部）
        min_height_frac : 峰高下限（相对 max 的比例，0~1）
        refine : 是否做亚网格抛物面细化（边界峰默认跳过）
        q_shape : (nx, ny) 显式网格形状；None 时从 q 唯一值推断并验证

        返回
        ----
        dict：status / status_reason / signal_used / n_peaks / peaks / resolution /
        max_signal / invalid_signal_points / 透传元数据（definition、
        is_restricted 等，缺失时保留缺失状态）
        """
        if signal not in ('abs', 'real'):
            raise ValueError(f"signal 必须为 'abs' 或 'real'，得到 {signal}")
        q = np.asarray(sf_result['q'], dtype=float)
        S = np.asarray(sf_result['S_q'], dtype=complex)
        if len(q) != len(S):
            raise ValueError("sf_result 中 q 与 S_q 长度不一致")

        grid_q, qx_axis, qy_axis, dq_x, dq_y, uniform_x, uniform_y = \
            SpinStructureFactorPeakAnalyzer._grid_from_q(q, q_shape)
        nx, ny = grid_q.shape[0], grid_q.shape[1]
        grid_uniform = bool(uniform_x and uniform_y)

        sig_flat = np.abs(S) if signal == 'abs' else S.real
        sig2d = sig_flat.reshape(nx, ny)
        max_signal = float(np.max(np.abs(S)))

        # ---- 信号状态判定 ----
        n_invalid = int(np.sum(~np.isfinite(sig2d)))
        if n_invalid == nx * ny:
            return SpinStructureFactorPeakAnalyzer._no_peak_report(
                sf_result, signal, grid_q, qx_axis, qy_axis, dq_x, dq_y,
                'all_invalid', '全部信号点为 NaN/Inf', n_invalid, max_signal, grid_uniform)
        valid_sig = sig2d[np.isfinite(sig2d)]
        if len(valid_sig) == 0:
            return SpinStructureFactorPeakAnalyzer._no_peak_report(
                sf_result, signal, grid_q, qx_axis, qy_axis, dq_x, dq_y,
                'all_invalid', '无有效信号点', n_invalid, max_signal, grid_uniform)
        vmax = float(np.max(valid_sig))
        vmin = float(np.min(valid_sig))
        if abs(vmax) < 1e-300:
            return SpinStructureFactorPeakAnalyzer._no_peak_report(
                sf_result, signal, grid_q, qx_axis, qy_axis, dq_x, dq_y,
                'all_zero', '信号全为零', n_invalid, max_signal, grid_uniform)
        if (vmax - vmin) < 1e-12 * max(abs(vmax), abs(vmin)):
            return SpinStructureFactorPeakAnalyzer._no_peak_report(
                sf_result, signal, grid_q, qx_axis, qy_axis, dq_x, dq_y,
                'flat', f'信号平坦（动态范围 {vmax - vmin:.3e} < 1e-12×幅度）',
                n_invalid, max_signal, grid_uniform)

        # ---- 8 邻域严格局部极大（NaN 邻点不构成压制） ----
        candidates = []
        for i in range(nx):
            for j in range(ny):
                v = sig2d[i, j]
                if not np.isfinite(v):
                    continue
                is_max = True
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            nv = sig2d[ni, nj]
                            if np.isfinite(nv) and nv >= v:
                                is_max = False
                                break
                    if not is_max:
                        break
                if is_max:
                    candidates.append((i, j, v))
        height_floor = min_height_frac * vmax
        candidates = [c for c in candidates if c[2] >= height_floor]
        if not candidates:
            return SpinStructureFactorPeakAnalyzer._no_peak_report(
                sf_result, signal, grid_q, qx_axis, qy_axis, dq_x, dq_y,
                'no_peak_above_threshold',
                f'无高于 min_height_frac×max = {height_floor:.4g} 的局部极大',
                n_invalid, max_signal, grid_uniform)

        peaks = []
        for i, j, v in candidates:
            peak = {
                'q_original': (float(qx_axis[i]), float(qy_axis[j])),
                'grid_index': (int(i), int(j)),
                'height': float(v),
                'boundary': bool(i == 0 or i == nx - 1 or j == 0 or j == ny - 1),
                'q_refined': None,
                'refine_status': 'not_attempted',
                'fwhm_qx': None,
                'fwhm_qy': None,
                'fwhm_note': '',
                'resolution_limited': False,
                'merged_risk': False,
            }
            # 亚网格细化：边界峰跳过（窗口截断，抛物面外推不可靠）；
            # 非等距网格跳过（索引坐标拟合失真，保留原始网格峰位）
            if refine and not peak['boundary'] and nx >= 3 and ny >= 3 and grid_uniform:
                SpinStructureFactorPeakAnalyzer._refine_peak(
                    sig2d, qx_axis, qy_axis, i, j, peak)
            else:
                if peak['boundary']:
                    peak['refine_status'] = 'skipped_boundary'
                elif refine and not grid_uniform:
                    peak['refine_status'] = 'skipped_nonuniform'
                else:
                    peak['refine_status'] = 'not_attempted'
                peak['q_refined'] = peak['q_original']
            # FWHM（实际 q 坐标线性插值）
            SpinStructureFactorPeakAnalyzer._fwhm(
                sig2d, qx_axis, qy_axis, i, j, dq_x, dq_y, peak)
            peaks.append(peak)

        # ---- 近邻峰风险（基于物理 q 距离，不用网格索引距离） ----
        diag_step = float(np.hypot(dq_x, dq_y)) if np.isfinite(dq_x) and np.isfinite(dq_y) else np.nan
        for a in range(len(peaks)):
            for b in range(a + 1, len(peaks)):
                dq = float(np.hypot(peaks[a]['q_original'][0] - peaks[b]['q_original'][0],
                                    peaks[a]['q_original'][1] - peaks[b]['q_original'][1]))
                if np.isfinite(diag_step) and dq < 2.0 * diag_step:
                    peaks[a]['merged_risk'] = True
                    peaks[b]['merged_risk'] = True

        report = {
            'status': 'ok',
            'status_reason': '',
            'signal_used': signal,
            'n_peaks': len(peaks),
            'peaks': peaks,
            'resolution': {'dq_x': dq_x, 'dq_y': dq_y,
                           'grid_uniform': grid_uniform,
                           'note': '亚网格细化峰位不代表真实物理分辨率；'
                                   '开放方向（如 XC 轴向 x）另有有限窗口限制 '
                                   '~2π/(窗口长度)'},
            'max_signal': max_signal,
            'invalid_signal_points': n_invalid,
        }
        for f in _PASS_THROUGH_FIELDS:
            if f in sf_result:
                report[f] = sf_result[f]
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _no_peak_report(sf_result, signal, grid_q, qx_axis, qy_axis,
                        dq_x, dq_y, status, reason, n_invalid, max_signal,
                        grid_uniform=None):
        report = {
            'status': status,
            'status_reason': reason,
            'signal_used': signal,
            'n_peaks': 0,
            'peaks': [],
            'resolution': {'dq_x': dq_x, 'dq_y': dq_y,
                           'grid_uniform': grid_uniform,
                           'note': '亚网格细化峰位不代表真实物理分辨率；'
                                   '开放方向（如 XC 轴向 x）另有有限窗口限制 '
                                   '~2π/(窗口长度)'},
            'max_signal': max_signal,
            'invalid_signal_points': n_invalid,
        }
        for f in _PASS_THROUGH_FIELDS:
            if f in sf_result:
                report[f] = sf_result[f]
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _refine_peak(sig2d, qx_axis, qy_axis, i, j, peak):
        """2D 抛物面细化：s ≈ a + b·u + c·v + d·u² + e·v² + f·u·v
        （u、v 为网格索引坐标）。仅当拟合有效、Hessian 负定（局部极大）、
        顶点偏移 ≤ 0.75 步长时接受；否则回退原始峰位并记录原因。"""
        n = 1   # 邻域半径（3×3）
        us, vs, ss = [], [], []
        for di in (-n, 0, n):
            for dj in (-n, 0, n):
                ni, nj = i + di, j + dj
                if 0 <= ni < sig2d.shape[0] and 0 <= nj < sig2d.shape[1]:
                    sv = sig2d[ni, nj]
                    if np.isfinite(sv):
                        us.append(di)
                        vs.append(dj)
                        ss.append(sv)
        if len(ss) < 6:
            peak['refine_status'] = 'fit_failed'
            peak['q_refined'] = peak['q_original']
            return
        X = np.column_stack([np.ones(len(us)), us, vs,
                             [u * u for u in us], [v * v for v in vs],
                             [u * v for u, v in zip(us, vs)]])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, ss, rcond=None)
        except Exception:
            peak['refine_status'] = 'fit_failed'
            peak['q_refined'] = peak['q_original']
            return
        a, b, c, d, e, f = (float(x) for x in coef)
        hess = np.array([[2 * d, f], [f, 2 * e]])
        try:
            ev = np.linalg.eigvalsh(hess)
        except Exception:
            peak['refine_status'] = 'fit_failed'
            peak['q_refined'] = peak['q_original']
            return
        if not (ev[0] < -1e-12 and ev[1] < -1e-12):
            peak['refine_status'] = 'not_concave'
            peak['q_refined'] = peak['q_original']
            return
        try:
            delta = -np.linalg.solve(hess, np.array([b, c]))
        except Exception:
            peak['refine_status'] = 'fit_failed'
            peak['q_refined'] = peak['q_original']
            return
        if max(abs(delta[0]), abs(delta[1])) > 0.75:
            peak['refine_status'] = 'vertex_out_of_range'
            peak['q_refined'] = peak['q_original']
            return
        peak['refine_status'] = 'refined'
        peak['q_refined'] = (float(qx_axis[i] + delta[0] * (qx_axis[min(i + 1, len(qx_axis) - 1)] - qx_axis[max(i - 1, 0)]) / 2.0),
                             float(qy_axis[j] + delta[1] * (qy_axis[min(j + 1, len(qy_axis) - 1)] - qy_axis[max(j - 1, 0)]) / 2.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _fwhm(sig2d, qx_axis, qy_axis, i, j, dq_x, dq_y, peak):
        """沿 qx/qy 方向的线性插值 FWHM（实际 q 坐标）。
        半高区间触及网格边界 → 无法可靠求解 → None + 原因；
        峰宽小于采样步长 → resolution_limited=True。"""
        v = peak['height']
        half = 0.5 * v
        # 方向与截面正确配对：fwhm_qx 沿 qx 方向（固定 j，变 i，坐标 qx_axis）；
        # fwhm_qy 沿 qy 方向（固定 i，变 j，坐标 qy_axis）
        for axis, (coord, values, dq) in (
                ('fwhm_qx', (qx_axis, sig2d[:, j], dq_x)),
                ('fwhm_qy', (qy_axis, sig2d[i, :], dq_y))):
            vals = values
            above = np.isfinite(vals) & (vals >= half)
            idx = np.where(above)[0]
            if len(idx) == 0 or not np.isfinite(dq):
                peak[axis] = None
                peak['fwhm_note'] += f'{axis} 半高区间不存在；'
                continue
            lo, hi = int(idx[0]), int(idx[-1])
            if lo == 0 or hi == len(coord) - 1:
                peak[axis] = None
                peak['fwhm_note'] += f'{axis} 半高区间触及网格边界，无法可靠求解；'
                continue
            # 线性插值交点（实际 q 坐标）
            left = SpinStructureFactorPeakAnalyzer._interp_cross(
                coord, vals, lo - 1, lo, half)
            right = SpinStructureFactorPeakAnalyzer._interp_cross(
                coord, vals, hi, hi + 1, half)
            if left is None or right is None:
                peak[axis] = None
                peak['fwhm_note'] += f'{axis} 插值失败；'
                continue
            fwhm = abs(right - left)
            peak[axis] = float(fwhm)
            # 半高区间跨越的采样点少于 ~2 个步长时，线性插值 FWHM 由少数
            # 采样点决定（欠采样下插值值收敛到步长量级），标分辨率受限
            if fwhm < 2.0 * dq:
                peak['resolution_limited'] = True
                peak['fwhm_note'] += f'{axis} 峰宽 {fwhm:.4g} 不足 2 个采样步长 {dq:.4g}，分辨率受限；'

    # ------------------------------------------------------------------
    @staticmethod
    def _interp_cross(coord, vals, k1, k2, target):
        """相邻两点 (k1, k2) 间信号跨过 target 时的线性插值坐标。"""
        v1, v2 = vals[k1], vals[k2]
        if not (np.isfinite(v1) and np.isfinite(v2)):
            return None
        if (v1 - target) * (v2 - target) >= 0:
            return None
        t = (target - v1) / (v2 - v1)
        return float(coord[k1] + t * (coord[k2] - coord[k1]))

    # ------------------------------------------------------------------
    @staticmethod
    def plot_peaks(sf_result: dict, peak_report: dict, save_path: str = None,
                   show: bool = True):
        """2D 峰值可视化：|S(q)|（或 real 信号）颜色图 + 原始/细化峰位标记。"""
        import matplotlib.pyplot as plt
        q = np.asarray(sf_result['q'], dtype=float)
        S = np.asarray(sf_result['S_q'], dtype=complex)
        grid_q, qx_axis, qy_axis, _, _, _, _ = \
            SpinStructureFactorPeakAnalyzer._grid_from_q(q)
        nx, ny = grid_q.shape[0], grid_q.shape[1]
        sig = (np.abs(S) if peak_report.get('signal_used', 'abs') == 'abs'
               else S.real).reshape(nx, ny)
        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(qx_axis, qy_axis, sig.T, shading='auto', cmap='viridis')
        fig.colorbar(mesh, ax=ax, label=peak_report.get('signal_used', 'abs'))
        for p in peak_report.get('peaks', []):
            qo = p['q_original']
            ax.plot(qo[0], qo[1], 'o', mfc='none', mec='red', ms=10, mew=2,
                    label='grid peak' if p is peak_report['peaks'][0] else None)
            if p['q_refined'] is not None and p['refine_status'] == 'refined':
                qr = p['q_refined']
                ax.plot(qr[0], qr[1], 'x', color='cyan', ms=9, mew=2,
                        label='refined peak' if p is peak_report['peaks'][0] else None)
        ax.set_xlabel('qx')
        ax.set_ylabel('qy')
        ax.set_title(f"structure factor peaks (status: {peak_report.get('status')})")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=9)
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig
