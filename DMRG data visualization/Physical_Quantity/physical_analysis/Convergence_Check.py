# -*- coding: utf-8 -*-
"""
DMRG 收敛诊断（P1 阶段 1：叶子模块，只依赖 numpy）。

只回答"数值是否稳定（bond dimension 是否还需增加）"，**不回答
"能否判相"**——判相还需要尺寸序列等其他证据。结论三档：
- converged：全部可用检查通过（按当前阈值，数值稳定）
- needs_higher_D：存在未通过项（数值未收敛，建议增加 D），或
  证据不足无法对比（单档数据）
- suspicious：出现数值振荡信号（如不同 D 档分类不一致）

阈值纪律：DEFAULT_THRESHOLDS 是默认值（部分来自项目经验观察），
**未获物理授权**——报告必须注明阈值来源状态；调用方可显式传入
thresholds 覆盖。
"""
import numpy as np

# 默认阈值（未获物理授权；e0_rel_tol 为工程惯例量级，
# su2_max 沿用 symmetry_check 的经验"警惕"线 1e-3，其余为默认设定）
DEFAULT_THRESHOLDS = {
    'e0_rel_tol': 1e-4,       # 相邻 D 档相对能量差上限
    'truncerr_max': 1e-7,     # 末段截断误差上限
    'su2_max': 1e-3,          # SU(2) 一致性残差上限（经验警惕线）
    'param_n_sigma': 2.0,     # 参数误差内一致：|Δ| ≤ n·max(σ1, σ2)
}

VERDICT_CONVERGED = 'converged'
VERDICT_NEEDS_HIGHER_D = 'needs_higher_D'
VERDICT_SUSPICIOUS = 'suspicious'
VERDICTS = (VERDICT_CONVERGED, VERDICT_NEEDS_HIGHER_D, VERDICT_SUSPICIOUS)


def _mk(name, passed, evidence, note=''):
    """检查条目：passed=True/False/None（None = NA 证据不足）。"""
    return {'name': name, 'passed': passed, 'evidence': evidence,
            'note': note}


def _dims_sorted(mapping):
    return sorted((mapping or {}).items(), key=lambda kv: _dim_key(kv[0]))


def _dim_key(d):
    try:
        return int(d)
    except (TypeError, ValueError):
        return 0


def check_e0_saturation(e0_by_dim, thresholds):
    """E0 饱和：相邻 D 档相对能量差 ≤ e0_rel_tol。"""
    th = thresholds
    dims = _dims_sorted(e0_by_dim)
    if len(dims) < 2:
        return _mk('e0_saturation', None,
                   '仅单个 D 档，无法对比（需要更多档位）')
    worst = 0.0
    worst_pair = None
    for (d1, e1), (d2, e2) in zip(dims, dims[1:]):
        rel = abs(e2 - e1) / max(abs(e1), abs(e2), 1e-30)
        if rel > worst:
            worst, worst_pair = rel, (d1, d2)
    ok = worst <= th['e0_rel_tol']
    return _mk('e0_saturation', ok,
               f'相邻 D 档最大相对能量差 {worst:.3g}'
               f'（阈值 {th["e0_rel_tol"]:.0e}）'
               + ('' if worst_pair is None else
                  f'，出现在 {worst_pair[0]}→{worst_pair[1]}'))


def check_truncerr(truncerr_by_dim, thresholds):
    """TruncErr：各档末段截断误差 ≤ truncerr_max。"""
    th = thresholds
    if not truncerr_by_dim:
        return _mk('truncerr', None, '无 TruncErr 记录')
    fails = [(d, v) for d, v in _dims_sorted(truncerr_by_dim)
             if v > th['truncerr_max']]
    ok = not fails
    return _mk('truncerr', ok,
               f'{len(truncerr_by_dim)} 档 TruncErr，'
               f'超限档: {fails if fails else "无"}'
               f'（阈值 {th["truncerr_max"]:.0e}）')


def check_su2(su2_by_dim, thresholds):
    """SU(2) 一致性残差：各档 max(err_ad, err_bc) ≤ su2_max。

    su2_by_dim 的值是标量（每档已取 max(err_ad, err_bc)）。
    """
    th = thresholds
    if not su2_by_dim:
        return _mk('su2', None, '无 SU(2) 残差数据')
    worst = 0.0
    worst_d = None
    for d, w_val in _dims_sorted(su2_by_dim):
        w = float(w_val)
        if w > worst:
            worst, worst_d = w, d
    ok = worst <= th['su2_max']
    return _mk('su2', ok,
               f'各档最大残差 {worst:.3g}（D={worst_d}；'
               f'阈值 {th["su2_max"]:.0e}——残差随 D 减小是收敛的'
               '正向信号，但归因不唯一，见协议锁定清单 §2.1）')


def check_classification_consistency(classes_by_dim, thresholds):
    """分类一致：各 D 档分类完全相同。"""
    if len(classes_by_dim or {}) < 2:
        return _mk('classification_consistency', None,
                   '仅单个 D 档，无法对比')
    classes = set(classes_by_dim.values())
    ok = len(classes) == 1
    return _mk('classification_consistency', ok,
               f'各档分类: {dict(_dims_sorted(classes_by_dim))}'
               + ('（一致）' if ok else '（不一致——数值振荡信号）'))


def check_parameter_consistency(params_by_dim, errors_by_dim, thresholds):
    """参数误差内一致：同名参数 |Δ| ≤ n_sigma·max(σ1, σ2)。

    params_by_dim: {D: {param_name: value}}
    errors_by_dim: {D: {param_name: sigma}}（可为空 → 该项 NA）
    """
    th = thresholds
    dims = _dims_sorted(params_by_dim)
    if len(dims) < 2:
        return _mk('parameter_consistency', None,
                   '仅单个 D 档，无法对比')
    names = set()
    for _, p in dims:
        names |= set((p or {}).keys())
    bad = []
    na = True
    for name in sorted(names):
        vals = [(d, p.get(name)) for d, p in dims
                if p.get(name) is not None]
        sigs = {d: (errors_by_dim or {}).get(d, {}).get(name)
                for d, _ in vals}
        if len(vals) < 2 or any(sigs[d] is None for d, _ in vals):
            continue
        na = False
        (d1, v1), (d2, v2) = vals[0], vals[1]
        tol = th['param_n_sigma'] * max(sigs[d1], sigs[d2])
        if abs(v2 - v1) > tol:
            bad.append(f'{name}: {v1:.4g} vs {v2:.4g}（容差 {tol:.3g}）')
    if na:
        return _mk('parameter_consistency', None,
                   '参数或误差缺失，无法对比')
    ok = not bad
    return _mk('parameter_consistency', ok,
               ('各参数在误差内一致' if ok
                else f'超容差参数: {bad}；n_sigma={th["param_n_sigma"]}'))


def convergence_verdict(e0_by_dim=None, truncerr_by_dim=None,
                        su2_by_dim=None, classes_by_dim=None,
                        params_by_dim=None, param_errors_by_dim=None,
                        thresholds=None) -> dict:
    """聚合五项检查 → 三档结论 + 证据明细。

    阈值来源：thresholds=None 时用 DEFAULT_THRESHOLDS（默认值，
    **未获物理授权**——报告层必须标注）。
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    checks = [
        check_e0_saturation(e0_by_dim, th),
        check_truncerr(truncerr_by_dim, th),
        check_su2(su2_by_dim, th),
        check_classification_consistency(classes_by_dim, th),
        check_parameter_consistency(params_by_dim, param_errors_by_dim, th),
    ]
    failed = [c for c in checks if c['passed'] is False]
    n_na = sum(1 for c in checks if c['passed'] is None)
    n_ok = sum(1 for c in checks if c['passed'] is True)
    if any(c['name'] == 'classification_consistency'
           and c['passed'] is False for c in checks):
        verdict = VERDICT_SUSPICIOUS
        reason = ('分类随 D 变化：数值振荡信号，优先排查收敛性'
                  '（分类变化不等于物理变化）')
    elif failed:
        verdict = VERDICT_NEEDS_HIGHER_D
        reason = (f'{len(failed)} 项未通过'
                  f'（{[c["name"] for c in failed]}）：'
                  '数值未收敛，建议增加 bond dimension')
    elif n_ok and n_ok + n_na == len(checks):
        verdict = VERDICT_CONVERGED
        reason = f'全部 {n_ok} 项可用检查通过（{n_na} 项证据不足）'
    else:
        verdict = VERDICT_NEEDS_HIGHER_D
        reason = ('证据不足：单档数据无法对比（需要更多 D 档），'
                  '不能下"已收敛"结论')
    return {'verdict': verdict, 'reason': reason, 'checks': checks,
            'thresholds': dict(th),
            'threshold_status': 'defaults_not_authorized'}
