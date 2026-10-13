# -*- coding: utf-8 -*-
"""
统一分析入口（D3.7：薄分派层）

调用方通过 (observable, operation) 选择分析任务，获得 AnalysisReport。
只做路由分派，不复制底层算法、不重构现有模块：

- 每条路由调用 Analysis_Integration 的既有包装函数
- 不捕获底层异常（原样传播）
- 未知 observable / 不支持的 operation 抛 ValueError（含支持列表）
- 缺少必需参数 / 多余参数 / 不支持 config 的链路收到 config
  均由 Python 签名机制抛 TypeError，不静默忽略
- 不新增批量入口、物理相态判定或证据融合
"""
from .Analysis_Contract import (
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_CORRELATION,
    OBSERVABLE_CHARGE_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_ENERGY,
)
from .Analysis_Integration import (
    analyze_with_report, structure_factor_report,
    peak_analysis_report, energy_extrapolation_report,
    charge_structure_factor_report,
)
from .Super_Correlation_Analyzer import SuperCorrDecayAnalyzer
from .Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer
from .Charge_Correlation_Analyzer import ChargeCorrDecayAnalyzer


# --------------------------------------------------------------------------
def _decay_route(analyzer_cls, observable):
    """衰减链路闭包：record_list 必需，config 透传，其余 kwargs 直通
    （如 Spin/Charge 的 N、a）。"""
    def route(record_list, config=None, **kwargs):
        return analyze_with_report(analyzer_cls, record_list, observable,
                                   config, **kwargs)
    return route


# (observable, operation) → 路由函数（签名即参数规则，无通用参数映射）
_ROUTES = {
    (OBSERVABLE_SUPERCONDUCTING_CORRELATION, 'decay'):
        _decay_route(SuperCorrDecayAnalyzer, OBSERVABLE_SUPERCONDUCTING_CORRELATION),
    (OBSERVABLE_SPIN_CORRELATION, 'decay'):
        _decay_route(SpinCorrDecayAnalyzer, OBSERVABLE_SPIN_CORRELATION),
    (OBSERVABLE_CHARGE_CORRELATION, 'decay'):
        _decay_route(ChargeCorrDecayAnalyzer, OBSERVABLE_CHARGE_CORRELATION),
    (OBSERVABLE_SPIN_STRUCTURE_FACTOR, 'estimate'): structure_factor_report,
    (OBSERVABLE_SPIN_STRUCTURE_FACTOR, 'peak_analysis'): peak_analysis_report,
    (OBSERVABLE_CHARGE_STRUCTURE_FACTOR, 'estimate'): charge_structure_factor_report,
    (OBSERVABLE_ENERGY, 'extrapolate'): energy_extrapolation_report,
}

# 只有一种操作的观测量 → 默认操作（保持常用调用简洁）；
# spin_structure_factor 有两种操作，无默认（必须显式指定，不猜测）
_DEFAULT_OPERATION = {
    OBSERVABLE_SUPERCONDUCTING_CORRELATION: 'decay',
    OBSERVABLE_SPIN_CORRELATION: 'decay',
    OBSERVABLE_CHARGE_CORRELATION: 'decay',
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR: 'estimate',
    OBSERVABLE_ENERGY: 'extrapolate',
}

# 支持 config 参数的链路（仅衰减分析；其余链路收到 config 抛 TypeError）
_CONFIG_SUPPORTED = {'decay'}


def run_analysis(observable, *, operation=None, config=None, **kwargs):
    """统一分析入口。

    参数
    ----
    observable : Analysis_Contract 的 OBSERVABLE_* 常量之一
    operation  : 操作名。单操作观测量可省略（默认值）；自旋结构因子
                 必须显式 'estimate' 或 'peak_analysis'
    config     : AnalyzerConfig，仅衰减链路支持；其余链路传入非 None
                 会抛 TypeError（不静默忽略）
    **kwargs   : 按所选路由的真实签名透传（缺参/多参由 Python
                 TypeError 自然报错）

    返回
    ----
    AnalysisReport（payload/状态映射/序列化语义与既有包装函数完全一致）
    """
    if observable not in {obs for obs, _ in _ROUTES}:
        raise ValueError(
            f"未知 observable: {observable!r}，支持: "
            f"{sorted({obs for obs, _ in _ROUTES})}")
    if operation is None:
        operation = _DEFAULT_OPERATION.get(observable)
        if operation is None:
            raise ValueError(
                f"observable={observable!r} 支持两种操作，必须显式指定 "
                f"operation='estimate' 或 operation='peak_analysis'")
    route = _ROUTES.get((observable, operation))
    if route is None:
        supported = [op for (obs, op) in _ROUTES if obs == observable]
        raise ValueError(
            f"不支持的 (observable, operation) 组合: "
            f"({observable!r}, {operation!r})，该 observable 支持: {supported}")
    if operation not in _CONFIG_SUPPORTED and config is not None:
        raise TypeError(
            f"observable={observable!r} operation={operation!r} "
            f"不支持 config 参数")
    if operation in _CONFIG_SUPPORTED:
        # 衰减路由闭包签名 (record_list, config=None, **kwargs)：
        # record_list 缺失/多余参数由 Python 签名机制报 TypeError
        return route(config=config, **kwargs)
    return route(**kwargs)
