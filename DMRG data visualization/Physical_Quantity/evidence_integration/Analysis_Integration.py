# -*- coding: utf-8 -*-
"""
统一契约集成层（SC / Spin 衰减分析器包装 + Spin Structure Factor 包装）

将现有分析器的输出包装为 AnalysisReport，供后续证据融合
（PhaseDiagnosis）消费。纯增量层：

- 不修改任何现有分析器的返回类型、数值算法或输出结构
- 数值路径与直接调用 analyze()/estimator/find_peaks() 完全一致
  （回归基线不受影响）
- 不捕获并吞掉分析器异常——原始异常（如空输入 ValueError）原样传播
- 通过 **kwargs 兼容 SC 与 Spin 适配层的 analyze 签名差异
  （SuperCorrDecayAnalyzer.analyze(record_list, config)；
   SpinCorrDecayAnalyzer.analyze(record_list, config, N, a)）
- 完整候选模型、模型比较、振荡分析、去趋势结果、稳定性扫描、
  结构因子全量 S(q) 与峰值细节等均保留在 report.payload
- 不把 SC 的 SU(2) 对称性检查结果（SuperCorrProcessor.symmetry_check
  的独立输出）伪装成衰减分析器自身的输出
- 不把峰值位置、峰数或峰值状态映射为物理相态结论
"""

import numpy as np

from .Analysis_Contract import (
    AnalysisReport, from_decay_result, from_structure_factor, from_peak_report,
    from_energy_extrapolation, from_density_profile,
    from_charge_spin_peak_alignment,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR, ANALYZER_SC_PAIR_SF,
    OBSERVABLE_SC_PAIRING_SYMMETRY, ANALYZER_SC_PAIRING_SYMMETRY,
    OBSERVABLE_LOCAL_DENSITY,
    STATUS_OK, STATUS_FAILED, STATUS_INCONCLUSIVE,
)
from ..data_records.Physical_Data_Record import DensityRecord
from ..physical_analysis.Density_Process import DensityProcessor
from ..physical_analysis.Spin_Structure_Factor_Analyzer import SpinStructureFactorPeakAnalyzer
from ..physical_analysis.Energy_Process import EnergyProcessor
from ..physical_analysis.Charge_Structure_Factor import ChargeStructureFactorEstimator
from ..physical_analysis.SC_Pair_Structure_Factor import SCPairStructureFactorEstimator

# 拟合质量指标：按 FitResult 实际字段名（r2/rss/rmse/aic/aicc/bic）提取，
# 存在才收录；不编造缺失数值
_FIT_QUALITY_FIELDS = ('r2', 'rss', 'rmse', 'aic', 'aicc', 'bic')


def analyze_with_report(analyzer_cls, record_list, observable,
                        config=None, **kwargs) -> AnalysisReport:
    """调用指定衰减分析器并包装为 AnalysisReport。

    参数
    ----
    analyzer_cls : SuperCorrDecayAnalyzer 或 SpinCorrDecayAnalyzer 类
    record_list : 已合成的关联记录列表（与分析器 analyze 的输入一致）
    observable  : Analysis_Contract 中的物理量常量
    config      : AnalyzerConfig（None 用分析器默认）
    **kwargs    : 透传给 analyze 的专属参数（如 Spin 的 N、a）

    返回
    ----
    AnalysisReport：payload 为原始 DecayAnalysisResult 对象引用。
    metadata 在契约映射之外补充（不修改 AnalysisReport 字段定义）：
    - metadata['param_errors']：envelope_fit.param_errors 的副本
      （仅拟合存在且非空时收录）
    - metadata['fit_quality']：envelope_fit 上实际存在的质量指标
      （r2/rss/rmse/aic/aicc/bic），存在才收录，值含 NaN 时如实保留
      （序列化时按契约转 null）
    拟合参数值仍只在 parameters；误差与质量指标不混入 parameters。
    """
    result = analyzer_cls.analyze(record_list, config, **kwargs)
    report = from_decay_result(result, analyzer_cls.__name__, observable)

    # 集成层补充：误差棒与拟合优度（独立存放，不污染 parameters）
    env = getattr(result, 'envelope_fit', None)
    if env is not None:
        param_errors = getattr(env, 'param_errors', None)
        if param_errors:
            report.metadata['param_errors'] = dict(param_errors)
        fit_quality = {}
        for name in _FIT_QUALITY_FIELDS:
            value = getattr(env, name, None)
            if value is not None:
                fit_quality[name] = value
        if fit_quality:
            report.metadata['fit_quality'] = fit_quality
    return report


def structure_factor_report(estimator_fn, estimator_arg, q_list, N, a=1.0,
                            **kwargs) -> AnalysisReport:
    """调用 SpinStructureFactorEstimator 入口并包装为 AnalysisReport。

    参数
    ----
    estimator_fn  : SpinStructureFactorEstimator.estimate_sz_sz_from_file
                    或 estimate_spin_dot_from_records（调用方传入实际入口，
                    不修改 estimator 签名）
    estimator_arg : 入口的第一个位置参数（文件路径 / record_list）
    q_list, N, a  : 与 estimator 入口一致的参数
    **kwargs      : 其余入口专属参数（如 estimate_sz_sz_from_file 的 source）

    语义约定：
    - status 与 data_restricted 独立：受限数据（is_restricted=True）仍可
      status='ok'（计算正常完成），两字段不互相推导
    - missing_pairs 非空（>0）时在 from_structure_factor 之外追加一条
      数据质量警告（None=未评估、空=无缺失时均不追加）；不修改契约模块
    - 不捕获底层异常（空文件路径 FileNotFoundError 等原样传播）
    - payload 保留 estimator 原始 dict 引用；complex 数组 S_q 在 payload
      中，默认序列化不展开；显式 include_payload=True 序列化含 complex
      的 payload 仍按契约严格拒绝（TypeError），不添加宽松兜底
    """
    sf_result = estimator_fn(estimator_arg, q_list, N, a, **kwargs)
    report = from_structure_factor(sf_result)
    missing = sf_result.get('missing_pairs')
    if missing:
        report.warnings.append(
            f"{len(missing)} 个预期关联对缺失（受限估计）")
    return report


def peak_analysis_report(sf_result, signal='abs', min_height_frac=0.2,
                         refine=True, q_shape=None) -> AnalysisReport:
    """调用 SpinStructureFactorPeakAnalyzer.find_peaks 并包装为 AnalysisReport。

    sf_result 同时传给 from_peak_report 作为 data_restricted 回退源
    （峰值报告缺 is_restricted 透传字段时按契约规则回退）。

    语义约定：
    - parameters 保持契约既有结构（n_peaks / 原始网格峰位 / 峰高）；
      q_refined 与 fwhm 等细节不并入 parameters，经 payload 访问
    - 峰值位置、峰数与峰值状态是数值层面的检测结果，不映射为
      AFM 或其他物理相态结论
    - 不捕获底层异常；不修改 find_peaks 签名
    """
    peak_dict = SpinStructureFactorPeakAnalyzer.find_peaks(
        sf_result, signal=signal, min_height_frac=min_height_frac,
        refine=refine, q_shape=q_shape)
    report = from_peak_report(peak_dict, sf_result)
    # q_grid 摘要入 metadata（与 charge_peak_analysis_report 同模式；
    # 供 O4 峰位对齐做可比性检查，缺失时保持原有行为不伪造）
    if isinstance(sf_result, dict) and sf_result.get('q_grid') is not None:
        report.metadata['q_grid'] = sf_result['q_grid']
    return report


def energy_extrapolation_report(records, power: int = 1,
                                threshold: float = 2.0) -> AnalysisReport:
    """调用 EnergyProcessor.extrapolate 并包装为 AnalysisReport。

    语义约定（与契约 from_energy_extrapolation 一致）：
    - 复用契约适配器，不重复实现拟合、异常点剔除或结果映射
    - 不捕获底层异常（records < 3 的 ValueError 等原样传播）
    - 不修改绘图/打印行为（extrapolate 的 plt.show 与候选异常点 print
      副作用保持不变）
    - 不通过解析 stdout 获取异常点信息；MAD 诊断经
      diagnose_outliers 结构化接入 metadata['outlier_diagnosis']
      （B3-A9，复用同一诊断算法，不复制）。候选异常点仅为敏感性
      信息：不进入 warnings、不改变正式 E_inf（恒为全点拟合）
    - 误差棒与拟合优度在 extrapolate 返回值中不存在，按契约保持
      空值语义，不编造
    - 单一外推结果不映射任何物理相态结论
    """
    e_inf = EnergyProcessor.extrapolate(records, power, threshold)
    report = from_energy_extrapolation(
        e_inf, [r.N for r in records], power, threshold)
    report.metadata['outlier_diagnosis'] = \
        EnergyProcessor.diagnose_outliers(records, power, threshold)
    return report


def charge_structure_factor_report(nf_path, corr_paths, q_list, N,
                                   a=1.0, sequences=None) -> AnalysisReport:
    """调用 ChargeStructureFactorEstimator.estimate_from_files 并包装为
    AnalysisReport。

    契约适配器 from_structure_factor 的 analyzer/observable 按自旋结构因子
    硬编码；本包装在适配后覆写为电荷标识（不改契约适配器签名）。
    其余语义与 from_structure_factor 一致：status 与 data_restricted 正交
    （受限数据仍 status='ok'）、missing_pairs 空、warnings 由实际计数构造、
    payload 保留原始 dict（含复数 S_q，默认不序列化）。
    """
    sf_result = ChargeStructureFactorEstimator.estimate_from_files(
        nf_path, corr_paths, q_list, N, a, sequences)
    report = from_structure_factor(sf_result)
    report.observable = OBSERVABLE_CHARGE_STRUCTURE_FACTOR
    report.analyzer = 'ChargeStructureFactorEstimator'
    # 实际 q 网格统计入 metadata（与 fit_quality 同模式，不改契约适配器；
    # 缺失时保持原有行为，不伪造默认值）
    if sf_result.get('q_grid') is not None:
        report.metadata['q_grid'] = sf_result['q_grid']
    return report


def charge_spin_peak_alignment_report(charge_peak_report,
                                      spin_peak_report) -> AnalysisReport:
    """电荷-自旋峰位对齐观测（O4，P1-G.4）：比较两个峰报告的峰位
    集合并包装为 AnalysisReport。

    输入：两个峰值 AnalysisReport（charge_structure_factor /
    spin_structure_factor 的 peak_analysis 路由输出）。

    语义约定：
    - 峰位取 q_original（网格坐标）：两报告同 q 网格时直接可比；
      q_refined 是抛物面外推，两物理量的外推差异会引入伪差，不使用
    - q_grid 可比性检查（摘要级，全等才可比）：任一缺失 →
      missing_q_grid；摘要不等 → inconsistent_q_grid；一致但任一侧
      无峰 → no_peaks；一致且双侧有峰 → consistent_q_grid。前两种
      情形不计算 q 距离（字段 None/空），warnings 明示——不可比是
      观测事实，不是计算失败也不是相级结论
    - 数值输出：逐电荷峰最近自旋峰（q_charge/q_spin/|dq|）及对称
      方向、min/median/max dq——峰位重合不构成耦合序支持，峰位
      偏离不构成反证
    - qy 周期折叠（P1-G.6）：XC 圆柱 y 方向周期 Ly=2√3·a，格点
      y 取值为 0.5√3·a 的整数倍 → qy 等价周期 Q_qy=2π/(0.5√3·a)
      =4π/(√3·a)（由基元模板 y 坐标推导验证；不是通用 2π/Ly）。
      Q_qy 从两峰报告 payload['meta']（_xc_geometry_meta）的 a 读取
      （Ly 交叉验证），任一侧缺失或 a 不一致 → 周期字段为 None、
      note 说明，原始距离不受影响。周期折叠后的距离以 *_periodic
      字段输出，原始距离字段语义不变（向后兼容，不混淆原始值与
      周期值）；periodic 距离同为描述性观测，不升级为耦合序或
      Stripe 证据
    - 不捕获底层异常（非 AnalysisReport 输入 TypeError 等原样传播）
    - 不修改 AUTHORIZED_RULES / 规则引擎 / 峰检测算法
    """
    for name, rep in (('charge', charge_peak_report),
                      ('spin', spin_peak_report)):
        if not isinstance(rep, AnalysisReport):
            raise TypeError(
                f'{name}_peak_report 必须为 AnalysisReport，'
                f'得到 {type(rep).__name__}')
    c_payload = charge_peak_report.payload or {}
    s_payload = spin_peak_report.payload or {}
    c_qgrid = charge_peak_report.metadata.get('q_grid')
    s_qgrid = spin_peak_report.metadata.get('q_grid')
    c_peaks = [tuple(p['q_original']) for p in c_payload.get('peaks', [])
               if p.get('q_original') is not None]
    s_peaks = [tuple(p['q_original']) for p in s_payload.get('peaks', [])
               if p.get('q_original') is not None]

    align = {
        'n_charge_peaks': len(c_peaks),
        'n_spin_peaks': len(s_peaks),
        'charge_peak_nearest_spin': [],
        'spin_peak_nearest_charge': [],
        'min_dq': None, 'median_dq': None, 'max_dq': None,
        'charge_peak_nearest_spin_periodic': [],
        'spin_peak_nearest_charge_periodic': [],
        'min_dq_periodic': None, 'median_dq_periodic': None,
        'max_dq_periodic': None,
        'qy_period': None,
        'periodic_distance_available': False,
        'periodic_note': '',
        'charge_q_grid': c_qgrid, 'spin_q_grid': s_qgrid,
    }
    if c_qgrid is None or s_qgrid is None:
        align['comparability'] = 'missing_q_grid'
        align['comparability_note'] = \
            '一侧或两侧峰报告缺少 q_grid 元数据，无法确认可比性，不计算峰位距离'
    elif c_qgrid != s_qgrid:
        align['comparability'] = 'inconsistent_q_grid'
        align['comparability_note'] = \
            '两侧 q_grid 摘要不一致（n_qx/n_qy/端点），峰位不可比，不计算距离'
    elif not c_peaks or not s_peaks:
        align['comparability'] = 'no_peaks'
        align['comparability_note'] = \
            'q_grid 一致但至少一侧无峰，无对齐对象'
    else:
        align['comparability'] = 'consistent_q_grid'
        align['comparability_note'] = \
            'q_grid 摘要一致且双侧有峰，峰位距离已计算（q_original 网格坐标）'
        for qc in c_peaks:
            best = min(s_peaks,
                       key=lambda qs: (qs[0] - qc[0]) ** 2
                       + (qs[1] - qc[1]) ** 2)
            dq = float(((best[0] - qc[0]) ** 2
                        + (best[1] - qc[1]) ** 2) ** 0.5)
            align['charge_peak_nearest_spin'].append(
                [list(qc), list(best), dq])
        for qs in s_peaks:
            best = min(c_peaks,
                       key=lambda qc: (qc[0] - qs[0]) ** 2
                       + (qc[1] - qs[1]) ** 2)
            dq = float(((best[0] - qs[0]) ** 2
                        + (best[1] - qs[1]) ** 2) ** 0.5)
            align['spin_peak_nearest_charge'].append(
                [list(qs), list(best), dq])
        # min/median/max dq 统计双向最近邻距离的并集（电荷→自旋 与
        # 自旋→电荷 两方向都计入：一侧峰多于另一侧时不会漏掉
        # 无对齐对象的方向）
        dqs = ([r[2] for r in align['charge_peak_nearest_spin']]
               + [r[2] for r in align['spin_peak_nearest_charge']])
        align['min_dq'] = float(min(dqs))
        align['median_dq'] = float(np.median(dqs))
        align['max_dq'] = float(max(dqs))

        # ---- qy 周期折叠距离（P1-G.6；原始距离字段保持不变） ----
        def _meta_a(rep):
            meta = (rep.payload or {}).get('meta') or {}
            try:
                a = float(meta.get('a'))
                ly = float(meta.get('Ly'))
            except (TypeError, ValueError):
                return None
            # Ly 与 a 交叉验证（XC 构型 Ly = 2√3·a，容差 1e-6 相对）
            if abs(ly - 2.0 * np.sqrt(3.0) * a) <= 1e-6 * max(ly, 1.0):
                return a
            return None

        a_c = _meta_a(charge_peak_report)
        a_s = _meta_a(spin_peak_report)
        if a_c is None or a_s is None:
            align['periodic_note'] = \
                '峰报告缺少 _xc_geometry_meta（a/Ly）或 Ly 与 a 不一致，' \
                '周期折叠距离不可用（原始距离不受影响）'
        elif abs(a_c - a_s) > 1e-9 * max(a_c, 1.0):
            align['periodic_note'] = \
                f'两侧几何 a 不一致（{a_c} vs {a_s}），周期折叠距离不可用'
        else:
            qy_period = 4.0 * np.pi / (np.sqrt(3.0) * a_c)
            align['qy_period'] = float(qy_period)
            align['periodic_distance_available'] = True
            align['periodic_note'] = (
                f'qy 等价周期 Q=2π/(0.5√3·a)=4π/(√3·a)='
                f'{qy_period:.6g}（由 XC 基元 y 格点步长推导；'
                'a 取自两报告的 _xc_geometry_meta）')

            def _fold(dy):
                return (dy + qy_period / 2.0) % qy_period - qy_period / 2.0

            def _nearest(q, pool):
                best, best_d = None, float('inf')
                for qp in pool:
                    d = float(((qp[0] - q[0]) ** 2
                               + _fold(qp[1] - q[1]) ** 2) ** 0.5)
                    if d < best_d:
                        best, best_d = qp, d
                return best, best_d

            for qc in c_peaks:
                best, d = _nearest(qc, s_peaks)
                align['charge_peak_nearest_spin_periodic'].append(
                    [list(qc), list(best), d])
            for qs in s_peaks:
                best, d = _nearest(qs, c_peaks)
                align['spin_peak_nearest_charge_periodic'].append(
                    [list(qs), list(best), d])
            dqs_p = ([r[2] for r in
                      align['charge_peak_nearest_spin_periodic']]
                     + [r[2] for r in
                        align['spin_peak_nearest_charge_periodic']])
            align['min_dq_periodic'] = float(min(dqs_p))
            align['median_dq_periodic'] = float(np.median(dqs_p))
            align['max_dq_periodic'] = float(max(dqs_p))
    return from_charge_spin_peak_alignment(align)


def charge_peak_analysis_report(sf_result, signal='abs',
                                min_height_frac=0.2, refine=True,
                                q_shape=None) -> AnalysisReport:
    """电荷结构因子的数值峰值分析：复用 SpinStructureFactorPeakAnalyzer.
    find_peaks（零修改）并包装为 AnalysisReport。

    语义约定：
    - find_peaks 的输入契约是通用 compute_structure_factor dict
      （q / S_q 同长）；电荷 SF 经同一通用核输出（归一化同为
      1/N），数值检测约定同源。但电荷侧与自旋侧不默认完全相同：
      definition 为 'charge_density_connected'（vs 'sz_sz'/'spin_dot'）
      且电荷 SF 恒 is_restricted=True（200×200 关联矩阵不存在），
      由峰报告的透传字段区分
    - 峰位置 / 峰高 / FWHM（半高线性插值，沿 qx/qy 截面）是数值
      检测结果，不映射为 CDW/Stripe 或其他相态结论；min_height_frac
      是数值检测参数，不是物理阈值
    - analyzer 保持算法出处 'SpinStructureFactorPeakAnalyzer'；
      observable 覆写为 charge_structure_factor（与自旋峰值条目区分）
    - metadata 在 from_peak_report 既有映射之外追加 sf_result 的
      q_grid 摘要（存在时，与 charge_structure_factor_report 同模式）
    - 不捕获底层异常；不修改 find_peaks 及其 FWHM/细化逻辑
    """
    peak_dict = SpinStructureFactorPeakAnalyzer.find_peaks(
        sf_result, signal=signal, min_height_frac=min_height_frac,
        refine=refine, q_shape=q_shape)
    report = from_peak_report(peak_dict, sf_result)
    report.observable = OBSERVABLE_CHARGE_STRUCTURE_FACTOR
    if isinstance(sf_result, dict) and sf_result.get('q_grid') is not None:
        report.metadata['q_grid'] = sf_result['q_grid']
    return report


def sc_pair_structure_factor_report(direction, refbond_dir, q_list, N,
                                    a=1.0, include_channels=True) -> AnalysisReport:
    """调用 SCPairStructureFactorEstimator.estimate_from_files 并包装为
    AnalysisReport。

    契约适配器 from_structure_factor 的 analyzer/observable 按自旋结构因子
    硬编码；本包装在适配后覆写为 SC 配对标识（不改契约适配器签名）。
    语义与 from_structure_factor 一致：status 与 data_restricted 正交
    （受限数据恒 is_restricted=True，仍 status='ok'）、warnings 由实际
    计数构造、payload 保留原始 dict（含复数 S_q 与 channel_sf，默认不
    序列化）。
    SC 专属身份字段（存在才收录，缺失不编造）入 metadata：direction /
    ref_bond / target_bond_count / channel_target_consistency；q_grid 与
    电荷包装同模式。
    不捕获底层异常（文件缺失 FileNotFoundError / 通道集合不一致
    ValueError 等原样传播）。
    """
    sf_result = SCPairStructureFactorEstimator.estimate_from_files(
        direction, refbond_dir, q_list, N, a, include_channels)
    report = from_structure_factor(sf_result)
    report.observable = OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR
    report.analyzer = ANALYZER_SC_PAIR_SF
    if sf_result.get('q_grid') is not None:
        report.metadata['q_grid'] = sf_result['q_grid']
    for k in ('direction', 'channel', 'ref_bond', 'target_bond_count'):
        if k in sf_result:
            report.metadata[k] = sf_result[k]
    if 'channel_target_consistency' in sf_result:
        report.metadata['channel_target_consistency'] = \
            sf_result['channel_target_consistency']
    return report


def density_profile_report(nf_path, N, a=1.0) -> AnalysisReport:
    """读取 nf.json 局域密度数据并包装为 AnalysisReport（观测层）。

    链：open(nf_path) → DensityRecord.extract_data →
    DensityProcessor.profile_summary → from_density_profile。

    语义约定：
    - 只产生数值观测（逐格点向量/行平均/子晶格平均/晶胞分组平均 +
      几何边界元数据），不输出任何相级判定（不判 Stripe）
    - 不复制或修改 DensityProcessor 的可视化逻辑（本包装只调用
      profile_summary 数值摘要）
    - 不捕获底层异常：文件缺失 FileNotFoundError、畸形 JSON
      ValueError、格点数不匹配 ValueError、N 非 8 倍数 ValueError
      均原样传播（不静默补默认值）
    - payload 保留 profile_summary 原始 dict 引用（默认不序列化）
    """
    with open(nf_path, encoding='utf-8') as f:
        records = DensityRecord.extract_data(f.read(), N, a)
    profile = DensityProcessor.profile_summary(records, N, a)
    return from_density_profile(profile)


def sc_pairing_symmetry_report(report_dict) -> AnalysisReport:
    """把 PairingSymmetryAnalyzer 的原生输出包装为 AnalysisReport
    （P1-E.2 / P1-E.3.1 双输入结构适配）。

    支持三种输入结构（按实际输出核对，不按设计文档推测）：
    1. run_stage4 输出：顶层含 magnitude/sign/phase 三通道状态 +
       run_status
    2. run_candidate_observations 输出：顶层含 layers（各层 status/
       blocking_reasons/quality_warnings/sign_evidence）与
       supported_candidate_status；**无三通道顶层字段、无 phase
       通道**（phase 能力限制不在此结构，不编造）
    3. 复合输入 {'stage4': <run_stage4 输出>,
                'observations': <run_candidate_observations 输出>}：
       三通道状态取自 stage4、层摘要与质量警告取自 observations——
       完整证据链推荐使用此形式

    语义约定：
    - 输入 None → status=failed（沿用契约适配器的失败约定，不伪造）；
      非 None 且 run_status=='ok' 或缺失 → status=ok；其他 run_status
      值 → STATUS_INCONCLUSIVE（计算完成但运行状态异常，保守不冒充
      成功）。复合输入中 run_status 取自 stage4 部分
    - payload：原生 dict 引用（默认不序列化）；缺失通道保持缺失，
      不自动补成成功或失败结论
    - metadata 仅记录实际存在的字段：三通道状态摘要（magnitude_state/
      sign_state/phase_state，仅 stage4/复合输入可得）、
      supported_candidate_status、cross_element_sign_structure（各层
      sign_evidence）、layer_states（observations 各层 status/阻断数/
      支持判定状态摘要）——缺失键不编造
    - warnings：各层 quality_warnings 展平去重副本
    - data_restricted=None：保留未知语义，不默认解释为不受限
    - phase=blocked_by_data 是数据能力限制（实值数据无复相位），
      不解释为 d+id/TRSB 反证；不注册物理判据、不改变冻结状态语义
    """
    observable = OBSERVABLE_SC_PAIRING_SYMMETRY
    analyzer = ANALYZER_SC_PAIRING_SYMMETRY
    if report_dict is None:
        return AnalysisReport(analyzer=analyzer, observable=observable,
                              status=STATUS_FAILED)
    stage4_part = report_dict.get('stage4') \
        if 'stage4' in report_dict and 'observations' in report_dict \
        else (report_dict if 'magnitude' in report_dict else None)
    obs_part = report_dict.get('observations') \
        if 'stage4' in report_dict and 'observations' in report_dict \
        else (report_dict if 'layers' in report_dict else None)

    run_status = None
    if isinstance(stage4_part, dict):
        run_status = stage4_part.get('run_status')
    elif isinstance(obs_part, dict):
        run_status = obs_part.get('run_status')
    if run_status is None or run_status == 'ok':
        status = STATUS_OK
    else:
        status = STATUS_INCONCLUSIVE

    metadata = {}
    # 三通道状态：仅 stage4 部分存在（observations 无顶层通道，不编造）
    if isinstance(stage4_part, dict):
        for ch in ('magnitude', 'sign', 'phase'):
            ch_cfg = stage4_part.get(ch)
            if isinstance(ch_cfg, dict):
                metadata[f'{ch}_state'] = ch_cfg.get('state')
    # observations 部分摘要
    if isinstance(obs_part, dict):
        if obs_part.get('supported_candidate_status') is not None:
            metadata['supported_candidate_status'] = \
                obs_part['supported_candidate_status']
        cs = set()
        layer_states = {}
        for L, linfo in (obs_part.get('layers') or {}).items():
            if not isinstance(linfo, dict):
                continue
            layer_states[str(L)] = {
                'status': linfo.get('status'),
                'n_blocking': len(linfo.get('blocking_reasons') or []),
                'supported_candidate_status': linfo.get(
                    'supported_candidate_status'),
            }
            se = linfo.get('sign_evidence') or {}
            if se.get('cross_element_sign_structure') is not None:
                cs.add(se['cross_element_sign_structure'])
        if layer_states:
            metadata['layer_states'] = layer_states
        if cs:
            metadata['cross_element_sign_structure'] = (
                cs.pop() if len(cs) == 1 else sorted(cs))
    warnings = []
    for linfo in ((obs_part or {}).get('layers') or {}).values():
        for w in (linfo or {}).get('quality_warnings') or []:
            if w not in warnings:
                warnings.append(w)
    return AnalysisReport(
        analyzer=analyzer, observable=observable, status=status,
        classification=None, parameters={}, evidence=None,
        data_restricted=None, warnings=warnings, metadata=metadata,
        payload=report_dict)


def sc_pairing_symmetry_analysis_report(condition, config=None) \
        -> AnalysisReport:
    """配对对称性标准入口（P1-E.4）：从同一 condition 计算 stage4 与
    observations 并复合包装。

    - condition：build_comparable_set 的输入 dict（doping/N/bond_dim/
      super_corr_root，refbonds 可选）
    - config：PairingSymmetryConfig（None → 默认配置）
    - **同源构造性保证**：stage4 与 observations 由同一个
      build_comparable_set 返回对象生成（本函数内部），不存在外部
      组合不同数据源的风险——来源一致性由构造方式保证，不依赖调用方
    - 不捕获底层异常（目录缺失 FileNotFoundError、分析失败等原样传播）
    - 不注册物理判据；AUTHORIZED_RULES 保持空
    """
    # 函数级延迟导入：避免 Analysis_Integration ↔ Pairing_Symmetry_Analyzer 环
    from ..physical_analysis.Pairing_Symmetry_Analyzer import (
        build_comparable_set, run_stage4, run_candidate_observations,
        PairingSymmetryConfig)
    config = config if config is not None else PairingSymmetryConfig()
    rep = build_comparable_set(condition, config)
    stage4 = run_stage4(rep, config)
    observations = run_candidate_observations(rep, config)
    return sc_pairing_symmetry_report(
        {'stage4': stage4, 'observations': observations})
