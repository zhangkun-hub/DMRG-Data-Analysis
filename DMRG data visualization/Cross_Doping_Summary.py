# -*- coding: utf-8 -*-
"""
跨 doping 受控数值汇总脚本（PhaseDiagnosis-P4.3）

将指定 doping 集合（默认 7 个：d1_10/d1_20/d1_25/d1_100/d2_25/d3_20/d3_25）
在共同条件（默认 N=200、bond dimension=1500）下可用的分析结果，逐条运行
既有统一入口 run_analysis 并汇总为结构化 CSV + JSON 输出，供后续人工审查
与受控并列比较。

覆盖的分析（按实际存在的数据，不做完整性假定）：
- 超导关联衰减：每个发现的参考键目录 × A/B/C 三个键方向（12 文件合成
  <Δ†Δ> 后经 SuperCorrDecayAnalyzer 分析）
- 自旋结构因子：sz_sz（measure_49/sz.json，源点 0）+ spin_dot
  （measure_49/50 三分量合成，源点 49/50）的受限估计 + 峰值分析
- 电荷结构因子：nf + nf49nf + nf50nf 的连通形式受限估计

职责边界（严格）：
- 只做「运行既有统一入口 + 汇总 + 机械可比性分组」，不引入任何新拟合
  或新统计量；不修改既有分析器 / 契约 / P1/P2/P3 实现
- 不判定单调性、物理趋势、物理相态或趋势显著性：数值随 doping 的变化
  只作为观测结果逐行呈现，本脚本不生成任何跨 doping 物理结论
- 可比性分组是机械条件检查（N / bond dimension / 参考键 / 键方向 /
  源点集合 / q_grid 摘要）。'direct' 仅表示分析条件一致，不表示
  DMRG 收敛精度相同，也不表示物理结论可比
- q_grid 只依据现有审计能力判断：SF 估计行复用 P2 摘要级逐对审计；
  峰值行按其对应 SF 结果的 q 坐标（payload['q']）由本脚本计算摘要。
  摘要一致时明确标记「摘要一致」，不声称完整坐标已验证一致
- 不同参考键、不同源点集合不自动合并为同一比较组；N=400 数据绝不与
  N=200 混组（本脚本只处理命令行指定的单一 (N, bond_dim) 组合，
  需要 N=400 时单独运行产生独立输出）
- 缺失数据 / 分析失败 / 受限数据保留明确状态（failed 行 + failed_type /
  failed_message），不静默跳过、不伪装成功
- 重复运行确定性：输出整体覆盖重写，绝不追加旧记录
- 复用 PhaseDiagnosis P1/P2/P3：collect_evidence / audit_comparability /
  evidence_sufficiency_check（rules=None → overall='unknown'，如实记录
  未提供经确认的充分性标准；不编造置信度）

运行方式（在 DMRG data visualization/ 目录下）：
    python Cross_Doping_Summary.py --outdir cross_doping_summary_output

输出（--outdir 指定目录，默认 <脚本目录>/cross_doping_summary_output/）：
- cross_doping_summary.csv：一行一条分析（含失败条目），固定列序；
  不适用字段留空（缺失不编造默认值）
- cross_doping_summary.json：config 快照 + rows + 可比性分组 +
  PhaseDiagnosisReport（P1/P2/P3 完整结果）+ 语义说明
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Physical_Quantity.Analysis_Contract import (
    _convert, STATUS_FAILED,
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
)
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, audit_comparability, evidence_sufficiency_check)
from Physical_Quantity.Physical_Data_Record import (
    SuperCorrRecord, CorrelationRecord)
from Physical_Quantity.Correlation_Process import CorrelationProcessor
from Physical_Quantity.Super_Correlation_Process import SuperCorrProcessor
from Physical_Quantity.Spin_Structure_Factor import (
    xc_momentum_grid, SpinStructureFactorEstimator, _q_grid_stats)
from Physical_Quantity.Correlation_Decay_Analyzer import AnalyzerConfig

DEFAULT_DOPINGS = ['d1_10', 'd1_20', 'd1_25', 'd1_100', 'd2_25', 'd3_20', 'd3_25']
DEFAULT_T_J = 't_J_3'
DEFAULT_BOND_DIM = '1500'
DEFAULT_N = 200

SC_KEYS = ('A', 'B', 'C')
SC_CHANNELS = ('a', 'b', 'c', 'd')
_SUPERCORR_DIR_RE = re.compile(r'^super_corr_(\d+)_(\d+)$')

ANALYZER_SC_DECAY = 'SuperCorrDecayAnalyzer'
ANALYZER_SPIN_SF = 'SpinStructureFactorEstimator'
ANALYZER_PEAK = 'SpinStructureFactorPeakAnalyzer'
ANALYZER_CHARGE_SF = 'ChargeStructureFactorEstimator'
_SF_ESTIMATOR_ANALYZERS = (ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF)

# 源点/参考键身份来源标记（P3 修复；取值约定见 write_outputs 语义块）
KIND_REFBOND_DIR = 'refbond_dir'                # SC：参考键目录名
KIND_PAYLOAD_FIELD = 'payload_field'            # 电荷：payload 实测字段
KIND_SCRIPT_CONVENTION = 'script_convention'    # 自旋：脚本输入约定
# 代表"源点身份"的标记（比较逻辑据此识别源点差异，不按分析器名称推测）
_SOURCE_KINDS = (KIND_PAYLOAD_FIELD, KIND_SCRIPT_CONVENTION)

# 可比性分组四档：前三档对应任务约定（条件一致 / 已知差异 / 关键条件
# 不一致或未知），'sole' 为无同类条目（无并列比较对象）的扩展标注
BUCKET_DIRECT = 'direct'
BUCKET_CAUTIOUS = 'cautious'
BUCKET_NOT_DIRECT = 'not_direct'
BUCKET_SOLE = 'sole'
_BUCKETS = (BUCKET_DIRECT, BUCKET_CAUTIOUS, BUCKET_NOT_DIRECT, BUCKET_SOLE)

# CSV 固定列序（新字段须同步更新本元组与 build_rows 的 row 模板）
CSV_COLUMNS = (
    'entry_index', 'doping', 'N', 'bond_dim', 'observable', 'analyzer',
    'refbond_or_source', 'source_identity_kind', 'key_or_definition',
    'status', 'classification', 'evidence', 'data_restricted',
    'fit_model', 'fit_r_min', 'fit_r_max', 'envelope_mode', 'verdict',
    'n_points', 'n_distances',
    'params_json', 'fit_quality_json', 'param_errors_json',
    'n_peaks', 'peak_estimate_entry_index', 'measured_pair_count',
    'source_coverage',
    'imaginary_residual', 'q_grid_json', 'diagonal_source_sites_json',
    'warnings_json',
    'failed_type', 'failed_message',
    'comparability_bucket', 'comparability_reason',
)


# ==========================================================================
# 数据加载（默认实现；测试可注入 Loaders 替换，不经文件）
# ==========================================================================
def load_super_corr_records(refbond_dir, refbond, key, N, a):
    """读取一个参考键目录下 key 方向的 4 通道文件并合成为 <Δ†Δ> 记录。
    文件名约定：scs{key}{ch}_fix_refbond_{i}_{j}.json。"""
    channels = {}
    for ch in SC_CHANNELS:
        path = os.path.join(
            refbond_dir,
            f'scs{key}{ch}_fix_refbond_{refbond[0]}_{refbond[1]}.json')
        with open(path, encoding='utf-8') as f:
            channels[ch] = SuperCorrRecord.extract_data(f.read(), N, a)
    return SuperCorrProcessor.Combined_super_correlation(
        channels['a'], channels['b'], channels['c'], channels['d'])


def load_spin_dot_records(phys_dir, N, a):
    """读取 measure_49/50 的自旋三分量并合成为 <S_i·S_j> 记录
    （与既有 notebook / 端到端测试同款 sequence 约定）。"""
    recs = []
    for measure, ver, seq in (('measure_49', '49', range(1, N + 100, 4)),
                              ('measure_50', '50', range(2, N + 100, 4))):
        base = os.path.join(phys_dir, measure)
        sz = CorrelationRecord.extract_data(
            open(os.path.join(base, f'sz{ver}sz.json'), encoding='utf-8').read(),
            seq, N, a)
        su = CorrelationRecord.extract_data(
            open(os.path.join(base, f'sp{ver}sm.json'), encoding='utf-8').read(),
            seq, N, a)
        sd = CorrelationRecord.extract_data(
            open(os.path.join(base, f'sm{ver}sp.json'), encoding='utf-8').read(),
            seq, N, a)
        recs.extend(CorrelationProcessor.Combined_spin_correlation(sz, su, sd))
    return recs


class Loaders:
    """可注入的加载器集合（测试用 mock 替换，避免依赖完整科研数据目录）。"""

    def __init__(self, super_corr=None, spin_dot=None):
        self.super_corr = super_corr or load_super_corr_records
        self.spin_dot = spin_dot or load_spin_dot_records


# ==========================================================================
# 单个 doping 的分析运行（复用统一入口 run_analysis）
# ==========================================================================
def analyze_doping(doping, *, root, t_j, bond_dim, N, a, q_list,
                   config=None, refbonds=None, loaders=None) -> list:
    """对单个 doping 运行全部可用分析，返回 outcome 列表（顺序确定）。

    outcome = {'context': {...}, 'report': AnalysisReport|None,
               'failure': {'type', 'message'}|None}
    context 键：doping/N/bond_dim/observable/analyzer/refbond_or_source/
    key_or_definition（失败条目同样保留所尝试的分析身份，不伪造来源）。

    失败约定：仅捕获 FileNotFoundError / ValueError（数据级失败，如文件
    缺失 / 格式错误 / 空输入）记录进 failure；其余异常（潜在程序缺陷）
    原样传播。前置分析失败时，依赖它的峰值分析记录 DependencyFailed
    （明确状态，不静默跳过）。参考键按目录发现，不假定固定集合。
    """
    loaders = loaders or Loaders()
    outcomes = []

    def _ctx(observable, analyzer, refbond_or_source, key_or_definition):
        return {'doping': doping, 'N': N, 'bond_dim': bond_dim,
                'observable': observable, 'analyzer': analyzer,
                'refbond_or_source': refbond_or_source,
                'key_or_definition': key_or_definition}

    def _fail(context, type_, message):
        outcomes.append({'context': context, 'report': None,
                         'failure': {'type': type_, 'message': message}})

    def _try(context, fn, *args, **kwargs):
        try:
            report = fn(*args, **kwargs)
        except (FileNotFoundError, ValueError) as e:
            outcomes.append({'context': context, 'report': None,
                             'failure': {'type': type(e).__name__,
                                         'message': str(e)}})
        else:
            outcomes.append({'context': context, 'report': report,
                             'failure': None})

    # ---- 超导关联衰减（参考键目录发现，逐键 A/B/C） ----
    super_dir = os.path.join(root, t_j, doping,
                             f'{bond_dim}_measure_super_corr')
    try:
        dir_names = sorted(os.listdir(super_dir))
    except FileNotFoundError as e:
        _fail(_ctx(OBSERVABLE_SUPERCONDUCTING_CORRELATION, ANALYZER_SC_DECAY,
                   '', ''),
              type(e).__name__, f'super_corr 目录缺失或不可读: {e}')
    else:
        for name in dir_names:
            m = _SUPERCORR_DIR_RE.match(name)
            if not m:
                continue
            refbond = (int(m.group(1)), int(m.group(2)))
            refbond_tag = f'{refbond[0]}_{refbond[1]}'
            if refbonds is not None and refbond_tag not in refbonds:
                continue
            ref_dir = os.path.join(super_dir, name)
            for key in SC_KEYS:
                ctx = _ctx(OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                           ANALYZER_SC_DECAY, refbond_tag, key)

                def _sc_analysis(_dir=ref_dir, _rb=refbond, _key=key):
                    recs = loaders.super_corr(_dir, _rb, _key, N, a)
                    return run_analysis(
                        OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                        config=config, record_list=recs)

                _try(ctx, _sc_analysis)

    # ---- 自旋结构因子：sz_sz（源点 0，旧版 sz.json） ----
    phys_dir = os.path.join(root, t_j, doping,
                            f'{bond_dim}_measure_phys_qty')
    sz_path = os.path.join(phys_dir, 'measure_49', 'sz.json')
    ctx_sz = _ctx(OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_SPIN_SF,
                  '0', 'sz_sz')
    _try(ctx_sz, run_analysis, OBSERVABLE_SPIN_STRUCTURE_FACTOR,
         operation='estimate',
         estimator_fn=SpinStructureFactorEstimator.estimate_sz_sz_from_file,
         estimator_arg=sz_path, q_list=q_list, N=N, a=a)
    if outcomes[-1]['report'] is not None:
        ctx_sz_peak = _ctx(OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_PEAK,
                           '0', 'sz_sz')
        _try(ctx_sz_peak, run_analysis, OBSERVABLE_SPIN_STRUCTURE_FACTOR,
             operation='peak_analysis',
             sf_result=outcomes[-1]['report'].payload)
    else:
        _fail(_ctx(OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_PEAK, '0', 'sz_sz'),
              'DependencyFailed', '前置 sz_sz 估计失败，跳过峰值分析')

    # ---- 自旋结构因子：spin_dot（源点 49/50 三分量合成） ----
    ctx_dot = _ctx(OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_SPIN_SF,
                   '49_50', 'spin_dot')

    def _spin_dot_analysis():
        recs = loaders.spin_dot(phys_dir, N, a)
        return run_analysis(
            OBSERVABLE_SPIN_STRUCTURE_FACTOR, operation='estimate',
            estimator_fn=SpinStructureFactorEstimator.estimate_spin_dot_from_records,
            estimator_arg=recs, q_list=q_list, N=N, a=a)

    _try(ctx_dot, _spin_dot_analysis)
    if outcomes[-1]['report'] is not None:
        ctx_dot_peak = _ctx(OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_PEAK,
                            '49_50', 'spin_dot')
        _try(ctx_dot_peak, run_analysis, OBSERVABLE_SPIN_STRUCTURE_FACTOR,
             operation='peak_analysis',
             sf_result=outcomes[-1]['report'].payload)
    else:
        _fail(_ctx(OBSERVABLE_SPIN_STRUCTURE_FACTOR, ANALYZER_PEAK,
                   '49_50', 'spin_dot'),
              'DependencyFailed', '前置 spin_dot 估计失败，跳过峰值分析')

    # ---- 电荷结构因子（nf + nf49nf + nf50nf 连通形式） ----
    ctx_chg = _ctx(OBSERVABLE_CHARGE_STRUCTURE_FACTOR, ANALYZER_CHARGE_SF,
                   '', 'charge_density_connected')

    def _charge_analysis():
        return run_analysis(
            OBSERVABLE_CHARGE_STRUCTURE_FACTOR, operation='estimate',
            nf_path=os.path.join(phys_dir, 'measure_49', 'nf.json'),
            corr_paths=[os.path.join(phys_dir, 'measure_49', 'nf49nf.json'),
                        os.path.join(phys_dir, 'measure_50', 'nf50nf.json')],
            q_list=q_list, N=N, a=a)

    _try(ctx_chg, _charge_analysis)
    return outcomes


# ==========================================================================
# 汇总：行构建（P1 契约字段 + 报告既有数值质量信息）
# ==========================================================================
def _empty_row(entry_index, context):
    """固定列序的空行模板：不适用字段留空，缺失不编造默认值。"""
    return {
        'entry_index': entry_index,
        'doping': context['doping'],
        'N': context['N'],
        'bond_dim': context['bond_dim'],
        'observable': context['observable'],
        'analyzer': context['analyzer'],
        'refbond_or_source': context['refbond_or_source'],
        'source_identity_kind': '',
        'key_or_definition': context['key_or_definition'],
        'status': '', 'classification': '', 'evidence': '',
        'data_restricted': '',
        'fit_model': '', 'fit_r_min': '', 'fit_r_max': '',
        'envelope_mode': '', 'verdict': '', 'n_points': '', 'n_distances': '',
        'params_json': '', 'fit_quality_json': '', 'param_errors_json': '',
        'n_peaks': '', 'peak_estimate_entry_index': '',
        'measured_pair_count': '', 'source_coverage': '',
        'imaginary_residual': '', 'q_grid_json': '',
        'diagonal_source_sites_json': '',
        'warnings_json': '',
        'failed_type': '', 'failed_message': '',
        'comparability_bucket': '', 'comparability_reason': '',
    }


def _json_dump(value):
    """dict/list → JSON 字符串（非有限浮点经 _convert 转 null）。"""
    return json.dumps(_convert(value), ensure_ascii=False, sort_keys=True)


def _restricted_label(value):
    """data_restricted 三值语义 → 文本标签（None=未知，不默认 False）。"""
    if value is True:
        return 'restricted'
    if value is False:
        return 'unrestricted'
    return 'unknown'


def build_rows(outcomes, phase_report) -> list:
    """outcomes → 行列表（与 phase_report.entries 同序对齐）。

    从 AnalysisReport 提取：entry 四态字段 + metadata 的拟合窗口 /
    verdict / 点数 / 数值质量信息 + payload 的拟合模型名 / q_grid /
    diagonal_source_sites。电荷行的源点身份以 payload 实际
    diagonal_source_sites 为准（不同源点集合不默认可比）。
    """
    entries = phase_report.entries
    if len(entries) != len(outcomes):
        raise RuntimeError(
            f'entries 与 outcomes 长度不一致: {len(entries)} vs {len(outcomes)}')
    rows = []
    for i, (outcome, entry) in enumerate(zip(outcomes, entries)):
        row = _empty_row(i, outcome['context'])
        row['status'] = entry['status']
        # 缺失（None）统一留空，与行模板语义一致（不编造默认值）
        row['classification'] = entry['classification'] or ''
        row['evidence'] = entry['evidence'] or ''
        row['data_restricted'] = _restricted_label(entry['data_restricted'])
        row['warnings_json'] = _json_dump(list(entry['warnings']))
        # P3 修复：身份来源标记（由分析器与实际数据结构共同决定）。
        # 电荷行仅在 payload 实测存在 diagonal_source_sites 时于下方
        # 标记 payload_field；失败/字段缺失保持 ''（无法确定）。
        if outcome['context']['analyzer'] == ANALYZER_SC_DECAY:
            row['source_identity_kind'] = KIND_REFBOND_DIR
        elif outcome['context']['analyzer'] in (ANALYZER_SPIN_SF,
                                                ANALYZER_PEAK):
            row['source_identity_kind'] = KIND_SCRIPT_CONVENTION
        report = outcome['report']
        if report is None:
            row['failed_type'] = outcome['failure']['type']
            row['failed_message'] = outcome['failure']['message']
            rows.append(row)
            continue
        md = report.metadata or {}
        if report.analyzer == ANALYZER_SC_DECAY:
            for src, dst in (('r_min', 'fit_r_min'), ('r_max', 'fit_r_max'),
                             ('verdict', 'verdict'), ('n_points', 'n_points'),
                             ('n_distances', 'n_distances')):
                if md.get(src) is not None:
                    row[dst] = md[src]
            if md.get('fit_quality'):
                row['fit_quality_json'] = _json_dump(md['fit_quality'])
            if md.get('param_errors'):
                row['param_errors_json'] = _json_dump(md['param_errors'])
            if report.parameters:
                row['params_json'] = _json_dump(report.parameters)
            row['fit_model'] = getattr(report.payload, 'recommended_model', '')
            row['envelope_mode'] = getattr(
                report.payload, 'envelope_mode_used', '')
        elif report.analyzer in _SF_ESTIMATOR_ANALYZERS:
            for src, dst in (('measured_pair_count', 'measured_pair_count'),
                             ('source_coverage', 'source_coverage'),
                             ('imaginary_residual', 'imaginary_residual')):
                if md.get(src) is not None:
                    row[dst] = md[src]
            payload = report.payload if isinstance(report.payload, dict) else {}
            if payload.get('q_grid') is not None:
                row['q_grid_json'] = _json_dump(payload['q_grid'])
            if payload.get('diagonal_source_sites') is not None:
                row['diagonal_source_sites_json'] = _json_dump(
                    payload['diagonal_source_sites'])
                # 电荷源点身份以实际数据为准（机械字段，不假定 49/50）
                row['refbond_or_source'] = '_'.join(
                    str(int(v)) for v in sorted(payload['diagonal_source_sites']))
                # P3 修复：身份来源为 payload 实测字段（对角补全源点，
                # 不等同于完整测量源点集合）
                row['source_identity_kind'] = KIND_PAYLOAD_FIELD
        elif report.analyzer == ANALYZER_PEAK:
            if report.parameters.get('n_peaks') is not None:
                row['n_peaks'] = report.parameters['n_peaks']
            # P5 + 关联键修复：峰值行 q_grid 摘要按其真正对应的 SF
            # estimate outcome 的 payload['q'] 逐行计算（峰值 payload
            # 无网格字段）。对应关系要求唯一匹配（全部条件：位置在前 +
            # analyzer ∈ 估计器 + doping/N/bond_dim/observable/
            # refbond_or_source/key_or_definition 全等）；零个或多个
            # 候选 → 保守留空（unknown），不用全局 q_list 或其他候选
            # 结果补齐
            ctx = outcome['context']
            candidates = [
                k for k in range(i)
                if (outcomes[k]['report'] is not None
                    and outcomes[k]['report'].analyzer
                    in _SF_ESTIMATOR_ANALYZERS
                    and outcomes[k]['context']['doping'] == ctx['doping']
                    and outcomes[k]['context']['N'] == ctx['N']
                    and outcomes[k]['context']['bond_dim'] == ctx['bond_dim']
                    and outcomes[k]['context']['observable']
                    == ctx['observable']
                    and outcomes[k]['context']['refbond_or_source']
                    == ctx['refbond_or_source']
                    and outcomes[k]['context']['key_or_definition']
                    == ctx['key_or_definition'])]
            if len(candidates) == 1:
                est = outcomes[candidates[0]]
                row['peak_estimate_entry_index'] = candidates[0]
                prev_payload = (est['report'].payload
                                if isinstance(est['report'].payload, dict)
                                else None)
                prev_q = prev_payload.get('q') if prev_payload else None
                if prev_q is not None:
                    arr = np.asarray(prev_q, dtype=float)
                    if arr.ndim == 2 and arr.shape[0] > 0:
                        qg = _q_grid_stats(arr)
                        if qg['n_qx'] > 0:
                            row['q_grid_json'] = _json_dump(qg)
        rows.append(row)
    return rows


# ==========================================================================
# 可比性分组（机械条件检查；P2 审计结果只读引用）
# ==========================================================================
def _identity(row):
    return (row['observable'], row['analyzer'], row['N'], row['bond_dim'],
            row['refbond_or_source'], row['key_or_definition'])


def _coarse(row):
    return (row['observable'], row['analyzer'], row['N'], row['bond_dim'])


def _obs_analyzer(row):
    return (row['observable'], row['analyzer'])


def _is_source_identity(row):
    """身份列是否代表源点集合：按行内 source_identity_kind 标记判定
    （payload_field / script_convention = 源点身份），不按分析器名称推测。"""
    return row.get('source_identity_kind', '') in _SOURCE_KINDS


def assign_comparability(rows, phase_report):
    """逐行赋可比性档位并生成分组摘要（就地修改 rows 并返回）。

    规则（机械条件检查，无物理判据）：
    - failed 行 → 'not_direct'（无数值参与并列比较）
    - 同身份组（observable/analyzer/N/bond_dim/参考键或源点/键方向或定义
      全等）内存在成功条目且附加检查全部通过 → 'direct'：
        SF 估计行：P2 q_grid 逐对状态 consistent（仅摘要级，完整坐标
        未验证）；电荷行另需 P2 source_set consistent
        （diagonal_source_sites 集合相等，不等同于完整测量源点集合）
        峰值行：行内 q_grid 摘要一致（build_rows 按唯一匹配的前置
        estimate 的 payload['q'] 逐行计算，缺失 → unknown），且对应
        estimate 的 P2 逐对审计为 consistent（unknown/inconsistent/无对
        → 不视为一致，reason 注明限制来自前置 estimate 的 P2 审计）
        衰减行：无附加检查（身份全等即条件一致；不表示 DMRG 收敛精度
        相同）
    - 同身份组内成功条目存在但附加检查不通过 → 'not_direct'
    - 同身份组内全部失败 → 'not_direct'（无数值可并列比较）
    - 同 (observable/analyzer/N/bond_dim) 但身份不同 → 先过硬性限制，
      再考虑 'cautious'（已知差异）
    - 同 (observable/analyzer) 但 N 或 bond_dim 不同 → 'not_direct'
      （关键条件不一致）
    - 无同 (observable/analyzer) 条目 → 'sole'（无并列比较对象）

    优先级语义（U1 锁定）：
    - 硬性限制优先：failed、N/bond_dim 不一致、源点身份不同或未知
      （source_identity_kind 为 payload_field/script_convention 且
      身份值不同）、P2 逐对状态 inconsistent/unknown、峰值输入 q 坐标
      摘要不一致，一律 not_direct，不被 cautious 等宽松分支覆盖
    - cautious 仅适用于满足上述前置条件的已知差异（SC 参考键 / 键方向
      或定义不同）
    - 同身份组内全部失败 → not_direct，不降级到 cautious
    - sole 仅表示无并列比较对象，不是比较结论
    """
    consistency = phase_report.consistency or {}
    qpair_map = {}
    for p in consistency.get('q_grid', {}).get('pairs', []):
        if len(p.get('entries', [])) == 2:
            qpair_map[(p['entries'][0], p['entries'][1])] = p['status']
    srcpair_map = {}
    for p in consistency.get('source_set', {}).get('pairs', []):
        if len(p.get('entries', [])) == 2:
            srcpair_map[(p['entries'][0], p['entries'][1])] = p['status']

    def _pair_status(pair_map, i, j):
        return pair_map.get((i, j), pair_map.get((j, i)))

    for i, row in enumerate(rows):
        if row['status'] == STATUS_FAILED:
            row['comparability_bucket'] = BUCKET_NOT_DIRECT
            row['comparability_reason'] = (
                f"分析失败（{row['failed_type']}: "
                f"{row['failed_message'][:80]}），无数值参与并列比较")
            continue
        same_id = [j for j, r in enumerate(rows)
                   if j != i and _identity(r) == _identity(row)]
        same_id_ok = [j for j in same_id if rows[j]['status'] != STATUS_FAILED]
        n_failed_peer = len(same_id) - len(same_id_ok)
        if same_id_ok:
            checks = []   # (kind, message)，kind ∈ {'direct', 'fail'}
            if row['analyzer'] in _SF_ESTIMATOR_ANALYZERS:
                for j in same_id_ok:
                    qs = _pair_status(qpair_map, i, j)
                    if qs == 'consistent':
                        checks.append(('direct', 'q_grid 摘要一致'
                                               '（完整坐标未验证）'))
                    elif qs == 'inconsistent':
                        checks.append(('fail', 'q_grid 摘要不一致'))
                    else:
                        checks.append(('fail', 'q_grid 摘要未知或未审计'))
                if row['analyzer'] == ANALYZER_CHARGE_SF:
                    for j in same_id_ok:
                        ss = _pair_status(srcpair_map, i, j)
                        if ss == 'consistent':
                            checks.append(('direct', 'diagonal_source_sites '
                                                     '集合相等（不等同于完整测量'
                                                     '源点集合）'))
                        elif ss == 'inconsistent':
                            checks.append(('fail', 'diagonal_source_sites '
                                                   '集合不等'))
                        else:
                            checks.append(('fail', 'diagonal_source_sites '
                                                   '未知或未审计'))
            elif row['analyzer'] == ANALYZER_PEAK:
                # P5 修复：与每个同身份 peer 比较行内 q_grid 摘要
                # （build_rows 按唯一匹配的前置 estimate 的 payload['q']
                # 逐行计算；缺失 → unknown，不用全局 q_list 补齐）
                # P2 传播修复（方案 b）：另查各自对应 estimate 的 P2
                # 逐对审计——unknown/inconsistent/无对 → 不视为
                # consistent，Peak 不得判 direct（reason 注明限制来自
                # 前置 estimate 的 P2 审计）；不复制 estimate 的
                # classification/status 等物理判断
                est_i = row['peak_estimate_entry_index']
                for j in same_id_ok:
                    gi, gj = row['q_grid_json'], rows[j]['q_grid_json']
                    if not gi or not gj:
                        checks.append(('fail', '峰值分析输入 q 坐标摘要缺失'
                                               '（unknown）'))
                    elif gi == gj:
                        checks.append(('direct', 'q_grid 摘要一致（该峰值'
                                                 '分析对应 SF 结果的 q 坐标'
                                                 '摘要；完整坐标未验证）'))
                    else:
                        checks.append(('fail', '峰值分析输入 q 坐标摘要'
                                               '不一致'))
                    est_j = rows[j]['peak_estimate_entry_index']
                    # 注意：索引 0 是合法值，必须与空字符串显式比较
                    # （不能用 not est_i 判断，0 会被误判为缺失）
                    if est_i == '' or est_j == '':
                        checks.append(('fail', '对应 estimate 无法唯一匹配'
                                               '（unknown），P2 审计不可用'))
                        continue
                    ei, ej = int(est_i), int(est_j)
                    qs = _pair_status(qpair_map, ei, ej)
                    if qs == 'consistent':
                        pass   # 对应 estimate 的 P2 审计一致
                    elif qs in ('inconsistent', 'unknown'):
                        checks.append(('fail', f'对应 estimate 的 P2 q_grid '
                                               f'审计不一致或未知（{qs}）'))
                    else:
                        checks.append(('fail', '对应 estimate 无 P2 逐对'
                                               '审计结果，不能视为一致'))
                    if rows[ei]['analyzer'] == ANALYZER_CHARGE_SF:
                        ss = _pair_status(srcpair_map, ei, ej)
                        if ss == 'consistent':
                            pass
                        elif ss in ('inconsistent', 'unknown'):
                            checks.append(('fail', f'对应 estimate 的 P2 '
                                                   f'diagonal_source_sites '
                                                   f'审计不一致或未知'
                                                   f'（{ss}）'))
                        else:
                            checks.append(('fail', '对应 estimate 无 P2 源点'
                                                   '审计结果，不能视为'
                                                   '一致'))
            # 衰减行：无附加检查（身份全等即条件一致）
            direct_msgs = [m for k, m in checks if k == 'direct']
            fail_msgs = [m for k, m in checks if k == 'fail']
            if not fail_msgs:
                base = ('条件一致（N/bond dimension/参考键或源点/键方向或'
                        '定义相同；不表示 DMRG 收敛精度相同），可直接并列'
                        '比较')
                # P4 修复：去重（同身份多 peer 会产生重复消息）；
                # sorted(set(...)) 保证顺序确定性
                extra = ('；' + '；'.join(sorted(set(direct_msgs)))
                         if direct_msgs else '')
                if n_failed_peer:
                    extra += f'；同组另有 {n_failed_peer} 条分析失败（无数值）'
                # U2 补充：脚本约定来源的源点身份在 direct 原因中注明
                # （未经报告字段验证，不代表已验证 source set）
                if row['source_identity_kind'] == KIND_SCRIPT_CONVENTION:
                    extra += '；源点身份来自脚本约定，未经报告字段验证'
                row['comparability_bucket'] = BUCKET_DIRECT
                row['comparability_reason'] = base + extra
            else:
                row['comparability_bucket'] = BUCKET_NOT_DIRECT
                row['comparability_reason'] = (
                    '关键条件不一致或未知：' + '；'.join(fail_msgs))
            continue
        if same_id:
            row['comparability_bucket'] = BUCKET_NOT_DIRECT
            row['comparability_reason'] = (
                '同条件组内无成功条目可并列比较（其余均分析失败）')
            continue
        same_coarse_diff = [j for j, r in enumerate(rows)
                            if j != i and _coarse(r) == _coarse(row)
                            and _identity(r) != _identity(row)]
        if same_coarse_diff:
            peers = [(j, rows[j]) for j in same_coarse_diff]
            hard_fails = []
            # P1 修复（硬性限制）：任一行身份列代表源点身份
            # （source_identity_kind 为 payload_field / script_convention）
            # 且身份值不同 → 源点差异，不得默认可比（不落 cautious）
            if (_is_source_identity(row)
                    or any(_is_source_identity(r) for _, r in peers)):
                if any(r['refbond_or_source'] != row['refbond_or_source']
                       for _, r in peers):
                    hard_fails.append('源点身份不同或未知，不得默认可比')
            # P2 修复（硬性限制）：SF 估计行同分析器 peer 的 P2 逐对
            # 状态为 inconsistent/unknown 时，不被 cautious 覆盖
            # （P2 只对同 analyzer 逐对；跨分析器无对 → 不检查）
            if row['analyzer'] in _SF_ESTIMATOR_ANALYZERS:
                for j, r in peers:
                    if r['analyzer'] != row['analyzer']:
                        continue
                    qs = _pair_status(qpair_map, i, j)
                    if qs in ('inconsistent', 'unknown'):
                        hard_fails.append(
                            f'P2 q_grid 摘要不一致或未知（{qs}）')
                    if row['analyzer'] == ANALYZER_CHARGE_SF:
                        ss = _pair_status(srcpair_map, i, j)
                        if ss in ('inconsistent', 'unknown'):
                            hard_fails.append(
                                f'P2 diagonal_source_sites 不一致或未知'
                                f'（{ss}）')
            if hard_fails:
                row['comparability_bucket'] = BUCKET_NOT_DIRECT
                row['comparability_reason'] = (
                    '关键条件不一致或未知：'
                    + '；'.join(sorted(set(hard_fails))))
                continue
            # 已知差异（仅剩 SC 参考键 / 键方向或定义；源点差异已被
            # 上方硬性限制拦截）→ cautious
            diffs = set()
            for _, r in peers:
                if r['refbond_or_source'] != row['refbond_or_source']:
                    diffs.add('参考键')
                if r['key_or_definition'] != row['key_or_definition']:
                    diffs.add('键方向或定义')
            row['comparability_bucket'] = BUCKET_CAUTIOUS
            row['comparability_reason'] = (
                '存在已知差异（' + '/'.join(sorted(diffs))
                + '不同），只能谨慎比较')
            continue
        same_obs_ana = [j for j, r in enumerate(rows)
                        if j != i and _obs_analyzer(r) == _obs_analyzer(row)]
        if same_obs_ana:
            row['comparability_bucket'] = BUCKET_NOT_DIRECT
            row['comparability_reason'] = (
                '关键条件不一致（N 或 bond dimension 不同），不可直接比较')
            continue
        row['comparability_bucket'] = BUCKET_SOLE
        row['comparability_reason'] = '无同类条目（无并列比较对象）'

    # ---- 分组摘要（按身份分组；组档位 = 成功成员档位的共同值） ----
    groups = []
    group_index = {}
    for i, row in enumerate(rows):
        idt = _identity(row)
        if idt not in group_index:
            group_index[idt] = len(groups)
            groups.append({
                'observable': row['observable'], 'analyzer': row['analyzer'],
                'N': row['N'], 'bond_dim': row['bond_dim'],
                'refbond_or_source': row['refbond_or_source'],
                'key_or_definition': row['key_or_definition'],
                'member_indices': [], 'n_ok': 0, 'n_failed': 0,
                'bucket': '', 'reason': ''})
        g = groups[group_index[idt]]
        g['member_indices'].append(i)
        if row['status'] == STATUS_FAILED:
            g['n_failed'] += 1
        else:
            g['n_ok'] += 1
    for g in groups:
        ok_rows = [rows[i] for i in g['member_indices']
                   if rows[i]['status'] != STATUS_FAILED]
        if not ok_rows:
            g['bucket'] = BUCKET_NOT_DIRECT
            g['reason'] = '组内全部失败，无数值可并列比较'
        else:
            buckets = {r['comparability_bucket'] for r in ok_rows}
            g['bucket'] = buckets.pop() if len(buckets) == 1 else 'mixed'
            g['reason'] = '；'.join(sorted(
                {r['comparability_reason'] for r in ok_rows}))
    return rows, groups


# ==========================================================================
# 汇总主流程（P1 collect → P2 audit → P3 sufficiency(rules=None)）
# ==========================================================================
def summarize(outcomes, q_list=None, config=None) -> dict:
    """outcomes → 完整汇总 dict：rows / comparability_groups /
    phase_report / config。

    复用 PhaseDiagnosis P1/P2/P3；evidence_sufficiency_check 以
    rules=None 调用（如实记录未提供经确认的充分性标准，overall 恒为
    'unknown'，不编造置信度）。

    q_list 参数仅保留调用方兼容：P5 修复后可比性比较不再使用全局
    q_list（峰值行 q_grid 由 build_rows 按前置 outcome 的
    payload['q'] 逐行计算）。
    """
    labels = [
        f"{o['context']['doping']}|{o['context']['observable']}|"
        f"{o['context']['analyzer']}|{o['context']['refbond_or_source']}|"
        f"{o['context']['key_or_definition']}" for o in outcomes]
    phase_report = collect_evidence([o['report'] for o in outcomes], labels)
    audit_comparability(phase_report)
    evidence_sufficiency_check(phase_report, rules=None)

    rows = build_rows(outcomes, phase_report)
    rows, groups = assign_comparability(rows, phase_report)

    cfg = dict(config or {})
    if 'analysis_version' not in cfg:
        for o in outcomes:
            rep = o['report']
            if rep is not None and rep.metadata.get('analysis_version'):
                cfg['analysis_version'] = rep.metadata['analysis_version']
                break
    return {'rows': rows, 'comparability_groups': groups,
            'phase_report': phase_report, 'config': cfg}


# ==========================================================================
# 输出（CSV + JSON；整体覆盖，绝不追加）
# ==========================================================================
def _csv_cell(value):
    """CSV 单元格：None/非有限浮点 → 空字符串；其余 str()。"""
    if value is None:
        return ''
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


def write_outputs(summary, outdir) -> dict:
    """写 CSV + JSON（utf-8-sig CSV 便于 Excel 打开；JSON 严格
    allow_nan=False）。输出整体覆盖重写，重复执行不追加旧记录。"""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'cross_doping_summary.csv')
    json_path = os.path.join(outdir, 'cross_doping_summary.json')
    rows = summary['rows']

    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k, ''))
                             for k in CSV_COLUMNS})

    doc = {
        'generated_by': 'Cross_Doping_Summary (PhaseDiagnosis-P4.3)',
        'config': _convert(summary['config']),
        'rows': [{k: _convert(r.get(k, '')) for k in CSV_COLUMNS}
                 for r in rows],
        'comparability_groups': _convert(summary['comparability_groups']),
        'phase_report': summary['phase_report'].to_dict(),
        'semantics': {
            'missing_fields': '空字符串 / JSON null 表示该字段对该分析'
                              '不适用或缺失，未填入虚构默认值',
            'data_restricted': 'restricted/unrestricted/unknown'
                               '（None=未知，不默认 unrestricted）',
            'comparability_buckets': {
                BUCKET_DIRECT: '条件一致，可直接并列比较（仅分析条件；'
                               '不表示 DMRG 收敛精度相同）',
                BUCKET_CAUTIOUS: '存在已知差异（SC 参考键或键方向/定义'
                                 '不同），只能谨慎比较',
                BUCKET_NOT_DIRECT: '关键条件不一致或未知（N/bond dimension'
                                   '不同、源点身份不同或未知、P2 q_grid 或'
                                   'source_set 状态不一致或未知、峰值输入 q'
                                   '坐标摘要不一致、对应 estimate 的 P2 审计'
                                   '不一致或未知，或分析失败无数值），'
                                   '不可直接比较',
                BUCKET_SOLE: '无同类条目（无并列比较对象），不是比较结论',
            },
            'comparability_priority': '硬性限制（failed / N / bond_dim / '
                                      '源点身份 / P2 逐对状态 / 峰值输入 q '
                                      '摘要）优先于 cautious；同身份组全部'
                                      '失败 → not_direct（不降级）',
            'source_identity_kind': {
                KIND_REFBOND_DIR: 'SC：参考键身份来自参考键目录名',
                KIND_PAYLOAD_FIELD: '电荷：来自 payload 的 diagonal_source_'
                                    'sites（对角补全源点，不等同于完整测量'
                                    '源点集合）',
                KIND_SCRIPT_CONVENTION: '自旋：脚本输入约定，未经分析报告'
                                        '字段验证，不代表已验证 source set；'
                                        '不同源点身份不判 direct',
                '': '不适用或无法确定（如失败行、字段缺失）',
            },
            'peak_estimate_entry_index': '峰值行唯一匹配的前置 estimate 的'
                                         'entry 索引（空=零个或多个候选，'
                                         '无法唯一匹配，q 摘要保守留空）',
            'q_grid_comparison_level': 'summary（仅 n_qx/n_qy/端点摘要；'
                                       '完整坐标未验证）',
            'no_physical_conclusions': '本汇总不判定物理趋势/单调性/物理相态'
                                       '/趋势显著性；数值随 doping 的变化仅'
                                       '作观测呈现；sufficiency overall='
                                       'unknown 表示未提供经确认的充分性标准',
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
    return {'csv': csv_path, 'json': json_path}


# ==========================================================================
# 控制台摘要（仅事实呈现，无趋势/相态结论）
# ==========================================================================
def _print_console(summary, paths, config):
    rows = summary['rows']
    by_doping = {}
    for r in rows:
        d = by_doping.setdefault(
            r['doping'], {'sc': [0, 0], 'spin_sf': [0, 0],
                          'peak': [0, 0], 'charge': [0, 0]})
        if r['observable'] == OBSERVABLE_SUPERCONDUCTING_CORRELATION:
            key = 'sc'
        elif r['analyzer'] == ANALYZER_PEAK:
            key = 'peak'
        elif r['observable'] == OBSERVABLE_SPIN_STRUCTURE_FACTOR:
            key = 'spin_sf'
        else:
            key = 'charge'
        d[key][0 if r['status'] != STATUS_FAILED else 1] += 1

    print(f"\n===== 跨 doping 汇总（N={config.get('N')}, "
          f"bond_dim={config.get('bond_dim')}）=====")
    print(f"{'doping':10} {'SC ok/fail':>10} {'spin_sf ok/fail':>15} "
          f"{'peak ok/fail':>13} {'charge ok/fail':>14}")
    for doping in sorted(by_doping):
        d = by_doping[doping]
        print(f"{doping:10} {d['sc'][0]:>3}/{d['sc'][1]:<5} "
              f"{d['spin_sf'][0]:>5}/{d['spin_sf'][1]:<8} "
              f"{d['peak'][0]:>5}/{d['peak'][1]:<6} "
              f"{d['charge'][0]:>6}/{d['charge'][1]:<6}")

    print("\n===== 可比性分组（direct/cautious/not_direct/sole）=====")
    for g in summary['comparability_groups']:
        idt = (f"{g['observable']}|{g['analyzer']}|"
               f"ref={g['refbond_or_source'] or '-'}|"
               f"{g['key_or_definition'] or '-'}")
        print(f"  [{g['bucket']:9}] {idt}  "
              f"ok={g['n_ok']}/fail={g['n_failed']}  {g['reason'][:100]}")
    print(f"\n输出: {paths['csv']}\n      {paths['json']}")
    print("说明: direct 仅表示分析条件一致（q_grid 仅摘要级、完整坐标未"
          "验证），不表示物理结论可比；本汇总不判定任何物理趋势或相态。")


# ==========================================================================
# CLI 入口
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='跨 doping 受控数值汇总（PhaseDiagnosis-P4.3）：'
                    '逐 doping 运行既有统一入口并输出 CSV/JSON 汇总，'
                    '不判定物理趋势或相态')
    parser.add_argument('--root', default=os.path.join(SCRIPT_DIR, 'Data_N200'),
                        help='数据根目录（默认 <脚本目录>/Data_N200）')
    parser.add_argument('--t-j', dest='t_j', default=DEFAULT_T_J,
                        help='t/J 目录名（默认 t_J_3）')
    parser.add_argument('--dopings', default=','.join(DEFAULT_DOPINGS),
                        help='逗号分隔的 doping 列表（默认 7 个共同 doping）')
    parser.add_argument('--bond-dim', dest='bond_dim', default=DEFAULT_BOND_DIM,
                        help='bond dimension（默认 1500）')
    parser.add_argument('--N', type=int, default=DEFAULT_N,
                        help='系统格点数（默认 200；N=400 数据请单独运行，'
                             '不与 N=200 混组）')
    parser.add_argument('--n-qx', type=int, default=21,
                        help='动量网格 qx 点数（默认 21）')
    parser.add_argument('--n-qy', type=int, default=4,
                        help='动量网格 qy 点数（默认 4）')
    parser.add_argument('--a', type=float, default=1.0, help='晶格常数')
    parser.add_argument('--refbonds', default=None,
                        help='逗号分隔的参考键过滤（如 49_50,50_51）；'
                             '默认全部发现的参考键')
    parser.add_argument('--outdir',
                        default=os.path.join(SCRIPT_DIR,
                                             'cross_doping_summary_output'),
                        help='输出目录（默认 <脚本目录>/'
                             'cross_doping_summary_output）')
    args = parser.parse_args(argv)

    dopings = [d.strip() for d in args.dopings.split(',') if d.strip()]
    refbonds = None
    if args.refbonds:
        refbonds = {rb.strip() for rb in args.refbonds.split(',')
                    if rb.strip()}
    q_list, q_meta = xc_momentum_grid(args.N, a=args.a,
                                      n_qx=args.n_qx, n_qy=args.n_qy)

    outcomes = []
    for doping in dopings:
        outcomes.extend(analyze_doping(
            doping, root=args.root, t_j=args.t_j, bond_dim=args.bond_dim,
            N=args.N, a=args.a, q_list=q_list, refbonds=refbonds))

    config = {
        'root': args.root, 't_j': args.t_j, 'dopings': dopings,
        'bond_dim': args.bond_dim, 'N': args.N, 'a': args.a,
        'n_qx': args.n_qx, 'n_qy': args.n_qy, 'q_meta': q_meta,
        'refbonds_filter': sorted(refbonds) if refbonds else None,
        'analyzer_config': 'default（AnalyzerConfig 默认阈值快照见下）',
        'analyzer_config_snapshot': vars(AnalyzerConfig()),
    }
    summary = summarize(outcomes, q_list, config=config)
    paths = write_outputs(summary, args.outdir)
    _print_console(summary, paths, config)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
