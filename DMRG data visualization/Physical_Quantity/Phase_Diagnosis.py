# -*- coding: utf-8 -*-
"""
多观测量证据汇总契约（PhaseDiagnosis-P1：纯汇总，无物理判据）

将多份 AnalysisReport 收集为统一、可追溯的 PhaseDiagnosisReport。
第一版只做：
- 结构化摘要（各报告 observable/analyzer/status/classification/evidence/
  data_restricted/warnings 原样保留）
- 调用方标签一一对应（错配按明确契约报错，不静默）
- status 计数汇总
- 规则执行记录（rule_log，支持后续审计）
- 严格 JSON 序列化（禁止 NaN/Infinity 字面量）

语义边界：
- evidence_collected 仅表示"汇总完成"，不表示物理证据充分
- INCONCLUSIVE 状态不被转换为成功或失败；failed 报告保留在汇总中
- classification / status / data_restricted 独立记录，不隐式归一化
- 报告引用（report_refs）仅内存保留可追溯；to_dict/to_json 默认不含
  （需要完整报告请对各 AnalysisReport 单独调用 to_json）
- 不引入物理相态判断、加权评分、概率或置信度；不自动记录文件路径
- 不修改 Analysis_Contract / Analysis_Integration / Analysis_Entry
"""

import json
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .Analysis_Contract import (
    AnalysisReport, STATUS_FAILED, _convert, ANALYZER_ENERGY,
    ANALYZER_SC_PAIRING_SYMMETRY, ANALYZER_DECAY_SUPERCONDUCTING,
    ANALYZER_DENSITY,
    ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT,
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR, OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR, OBSERVABLE_SC_PAIRING_SYMMETRY,
    OBSERVABLE_LOCAL_DENSITY,
    OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
    OBSERVABLE_ENERGY)

# 结构因子估计报告的精确识别标识（禁止宽泛字符串包含判断）
ANALYZER_SPIN_SF = 'SpinStructureFactorEstimator'
ANALYZER_CHARGE_SF = 'ChargeStructureFactorEstimator'
ANALYZER_PEAK = 'SpinStructureFactorPeakAnalyzer'
ANALYZER_SC_PAIR_SF = 'SCPairStructureFactorEstimator'
_SF_ESTIMATOR_ANALYZERS = (ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF,
                           ANALYZER_SC_PAIR_SF)

# 可比性检查状态（四态，语义见 audit_comparability docstring）
CMP_CONSISTENT = 'consistent'
CMP_INCONSISTENT = 'inconsistent'
CMP_UNKNOWN = 'unknown'
CMP_NOT_APPLICABLE = 'not_applicable'

# 证据充分性状态（保守语义，见 evidence_sufficiency_check docstring）
SUFF_SUFFICIENT = 'sufficient'
SUFF_INSUFFICIENT = 'insufficient'
SUFF_UNKNOWN = 'unknown'

# 汇总结论枚举（语义见模块 docstring）
CONCLUSION_EVIDENCE_COLLECTED = 'evidence_collected'
CONCLUSION_INCONCLUSIVE = 'INCONCLUSIVE'

# entries 的固定键集（新报告类型扩展时需显式更新并补测试）
_ENTRY_KEYS = ('observable', 'analyzer', 'status', 'classification',
               'evidence', 'data_restricted', 'warnings', 'label')


@dataclass
class PhaseDiagnosisReport:
    """多观测量证据汇总结果。"""
    entries: list = field(default_factory=list)          # 结构化摘要列表
    status_summary: dict = field(default_factory=dict)   # {status: 计数}
    conclusion: str = CONCLUSION_INCONCLUSIVE            # 仅两个枚举值
    rule_log: list = field(default_factory=list)         # 规则执行记录（可审计）
    report_refs: list = field(default_factory=list)      # 原始报告引用（内存可追溯）
    labels: list = field(default_factory=list)           # 调用方标签（与 entries 同序）
    consistency: dict = field(default_factory=dict)      # P2 可比性审计结果（默认空）
    sufficiency: dict = field(default_factory=dict)      # P3 证据充分性结果（默认空）
    candidate_verdicts: dict = field(default_factory=dict)  # P1-D.2 候选级九态判定
    verdict_summary: dict = field(default_factory=dict)     # P1-D.2 六组汇总
    rule_trace: list = field(default_factory=list)          # P1-D.2 规则执行轨迹

    def to_dict(self) -> dict:
        """紧凑 dict：entries/status_summary/conclusion/rule_log/labels/
        consistency/sufficiency。report_refs 默认不序列化（引用完整性由
        内存保证；需要完整报告请对各 AnalysisReport 单独调用 to_json）。
        非有限浮点转 null。"""
        return {
            'entries': _convert(self.entries),
            'status_summary': _convert(self.status_summary),
            'conclusion': self.conclusion,
            'rule_log': _convert(self.rule_log),
            'labels': _convert(self.labels),
            'consistency': _convert(self.consistency),
            'sufficiency': _convert(self.sufficiency),
            'candidate_verdicts': _convert(self.candidate_verdicts),
            'verdict_summary': _convert(self.verdict_summary),
            'rule_trace': _convert(self.rule_trace),
        }

    def to_json(self, path: str) -> None:
        """严格 JSON 落盘（allow_nan=False 双保险）。"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False,
                      indent=2, allow_nan=False)


def collect_evidence(reports, labels=None) -> PhaseDiagnosisReport:
    """收集多份 AnalysisReport 的结构化证据摘要。

    参数
    ----
    reports : list[AnalysisReport]，允许 None 元素（视为 failed 条目，
              该条目的 observable/analyzer 记 None，不伪造来源）
    labels  : list[str|None] 或 None——与 reports 一一对应的调用方标签
              （如 doping/bond_dim/measure 组合；None 表示未提供标签）。
              长度不匹配 → ValueError；元素非 str/None → TypeError；
              不做任何隐式对齐或默认命名。

    结论规则（写入 rule_log）：
    - entries 非空且存在至少一个 status != 'failed' 的条目
      → conclusion = 'evidence_collected'（仅表示汇总完成，不代表
        物理证据充分）
    - 否则（空输入或全部 failed）→ conclusion = 'INCONCLUSIVE'
    """
    if reports is None:
        reports = []
    if labels is None:
        labels = [None] * len(reports)
    if len(labels) != len(reports):
        raise ValueError(
            f"labels 与 reports 长度不一致: {len(labels)} vs {len(reports)}")

    entries = []
    report_refs = []
    for rep, label in zip(reports, labels):
        if label is not None and not isinstance(label, str):
            raise TypeError(
                f"label 必须为 str 或 None，得到 {type(label).__name__}")
        if rep is None:
            entries.append({
                'observable': None, 'analyzer': None,
                'status': STATUS_FAILED,
                'classification': None, 'evidence': None,
                'data_restricted': None, 'warnings': [], 'label': label,
            })
            report_refs.append(None)
            continue
        if not isinstance(rep, AnalysisReport):
            raise TypeError(
                f"reports 元素必须为 AnalysisReport 或 None，"
                f"得到 {type(rep).__name__}")
        entries.append({
            'observable': rep.observable,
            'analyzer': rep.analyzer,
            'status': rep.status,
            'classification': rep.classification,
            'evidence': rep.evidence,
            'data_restricted': rep.data_restricted,
            'warnings': list(rep.warnings),
            'label': label,
        })
        report_refs.append(rep)

    status_summary = {}
    for e in entries:
        status_summary[e['status']] = status_summary.get(e['status'], 0) + 1

    conclusion = (CONCLUSION_EVIDENCE_COLLECTED
                  if entries and any(e['status'] != STATUS_FAILED for e in entries)
                  else CONCLUSION_INCONCLUSIVE)

    rule_log = [
        f"collect_evidence: 输入 {len(entries)} 份报告，"
        f"标签一一对应校验通过（labels={len(labels)}）",
        f"collect_evidence: status 汇总 {status_summary}",
        f"collect_evidence: conclusion={conclusion}"
        f"（evidence_collected 仅表示汇总完成，不代表物理证据充分；"
        f"INCONCLUSIVE 状态不被转换为成功或失败）",
    ]
    return PhaseDiagnosisReport(
        entries=entries, status_summary=status_summary,
        conclusion=conclusion, rule_log=rule_log,
        report_refs=report_refs,
        labels=[e['label'] for e in entries])


# ==========================================================================
# P2：可比性审计（source set 与 q_grid 摘要；纯机械检查，无物理判据）
# ==========================================================================
def _is_int_list(value) -> bool:
    """严格整数列表校验（bool 是 int 子类，须用 type 精确判断）。"""
    return (isinstance(value, list)
            and all(type(x) is int for x in value))


def _qgrid_ok(value) -> bool:
    """q_grid 摘要结构完整且非空：6 键全在、n>0、端点非 None。"""
    if not isinstance(value, dict):
        return False
    if set(value.keys()) != {'n_qx', 'n_qy', 'qx_min', 'qx_max',
                             'qy_min', 'qy_max'}:
        return False
    if not (isinstance(value['n_qx'], int) and isinstance(value['n_qy'], int)
            and value['n_qx'] > 0 and value['n_qy'] > 0):
        return False
    for k in ('qx_min', 'qx_max', 'qy_min', 'qy_max'):
        if value[k] is None:
            return False
    return True


def _summarize_groups(pairs, group_key='analyzer'):
    """组级汇总（逐对结果优先保留，组汇总不丢失信息）：
    - 含 inconsistent → inconsistent
    - 否则全 consistent → consistent
    - 否则全 not_applicable → not_applicable
    - 其余（含 unknown 或无对）→ unknown"""
    by_group = {}
    for p in pairs:
        by_group.setdefault(p[group_key], []).append(p['status'])
    out = {}
    for grp, sts in by_group.items():
        if CMP_INCONSISTENT in sts:
            out[grp] = CMP_INCONSISTENT
        elif all(s == CMP_CONSISTENT for s in sts):
            out[grp] = CMP_CONSISTENT
        elif all(s == CMP_NOT_APPLICABLE for s in sts):
            out[grp] = CMP_NOT_APPLICABLE
        else:
            out[grp] = CMP_UNKNOWN
    return out


def audit_comparability(phase_report: PhaseDiagnosisReport) -> PhaseDiagnosisReport:
    """P2 可比性审计：对已汇总报告做 source set 与 q_grid 摘要的逐对检查。

    规则（全部为机械检查，无物理判据）：
    - 报告类型用 analyzer/observable 精确值识别（禁止宽泛字符串包含）；
      Spin SF 估计、Charge SF 估计、峰值分析严格分开
    - source set：仅比较 Charge SF 的 payload['diagonal_source_sites']
      （集合比较，顺序无关）；该字段语义是"对角补全源点"，不等同于
      无条件完整的测量源点集合（reason 中明示）。source_coverage 数量
      相同不能视为集合相同。自旋 SF 无源点身份字段 → unknown（不推测）。
      衰减/能量/峰值 → not_applicable
    - q_grid：仅摘要级比较（n_qx/n_qy/端点），同 analyzer 组内逐对；
      摘要全等 → consistent 且 comparison_level='summary'（完整网格坐标
      未验证）；摘要不同 → inconsistent；缺失/结构不完整/空网格/端点
      None → unknown；空摘要相同不得判 consistent
    - failed / None 报告不参与一致断言，其检查标记 unknown 并保留原 status
    - 单条报告 → unknown（无可比较对象），不判 consistent
    - 多报告逐对全覆盖（combinations），组级汇总保留逐对结果
    - 不修改原始 AnalysisReport / payload / 已有证据条目；结果写入
      phase_report.consistency 并返回同一对象
    """
    entries = phase_report.entries
    refs = phase_report.report_refs
    src_pairs = []
    qgrid_pairs = []
    # 幂等性：清理本函数上一次生成、以 'audit_comparability:' 为前缀的
    # 日志（只匹配本函数的记录，不误删 collect_evidence 等其他来源的
    # 条目及其顺序），再追加本次审计记录
    log = [line for line in phase_report.rule_log
           if not line.startswith('audit_comparability:')]

    def _valid(i):
        e = entries[i]
        r = refs[i]
        return (e['status'] != STATUS_FAILED and r is not None
                and isinstance(r, AnalysisReport))

    # ---- 精确识别 SF 估计条目（analyzer 全等） ----
    spin_idxs = [i for i, e in enumerate(entries)
                 if e['analyzer'] == ANALYZER_SPIN_SF]
    charge_idxs = [i for i, e in enumerate(entries)
                   if e['analyzer'] == ANALYZER_CHARGE_SF]
    scpair_idxs = [i for i, e in enumerate(entries)
                   if e['analyzer'] == ANALYZER_SC_PAIR_SF]
    known_sf = {ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF, ANALYZER_SC_PAIR_SF}

    # ==================================================================
    # source set 检查
    # ==================================================================
    # Charge SF：diagonal_source_sites 集合逐对比较
    valid_charge = [i for i in charge_idxs if _valid(i)]
    for a, b in combinations(valid_charge, 2):
        sa = (refs[a].payload or {}).get('diagonal_source_sites')
        sb = (refs[b].payload or {}).get('diagonal_source_sites')
        note = ('针对 diagonal_source_sites（对角补全源点），'
                '不等同于无条件完整的测量源点集合')
        if not (_is_int_list(sa) and _is_int_list(sb)):
            status, reason = CMP_UNKNOWN, 'diagonal_source_sites 缺失或类型不合法'
        elif set(sa) == set(sb):
            status, reason = CMP_CONSISTENT, '集合相等（顺序无关）'
        else:
            status, reason = (CMP_INCONSISTENT,
                              f'集合不等: {sorted(sa)} vs {sorted(sb)}')
        src_pairs.append({'type': 'source_set', 'analyzer': ANALYZER_CHARGE_SF,
                          'entries': [a, b], 'status': status,
                          'comparison_level': 'set', 'reason': f'{reason}；{note}'})
    if len(valid_charge) == 1:
        src_pairs.append({'type': 'source_set', 'analyzer': ANALYZER_CHARGE_SF,
                          'entries': [valid_charge[0]], 'status': CMP_UNKNOWN,
                          'comparison_level': 'set',
                          'reason': '无可比较对象（仅一条有效报告）'})
    for i in charge_idxs:
        if not _valid(i):
            src_pairs.append({'type': 'source_set', 'analyzer': ANALYZER_CHARGE_SF,
                              'entries': [i], 'status': CMP_UNKNOWN,
                              'comparison_level': 'set',
                              'reason': f'报告 status={entries[i]["status"]}，'
                                        f'不参与一致断言（原状态保留）'})
    if spin_idxs:
        src_pairs.append({'type': 'source_set', 'analyzer': ANALYZER_SPIN_SF,
                          'entries': spin_idxs, 'status': CMP_UNKNOWN,
                          'comparison_level': 'set',
                          'reason': '自旋结构因子报告无源点身份集合字段，'
                                    '不推测源点'})
    if scpair_idxs:
        src_pairs.append({'type': 'source_set', 'analyzer': ANALYZER_SC_PAIR_SF,
                          'entries': scpair_idxs, 'status': CMP_UNKNOWN,
                          'comparison_level': 'set',
                          'reason': 'SC 配对结构因子的源点身份为参考键'
                                    '（单键，非源点集合），P2 不在此审计'
                                    '（身份由汇总上下文 refbond_dir 标识）'})
    # 其余类型（衰减/能量/峰值）→ not_applicable（组级一条）
    others = [i for i, e in enumerate(entries) if e['analyzer'] not in known_sf]
    if others:
        src_pairs.append({'type': 'source_set', 'analyzer': 'others',
                          'entries': others, 'status': CMP_NOT_APPLICABLE,
                          'comparison_level': 'set',
                          'reason': '非结构因子估计报告，无源点集合概念'})

    # ==================================================================
    # q_grid 检查（同 analyzer 组内逐对，摘要级）
    # ==================================================================
    sf_idxs = spin_idxs + charge_idxs + scpair_idxs
    for a, b in combinations(sf_idxs, 2):
        if entries[a]['analyzer'] != entries[b]['analyzer']:
            continue   # 不同 SF 类型不混组比较
        if not (_valid(a) and _valid(b)):
            continue   # 非有效条目由下方单独记录
        ga = (refs[a].payload or {}).get('q_grid')
        gb = (refs[b].payload or {}).get('q_grid')
        if not (_qgrid_ok(ga) and _qgrid_ok(gb)):
            status, reason = (CMP_UNKNOWN,
                              'q_grid 缺失/结构不完整/空网格/端点 None')
        elif ga == gb:
            status, reason = (CMP_CONSISTENT,
                              '摘要全等（n_qx/n_qy/端点）；完整网格坐标未验证')
        else:
            diff = [k for k in ga if ga[k] != gb[k]]
            status, reason = (CMP_INCONSISTENT, f'摘要不等（差异键: {diff}）')
        qgrid_pairs.append({'type': 'q_grid', 'analyzer': entries[a]['analyzer'],
                            'entries': [a, b], 'status': status,
                            'comparison_level': 'summary', 'reason': reason})
    # 单条有效 / 非有效条目
    for i in sf_idxs:
        if _valid(i) and not any(i in p['entries'] for p in qgrid_pairs):
            qgrid_pairs.append({'type': 'q_grid', 'analyzer': entries[i]['analyzer'],
                                'entries': [i], 'status': CMP_UNKNOWN,
                                'comparison_level': 'summary',
                                'reason': '无可比较对象（仅一条有效报告）'})
        if not _valid(i):
            qgrid_pairs.append({'type': 'q_grid', 'analyzer': entries[i]['analyzer'],
                                'entries': [i], 'status': CMP_UNKNOWN,
                                'comparison_level': 'summary',
                                'reason': f'报告 status={entries[i]["status"]}，'
                                          f'不参与一致断言（原状态保留）'})
    if others:
        qgrid_pairs.append({'type': 'q_grid', 'analyzer': 'others',
                            'entries': others, 'status': CMP_NOT_APPLICABLE,
                            'comparison_level': 'summary',
                            'reason': '非结构因子估计报告，无 q_grid 元数据'})

    consistency = {
        'source_set': {'pairs': src_pairs,
                       'group_summary': _summarize_groups(src_pairs)},
        'q_grid': {'pairs': qgrid_pairs,
                   'group_summary': _summarize_groups(qgrid_pairs)},
    }
    log.append(f"audit_comparability: source_set 逐对 {len(src_pairs)} 条，"
               f"组汇总 {consistency['source_set']['group_summary']}")
    log.append(f"audit_comparability: q_grid 逐对 {len(qgrid_pairs)} 条"
               f"（摘要级，完整坐标未验证），"
               f"组汇总 {consistency['q_grid']['group_summary']}")
    log.append("audit_comparability: 组汇总规则——含 inconsistent→inconsistent；"
               "全 consistent→consistent；全 not_applicable→not_applicable；"
               "其余→unknown；逐对结果优先保留")

    phase_report.consistency = consistency
    phase_report.rule_log = log
    return phase_report


# ==========================================================================
# P3：证据充分性检查（机械事实清单；无物理判据、无阈值、无相态标签）
# ==========================================================================
def evidence_sufficiency_check(phase_report: PhaseDiagnosisReport,
                               *, rules=None) -> PhaseDiagnosisReport:
    """对已汇总报告做证据充分性检查，结果写入 phase_report.sufficiency。

    每个 observable 分组记录：
    - has_any_valid_report：是否存在 status != 'failed' 的有效报告
    - data_restriction_status：'restricted' | 'unrestricted' | 'unknown'
      （只统计有效条目；True/False 混合或含 None → unknown 并记录冲突）
    - comparability：{analyzer: P2 group_summary 状态}，只读取已有
      consistency 信息（不重新审计）；缺失 → unknown
    - notes：仅实际可验证的机械事实
    - entry_indices：对应原始 entries 索引

    overall 保守语义（仅三个枚举值）：
    - 空输入 → 'insufficient'（没有可检查的证据条目；不表示物理相态不存在）
    - 全部条目 failed/None → 'insufficient'（没有有效报告；不表示物理证据不存在）
    - 存在有效报告：
        rules=None → 'unknown'（存在有效报告，但未提供经确认的充分性标准）
        rules 提供且全部 condition 通过 → 'sufficient'
        rules 提供但任一不通过 → 'unknown'
      rules 为 list[dict]，每项 {'name', 'description', 'condition'}，
      condition(entries, per_observable) -> bool，仅执行显式传入的
      机械规则；不引入默认物理阈值
    - sufficient 只能由显式规则产生；P2 可比性状态不改变 overall；
      has_any_valid_report=True 不直接转换为 sufficient

    幂等：sufficiency 字段整体替换；日志按前缀清理再追加。
    不修改原始 AnalysisReport/payload/entries/labels/report_refs。
    """
    entries = phase_report.entries
    log = [line for line in phase_report.rule_log
           if not line.startswith('evidence_sufficiency_check:')]

    # ---- 按 observable 分组（键为 str；None observable → 'None'） ----
    grouped = {}
    for i, e in enumerate(entries):
        key = str(e['observable'])
        grp = grouped.setdefault(
            key, {'entry_indices': [], 'statuses': [], 'restrictions': [],
                  'analyzers': set()})
        grp['entry_indices'].append(i)
        grp['statuses'].append(e['status'])
        grp['restrictions'].append(e['data_restricted'])
        if e['analyzer'] is not None:
            grp['analyzers'].add(e['analyzer'])

    # ---- P2 consistency 只读引用（不重新审计） ----
    group_summary_qg = phase_report.consistency.get(
        'q_grid', {}).get('group_summary', {})

    per_observable = {}
    for key, grp in grouped.items():
        valid = [i for i, s in zip(grp['entry_indices'], grp['statuses'])
                 if s != STATUS_FAILED]
        notes = []
        # data_restriction_status：只看有效条目的值
        drs = [entries[i]['data_restricted'] for i in valid]
        if not valid:
            dr_status = SUFF_UNKNOWN
            notes.append('无有效报告（该 observable 全部 failed/None）')
        elif all(d is True for d in drs):
            dr_status = 'restricted'
        elif all(d is False for d in drs):
            dr_status = 'unrestricted'
        else:
            dr_status = SUFF_UNKNOWN
            if any(d is None for d in drs):
                notes.append('data_restricted 缺失或为 None，标记 unknown（不默认 unrestricted）')
            if any(d is True for d in drs) and any(d is False for d in drs):
                notes.append('同一 observable 内限制状态冲突（True 与 False 并存），标记 unknown')
        notes.append(f'有效报告 {len(valid)}/{len(grp["entry_indices"])} 条')
        # comparability：该 observable 内 analyzer → P2 组汇总（缺失 unknown）
        comparability = {}
        if not phase_report.consistency:
            notes.append('consistency 未审计（P2），可比性标记 unknown')
        for a in sorted(grp['analyzers']):
            comparability[a] = group_summary_qg.get(a, SUFF_UNKNOWN)
        per_observable[str(key)] = {
            'has_any_valid_report': bool(valid),
            'data_restriction_status': dr_status,
            'comparability': comparability,
            'notes': notes,
            'entry_indices': list(grp['entry_indices']),
        }

    # ---- overall（保守语义） ----
    any_valid = any(p['has_any_valid_report'] for p in per_observable.values())
    if not entries:
        overall = SUFF_INSUFFICIENT
        overall_note = '空输入：没有可检查的证据条目（不表示物理相态不存在）'
    elif not any_valid:
        overall = SUFF_INSUFFICIENT
        overall_note = ('全部条目 failed/None：没有有效报告'
                        '（不表示物理证据不存在）')
    elif rules is None:
        overall = SUFF_UNKNOWN
        overall_note = '存在有效报告，但未提供经确认的充分性标准（rules=None）'
    else:
        results = []
        for rule in rules:
            passed = bool(rule['condition'](entries, per_observable))
            results.append((rule['name'], passed))
            log.append(f"evidence_sufficiency_check: 规则 '{rule['name']}' "
                       f"({rule.get('description', '')}) → "
                       f"{'通过' if passed else '不通过'}")
        if results and all(p for _, p in results):
            overall = SUFF_SUFFICIENT
            overall_note = ('显式规则全部通过：'
                            + ', '.join(n for n, _ in results))
        else:
            overall = SUFF_UNKNOWN
            overall_note = ('存在有效报告，但显式规则未全部通过'
                            f'（{results}）')

    log.append(f"evidence_sufficiency_check: overall={overall}——{overall_note}")
    log.append("evidence_sufficiency_check: sufficient 仅由显式规则产生；"
               "P2 可比性状态不改变 overall；insufficient 仅指无可用报告，"
               "不是对物理证据不存在的断言")

    phase_report.sufficiency = {
        'overall': overall,
        'overall_note': overall_note,
        'per_observable': per_observable,
    }
    phase_report.rule_log = log
    return phase_report


# ==========================================================================
# P1-C：标准证据映射（纯函数，不修改传入对象）
# ==========================================================================
# 映射层允许的证据类别（描述性多标签，不代表证据强度或物理判定）。
# 不含 order_evidence / counter_evidence——正式判据证据只能由
# P1-D 已授权规则引擎的判定结果产生，映射层绝不从观测/诊断自动生成。
EVIDENCE_OBSERVATION = 'observation'
EVIDENCE_DERIVED_OBSERVATION = 'derived_observation'
EVIDENCE_QUALITY_DIAGNOSTIC = 'quality_diagnostic'
EVIDENCE_CAPABILITY_LIMITATION = 'capability_limitation'
EVIDENCE_NOT_AVAILABLE = 'not_available'
EVIDENCE_CATEGORIES = (EVIDENCE_OBSERVATION, EVIDENCE_DERIVED_OBSERVATION,
                       EVIDENCE_QUALITY_DIAGNOSTIC,
                       EVIDENCE_CAPABILITY_LIMITATION, EVIDENCE_NOT_AVAILABLE)

# 衰减分析器精确标识（与 Analysis_Entry 路由一致）
_DECAY_ANALYZERS = ('SuperCorrDecayAnalyzer', 'SpinCorrDecayAnalyzer',
                    'ChargeCorrDecayAnalyzer', 'GreenFunctionDecayAnalyzer')

# 固定键集：仅从实际存在的报告字段提取，缺失一律 None——不推断、
# 不从其他字段计算、不对大型元数据自定义摘要算法
_RANGE_KEYS = ('N', 'N_values', 'n_points', 'n_distances', 'n_raw_points',
               'r_min', 'r_max', 'definition', 'n_observed', 'q_grid',
               'lattice_geometry', 'boundary', 'unit_cell_size')
_CAPABILITY_KEYS = ('is_restricted', 'expected_pair_count', 'pair_coverage',
                    'source_coverage', 'record_source', 'imaginary_residual',
                    'phase_capability')


def _mapping_append_unique(item, category):
    if category not in item['categories']:
        item['categories'].append(category)


def _map_single_entry(e, idx, rep):
    """单条 entry → 标准证据条目（纯函数；rep 为对应 report_refs 元素）。"""
    rep_ok = isinstance(rep, AnalysisReport)
    item = {
        'evidence_id': f'e{idx}',
        'entry_index': idx,
        'observable': e.get('observable'),
        'analyzer': e.get('analyzer'),
        'status': e.get('status'),
        'classification': e.get('classification'),
        'evidence': e.get('evidence'),
        'data_restricted': e.get('data_restricted'),
        'warnings': list(e.get('warnings') or []),
        'label': e.get('label'),
        'categories': [],
        'parameters': _convert(rep.parameters) if rep_ok else {},
        'metadata': _convert(rep.metadata) if rep_ok else {},
        'data_range': {k: None for k in _RANGE_KEYS},
        'capability': {k: None for k in _CAPABILITY_KEYS},
        'payload_available': bool(rep_ok and rep.payload is not None),
        'payload_note': None,
        'mapping_note': None,
    }
    if rep is None:
        item['categories'] = [EVIDENCE_NOT_AVAILABLE]
        item['payload_note'] = '报告缺失（None 输入）：无 payload'
        return item
    if not rep_ok:
        item['categories'] = [EVIDENCE_NOT_AVAILABLE]
        item['mapping_note'] = (f'report_refs[{idx}] 不是 AnalysisReport'
                                f'（实际 {type(rep).__name__}），保守归'
                                ' not_available')
        return item
    if rep.status == STATUS_FAILED:
        item['categories'] = [EVIDENCE_NOT_AVAILABLE]
        item['payload_note'] = ('报告 status=failed：分析失败；原始失败'
                                '信息见 parameters/warnings，不伪造'
                                ' observable/analyzer 来源')
        return item
    # ---- 按 analyzer 精确标识分类（不宽泛字符串包含） ----
    a = rep.analyzer
    if a in _DECAY_ANALYZERS:
        if item['parameters']:
            _mapping_append_unique(item, EVIDENCE_OBSERVATION)
        if rep.classification is not None:
            # 原分析器输出的数值拟合分类（8 类）——仅保留为派生观测，
            # 不据此生成相态支持或反证
            _mapping_append_unique(item, EVIDENCE_DERIVED_OBSERVATION)
        if rep.warnings or rep.metadata.get('verdict') is not None \
                or rep.metadata.get('param_errors') \
                or rep.metadata.get('fit_quality'):
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        for k in ('n_points', 'n_distances', 'n_raw_points',
                  'r_min', 'r_max'):
            if k in rep.metadata:
                item['data_range'][k] = rep.metadata[k]
    elif a in (ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF, ANALYZER_SC_PAIR_SF):
        if item['payload_available']:
            _mapping_append_unique(item, EVIDENCE_OBSERVATION)
            item['payload_note'] = ('payload 含 S(q) 等完整结果，经 '
                                    'report_refs[entry_index].payload 访问；'
                                    '映射层不复制不序列化')
        else:
            item['payload_note'] = 'payload 不存在（None）：S(q) 数据不可用'
        if rep.warnings:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        md = rep.metadata
        # capability_limitation 触发条件只用"键存在且取特定值"判断——
        # 键不存在（缺失）不代表受限，不推断
        if md.get('is_restricted') is True \
                or ('expected_pair_count' in md
                    and md['expected_pair_count'] is None) \
                or md.get('record_source') is not None:
            _mapping_append_unique(item, EVIDENCE_CAPABILITY_LIMITATION)
        for k in ('N', 'definition', 'q_grid', 'n_observed'):
            if k in md:
                item['data_range'][k] = md[k]
        for k in _CAPABILITY_KEYS:
            if k in md:
                item['capability'][k] = md[k]
    elif a == ANALYZER_PEAK:
        if item['parameters']:
            # 峰值是数值检测结果（从 S(q) 计算），非原始观测
            _mapping_append_unique(item, EVIDENCE_DERIVED_OBSERVATION)
        if rep.warnings:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        for k in ('N', 'definition'):
            if k in rep.metadata:
                item['data_range'][k] = rep.metadata[k]
    elif a == ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT:
        # P1-G.4：电荷-自旋峰位对齐是派生数值观测（峰位距离），
        # 不构成任何相级支持或反证（峰位重合 ≠ 耦合序、偏离 ≠ 无耦合）。
        # comparability 经 item['metadata'] 保留（不扩 _RANGE_KEYS 键集）
        if item['parameters']:
            _mapping_append_unique(item, EVIDENCE_DERIVED_OBSERVATION)
        if rep.warnings:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
    elif a == ANALYZER_ENERGY:
        if item['parameters']:
            _mapping_append_unique(item, EVIDENCE_OBSERVATION)
        if rep.metadata.get('outlier_diagnosis') is not None:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        if 'N_values' in rep.parameters:
            item['data_range']['N_values'] = rep.parameters['N_values']
    elif a == ANALYZER_SC_PAIRING_SYMMETRY:
        # P1-E.2：配对对称性报告（sc_pairing_symmetry_report 包装）。
        # payload 含三通道原生结果；phase=blocked_by_data 是数据能力
        # 限制，不解释为 d+id/TRSB 反证；缺失通道不伪造观测
        if item['payload_available']:
            _mapping_append_unique(item, EVIDENCE_OBSERVATION)
            item['payload_note'] = ('payload 含 magnitude/sign/phase 三通道'
                                    '原生结果，经 '
                                    'report_refs[entry_index].payload 访问；'
                                    '映射层不复制不序列化')
        else:
            item['payload_note'] = 'payload 不存在（None）：配对对称性结果不可用'
        if rep.warnings:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        md = rep.metadata
        if md.get('phase_state') == 'blocked_by_data':
            _mapping_append_unique(item, EVIDENCE_CAPABILITY_LIMITATION)
            item['capability']['phase_capability'] = 'blocked_by_data'
    elif a == ANALYZER_DENSITY:
        # P1-G.1：局域密度数值观测。parameters 含逐格点向量/行平均/
        # 子晶格平均/晶胞分组平均；N 与几何边界元数据入 data_range
        # （供空间观测消费）。密度空间分布是观测，不构成任何相级
        # 支持或反证（不判 Stripe）。
        if item['parameters']:
            _mapping_append_unique(item, EVIDENCE_OBSERVATION)
        if rep.warnings:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        md = rep.metadata
        for k in ('N', 'lattice_geometry', 'boundary', 'unit_cell_size'):
            if k in md:
                item['data_range'][k] = md[k]
    else:
        # 未知 analyzer：仅按实际存在的内容保守归类
        if item['parameters']:
            _mapping_append_unique(item, EVIDENCE_OBSERVATION)
        if rep.warnings:
            _mapping_append_unique(item, EVIDENCE_QUALITY_DIAGNOSTIC)
        item['mapping_note'] = (f'未知 analyzer {a!r}：仅按实际存在字段'
                                '保守归类')
    return item


def build_evidence_items(phase_report):
    """把 PhaseDiagnosisReport 的 entries/report_refs 映射为标准证据条目
    列表（P1-C，纯函数）。

    - 不修改传入的 phase_report 或任何 AnalysisReport；返回值由调用方
      决定是否保存（如赋值给 phase_report.evidence_items）
    - 每条条目为可 JSON 序列化基础类型（dict/list/str/float/None），
      不含 report 对象引用；payload 不复制，经稳定的 entry_index 关联
      report_refs[entry_index].payload 访问（payload_available/
      payload_note 说明可用性与访问方式）
    - 字段仅从实际存在的报告字段提取；未知/缺失填 None，不推断
    - categories ⊆ EVIDENCE_CATEGORIES（五类，描述性多标签，可为空
      列表）；不产生 order_evidence/counter_evidence，不修改候选
      verdict 或总体 conclusion
    - failed/None 报告 → not_available，保留原失败信息与索引关系，
      不伪造 observable/analyzer
    - 索引越界（report_refs 短于 entries）→ 该条目按 None 报告处理；
      重复调用结果确定（纯函数）
    """
    entries = getattr(phase_report, 'entries', None) or []
    refs = getattr(phase_report, 'report_refs', None) or []
    items = []
    for idx, e in enumerate(entries):
        rep = refs[idx] if idx < len(refs) else None
        items.append(_map_single_entry(e, idx, rep))
    return items


# ==========================================================================
# P1-D.2：通用候选判定规则引擎（P1-B.2/P1-D.1.3 冻结语义）
# ==========================================================================
# ---- 规则级六态（枚举冻结，不扩展） ----
RULE_EXECUTED_SUPPORTED = 'executed_supported'
RULE_EXECUTED_UNSATISFIED = 'executed_unsatisfied'
RULE_COUNTER_TRIGGERED = 'counter_triggered'
RULE_NOT_APPLICABLE = 'not_applicable'
RULE_REJECTED = 'rule_rejected'
RULE_GATE_FAILED = 'gate_failed'
RULE_LEVEL_STATES = (RULE_EXECUTED_SUPPORTED, RULE_EXECUTED_UNSATISFIED,
                     RULE_COUNTER_TRIGGERED, RULE_NOT_APPLICABLE,
                     RULE_REJECTED, RULE_GATE_FAILED)

# ---- 候选级九态（枚举冻结，不扩展） ----
VERDICT_SUPPORTED = 'supported'
VERDICT_CONTRADICTED = 'contradicted'
VERDICT_CONFLICTING = 'conflicting'
VERDICT_INSUFFICIENT = 'insufficient'
VERDICT_QUALITY_FAILED = 'quality_failed'
VERDICT_CRITERIA_UNDEFINED = 'criteria_undefined'
VERDICT_DATA_MISSING = 'data_missing'
VERDICT_ANALYSIS_UNAVAILABLE = 'analysis_unavailable'
VERDICT_NOT_MEASURABLE = 'not_measurable'
VERDICT_STATES = (VERDICT_SUPPORTED, VERDICT_CONTRADICTED,
                  VERDICT_CONFLICTING, VERDICT_INSUFFICIENT,
                  VERDICT_QUALITY_FAILED, VERDICT_CRITERIA_UNDEFINED,
                  VERDICT_DATA_MISSING, VERDICT_ANALYSIS_UNAVAILABLE,
                  VERDICT_NOT_MEASURABLE)
# 缺失解释优先级（仅用于多项缺失时选择主 verdict；detail 保留全部）
_MISSING_PRIORITY = {VERDICT_NOT_MEASURABLE: 3,
                     VERDICT_ANALYSIS_UNAVAILABLE: 2,
                     VERDICT_DATA_MISSING: 1}

# ---- 总体 conclusion 七值 ----
CONCLUSION_CONFLICTING = 'conflicting'
CONCLUSION_MIXED = 'mixed'
CONCLUSION_PARTIALLY_UNRESOLVED = 'partially_unresolved'
CONCLUSION_SUPPORTED = 'supported'
CONCLUSION_CONTRADICTED_ALL = 'contradicted_all'
# CONCLUSION_INCONCLUSIVE / CONCLUSION_EVIDENCE_COLLECTED 沿用既有常量

# ---- 授权清单 ----
# 引擎默认空表：仅凭规则对象存在不执行。P1-F.3 首批授权（观测级 × 3 +
# 能力边界 × 2，criteria_version=phase_criteria_v1_observation）在文件
# 末尾"P1-F.3：首批规则定义"一节注册（AUTHORIZED_RULES.update(...)），
# 规则定义与授权相邻放置便于审计。未授权任何相级判定规则（支持/反证
# 条款只能由后续单独授权的规则产生）。
# 授权必须绑定 rule_id + criteria_version；仅凭规则对象存在不执行。
AUTHORIZED_RULES = {}


def _safe_payloads(entries, refs):
    """构造 {entry_index: payload|None} 安全访问字典（越界/引用失效/
    访问异常 → None，不传播）。规则回调禁止直接访问 report_refs。"""
    out = {}
    for idx in range(len(entries)):
        payload = None
        if idx < len(refs) and isinstance(refs[idx], AnalysisReport):
            try:
                payload = refs[idx].payload
            except Exception:
                payload = None
        out[idx] = payload
    return out


def _match_required_evidence(specs, items):
    """required_evidence 逐项匹配 P1-C 标准证据条目。

    匹配规则：spec 各维度（observable/analyzer/category）None = 不约束；
    精确相等匹配；failed/not_available 条目不参与。
    missing_interpretation 未声明/未知/缺 basis → 保守 data_missing；
    not_measurable/analysis_unavailable 必须带 basis（依据说明），
    缺失即降级——引擎不校验依据的物理正确性（属授权流程），只强制
    其存在以保证可追溯。
    返回 (matched_specs, missing)：匹配成功项列表与缺失项列表。
    """
    matched_specs = {}
    missing = []
    for i, spec in enumerate(specs):
        matched = [it for it in items
                   if (spec.get('observable') is None
                       or it['observable'] == spec['observable'])
                   and (spec.get('analyzer') is None
                        or it['analyzer'] == spec['analyzer'])
                   and (spec.get('category') is None
                        or spec['category'] in it['categories'])
                   and it['status'] != STATUS_FAILED
                   and EVIDENCE_NOT_AVAILABLE not in it['categories']]
        min_n = int(spec.get('min_entries', 1))
        if len(matched) >= min_n:
            matched_specs[i] = [it['entry_index'] for it in matched]
            continue
        interp = spec.get('missing_interpretation')
        basis = spec.get('basis')
        if interp is None:
            interp, detail = VERDICT_DATA_MISSING, \
                '未声明 missing_interpretation，保守按 data_missing'
        elif interp not in (VERDICT_DATA_MISSING,
                            VERDICT_ANALYSIS_UNAVAILABLE,
                            VERDICT_NOT_MEASURABLE):
            interp, detail = VERDICT_DATA_MISSING, \
                f'未知 interpretation {interp!r}，保守按 data_missing'
        elif interp != VERDICT_DATA_MISSING and not basis:
            interp, detail = VERDICT_DATA_MISSING, \
                f'声明 {interp} 但缺少 basis（依据说明），降级 data_missing'
        else:
            detail = basis
        missing.append({
            'spec': {k: v for k, v in spec.items()
                     if k not in ('missing_interpretation', 'basis')},
            'interpretation': interp, 'basis': basis, 'detail': detail,
            'matched_count': len(matched)})
    return matched_specs, missing


def _execute_rule(rule, items, payloads, authorized):
    """执行单条规则 → rule_trace 记录。

    rule_level_state 六态之一或 None（规则适用但未进入判定：证据缺失
    或执行异常）；execution_status（completed/exception）与
    evidence_missing / exception 独立字段严格区分四类情形。
    """
    tr = {
        'rule_id': rule.get('rule_id'),
        'candidate': rule.get('candidate'),
        'criteria_version': rule.get('criteria_version'),
        'rule_level_state': None,
        'execution_status': 'completed',
        'evidence_missing': None,
        'exception': None,
        'entry_indices': [],
        'support_evaluation': None,
        'counter_evaluation': None,
        'reason': '',
    }

    def _exception(stage, ex, extra=None):
        tr['execution_status'] = 'exception'
        tr['exception'] = {'stage': stage, 'type': type(ex).__name__,
                           'message': str(ex)}
        if extra:
            tr['exception'].update(extra)
        tr['reason'] = f'{stage} 回调异常（{type(ex).__name__}: {ex}）'

    # ---- 授权校验（rule_id + criteria_version 双绑定） ----
    if tr['rule_id'] not in authorized \
            or authorized[tr['rule_id']] != tr['criteria_version']:
        tr['rule_level_state'] = RULE_REJECTED
        tr['reason'] = ('规则未授权或版本不匹配（授权清单中: '
                        + repr(authorized.get(tr['rule_id'], '未注册')) + '）')
        return tr

    # ---- applicability（不含证据存在性判断） ----
    try:
        app = rule['applicability'](items)
        applicable = bool(app.get('applicable'))
        app_reason = app.get('reason', '')
    except Exception as ex:
        _exception('applicability', ex)
        return tr
    if not applicable:
        tr['rule_level_state'] = RULE_NOT_APPLICABLE
        tr['reason'] = f'规则不适用: {app_reason or "（无原因记录）"}'
        return tr

    # ---- required_evidence 匹配（缺失 → 独立字段，不记 gate_failed） ----
    matched_specs, missing = _match_required_evidence(
        rule.get('required_evidence') or [], items)
    if missing:
        tr['evidence_missing'] = missing
        tr['reason'] = ('规则适用但必需证据缺失（详见 evidence_missing；'
                        'rule_level_state=None 表示未进入判定）')
        return tr
    idxs = sorted({i for lst in matched_specs.values() for i in lst})
    tr['entry_indices'] = idxs
    matched = [it for it in items if it['entry_index'] in idxs]
    matched_payloads = {i: payloads.get(i) for i in idxs}

    # ---- quality_gates（回调正常执行且明确返回 passed=False 才算失败） ----
    for gate in rule.get('quality_gates') or []:
        try:
            res = gate['check'](matched, matched_payloads)
            if not bool(res.get('passed')):
                tr['rule_level_state'] = RULE_GATE_FAILED
                tr['reason'] = (f"质量门 '{gate.get('name', '?')}' 明确失败: "
                                + str(res.get('detail', '无细节')))
                return tr
        except Exception as ex:
            _exception('quality_gates', ex, {'gate': gate.get('name')})
            return tr

    # ---- evaluation / counter_evidence（异常不贡献支持/反证） ----
    try:
        support = bool(rule['evaluation'](matched, matched_payloads))
    except Exception as ex:
        _exception('evaluation', ex)
        return tr
    counter = None
    ctr_cfg = rule.get('counter_evidence')
    if ctr_cfg is not None:
        try:
            counter = bool(ctr_cfg['evaluation'](matched, matched_payloads))
        except Exception as ex:
            _exception('counter_evidence', ex)
            return tr
    tr['support_evaluation'] = support
    tr['counter_evaluation'] = counter
    if counter:
        tr['rule_level_state'] = RULE_COUNTER_TRIGGERED
        note = ctr_cfg.get('note') or ''
        tr['reason'] = '正式反证条款成立' + (f'；{note}' if note else '')
    elif support:
        tr['rule_level_state'] = RULE_EXECUTED_SUPPORTED
        tr['reason'] = '支持条件成立'
    else:
        tr['rule_level_state'] = RULE_EXECUTED_UNSATISFIED
        tr['reason'] = '支持条件不成立（不构成反证）'
    return tr


def _evaluate_candidate(cand, rules, items, payloads, authorized):
    """单候选多规则聚合（P1-D.1.3 决策表）。"""
    traces = [r for r in (_execute_rule(rule, items, payloads, authorized)
                          for rule in rules)]
    executed = [t for t in traces
                if t['rule_level_state'] in (RULE_EXECUTED_SUPPORTED,
                                             RULE_EXECUTED_UNSATISFIED,
                                             RULE_COUNTER_TRIGGERED)]
    rejected = [t for t in traces if t['rule_level_state'] == RULE_REJECTED]
    not_appl = [t for t in traces
                if t['rule_level_state'] == RULE_NOT_APPLICABLE]
    gate_fails = [t for t in traces
                  if t['rule_level_state'] == RULE_GATE_FAILED]
    exceptions = [t for t in traces
                  if t['execution_status'] == 'exception']
    missing_rules = [t for t in traces if t['evidence_missing']]
    all_missing = sorted({m['interpretation']
                          for t in missing_rules
                          for m in t['evidence_missing']},
                         key=lambda v: -_MISSING_PRIORITY[v])

    if not traces:
        verdict, reason = VERDICT_CRITERIA_UNDEFINED, '该候选无规则'
    elif len(rejected) == len(traces):
        verdict, reason = VERDICT_CRITERIA_UNDEFINED, \
            '全部规则未授权或版本不匹配（拒绝执行）'
    elif len(not_appl) == len(traces):
        verdict, reason = VERDICT_CRITERIA_UNDEFINED, \
            '全部规则不适用（判据不可执行）'
    elif not executed and exceptions:
        verdict, reason = VERDICT_ANALYSIS_UNAVAILABLE, \
            ('全部已授权且适用规则执行异常，本次诊断不可用'
             '（异常详情见 execution_errors）')
    elif not executed and gate_fails:
        verdict, reason = VERDICT_QUALITY_FAILED, \
            '全部可执行规则质量门失败（数据质量不合格，不等于候选被否定）'
    elif not executed and missing_rules:
        verdict = all_missing[0]
        reason = ('全部可执行规则均因必需证据缺失未进入判定；'
                  f'缺失解释: {all_missing}（优先级取最高项）')
    elif not executed:
        verdict, reason = VERDICT_CRITERIA_UNDEFINED, \
            '无有效判定结果（详见 rule_trace）'
    else:
        any_support = any(t['support_evaluation'] is True for t in executed)
        any_counter = any(t['counter_evaluation'] is True for t in executed)
        if any_support and any_counter:
            verdict, reason = VERDICT_CONFLICTING, \
                '支持条款与正式反证条款同时成立（证据冲突）'
        elif any_counter:
            verdict, reason = VERDICT_CONTRADICTED, \
                '正式反证成立且支持条件不成立'
        elif any_support:
            verdict, reason = VERDICT_SUPPORTED, \
                '支持条件成立且无正式反证'
        else:
            verdict, reason = VERDICT_INSUFFICIENT, \
                '支持条件不成立且无正式反证（证据不足以判断）'

    detail = {
        'rejected': [t['rule_id'] for t in rejected],
        'not_applicable': [{'rule_id': t['rule_id'], 'reason': t['reason']}
                           for t in not_appl],
        'gate_failures': [{'rule_id': t['rule_id'], 'reason': t['reason']}
                          for t in gate_fails],
        'execution_errors': [t['exception'] for t in exceptions],
        'evidence_missing': [m for t in missing_rules
                             for m in t['evidence_missing']],
    }
    return {'candidate': cand, 'verdict': verdict, 'reason': reason,
            'detail': detail}, traces


def _summarize_verdicts(verdicts):
    """候选级九态 → 六组汇总（P1-B.1 分组）。"""
    groups = {
        'supported_candidates': [], 'contradicted_candidates': [],
        'conflicting_candidates': [],
        'insufficient_or_quality_failed': [],
        'criteria_undefined_candidates': [], 'unresolved_candidates': [],
    }
    counts = {v: 0 for v in VERDICT_STATES}
    for cand, v in verdicts.items():
        counts[v['verdict']] += 1
        if v['verdict'] == VERDICT_SUPPORTED:
            groups['supported_candidates'].append(cand)
        elif v['verdict'] == VERDICT_CONTRADICTED:
            groups['contradicted_candidates'].append(cand)
        elif v['verdict'] == VERDICT_CONFLICTING:
            groups['conflicting_candidates'].append(cand)
        elif v['verdict'] in (VERDICT_INSUFFICIENT, VERDICT_QUALITY_FAILED):
            groups['insufficient_or_quality_failed'].append(cand)
        elif v['verdict'] == VERDICT_CRITERIA_UNDEFINED:
            groups['criteria_undefined_candidates'].append(cand)
        else:   # data_missing / analysis_unavailable / not_measurable
            groups['unresolved_candidates'].append(cand)
    return {'counts': counts, **groups}


def _aggregate_conclusion(verdicts):
    """候选级九态 → 总体 conclusion 七值（P1-B.2 优先级 1→7，互斥）。"""
    vs = [v['verdict'] for v in verdicts.values()]
    if not vs:
        return CONCLUSION_EVIDENCE_COLLECTED
    if VERDICT_CONFLICTING in vs:
        return CONCLUSION_CONFLICTING
    has_sup = VERDICT_SUPPORTED in vs
    has_ctr = VERDICT_CONTRADICTED in vs
    unresolved = {VERDICT_CRITERIA_UNDEFINED, VERDICT_DATA_MISSING,
                  VERDICT_ANALYSIS_UNAVAILABLE, VERDICT_NOT_MEASURABLE}
    if has_sup and has_ctr:
        return CONCLUSION_MIXED
    if (has_sup or has_ctr) and any(v in unresolved for v in vs):
        return CONCLUSION_PARTIALLY_UNRESOLVED
    if has_sup:
        return CONCLUSION_SUPPORTED
    if all(v == VERDICT_CONTRADICTED for v in vs):
        return CONCLUSION_CONTRADICTED_ALL
    return CONCLUSION_INCONCLUSIVE


def run_candidate_rules(phase_report, rules, authorized=None, candidates=None):
    """P1-D.2 通用候选判定规则引擎入口。

    - rules：list[CandidateRule]（rule_id/candidate/criteria_version/
      applicability/required_evidence/quality_gates/evaluation/
      counter_evidence/result_mapping）
    - authorized：{rule_id: criteria_version}；None → 模块级
      AUTHORIZED_RULES（初始空表——不注册真实物理判据；未授权规则
      拒绝执行，仅凭规则对象存在不执行）
    - candidates：可选显式候选清单（list[str]）。None → 候选集合仅由
      rules 的 candidate 字段定义（原有行为）；显式清单中没有任何
      规则的候选 → verdict=criteria_undefined（判定不可执行，reason
      '该候选无规则'）。**候选清单不赋予规则授权**——授权只来自
      authorized/AUTHORIZED_RULES
    - 消费 P1-C build_evidence_items 的标准证据条目；规则回调只接收
      (matched_items, matched_payloads)，payload 经引擎安全访问字典
      提供（越界/失效 → None），禁止直接访问 report_refs
    - 写入 phase_report.candidate_verdicts / verdict_summary /
      rule_trace / conclusion（七值）；rule_log 追加摘要文本
    - 不修改 entries/report_refs/consistency/sufficiency；不产生
      order_evidence/counter_evidence 证据类别
    """
    authorized = AUTHORIZED_RULES if authorized is None else authorized
    entries = phase_report.entries
    refs = phase_report.report_refs
    items = build_evidence_items(phase_report)
    payloads = _safe_payloads(entries, refs)

    by_candidate = {}
    for rule in rules:
        by_candidate.setdefault(rule['candidate'], []).append(rule)
    for cand in (candidates or []):
        by_candidate.setdefault(cand, [])   # 显式候选：无规则 → 空列表

    verdicts = {}
    trace = []
    for cand, cand_rules in by_candidate.items():
        v, t = _evaluate_candidate(cand, cand_rules, items, payloads,
                                   authorized)
        verdicts[cand] = v
        trace.extend(t)

    phase_report.candidate_verdicts = verdicts
    phase_report.verdict_summary = _summarize_verdicts(verdicts)
    phase_report.rule_trace = trace
    phase_report.conclusion = _aggregate_conclusion(verdicts)
    phase_report.rule_log = phase_report.rule_log + [
        f"run_candidate_rules: 执行 {len(trace)} 条规则、"
        f"{len(verdicts)} 个候选，conclusion={phase_report.conclusion}；"
        "candidate_verdicts 为候选级九态判定（supported/contradicted 仅"
        "由已授权规则产生），未决候选与执行异常见 verdict_detail 与"
        "rule_trace"]
    return phase_report


# ==========================================================================
# P1-F.3：首批规则定义（观测级 × 3 + 能力边界 × 2；无相级判定规则）
# ==========================================================================
# criteria_version：phase_criteria_v1_observation（P1-F.3 任务书建议版本，
# 最终命名待确认）。
#
# 接口决策（P1-F.3 关键要求核验）：当前引擎规则层只输出候选级 verdict
# （九态），无独立"观测结果"类型；但 verdict 空间含非判定态
# （insufficient / data_missing / not_measurable / quality_failed 等），
# 观测内容本身由 P1-C 证据层表达（observation / derived_observation /
# quality_diagnostic / capability_limitation / not_available 五类）。
# 因此观测级规则按以下方式注册，不修改冻结架构与状态枚举：
#
# - 观测级规则（sc_assoc_v1/cdw_v1/sdw_v1）：evaluation 恒 False（无支持
#   条款）、counter_evidence 为 None（无正式反证条款）→ 候选 verdict 结构
#   上永不 supported / contradicted / conflicting；观测存在且纪律门通过时
#   verdict=insufficient（"证据不足以判断"——观测级规则不提供相级判定），
#   观测缺失时 data_missing，纪律门失败时 quality_failed。候选 ↔ 观测条目
#   的关联由 rule_trace.entry_indices 记录；规则级 reason 为引擎通用文案，
#   观测级语义由 rule_id + criteria_version + 本节注释承载。
# - required_evidence 中 min_entries=0 的 spec 表示"观测存在时纳入关联、
#   缺失不阻塞"的附属证据（引擎原语义 len(matched) >= min_n）。
# - 能力边界规则（did_phase_v1/pdw_v1）：required_evidence 含一条永不可
#   匹配的 spec（missing_interpretation='not_measurable' + basis 依据
#   说明）→ 候选 verdict 恒 not_measurable（缺失解释优先级最高），永不
#   supported / contradicted。
# - P1-G.4 起 stripe_v1 以能力边界规则形式注册（与 did/pdw 同模式）：
#   O5 审查确认 N200/N400 测量源码版本不同（N200 早于 44269cb、
#   N400 在其后）、D 调度不同、源点身份不同且仅两个尺寸——体相
#   条纹序判定不可测量，恒 not_measurable（永不 contradicted）。
# - 仍暂缓：sc_lro_v1（不实现，不授权）。未授权规则即使存在规则
#   对象也不执行。
P1F3_CRITERIA_VERSION = 'phase_criteria_v1_observation'


def _observation_evaluation(matched_items, matched_payloads):
    """观测级规则统一 evaluation：恒 False。

    返回 True 会触发候选级 supported——观测级规则结构上禁止该路径：
    幂律衰减不确立配对（sc）、指数衰减不反证（sc）、q=0 增强不确认
    （sc）、峰值数值不判定（cdw/sdw）。支持/反证只能由后续单独授权
    的相级判定规则产生。
    """
    return False


# ---- 电荷 SF 连通关联定义（与 Charge_Structure_Factor 模块的
# DEF_CHARGE_DENSITY_CONNECTED 一致；本模块不 import 叶子模块，
# 按 P1-C 报告的 metadata['definition'] 字符串核对） ----
CHARGE_DEFINITION_CONNECTED = 'charge_density_connected'


def _cdw_connected_definition_gate(matched_items, matched_payloads):
    """纪律门：电荷结构因子必须是连通关联定义（任务书"必须连通关联"）。

    定义缺失或非连通 → 明确不通过（gate_failed → 候选 quality_failed，
    数据质量不合格，不等于候选被否定）。不施加任何峰值阈值或峰显著性
    判据（任务书"无通用峰值阈值"）。
    """
    for it in matched_items:
        d = it.get('data_range', {}).get('definition')
        if d != CHARGE_DEFINITION_CONNECTED:
            return {'passed': False,
                    'detail': (f'电荷 SF 定义 {d!r} 非连通关联（要求 '
                               f'{CHARGE_DEFINITION_CONNECTED!r}）；'
                               '原始密度关联不可作为 CDW 观测')}
    return {'passed': True,
            'detail': f'全部 {len(matched_items)} 条电荷 SF 为连通关联定义'}


# sdw_v1 纪律门：波矢物理标签必须在晶格几何核对前保持未定——参数层出现
# 以下任一标签键即视为提前解释，门不通过（保守）。
_SDW_FORBIDDEN_LABEL_KEYS = ('afm', 'afm_label', 'gamma', 'gamma_label',
                             'k_label', 'wavevector_label', 'phase_label',
                             'order_label')


def _sdw_unlabeled_gate(matched_items, matched_payloads):
    """纪律门：自旋 SF / 峰值观测的参数层不得携带 AFM/Γ/K 等物理波矢
    标签（任务书：波矢标签在晶格几何核对前必须保持未定）。

    现有峰值报告参数仅含数值键（n_peaks/peak_positions/peak_heights），
    本门恒通过；若未来报告加入标签键，门失败 → quality_failed 触发复核。
    """
    for it in matched_items:
        hits = [k for k in _SDW_FORBIDDEN_LABEL_KEYS
                if k in (it.get('parameters') or {})]
        if hits:
            return {'passed': False,
                    'detail': (f'条目 e{it["entry_index"]} 参数含波矢物理'
                               f'标签键 {hits}（波矢解释未获授权，须保持'
                               '未定）')}
    return {'passed': True,
            'detail': '参数层无 AFM/Γ/K 等波矢物理标签（数值位置保持未定）'}


RULE_SC_ASSOC_V1 = {
    'rule_id': 'sc_assoc_v1',
    'candidate': 'superconductivity',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '观测级规则：检查配对关联观测存在性并记录，不判定相'},
    'required_evidence': [
        # 必需：SC 衰减观测（配对关联衰减记录，含数值拟合分类/指数）
        {'observable': OBSERVABLE_SUPERCONDUCTING_CORRELATION,
         'analyzer': ANALYZER_DECAY_SUPERCONDUCTING,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 1},
        # 附属：SC 配对结构因子（存在时纳入 entry_indices 关联）
        {'observable': OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
         'analyzer': ANALYZER_SC_PAIR_SF,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 0},
        # 附属：配对对称性观测（存在时纳入关联）
        {'observable': OBSERVABLE_SC_PAIRING_SYMMETRY,
         'analyzer': ANALYZER_SC_PAIRING_SYMMETRY,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 0},
    ],
    'quality_gates': [],
    'evaluation': _observation_evaluation,
    'counter_evidence': None,
}
# 禁止条款（结构保证，无运行时判定）：幂律衰减 → 不判支持；指数衰减 →
# 不判反证；q=0 增强 → 不判确认；Luther-Emery 关系 → 不作为必要条件；
# 不输出任何长程序结论。


RULE_CDW_V1 = {
    'rule_id': 'cdw_v1',
    'candidate': 'cdw',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '观测级规则：记录电荷结构因子观测（必须连通关联），不判定相'},
    'required_evidence': [
        # 必需：电荷 SF 观测（连通关联定义由质量门强制）
        {'observable': OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
         'analyzer': ANALYZER_CHARGE_SF,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 1},
    ],
    'quality_gates': [{'name': 'connected_definition',
                       'check': _cdw_connected_definition_gate}],
    'evaluation': _observation_evaluation,
    'counter_evidence': None,
}
# 禁止条款：无通用峰值阈值——峰值数值本身不构成任何相级支持/反证。


RULE_SDW_V1 = {
    'rule_id': 'sdw_v1',
    'candidate': 'sdw',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '观测级规则：记录自旋结构因子与峰值 q 位置，不判定相'},
    'required_evidence': [
        # 必需：自旋 SF 观测
        {'observable': OBSERVABLE_SPIN_STRUCTURE_FACTOR,
         'analyzer': ANALYZER_SPIN_SF,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 1},
        # 附属：峰值数值观测（存在时纳入关联）
        {'observable': OBSERVABLE_SPIN_STRUCTURE_FACTOR,
         'analyzer': ANALYZER_PEAK,
         'category': EVIDENCE_DERIVED_OBSERVATION, 'min_entries': 0},
    ],
    'quality_gates': [{'name': 'wavevector_unlabeled',
                       'check': _sdw_unlabeled_gate}],
    'evaluation': _observation_evaluation,
    'counter_evidence': None,
}
# 禁止条款：q 位置不解释为 AFM/Γ/K（波矢标签在晶格几何核对前保持未定）；
# ⟨Sz⟩≈0 不排除 SDW；SU(2) 偏差数值不构成相级支持/反证。


RULE_DID_PHASE_V1 = {
    'rule_id': 'did_phase_v1',
    'candidate': 'd_id_phase',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '能力边界规则：实值数据下 d+id/TRSB 判定不可测量'},
    'required_evidence': [
        # 附属：配对对称性观测（存在时纳入关联；缺失不阻塞能力结论）
        {'observable': OBSERVABLE_SC_PAIRING_SYMMETRY,
         'analyzer': ANALYZER_SC_PAIRING_SYMMETRY,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 0},
        # 必需但永不可匹配：复相位信息——d+id/TRSB 判定的必要输入
        {'observable': 'complex_phase_data',
         'analyzer': None, 'category': None, 'min_entries': 1,
         'missing_interpretation': VERDICT_NOT_MEASURABLE,
         'basis': ('d+id/TRSB 相判定需要配对关联的复相对相位信息；'
                   '当前全部数据为实值 MPS 基态（测量源码 USE_REAL=1，'
                   'B1.1 已核实），数据虚部恒 0，复相位/手性原则上不可'
                   '恢复——该判定在当前数据下不可测量')},
    ],
    'quality_gates': [],
    'evaluation': _observation_evaluation,   # 结构上不可达：必需证据恒缺失
    'counter_evidence': None,
}


RULE_PDW_V1 = {
    'rule_id': 'pdw_v1',
    'candidate': 'pdw',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '能力边界规则：单参考键受限数据下 PDW 判定不可测量'},
    'required_evidence': [
        # 附属：SC 配对结构因子（存在时纳入关联）
        {'observable': OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
         'analyzer': ANALYZER_SC_PAIR_SF,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 0},
        # 必需但永不可匹配：多参考键配对矩阵——PDW 空间调制序判定的必要输入
        {'observable': 'pairing_matrix_multi_refbond',
         'analyzer': None, 'category': None, 'min_entries': 1,
         'missing_interpretation': VERDICT_NOT_MEASURABLE,
         'basis': ('PDW 空间调制序判定需要不同参考键的配对关联矩阵'
                   '（P(q) 多行）；当前测量为单参考键受限数据（P(q) 仅'
                   '一行），调制序不可测量；不使用 Q_PDW≈2Q_stripe 文献'
                   '关系作为判定条件')},
    ],
    'quality_gates': [],
    'evaluation': _observation_evaluation,   # 结构上不可达：必需证据恒缺失
    'counter_evidence': None,
}


RULE_STRIPE_V1 = {
    'rule_id': 'stripe_v1',
    'candidate': 'stripe',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '能力边界规则：体相条纹序在现有数据下不可测量'},
    'required_evidence': [
        # 附属：O1 密度观测 / O2 电荷峰 / O4 峰位对齐（存在时纳入关联）
        {'observable': OBSERVABLE_LOCAL_DENSITY,
         'analyzer': ANALYZER_DENSITY,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 0},
        {'observable': OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
         'analyzer': ANALYZER_PEAK,
         'category': EVIDENCE_DERIVED_OBSERVATION, 'min_entries': 0},
        {'observable': OBSERVABLE_CHARGE_SPIN_PEAK_ALIGNMENT,
         'analyzer': ANALYZER_CHARGE_SPIN_PEAK_ALIGNMENT,
         'category': EVIDENCE_DERIVED_OBSERVATION, 'min_entries': 0},
        # 必需但永不可匹配：同口径多尺寸序列——体相条纹序判定的必要输入
        {'observable': 'stripe_size_series',
         'analyzer': None, 'category': None, 'min_entries': 1,
         'missing_interpretation': VERDICT_NOT_MEASURABLE,
         'basis': ('体相条纹序判定需要同测量源码版本 / 同 DMRG 协议 / '
                   '同源点口径的多尺寸密度与连通电荷 SF 序列（用于区分'
                   '开边界诱导调制与体相调制、建立尺寸趋势）；当前仅有 '
                   'N200/N400 两个尺寸，且 O5 审查确认：测量源码版本不同'
                   '（N200 数据早于测量源码 44269cb，N400 在其后）、D 调度'
                   '不同（1500/2000→5000 vs 500/1000/1500/2000→5000）、'
                   '源点身份不同（49/50 vs 97/98）——空间调制序的体相'
                   '解释不可测量；单个 SF 峰、单个密度调制与两尺寸对比'
                   '均不构成体相条纹证据，也不构成其反证')},
    ],
    'quality_gates': [],
    'evaluation': _observation_evaluation,   # 结构上不可达：必需证据恒缺失
    'counter_evidence': None,
}


# ---- P1-G.28：有限尺寸分析能力边界规则 ----
# 语义声明：以下两条规则的 candidate（'gap_scaling' / 'width_scaling'）是
# 有限尺寸分析能力诊断对象，**不是物理相候选**——引擎机制层的 candidate
# 字段是通用诊断对象键（P1-G.28-A 审查核实：_evaluate_candidate /
# _summarize_verdicts 对 candidate 字符串零类型区分），但本文件规则定义
# 的候选集合按物理相语境维护，能力诊断对象显式标注以免混用。
# verdict 恒 not_measurable（与 did/pdw/stripe 同模式）：表达"该分析在
# 当前数据能力下不可开展"，不表达任何物理量/物理相的存在性结论。

RULE_GAP_V1 = {
    'rule_id': 'gap_v1',
    'candidate': 'gap_scaling',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '能力边界规则（分析能力诊断对象，非物理相候选）：'
                  '能隙定量分析在当前数据下不可测量'},
    'required_evidence': [
        # 附属：能量外推观测（存在时纳入关联；缺失不阻塞能力结论）
        {'observable': OBSERVABLE_ENERGY,
         'analyzer': ANALYZER_ENERGY,
         'category': EVIDENCE_OBSERVATION, 'min_entries': 0},
        # 必需但永不可匹配：激发态/扇区能量差——能隙分析的唯一有效输入
        {'observable': 'gap_excitation_energy',
         'analyzer': None, 'category': None, 'min_entries': 1,
         'missing_interpretation': VERDICT_NOT_MEASURABLE,
         'basis': ('能隙定量分析（自旋/电荷/单粒子隙及其尺寸标度）需要'
                   '激发态能量或不同粒子数扇区的基态能量差作为输入；'
                   '当前全部可用能量数据（Data_Energy 6 doping × 5 N '
                   '共 30 份日志）均为单扇区基态 DMRG 优化日志（仅 '
                   'E0/TruncErr 扫描行，无激发态或扇区分辨信息，P1-G.26 '
                   '已核实）。禁止从基态总能量随掺杂或系统尺寸的变化'
                   '直接构造激发态能隙。not_measurable 表示输入数据不足，'
                   '不表示能隙为零、能隙不存在或任何物理相被排除')},
    ],
    'quality_gates': [],
    'evaluation': _observation_evaluation,   # 结构上不可达：必需证据恒缺失
    'counter_evidence': None,
}
# 禁止条款：不得由 E∞(δ)、E∞(1/N) 或任何基态总能曲线构造能隙输入；
# 能量外推观测仅作附属关联，不构成能隙证据。


RULE_WIDTH_V1 = {
    'rule_id': 'width_v1',
    'candidate': 'width_scaling',
    'criteria_version': P1F3_CRITERIA_VERSION,
    'applicability': lambda items: {
        'applicable': True,
        'reason': '能力边界规则（分析能力诊断对象，非物理相候选）：'
                  '圆柱宽度依赖分析在当前数据下不可测量'},
    'required_evidence': [
        # 必需但永不可匹配：可比多宽度序列——宽度外推的唯一有效输入
        {'observable': 'cylinder_width_series',
         'analyzer': None, 'category': None, 'min_entries': 1,
         'missing_interpretation': VERDICT_NOT_MEASURABLE,
         'basis': ('圆柱宽度依赖分析与宽度外推需要满足可比性要求的多'
                   '宽度序列（同一系统参数、同一测量协议下多个不同的 '
                   'y 方向宽度）；当前全部可用数据均为 XC 构型（y 方向'
                   '恒 2 胞宽，Ly=2√3·a），改变 N 只对应 x 方向长度'
                   '（胞数）变化，不能当作宽度扫描。not_measurable 表示'
                   '数据能力不足，不表示宽度依赖为零，也不表示宽度方向'
                   '的物理行为不存在')},
    ],
    'quality_gates': [],
    'evaluation': _observation_evaluation,   # 结构上不可达：必需证据恒缺失
    'counter_evidence': None,
}
# 禁止条款：不得把沿圆柱长度方向改变 N 当作改变圆柱宽度。


# ---- 流程层注册入口（P1-G.6：候选级冲突约束，不动冻结引擎） ----
_REGISTERED_RULES = {}   # rule_id -> 规则 dict 副本（流程层注册表）


def _rule_kind(rule):
    """规则类别（流程层注册约束用，结构判定 + 显式声明）：

    - 'capability_boundary'：required_evidence 含
      missing_interpretation='not_measurable' 的 spec（数据能力硬缺口，
      恒 not_measurable）
    - 'phase_judgment'：counter_evidence 非 None（正式反证条款），或
      rule dict 显式声明 rule_kind='phase_judgment'（支持级规则必须
      显式声明，注册入口强制分类）
    - 'observation'：其余（观测级，evaluation 恒 False 且无反证条款）
    """
    if any(spec.get('missing_interpretation') == VERDICT_NOT_MEASURABLE
           for spec in (rule.get('required_evidence') or [])):
        return 'capability_boundary'
    if rule.get('counter_evidence') is not None \
            or rule.get('rule_kind') == 'phase_judgment':
        return 'phase_judgment'
    return 'observation'


def register_candidate_rules(rules, version, replace=None):
    """流程层规则注册入口：候选级冲突检查后更新 AUTHORIZED_RULES。

    约束（P1-G.6 / P1-G.8，防能力边界被同候选相级规则静默覆盖）：
    - 同一候选上 capability_boundary 与 phase_judgment 不得同时注册
      （无论新旧版本、是否同 rule_id）——能力边界表示该候选在当前
      数据能力下不可测量，与相级判定矛盾；冲突 → ValueError，
      不静默并存；显式替换其他规则不能绕过该冲突检查
    - 替换必须显式（P3-1）：已注册的 rule_id 再次注册（含不同版本
      升级），必须将该 rule_id 列入 replace，否则 ValueError——
      不隐式覆盖
    - replace 中的 rule_id 必须已注册（P3-2）：含未注册 ID →
      ValueError 并指出无效 ID，不静默忽略；重复 ID 去重处理，
      无副作用
    - 相级规则必须可分类：phase_judgment 由 counter_evidence 或
      rule_kind 显式声明识别；evaluation 非观测级统一函数且未声明
      rule_kind 的规则无法确认类别 → ValueError（不猜测）
    - 原子性：全部验证（输入 / replace 存在性 / 重复 ID / 规则分类 /
      候选级冲突）在修改注册表之前完成；任一失败路径注册表零修改
    - 授权仍由 AUTHORIZED_RULES 双绑定执行（本入口只做注册约束，
      不改 _execute_rule / _evaluate_candidate 的冻结语义）

    注意（如实声明）：run_candidate_rules 是通用引擎入口，调用方可
    显式传入 authorized 与任意 rules 列表绕过本注册约束——流程层
    约束覆盖注册路径；引擎路径的候选级冲突检查不在本阶段实现。
    """
    replace = set(replace or [])          # 重复 ID 去重（行为稳定）
    new_by_id = {r['rule_id']: dict(r) for r in rules}
    if len(new_by_id) != len(rules):
        raise ValueError('本次注册存在重复 rule_id，拒绝注册')
    # ---- P3-2：replace 目标必须已注册（先于任何修改） ----
    invalid_replace = sorted(rid for rid in replace
                             if rid not in _REGISTERED_RULES)
    if invalid_replace:
        raise ValueError(
            f"replace 含未注册的规则 ID {invalid_replace}"
            '（幽灵替换目标，拒绝注册——注册表未修改）')
    # ---- P3-1：已注册 rule_id 再注册必须显式列入 replace ----
    implicit = sorted(rid for rid in new_by_id
                      if rid in _REGISTERED_RULES and rid not in replace)
    if implicit:
        raise ValueError(
            f"规则 {implicit} 已注册：同 rule_id 再注册（含版本升级）"
            '必须显式声明 replace=[该 rule_id]，禁止隐式覆盖'
            '（注册表未修改）')
    kinds = {}
    for rule in new_by_id.values():
        kind = _rule_kind(rule)
        if kind == 'observation' \
                and rule.get('evaluation') is not _observation_evaluation:
            raise ValueError(
                f"规则 {rule['rule_id']!r} 无法确认类别：evaluation 不是"
                '观测级统一函数且未声明 rule_kind / 无反证条款——'
                '相级规则必须显式声明 rule_kind=\'phase_judgment\'')
        kinds[rule['rule_id']] = kind
    # 与已注册规则逐候选冲突检查（不按 rule_id 判断冲突；
    # replace 中的 ID 是被替换对象，不参与冲突）
    for rule in new_by_id.values():
        for rid, reg in _REGISTERED_RULES.items():
            if reg['candidate'] != rule['candidate']:
                continue
            if rid in replace:
                continue
            if {kinds[rule['rule_id']], _rule_kind(reg)} \
                    == {'capability_boundary', 'phase_judgment'}:
                raise ValueError(
                    f"候选 {rule['candidate']!r} 上能力边界规则 "
                    f"{rid!r} 与相级判定规则 {rule['rule_id']!r} 冲突；"
                    '能力边界未解决前不得注册相级判定（如需替换，'
                    '显式传入 replace=[被替换 rule_id]）')
    # ---- 全部验证通过后统一修改 ----
    for rid in list(replace):
        _REGISTERED_RULES.pop(rid, None)
        AUTHORIZED_RULES.pop(rid, None)
    for rid, rule in new_by_id.items():
        _REGISTERED_RULES[rid] = dict(rule)
        AUTHORIZED_RULES[rid] = version
    return list(new_by_id.keys())


# ---- 注册（观测级 + 能力边界；无相级判定规则） ----
# sc_lro_v1 仍暂缓：不定义、不注册、不授权。
P1F3_RULES = (RULE_SC_ASSOC_V1, RULE_CDW_V1, RULE_SDW_V1,
              RULE_DID_PHASE_V1, RULE_PDW_V1, RULE_STRIPE_V1)
register_candidate_rules(P1F3_RULES, P1F3_CRITERIA_VERSION)

# P1-G.28：Gap/Width 能力边界规则（独立元组，独立注册；不并入 P1F3_RULES）。
# 注册只更新授权表；候选 verdict 仅在调用方显式把 GAPW_RULES 传入
# run_candidate_rules 时产生——注册不等于生产管线接入这些诊断。
GAPW_RULES = (RULE_GAP_V1, RULE_WIDTH_V1)
register_candidate_rules(GAPW_RULES, P1F3_CRITERIA_VERSION)
