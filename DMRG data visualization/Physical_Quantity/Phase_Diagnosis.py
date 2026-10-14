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
    AnalysisReport, STATUS_FAILED, _convert)

# 结构因子估计报告的精确识别标识（禁止宽泛字符串包含判断）
ANALYZER_SPIN_SF = 'SpinStructureFactorEstimator'
ANALYZER_CHARGE_SF = 'ChargeStructureFactorEstimator'
ANALYZER_PEAK = 'SpinStructureFactorPeakAnalyzer'
_SF_ESTIMATOR_ANALYZERS = (ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF)

# 可比性检查状态（四态，语义见 audit_comparability docstring）
CMP_CONSISTENT = 'consistent'
CMP_INCONSISTENT = 'inconsistent'
CMP_UNKNOWN = 'unknown'
CMP_NOT_APPLICABLE = 'not_applicable'

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

    def to_dict(self) -> dict:
        """紧凑 dict：entries/status_summary/conclusion/rule_log/labels/
        consistency。report_refs 默认不序列化（引用完整性由内存保证；
        需要完整报告请对各 AnalysisReport 单独调用 to_json）。
        非有限浮点转 null。"""
        return {
            'entries': _convert(self.entries),
            'status_summary': _convert(self.status_summary),
            'conclusion': self.conclusion,
            'rule_log': _convert(self.rule_log),
            'labels': _convert(self.labels),
            'consistency': _convert(self.consistency),
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
    known_sf = {ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF}

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
    sf_idxs = spin_idxs + charge_idxs
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
