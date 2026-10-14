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

import numpy as np

from .Analysis_Contract import (
    AnalysisReport, STATUS_FAILED, _convert)

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

    def to_dict(self) -> dict:
        """紧凑 dict：entries/status_summary/conclusion/rule_log/labels。
        report_refs 默认不序列化（引用完整性由内存保证；需要完整报告
        请对各 AnalysisReport 单独调用 to_json）。非有限浮点转 null。"""
        return {
            'entries': _convert(self.entries),
            'status_summary': _convert(self.status_summary),
            'conclusion': self.conclusion,
            'rule_log': _convert(self.rule_log),
            'labels': _convert(self.labels),
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
