# -*- coding: utf-8 -*-
"""
PhaseDiagnosis P1 测试：多观测量证据汇总契约（纯汇总，无物理判据）

合成 AnalysisReport 覆盖：
- 多观测量正常汇总
- ok / inconclusive / failed 状态并存与计数
- classification/evidence/data_restricted 为 None 的原样保留
- 标签一一对应及非法输入（长度不匹配 / 类型错误 / None 标签）
- 空输入行为
- JSON 无 NaN/Infinity 字面量、无 payload/引用外泄
- 原始报告关键字段与汇总前一致
- evidence_collected 不被解释为物理结论（conclusion 仅两个枚举 + rule_log 文本）
- entries 键集固定、无路径、无置信度/概率字段

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_phase_diagnosis.py
"""
import sys
import os
import json
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK, STATUS_INCONCLUSIVE, STATUS_FAILED,
    OBSERVABLE_SPIN_CORRELATION, OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR, OBSERVABLE_ENERGY)
from Physical_Quantity.Phase_Diagnosis import (
    PhaseDiagnosisReport, collect_evidence,
    CONCLUSION_EVIDENCE_COLLECTED, CONCLUSION_INCONCLUSIVE)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def make_report(observable, analyzer='TestAnalyzer', status=STATUS_OK,
                classification=None, evidence=None, data_restricted=None,
                warnings=None):
    return AnalysisReport(
        analyzer=analyzer, observable=observable, status=status,
        classification=classification, evidence=evidence,
        data_restricted=data_restricted,
        warnings=warnings if warnings is not None else [])


# ==========================================================================
# A. 多观测量正常汇总与状态并存
# ==========================================================================
print('===== A. 多观测量汇总 =====')
r_ok = make_report(OBSERVABLE_SPIN_CORRELATION, 'SpinCorrDecayAnalyzer',
                   classification='power_law', evidence='moderate')
r_inc = make_report(OBSERVABLE_SPIN_STRUCTURE_FACTOR, 'SpinStructureFactorPeakAnalyzer',
                    status=STATUS_INCONCLUSIVE, data_restricted=True,
                    warnings=['信号全无效'])
r_fail = None   # 失败的输入报告（契约：保留为 failed 条目）
r_chg = make_report(OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                    'ChargeStructureFactorEstimator',
                    data_restricted=True, warnings=['2 个预期关联对缺失（受限估计）'])

rep = collect_evidence([r_ok, r_inc, r_fail, r_chg],
                       labels=['spin d1_100', 'spin_sf', 'energy_failed', 'charge_sf'])
check('A', 'entries 含 4 条且键集固定',
      len(rep.entries) == 4
      and all(set(e.keys()) == {'observable', 'analyzer', 'status', 'classification',
                                'evidence', 'data_restricted', 'warnings', 'label'}
              for e in rep.entries))
check('A', 'status_summary 计数正确（ok=2, inconclusive=1, failed=1）',
      rep.status_summary == {STATUS_OK: 2, STATUS_INCONCLUSIVE: 1, STATUS_FAILED: 1},
      rep.status_summary)
check('A', 'failed 条目保留且 observable/analyzer 为 None（不伪造来源）',
      rep.entries[2]['status'] == STATUS_FAILED
      and rep.entries[2]['observable'] is None
      and rep.entries[2]['analyzer'] is None)
check('A', 'INCONCLUSIVE 状态不被转换为成功或失败',
      rep.entries[1]['status'] == STATUS_INCONCLUSIVE
      and rep.entries[1]['classification'] is None
      and rep.entries[1]['data_restricted'] is True)
check('A', 'classification/evidence/data_restricted 独立原样记录',
      rep.entries[0]['classification'] == 'power_law'
      and rep.entries[0]['evidence'] == 'moderate'
      and rep.entries[0]['data_restricted'] is None
      and rep.entries[3]['data_restricted'] is True)

# ==========================================================================
# B. 标签契约与非法输入
# ==========================================================================
print('===== B. 标签与非法输入 =====')
check('B', '标签一一对应（与 entries 同序）',
      rep.labels == ['spin d1_100', 'spin_sf', 'energy_failed', 'charge_sf'])
check('B', 'None 标签允许（未提供标签语义）',
      collect_evidence([r_ok], labels=[None]).labels == [None])
try:
    collect_evidence([r_ok], labels=['a', 'b'])
    check('B', '标签长度不匹配 → ValueError', False, '未抛异常')
except ValueError:
    check('B', '标签长度不匹配 → ValueError', True)
try:
    collect_evidence([r_ok], labels=[42])
    check('B', '标签非 str/None → TypeError', False, '未抛异常')
except TypeError:
    check('B', '标签非 str/None → TypeError', True)
try:
    collect_evidence([{'not': 'a report'}], labels=['x'])
    check('B', '非 AnalysisReport 元素 → TypeError', False, '未抛异常')
except TypeError:
    check('B', '非 AnalysisReport 元素 → TypeError', True)

# ==========================================================================
# C. 空输入与结论枚举
# ==========================================================================
print('===== C. 空输入与结论 =====')
rep_empty = collect_evidence([])
check('C', '空输入：entries 为空、conclusion=INCONCLUSIVE',
      rep_empty.entries == [] and rep_empty.conclusion == CONCLUSION_INCONCLUSIVE
      and rep_empty.status_summary == {})
rep_allfail = collect_evidence([None, None], labels=['a', 'b'])
check('C', '全部 failed：conclusion=INCONCLUSIVE（不伪造证据）',
      rep_allfail.conclusion == CONCLUSION_INCONCLUSIVE)
check('C', '存在非 failed 条目：conclusion=evidence_collected（仅汇总完成）',
      rep.conclusion == CONCLUSION_EVIDENCE_COLLECTED)
check('C', 'rule_log 声明 evidence_collected 不代表物理证据充分',
      any('不代表物理证据充分' in line for line in rep.rule_log))
check('C', 'conclusion 只取两个枚举值（无相态名/置信度）',
      rep.conclusion in (CONCLUSION_EVIDENCE_COLLECTED, CONCLUSION_INCONCLUSIVE)
      and 'conf' not in rep.to_dict())

# ==========================================================================
# D. 序列化与原始报告保护
# ==========================================================================
print('===== D. 序列化与报告保护 =====')
tmp = tempfile.mkdtemp(prefix='phase_diag_')
json_path = os.path.join(tmp, 'report.json')
rep.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
back = json.loads(raw)
check('D', 'JSON：无 NaN/Infinity 字面量、无 payload/引用键',
      'NaN' not in raw and 'Infinity' not in raw
      and '"payload"' not in raw and '"report_refs"' not in raw)
check('D', 'JSON 读回：entries/status_summary/conclusion 一致',
      back['entries'][0]['classification'] == 'power_law'
      and back['status_summary'] == {'ok': 2, 'inconclusive': 1, 'failed': 1}
      and back['conclusion'] == CONCLUSION_EVIDENCE_COLLECTED)
check('D', '原始报告引用内存保留（可追溯），字段未被修改',
      rep.report_refs[0] is r_ok and rep.report_refs[3] is r_chg
      and r_ok.classification == 'power_law'
      and r_inc.warnings == ['信号全无效'])
check('D', '输出不含路径与置信度/概率/误差字段',
      'C:' not in raw and 'confidence' not in raw and 'probability' not in raw
      and 'error' not in raw)

# ==========================================================================
# 汇总
# ==========================================================================
print()
print('=' * 60)
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in RESULTS:
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    print(f'{cat:4}: {n_pass}/{len(RESULTS[cat])} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)
