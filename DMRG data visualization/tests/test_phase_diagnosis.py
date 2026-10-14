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
    PhaseDiagnosisReport, collect_evidence, audit_comparability,
    CONCLUSION_EVIDENCE_COLLECTED, CONCLUSION_INCONCLUSIVE,
    ANALYZER_SPIN_SF, ANALYZER_CHARGE_SF, ANALYZER_PEAK,
    CMP_CONSISTENT, CMP_INCONSISTENT, CMP_UNKNOWN, CMP_NOT_APPLICABLE)

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
# E. P2 可比性审计（audit_comparability）
# ==========================================================================
print('===== E. P2 可比性审计 =====')


def make_sf_report(analyzer, sites=None, qgrid=None, status=STATUS_OK,
                   label='sf'):
    """构造带 payload 的 SF 估计报告（payload 为 sf dict 简化版，
    挂载到 AnalysisReport.payload 供审计函数读取）"""
    payload = {}
    if sites is not None:
        payload['diagonal_source_sites'] = sites
    if qgrid is not None:
        payload['q_grid'] = qgrid
    rep = make_report(OBSERVABLE_CHARGE_STRUCTURE_FACTOR
                      if analyzer == ANALYZER_CHARGE_SF
                      else OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                      analyzer=analyzer, status=status,
                      data_restricted=True)
    rep.payload = payload
    return rep, payload


Q1 = {'n_qx': 11, 'n_qy': 4, 'qx_min': -1.047, 'qx_max': 1.047,
      'qy_min': 0.0, 'qy_max': 5.441}
Q2 = {'n_qx': 21, 'n_qy': 4, 'qx_min': -1.047, 'qx_max': 1.047,
      'qy_min': 0.0, 'qy_max': 5.441}

r_c1, p_c1 = make_sf_report(ANALYZER_CHARGE_SF, sites=[49, 50], qgrid=Q1)
r_c2, p_c2 = make_sf_report(ANALYZER_CHARGE_SF, sites=[49, 50], qgrid=Q1)
r_c3, p_c3 = make_sf_report(ANALYZER_CHARGE_SF, sites=[97, 98], qgrid=Q2)
r_c4, p_c4 = make_sf_report(ANALYZER_CHARGE_SF, sites=[50, 49], qgrid=Q1)
r_s1, p_s1 = make_sf_report(ANALYZER_SPIN_SF, qgrid=Q1)
r_p1, _ = make_sf_report(ANALYZER_PEAK, qgrid=None)

# E1. 源点集合相同 → consistent（r_c1 vs r_c2）
audit_e1 = audit_comparability(collect_evidence([r_c1, r_c2],
                                                labels=['a', 'b']))
pairs_e1 = audit_e1.consistency['source_set']['pairs']
check('E', 'E1 源点集合相同 → consistent（集合比较）',
      any(p['status'] == CMP_CONSISTENT and p['entries'] == [0, 1]
          for p in pairs_e1),
      [(p['status'], p['entries']) for p in pairs_e1])

# E2. 源点集合不同 → inconsistent
audit_e2 = audit_comparability(collect_evidence([r_c1, r_c3],
                                                labels=['a', 'b']))
check('E', 'E2 源点集合不同 → inconsistent',
      any(p['status'] == CMP_INCONSISTENT
          for p in audit_e2.consistency['source_set']['pairs']))

# E3. 顺序不同同集合 → consistent（r_c1 vs r_c4：49,50 vs 50,49）
audit_e3 = audit_comparability(collect_evidence([r_c1, r_c4],
                                                labels=['a', 'b']))
check('E', 'E3 集合相同但顺序不同 → consistent',
      any(p['status'] == CMP_CONSISTENT
          for p in audit_e3.consistency['source_set']['pairs']))

# E4. 覆盖率相同但源点不同 → inconsistent（用 diagonal_source_sites 区分）
audit_e4 = audit_comparability(collect_evidence([r_c2, r_c3],
                                                labels=['a', 'b']))
check('E', 'E4 源点不同（覆盖率同量级亦不同集合）→ inconsistent',
      any(p['status'] == CMP_INCONSISTENT
          for p in audit_e4.consistency['source_set']['pairs']))

# E5. q_grid 摘要相同 → consistent 且 comparison_level=summary
audit_e5 = audit_comparability(collect_evidence([r_c1, r_c2],
                                                labels=['a', 'b']))
check('E', 'E5 q_grid 摘要相同 → consistent + comparison_level=summary',
      any(p['status'] == CMP_CONSISTENT
          and p['comparison_level'] == 'summary'
          and '完整网格坐标未验证' in p['reason']
          for p in audit_e5.consistency['q_grid']['pairs']))

# E6. q_grid 摘要不同 → inconsistent
audit_e6 = audit_comparability(collect_evidence([r_c1, r_c3],
                                                labels=['a', 'b']))
check('E', 'E6 q_grid 摘要不同 → inconsistent（差异键记录）',
      any(p['status'] == CMP_INCONSISTENT and 'n_qx' in p['reason']
          for p in audit_e6.consistency['q_grid']['pairs']))

# E7. 摘要相同但完整坐标不同：第一版不声称完整网格一致（reason 明示）
r_c5, p_c5 = make_sf_report(ANALYZER_CHARGE_SF, sites=[49, 50], qgrid=Q1)
p_c5['q_grid_same_summary_but_different_coords'] = True   # 模拟内部坐标不同
audit_e7 = audit_comparability(collect_evidence([r_c1, r_c5],
                                                labels=['a', 'b']))
check('E', 'E7 摘要级一致不声称完整网格一致（comparison_level=summary 明示）',
      any(p['status'] == CMP_CONSISTENT and p['comparison_level'] == 'summary'
          for p in audit_e7.consistency['q_grid']['pairs']))

# E8. 自旋 SF 源点集合缺失 → unknown（不推测）
audit_e8 = audit_comparability(collect_evidence([r_s1, r_s1],
                                                labels=['a', 'b']))
check('E', 'E8 自旋 SF 源点集合 → unknown（不推测源点）',
      any(p['status'] == CMP_UNKNOWN and '不推测源点' in p['reason']
          for p in audit_e8.consistency['source_set']['pairs']))

# E9. 衰减/能量链路 → not_applicable
r_dec = make_report(OBSERVABLE_SPIN_CORRELATION, 'SpinCorrDecayAnalyzer',
                    classification='power_law')
audit_e9 = audit_comparability(collect_evidence([r_dec],
                                                labels=['decay']))
check('E', 'E9 衰减链路 source_set/q_grid → not_applicable',
      any(p['status'] == CMP_NOT_APPLICABLE
          for p in audit_e9.consistency['source_set']['pairs'])
      and any(p['status'] == CMP_NOT_APPLICABLE
              for p in audit_e9.consistency['q_grid']['pairs']))

# E10. failed/None 报告参与 → unknown 且保留原 status
r_fail_sf, _ = make_sf_report(ANALYZER_CHARGE_SF, sites=[49, 50], qgrid=Q1,
                              status=STATUS_FAILED)
audit_e10 = audit_comparability(collect_evidence([r_c1, r_fail_sf],
                                                 labels=['ok', 'fail']))
check('E', 'E10 failed 条目 → 一致断言 unknown（原状态保留）',
      any(p['status'] == CMP_UNKNOWN and 'status=failed' in p['reason']
          for p in audit_e10.consistency['q_grid']['pairs']))

# E11. 空/不完整 q_grid → unknown
r_c6, p_c6 = make_sf_report(ANALYZER_CHARGE_SF, sites=[49, 50],
                            qgrid={'n_qx': 0, 'n_qy': 0, 'qx_min': None,
                                   'qx_max': None, 'qy_min': None, 'qy_max': None})
audit_e11 = audit_comparability(collect_evidence([r_c1, r_c6],
                                                 labels=['a', 'b']))
check('E', 'E11 空 q_grid → unknown（不判 consistent）',
      any(p['status'] == CMP_UNKNOWN
          for p in audit_e11.consistency['q_grid']['pairs']))

# E12. 单条报告 → 无可比较对象（unknown，不判 consistent）
audit_e12 = audit_comparability(collect_evidence([r_c1], labels=['only']))
check('E', 'E12 单条报告 → 无可比较对象（unknown）',
      any(p['status'] == CMP_UNKNOWN and '无可比较对象' in p['reason']
          for p in audit_e12.consistency['q_grid']['pairs']))

# E13. 多报告逐对全覆盖（3 条 → 3 对，无遗漏）
audit_e13 = audit_comparability(collect_evidence([r_c1, r_c2, r_c3],
                                                 labels=['a', 'b', 'c']))
n_pairs = len([p for p in audit_e13.consistency['q_grid']['pairs']
               if len(p['entries']) == 2])
check('E', 'E13 三条报告逐对全覆盖（3 对，无遗漏组合）', n_pairs == 3,
      f'{n_pairs} 对')

# E14. 审计前后原始报告与 payload 不变
snap_payload = dict(p_c1) if p_c1 else None
snap_cls = r_c1.classification
audit_comparability(collect_evidence([r_c1, r_c2], labels=['a', 'b']))
check('E', 'E14 审计不修改原始报告与 payload',
      r_c1.classification == snap_cls
      and r_c1.payload is not None
      and all((r_c1.payload or {}).get(k) == snap_payload.get(k)
              for k in snap_payload))

# E15. report_refs/entries/labels 索引顺序保持对应
audit_e15 = audit_comparability(collect_evidence([r_c1, r_c2, r_c3],
                                                 labels=['x', 'y', 'z']))
pr = audit_e15
check('E', 'E15 审计后 report_refs/entries/labels 顺序对应不变',
      len(pr.report_refs) == 3 and pr.report_refs[0] is r_c1
      and pr.labels == ['x', 'y', 'z']
      and pr.entries[2]['label'] == 'z')

# E16. 严格 JSON 序列化含 consistency（无 NaN 字面量）
tmp_e = tempfile.mkdtemp(prefix='phase_p2_')
json_e = os.path.join(tmp_e, 'report.json')
audit_e5.to_json(json_e)
raw_e = open(json_e, encoding='utf-8').read()
back_e = json.loads(raw_e)
check('E', 'E16 JSON：consistency 可读回、无 NaN/Infinity',
      'NaN' not in raw_e and 'Infinity' not in raw_e
      and 'consistency' in back_e
      and back_e['consistency']['q_grid']['pairs'])

# E17. 幂等性：重复调用 audit_comparability
pr_idem = collect_evidence([r_c1, r_c2], labels=['a', 'b'])
pr_idem.rule_log.append('custom_note: 预先存在的非审计日志')
audit_comparability(pr_idem)
n_after1 = len([l for l in pr_idem.rule_log
                if l.startswith('audit_comparability:')])
consistency_1 = pr_idem.consistency
audit_comparability(pr_idem)
n_after2 = len([l for l in pr_idem.rule_log
                if l.startswith('audit_comparability:')])
consistency_2 = pr_idem.consistency
audit_comparability(pr_idem)
n_after3 = len([l for l in pr_idem.rule_log
                if l.startswith('audit_comparability:')])
check('E', 'E17a 连续两次调用：consistency 一致',
      consistency_1 == consistency_2)
check('E', 'E17b 重复调用不累积审计日志（前缀条数稳定为 3）',
      n_after1 == 3 and n_after2 == 3 and n_after3 == 3,
      f'{n_after1}/{n_after2}/{n_after3}')
check('E', 'E17c 三次调用日志数量稳定',
      len([l for l in pr_idem.rule_log
           if l.startswith('audit_comparability:')]) == 3)
check('E', 'E17d 非本函数日志保留且顺序不变（custom_note 恰一条、位于 collect_evidence 日志之后）',
      [l for l in pr_idem.rule_log if l.startswith('custom_note')]
      == ['custom_note: 预先存在的非审计日志']
      and pr_idem.rule_log[3] == 'custom_note: 预先存在的非审计日志',
      pr_idem.rule_log)
check('E', 'E17e 原始报告与 payload 未被修改（幂等调用后）',
      r_c1.classification is None
      and (r_c1.payload or {}).get('diagonal_source_sites') == [49, 50])

# 组级汇总：存在不一致时组汇总不得为 consistent
check('E', '组级汇总：含 inconsistent 时组汇总为 inconsistent（不汇总为 consistent）',
      audit_e6.consistency['q_grid']['group_summary'].get(ANALYZER_CHARGE_SF)
      == CMP_INCONSISTENT,
      audit_e6.consistency['q_grid']['group_summary'])

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
