# -*- coding: utf-8 -*-
"""
Phase_Diagnosis P1-D.2 通用规则引擎定向测试。

所有测试规则均为**合成规则**（仅验证软件执行逻辑，不代表任何真实
物理判据）；AUTHORIZED_RULES 保持空表，授权经 run_candidate_rules
的 authorized 参数显式注入。

覆盖（P1-D.1.3 冻结语义）：
- 规则级六态 / 候选级九态 / 总体 conclusion 七值
- 授权与版本校验（rule_rejected）
- 适用性（not_applicable）与证据缺失（evidence_missing + rule_level_
  state=None）严格区分；证据缺失不伪装 gate_failed
- missing_interpretation 保守默认 data_missing；not_measurable 无
  basis 降级；basis 齐备按优先级（not_measurable > analysis_
  unavailable > data_missing）
- 软件执行异常（execution_status=exception）不贡献支持/反证；全部
  异常 → analysis_unavailable；部分异常不覆盖有效判定
- 支持/未满足/正式反证/冲突组合（executed_unsatisfied 不构成反证）
- 多规则多候选聚合与 conclusion 七值关键边界
- 序列化、旧接口兼容与输入不被修改

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_phase_diagnosis_rules.py
"""
import json
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK, STATUS_FAILED)
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, build_evidence_items, run_candidate_rules,
    EVIDENCE_OBSERVATION,
    RULE_EXECUTED_SUPPORTED, RULE_EXECUTED_UNSATISFIED,
    RULE_COUNTER_TRIGGERED, RULE_NOT_APPLICABLE, RULE_REJECTED,
    RULE_GATE_FAILED,
    VERDICT_SUPPORTED, VERDICT_CONTRADICTED, VERDICT_CONFLICTING,
    VERDICT_INSUFFICIENT, VERDICT_QUALITY_FAILED, VERDICT_CRITERIA_UNDEFINED,
    VERDICT_DATA_MISSING, VERDICT_ANALYSIS_UNAVAILABLE, VERDICT_NOT_MEASURABLE,
    CONCLUSION_CONFLICTING, CONCLUSION_MIXED,
    CONCLUSION_PARTIALLY_UNRESOLVED, CONCLUSION_SUPPORTED,
    CONCLUSION_CONTRADICTED_ALL, CONCLUSION_INCONCLUSIVE,
    CONCLUSION_EVIDENCE_COLLECTED)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ---- 合成报告：一条自旋衰减 observation 证据 ----
def spin_decay_rep():
    return AnalysisReport(
        analyzer='SpinCorrDecayAnalyzer', observable='spin_correlation',
        status=STATUS_OK, classification='power_law',
        parameters={'K': 1.5, 'A': 0.1}, evidence='strong',
        data_restricted=None, warnings=[],
        metadata={'r_min': 2.0, 'r_max': 20.0, 'n_points': 18},
        payload=None)


def make_phase():
    return collect_evidence([spin_decay_rep()], labels=['L0'])


# ---- 合成规则构造（非物理判据） ----
def make_rule(rule_id, candidate='SC', version='v1', applicable=True,
              required=None, gates=None, support=True, counter=None,
              raise_stage=None, app_reason=''):
    """合成规则。raise_stage ∈ {None,'applicability','quality_gates',
    'evaluation','counter_evidence'} 时对应回调抛 RuntimeError。"""
    def boom(*args, **kwargs):
        raise RuntimeError(f'合成回调异常 @{raise_stage}')

    def appl(items):
        if raise_stage == 'applicability':
            boom()
        return {'applicable': applicable, 'reason': app_reason}

    def gate(items, payloads):
        if raise_stage == 'quality_gates':
            boom()
        return {'passed': True, 'detail': ''}

    def evl(items, payloads):
        if raise_stage == 'evaluation':
            boom()
        return support

    ctr = None
    if counter is not None:
        def ctr_fn(items, payloads):
            if raise_stage == 'counter_evidence':
                boom()
            return counter
        ctr = {'evaluation': ctr_fn, 'note': '合成反证条款（非物理判据）'}
    if required is None:
        required = [{'observable': 'spin_correlation',
                     'analyzer': 'SpinCorrDecayAnalyzer',
                     'category': EVIDENCE_OBSERVATION, 'min_entries': 1}]
    return {'rule_id': rule_id, 'candidate': candidate,
            'criteria_version': version, 'applicability': appl,
            'required_evidence': required,
            'quality_gates': (gates if gates is not None
                              else [{'name': 'g_ok', 'check': gate}]),
            'evaluation': evl, 'counter_evidence': ctr,
            'result_mapping': None}


# ======================================================================
# A. 授权与拒绝
# ======================================================================
print('===== A. 授权与拒绝 =====')
ph_a1 = make_phase()
run_candidate_rules(ph_a1, [make_rule('r1')], authorized={})
check('A', '未授权规则：rule_rejected（不执行）',
      ph_a1.rule_trace[0]['rule_level_state'] == RULE_REJECTED
      and ph_a1.candidate_verdicts['SC']['verdict']
      == VERDICT_CRITERIA_UNDEFINED,
      str(ph_a1.rule_trace[0]['rule_level_state']))
ph_a2 = make_phase()
run_candidate_rules(ph_a2, [make_rule('r1', version='v2')],
                    authorized={'r1': 'v1'})
check('A', '版本不匹配：rule_rejected + 候选 criteria_undefined',
      ph_a2.rule_trace[0]['rule_level_state'] == RULE_REJECTED
      and ph_a2.candidate_verdicts['SC']['verdict']
      == VERDICT_CRITERIA_UNDEFINED,
      str(ph_a2.rule_trace[0]['reason']))
ph_a3 = make_phase()
run_candidate_rules(ph_a3, [], authorized={})
check('A', '无规则执行：candidate_verdicts 空 + conclusion=evidence_collected',
      ph_a3.candidate_verdicts == {}
      and ph_a3.conclusion == CONCLUSION_EVIDENCE_COLLECTED)

# ======================================================================
# B. 适用性与证据缺失
# ======================================================================
print('===== B. 适用性与证据缺失 =====')
ph_b1 = make_phase()
run_candidate_rules(ph_b1, [make_rule('r1', applicable=False,
                                      app_reason='数据组不满足')],
                    authorized={'r1': 'v1'})
check('B', '全部 not_applicable：候选 criteria_undefined（判定不可执行）',
      ph_b1.rule_trace[0]['rule_level_state'] == RULE_NOT_APPLICABLE
      and ph_b1.candidate_verdicts['SC']['verdict']
      == VERDICT_CRITERIA_UNDEFINED
      and '不适用' in ph_b1.rule_trace[0]['reason'])

# 证据缺失：未声明 interpretation → 保守 data_missing
ph_b2 = make_phase()
run_candidate_rules(ph_b2, [make_rule(
    'r1', required=[{'observable': 'charge_structure_factor',
                     'analyzer': 'ChargeStructureFactorEstimator',
                     'category': EVIDENCE_OBSERVATION, 'min_entries': 1}])],
    authorized={'r1': 'v1'})
tr_b2 = ph_b2.rule_trace[0]
check('B', '规则适用但证据缺失：rule_level_state=None + evidence_missing '
      '非空（不是 not_applicable 也不是 gate_failed）',
      tr_b2['rule_level_state'] is None
      and tr_b2['evidence_missing'] is not None
      and tr_b2['execution_status'] == 'completed',
      str(tr_b2['rule_level_state']))
check('B', 'missing_interpretation 未声明：保守 data_missing',
      tr_b2['evidence_missing'][0]['interpretation'] == 'data_missing'
      and ph_b2.candidate_verdicts['SC']['verdict'] == VERDICT_DATA_MISSING,
      str(tr_b2['evidence_missing'][0]))

# not_measurable 无 basis → 降级
ph_b3 = make_phase()
run_candidate_rules(ph_b3, [make_rule(
    'r1', required=[{'observable': 'charge_structure_factor',
                     'analyzer': 'ChargeStructureFactorEstimator',
                     'category': EVIDENCE_OBSERVATION, 'min_entries': 1,
                     'missing_interpretation': 'not_measurable'}])],
    authorized={'r1': 'v1'})
check('B', 'not_measurable 缺 basis：降级 data_missing',
      ph_b3.candidate_verdicts['SC']['verdict'] == VERDICT_DATA_MISSING
      and '降级' in ph_b3.rule_trace[0]['evidence_missing'][0]['detail'],
      str(ph_b3.rule_trace[0]['evidence_missing'][0]))

# not_measurable 有 basis → 生效；优先级 not_measurable > data_missing
ph_b4 = make_phase()
run_candidate_rules(ph_b4, [make_rule(
    'r1', required=[
        {'observable': 'charge_structure_factor',
         'analyzer': 'ChargeStructureFactorEstimator',
         'category': EVIDENCE_OBSERVATION, 'min_entries': 1},
        {'observable': 'phase_info', 'analyzer': 'ComplexPairingAnalyzer',
         'category': EVIDENCE_OBSERVATION, 'min_entries': 1,
         'missing_interpretation': 'not_measurable',
         'basis': '实值数据无复相位（合成测试依据，非物理判据）'}])],
    authorized={'r1': 'v1'})
check('B', '多缺失解释：主 verdict 取最高优先级 not_measurable（basis 齐备）'
      '且 detail 保留全部缺失',
      ph_b4.candidate_verdicts['SC']['verdict'] == VERDICT_NOT_MEASURABLE
      and len(ph_b4.rule_trace[0]['evidence_missing']) == 2,
      str(ph_b4.candidate_verdicts['SC']['verdict']))

# ======================================================================
# C. 正常判定与组合
# ======================================================================
print('===== C. 正常判定与组合 =====')
ph_c1 = make_phase()
run_candidate_rules(ph_c1, [make_rule('r1', support=True)],
                    authorized={'r1': 'v1'})
check('C', '支持成立无反证：rule=executed_supported、候选=supported、'
      'conclusion=supported',
      ph_c1.rule_trace[0]['rule_level_state'] == RULE_EXECUTED_SUPPORTED
      and ph_c1.candidate_verdicts['SC']['verdict'] == VERDICT_SUPPORTED
      and ph_c1.conclusion == CONCLUSION_SUPPORTED,
      str(ph_c1.candidate_verdicts['SC']['verdict']))

ph_c2 = make_phase()
run_candidate_rules(ph_c2, [make_rule('r1', support=False)],
                    authorized={'r1': 'v1'})
check('C', '全部 executed_unsatisfied：候选 insufficient（不构成反证）、'
      'conclusion=INCONCLUSIVE',
      ph_c2.rule_trace[0]['rule_level_state'] == RULE_EXECUTED_UNSATISFIED
      and ph_c2.candidate_verdicts['SC']['verdict'] == VERDICT_INSUFFICIENT
      and ph_c2.conclusion == CONCLUSION_INCONCLUSIVE,
      str(ph_c2.candidate_verdicts['SC']['verdict']))

ph_c3 = make_phase()
run_candidate_rules(ph_c3, [make_rule('r1', support=False, counter=True)],
                    authorized={'r1': 'v1'})
check('C', '正式反证成立且支持不成立：rule=counter_triggered、'
      '候选=contradicted',
      ph_c3.rule_trace[0]['rule_level_state'] == RULE_COUNTER_TRIGGERED
      and ph_c3.candidate_verdicts['SC']['verdict'] == VERDICT_CONTRADICTED,
      str(ph_c3.rule_trace[0]['rule_level_state']))

ph_c4 = make_phase()
run_candidate_rules(ph_c4, [make_rule('r1', support=True, counter=True)],
                    authorized={'r1': 'v1'})
check('C', '支持∧反证并存：候选 conflicting、conclusion=conflicting',
      ph_c4.candidate_verdicts['SC']['verdict'] == VERDICT_CONFLICTING
      and ph_c4.conclusion == CONCLUSION_CONFLICTING,
      str(ph_c4.candidate_verdicts['SC']['verdict']))

# 质量门失败
def gate_fail(items, payloads):
    return {'passed': False, 'detail': '合成质量门失败'}


ph_c5 = make_phase()
run_candidate_rules(ph_c5, [make_rule('r1', gates=[
    {'name': 'g_fail', 'check': gate_fail}])], authorized={'r1': 'v1'})
check('C', '质量门明确失败：gate_failed → 候选 quality_failed（无 '
      'evidence_missing 混入）',
      ph_c5.rule_trace[0]['rule_level_state'] == RULE_GATE_FAILED
      and ph_c5.rule_trace[0]['evidence_missing'] is None
      and ph_c5.candidate_verdicts['SC']['verdict'] == VERDICT_QUALITY_FAILED,
      str(ph_c5.rule_trace[0]['reason']))

# ======================================================================
# D. 执行异常
# ======================================================================
print('===== D. 执行异常 =====')
ph_d1 = make_phase()
run_candidate_rules(ph_d1, [make_rule('r1', raise_stage='evaluation')],
                    authorized={'r1': 'v1'})
tr_d1 = ph_d1.rule_trace[0]
check('D', '全部规则执行异常：候选 analysis_unavailable + exception 详情'
      '（rule_level_state=None、非 gate_failed 非反证）',
      tr_d1['execution_status'] == 'exception'
      and tr_d1['rule_level_state'] is None
      and tr_d1['exception']['stage'] == 'evaluation'
      and tr_d1['evidence_missing'] is None
      and ph_d1.candidate_verdicts['SC']['verdict']
      == VERDICT_ANALYSIS_UNAVAILABLE
      and ph_d1.candidate_verdicts['SC']['detail']['execution_errors'],
      str(tr_d1['exception']))

ph_d2 = make_phase()
run_candidate_rules(ph_d2, [make_rule('r1', raise_stage='evaluation'),
                            make_rule('r2', support=True)],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('D', '部分异常+有效支持：支持不被覆盖，异常保留 detail',
      ph_d2.candidate_verdicts['SC']['verdict'] == VERDICT_SUPPORTED
      and len(ph_d2.candidate_verdicts['SC']['detail']['execution_errors']) == 1,
      str(ph_d2.candidate_verdicts['SC']['verdict']))

ph_d3 = make_phase()
run_candidate_rules(ph_d3, [make_rule('r1', raise_stage='applicability')],
                    authorized={'r1': 'v1'})
check('D', 'applicability 回调异常：exception 记录 stage=applicability',
      ph_d3.rule_trace[0]['execution_status'] == 'exception'
      and ph_d3.rule_trace[0]['exception']['stage'] == 'applicability',
      str(ph_d3.rule_trace[0]['exception']))

# ======================================================================
# E. 证据缺失与有效判定并存
# ======================================================================
print('===== E. 缺失与有效判定并存 =====')
ph_e1 = make_phase()
run_candidate_rules(ph_e1, [
    make_rule('r1', required=[{'observable': 'charge_structure_factor',
                               'analyzer': 'ChargeStructureFactorEstimator',
                               'category': EVIDENCE_OBSERVATION,
                               'min_entries': 1}]),
    make_rule('r2', support=True)], authorized={'r1': 'v1', 'r2': 'v1'})
check('E', '证据缺失不覆盖其他规则有效判定：候选 supported；缺失解释'
      '保留 verdict_detail',
      ph_e1.candidate_verdicts['SC']['verdict'] == VERDICT_SUPPORTED
      and len(ph_e1.candidate_verdicts['SC']['detail']['evidence_missing']) == 1,
      str(ph_e1.candidate_verdicts['SC']['verdict']))

# ======================================================================
# F. 多候选聚合与 conclusion 七值边界
# ======================================================================
print('===== F. 多候选聚合与 conclusion =====')

ph_f1 = make_phase()
run_candidate_rules(ph_f1, [make_rule('r1', candidate='SC', support=True),
                            make_rule('r2', candidate='CDW',
                                      support=False, counter=True)],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('F', '一支持一否定：conclusion=mixed',
      ph_f1.conclusion == CONCLUSION_MIXED
      and ph_f1.candidate_verdicts['SC']['verdict'] == VERDICT_SUPPORTED
      and ph_f1.candidate_verdicts['CDW']['verdict'] == VERDICT_CONTRADICTED,
      ph_f1.conclusion)

ph_f2 = make_phase()
run_candidate_rules(ph_f2, [make_rule('r1', candidate='SC', support=True),
                            make_rule('r2', candidate='CDW',
                                      applicable=False)],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('F', '一支持一未决（criteria_undefined）：conclusion=partially_unresolved',
      ph_f2.conclusion == CONCLUSION_PARTIALLY_UNRESOLVED,
      ph_f2.conclusion)

ph_f3 = make_phase()
run_candidate_rules(ph_f3, [make_rule('r1', candidate='SC',
                                      support=False, counter=True),
                            make_rule('r2', candidate='CDW',
                                      support=False, counter=True)],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('F', '全部否定：conclusion=contradicted_all',
      ph_f3.conclusion == CONCLUSION_CONTRADICTED_ALL,
      ph_f3.conclusion)

ph_f4 = make_phase()
run_candidate_rules(ph_f4, [make_rule('r1', candidate='SC', support=True),
                            make_rule('r2', candidate='CDW', support=True)],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('F', '多候选同时支持：conclusion=supported',
      ph_f4.conclusion == CONCLUSION_SUPPORTED, ph_f4.conclusion)

ph_f5 = make_phase()
run_candidate_rules(ph_f5, [make_rule('r1', candidate='SC', support=False),
                            make_rule('r2', candidate='CDW',
                                      raise_stage='evaluation')],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('F', '部分判定+部分异常：conclusion=INCONCLUSIVE（无支持/否定/冲突）',
      ph_f5.conclusion == CONCLUSION_INCONCLUSIVE
      and ph_f5.candidate_verdicts['CDW']['verdict']
      == VERDICT_ANALYSIS_UNAVAILABLE,
      ph_f5.conclusion)

# ======================================================================
# G. 序列化、输入保护、确定性
# ======================================================================
print('===== G. 序列化与输入保护 =====')
ph_g = make_phase()
snapshot_entries = [dict(e) for e in ph_g.entries]
concl_before = ph_g.conclusion
run_candidate_rules(ph_g, [make_rule('r1', support=True)],
                    authorized={'r1': 'v1'})
check('G', '规则执行不改动 entries 内容',
      [dict(e) for e in ph_g.entries] == snapshot_entries)
check('G', '未执行时 conclusion 保持 evidence_collected（旧行为）',
      concl_before == CONCLUSION_EVIDENCE_COLLECTED)
ph_g2 = make_phase()
run_candidate_rules(ph_g2, [make_rule('r1', support=True)],
                    authorized={'r1': 'v1'})
run_candidate_rules(ph_g2, [make_rule('r1', support=True)],
                    authorized={'r1': 'v1'})
check('G', '重复执行：rule_trace 幂等替换（长度不累积）',
      len(ph_g2.rule_trace) == 1)
tmp = tempfile.mkdtemp(prefix='phase_rules_')
json_path = os.path.join(tmp, 'report.json')
ph_g.to_json(json_path)
raw = open(json_path, encoding='utf-8').read()
back = json.loads(raw)
check('G', 'to_json 含新键且严格可解析（无 NaN/Infinity）',
      'NaN' not in raw and 'Infinity' not in raw
      and 'candidate_verdicts' in back and 'rule_trace' in back
      and 'verdict_summary' in back
      and back['candidate_verdicts']['SC']['verdict'] == VERDICT_SUPPORTED)

# ======================================================================
# H. P1-D.2.2：候选清单与混合状态收尾
# ======================================================================
print('===== H. 候选清单与混合状态 =====')
from Physical_Quantity.Phase_Diagnosis import (
    AUTHORIZED_RULES, P1F3_RULES, GAPW_RULES)

# H1 显式候选清单：无规则候选 → criteria_undefined（判定不可执行）
ph_h1 = make_phase()
run_candidate_rules(ph_h1, [], authorized={}, candidates=['SC', 'CDW'])
check('H', '显式候选无规则：两候选均 criteria_undefined（reason 该候选无规则）'
      '、conclusion=INCONCLUSIVE',
      ph_h1.candidate_verdicts['SC']['verdict'] == VERDICT_CRITERIA_UNDEFINED
      and ph_h1.candidate_verdicts['CDW']['verdict']
      == VERDICT_CRITERIA_UNDEFINED
      and '该候选无规则' in ph_h1.candidate_verdicts['SC']['reason']
      and ph_h1.conclusion == CONCLUSION_INCONCLUSIVE,
      str(ph_h1.candidate_verdicts))

# H2 candidates=None 原有行为：候选集合仅来自 rules
ph_h2 = make_phase()
run_candidate_rules(ph_h2, [make_rule('r1', candidate='SC', support=True)],
                    authorized={'r1': 'v1'})
check('H', 'candidates=None：候选集合仅来自 rules（无额外候选）',
      list(ph_h2.candidate_verdicts.keys()) == ['SC']
      and ph_h2.conclusion == CONCLUSION_SUPPORTED,
      str(list(ph_h2.candidate_verdicts.keys())))

# H3 候选清单不赋予授权：清单中的无规则候选不得获 supported
ph_h3 = make_phase()
run_candidate_rules(ph_h3, [], authorized={}, candidates=['SC'])
check('H', '候选清单不赋予规则授权（无规则候选为 criteria_undefined，'
      '绝非 supported）',
      ph_h3.candidate_verdicts['SC']['verdict'] == VERDICT_CRITERIA_UNDEFINED,
      str(ph_h3.candidate_verdicts['SC']['verdict']))

# H4 quality_gates 回调异常
ph_h4 = make_phase()
run_candidate_rules(ph_h4, [make_rule('r1', raise_stage='quality_gates')],
                    authorized={'r1': 'v1'})
check('H', 'quality_gates 回调异常：exception.stage=quality_gates、'
      '候选 analysis_unavailable（不伪装 gate_failed）',
      ph_h4.rule_trace[0]['execution_status'] == 'exception'
      and ph_h4.rule_trace[0]['exception']['stage'] == 'quality_gates'
      and ph_h4.rule_trace[0]['rule_level_state'] is None
      and ph_h4.candidate_verdicts['SC']['verdict']
      == VERDICT_ANALYSIS_UNAVAILABLE,
      str(ph_h4.rule_trace[0]['exception']))

# H5 counter_evidence 回调异常
ph_h5 = make_phase()
run_candidate_rules(ph_h5, [make_rule('r1', counter=True,
                                      raise_stage='counter_evidence')],
                    authorized={'r1': 'v1'})
check('H', 'counter_evidence 回调异常：exception.stage=counter_evidence、'
      '不贡献反证（候选 analysis_unavailable）',
      ph_h5.rule_trace[0]['exception']['stage'] == 'counter_evidence'
      and ph_h5.candidate_verdicts['SC']['verdict']
      == VERDICT_ANALYSIS_UNAVAILABLE
      and ph_h5.rule_trace[0]['counter_evaluation'] is None,
      str(ph_h5.rule_trace[0]['exception']))

# H6 跨规则 executed_unsatisfied + counter_triggered → contradicted
ph_h6 = make_phase()
run_candidate_rules(ph_h6, [make_rule('r1', support=False),
                            make_rule('r2', support=False, counter=True)],
                    authorized={'r1': 'v1', 'r2': 'v1'})
check('H', '跨规则 unsatisfied+反证：候选 contradicted（unsatisfied 不构成'
      '反证，但正式反证成立且无支持）',
      ph_h6.candidate_verdicts['SC']['verdict'] == VERDICT_CONTRADICTED,
      str(ph_h6.candidate_verdicts['SC']['verdict']))

# H7 多规则版本并存：同 rule_id 不同 version，仅授权版本执行
ph_h7 = make_phase()
run_candidate_rules(ph_h7, [make_rule('r1', version='v1', support=True),
                            make_rule('r1', version='v2', support=False,
                                      counter=True)],
                    authorized={'r1': 'v1'})
states7 = [t['rule_level_state'] for t in ph_h7.rule_trace]
check('H', '多版本并存：v1 执行（executed_supported）、v2 拒绝（rule_rejected）'
      '、候选 supported',
      sorted(states7) == [RULE_EXECUTED_SUPPORTED, RULE_REJECTED]
      and ph_h7.candidate_verdicts['SC']['verdict'] == VERDICT_SUPPORTED,
      str(states7))

# H8 混合：gate_failed + exception，无有效判定 → analysis_unavailable，
#    两类 trace 均保留
ph_h8 = make_phase()
run_candidate_rules(ph_h8, [make_rule('r1', gates=[
    {'name': 'g_fail', 'check': gate_fail}]),
    make_rule('r2', raise_stage='evaluation')],
    authorized={'r1': 'v1', 'r2': 'v1'})
d8 = ph_h8.candidate_verdicts['SC']
check('H', '混合 gate+exception 无有效判定：analysis_unavailable + '
      'gate_failures 与 execution_errors 各 1（不互相冒充）',
      d8['verdict'] == VERDICT_ANALYSIS_UNAVAILABLE
      and len(d8['detail']['gate_failures']) == 1
      and len(d8['detail']['execution_errors']) == 1,
      str(d8['verdict']))

# H9 混合：gate_failed + evidence_missing，无有效判定无异常 →
#    quality_failed，缺失细节全部保留
ph_h9 = make_phase()
run_candidate_rules(ph_h9, [make_rule('r1', gates=[
    {'name': 'g_fail', 'check': gate_fail}]),
    make_rule('r2', required=[{'observable': 'charge_structure_factor',
                               'analyzer': 'ChargeStructureFactorEstimator',
                               'category': EVIDENCE_OBSERVATION,
                               'min_entries': 1}])],
    authorized={'r1': 'v1', 'r2': 'v1'})
d9 = ph_h9.candidate_verdicts['SC']
check('H', '混合 gate+missing 无有效判定无异常：quality_failed + '
      'evidence_missing 全量保留',
      d9['verdict'] == VERDICT_QUALITY_FAILED
      and len(d9['detail']['evidence_missing']) == 1
      and len(d9['detail']['execution_errors']) == 0,
      str(d9['verdict']))

# H10 生产默认授权：恰为 8 条（观测级 ×3 + 能力边界 ×3 + P1-G.28
# Gap/Width 能力边界 ×2），全部无相级支持/反证条款（不注册真实物理判据）
check('H', 'AUTHORIZED_RULES 恰为 8 条（含 stripe/gap/width 能力边界）'
      '且无相级判定规则',
      set(AUTHORIZED_RULES)
      == {'sc_assoc_v1', 'cdw_v1', 'sdw_v1', 'did_phase_v1', 'pdw_v1',
          'stripe_v1', 'gap_v1', 'width_v1'}
      and all(r.get('counter_evidence') is None
              for r in P1F3_RULES + GAPW_RULES))

# H11 rule_trace 关键字段完整性
tr_h11 = ph_h7.rule_trace[0]
check('H', 'rule_trace 字段完整（11 键）',
      all(k in tr_h11 for k in
          ('rule_id', 'candidate', 'criteria_version', 'rule_level_state',
           'execution_status', 'evidence_missing', 'exception',
           'entry_indices', 'support_evaluation', 'counter_evaluation',
           'reason')),
      str(sorted(tr_h11.keys())))

# ======================================================================
# 汇总
# ======================================================================
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
