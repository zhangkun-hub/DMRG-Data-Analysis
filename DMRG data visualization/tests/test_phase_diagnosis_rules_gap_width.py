# -*- coding: utf-8 -*-
"""
Gap/Width 能力边界规则测试（P1-G.28）。

覆盖：
- A. 注册与授权：gap_v1/width_v1 经注册入口进入授权表与注册表，
  规则类别为 capability_boundary，criteria_version 双绑定
- B. 显式执行 GAPW_RULES：空报告下 gap_scaling/width_scaling 恒
  not_measurable（evidence_missing 带 interpretation + basis），
  结构上不可能 supported/contradicted
- C. 含能量外推观测的报告：附属观测仅作关联（entry_indices），
  不改变能隙输入缺失的事实（仍 not_measurable）；宽度同理
- D. 既有注册守卫回归：同候选相级规则注册被拒、P3-1/P3-2、
  版本失配执行被拒、守卫失败零修改
- E. 仅执行 P1F3_RULES 时历史候选 verdict 与 conclusion 不变；
  sc_lro_v1 仍冻结；状态枚举不变

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_phase_diagnosis_rules_gap_width.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    from_energy_extrapolation, OBSERVABLE_ENERGY, ANALYZER_ENERGY)
from Physical_Quantity.Phase_Diagnosis import (
    AUTHORIZED_RULES, _REGISTERED_RULES, register_candidate_rules,
    _observation_evaluation, _rule_kind,
    P1F3_RULES, GAPW_RULES, P1F3_CRITERIA_VERSION,
    RULE_GAP_V1, RULE_WIDTH_V1,
    collect_evidence, build_evidence_items, run_candidate_rules,
    RULE_LEVEL_STATES, VERDICT_STATES,
    RULE_REJECTED, VERDICT_NOT_MEASURABLE, VERDICT_SUPPORTED,
    VERDICT_CONTRADICTED)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def snap():
    return dict(AUTHORIZED_RULES), dict(_REGISTERED_RULES)


def restore(s):
    AUTHORIZED_RULES.clear()
    _REGISTERED_RULES.clear()
    AUTHORIZED_RULES.update(s[0])
    _REGISTERED_RULES.update(s[1])


S = snap()   # 生产状态快照：8 条规则

# ======================================================================
# A. 注册与授权
# ======================================================================
print('===== A. 注册与授权 =====')
check('A', 'gap_v1 / width_v1 均在授权表（经注册入口注册）',
      'gap_v1' in AUTHORIZED_RULES and 'width_v1' in AUTHORIZED_RULES)
check('A', '授权绑定 criteria_version=phase_criteria_v1_observation',
      AUTHORIZED_RULES.get('gap_v1') == P1F3_CRITERIA_VERSION
      and AUTHORIZED_RULES.get('width_v1') == P1F3_CRITERIA_VERSION)
check('A', '注册表保存规则对象且 candidate 正确',
      _REGISTERED_RULES['gap_v1']['candidate'] == 'gap_scaling'
      and _REGISTERED_RULES['width_v1']['candidate'] == 'width_scaling')
check('A', '两条规则类别均为 capability_boundary（能力边界，非相级判定）',
      _rule_kind(RULE_GAP_V1) == 'capability_boundary'
      and _rule_kind(RULE_WIDTH_V1) == 'capability_boundary')
check('A', 'P1F3_RULES 元组内容不变（6 条，无 gap/width）',
      [r['rule_id'] for r in P1F3_RULES]
      == ['sc_assoc_v1', 'cdw_v1', 'sdw_v1',
          'did_phase_v1', 'pdw_v1', 'stripe_v1'])
check('A', 'GAPW_RULES 为独立二元组',
      [r['rule_id'] for r in GAPW_RULES] == ['gap_v1', 'width_v1'])

# ======================================================================
# B. 空报告显式执行：恒 not_measurable
# ======================================================================
print('===== B. 空报告显式执行 GAPW_RULES =====')
ph_b = run_candidate_rules(collect_evidence([]), GAPW_RULES)
check('B', '候选集合恰为两个能力诊断对象（无其它候选）',
      set(ph_b.candidate_verdicts) == {'gap_scaling', 'width_scaling'},
      str(set(ph_b.candidate_verdicts)))
for cand, rid in (('gap_scaling', 'gap_v1'), ('width_scaling', 'width_v1')):
    v = ph_b.candidate_verdicts[cand]
    tr = next(t for t in ph_b.rule_trace if t['rule_id'] == rid)
    check('B', f'{cand} → verdict 恰为 not_measurable',
          v['verdict'] == VERDICT_NOT_MEASURABLE, v['verdict'])
    check('B', f'{cand} → 非 supported / contradicted（结构保证）',
          v['verdict'] != VERDICT_SUPPORTED
          and v['verdict'] != VERDICT_CONTRADICTED)
    check('B', f'{cand} → rule_trace 记录证据缺失且带 basis 依据',
          len(tr['evidence_missing']) == 1
          and tr['evidence_missing'][0]['interpretation']
          == VERDICT_NOT_MEASURABLE
          and bool(tr['evidence_missing'][0].get('basis')))
    check('B', f'{cand} → 支持/反证求值均非 True（evaluation 恒 False + '
          'counter_evidence=None）',
          tr['support_evaluation'] is not True
          and tr['counter_evaluation'] is not True
          and RULE_GAP_V1['evaluation'] is _observation_evaluation
          and RULE_GAP_V1['counter_evidence'] is None
          and RULE_WIDTH_V1['evaluation'] is _observation_evaluation
          and RULE_WIDTH_V1['counter_evidence'] is None)
check('B', '空报告 + GAPW_RULES 的 conclusion=INCONCLUSIVE',
      ph_b.conclusion == 'INCONCLUSIVE', ph_b.conclusion)

# ======================================================================
# C. 含能量外推观测：附属关联不冒充能隙输入
# ======================================================================
print('===== C. 含能量观测的报告 =====')
rep_energy = from_energy_extrapolation(e_inf=-0.50,
                                       ns=[80, 160, 240, 320, 400])
ph_c = run_candidate_rules(collect_evidence([rep_energy]), GAPW_RULES)
v_gap = ph_c.candidate_verdicts['gap_scaling']
tr_gap = next(t for t in ph_c.rule_trace if t['rule_id'] == 'gap_v1')
check('C', '能量外推条目进入证据层（observation 类）',
      any(it['analyzer'] == ANALYZER_ENERGY
          and it['observable'] == OBSERVABLE_ENERGY
          for it in build_evidence_items(collect_evidence([rep_energy]))))
check('C', '必需能隙输入仍缺失（matched_count=0），缺失路径提前返回'
      '（entry_indices 保持空，引擎既定语义）',
      tr_gap['evidence_missing'][0]['matched_count'] == 0
      and tr_gap.get('entry_indices') == [])
check('C', 'gap_scaling 仍恒 not_measurable（能量观测不冒充能隙证据）',
      v_gap['verdict'] == VERDICT_NOT_MEASURABLE, v_gap['verdict'])
v_wid = ph_c.candidate_verdicts['width_scaling']
check('C', 'width_scaling 仍恒 not_measurable（无宽度序列）',
      v_wid['verdict'] == VERDICT_NOT_MEASURABLE, v_wid['verdict'])
check('C', '占位 observable 未被伪造为已具备（gap_excitation_energy / '
      'cylinder_width_series 不在证据条目中）',
      not any(it.get('observable') in ('gap_excitation_energy',
                                       'cylinder_width_series')
              for it in collect_evidence([rep_energy]).entries))

# ======================================================================
# D. 既有守卫回归（快照/恢复，不污染生产注册表）
# ======================================================================
print('===== D. 既有守卫回归 =====')


def tables():
    return (sorted(AUTHORIZED_RULES.items()),
            sorted((k, v['candidate'], v.get('rule_kind'))
                   for k, v in _REGISTERED_RULES.items()))


before = tables()
sup_gap = {'rule_id': 'gap_support_v1', 'candidate': 'gap_scaling',
           'criteria_version': P1F3_CRITERIA_VERSION,
           'applicability': lambda items: {'applicable': True},
           'required_evidence': [], 'quality_gates': [],
           'evaluation': lambda m, p: True, 'counter_evidence': None,
           'rule_kind': 'phase_judgment'}
try:
    register_candidate_rules([sup_gap], P1F3_CRITERIA_VERSION)
    check('D', '同候选（gap_scaling）相级规则注册 → ValueError', False,
          '未抛异常')
except ValueError as e:
    check('D', '同候选（gap_scaling）相级规则注册 → ValueError',
          '冲突' in str(e), str(e)[:80])
check('D', '拒绝后两注册表与调用前完全一致',
      tables() == before)

before = tables()
try:
    register_candidate_rules([RULE_GAP_V1], P1F3_CRITERIA_VERSION)
    check('D', '同 rule_id（gap_v1）未声明 replace → ValueError（P3-1）',
          False, '未抛异常')
except ValueError as e:
    check('D', '同 rule_id（gap_v1）未声明 replace → ValueError（P3-1）',
          '已注册' in str(e) and 'replace' in str(e), str(e)[:80])
check('D', 'P3-1 失败后两注册表零修改', tables() == before)

before = tables()
try:
    register_candidate_rules(
        [{'rule_id': 'fresh_d3', 'candidate': 'other_diag',
          'criteria_version': P1F3_CRITERIA_VERSION,
          'applicability': lambda items: {'applicable': True},
          'required_evidence': [], 'quality_gates': [],
          'evaluation': _observation_evaluation, 'counter_evidence': None}],
        P1F3_CRITERIA_VERSION, replace=['ghost_id'])
    check('D', 'replace 含未注册 ID → ValueError（P3-2）', False,
          '未抛异常')
except ValueError as e:
    check('D', 'replace 含未注册 ID → ValueError（P3-2）',
          'ghost_id' in str(e) and '未注册' in str(e), str(e)[:80])
check('D', 'P3-2 失败后两注册表零修改', tables() == before)
restore(S)

# 版本失配：注册表有 v1 授权时，执行旧版本规则对象 → 拒绝
old_gap = {**RULE_GAP_V1, 'criteria_version': 'v0_old'}
ph_dv = run_candidate_rules(collect_evidence([]), [old_gap])
check('D', '版本不匹配仍被授权校验拒绝（rule_rejected）',
      ph_dv.rule_trace[0]['rule_level_state'] == RULE_REJECTED)
check('D', '版本失配下候选 verdict 为 criteria_undefined（不降级为能力'
      '结论）',
      ph_dv.candidate_verdicts['gap_scaling']['verdict']
      == 'criteria_undefined')
restore(S)

# ======================================================================
# E. P1F3_RULES 历史行为不变 + 冻结项
# ======================================================================
print('===== E. 历史行为与冻结项 =====')
ph_e = run_candidate_rules(collect_evidence([]), P1F3_RULES)
check('E', '仅执行 P1F3_RULES：候选集合不变（6 个，无 gap/width）',
      set(ph_e.candidate_verdicts)
      == {'superconductivity', 'cdw', 'sdw', 'd_id_phase', 'pdw',
          'stripe'})
check('E', 'P1F3 六候选历史 verdict 不变（3 data_missing + '
      '3 not_measurable）',
      {c: v['verdict'] for c, v in ph_e.candidate_verdicts.items()}
      == {'superconductivity': 'data_missing', 'cdw': 'data_missing',
          'sdw': 'data_missing', 'd_id_phase': 'not_measurable',
          'pdw': 'not_measurable', 'stripe': 'not_measurable'},
      str({c: v['verdict'] for c, v in ph_e.candidate_verdicts.items()}))
check('E', 'P1F3_RULES 空报告 conclusion 不变（INCONCLUSIVE）',
      ph_e.conclusion == 'INCONCLUSIVE', ph_e.conclusion)
check('E', 'sc_lro_v1 仍冻结：不在授权表、注册表与 P1F3/GAPW 元组',
      'sc_lro_v1' not in AUTHORIZED_RULES
      and 'sc_lro_v1' not in _REGISTERED_RULES
      and 'sc_lro_v1' not in [r['rule_id'] for r in P1F3_RULES]
      and 'sc_lro_v1' not in [r['rule_id'] for r in GAPW_RULES])
check('E', '状态枚举冻结不变（规则级 6 态 / 候选级 9 态）',
      len(RULE_LEVEL_STATES) == 6 and len(VERDICT_STATES) == 9)

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
