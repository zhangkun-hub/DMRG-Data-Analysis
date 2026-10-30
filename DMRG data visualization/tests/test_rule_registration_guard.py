# -*- coding: utf-8 -*-
"""
规则注册入口约束测试（P1-G.6 阶段 B）。

覆盖：
1. 单独能力边界规则：注册成功且 verdict 维持 not_measurable
2. 能力边界 + 同候选支持规则（rule_kind='phase_judgment'）：
   注册被拒绝（ValueError），不得静默并存
3. 能力边界 + 同候选反证规则（counter_evidence 非 None）：
   注册被拒绝
4. 不同候选的规则互不干扰：注册成功
5. 版本不匹配仍被授权校验拒绝（引擎既有语义，注册后执行验证）
6. 未声明类别且 evaluation 非观测级函数 → 注册拒绝（不猜测）
7. replace 显式替换路径：同 rule_id 升级成功；替换后旧版本规则对象
   执行被拒绝
8. 绕过路径现状（如实固化）：run_candidate_rules 显式 authorized +
   任意 rules 不经过注册入口——引擎不检查候选级冲突（文档性断言，
   防未来误认为引擎有该保护）
9. 生产授权表内容不变（8 条）与历史行为回归

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_rule_registration_guard.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Phase_Diagnosis import (
    AUTHORIZED_RULES, _REGISTERED_RULES, register_candidate_rules,
    _observation_evaluation, P1F3_RULES, GAPW_RULES, P1F3_CRITERIA_VERSION,
    collect_evidence, run_candidate_rules,
    RULE_REJECTED, RULE_STRIPE_V1,
    VERDICT_NOT_MEASURABLE, VERDICT_SUPPORTED)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def make_rule(rule_id, candidate, kind=None, counter=False,
              eval_fn=None):
    return {'rule_id': rule_id, 'candidate': candidate,
            'criteria_version': P1F3_CRITERIA_VERSION,
            'applicability': lambda items: {'applicable': True},
            'required_evidence': [], 'quality_gates': [],
            'evaluation': (eval_fn if eval_fn is not None
                           else (lambda m, p: True)),
            'counter_evidence': (
                {'evaluation': lambda m, p: True, 'note': 't'}
                if counter else None),
            **({'rule_kind': kind} if kind else {})}


def snap():
    """保存并恢复生产注册表（测试不得改变生产状态）。"""
    return dict(AUTHORIZED_RULES), dict(_REGISTERED_RULES)


def restore(s):
    AUTHORIZED_RULES.clear()
    _REGISTERED_RULES.clear()
    AUTHORIZED_RULES.update(s[0])
    _REGISTERED_RULES.update(s[1])


S = snap()   # 生产状态快照：8 条规则

# ======================================================================
# A. 生产注册表内容与单独能力边界行为
# ======================================================================
print('===== A. 生产状态 =====')
check('A', '生产授权表恰为 8 条（P1F3_RULES ∪ GAPW_RULES 经注册入口注册）',
      set(AUTHORIZED_RULES) == {r['rule_id'] for r in P1F3_RULES + GAPW_RULES}
      and len(_REGISTERED_RULES) == 8)
check('A', '注册表保存规则对象副本且规则类别可识别',
      all(_REGISTERED_RULES[r['rule_id']]['candidate'] == r['candidate']
          for r in P1F3_RULES + GAPW_RULES))
ph_a = run_candidate_rules(collect_evidence([]), P1F3_RULES)
check('A', '单独能力边界规则维持 not_measurable（stripe）',
      ph_a.candidate_verdicts['stripe']['verdict']
      == VERDICT_NOT_MEASURABLE)

# ======================================================================
# B. 能力边界 + 同候选支持规则：注册拒绝
# ======================================================================
print('===== B. 能力边界 × 支持规则冲突 =====')
sup_stripe = make_rule('stripe_support_v1', 'stripe',
                       kind='phase_judgment')
try:
    register_candidate_rules([sup_stripe], P1F3_CRITERIA_VERSION)
    check('B', '同候选支持规则注册 → ValueError（拒绝并存）', False,
          '未抛异常')
except ValueError:
    check('B', '同候选支持规则注册 → ValueError（拒绝并存）', True)
check('B', '拒绝后授权表未变化（不得静默并存）',
      'stripe_support_v1' not in AUTHORIZED_RULES
      and set(AUTHORIZED_RULES)
      == {r['rule_id'] for r in P1F3_RULES + GAPW_RULES})

# 反证规则同候选（d_id_phase）
ctr_did = make_rule('did_counter_v1', 'd_id_phase', counter=True)
try:
    register_candidate_rules([ctr_did], P1F3_CRITERIA_VERSION)
    check('B', '同候选反证规则注册 → ValueError（拒绝并存）', False,
          '未抛异常')
except ValueError:
    check('B', '同候选反证规则注册 → ValueError（拒绝并存）', True)
check('B', '拒绝后授权表未变化',
      'did_counter_v1' not in AUTHORIZED_RULES)

# ======================================================================
# C. 不同候选互不干扰
# ======================================================================
print('===== C. 不同候选 =====')
sup_afm = make_rule('afm_support_v1', 'afm', kind='phase_judgment')
ids_c = register_candidate_rules([sup_afm], P1F3_CRITERIA_VERSION)
check('C', '不同候选支持规则注册成功（互不干扰）',
      ids_c == ['afm_support_v1']
      and AUTHORIZED_RULES['afm_support_v1'] == P1F3_CRITERIA_VERSION)
restore(S)

# ======================================================================
# D. 未分类规则拒绝 + 版本失配拒绝
# ======================================================================
print('===== D. 类别与版本 =====')
mystery = make_rule('mystery_v1', 'afm')   # evaluation 非观测级、无声明
try:
    register_candidate_rules([mystery], P1F3_CRITERIA_VERSION)
    check('D', '未声明类别且 evaluation 非观测级 → ValueError', False,
          '未抛异常')
except ValueError:
    check('D', '未声明类别且 evaluation 非观测级 → ValueError', True)

obs_rule = make_rule('obs_afm_v1', 'afm', eval_fn=_observation_evaluation)
register_candidate_rules([obs_rule], P1F3_CRITERIA_VERSION)
restore(S)
# 版本失配（注册表有 v1 时，执行 v0 规则对象）
old_obj = {**RULE_STRIPE_V1, 'criteria_version': 'v0_old'}
ph_d = run_candidate_rules(collect_evidence([]), [old_obj])
check('D', '版本不匹配仍被授权校验拒绝（rule_rejected）',
      ph_d.rule_trace[0]['rule_level_state'] == RULE_REJECTED)
restore(S)

# ======================================================================
# E. replace 显式替换路径
# ======================================================================
print('===== E. replace 替换 =====')
upgraded = {**RULE_STRIPE_V1, 'rule_kind': 'phase_judgment',
            'evaluation': lambda m, p: True}
ids_e = register_candidate_rules(
    [upgraded], 'phase_criteria_v2_stripe_support',
    replace=['stripe_v1'])
check('E', 'replace=True 同 rule_id 升级成功（新版本入授权表）',
      AUTHORIZED_RULES['stripe_v1'] == 'phase_criteria_v2_stripe_support'
      and _REGISTERED_RULES['stripe_v1'].get('rule_kind')
      == 'phase_judgment')
# 旧版本规则对象（v1）执行 → 拒绝
ph_e = run_candidate_rules(collect_evidence([]), [RULE_STRIPE_V1])
check('E', '升级后旧版本规则对象执行被拒（rule_rejected）',
      ph_e.rule_trace[0]['rule_level_state'] == RULE_REJECTED)
restore(S)
check('E', '恢复生产注册表（8 条、原版本）',
      AUTHORIZED_RULES['stripe_v1'] == P1F3_CRITERIA_VERSION
      and len(AUTHORIZED_RULES) == 8)

# ======================================================================
# F. 绕过路径现状（如实固化：引擎不检查候选级冲突）
# ======================================================================
print('===== F. 绕过路径现状 =====')
ph_f = run_candidate_rules(
    collect_evidence([]), [RULE_STRIPE_V1, sup_stripe],
    authorized={'stripe_v1': P1F3_CRITERIA_VERSION,
                'stripe_support_v1': P1F3_CRITERIA_VERSION})
check('F', '显式 authorized + 任意 rules 不经注册入口（文档性现状）：'
      '引擎路径混合执行，verdict=supported——候选级冲突保护只存在于'
      '注册入口，不在引擎',
      ph_f.candidate_verdicts['stripe']['verdict'] == VERDICT_SUPPORTED,
      ph_f.candidate_verdicts['stripe']['verdict'])
restore(S)

# ======================================================================
# H. P3-1/P3-2：显式替换语义与幽灵替换 ID（P1-G.8）
# ======================================================================
print('===== H. 显式替换语义（P1-G.8） =====')


def tables():
    """当前两表快照（校验失败路径零修改用）。"""
    return (sorted(AUTHORIZED_RULES.items()),
            sorted((k, v['candidate'], v.get('rule_kind'))
                   for k, v in _REGISTERED_RULES.items()))


def assert_unchanged(name, before):
    check('H', f'{name} 失败后两注册表与调用前完全一致',
          tables() == before, str(tables())[:100])


# H1 同 rule_id 未声明 replace → 拒绝
before = tables()
try:
    register_candidate_rules([RULE_STRIPE_V1], P1F3_CRITERIA_VERSION)
    check('H', '同 rule_id 未声明 replace → ValueError', False, '未抛异常')
except ValueError as e:
    check('H', '同 rule_id 未声明 replace → ValueError',
          '已注册' in str(e) and 'replace' in str(e), str(e)[:80])
assert_unchanged('同 id 未声明 replace', before)

# H2 同 rule_id 不同版本未声明 replace → 拒绝
before = tables()
v2_obj = {**RULE_STRIPE_V1, 'criteria_version': 'v2_new'}
try:
    register_candidate_rules([v2_obj], 'v2_new')
    check('H', '同 rule_id 不同版本未声明 replace → ValueError', False,
          '未抛异常')
except ValueError:
    check('H', '同 rule_id 不同版本未声明 replace → ValueError', True)
assert_unchanged('同 id 不同版本未声明 replace', before)

# H3 replace 含不存在的 ID → 拒绝
before = tables()
try:
    register_candidate_rules([make_rule('fresh_h3', 'afm',
                                        eval_fn=_observation_evaluation)],
                             P1F3_CRITERIA_VERSION,
                             replace=['ghost_id'])
    check('H', 'replace 含未注册 ID → ValueError 且指出无效 ID', False,
          '未抛异常')
except ValueError as e:
    check('H', 'replace 含未注册 ID → ValueError 且指出无效 ID',
          'ghost_id' in str(e) and '未注册' in str(e), str(e)[:80])
assert_unchanged('幽灵 replace', before)

# H4 replace 混合有效 + 无效 → 拒绝且有效目标未被提前删除
before = tables()
try:
    register_candidate_rules([make_rule('fresh_h4', 'afm',
                                        eval_fn=_observation_evaluation)],
                             P1F3_CRITERIA_VERSION,
                             replace=['stripe_v1', 'ghost_id'])
    check('H', 'replace 混合有效+无效 → ValueError（不提前删除有效目标）',
          False, '未抛异常')
except ValueError:
    check('H', 'replace 混合有效+无效 → ValueError（不提前删除有效目标）',
          True)
assert_unchanged('混合 replace', before)
check('H', '有效目标 stripe_v1 未被删除（仍注册）',
      'stripe_v1' in AUTHORIZED_RULES)

# H5 多个有效替换目标全部验证后替换成功
before = tables()
register_candidate_rules(
    [make_rule('tmp_h5a', 'afm', eval_fn=_observation_evaluation),
     make_rule('tmp_h5b', 'valence_bond', eval_fn=_observation_evaluation)],
    P1F3_CRITERIA_VERSION)
n_before = len(AUTHORIZED_RULES)
ids_h5 = register_candidate_rules(
    [make_rule('new_h5a', 'afm', eval_fn=_observation_evaluation),
     make_rule('new_h5b', 'valence_bond', eval_fn=_observation_evaluation)],
    P1F3_CRITERIA_VERSION,
    replace=['tmp_h5a', 'tmp_h5b'])
check('H', '多有效替换目标全部验证后替换成功（旧 ID 移除、新 ID 生效）',
      ids_h5 == ['new_h5a', 'new_h5b']
      and 'tmp_h5a' not in AUTHORIZED_RULES
      and 'tmp_h5b' not in AUTHORIZED_RULES
      and 'new_h5a' in AUTHORIZED_RULES
      and 'new_h5b' in AUTHORIZED_RULES
      and len(AUTHORIZED_RULES) == n_before)
restore(S)

# H6 显式替换其他规则不能绕过候选级冲突
before = tables()
sup_stripe2 = make_rule('stripe_support_v2', 'stripe',
                        kind='phase_judgment')
try:
    register_candidate_rules([sup_stripe2], P1F3_CRITERIA_VERSION,
                             replace=['did_phase_v1'])
    check('H', 'replace 其他候选规则不能绕过候选级冲突 → ValueError',
          False, '未抛异常')
except ValueError as e:
    check('H', 'replace 其他候选规则不能绕过候选级冲突 → ValueError',
          '冲突' in str(e), str(e)[:80])
assert_unchanged('replace 不绕过冲突', before)
check('H', '被声明替换的 did_phase_v1 未被删除',
      'did_phase_v1' in AUTHORIZED_RULES)

# H7 重复 replace ID 去重稳定（不产生副作用）
before = tables()
register_candidate_rules(
    [make_rule('tmp_h7', 'afm', eval_fn=_observation_evaluation)],
    P1F3_CRITERIA_VERSION)
ids_h7 = register_candidate_rules(
    [make_rule('new_h7', 'afm', eval_fn=_observation_evaluation)],
    P1F3_CRITERIA_VERSION,
    replace=['tmp_h7', 'tmp_h7', 'tmp_h7'])
check('H', '重复 replace ID 去重处理（替换成功且无副作用）',
      ids_h7 == ['new_h7'] and 'tmp_h7' not in AUTHORIZED_RULES
      and 'new_h7' in AUTHORIZED_RULES)
restore(S)

# ======================================================================
# G. 历史行为回归（生产注册表未变）
# ======================================================================
print('===== G. 历史行为回归 =====')
ph_g = run_candidate_rules(collect_evidence([]), P1F3_RULES)
check('G', '生产 6 候选历史 verdict 不变（空报告）',
      {c: v['verdict'] for c, v in ph_g.candidate_verdicts.items()}
      == {'superconductivity': 'data_missing', 'cdw': 'data_missing',
          'sdw': 'data_missing', 'd_id_phase': 'not_measurable',
          'pdw': 'not_measurable', 'stripe': 'not_measurable'},
      str(ph_g.candidate_verdicts))
check('G', '生产授权表 8 条内容与版本不变',
      set(AUTHORIZED_RULES) == {'sc_assoc_v1', 'cdw_v1', 'sdw_v1',
                                'did_phase_v1', 'pdw_v1', 'stripe_v1',
                                'gap_v1', 'width_v1'}
      and all(v == P1F3_CRITERIA_VERSION
              for v in AUTHORIZED_RULES.values()))

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
