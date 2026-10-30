# -*- coding: utf-8 -*-
"""
P1-F.3 首批规则（观测级 × 3 + 能力边界 × 2）合成测试。

覆盖：
- 授权清单：仅 5 条规则、criteria_version=phase_criteria_v1_observation；
  stripe_v1 / sc_lro_v1 暂缓不注册
- 观测级结构保证：sc_assoc_v1/cdw_v1/sdw_v1 永不 supported/contradicted/
  conflicting——幂律不确立、指数不反证、q=0 增强不确认、Luther-Emery
  不作必要条件、无长程序输出（evaluation 恒 False + 无 counter 条款）
- sc_assoc_v1：证据存在 → insufficient + entry_indices 关联全部 SC 观测；
  证据缺失 → data_missing
- cdw_v1 纪律门：连通关联定义通过；非连通/定义缺失 → gate_failed →
  quality_failed；无电荷 SF → data_missing；无通用峰值阈值
- sdw_v1 纪律门：参数层波矢物理标签（AFM/Γ/K）保持未定；标签出现 →
  gate_failed → quality_failed；峰值观测存在时纳入 entry_indices；
  峰值缺失（min_entries=0 附属 spec）不阻塞
- did_phase_v1 / pdw_v1 能力边界：恒 not_measurable（含空报告），
  evidence_missing 携带 basis 依据说明，永不 supported/contradicted
- 版本失配 → rule_rejected → 候选 criteria_undefined
- 序列化冒烟（candidate_verdicts 可 to_dict/to_json）

全部合成输入（AnalysisReport 直构），不读取科研数据、不修改冻结引擎。

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_phase_diagnosis_rules_observation.py
"""
import json
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK, OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR, OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR, OBSERVABLE_SC_PAIRING_SYMMETRY,
    ANALYZER_SC_PAIR_SF, ANALYZER_SC_PAIRING_SYMMETRY)
from Physical_Quantity.Phase_Diagnosis import (
    collect_evidence, run_candidate_rules, AUTHORIZED_RULES, P1F3_RULES,
    GAPW_RULES, P1F3_CRITERIA_VERSION,
    VERDICT_SUPPORTED, VERDICT_CONTRADICTED, VERDICT_CONFLICTING,
    VERDICT_INSUFFICIENT, VERDICT_QUALITY_FAILED, VERDICT_DATA_MISSING,
    VERDICT_CRITERIA_UNDEFINED, VERDICT_NOT_MEASURABLE,
    RULE_EXECUTED_UNSATISFIED, RULE_GATE_FAILED, RULE_REJECTED,
    CONCLUSION_INCONCLUSIVE,
)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def make_rep(analyzer, observable, status=STATUS_OK, parameters=None,
             classification=None, payload=None, metadata=None):
    return AnalysisReport(
        analyzer=analyzer, observable=observable, status=status,
        classification=classification, parameters=parameters or {},
        evidence=None, data_restricted=None, warnings=[],
        metadata=metadata or {}, payload=payload)


def sc_decay_rep(classification='power_law'):
    return make_rep('SuperCorrDecayAnalyzer',
                    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                    parameters={'K': 1.7}, classification=classification)


def sc_pair_sf_rep():
    return make_rep(ANALYZER_SC_PAIR_SF, OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                    payload={'S_q': [1.0, 0.9]})


def pair_sym_rep():
    return make_rep(ANALYZER_SC_PAIRING_SYMMETRY,
                    OBSERVABLE_SC_PAIRING_SYMMETRY,
                    payload={'phase': {'state': 'blocked_by_data'}})


def charge_sf_rep(definition='charge_density_connected'):
    return make_rep('ChargeStructureFactorEstimator',
                    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                    payload={'S_q': [1.0]},
                    metadata={'definition': definition})


def spin_sf_rep():
    return make_rep('SpinStructureFactorEstimator',
                    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                    payload={'S_q': [1.0]}, metadata={'definition': 'sz_sz'})


def peak_rep(parameters=None):
    p = parameters if parameters is not None else {
        'n_peaks': 1, 'peak_positions': [[0.0, 0.0]],
        'peak_heights': [12.0]}
    return make_rep('SpinStructureFactorPeakAnalyzer',
                    OBSERVABLE_SPIN_STRUCTURE_FACTOR, parameters=p,
                    payload={'peaks': []})


def trace_of(ph, rule_id):
    return next(t for t in ph.rule_trace if t['rule_id'] == rule_id)


def run(reps):
    ph = collect_evidence(reps)
    return run_candidate_rules(ph, P1F3_RULES)


# ======================================================================
# A. 授权清单与规则对象结构
# ======================================================================
print('===== A. 授权清单 =====')
check('A', 'AUTHORIZED_RULES 恰为 8 条（P1F3 观测级 ×3 + 能力边界 ×3 + '
      'P1-G.28 Gap/Width 能力边界 ×2）',
      set(AUTHORIZED_RULES) == {'sc_assoc_v1', 'cdw_v1', 'sdw_v1',
                                'did_phase_v1', 'pdw_v1', 'stripe_v1',
                                'gap_v1', 'width_v1'},
      str(AUTHORIZED_RULES))
check('A', '全部授权绑定 criteria_version=phase_criteria_v1_observation',
      P1F3_CRITERIA_VERSION == 'phase_criteria_v1_observation'
      and all(v == P1F3_CRITERIA_VERSION for v in AUTHORIZED_RULES.values()))
check('A', 'P1F3_RULES ∪ GAPW_RULES 与授权清单一一对应，候选集合正确',
      {r['rule_id'] for r in P1F3_RULES + GAPW_RULES} == set(AUTHORIZED_RULES)
      and {r['candidate'] for r in P1F3_RULES + GAPW_RULES}
      == {'superconductivity', 'cdw', 'sdw', 'd_id_phase', 'pdw',
          'stripe', 'gap_scaling', 'width_scaling'},
      str({r['candidate'] for r in P1F3_RULES + GAPW_RULES}))
check('A', 'sc_lro_v1 仍暂缓：不定义不注册',
      'sc_lro_v1' not in AUTHORIZED_RULES
      and 'sc_lro_v1' not in [r['rule_id'] for r in P1F3_RULES])
check('A', '全部规则无正式反证条款（counter_evidence 恒 None）',
      all(r.get('counter_evidence') is None for r in P1F3_RULES))
check('A', '全部规则具备 evaluation 回调（观测级统一恒 False）',
      all(callable(r.get('evaluation')) for r in P1F3_RULES))

# ======================================================================
# B. sc_assoc_v1：观测级结构保证（幂律不确立 / q=0 增强不确认）
# ======================================================================
print('===== B. sc_assoc_v1 观测级 =====')
ph = run([sc_decay_rep('power_law'), sc_pair_sf_rep(), pair_sym_rep()])
v = ph.candidate_verdicts['superconductivity']
check('B', '幂律衰减 + q=0 附近 SF 观测存在 → verdict=insufficient'
      '（幂律不确立、q=0 增强不确认、无长程序输出）',
      v['verdict'] == VERDICT_INSUFFICIENT, v['verdict'])
check('B', 'superconductivity 永不 supported/contradicted/conflicting',
      v['verdict'] not in (VERDICT_SUPPORTED, VERDICT_CONTRADICTED,
                           VERDICT_CONFLICTING))
t = trace_of(ph, 'sc_assoc_v1')
check('B', '规则级状态 executed_unsatisfied（支持条件不成立，不构成反证）',
      t['rule_level_state'] == RULE_EXECUTED_UNSATISFIED
      and t['support_evaluation'] is False
      and t['counter_evaluation'] is None,
      str((t['rule_level_state'], t['support_evaluation'],
           t['counter_evaluation'])))
check('B', 'entry_indices 关联全部 3 条 SC 观测条目（衰减/SF/配对对称性）',
      sorted(t['entry_indices']) == [0, 1, 2], str(t['entry_indices']))
check('B', 'evidence_missing 为 None（证据存在，规则正常执行）',
      t['evidence_missing'] is None)
check('B', '总体 conclusion=INCONCLUSIVE（观测级规则不产生相级结论）',
      ph.conclusion == CONCLUSION_INCONCLUSIVE, ph.conclusion)
check('B', 'verdict_summary 中 supported/contradicted/conflicting 组全空',
      not ph.verdict_summary['supported_candidates']
      and not ph.verdict_summary['contradicted_candidates']
      and not ph.verdict_summary['conflicting_candidates'])

# ======================================================================
# C. sc_assoc_v1：指数衰减不反证
# ======================================================================
print('===== C. 指数衰减不反证 =====')
for cls in ('exponential', 'oscillatory_exponential'):
    ph_c = run([sc_decay_rep(cls)])
    vc = ph_c.candidate_verdicts['superconductivity']
    check('C', f'classification={cls} → 仍 insufficient（指数不反证）',
          vc['verdict'] == VERDICT_INSUFFICIENT
          and vc['verdict'] != VERDICT_CONTRADICTED, vc['verdict'])
    check('C', f'classification={cls} → 规则级 executed_unsatisfied'
          f'（无反证条款触发）',
          trace_of(ph_c, 'sc_assoc_v1')['rule_level_state']
          == RULE_EXECUTED_UNSATISFIED)

# ======================================================================
# D. sc_assoc_v1：无证据 → data_missing
# ======================================================================
print('===== D. 无 SC 证据 =====')
ph_d = run([charge_sf_rep()])
vd = ph_d.candidate_verdicts['superconductivity']
check('D', '无 SC 衰减证据 → data_missing（不伪造观测）',
      vd['verdict'] == VERDICT_DATA_MISSING, vd['verdict'])
td = trace_of(ph_d, 'sc_assoc_v1')
check('D', '缺失 spec 指向 superconducting_correlation，解释为 data_missing',
      any(m['spec'].get('observable') == 'superconducting_correlation'
          and m['interpretation'] == VERDICT_DATA_MISSING
          for m in (td['evidence_missing'] or [])),
      str(td['evidence_missing']))

# ======================================================================
# E. cdw_v1：连通关联定义纪律门（无通用峰值阈值）
# ======================================================================
print('===== E. cdw_v1 连通定义纪律门 =====')
ph_e1 = run([charge_sf_rep('charge_density_connected')])
ve1 = ph_e1.candidate_verdicts['cdw']
check('E', '连通关联定义 → verdict=insufficient（观测记录，不判定）',
      ve1['verdict'] == VERDICT_INSUFFICIENT
      and trace_of(ph_e1, 'cdw_v1')['rule_level_state']
      == RULE_EXECUTED_UNSATISFIED, ve1['verdict'])

ph_e2 = run([charge_sf_rep('raw_density_correlation')])
ve2 = ph_e2.candidate_verdicts['cdw']
te2 = trace_of(ph_e2, 'cdw_v1')
check('E', '非连通定义 → 纪律门失败 → quality_failed（不等于否定）',
      ve2['verdict'] == VERDICT_QUALITY_FAILED
      and te2['rule_level_state'] == RULE_GATE_FAILED
      and '非连通' in te2['reason'], str((ve2['verdict'], te2['reason'])))

ph_e3 = run([charge_sf_rep(None)])
ve3 = ph_e3.candidate_verdicts['cdw']
check('E', '定义缺失（无法确认连通）→ 保守 gate_failed → quality_failed',
      ve3['verdict'] == VERDICT_QUALITY_FAILED
      and trace_of(ph_e3, 'cdw_v1')['rule_level_state'] == RULE_GATE_FAILED,
      ve3['verdict'])

ph_e4 = run([spin_sf_rep()])
check('E', '无电荷 SF → data_missing',
      ph_e4.candidate_verdicts['cdw']['verdict'] == VERDICT_DATA_MISSING)
check('E', '规则无峰值阈值条款（structure：仅一个 connected_definition 门）',
      [g['name'] for r in P1F3_RULES if r['rule_id'] == 'cdw_v1'
       for g in r.get('quality_gates') or []] == ['connected_definition'])

# ======================================================================
# F. sdw_v1：波矢未标签纪律门 + 峰值关联
# ======================================================================
print('===== F. sdw_v1 波矢未标签纪律门 =====')
ph_f1 = run([spin_sf_rep(), peak_rep()])
vf1 = ph_f1.candidate_verdicts['sdw']
tf1 = trace_of(ph_f1, 'sdw_v1')
check('F', '自旋 SF + 峰值观测 → insufficient（q 位置记录，不解释）',
      vf1['verdict'] == VERDICT_INSUFFICIENT
      and tf1['rule_level_state'] == RULE_EXECUTED_UNSATISFIED, vf1['verdict'])
check('F', 'entry_indices 关联 SF 与峰值两条条目（min_entries=0 附属 spec）',
      sorted(tf1['entry_indices']) == [0, 1], str(tf1['entry_indices']))

ph_f2 = run([spin_sf_rep(), peak_rep({
    'n_peaks': 1, 'afm_label': 'AFM',
    'peak_positions': [[0.0, 0.0]], 'peak_heights': [12.0]})])
vf2 = ph_f2.candidate_verdicts['sdw']
tf2 = trace_of(ph_f2, 'sdw_v1')
check('F', '参数层出现 AFM 标签 → 纪律门失败 → quality_failed（波矢未授权解释）',
      vf2['verdict'] == VERDICT_QUALITY_FAILED
      and tf2['rule_level_state'] == RULE_GATE_FAILED
      and '波矢' in tf2['reason'], str((vf2['verdict'], tf2['reason'])))

ph_f3 = run([spin_sf_rep()])
tf3 = trace_of(ph_f3, 'sdw_v1')
check('F', '无峰值观测（附属 spec）→ 不阻塞，仍 insufficient 且关联 SF 条目',
      ph_f3.candidate_verdicts['sdw']['verdict'] == VERDICT_INSUFFICIENT
      and tf3['entry_indices'] == [0], str(tf3['entry_indices']))

ph_f4 = run([charge_sf_rep()])
check('F', '无自旋 SF → data_missing',
      ph_f4.candidate_verdicts['sdw']['verdict'] == VERDICT_DATA_MISSING)

# ======================================================================
# G. did_phase_v1：能力边界（实值数据恒 not_measurable）
# ======================================================================
print('===== G. did_phase_v1 能力边界 =====')
ph_g1 = run([sc_decay_rep(), pair_sym_rep()])
vg1 = ph_g1.candidate_verdicts['d_id_phase']
tg1 = trace_of(ph_g1, 'did_phase_v1')
check('G', '配对对称性观测存在 → verdict=not_measurable（恒，非支持非反证）',
      vg1['verdict'] == VERDICT_NOT_MEASURABLE
      and vg1['verdict'] not in (VERDICT_SUPPORTED, VERDICT_CONTRADICTED),
      vg1['verdict'])
check('G', 'evidence_missing 含 not_measurable 解释且 basis 追溯实值数据依据',
      any(m['interpretation'] == VERDICT_NOT_MEASURABLE
          and m['basis'] and 'USE_REAL=1' in m['basis']
          for m in (tg1['evidence_missing'] or [])),
      str(tg1['evidence_missing']))
check('G', '缺相位信息 spec 的 observable 标记为 complex_phase_data（自描述）',
      any(m['spec'].get('observable') == 'complex_phase_data'
          for m in (tg1['evidence_missing'] or [])))

ph_g2 = run([])
check('G', '空报告（无任何证据）→ 仍 not_measurable（能力结论不依赖观测存在）',
      ph_g2.candidate_verdicts['d_id_phase']['verdict']
      == VERDICT_NOT_MEASURABLE,
      ph_g2.candidate_verdicts['d_id_phase']['verdict'])

# ======================================================================
# H. pdw_v1：能力边界（单参考键受限恒 not_measurable）
# ======================================================================
print('===== H. pdw_v1 能力边界 =====')
ph_h1 = run([sc_decay_rep(), sc_pair_sf_rep()])
vh1 = ph_h1.candidate_verdicts['pdw']
th1 = trace_of(ph_h1, 'pdw_v1')
check('H', 'SC 配对 SF 观测存在 → verdict=not_measurable（恒）',
      vh1['verdict'] == VERDICT_NOT_MEASURABLE, vh1['verdict'])
check('H', 'basis 说明单参考键限制且不使用 Q_PDW≈2Q_stripe 文献关系',
      any(m['interpretation'] == VERDICT_NOT_MEASURABLE
          and m['basis'] and '单参考键' in m['basis']
          and 'Q_PDW' in m['basis']
          for m in (th1['evidence_missing'] or [])),
      str(th1['evidence_missing']))
check('H', '空报告 → 仍 not_measurable',
      run([]).candidate_verdicts['pdw']['verdict']
      == VERDICT_NOT_MEASURABLE)

# ======================================================================
# H2. stripe_v1：能力边界（缺少同口径尺寸序列恒 not_measurable）
# ======================================================================
print('===== H2. stripe_v1 能力边界 =====')
# 附属观测：密度 + 电荷峰值 + 对齐（存在时纳入关联，不阻塞能力结论）
ph_s1 = run([make_rep('DensityEstimator', 'local_density',
                      parameters={'N': 200, 'density_by_site': [0.9] * 200},
                      payload={}),
             make_rep('SpinStructureFactorPeakAnalyzer',
                      'charge_structure_factor',
                      parameters={'n_peaks': 1, 'peak_positions': [[0.5, 0]],
                                  'peak_heights': [2.0]},
                      payload={'peaks': []}),
             make_rep('ChargeSpinPeakAlignmentEstimator',
                      'charge_spin_peak_alignment',
                      parameters={'min_dq': 0.0})])
vs1 = ph_s1.candidate_verdicts['stripe']
ts1 = trace_of(ph_s1, 'stripe_v1')
check('H2', 'O1/O2/O4 观测存在 → stripe verdict=not_measurable（恒，非反证）',
      vs1['verdict'] == VERDICT_NOT_MEASURABLE
      and vs1['verdict'] != VERDICT_CONTRADICTED, vs1['verdict'])
check('H2', 'basis 追溯 O5 审查依据（源码版本/D 调度/源点身份/两尺寸）',
      any(m['interpretation'] == VERDICT_NOT_MEASURABLE
          and m['basis'] and '44269cb' in m['basis']
          and '源点身份' in m['basis']
          for m in (ts1['evidence_missing'] or [])),
      str(ts1['evidence_missing']))
check('H2', '空报告 → 仍 not_measurable（能力结论不依赖观测存在）',
      run([]).candidate_verdicts['stripe']['verdict']
      == VERDICT_NOT_MEASURABLE)

# ======================================================================
# I. 授权一致性：版本失配拒绝
# ======================================================================
print('===== I. 授权一致性 =====')
bad_rule = {**P1F3_RULES[0], 'criteria_version': 'other_version'}
ph_i = collect_evidence([sc_decay_rep()])
ph_i = run_candidate_rules(ph_i, [bad_rule])
ti = ph_i.rule_trace[0]
check('I', 'criteria_version 失配 → rule_rejected（授权双绑定）',
      ti['rule_level_state'] == RULE_REJECTED
      and '未授权' in ti['reason'], str((ti['rule_level_state'],
                                         ti['reason'])))
check('I', '全部规则被拒绝 → 候选 criteria_undefined（不冒充判定）',
      ph_i.candidate_verdicts['superconductivity']['verdict']
      == VERDICT_CRITERIA_UNDEFINED,
      ph_i.candidate_verdicts['superconductivity']['verdict'])

# ======================================================================
# J. 序列化冒烟
# ======================================================================
print('===== J. 序列化 =====')
ph_j = run([sc_decay_rep(), sc_pair_sf_rep(), pair_sym_rep(),
            charge_sf_rep(), spin_sf_rep(), peak_rep()])
d_j = ph_j.to_dict()
check('J', 'to_dict 含 6 候选 verdicts 与完整 rule_trace',
      set(d_j['candidate_verdicts'])
      == {'superconductivity', 'cdw', 'sdw', 'd_id_phase', 'pdw',
          'stripe'}
      and len(d_j['rule_trace']) == 6)
path_j = os.path.join(tempfile.mkdtemp(prefix='p1f3_'), 'ph.json')
ph_j.to_json(path_j)
with open(path_j, encoding='utf-8') as f:
    back = json.load(f)
check('J', 'to_json 严格落盘可回读且 verdict 一致',
      back['candidate_verdicts'] == d_j['candidate_verdicts'])

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
