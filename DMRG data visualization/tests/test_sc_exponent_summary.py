# -*- coding: utf-8 -*-
"""
SC 指数跨数据汇总测试（D1：SC_Exponent_Summary.py）

覆盖：
- A. 数值不变性：合成已知 K 幂律数据经真实 run_analysis 分析后，汇总
  提取的指数/误差与 report.payload 完全一致（汇总层不重新拟合）
- B. only_observables 筛选：None 保持全量行为；SC-only 不触发其他
  分析器；未知 observable → ValueError
- C. 四态分类：exponent_ok / unstable / data_insufficient / input_missing
  各自构造真实合法结构验证映射；失败原因原样保留；稳定性报告缺失不
  伪装成功
- D. 指数区分：K 与 xi 分列、分类型、分组建，不混列
- E. 候选模型：缺失候选模型不崩溃；存在时 K/误差/拟合空间/AICc 被
  准确提取
- F. 分组规则：同身份同指数类型同组；N / 方向 / refbond / 指数类型
  任一不同即分组建
- G. 输出：严格 JSON（无 NaN/Infinity 字面量）、CSV 固定列序、
  重复执行字节一致
- H. 真实数据 smoke（数据目录存在时运行）：d1_100/N200/refbond 49_50
  经 analyze_doping(only SC) + summarize + 指数行构建，状态合法且
  指数与 cds 行 params_json 同源逐位一致

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_sc_exponent_summary.py
"""
import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

_SPEC_CDS = importlib.util.spec_from_file_location(
    'cross_doping_summary', os.path.join(ROOT, 'Cross_Doping_Summary.py'))
cds = importlib.util.module_from_spec(_SPEC_CDS)
_SPEC_CDS.loader.exec_module(cds)

_SPEC_SES = importlib.util.spec_from_file_location(
    'sc_exponent_summary', os.path.join(ROOT, 'SC_Exponent_Summary.py'))
ses = importlib.util.module_from_spec(_SPEC_SES)
_SPEC_SES.loader.exec_module(ses)

from Physical_Quantity.Analysis_Contract import (
    AnalysisReport, STATUS_OK,
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR)
from Physical_Quantity.Physical_Data_Record import SuperCorrRecord
from Physical_Quantity.Correlation_Decay_Analyzer import (
    FitResult, StabilityReport)
from Physical_Quantity.Analysis_Entry import run_analysis

SC_OBS = OBSERVABLE_SUPERCONDUCTING_CORRELATION

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


# ---- 通用构造器：cds 行模板 + outcome ----
def make_cds_row(**over):
    row = {'doping': 'synth', 'N': 200, 'bond_dim': '1500',
           'refbond_or_source': '49_50', 'key_or_definition': 'A',
           'comparability_bucket': 'sole', 'comparability_reason': '',
           'warnings_json': '[]', 'classification': '',
           'evidence': '', 'fit_r_min': '', 'fit_r_max': '',
           'n_points': '', 'verdict': '', 'fit_model': '',
           'envelope_mode': ''}
    row.update(over)
    return row


def make_payload(classification='power_law', envelope=None, stability=None,
                 candidate_fits=None):
    return SimpleNamespace(
        classification=classification, evidence_level='weak',
        envelope_fit=envelope, stability=stability,
        candidate_fits=candidate_fits,
        recommended_model=classification, envelope_mode_used='all_points')


def make_report(payload, status=STATUS_OK, classification=None):
    return AnalysisReport(
        analyzer='SuperCorrDecayAnalyzer', observable=SC_OBS,
        status=status,
        classification=(payload.classification
                        if classification is None else classification),
        parameters={}, evidence='weak', data_restricted=None,
        warnings=[], metadata={}, payload=payload)


ENV_POWER = FitResult(model='envelope_power', fit_space='log_absC',
                      r2_space='log', params={'A': 1e-4, 'K': 2.0},
                      param_errors={'A': 1e-5, 'K': 0.1},
                      aicc=-10.0, bic=-9.0, error_source='pcov')
ENV_EXP = FitResult(model='envelope_exp', fit_space='log_absC',
                    r2_space='log', params={'A': 1e-4, 'xi': 12.5},
                    param_errors={'A': 1e-5, 'xi': 0.8},
                    aicc=-8.0, bic=-7.0, error_source='pcov')
ST_OK = StabilityReport(verdict='moderate', rel_mad_K=0.2,
                        model_consistency=1.0, spearman_rho=0.1,
                        recommended_r_min=3.0)
ST_STABLE = StabilityReport(verdict='stable', rel_mad_K=0.1,
                            model_consistency=1.0, spearman_rho=0.0,
                            recommended_r_min=3.0)


# ==========================================================================
# A. 数值不变性（合成已知 K 幂律 + 真实分析器；汇总层不重新拟合）
# ==========================================================================
print('===== A. 数值不变性 =====')
rng = np.random.default_rng(7)
_dists = np.arange(3.0, 39.0, 3.0)          # 12 个距离点
_K_true = 2.0
_base = 1e-4 * _dists ** (-_K_true)
_C = _base + 0.05 * np.abs(_base) * rng.standard_normal(len(_dists))
recs_synth = [SuperCorrRecord((49, 50), (52 + 3 * i, 53 + 3 * i), d, v)
              for i, (d, v) in enumerate(zip(_dists, _C))]
rep_synth = run_analysis(SC_OBS, record_list=recs_synth)
payload = rep_synth.payload
check('A', '合成数据：分析完成且包络为幂律形式（K 存在）',
      payload is not None and 'K' in payload.envelope_fit.params,
      payload.envelope_fit.model if payload else '')
row_a = ses.build_exponent_row(
    0,
    make_cds_row(classification=rep_synth.classification,
                 evidence=rep_synth.evidence,
                 fit_r_min=rep_synth.metadata['r_min'],
                 fit_r_max=rep_synth.metadata['r_max'],
                 n_points=rep_synth.metadata['n_points'],
                 verdict=rep_synth.metadata['verdict'],
                 fit_model=payload.recommended_model,
                 envelope_mode=payload.envelope_mode_used,
                 warnings_json=json.dumps(rep_synth.warnings)),
    {'report': rep_synth, 'failure': None})
check('A', '汇总提取指数与 payload 逐位一致（不重新拟合）',
      row_a['exponent_value'] == payload.envelope_fit.params['K']
      and row_a['exponent_error'] == payload.envelope_fit.param_errors['K'],
      f"{row_a['exponent_value']} vs {payload.envelope_fit.params['K']}")
check('A', '稳定性细节与 payload 逐位一致',
      row_a['rel_mad_K'] == payload.stability.rel_mad_K
      and row_a['model_consistency'] == payload.stability.model_consistency
      and row_a['spearman_rho'] == payload.stability.spearman_rho)
check('A', '已知 K=2.0 被现有分析器恢复（宽松容差）',
      abs(row_a['exponent_value'] - 2.0) < 0.5,
      f"K={row_a['exponent_value']}")
check('A', '状态在四分类枚举内且与分类函数一致',
      row_a['state'] in ses._STATES
      and row_a['state'] == ses.classify_state(rep_synth, None)[0],
      row_a['state'])

# ==========================================================================
# B. only_observables 筛选行为
# ==========================================================================
print('===== B. only_observables 筛选 =====')
called = []


def fake_run(observable, *, operation=None, config=None, **kwargs):
    called.append((observable, operation))
    return AnalysisReport(analyzer='FakeAnalyzer', observable=observable,
                          status=STATUS_OK, classification=None,
                          parameters={}, evidence=None, data_restricted=None,
                          warnings=[], metadata={}, payload=None)


fake_loaders = cds.Loaders(super_corr=lambda _d, _r, _k, _N, _a: ['fake'],
                           spin_dot=lambda _d, _N, _a: ['fake'])
tmp_root = os.path.join(tempfile.mkdtemp(prefix='d1_tree_'), 'root')
os.makedirs(os.path.join(
    tmp_root, 't_J_3', 'd1', '1500_measure_super_corr', 'super_corr_0_1'))

orig_run = cds.run_analysis
cds.run_analysis = fake_run
try:
    outs_all = cds.analyze_doping(
        'd1', root=tmp_root, t_j='t_J_3', bond_dim='1500', N=200, a=1.0,
        q_list=[[0.0, 0.0]], loaders=fake_loaders)
    n_all = len(called)
    called.clear()
    outs_sc = cds.analyze_doping(
        'd1', root=tmp_root, t_j='t_J_3', bond_dim='1500', N=200, a=1.0,
        q_list=[[0.0, 0.0]], loaders=fake_loaders,
        only_observables={SC_OBS})
    called_sc = list(called)
    called.clear()
    try:
        cds.analyze_doping(
            'd1', root=tmp_root, t_j='t_J_3', bond_dim='1500', N=200, a=1.0,
            q_list=[[0.0, 0.0]], loaders=fake_loaders,
            only_observables={'bogus_observable'})
        raised = False
    except ValueError:
        raised = True
finally:
    cds.run_analysis = orig_run

check('B', 'only_observables=None：全量 11 次分析调用（3 SC + 3 SC 配对 SF '
      '+ 2+2 自旋 + 1 电荷）',
      n_all == 11 and len(outs_all) == 11, f'{n_all} 次')
check('B', 'SC-only：只调用 SC 衰减（3 次），不触发其他分析器',
      len(called_sc) == 3
      and all(obs == SC_OBS and op is None for obs, op in called_sc)
      and len(outs_sc) == 3,
      called_sc)
check('B', 'SC-only 筛选发生在调用之前（其他 observable 零调用）',
      not any(obs != SC_OBS for obs, _ in called_sc))
check('B', '未知 observable → ValueError（防拼写错误静默空跑）', raised)

# ==========================================================================
# C. 四态分类
# ==========================================================================
print('===== C. 四态分类 =====')


def state_of(report, failure=None):
    return ses.classify_state(report, failure)[0]


check('C', 'verdict=moderate + 幂律包络 → exponent_ok（moderate 不等于失败）',
      state_of(make_report(make_payload(
          envelope=ENV_POWER, stability=ST_OK))) == ses.STATE_EXPONENT_OK)
check('C', 'verdict=stable → exponent_ok',
      state_of(make_report(make_payload(
          envelope=ENV_POWER, stability=ST_STABLE))) == ses.STATE_EXPONENT_OK)
check('C', 'verdict=unstable → unstable',
      state_of(make_report(make_payload(
          envelope=ENV_POWER,
          stability=StabilityReport(verdict='unstable'))))
      == ses.STATE_UNSTABLE)
check('C', 'verdict=insufficient → data_insufficient',
      state_of(make_report(make_payload(
          envelope=ENV_POWER,
          stability=StabilityReport(verdict='insufficient'))))
      == ses.STATE_DATA_INSUFFICIENT)
check('C', 'classification=INCONCLUSIVE → data_insufficient',
      state_of(make_report(make_payload(
          classification='INCONCLUSIVE',
          envelope=ENV_POWER, stability=ST_OK)))
      == ses.STATE_DATA_INSUFFICIENT)
check('C', '包络拟合缺失 → unstable（不伪装成功）',
      state_of(make_report(make_payload(
          envelope=None, stability=ST_OK))) == ses.STATE_UNSTABLE)
check('C', '稳定性报告缺失 → unstable（不伪装成功）',
      state_of(make_report(make_payload(
          envelope=ENV_POWER, stability=None))) == ses.STATE_UNSTABLE)
check('C', 'FileNotFoundError 失败 → input_missing',
      state_of(None, {'type': 'FileNotFoundError', 'message': '缺失'})
      == ses.STATE_INPUT_MISSING)
check('C', 'ValueError 失败 → data_insufficient（原始类型与消息保留）',
      state_of(None, {'type': 'ValueError', 'message': '空输入'})
      == ses.STATE_DATA_INSUFFICIENT)

row_miss = ses.build_exponent_row(
    0, make_cds_row(), {'report': None,
                        'failure': {'type': 'FileNotFoundError',
                                    'message': 'super_corr 目录缺失'}})
check('C', '失败行：failed_type/message 保留、指数字段留空、不填零',
      row_miss['failed_type'] == 'FileNotFoundError'
      and '缺失' in row_miss['failed_message']
      and row_miss['exponent_value'] == ''
      and row_miss['state'] == ses.STATE_INPUT_MISSING)

# ==========================================================================
# D. 指数区分（K 与 xi）
# ==========================================================================
print('===== D. 指数区分 =====')
row_k = ses.build_exponent_row(
    0, make_cds_row(), {'report': make_report(make_payload(
        envelope=ENV_POWER, stability=ST_OK)), 'failure': None})
row_xi = ses.build_exponent_row(
    1, make_cds_row(doping='synth2'), {'report': make_report(make_payload(
        envelope=ENV_EXP, stability=ST_OK)), 'failure': None})
check('D', 'K 行：type=K、form=power、值与误差正确',
      row_k['exponent_type'] == 'K' and row_k['envelope_form'] == 'power'
      and row_k['exponent_value'] == 2.0 and row_k['exponent_error'] == 0.1)
check('D', 'xi 行：type=xi、form=exp、值与误差正确',
      row_xi['exponent_type'] == 'xi' and row_xi['envelope_form'] == 'exp'
      and row_xi['exponent_value'] == 12.5 and row_xi['exponent_error'] == 0.8)
groups_d = ses.build_groups([row_k, row_xi])
check('D', 'K 与 xi 不共组（身份其余全等也分组建）',
      len(groups_d) == 2
      and {g['exponent_type'] for g in groups_d} == {'K', 'xi'})

# ==========================================================================
# E. 候选模型证据
# ==========================================================================
print('===== E. 候选模型 =====')
cands = {
    'power_law': FitResult(model='power_law', fit_space='linear_absC',
                           r2_space='linear',
                           params={'A': 5e-5, 'K': 1.9},
                           param_errors={'A': 1e-5, 'K': 0.2},
                           aicc=5.0, bic=6.0),
    'power_law_C': None, 'osc_power': None, 'osc_power_C': None,
}
row_e = ses.build_exponent_row(
    0, make_cds_row(), {'report': make_report(make_payload(
        envelope=ENV_POWER, stability=ST_OK, candidate_fits=cands)),
        'failure': None})
ev = json.loads(row_e['candidate_models_json'])
check('E', '候选模型证据：K/误差/拟合空间/AICc 被准确提取，None 模型跳过',
      set(ev) == {'power_law'}
      and ev['power_law']['K'] == 1.9 and ev['power_law']['K_err'] == 0.2
      and ev['power_law']['fit_space'] == 'linear_absC'
      and ev['power_law']['aicc'] == 5.0, ev)
row_e2 = ses.build_exponent_row(
    0, make_cds_row(), {'report': make_report(make_payload(
        envelope=ENV_POWER, stability=ST_OK, candidate_fits=None)),
        'failure': None})
check('E', '候选模型缺失：不崩溃、列留空（不编造证据）',
      row_e2['candidate_models_json'] == '')

# ==========================================================================
# F. 分组规则
# ==========================================================================
print('===== F. 分组规则 =====')


def ok_row(i, doping='d1', N=200, bond_dim='1500', ref='49_50',
           direction='A', exp_type='K'):
    env = ENV_POWER if exp_type == 'K' else ENV_EXP
    r = ses.build_exponent_row(
        i, make_cds_row(doping=doping, N=N, bond_dim=bond_dim,
                        refbond_or_source=ref,
                        key_or_definition=direction),
        {'report': make_report(make_payload(
            envelope=env, stability=ST_OK)), 'failure': None})
    return r


rows_f = [ok_row(0, doping='d1_10'), ok_row(1, doping='d1_20'),
          ok_row(2, doping='d1_10', direction='B'),
          ok_row(3, doping='d1_10', ref='50_51'),
          ok_row(4, doping='d1_10', exp_type='xi'),
          ok_row(5, doping='d1_10', N=400)]
groups_f = ses.build_groups(rows_f)
by_key = {(g['N'], g['bond_dim'], g['refbond'], g['direction'],
           g['exponent_type']): g for g in groups_f}
check('F', '同身份同指数类型：跨 doping 同组并列（2 成员）',
      by_key[(200, '1500', '49_50', 'A', 'K')]['member_indices'] == [0, 1]
      and by_key[(200, '1500', '49_50', 'A', 'K')]['n_total'] == 2)
check('F', '方向不同 → 分组建；refbond 不同 → 分组建；指数类型不同 → '
      '分组建；N 不同 → 分组建',
      len(groups_f) == 5
      and all(len(g['member_indices']) == 1
              for g in groups_f
              if g['member_indices'] != [0, 1]),
      [g['member_indices'] for g in groups_f])
check('F', '分组不计算跨组平均指数（组只含身份与状态计数）',
      all('exponent_value' not in g for g in groups_f)
      and by_key[(200, '1500', '49_50', 'A', 'K')]['n_exponent_ok'] == 2)

# ==========================================================================
# G. 输出：严格 JSON / CSV 列序 / 确定性覆盖
# ==========================================================================
print('===== G. 输出 =====')
outdir_g = os.path.join(tempfile.mkdtemp(prefix='d1_out_'), 'nested', 'out')
paths_g = ses.write_outputs(rows_f, groups_f, {'N': 200}, outdir_g)
check('G', '输出目录自动创建（含嵌套）且两文件生成',
      os.path.isfile(paths_g['csv']) and os.path.isfile(paths_g['json']))
raw_g = open(paths_g['json'], encoding='utf-8').read()
back_g = json.loads(raw_g)
check('G', '严格 JSON：无 NaN/Infinity 字面量、可解析、行数与列集正确',
      'NaN' not in raw_g and 'Infinity' not in raw_g
      and len(back_g['rows']) == 6
      and set(back_g['rows'][0].keys()) == set(ses.CSV_COLUMNS))
check('G', 'JSON 含语义说明（不判定物理相态/配对对称性声明）',
      'no_physical_conclusions' in back_g['semantics']
      and 'd+id' in back_g['semantics']['no_physical_conclusions'])
import csv
with open(paths_g['csv'], encoding='utf-8-sig', newline='') as f:
    lines_g = list(csv.DictReader(f))
check('G', 'CSV：固定列序与行数',
      list(lines_g[0].keys()) == list(ses.CSV_COLUMNS)
      and len(lines_g) == 6)
paths_g2 = ses.write_outputs(rows_f, groups_f, {'N': 200}, outdir_g)
check('G', '重复执行：输出字节一致（整体覆盖，不追加）',
      open(paths_g['csv'], 'rb').read() == open(paths_g2['csv'], 'rb').read()
      and open(paths_g['json'], 'rb').read()
      == open(paths_g2['json'], 'rb').read())

# ==========================================================================
# H. 真实数据 smoke（d1_100 / N200 / refbond 49_50；SC-only 全链路）
# ==========================================================================
print('===== H. 真实数据 smoke =====')
REAL_ROOT = os.path.join(ROOT, 'Data_N200')
if os.path.isdir(os.path.join(REAL_ROOT, 't_J_3', 'd1_100')):
    qs200, _ = cds.xc_momentum_grid(200, a=1.0, n_qx=11, n_qy=4)
    outs_real = cds.analyze_doping(
        'd1_100', root=REAL_ROOT, t_j='t_J_3', bond_dim='1500',
        N=200, a=1.0, q_list=qs200, refbonds={'49_50'},
        only_observables={SC_OBS})
    check('H', '真实 d1_100：SC-only 恰好 3 条（A/B/C）且全部成功',
          len(outs_real) == 3
          and all(o['failure'] is None for o in outs_real)
          and all(o['report'].status == 'ok' for o in outs_real),
          [o['context']['key_or_definition'] for o in outs_real])
    summ_real = cds.summarize(outs_real, qs200)
    rows_real = summ_real['rows']
    exp_rows_real = [ses.build_exponent_row(i, r, o)
                     for i, (r, o) in enumerate(zip(rows_real, outs_real))]
    check('H', '真实行：状态均在四分类内、指数类型/值存在且有限',
          all(r['state'] in ses._STATES for r in exp_rows_real)
          and all(r['exponent_type'] in ('K', 'xi')
                  and isinstance(r['exponent_value'], float)
                  and np.isfinite(r['exponent_value'])
                  for r in exp_rows_real),
          [(r['state'], r['exponent_type']) for r in exp_rows_real])
    # 与 cds 行 params_json 同源逐位一致（最强同源证据）
    same_source = True
    for er, cr in zip(exp_rows_real, rows_real):
        if not cr['params_json']:
            continue
        pj = json.loads(cr['params_json'])
        if er['exponent_type'] == 'K' and 'K' in pj:
            same_source = same_source and er['exponent_value'] == pj['K']
        elif er['exponent_type'] == 'xi' and 'xi' in pj:
            same_source = same_source and er['exponent_value'] == pj['xi']
        else:
            same_source = False
    check('H', '真实数据：指数与 cds 行 params_json 逐位一致（同源，'
          '无重拟合漂移）',
          same_source,
          [(r['exponent_type'], r['exponent_value']) for r in exp_rows_real])
    groups_h = ses.build_groups(exp_rows_real)
    check('H', '真实数据：按方向分组建（单 doping 内方向身份不同）',
          len(groups_h) == 3
          and all(g['n_total'] == 1 for g in groups_h))
    outdir_h = tempfile.mkdtemp(prefix='d1_real_out_')
    paths_h = ses.write_outputs(exp_rows_real, groups_h,
                                {'N': 200, 'dopings': ['d1_100']}, outdir_h)
    raw_h = open(paths_h['json'], encoding='utf-8').read()
    check('H', '真实输出：严格 JSON 可解析、无 NaN 字面量',
          'NaN' not in raw_h and json.loads(raw_h)['rows'])
else:
    check('H', '真实数据 smoke（数据目录不存在，跳过）', True,
          '跳过（Data_N200 不存在）')

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
