# -*- coding: utf-8 -*-
"""
一键受控证据报告（P1-G.29 提交 2）。

对单 doping 单参数组运行可用的物理量分析，产出 Markdown 人读报告
（report.md）+ 严格 JSON 机读底稿（report.json）+ 可选图件（plots/*.png）。

职责边界（硬约束）：
- 只输出观测数值 + 候选 verdict + 判据检查表；
- 不注册任何相级支持/反证规则（sc_lro_v1 冻结；相级判据需负责人批准），
  不输出"是 XX 相"的结论——渲染器不含任何相标签映射代码；
- 候选 verdict 由 PhaseDiagnosis 规则引擎的已注册规则产生（观测级恒
  insufficient/data_missing、能力边界恒 not_measurable），结论段只引用
  verdict 表数值，不重新解释；
- 纪律声明：软件正确性 ≠ 物理正确性；测试通过 ≠ 物理证明；本报告
  不是物理论文结论，是受控证据汇编。

运行方式（Calculation 环境，DMRG data visualization/ 目录下）：
    python -X utf8 Phase_Analysis_Report.py --doping d1_100
    python -X utf8 Phase_Analysis_Report.py --doping d1_10 \
        --energy-root Data_Energy --with-density-profile
"""
import argparse
import datetime
import json
import math
import os
import sys

# 绘图纪律：必须在 import 包模块（热导入 matplotlib）之前设定
os.environ.setdefault('MPLBACKEND', 'Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Physical_Quantity.Physical_Data_Record import DensityRecord  # noqa: E402
from Physical_Quantity.Analysis_Contract import (  # noqa: E402
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_ENERGY,
    OBSERVABLE_SC_PAIRING_SYMMETRY,
    OBSERVABLE_LOCAL_DENSITY,
    ANALYZER_ENERGY,
    ANALYZER_DENSITY,
    ANALYZER_SC_PAIRING_SYMMETRY,
    ANALYZER_PEAK)
from Physical_Quantity.Analysis_Entry import run_analysis  # noqa: E402
from Physical_Quantity.Phase_Diagnosis import (  # noqa: E402
    collect_evidence, audit_comparability, evidence_sufficiency_check,
    run_candidate_rules, P1F3_RULES, GAPW_RULES, AUTHORIZED_RULES,
    VERDICT_STATES)
from Physical_Quantity.Spin_Structure_Factor import xc_momentum_grid  # noqa: E402
from Physical_Quantity.Density_Process import DensityProcessor  # noqa: E402
from Physical_Quantity.Energy_Process import EnergyRecord  # noqa: E402
import Cross_Doping_Summary as cds  # noqa: E402

DEFAULT_T_J = 't_J_3'
DEFAULT_BOND_DIM = '1500'
DEFAULT_N = 200
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, 'phase_analysis_report_output')

# ==========================================================================
# 判据检查表（静态对照 docs/Phase_Diagnosis_Physical_Criteria_Matrix.md；
# 不修改文档、不越界判相。文档更新后由一致性测试报警）
# ==========================================================================
CHECKLIST_STATUSES = ('证据具备', '证据缺失', '不可测量', '未纳入')

CRITERIA_CHECKLIST = [
    # ---- 观测级 / 能力边界（矩阵 §6 规则行） ----
    {'criterion_id': 'sc_assoc_v1', 'section_ref': '§6 sc_assoc_v1',
     'candidate': 'SC', 'registration': 'observation',
     'required_evidence': 'superconducting_correlation 衰减 + '
                         'sc_pair_structure_factor + sc_pairing_symmetry',
     'evidence_observables': [OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                              OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                              OBSERVABLE_SC_PAIRING_SYMMETRY],
     'data_availability': 'SC 衰减/配对 SF：N200 四 doping 全有（1500/5000）；'
                          '配对对称可运行但候选支持判定恒 not_evaluated',
     'report_inclusion': 'included', 'status': '证据具备',
     'note': '观测级规则：不产生相级支持/反证；幂律衰减不确立配对',
     'doc_quoted': True},
    {'criterion_id': 'sc_lro_v1', 'section_ref': '§6 sc_lro_v1',
     'candidate': 'SC', 'registration': 'deferred',
     'required_evidence': '同口径多 N 衰减序列',
     'evidence_observables': [OBSERVABLE_SUPERCONDUCTING_CORRELATION],
     'data_availability': 'N200/N400 跨测量源码变更、D 调度不同、源点不同'
                          '（Protocol Lock §6 全部阻断）——无同口径尺寸序列',
     'report_inclusion': 'not_included_v1', 'status': '证据缺失',
     'note': 'sc_lro_v1 冻结未注册：不定义、不注册、不授权（长程序级判定'
             '需要新测量）',
     'doc_quoted': True},
    {'criterion_id': 'did_phase_v1', 'section_ref': '§6 did_phase_v1',
     'candidate': 'd+id', 'registration': 'capability_boundary',
     'required_evidence': '复数配对振幅',
     'evidence_observables': [OBSERVABLE_SC_PAIRING_SYMMETRY],
     'data_availability': '全部数据为实值 MPS 基态（USE_REAL=1），复相位'
                          '原则上不可恢复',
     'report_inclusion': 'opt_in', 'status': '不可测量',
     'note': '能力边界规则：实值数据下 d+id/TRSB 判定不可测量（恒 '
             'not_measurable）',
     'doc_quoted': True},
    {'criterion_id': 'cdw_v1', 'section_ref': '§6 cdw_v1',
     'candidate': 'CDW', 'registration': 'observation',
     'required_evidence': 'charge SF（连通）+ 电荷衰减',
     'evidence_observables': [OBSERVABLE_CHARGE_STRUCTURE_FACTOR],
     'data_availability': '连通电荷 SF：N200 四 doping 全有；电荷衰减本'
                          '报告 v1 未纳入（装载器无先例）',
     'report_inclusion': 'partial', 'status': '证据具备',
     'note': '观测级规则：峰值数值不判定 CDW；定义非连通被纪律门拒绝',
     'doc_quoted': True},
    {'criterion_id': 'sdw_v1', 'section_ref': '§6 sdw_v1',
     'candidate': 'SDW', 'registration': 'observation',
     'required_evidence': '自旋 SF + 峰值 + 自旋衰减',
     'evidence_observables': [OBSERVABLE_SPIN_STRUCTURE_FACTOR],
     'data_availability': '自旋 SF（sz_sz 源点 0 + spin_dot 源点 49/50）'
                          '与峰值分析：全有；自旋衰减本报告 v1 未纳入',
     'report_inclusion': 'partial', 'status': '证据具备',
     'note': '观测级规则：q 位置不解释为 AFM/Γ/K；⟨Sz⟩≈0 不排除 SDW',
     'doc_quoted': True},
    {'criterion_id': 'pdw_v1', 'section_ref': '§6 pdw_v1',
     'candidate': 'PDW', 'registration': 'capability_boundary',
     'required_evidence': '多参考键配对矩阵',
     'evidence_observables': [OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR],
     'data_availability': '单参考键受限数据（P(q) 仅一行）；多参考键矩阵'
                          '不存在',
     'report_inclusion': 'included', 'status': '不可测量',
     'note': '能力边界规则：PDW 空间调制序判定恒 not_measurable',
     'doc_quoted': True},
    {'criterion_id': 'stripe_v1', 'section_ref': '§6 stripe_v1',
     'candidate': 'Stripe', 'registration': 'capability_boundary',
     'required_evidence': 'nf + 电荷 SF + 自旋 SF + 同口径多尺寸序列',
     'evidence_observables': [OBSERVABLE_LOCAL_DENSITY,
                              OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                              OBSERVABLE_SPIN_STRUCTURE_FACTOR],
     'data_availability': '单尺寸数据（N200 与 N400 不可混组）；无同口径'
                          '多尺寸序列',
     'report_inclusion': 'partial', 'status': '不可测量',
     'note': '能力边界规则：体相条纹序判定恒 not_measurable（O5 审查）',
     'doc_quoted': True},
    # ---- P1-G.28 有限尺寸分析能力诊断（非相候选；GAPW 规则注释为源） ----
    {'criterion_id': 'gap_v1', 'section_ref': 'GAPW 规则（P1-G.28）',
     'candidate': 'gap_scaling（分析能力）', 'registration':
     'capability_boundary',
     'required_evidence': '激发态能量或粒子数扇区能量差',
     'evidence_observables': [OBSERVABLE_ENERGY],
     'data_availability': 'Data_Energy 30 份日志均为单扇区基态（无激发态'
                          '/扇区分辨信息）',
     'report_inclusion': 'opt_in', 'status': '不可测量',
     'note': '禁止从基态总能随 doping/N 变化构造能隙；not_measurable ≠ '
             '能隙为零/相被排除',
     'doc_quoted': False},
    {'criterion_id': 'width_v1', 'section_ref': 'GAPW 规则（P1-G.28）',
     'candidate': 'width_scaling（分析能力）', 'registration':
     'capability_boundary',
     'required_evidence': '可比多宽度序列（多个 y 方向宽度）',
     'evidence_observables': [],
     'data_availability': 'XC 构型 y 恒 2 胞宽；N 变化是长度方向，不构成'
                          '宽度扫描',
     'report_inclusion': 'not_applicable', 'status': '不可测量',
     'note': 'not_measurable ≠ 宽度依赖为零/物理行为不存在',
     'doc_quoted': False},
]

DISCIPLINE_STATEMENT = (
    '> 纪律声明：本报告是受控证据汇编，不是物理论文结论。软件正确性 ≠ '
    '物理正确性；测试通过 ≠ 物理证明。本报告未注册任何相级支持/反证规则'
    '（sc_lro_v1 冻结），候选 verdict 仅来自已注册规则：观测级规则不产生'
    '支持/反证，能力边界恒 not_measurable——insufficient ≠ 反证、'
    'quality_failed ≠ 否定、not_measurable ≠ 物理不存在。相态判定需要'
    '负责人批准的物理判据与达标证据。')


# ==========================================================================
# 数据收集区（每节独立；失败不阻断其他节）
# ==========================================================================
def _make_outcome(context, report=None, failure=None):
    """与 Cross_Doping_Summary.analyze_doping 的 outcome 同构。"""
    return {'context': context, 'report': report, 'failure': failure}


def run_core_section(doping, root, t_j, bond_dim, N, a, q_list, refbonds):
    """核心四类（SC 衰减/SC 配对 SF/自旋 SF/电荷 SF）复用 analyze_doping。"""
    try:
        outcomes = cds.analyze_doping(
            doping, root=root, t_j=t_j, bond_dim=bond_dim, N=N, a=a,
            q_list=q_list, refbonds=refbonds,
            only_observables={OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                              OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                              OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                              OBSERVABLE_CHARGE_STRUCTURE_FACTOR})
        return outcomes, None
    except Exception as ex:  # noqa: BLE001 数据收集节级隔离，记录不阻断
        return [], {'type': type(ex).__name__, 'message': str(ex)}


def run_energy_section(doping, energy_root, t_j, j_value):
    """Data_Energy 各 N 的基态日志 → EnergyRecord → E∞ 外推。"""
    outcomes = []
    base = os.path.join(energy_root, t_j, doping)
    if not os.path.isdir(base):
        return outcomes, {'type': 'FileNotFoundError',
                          'message': f'能量目录不存在: {base}'}
    records = []
    for n in (80, 160, 240, 320, 400):
        ndir = os.path.join(base, f'N{n}')
        ctx = {'doping': doping, 'N': n, 'bond_dim': '-',
               'observable': OBSERVABLE_ENERGY, 'analyzer': ANALYZER_ENERGY,
               'refbond_or_source': f'N{n}', 'key_or_definition': 'E0'}
        logs = sorted(os.path.join(ndir, f) for f in os.listdir(ndir)
                      if f.endswith('.log')) if os.path.isdir(ndir) else []
        if not logs:
            outcomes.append(_make_outcome(ctx, failure={
                'type': 'FileNotFoundError',
                'message': f'无优化日志: {ndir}'}))
            continue
        try:
            text = open(logs[0], encoding='utf-8', errors='replace').read()
            rec = EnergyRecord.extract_data(text, N=n, J=j_value)
            records.append(rec)
        except Exception as ex:  # noqa: BLE001
            outcomes.append(_make_outcome(ctx, failure={
                'type': type(ex).__name__, 'message': str(ex)}))
    if len(records) >= 3:
        ctx = {'doping': doping, 'N': 'extrap', 'bond_dim': '-',
               'observable': OBSERVABLE_ENERGY, 'analyzer': ANALYZER_ENERGY,
               'refbond_or_source': 'Einf', 'key_or_definition': 'E0'}
        try:
            outcomes.append(_make_outcome(ctx, report=run_analysis(
                OBSERVABLE_ENERGY, operation='extrapolate', records=records,
                power=1)))
        except Exception as ex:  # noqa: BLE001
            outcomes.append(_make_outcome(ctx, failure={
                'type': type(ex).__name__, 'message': str(ex)}))
    return outcomes, None


def run_density_section(doping, root, t_j, bond_dim, N, a):
    """密度 profile（nf.json；measure 版本目录按发现式定位）。"""
    phys_dir = os.path.join(root, t_j, doping,
                            f'{bond_dim}_measure_phys_qty')
    measures = sorted(os.path.basename(p) for p in
                      (os.path.join(phys_dir, d) for d in os.listdir(phys_dir)
                       if d.startswith('measure_'))
                      if os.path.isdir(p)) if os.path.isdir(phys_dir) else []
    if not measures:
        ctx = {'doping': doping, 'N': N, 'bond_dim': bond_dim,
               'observable': OBSERVABLE_LOCAL_DENSITY,
               'analyzer': ANALYZER_DENSITY,
               'refbond_or_source': '-', 'key_or_definition': 'nf'}
        return [_make_outcome(ctx, failure={'type': 'FileNotFoundError',
                                            'message': f'无 measure 目录: '
                                                       f'{phys_dir}'})], None
    nf_path = os.path.join(phys_dir, measures[0], 'nf.json')
    ctx = {'doping': doping, 'N': N, 'bond_dim': bond_dim,
           'observable': OBSERVABLE_LOCAL_DENSITY, 'analyzer': ANALYZER_DENSITY,
           'refbond_or_source': '-', 'key_or_definition': 'nf'}
    try:
        rep = run_analysis(OBSERVABLE_LOCAL_DENSITY, operation='profile',
                           nf_path=nf_path, N=N, a=a)
        return [_make_outcome(ctx, report=rep)], None
    except Exception as ex:  # noqa: BLE001
        return [_make_outcome(ctx, failure={'type': type(ex).__name__,
                                            'message': str(ex)})], None


def run_pairing_symmetry_section(doping, root, t_j, bond_dim, N):
    """配对对称性（D5；重，默认关闭）。"""
    ctx = {'doping': doping, 'N': N, 'bond_dim': bond_dim,
           'observable': OBSERVABLE_SC_PAIRING_SYMMETRY,
           'analyzer': ANALYZER_SC_PAIRING_SYMMETRY,
           'refbond_or_source': '-', 'key_or_definition': '-'}
    condition = {'doping': doping, 'N': N, 'bond_dim': bond_dim,
                 'super_corr_root': os.path.join(root, t_j, doping)}
    try:
        rep = run_analysis(OBSERVABLE_SC_PAIRING_SYMMETRY,
                           operation='analyze', condition=condition)
        return [_make_outcome(ctx, report=rep)], None
    except Exception as ex:  # noqa: BLE001
        return [_make_outcome(ctx, failure={'type': type(ex).__name__,
                                            'message': str(ex)})], None


# ==========================================================================
# 判据检查表：静态清单 + 本次运行采集计数
# ==========================================================================
def build_checklist(outcomes):
    """静态清单合并本次运行的实际采集结果（不改变静态 status 语义）。"""
    collected = {}
    for o in outcomes:
        ctx = o['context']
        key = (ctx['observable'], ctx['analyzer'])
        if o['report'] is not None:
            collected.setdefault(key, {'ok': 0, 'failed': 0})['ok'] += 1
        elif o.get('failure'):
            collected.setdefault(key, {'ok': 0, 'failed': 0})['failed'] += 1
    rows = []
    for item in CRITERIA_CHECKLIST:
        row = dict(item)
        row['collected'] = {
            obs: collected.get((obs, analyzer), {'ok': 0, 'failed': 0})
            for obs, analyzer in _observable_analyzer_pairs(item)}
        rows.append(row)
    return rows


def _observable_analyzer_pairs(item):
    """检查表 observables → (observable, analyzer) 候选对（尽量推断）。"""
    pairs = []
    for obs in item['evidence_observables']:
        if obs == OBSERVABLE_ENERGY:
            pairs.append((obs, ANALYZER_ENERGY))
        elif obs == OBSERVABLE_LOCAL_DENSITY:
            pairs.append((obs, ANALYZER_DENSITY))
        elif obs == OBSERVABLE_SC_PAIRING_SYMMETRY:
            pairs.append((obs, ANALYZER_SC_PAIRING_SYMMETRY))
        else:
            pairs.append((obs, None))   # analyzer 待运行结果回填
    return pairs


# ==========================================================================
# 绘图区（全部走既有函数；--no-plots 跳过；单图失败不阻断）
# ==========================================================================
def save_figures(outcomes, plots_dir, no_plots=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    refs = []
    if no_plots:
        return refs
    os.makedirs(plots_dir, exist_ok=True)

    # SC 衰减面板图（payload = DecayAnalysisResult）
    for o in outcomes:
        rep = o['report']
        if rep is None or rep.observable != OBSERVABLE_SUPERCONDUCTING_CORRELATION:
            continue
        if rep.payload is None:
            continue
        name = (f"sc_decay_{o['context']['refbond_or_source']}_"
                f"{o['context']['key_or_definition']}")
        path = os.path.join(plots_dir, name + '.png')
        try:
            rep.payload.plot_panel_summary(save_path=path)
            refs.append(os.path.join('plots', name + '.png'))
        except Exception:  # noqa: BLE001
            plt.close('all')
    # 自旋 SF 峰值图（同源点身份配对的 estimate + peak_analysis）
    est = [o for o in outcomes
           if o['report'] is not None
           and o['report'].observable == OBSERVABLE_SPIN_STRUCTURE_FACTOR]
    est_by_src = {}
    for o in est:
        if o['context']['refbond_or_source'] not in est_by_src:
            est_by_src[o['context']['refbond_or_source']] = o
    for o in outcomes:
        rep = o['report']
        if rep is None or rep.analyzer != ANALYZER_PEAK:
            continue
        src = o['context']['refbond_or_source']
        pair = est_by_src.get(src)
        if pair is None or pair['report'].payload is None \
                or rep.payload is None:
            continue
        name = f"spin_peaks_{src}_{o['context']['key_or_definition']}"
        path = os.path.join(plots_dir, name + '.png')
        try:
            from Physical_Quantity.Spin_Structure_Factor_Analyzer import (
                SpinStructureFactorPeakAnalyzer)
            SpinStructureFactorPeakAnalyzer.plot_peaks(
                pair['report'].payload, rep.payload, save_path=path)
            refs.append(os.path.join('plots', name + '.png'))
        except Exception:  # noqa: BLE001
            plt.close('all')
    return refs


# ==========================================================================
# 渲染区（无相标签映射代码——结构性保证"不输出是 XX 相"）
# ==========================================================================
def _fmt(v):
    if isinstance(v, float):
        return f'{v:.6g}' if math.isfinite(v) else 'null'
    return str(v)


def render_markdown(config, outcomes, summary, checklist, figure_refs,
                    section_failures) -> str:
    phase_report = summary['phase_report']
    lines = []
    lines.append('# 受控证据报告：' + config['doping'])
    lines.append('')
    lines.append(DISCIPLINE_STATEMENT)
    lines.append('')
    lines.append('## 0. 运行配置')
    lines.append('')
    lines.append('| 项 | 值 |')
    lines.append('|---|---|')
    for k in ('root', 't_j', 'doping', 'N', 'bond_dim', 'a', 'n_qx',
              'n_qy', 'energy_root', 'with_pairing_symmetry',
              'with_density_profile', 'no_plots', 'generated_at'):
        lines.append(f'| {k} | {config.get(k)} |')
    lines.append('')
    lines.append('## 1. 逐观测量分析结果')
    lines.append('')
    for o in outcomes:
        ctx = o['context']
        obs, an = ctx['observable'], ctx['analyzer']
        lines.append(f"### {obs} | {an} | "
                     f"ref={ctx['refbond_or_source']} | "
                     f"key={ctx['key_or_definition']}")
        lines.append('')
        if o.get('failure'):
            lines.append(f"- **失败**：{o['failure']['type']}："
                         f"{o['failure']['message']}")
            lines.append('')
            continue
        rep = o['report']
        lines.append(f"- status={rep.status}，"
                     f"classification={rep.classification}，"
                     f"data_restricted={rep.data_restricted}")
        if rep.parameters:
            lines.append(f"- parameters：{_fmt(rep.parameters)}")
        if rep.metadata:
            meta = {k: v for k, v in rep.metadata.items()
                    if isinstance(v, (str, int, float, bool, list, dict))}
            if meta:
                lines.append(f"- metadata：{_fmt(meta)}")
        if rep.warnings:
            lines.append(f"- warnings：{rep.warnings}")
        lines.append('')
    lines.append('## 2. 候选 verdict（规则引擎，仅已注册规则）')
    lines.append('')
    vd = phase_report.candidate_verdicts
    lines.append('| candidate | verdict | reason |')
    lines.append('|---|---|---|')
    for cand, v in vd.items():
        lines.append(f'| {cand} | {v["verdict"]} | {v["reason"]} |')
    lines.append('')
    vs = phase_report.verdict_summary
    lines.append(f"- 汇总：{_fmt(vs)}")
    lines.append(f"- 总体 conclusion：{phase_report.conclusion}")
    lines.append(f"- 规则执行 {len(phase_report.rule_trace)} 条，"
                 f"完整轨迹见 report.json 的 phase_report.rule_trace")
    lines.append('')
    lines.append('## 3. 判据检查表')
    lines.append('')
    lines.append('| criterion | candidate | 状态 | 依据 |')
    lines.append('|---|---|---|---|')
    for row in checklist:
        lines.append(f"| {row['criterion_id']} | {row['candidate']} | "
                     f"{row['status']} | {row['data_availability']} |")
    lines.append('')
    lines.append('## 4. 图件')
    lines.append('')
    if figure_refs:
        for ref in figure_refs:
            lines.append(f'- ![图]({ref})')
    else:
        lines.append('- （未生成图件：--no-plots 或无可绘图结果）')
    lines.append('')
    lines.append('## 5. 结论')
    lines.append('')
    lines.append('1. 观测数值汇总：见第 1 节各条目（status/parameters/'
                 'metadata 逐项可追溯，机读底稿 report.json 含全部 '
                 'AnalysisReport 序列化）。')
    lines.append('2. 候选状态：见第 2 节 verdict 表。当前注册规则下，'
                 '观测级候选为 insufficient/data_missing，能力边界候选为 '
                 'not_measurable——**本报告不据此判定任何物理相态**。')
    lines.append('3. 判据完备性：见第 3 节检查表。sc_lro_v1 冻结未注册；'
                 'GapScaling/WidthScaling 算法未实现；相级判据注册属物理'
                 '决策，需负责人批准。')
    lines.append('')
    lines.append(DISCIPLINE_STATEMENT)
    lines.append('')
    if section_failures:
        lines.append('## 附录：节级收集失败（不阻断其他节）')
        lines.append('')
        for name, f in section_failures.items():
            lines.append(f'- {name}: {f["type"]}: {f["message"]}')
        lines.append('')
    return '\n'.join(lines)


def build_json(config, outcomes, summary, checklist, figure_refs,
               section_failures) -> dict:
    return {
        'generated_by': 'Phase_Analysis_Report',
        'config': config,
        'reports': [
            {'context': o['context'],
             'report': (o['report'].to_dict()
                        if o['report'] is not None else None),
             'failure': o['failure']}
            for o in outcomes],
        'rows': summary['rows'],
        'comparability_groups': summary['comparability_groups'],
        'phase_report': summary['phase_report'].to_dict(),
        'checklist': checklist,
        'figure_refs': figure_refs,
        'section_failures': section_failures,
        'semantics': {
            'discipline': ('受控证据汇编：只输出观测数值 + 已注册规则的'
                           '候选 verdict + 判据检查表；无相级支持/反证规则'
                           '注册，无相态结论'),
            'verdict_states': list(VERDICT_STATES),
            'authorized_rules': dict(AUTHORIZED_RULES),
            'checklist_statuses': list(CHECKLIST_STATUSES),
        },
    }


# ==========================================================================
# 主入口
# ==========================================================================
def build_report(config) -> dict:
    """核心入口（独立于 main，便于测试）：返回完整报告 dict。"""
    outcomes = []
    section_failures = {}

    q_list, _q_meta = xc_momentum_grid(
        config['N'], a=config['a'], n_qx=config['n_qx'], n_qy=config['n_qy'])
    core, fail = run_core_section(
        config['doping'], config['root'], config['t_j'],
        config['bond_dim'], config['N'], config['a'], q_list,
        config['refbonds'])
    outcomes.extend(core)
    if fail:
        section_failures['core'] = fail

    if config.get('energy_root'):
        eng, fail = run_energy_section(config['doping'],
                                       config['energy_root'],
                                       config['t_j'], config['j_value'])
        outcomes.extend(eng)
        if fail:
            section_failures['energy'] = fail
    if config.get('with_density_profile'):
        den, _ = run_density_section(config['doping'], config['root'],
                                     config['t_j'], config['bond_dim'],
                                     config['N'], config['a'])
        outcomes.extend(den)
    if config.get('with_pairing_symmetry'):
        ps, fail = run_pairing_symmetry_section(
            config['doping'], config['root'], config['t_j'],
            config['bond_dim'], config['N'])
        outcomes.extend(ps)
        if fail:
            section_failures['pairing_symmetry'] = fail

    summary = cds.summarize(outcomes, q_list=q_list, config=config)
    # 规则引擎：显式传入已注册规则集（P1F3 + GAPW），授权来自注册表
    run_candidate_rules(summary['phase_report'],
                        list(P1F3_RULES) + list(GAPW_RULES))
    checklist = build_checklist(outcomes)
    figure_refs = save_figures(outcomes, os.path.join(config['outdir'],
                                                      'plots'),
                               no_plots=config.get('no_plots', False))
    md = render_markdown(config, outcomes, summary, checklist, figure_refs,
                         section_failures)
    return {'markdown': md,
            'json': build_json(config, outcomes, summary, checklist,
                               figure_refs, section_failures),
            'summary': summary, 'outcomes': outcomes}


def write_report(result, outdir) -> dict:
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, 'report.md')
    json_path = os.path.join(outdir, 'report.json')
    with open(md_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(result['markdown'])
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result['json'], f, ensure_ascii=False, indent=2,
                  allow_nan=False)
    return {'report.md': md_path, 'report.json': json_path}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='一键受控证据报告（单 doping 单参数组；不判相，'
                    '详见脚本 docstring）')
    parser.add_argument('--root', default=os.path.join(SCRIPT_DIR,
                                                       'Data_N200'))
    parser.add_argument('--t-j', default=DEFAULT_T_J)
    parser.add_argument('--doping', required=True,
                        help='单个 doping（如 d1_100）；不自动混 doping')
    parser.add_argument('--N', type=int, default=DEFAULT_N)
    parser.add_argument('--bond-dim', default=DEFAULT_BOND_DIM)
    parser.add_argument('--refbonds', default=None,
                        help='逗号分隔过滤（如 49_50,50_51）；默认全部发现')
    parser.add_argument('--n-qx', type=int, default=41)
    parser.add_argument('--n-qy', type=int, default=4)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--energy-root', default=None,
                        help='提供时运行能量外推（如 Data_Energy）')
    parser.add_argument('--j-value', type=float, default=3.0,
                        help='t-J 模型的 J 值（EnergyRecord 封装用）')
    parser.add_argument('--with-pairing-symmetry', action='store_true',
                        help='运行配对对称性分析（重，默认关闭）')
    parser.add_argument('--with-density-profile', action='store_true',
                        help='运行密度 profile 分析（默认关闭）')
    parser.add_argument('--no-plots', action='store_true',
                        help='跳过绘图')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)

    refbonds = ([s.strip() for s in args.refbonds.split(',') if s.strip()]
                if args.refbonds else None)
    config = {
        'root': args.root, 't_j': args.t_j, 'doping': args.doping,
        'N': args.N, 'bond_dim': args.bond_dim, 'refbonds': refbonds,
        'n_qx': args.n_qx, 'n_qy': args.n_qy, 'a': args.a,
        'energy_root': args.energy_root, 'j_value': args.j_value,
        'with_pairing_symmetry': args.with_pairing_symmetry,
        'with_density_profile': args.with_density_profile,
        'no_plots': args.no_plots, 'outdir': args.outdir,
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    result = build_report(config)
    paths = write_report(result, args.outdir)
    n_ok = sum(1 for o in result['outcomes'] if o['report'] is not None)
    n_fail = sum(1 for o in result['outcomes'] if o.get('failure'))
    print(f'分析完成：{n_ok} 项成功 / {n_fail} 项失败')
    print(f"输出: {paths['report.md']}")
    print(f"      {paths['report.json']}")
    print(f'候选 verdict 数: '
          f"{len(result['summary']['phase_report'].candidate_verdicts)}，"
          f"conclusion: {result['summary']['phase_report'].conclusion}")
    print('说明: 本报告只输出观测数值与已注册规则的候选 verdict，不判定'
          '物理相态；纪律声明见 report.md 头部。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
