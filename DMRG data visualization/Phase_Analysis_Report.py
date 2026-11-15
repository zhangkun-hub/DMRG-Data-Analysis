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
        --energy-root Data/real/Data_Energy --with-density-profile
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import warnings

# 批量出图（Agg 后端）下已知无害的绘图警告：plt.show() 无窗口可弹、
# 能量图的顶部次坐标轴与 tight_layout 不兼容——图均已通过 savefig 保存，
# 过滤掉以免刷屏；其余警告照常显示。
warnings.filterwarnings('ignore',
                        message='FigureCanvasAgg is non-interactive.*')
warnings.filterwarnings('ignore',
                        message='This figure includes Axes that are not '
                                'compatible with tight_layout.*')

# 绘图纪律：必须在 import 包模块（热导入 matplotlib）之前设定
os.environ.setdefault('MPLBACKEND', 'Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Physical_Quantity.data_records.Physical_Data_Record import DensityRecord  # noqa: E402
from Physical_Quantity.evidence_integration.Analysis_Contract import (  # noqa: E402
    OBSERVABLE_SUPERCONDUCTING_CORRELATION,
    OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
    OBSERVABLE_SPIN_STRUCTURE_FACTOR,
    OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
    OBSERVABLE_ENERGY,
    OBSERVABLE_SC_PAIRING_SYMMETRY,
    OBSERVABLE_LOCAL_DENSITY,
    OBSERVABLE_LOCAL_SZ,
    OBSERVABLE_PAIRING_G2_COMPARISON,
    ANALYZER_ENERGY,
    ANALYZER_DENSITY,
    ANALYZER_SZ_PROFILE,
    ANALYZER_PAIRING_G2,
    ANALYZER_SC_PAIRING_SYMMETRY,
    ANALYZER_PEAK)
from Physical_Quantity.evidence_integration.Analysis_Entry import run_analysis  # noqa: E402
from Physical_Quantity.phase_diagnosis.Phase_Diagnosis import (  # noqa: E402
    collect_evidence, audit_comparability, evidence_sufficiency_check,
    run_candidate_rules, P1F3_RULES, GAPW_RULES, AUTHORIZED_RULES,
    VERDICT_STATES)
from Physical_Quantity.physical_analysis.Spin_Structure_Factor import xc_momentum_grid  # noqa: E402
from Physical_Quantity.physical_analysis.Density_Process import DensityProcessor  # noqa: E402
from Physical_Quantity.physical_analysis.Energy_Process import EnergyRecord  # noqa: E402
from Physical_Quantity.physical_analysis.Convergence_Check import (  # noqa: E402
    convergence_verdict, VERDICTS)
from Physical_Quantity.physical_analysis.Correlation_Process import (  # noqa: E402
    CorrelationProcessor)
from Physical_Quantity.physical_analysis.Super_Correlation_Process import (  # noqa: E402
    SuperCorrProcessor, _refbond_direction)
from Physical_Quantity.data_records.DMRG_Log_Parser import (  # noqa: E402
    DMRGLogParser)
from Physical_Quantity.data_records.Physical_Data_Record import (  # noqa: E402
    CorrelationRecord, SuperCorrRecord, SingleParticleRecord)
from Physical_Quantity.physical_analysis.Correlation_Decay_Analyzer import (  # noqa: E402
    AnalyzerConfig)
import Cross_Doping_Summary as cds  # noqa: E402

DEFAULT_T_J = 't_J_3'
DEFAULT_BOND_DIM = '1500'
DEFAULT_N = 200
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'phase_analysis_report')

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
     'note': '观测级规则：不产生相级支持/反证；幂律衰减不确立配对；'
             '配对 vs G² 斜率比较（--with-g2-comparison）是参照基准数值'
             '诊断（不判相，阈值未授权）',
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
                          '与峰值分析：全有；局域 ⟨S^z_i⟩（sz.json）报告'
                          '接口已补齐（--with-sz-profile，观测层，未接入'
                          '规则）；自旋衰减本报告 v1 未纳入',
     'report_inclusion': 'partial', 'status': '证据具备',
     'note': '观测级规则：q 位置不解释为 AFM/Γ/K；⟨Sz⟩≈0 不排除 SDW；'
             'A/B 子格磁化差是数值观测，不判 AFM/SDW',
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
     'data_availability': 'Data/real/Data_Energy 30 份日志均为单扇区基态（无激发态'
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
    '负责人批准的物理判据与达标证据。SC 衰减参数误差兜底 bootstrap 用'
    '轻量次数 30（默认 200，耗时约 1/7）：误差估计更粗；分类结论不依赖'
    'bootstrap，不受影响。')


# ==========================================================================
# 数据收集区（每节独立；失败不阻断其他节）
# ==========================================================================
def _make_outcome(context, report=None, failure=None):
    """与 Cross_Doping_Summary.analyze_doping 的 outcome 同构。"""
    return {'context': context, 'report': report, 'failure': failure}


def run_core_section(doping, root, t_j, bond_dim, N, a, q_list, refbonds):
    """核心四类（SC 衰减/SC 配对 SF/自旋 SF/电荷 SF）复用 analyze_doping。

    SC 衰减用轻量 bootstrap（n=30，默认 200）：bootstrap 仅作 pcov 失效时
    的误差兜底、不参与分类与模型选择，轻量化只让参数误差更粗（与收敛
    报告的轻量配置一致），耗时约 1/7。
    """
    try:
        fast_config = AnalyzerConfig(bootstrap_n=30)
        outcomes = cds.analyze_doping(
            doping, root=root, t_j=t_j, bond_dim=bond_dim, N=N, a=a,
            q_list=q_list, refbonds=refbonds,
            only_observables={OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                              OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                              OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                              OBSERVABLE_CHARGE_STRUCTURE_FACTOR},
            config=fast_config)
        return outcomes, None
    except Exception as ex:  # noqa: BLE001 数据收集节级隔离，记录不阻断
        return [], {'type': type(ex).__name__, 'message': str(ex)}


def run_energy_section(doping, energy_root, t_j, j_value):
    """Data/real/Data_Energy 各 N 的基态日志 → EnergyRecord → E∞ 外推。"""
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


def run_g2_section(doping, root, t_j, bond_dim, N, a):
    """配对关联 vs 单粒子关联平方比较（每个 refbond × 键方向一行）。

    数值诊断（不判相）：SC 合成 ⟨Δ†Δ⟩（SuperCorrRecord）对比
    单粒子 G(r)（single_particle_corr2 的 ↓ 分量，corr2/corr4 等价）。
    """
    outcomes = []
    super_root = os.path.join(root, t_j, doping,
                              f'{bond_dim}_measure_super_corr')
    phys_dir = os.path.join(root, t_j, doping,
                            f'{bond_dim}_measure_phys_qty')
    measures = sorted(os.path.basename(p) for p in
                      (os.path.join(phys_dir, d) for d in os.listdir(phys_dir)
                       if d.startswith('measure_'))
                      if os.path.isdir(p)) if os.path.isdir(phys_dir) else []
    g_path = os.path.join(phys_dir, measures[0],
                          'single_particle_corr2.json') if measures else None
    if g_path is None or not os.path.isfile(g_path):
        return [], {'type': 'FileNotFoundError',
                    'message': f'无单粒子关联文件: '
                               f'{phys_dir}/measure_*/single_particle_corr2.json'}
    try:
        g_recs = SingleParticleRecord.extract_data(
            open(g_path, encoding='utf-8').read(), N, a)
    except Exception as ex:  # noqa: BLE001
        return [], {'type': type(ex).__name__, 'message': str(ex)}
    if not os.path.isdir(super_root):
        return [], {'type': 'FileNotFoundError',
                    'message': f'无超导目录: {super_root}'}
    for refdir in sorted(os.listdir(super_root)):
        m = re.match(r'super_corr_(\d+)_(\d+)$', refdir)
        if not m:
            continue
        refbond = (int(m.group(1)), int(m.group(2)))
        for key in 'ABC':
            ctx = {'doping': doping, 'N': N, 'bond_dim': bond_dim,
                   'observable': OBSERVABLE_PAIRING_G2_COMPARISON,
                   'analyzer': ANALYZER_PAIRING_G2,
                   'refbond_or_source': refdir,
                   'key_or_definition': key}
            try:
                pairing_recs = cds.load_super_corr_records(
                    os.path.join(super_root, refdir), refbond, key, N, a)
                rep = run_analysis(OBSERVABLE_PAIRING_G2_COMPARISON,
                                   operation='compare',
                                   pairing_recs=pairing_recs, g_recs=g_recs)
                outcomes.append(_make_outcome(ctx, report=rep))
            except Exception as ex:  # noqa: BLE001
                outcomes.append(_make_outcome(ctx, failure={
                    'type': type(ex).__name__, 'message': str(ex)}))
    return outcomes, None


def run_sz_section(doping, root, t_j, bond_dim, N, a):
    """局域自旋 profile（sz.json 单点 ⟨S^z_i⟩ 测量；观测层，不判相）。"""
    phys_dir = os.path.join(root, t_j, doping,
                            f'{bond_dim}_measure_phys_qty')
    measures = sorted(os.path.basename(p) for p in
                      (os.path.join(phys_dir, d) for d in os.listdir(phys_dir)
                       if d.startswith('measure_'))
                      if os.path.isdir(p)) if os.path.isdir(phys_dir) else []
    ctx = {'doping': doping, 'N': N, 'bond_dim': bond_dim,
           'observable': OBSERVABLE_LOCAL_SZ, 'analyzer': ANALYZER_SZ_PROFILE,
           'refbond_or_source': '-', 'key_or_definition': 'sz'}
    if not measures:
        return [_make_outcome(ctx, failure={'type': 'FileNotFoundError',
                                            'message': f'无 measure 目录: '
                                                       f'{phys_dir}'})], None
    sz_path = os.path.join(phys_dir, measures[0], 'sz.json')
    try:
        rep = run_analysis(OBSERVABLE_LOCAL_SZ, operation='profile',
                           sz_path=sz_path, N=N, a=a)
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
# 收敛诊断节（--convergence-dims）
# ==========================================================================
_E0_RE = re.compile(r'E0\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)')
_TE_RE = re.compile(r'TruncErr\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)')


def _find_log_for_dim(doping_dir, dim):
    """doping 目录下含该 D 档的优化日志（slurm-*-Zk_tJ.log）。

    匹配真实日志的阶段头行格式（如 'Bond dimension:\\t 1500/1500'）。
    """
    logs = [f for f in os.listdir(doping_dir)
            if f.endswith('.log') and 'Zk_tJ' in f]
    for name in logs:
        text = open(os.path.join(doping_dir, name), encoding='utf-8',
                    errors='replace').read()
        if f'{dim}/{dim}' in text:
            return os.path.join(doping_dir, name), text
    return None, None


def _su2_max_residual(super_root, dim, N, a):
    """该 D 档全部 refbond × 键方向的 SU(2) 通道残差最大值。"""
    root = os.path.join(super_root, f'{dim}_measure_super_corr')
    worst = 0.0
    n = 0
    if not os.path.isdir(root):
        return None
    for refdir in sorted(os.listdir(root)):
        m = re.match(r'super_corr_(\d+)_(\d+)$', refdir)
        if not m:
            continue
        refbond = (int(m.group(1)), int(m.group(2)))
        base = os.path.join(root, refdir)
        for key in 'ABC':
            recs = []
            for ch in 'abcd':
                path = os.path.join(
                    base, f'scs{key}{ch}_fix_refbond_{refbond[0]}'
                          f'_{refbond[1]}.json')
                if not os.path.isfile(path):
                    break
                recs.append(SuperCorrRecord.extract_data(
                    open(path, encoding='utf-8').read(), N, a))
            if len(recs) != 4:
                continue
            err_ad, err_bc = SuperCorrProcessor.symmetry_check(*recs)
            worst = max(worst, err_ad, err_bc)
            n += 1
    return worst if n else None


def _sc_params_by_dim(doping, root, t_j, dim, N, a, q_list, refbonds,
                      analyzer_config=None):
    """该 D 档 SC 衰减的分类组合与 (refbond,key)→(参数,误差)。

    analyzer_config：透传的 AnalyzerConfig（收敛报告用轻量 bootstrap
    配置提速；None 用分析器默认）。分类结论不依赖 bootstrap（bootstrap
    仅作 pcov 失效时的误差兜底），轻量配置只让参数误差更粗。
    """
    try:
        outs = cds.analyze_doping(
            doping, root=root, t_j=t_j, bond_dim=dim, N=N, a=a,
            q_list=q_list, refbonds=refbonds,
            only_observables={OBSERVABLE_SUPERCONDUCTING_CORRELATION},
            config=analyzer_config)
    except Exception:  # noqa: BLE001
        return None, None, None, None
    classes, detail, params, errs = [], {}, {}, {}
    for o in outs:
        rep = o['report']
        if rep is None or rep.observable != OBSERVABLE_SUPERCONDUCTING_CORRELATION:
            continue
        ctx = o['context']
        tag = f"{ctx['refbond_or_source']}_{ctx['key_or_definition']}"
        classes.append((tag, rep.classification))
        detail[tag] = rep.classification
        p, m = rep.parameters or {}, rep.metadata or {}
        pe = m.get('param_errors') or {}
        if p.get('K') is not None:
            params[f'{tag}_K'], errs[f'{tag}_K'] = p['K'], pe.get('K')
        elif p.get('xi') is not None:
            params[f'{tag}_xi'], errs[f'{tag}_xi'] = p['xi'], pe.get('xi')
    classes.sort()
    return (tuple(c for _, c in classes), params, errs,
            dict(sorted(detail.items())))


def run_convergence_section(config, dims):
    """收敛诊断：五项检查输入 + verdict + 证据图。

    返回 dict：inputs / verdict / figures / failures。
    证据图：能量收敛图 + 密度/自旋/超导两档 |差值| 图。
    阈值来源：Convergence_Check.DEFAULT_THRESHOLDS（未获物理授权）。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root, t_j, doping = config['root'], config['t_j'], config['doping']
    N, a = config['N'], config['a']
    doping_dir = os.path.join(root, t_j, doping)
    q_list, _ = xc_momentum_grid(N, a=a, n_qx=config['n_qx'],
                                 n_qy=config['n_qy'])
    e0_by_dim, te_by_dim, su2_by_dim = {}, {}, {}
    classes_by_dim, classes_detail_by_dim = {}, {}
    params_by_dim, errs_by_dim = {}, {}
    figures, failures = [], []
    dims = [str(d) for d in dims]
    # 轻量配置：bootstrap 仅作误差兜底且不参与分类/模型选择，收敛报告
    # 把 200 次降到 30 次（耗时 ~1/7，误差估计更粗）。分类结论不受影响。
    fast_config = AnalyzerConfig(bootstrap_n=30)

    for dim in dims:
        # E0 / TruncErr（优化日志）
        log_path, text = _find_log_for_dim(doping_dir, dim)
        if text is None:
            failures.append({'dim': dim, 'type': 'FileNotFoundError',
                             'message': f'无含 D={dim} 的优化日志'})
        else:
            e0s = [float(v) for v in _E0_RE.findall(text)]
            tes = [float(v) for v in _TE_RE.findall(text)]
            if e0s:
                e0_by_dim[dim] = e0s[-1]
            if tes:
                te_by_dim[dim] = max(tes[-5:])
        # SU(2) 残差
        su2_res = _su2_max_residual(os.path.join(root, t_j, doping),
                                    dim, N, a)
        if su2_res is not None:
            su2_by_dim[dim] = su2_res
        # SC 分类/参数
        cl, ps, es, detail = _sc_params_by_dim(
            doping, root, t_j, dim, N, a, q_list, config['refbonds'],
            analyzer_config=fast_config)
        if cl is not None:
            classes_by_dim[dim] = cl
        if detail is not None:
            classes_detail_by_dim[dim] = detail
        if ps:
            params_by_dim[dim], errs_by_dim[dim] = ps, es

    result = convergence_verdict(
        e0_by_dim=e0_by_dim or None, truncerr_by_dim=te_by_dim or None,
        su2_by_dim=su2_by_dim or None, classes_by_dim=classes_by_dim or None,
        params_by_dim=params_by_dim or None,
        param_errors_by_dim=errs_by_dim or None)

    # ---- 证据图（每图失败不阻断） ----
    if not config.get('no_plots') and len(dims) >= 2:
        d1, d2 = dims[0], dims[1]
        # 能量收敛图（合并两档日志 → DMRGLogParser：2 行子图，上为 E0、
        # 下为 |ΔE| 半对数，同存于 convergence_energy.png）
        try:
            parser = DMRGLogParser()
            texts = []
            for dim in dims:
                _, t = _find_log_for_dim(doping_dir, dim)
                if t:
                    texts.append(t)
            if texts:
                parser.parse_from_string('\n'.join(texts))
                parser.plot_DMRG_energy_convergence_analysis()
                path = os.path.join(config['outdir'], 'plots',
                                    'convergence_energy.png')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                plt.savefig(path)
                plt.close('all')
                figures.append(os.path.join('plots',
                                            'convergence_energy.png'))
        except Exception:  # noqa: BLE001
            plt.close('all')
        # 密度两档差值
        try:
            recs = {}
            for dim in (d1, d2):
                phys = os.path.join(root, t_j, doping,
                                    f'{dim}_measure_phys_qty')
                measures = [os.path.join(phys, d) for d in os.listdir(phys)
                            if d.startswith('measure_')]
                nf = os.path.join(measures[0], 'nf.json')
                recs[f'D{dim}'] = DensityRecord.extract_data(
                    open(nf, encoding='utf-8').read(), N, a)
            DensityProcessor.Line_Chart_Visualization_difference(recs)
            path = os.path.join(config['outdir'], 'plots',
                                'convergence_density_diff.png')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close('all')
            figures.append(os.path.join('plots',
                                        'convergence_density_diff.png'))
        except Exception:  # noqa: BLE001
            plt.close('all')
        # 自旋两档差值（A/B 子格分两行，组名即列标题 D = dim）
        try:
            sp = {}
            for dim in (d1, d2):
                phys = os.path.join(root, t_j, doping,
                                    f'{dim}_measure_phys_qty')
                recs = cds.load_spin_dot_records(phys, N, a)
                sp[f'D = {dim}'] = {
                    'A sublattice': [r for r in recs
                                     if r.source_point % 4 == 1],
                    'B sublattice': [r for r in recs
                                     if r.source_point % 4 == 2],
                }
            CorrelationProcessor.Spin_Corr_Visualization_difference(sp)
            path = os.path.join(config['outdir'], 'plots',
                                'convergence_spin_diff.png')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close('all')
            figures.append(os.path.join('plots',
                                        'convergence_spin_diff.png'))
        except Exception:  # noqa: BLE001
            plt.close('all')
        # 超导两档差值（行 = 3 个 refbond 键方向，列 = 两档 + 差值，
        # 每格内 target A/B/C 三键同图）
        try:
            sc = {}
            for dim in (d1, d2):
                root_dir = os.path.join(root, t_j, doping,
                                        f'{dim}_measure_super_corr')
                refdirs = [d for d in sorted(os.listdir(root_dir))
                           if re.match(r'super_corr_\d+_\d+$', d)]
                if not refdirs:
                    break
                labels = {'A': 'A key (blue bond)', 'B': 'B key (green bond)',
                          'C': 'C key (orange bond)'}
                per_ref = {}
                for rd in refdirs:
                    m = re.match(r'super_corr_(\d+)_(\d+)$', rd)
                    refbond = (int(m.group(1)), int(m.group(2)))
                    ref_key = _refbond_direction(*refbond)
                    ref_tag = f'reference {ref_key}: {refbond}'
                    per_ref[ref_tag] = {
                        label: cds.load_super_corr_records(
                            os.path.join(root_dir, rd), refbond, k, N, a)
                        for k, label in labels.items()}
                sc[f'D = {dim}'] = per_ref
            if len(sc) == 2:
                SuperCorrProcessor.Super_Corr_Visualization_difference_refbond(
                    sc)
                path = os.path.join(config['outdir'], 'plots',
                                    'convergence_supercorr_diff.png')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                plt.savefig(path)
                plt.close('all')
                figures.append(os.path.join('plots',
                                            'convergence_supercorr_diff.png'))
        except Exception:  # noqa: BLE001
            plt.close('all')

    return {'dims': dims, 'inputs': {
                'e0_by_dim': e0_by_dim, 'truncerr_by_dim': te_by_dim,
                'su2_by_dim': su2_by_dim, 'classes_by_dim': classes_by_dim,
                'classes_detail_by_dim': classes_detail_by_dim,
            },
            'verdict': result, 'figures': figures, 'failures': failures}


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
            from Physical_Quantity.physical_analysis.Spin_Structure_Factor_Analyzer import (
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
    if isinstance(v, list):
        if len(v) <= 3:
            return '[' + ', '.join(_fmt(x) for x in v) + ']'
        return '[' + ', '.join(_fmt(x) for x in v[:3]) + ', …]'
    return str(v)


def _core_fields(rep):
    """按观测量类型提炼核心数值（人读表格用；完整数据见 report.json）。"""
    p = rep.parameters or {}
    m = rep.metadata or {}
    obs = rep.observable
    if obs == OBSERVABLE_SUPERCONDUCTING_CORRELATION:
        fq = m.get('fit_quality') or {}
        pe = m.get('param_errors') or {}
        # 幂律族有 K 指数；指数族/非单调有 ξ 关联长度——按实际参数键自适应
        if p.get('K') is not None:
            exp_rows = [('K 指数', p.get('K')), ('ΔK', pe.get('K'))]
        elif p.get('xi') is not None:
            exp_rows = [('ξ 关联长度', p.get('xi')), ('Δξ', pe.get('xi'))]
        else:
            exp_rows = []
        return ([('分类', rep.classification)] + exp_rows
                + [('A 振幅', p.get('A')), ('r²', fq.get('r2')),
                   ('rmse', fq.get('rmse')), ('稳定性', m.get('verdict')),
                   ('拟合点数', m.get('n_points'))])
    if rep.analyzer == ANALYZER_PEAK:
        return [('峰数', p.get('n_peaks')),
                ('峰位 q', p.get('peak_positions')),
                ('峰高', p.get('peak_heights'))]
    if obs in (OBSERVABLE_SPIN_STRUCTURE_FACTOR,
               OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
               OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR):
        rows = []
        if m.get('measured_pair_count') is not None:
            rows.append(('已测对数', m.get('measured_pair_count')))
        if m.get('source_coverage') is not None:
            rows.append(('源点覆盖', m.get('source_coverage')))
        if m.get('imaginary_residual') is not None:
            rows.append(('虚部残差', m.get('imaginary_residual')))
        if m.get('channel_target_consistency') is not None:
            rows.append(('通道一致性', m.get('channel_target_consistency')))
        if rows:
            rows.append(('受限', rep.data_restricted))
        return rows
    if obs == OBSERVABLE_ENERGY:
        return [('E∞', p.get('E_inf')), ('外推幂次', p.get('power')),
                ('N 序列', p.get('N_values')),
                ('异常点诊断', m.get('outlier_diagnosis') is not None)]
    if obs == OBSERVABLE_LOCAL_DENSITY:
        return [('四行均值', p.get('row_mean')),
                ('A/B 子格均值', p.get('sub_lattice_mean')),
                ('胞均值数', len(p.get('unit_cell_mean') or []))]
    if obs == OBSERVABLE_LOCAL_SZ:
        return [('A/B 子格均值', p.get('sub_lattice_mean')),
                ('子格磁化差', p.get('staggered_magnetization')),
                ('四行均值', p.get('row_mean'))]
    if obs == OBSERVABLE_PAIRING_G2_COMPARISON:
        # 存储为 log 斜率（负值）；表格显示衰减指数（正值）
        return [('结论', p.get('verdict')),
                ('K_sc 指数',
                 None if p.get('k_sc') is None else -p['k_sc']),
                ('2·K_G 指数',
                 None if p.get('k_g2') is None else -p['k_g2']),
                ('斜率差', p.get('slope_diff'))]
    if obs == OBSERVABLE_SC_PAIRING_SYMMETRY:
        return [('阶段状态', p.get('phase') or rep.classification),
                ('元素数', p.get('n_elements'))]
    # 通用兜底：顶层标量参数（截断）
    scalars = [(k, v) for k, v in p.items()
               if isinstance(v, (int, float, str, bool))]
    return ([(k, v) for k, v in scalars[:4]]
            + [('data_restricted', rep.data_restricted)])


def _summarize_outcome_table(outcomes, observable, analyzer=None):
    """同 (observable, analyzer) 的 outcome → 紧凑表格（行=outcome）。"""
    rows = [o for o in outcomes
            if o['context']['observable'] == observable
            and (analyzer is None or o['context']['analyzer'] == analyzer)]
    ok_rows = [o for o in rows if o['report'] is not None]
    if not ok_rows:
        return None
    field_names = [k for k, _ in _core_fields(ok_rows[0]['report'])]
    header = ['refbond/源点', '键/定义', 'status'] + field_names
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '---|' * len(header)]
    for o in rows:
        ctx = o['context']
        if o['report'] is None:
            f = o.get('failure') or {}
            cells = [str(ctx['refbond_or_source']),
                     str(ctx['key_or_definition']), 'failed',
                     f"{f.get('type', '?')}: {(f.get('message') or '')[:40]}"]
            cells += [''] * (len(field_names) - 1)
        else:
            vals = dict(_core_fields(o['report']))
            cells = ([str(ctx['refbond_or_source']),
                      str(ctx['key_or_definition']), o['report'].status]
                     + [_fmt(vals.get(k)) for k in field_names])
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


VERDICT_EXPLAIN = {
    'supported': '支持条件成立（当前注册规则集不会产生此值）',
    'contradicted': '正式反证成立（当前注册规则集不会产生此值）',
    'conflicting': '支持与反证同时成立（当前不会产生）',
    'insufficient': '证据不足以判断（观测级规则的正常状态，非反证）',
    'quality_failed': '数据质量门失败（数据不合格，不等于被否定）',
    'criteria_undefined': '判据未定义或规则未授权',
    'data_missing': '必需观测缺失',
    'analysis_unavailable': '分析执行异常，本次不可用',
    'not_measurable': '当前数据能力不足，无法测量（不代表物理不存在）',
}


def render_markdown(config, outcomes, summary, checklist, figure_refs,
                    section_failures, convergence=None) -> str:
    phase_report = summary['phase_report']
    n_ok = sum(1 for o in outcomes if o['report'] is not None)
    n_fail = sum(1 for o in outcomes if o.get('failure'))
    lines = []
    lines.append(f"# 受控证据报告：{config['doping']}")
    lines.append('')
    lines.append('> 纪律声明：本报告只汇总数值观测与已注册规则的候选状态，'
                 '**不判定物理相态**。软件正确性 ≠ 物理正确性；'
                 'not_measurable ≠ 物理不存在；insufficient ≠ 反证。')
    lines.append('')
    lines.append('## 报告导读')
    lines.append('')
    lines.append('- **第 1 节**：各物理量的数值分析结果速览（核心指标表；'
                 '完整参数与警告见 report.json）')
    lines.append('- **第 2 节**：候选状态表（规则引擎输出；每格附中文含义）')
    lines.append('- **第 3 节**：判据检查表（每条判据的证据具备/缺失/'
                 '不可测量状态）')
    lines.append('- **第 4 节**：图件；**第 5 节**：结论')
    lines.append('')
    lines.append('## 0. 运行配置')
    lines.append('')
    lines.append('| 项 | 值 |')
    lines.append('|---|---|')
    for k, label in (('doping', 'doping'), ('N', 'N'), ('bond_dim', 'bond_dim'),
                     ('t_j', 't_J'), ('root', '数据目录'),
                     ('energy_root', '能量外推目录'),
                     ('n_qx', 'q 网格 qx 数'), ('n_qy', 'q 网格 qy 数'),
                     ('a', '晶格常数 a'),
                     ('with_pairing_symmetry', '配对对称分析'),
                     ('with_density_profile', '密度 profile'),
                     ('no_plots', '跳过绘图'), ('generated_at', '生成时间')):
        lines.append(f'| {label} | {config.get(k)} |')
    lines.append('')
    lines.append('## 1. 分析结果速览')
    lines.append('')
    lines.append(f'共 {len(outcomes)} 项分析：**{n_ok} 项成功**，'
                 f'{n_fail} 项失败（失败项单独列出，不阻断其他分析）。')
    lines.append('')
    seen = set()
    for obs in (OBSERVABLE_SUPERCONDUCTING_CORRELATION,
                OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR,
                OBSERVABLE_SPIN_STRUCTURE_FACTOR,
                OBSERVABLE_CHARGE_STRUCTURE_FACTOR,
                OBSERVABLE_ENERGY,
                OBSERVABLE_LOCAL_DENSITY,
                OBSERVABLE_LOCAL_SZ,
                OBSERVABLE_PAIRING_G2_COMPARISON,
                OBSERVABLE_SC_PAIRING_SYMMETRY):
        analyzers = sorted({o['context']['analyzer'] for o in outcomes
                            if o['context']['observable'] == obs})
        for an in analyzers:
            key = (obs, an)
            if key in seen:
                continue
            seen.add(key)
            table = _summarize_outcome_table(outcomes, obs, analyzer=an)
            if not table:
                continue
            lines.append(f'### {obs}（{an}）')
            lines.append('')
            lines.append(table)
            lines.append('')
    failures = [o for o in outcomes if o.get('failure')]
    if failures:
        lines.append('### 失败项')
        lines.append('')
        lines.append('| 观测 | ref/源点 | 原因 |')
        lines.append('|---|---|---|')
        for o in failures:
            f = o['failure']
            lines.append(f"| {o['context']['observable']} | "
                         f"{o['context']['refbond_or_source']} | "
                         f"{f['type']}: {f['message']} |")
        lines.append('')
    lines.append('> 完整参数、误差、warnings 与配置快照见 report.json 的 '
                 'reports 数组。')
    lines.append('')
    lines.append('## 2. 候选状态（规则引擎，仅已注册规则）')
    lines.append('')
    vd = phase_report.candidate_verdicts
    lines.append('| 候选 | 状态 | 含义 |')
    lines.append('|---|---|---|')
    for cand, v in vd.items():
        lines.append(f'| {cand} | {v["verdict"]} | '
                     f'{VERDICT_EXPLAIN.get(v["verdict"], v["verdict"])} |')
    lines.append('')
    lines.append(f'- 总体 conclusion：{phase_report.conclusion}')
    lines.append(f"- 规则执行 {len(phase_report.rule_trace)} 条，"
                 f"完整轨迹见 report.json 的 phase_report.rule_trace")
    lines.append('')
    lines.append('## 3. 判据检查表')
    lines.append('')
    lines.append('| 判据 | 物理量 | 状态 | 数据现状 |')
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
    if convergence is not None:
        lines.append('## 5. 收敛诊断（bond dimension 稳定性）')
        lines.append('')
        cv = convergence['verdict']
        lines.append(f"**结论：{cv['verdict']}**——{cv['reason']}")
        lines.append('')
        lines.append('> 注意：本开关只回答"数值是否稳定、D 是否还需增加"，'
                     '**不回答能否判相**（判相另需尺寸序列等证据）；阈值为'
                     '默认值（threshold_status: '
                     f"{cv['threshold_status']}，未获物理授权）。")
        lines.append('')
        lines.append('| 检查项 | 结果 | 证据 |')
        lines.append('|---|---|---|')
        for c in cv['checks']:
            mark = '✅' if c['passed'] is True else ('❌' if c['passed']
                                                    is False else '—（证据不足）')
            lines.append(f"| {c['name']} | {mark} | {c['evidence']} |")
        lines.append('')
        for ref in convergence['figures']:
            lines.append(f'- ![收敛证据]({ref})')
        if convergence['failures']:
            lines.append('')
            lines.append('**收集失败项**：')
            for f in convergence['failures']:
                lines.append(f"- D={f['dim']}: {f['type']}: {f['message']}")
        lines.append('')
        lines.append('## 6. 结论')
    else:
        lines.append('## 5. 结论')
    lines.append('')
    lines.append(f'1. **观测结果**：{n_ok} 项分析成功完成，核心数值见'
                 '第 1 节速览表；全部拟合参数与质量诊断可追溯到 report.json。')
    lines.append('2. **候选状态**：见第 2 节表。当前注册规则集下，'
                 '观测级候选为 insufficient/data_missing（证据不足以判断），'
                 '能力边界候选为 not_measurable（当前数据不可测量）——'
                 '**本报告不据此判定任何物理相态**。')
    lines.append('3. **判据完备性**：见第 3 节检查表。sc_lro_v1 冻结未注册；'
                 'GapScaling/WidthScaling 算法未实现；相级判据注册属物理'
                 '决策，需负责人批准。')
    lines.append('')
    if section_failures:
        lines.append('## 附录：节级收集失败（不阻断其他节）')
        lines.append('')
        for name, f in section_failures.items():
            lines.append(f'- {name}: {f["type"]}: {f["message"]}')
        lines.append('')
    return '\n'.join(lines)


def build_json(config, outcomes, summary, checklist, figure_refs,
               section_failures, convergence=None) -> dict:
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
        'convergence': convergence,
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
    if config.get('with_sz_profile'):
        sz, _ = run_sz_section(config['doping'], config['root'],
                               config['t_j'], config['bond_dim'],
                               config['N'], config['a'])
        outcomes.extend(sz)
    if config.get('with_g2_comparison'):
        g2, fail = run_g2_section(config['doping'], config['root'],
                                  config['t_j'], config['bond_dim'],
                                  config['N'], config['a'])
        outcomes.extend(g2)
        if fail:
            section_failures['g2_comparison'] = fail
    if config.get('with_pairing_symmetry'):
        ps, fail = run_pairing_symmetry_section(
            config['doping'], config['root'], config['t_j'],
            config['bond_dim'], config['N'])
        outcomes.extend(ps)
        if fail:
            section_failures['pairing_symmetry'] = fail
    convergence = None
    if config.get('convergence_dims'):
        convergence = run_convergence_section(config,
                                              config['convergence_dims'])

    summary = cds.summarize(outcomes, q_list=q_list, config=config)
    # 规则引擎：显式传入已注册规则集（P1F3 + GAPW），授权来自注册表
    run_candidate_rules(summary['phase_report'],
                        list(P1F3_RULES) + list(GAPW_RULES))
    checklist = build_checklist(outcomes)
    figure_refs = save_figures(outcomes, os.path.join(config['outdir'],
                                                      'plots'),
                               no_plots=config.get('no_plots', False))
    md = render_markdown(config, outcomes, summary, checklist, figure_refs,
                         section_failures, convergence)
    return {'markdown': md,
            'json': build_json(config, outcomes, summary, checklist,
                               figure_refs, section_failures, convergence),
            'summary': summary, 'outcomes': outcomes,
            'convergence': convergence}


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


def _fraction_to_doping(value):
    """空穴率输入 → 目录名：'1_10' → 'd1_10'；'d1_10' 直通；'0.1' 分数解析。"""
    v = str(value).strip()
    if re.match(r'^d\d+_\d+$', v):
        return v
    m = re.match(r'^(\d+)_(\d+)$', v)
    if m:
        return f'd{m.group(1)}_{m.group(2)}'
    from fractions import Fraction
    try:
        f = Fraction(v).limit_denominator(10000)
    except (ValueError, ZeroDivisionError):
        return None
    if f.numerator <= 0:
        return None
    return f'd{f.numerator}_{f.denominator}'


def _prompt_config(n_explicit, t_j_explicit, doping_explicit):
    """缺省参数的交互式获取：N（默认 200）、t/J（默认 3）、
    空穴率（默认 1_10，即 δ=0.1 → d1_10）。

    非交互环境（stdin 无输入，如管道/子进程）下回退默认值，不报错。
    """
    def _ask(prompt, default):
        try:
            raw = input(prompt).strip()
        except EOFError:
            return default
        return raw if raw else default

    n = n_explicit
    if n is None:
        n = int(_ask('请输入格点数 N（回车使用默认 200）: ', '200'))
    t_j = t_j_explicit
    if t_j is None:
        t_j_val = float(_ask('请输入 t/J（回车使用默认 3）: ', '3'))
        t_j = f't_J_{t_j_val:g}'
    doping = doping_explicit
    if doping is None:
        raw = _ask('请输入空穴率（如 1_10、d1_10 或 0.10；'
                   '回车使用默认 1_10）: ', '1_10')
        doping = _fraction_to_doping(raw)
        if doping is None:
            raise ValueError(f'无法解析空穴率输入: {raw!r}（示例: 1_10、'
                             '0.1、0.05、d1_10）')
    return n, t_j, doping


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='一键受控证据报告（单 doping 单参数组；不判相，'
                    '详见脚本 docstring）')
    parser.add_argument('--root', default=os.path.join(SCRIPT_DIR,
                                                       'Data/real/Data_N200'))
    parser.add_argument('--t-j', default=None,
                        help='t/J 目录后缀（如 t_J_3）；不提供时交互式'
                             '提示输入 t/J 值（默认 3）')
    parser.add_argument('--doping', default=None,
                        help='单个 doping（如 d1_10 或空穴率 1_10 / 0.1）；'
                             '不提供时交互式提示输入（默认 1_10）')
    parser.add_argument('--N', type=int, default=None,
                        help='格点数（默认 200）；不提供时交互式提示输入')
    parser.add_argument('--bond-dim', default=DEFAULT_BOND_DIM)
    parser.add_argument('--refbonds', default=None,
                        help='逗号分隔过滤（如 49_50,50_51）；默认全部发现')
    parser.add_argument('--n-qx', type=int, default=41)
    parser.add_argument('--n-qy', type=int, default=4)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--energy-root', default=None,
                        help='提供时运行能量外推（如 Data/real/Data_Energy）')
    parser.add_argument('--j-value', type=float, default=3.0,
                        help='t-J 模型的 J 值（EnergyRecord 封装用）')
    parser.add_argument('--with-pairing-symmetry', action='store_true',
                        help='运行配对对称性分析（重，默认关闭）')
    parser.add_argument('--with-density-profile', action='store_true',
                        help='运行密度 profile 分析（默认关闭）')
    parser.add_argument('--with-sz-profile', action='store_true',
                        help='运行局域自旋 profile 分析（sz.json 单点'
                             '⟨S^z_i⟩，默认关闭）')
    parser.add_argument('--with-g2-comparison', action='store_true',
                        help='运行配对关联 vs G² 衰减斜率比较（数值诊断，'
                             '不判相，默认关闭）')
    parser.add_argument('--no-plots', action='store_true',
                        help='跳过绘图')
    parser.add_argument('--convergence-dims', default=None,
                        help='逗号分隔的 D 档（如 1500,5000），提供时运行'
                             '收敛诊断（能量/TruncErr/SU(2)/分类/参数五项'
                             '检查 + 证据图）')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)

    refbonds = ([s.strip() for s in args.refbonds.split(',') if s.strip()]
                if args.refbonds else None)
    conv_dims = ([s.strip() for s in args.convergence_dims.split(',')
                  if s.strip()] if args.convergence_dims else None)
    n, t_j, doping = _prompt_config(args.N, args.t_j, args.doping)
    config = {
        'root': args.root, 't_j': t_j, 'doping': doping,
        'N': n, 'bond_dim': args.bond_dim, 'refbonds': refbonds,
        'n_qx': args.n_qx, 'n_qy': args.n_qy, 'a': args.a,
        'energy_root': args.energy_root, 'j_value': args.j_value,
        'with_pairing_symmetry': args.with_pairing_symmetry,
        'with_density_profile': args.with_density_profile,
        'with_sz_profile': args.with_sz_profile,
        'with_g2_comparison': args.with_g2_comparison,
        'no_plots': args.no_plots, 'outdir': args.outdir,
        'convergence_dims': conv_dims,
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
