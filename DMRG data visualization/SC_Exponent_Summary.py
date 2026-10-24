# -*- coding: utf-8 -*-
"""
SC 指数跨数据汇总脚本（D1：超导关联衰减指数的受控汇总）

复用 Cross_Doping_Summary.analyze_doping 的参考键目录发现、数据加载、
失败记录与统一入口 run_analysis，**不重新拟合**：指数与误差直接取自
report.payload（DecayAnalysisResult 的 envelope_fit / candidate_fits /
StabilityReport），与跨 doping 汇总 CSV 中的 params_json 数值同源。

职责边界（严格）：
- 只做「运行既有 SC 衰减分析 + 提取指数证据 + 机械分组呈现」，
  不实现任何拟合/模型选择/稳定性判据，不修改衰减分析器
- 幂律指数 K 与指数衰减长度 xi 是不同物理量：分列、分类型、分组建
  呈现，绝不混列或直接比较
- 指数状态四分类（见 _STATES 与 semantics）：exponent_ok / unstable /
  data_insufficient / input_missing——moderate 不等于失败（verdict 原文
  保留），失败原因原样保留，不伪装成功、不填零、不静默跳过
- 可比性分组复用 Cross_Doping_Summary.summarize 的机械规则
  （comparability_bucket 原样透传）；本脚本另按身份全等（N / bond
  dimension / 参考键 / 键方向 / 指数类型）分组并列展示，不同条件
  不合并为同一趋势
- 不计算跨组平均指数，不生成超导相结论，不判定 d+id 或任何物理相态

运行方式（在 DMRG data visualization/ 目录下）：
    python SC_Exponent_Summary.py --dopings d1_10,d1_20,d1_100,d2_25
        --outdir sc_exponent_summary_output

输出（--outdir 指定目录；整体覆盖写，不清理目录内其他文件）：
- sc_exponent_summary.csv：一行一条 SC 衰减分析（含失败条目），固定列序
- sc_exponent_summary.json：config 快照 + rows + 分组 + 语义说明
注意：本脚本默认输出目录未被 .gitignore 覆盖（与 Cross_Doping_Summary
的输出目录不同），正式运行建议 --outdir tests/artifacts/<name> 或
其他已忽略位置，避免未跟踪产物混入工作区。
"""

import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Cross_Doping_Summary as cds
from Physical_Quantity.Analysis_Contract import (
    _convert, OBSERVABLE_SUPERCONDUCTING_CORRELATION)
from Physical_Quantity.Spin_Structure_Factor import xc_momentum_grid

# ---- 指数状态四分类（映射规则见 classify_state docstring 与 semantics） ----
STATE_EXPONENT_OK = 'exponent_ok'
STATE_UNSTABLE = 'unstable'
STATE_DATA_INSUFFICIENT = 'data_insufficient'
STATE_INPUT_MISSING = 'input_missing'
_STATES = (STATE_EXPONENT_OK, STATE_UNSTABLE,
           STATE_DATA_INSUFFICIENT, STATE_INPUT_MISSING)

# ---- 候选模型证据范围（只收录存在且 valid 的模型；幂律族模型的 K） ----
_CANDIDATE_POWER_MODELS = ('power_law', 'power_law_C',
                           'osc_power', 'osc_power_C')

# CSV 固定列序（新字段须同步更新本元组与 build_exponent_row 模板）
CSV_COLUMNS = (
    'entry_index', 'doping', 'N', 'bond_dim', 'refbond', 'direction',
    'data_source', 'exponent_type', 'envelope_form', 'exponent_value',
    'exponent_error', 'error_source', 'fit_r_min', 'fit_r_max', 'n_points',
    'verdict', 'rel_mad_K', 'model_consistency', 'spearman_rho',
    'recommended_r_min', 'recommended_model', 'envelope_mode',
    'env_aicc', 'env_bic', 'classification', 'evidence',
    'state', 'state_reason',
    'comparability_bucket', 'comparability_reason',
    'candidate_models_json', 'warnings_json', 'failed_type', 'failed_message',
)


# ==========================================================================
# 指数行构建（读取 report.payload 的原始结果，不重新拟合）
# ==========================================================================
def classify_state(report, failure):
    """指数状态四分类（依据实际结果结构，优先保留原始 verdict 与失败原因）。

    映射规则（机械，无物理判据）：
    - 失败条目：FileNotFoundError → input_missing（输入文件/目录缺失）；
      其余数据级失败（ValueError 等）→ data_insufficient（原始类型与
      消息原样保留）——两者无法进一步细分时如实归入后者，不猜测
    - classification == 'INCONCLUSIVE' → data_insufficient（距离数不足等
      早退，无可用指数）
    - 包络拟合结果缺失（envelope_fit 为 None 或无 params）→ unstable
      （分析完成但无法提取指数，不伪装成功）
    - StabilityReport 缺失 → unstable（稳定性证据缺失，保守处理）
    - verdict == 'insufficient' → data_insufficient（r_min 扫描有效候选
      不足，包络指数证据不足）
    - verdict == 'unstable' → unstable
    - verdict ∈ ('stable', 'moderate') → exponent_ok（moderate 不等于
      失败，verdict 原文保留在独立列）
    """
    if report is None:
        if failure['type'] == 'FileNotFoundError':
            return STATE_INPUT_MISSING, '输入文件或目录缺失'
        return STATE_DATA_INSUFFICIENT, (
            f"数据级失败（{failure['type']}），原始原因见 failed_message")
    payload = report.payload
    envelope = getattr(payload, 'envelope_fit', None)
    stability = getattr(payload, 'stability', None)
    if getattr(payload, 'classification', '') == 'INCONCLUSIVE':
        return STATE_DATA_INSUFFICIENT, (
            'classification=INCONCLUSIVE（距离数不足等早退），无可用指数')
    if envelope is None or not getattr(envelope, 'params', None):
        return STATE_UNSTABLE, '包络拟合结果缺失，无法提取指数'
    if stability is None:
        return STATE_UNSTABLE, '稳定性报告缺失'
    if stability.verdict == 'insufficient':
        return STATE_DATA_INSUFFICIENT, (
            'verdict=insufficient（r_min 扫描有效候选不足）')
    if stability.verdict == 'unstable':
        return STATE_UNSTABLE, 'verdict=unstable（指数对拟合窗口不稳定）'
    return STATE_EXPONENT_OK, ''


def build_exponent_row(entry_index, cds_row, outcome):
    """cds 汇总行 + outcome（含 payload）→ SC 指数行（固定列序）。

    cds_row 提供身份/契约级字段与 comparability_bucket；payload 提供
    包络指数/误差/稳定性细节/候选模型证据。所有数值直接取自既有结果，
    无任何重算。缺失字段留空，不编造默认值。
    """
    row = {col: '' for col in CSV_COLUMNS}
    row['entry_index'] = entry_index
    row['doping'] = cds_row['doping']
    row['N'] = cds_row['N']
    row['bond_dim'] = cds_row['bond_dim']
    row['refbond'] = cds_row['refbond_or_source']
    row['direction'] = cds_row['key_or_definition']
    row['comparability_bucket'] = cds_row['comparability_bucket']
    row['comparability_reason'] = cds_row['comparability_reason']
    row['warnings_json'] = cds_row['warnings_json']
    # 数据来源标识：目录相对身份（项目约定不记录绝对路径）；
    # 失败条目无 refbond 时留空（无法确认来源，不猜测）
    if row['refbond']:
        row['data_source'] = (
            f"{cds_row['bond_dim']}_measure_super_corr/"
            f"super_corr_{row['refbond']}")

    report = outcome['report']
    row['state'], row['state_reason'] = classify_state(
        report, outcome['failure'])
    if report is None:
        row['failed_type'] = outcome['failure']['type']
        row['failed_message'] = outcome['failure']['message']
        return row

    # ---- 契约级字段（与 cds 行同源） ----
    row['classification'] = cds_row['classification']
    row['evidence'] = cds_row['evidence']
    row['fit_r_min'] = cds_row['fit_r_min']
    row['fit_r_max'] = cds_row['fit_r_max']
    row['n_points'] = cds_row['n_points']
    row['verdict'] = cds_row['verdict']
    row['recommended_model'] = cds_row['fit_model']
    row['envelope_mode'] = cds_row['envelope_mode']

    payload = report.payload
    envelope = getattr(payload, 'envelope_fit', None)
    stability = getattr(payload, 'stability', None)

    # ---- 主包络指数（K 与 xi 严格区分，不混列） ----
    if envelope is not None and getattr(envelope, 'params', None):
        p = envelope.params
        if 'K' in p:
            row['exponent_type'] = 'K'
            row['envelope_form'] = 'power'
            row['exponent_value'] = p['K']
            row['exponent_error'] = envelope.param_errors.get('K')
        elif 'xi' in p:
            row['exponent_type'] = 'xi'
            row['envelope_form'] = 'exp'
            row['exponent_value'] = p['xi']
            row['exponent_error'] = envelope.param_errors.get('xi')
        row['error_source'] = envelope.error_source
        row['env_aicc'] = envelope.aicc
        row['env_bic'] = envelope.bic

    # ---- 稳定性细节（StabilityReport 既有字段，缺失留空） ----
    if stability is not None:
        row['rel_mad_K'] = stability.rel_mad_K
        row['model_consistency'] = stability.model_consistency
        row['spearman_rho'] = stability.spearman_rho
        row['recommended_r_min'] = stability.recommended_r_min

    # ---- 候选模型证据（只收录存在且 valid 的幂律族模型；保留模型
    # 选择证据，不筛掉"不利于超导解释"的模型） ----
    cands = getattr(payload, 'candidate_fits', None)
    if cands:
        evidence = {}
        for name in _CANDIDATE_POWER_MODELS:
            fit = cands.get(name)
            if fit is None or not getattr(fit, 'valid', False):
                continue
            entry = {'model': name, 'fit_space': fit.fit_space,
                     'aicc': fit.aicc, 'bic': fit.bic}
            if 'K' in fit.params:
                entry['K'] = fit.params['K']
                entry['K_err'] = fit.param_errors.get('K')
            evidence[name] = entry
        if evidence:
            row['candidate_models_json'] = cds._json_dump(evidence)
    return row


# ==========================================================================
# 分组（身份全等：N/bond_dim/refbond/方向/指数类型；组间不合并趋势）
# ==========================================================================
def build_groups(exponent_rows):
    """按身份全等分组并列展示。K 与 xi 不共组；失败行（指数类型空）
    独立成组。只呈现成员与状态计数，不计算任何跨组平均指数。"""
    groups = []
    index = {}
    for i, r in enumerate(exponent_rows):
        key = (r['N'], r['bond_dim'], r['refbond'], r['direction'],
               r['exponent_type'])
        if key not in index:
            index[key] = len(groups)
            groups.append({
                'N': r['N'], 'bond_dim': r['bond_dim'],
                'refbond': r['refbond'], 'direction': r['direction'],
                'exponent_type': r['exponent_type'],
                'member_indices': [],
                'n_total': 0, 'n_exponent_ok': 0, 'n_unstable': 0,
                'n_data_insufficient': 0, 'n_input_missing': 0})
        g = groups[index[key]]
        g['member_indices'].append(i)
        g['n_total'] += 1
        key_map = {
            STATE_EXPONENT_OK: 'n_exponent_ok',
            STATE_UNSTABLE: 'n_unstable',
            STATE_DATA_INSUFFICIENT: 'n_data_insufficient',
            STATE_INPUT_MISSING: 'n_input_missing',
        }
        g[key_map[r['state']]] += 1
    return groups


# ==========================================================================
# 主流程（复用 analyze_doping 的 SC-only 筛选，只跑 SC 衰减）
# ==========================================================================
def run_summary(root, t_j, dopings, bond_dim, N, a, q_list, refbonds):
    """逐 doping 运行 SC 衰减分析（only_observables 筛选，其余分析不
    执行）→ cds.summarize 可比性分组 → 指数行构建 → 身份分组。"""
    outcomes = []
    for doping in dopings:
        outcomes.extend(cds.analyze_doping(
            doping, root=root, t_j=t_j, bond_dim=bond_dim, N=N, a=a,
            q_list=q_list, refbonds=refbonds,
            only_observables={OBSERVABLE_SUPERCONDUCTING_CORRELATION}))
    summary = cds.summarize(outcomes, q_list)
    rows = summary['rows']
    if len(rows) != len(outcomes):
        raise RuntimeError(
            f"cds 汇总行数与 outcomes 不一致: {len(rows)} vs {len(outcomes)}")
    exp_rows = [build_exponent_row(i, r, o)
                for i, (r, o) in enumerate(zip(rows, outcomes))]
    groups = build_groups(exp_rows)
    return exp_rows, groups, summary


# ==========================================================================
# 输出（CSV + 严格 JSON；整体覆盖写，不清理目录内其他文件）
# ==========================================================================
def write_outputs(exp_rows, groups, config, outdir) -> dict:
    """写 CSV（utf-8-sig，固定列序，NaN/None 保守空值）与 JSON
    （严格 allow_nan=False，NaN → null）。只覆盖本次指定的两个输出
    文件，不删除或清理输出目录内其他产物。"""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'sc_exponent_summary.csv')
    json_path = os.path.join(outdir, 'sc_exponent_summary.json')

    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        for row in exp_rows:
            writer.writerow({k: cds._csv_cell(row.get(k, ''))
                             for k in CSV_COLUMNS})

    doc = {
        'generated_by': 'SC_Exponent_Summary (D1)',
        'config': _convert(config),
        'rows': [{k: _convert(r.get(k, '')) for k in CSV_COLUMNS}
                 for r in exp_rows],
        'groups': _convert(groups),
        'semantics': {
            'no_physical_conclusions': (
                '本报告只整理 SC 关联衰减指数证据，不判定物理相态、'
                '配对对称性或 d+id；不计算跨组平均指数，不生成趋势结论'),
            'exponent_types': (
                'K 为幂律包络指数（A·r^(−K)，log|C| 空间拟合）；xi 为指数'
                '衰减长度（A·e^(−r/xi)）。两者是不同物理量，分列、分类型、'
                '分组建呈现，绝不混列或直接比较'),
            'states': {
                STATE_EXPONENT_OK: '指数估计成功（verdict 为 stable 或 '
                                   'moderate；moderate 不等于失败，verdict '
                                   '原文保留）',
                STATE_UNSTABLE: '拟合结果不稳定（verdict=unstable、包络'
                                '拟合或稳定性报告缺失）',
                STATE_DATA_INSUFFICIENT: '数据不足或数据级拟合失败'
                                         '（classification=INCONCLUSIVE、'
                                         'verdict=insufficient、或非'
                                         'FileNotFoundError 的失败）',
                STATE_INPUT_MISSING: '输入数据缺失（FileNotFoundError，'
                                     '文件或目录缺失）',
            },
            'state_mapping_limits': (
                '非 FileNotFoundError 的失败统一归入 data_insufficient：'
                '现有失败路径无法进一步区分"数据不足"与"数据损坏"'
                '（ValueError 等），原始类型与消息原样保留，不猜测'),
            'data_source': (
                '相对目录身份（如 1500_measure_super_corr/'
                'super_corr_49_50），不含绝对路径；失败条目无参考键时'
                '留空（无法确认来源）'),
            'comparability': (
                'comparability_bucket 复用 Cross_Doping_Summary 机械规则'
                '（direct 仅表示分析条件一致，不表示 DMRG 收敛精度相同）；'
                'groups 按身份全等（N/bond dimension/参考键/键方向/指数'
                '类型）并列展示，不同条件不合并'),
            'candidate_models': (
                '仅收录存在且 valid 的幂律族候选模型（power_law/'
                'power_law_C/osc_power/osc_power_C）的 K、误差、拟合空间'
                '与 AICc/BIC，保留模型选择证据，不筛选"有利"模型'),
            'missing_fields': '空字符串 / JSON null 表示该字段缺失或不适用'
                              '，未填入虚构默认值',
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
    return {'csv': csv_path, 'json': json_path}


# ==========================================================================
# 控制台摘要（仅事实呈现）
# ==========================================================================
def _print_console(exp_rows, groups, paths, config):
    by_doping = {}
    for r in exp_rows:
        d = by_doping.setdefault(r['doping'], {s: 0 for s in _STATES})
        d[r['state']] += 1
    print(f"\n===== SC 指数汇总（N={config.get('N')}, "
          f"bond_dim={config.get('bond_dim')}）=====")
    print(f"{'doping':10} {'ok':>4} {'unstable':>9} {'insufficient':>13} "
          f"{'missing':>8}")
    for doping in sorted(by_doping):
        d = by_doping[doping]
        print(f"{doping:10} {d[STATE_EXPONENT_OK]:>4} "
              f"{d[STATE_UNSTABLE]:>9} {d[STATE_DATA_INSUFFICIENT]:>13} "
              f"{d[STATE_INPUT_MISSING]:>8}")
    print("\n===== 分组（身份全等；组间不合并） =====")
    for g in groups:
        idt = (f"N={g['N']}|D={g['bond_dim']}|ref={g['refbond'] or '-'}|"
               f"{g['direction'] or '-'}|type={g['exponent_type'] or '-'}")
        print(f"  {idt}  ok={g['n_exponent_ok']} "
              f"unstable={g['n_unstable']} insuff={g['n_data_insufficient']} "
              f"missing={g['n_input_missing']}")
    print(f"\n输出: {paths['csv']}\n      {paths['json']}")
    print("说明: 本报告只整理指数证据，不判定物理相态/配对对称性/d+id；"
          "K 与 xi 分组呈现，不混列。")


# ==========================================================================
# CLI 入口
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='SC 指数跨数据汇总（D1）：逐 doping 运行既有 SC 衰减'
                    '分析并提取指数证据，不重新拟合，不判定物理相态')
    parser.add_argument('--root', default=os.path.join(SCRIPT_DIR, 'Data_N200'),
                        help='数据根目录（默认 <脚本目录>/Data_N200）')
    parser.add_argument('--t-j', dest='t_j', default=cds.DEFAULT_T_J,
                        help='t/J 目录名（默认 t_J_3）')
    parser.add_argument('--dopings', default=','.join(cds.DEFAULT_DOPINGS),
                        help='逗号分隔的 doping 列表（默认与跨 doping 汇总'
                             '一致；实际有数据的 4 个请显式指定）')
    parser.add_argument('--bond-dim', dest='bond_dim',
                        default=cds.DEFAULT_BOND_DIM,
                        help='bond dimension（默认 1500）')
    parser.add_argument('--N', type=int, default=cds.DEFAULT_N,
                        help='系统格点数（默认 200；N=400 请单独运行）')
    parser.add_argument('--n-qx', type=int, default=21,
                        help='动量网格 qx 点数（默认 21；SC 衰减不使用该'
                             '网格，仅为复用 analyze_doping 签名）')
    parser.add_argument('--n-qy', type=int, default=4,
                        help='动量网格 qy 点数（默认 4）')
    parser.add_argument('--a', type=float, default=1.0, help='晶格常数')
    parser.add_argument('--refbonds', default=None,
                        help='逗号分隔的参考键过滤（如 49_50,50_51）；'
                             '默认全部发现的参考键')
    parser.add_argument('--outdir',
                        default=os.path.join(SCRIPT_DIR,
                                             'sc_exponent_summary_output'),
                        help='输出目录（默认 <脚本目录>/'
                             'sc_exponent_summary_output，未被 .gitignore '
                             '覆盖，正式运行建议指定已忽略位置）')
    args = parser.parse_args(argv)

    dopings = [d.strip() for d in args.dopings.split(',') if d.strip()]
    refbonds = None
    if args.refbonds:
        refbonds = {rb.strip() for rb in args.refbonds.split(',')
                    if rb.strip()}
    q_list, q_meta = xc_momentum_grid(args.N, a=args.a,
                                      n_qx=args.n_qx, n_qy=args.n_qy)

    config = {
        'root': args.root, 't_j': args.t_j, 'dopings': dopings,
        'bond_dim': args.bond_dim, 'N': args.N, 'a': args.a,
        'n_qx': args.n_qx, 'n_qy': args.n_qy,
        'refbonds_filter': sorted(refbonds) if refbonds else None,
        'analyzer_config': 'default（AnalyzerConfig 默认阈值；与跨 doping '
                           '汇总一致，不重新拟合）',
    }
    exp_rows, groups, summary = run_summary(
        args.root, args.t_j, dopings, args.bond_dim, args.N, args.a,
        q_list, refbonds)
    paths = write_outputs(exp_rows, groups, config, args.outdir)
    _print_console(exp_rows, groups, paths, config)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
