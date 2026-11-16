# -*- coding: utf-8 -*-
"""
DMRG 收敛判断独立入口（P1 阶段 1）。

对指定参数组（N、t/J、doping）判断其数据的 bond dimension 收敛状态：
- 五项检查（E0 饱和 / TruncErr / SU(2) 残差 / 分类一致 / 参数误差内一致）
  → 三档结论（converged / needs_higher_D / suspicious）
- 证据：能量收敛图 + 密度/自旋/超导两档 |差值| 图

只回答"数值是否稳定、D 是否还需增加"，**不回答能否判相**。
阈值为默认值（未获物理授权，报告内标注 defaults_not_authorized）。

运行方式（Calculation 环境）：
    python -X utf8 Convergence_Report.py --doping d1_10 \
        --dims 1500,5000
"""
import argparse
import datetime
import json
import os
import re
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Phase_Analysis_Report as par  # noqa: E402

DEFAULT_T_J = 't_J_3'
DEFAULT_DIMS = '1500,5000'
DEFAULT_N = 200
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, 'output', 'convergence_report')

DISCIPLINE = (
    '> 阈值为默认值（defaults_not_authorized，未获物理授权）。'
    'converged ≠ 物理结论可信；needs_higher_D ≠ 数据作废。'
    '本报告只判断数值稳定性（bond dimension 收敛），不回答能否判相。'
    'SC 衰减参数误差兜底 bootstrap 用轻量次数 30（默认 200），'
    '误差估计更粗；分类结论不依赖 bootstrap，不受影响。')


# 交互提示与空穴率解析复用 Phase_Analysis_Report 的实现（默认 N=200、
# t/J=3、空穴率 1_10 → d1_10），两脚本行为一致
_fraction_to_doping = par._fraction_to_doping
_prompt_config = par._prompt_config


def _doping_display(doping):
    """d1_10 → '1/10=0.10'（无法解析时原样返回）。"""
    m = re.match(r'^d(\d+)_(\d+)$', doping)
    if not m:
        return doping
    num, den = int(m.group(1)), int(m.group(2))
    return f'{num}/{den} = {num / den:.2f}'


def _tj_display(t_j):
    """t_J_3 → 't/J=3'（无法解析时原样返回）。"""
    m = re.match(r'^t_J_(.+)$', t_j)
    return f'{m.group(1)}' if m else t_j


VERDICT_EXPLANATION = {
    'converged': ('全部可用检查通过：当前 D 档下数值稳定，该组数据可直接'
                  '用于论文引用，**不需要**更大的 D 计算。'),
    'needs_higher_D': ('能量或分类随 D 仍有明显变化：当前 D 尚未完全收敛，'
                       '**建议加算更大的 D**（如继续翻倍）后再引用结果。'),
    'suspicious': ('分类随 D 变化明显（数值振荡信号）：收敛性存疑，'
                   '**建议优先排查收敛性**（更大 D 或检查优化过程）；'
                   '在收敛问题解决前，引用该参数组结果需谨慎并注明。'),
}


def render_markdown(config, conv, outdir) -> str:
    cv = conv['verdict']
    lines = [
        f"# 收敛诊断报告：{config['doping']}"
        f"（{_doping_display(config['doping'])}，N={config['N']}，"
        f"{_tj_display(config['t_j'])}）",
        '',
        DISCIPLINE,
        '',
        '## 0. 运行配置',
        '',
        '| 项 | 值 |',
        '|---|---|',
        f"| doping | {_doping_display(config['doping'])} |",
        f"| N | {config['N']} |",
        f"| t/J | {_tj_display(config['t_j'])} |",
        f"| D 档 | {', '.join(conv['dims'])} |",
        f"| 生成时间 | {config['generated_at']} |",
        '',
        '## 1. 判定结论',
        '',
        f"**{cv['verdict']}**——{cv['reason']}",
        '',
        '## 收敛性说明',
        '',
        VERDICT_EXPLANATION.get(cv['verdict'], cv['reason']),
        '',
        '## 2. 检查明细',
        '',
        '| 检查项 | 结果 | 证据 |',
        '|---|---|---|',
    ]
    for c in cv['checks']:
        mark = ('✅ 通过' if c['passed'] is True else
                ('❌ 未通过' if c['passed'] is False else '— 证据不足'))
        note = f"（{c['note']}）" if c.get('note') else ''
        lines.append(f"| {c['name']} | {mark} | {c['evidence']}{note} |")
    # 分类逐方向对比表（classes_detail_by_dim 提供标签）
    detail = conv['inputs'].get('classes_detail_by_dim') or {}
    if len(detail) >= 2:
        dims = conv['dims']
        tags = sorted(set().union(*(set(v) for v in detail.values())))
        lines += ['', '### 衰减分类逐方向对比', '',
                  '| 参考键_方向 | ' + ' | '.join(f'D={d}' for d in dims)
                  + ' |',
                  '|' + '---|' * (len(dims) + 1)]
        for tag in tags:
            lines.append(f'| {tag} | ' + ' | '.join(
                str(detail[d].get(tag, '—')) for d in dims) + ' |')
    lines += [
        '',
        f"- 阈值来源状态：{cv['threshold_status']}",
        f"- 阈值数值：{json.dumps(cv['thresholds'], ensure_ascii=False)}",
        '',
        '## 3. 证据图',
        '',
    ]
    if conv['figures']:
        for ref in conv['figures']:
            lines.append(f'- ![收敛证据]({ref})')
    else:
        lines.append('- （--no-plots 或无证据图）')
    if conv['failures']:
        lines += ['', '## 附录：收集失败项', '']
        for f in conv['failures']:
            lines.append(f"- D={f['dim']}: {f['type']}: {f['message']}")
    lines.append('')
    return '\n'.join(lines)


def build_json(config, conv) -> dict:
    return {'generated_by': 'Convergence_Report', 'config': config,
            'verdict': conv['verdict'], 'inputs': conv['inputs'],
            'figures': conv['figures'], 'failures': conv['failures'],
            'semantics': {
                'discipline': ('只判断 bond dimension 收敛（数值稳定性），'
                               '不回答能否判相；阈值未获物理授权'),
                'threshold_status': conv['verdict']['threshold_status'],
            }}


def build_report(config) -> dict:
    """核心入口（独立于 main，便于测试）。"""
    config.setdefault('generated_at',
                      datetime.datetime.now().isoformat(timespec='seconds'))
    conv = par.run_convergence_section(config, config['dims'])
    md = render_markdown(config, conv, config['outdir'])
    return {'markdown': md, 'convergence': conv,
            'json': build_json(config, conv)}


def write_report(result, outdir) -> dict:
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, 'convergence_report.md')
    json_path = os.path.join(outdir, 'convergence_report.json')
    with open(md_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(result['markdown'])
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result['json'], f, ensure_ascii=False, indent=2,
                  allow_nan=False)
    return {'convergence_report.md': md_path,
            'convergence_report.json': json_path}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='DMRG 收敛诊断独立入口（bond dimension 稳定性；'
                    '不判相，详见脚本 docstring）')
    parser.add_argument('--root', default=os.path.join(SCRIPT_DIR,
                                                       'Data/complex/Data_N200'))
    parser.add_argument('--t-j', default=None,
                        help='t/J 目录后缀（如 t_J_3）；不提供时交互式'
                             '提示输入 t/J 值（默认 3）')
    parser.add_argument('--doping', default=None,
                        help='单个 doping（如 d1_10 或空穴率 0.1）；'
                             '不提供时交互式提示输入（默认 d=0.1）')
    parser.add_argument('--N', type=int, default=None,
                        help='格点数（默认 200）；不提供时交互式提示输入')
    parser.add_argument('--dims', default=None,
                        help='逗号分隔的 D 档（如 1500,5000）；不提供时'
                             '交互式提示输入（默认 1500,5000）')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--refbonds', default=None)
    parser.add_argument('--n-qx', type=int, default=41)
    parser.add_argument('--n-qy', type=int, default=4)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)

    n, t_j, doping = _prompt_config(args.N, args.t_j, args.doping)
    if args.dims is None:
        try:
            raw = input('请输入要比较的 D 档（逗号分隔，'
                        '回车使用默认 1500,5000）: ').strip()
        except EOFError:
            raw = ''
        dims_str = raw or DEFAULT_DIMS
    else:
        dims_str = args.dims
    dims = [s.strip() for s in dims_str.split(',') if s.strip()]
    # 校验各档在数据目录中存在（无数据可发现时跳过）
    doping_dir = os.path.join(args.root, t_j, doping)
    available = par.available_bond_dims(doping_dir)
    if available:
        missing = [d for d in dims if d not in available]
        if missing:
            print(f'错误：数据目录 {doping_dir} 中没有 '
                  f'D={",".join(missing)} 的数据；'
                  f'可比较的 bond dimension 档位: {", ".join(available)}')
            return 1
    config = {'root': args.root, 't_j': t_j, 'doping': doping,
              'N': n, 'a': args.a,
              'refbonds': ([s.strip() for s in args.refbonds.split(',')
                            if s.strip()] if args.refbonds else None),
              'n_qx': args.n_qx, 'n_qy': args.n_qy,
              'outdir': args.outdir, 'no_plots': args.no_plots,
              'dims': dims, 'bond_dim': dims[0] if dims else '1500'}
    result = build_report(config)
    paths = write_report(result, args.outdir)
    v = result['convergence']['verdict']
    print(f"收敛判定：{v['verdict']}——{v['reason']}")
    print(f"输出: {paths['convergence_report.md']}")
    print(f"      {paths['convergence_report.json']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
