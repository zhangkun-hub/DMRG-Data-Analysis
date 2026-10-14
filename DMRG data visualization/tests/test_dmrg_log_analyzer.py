# -*- coding: utf-8 -*-
"""
DMRG 优化日志末行 E0 提取测试（P2 方案 A）

覆盖：
- A. 末行提取：多 D 阶段、每阶段 sweep 编号从 1 重启、每 sweep 多条
  E0 行时精确提取最后一条（末行能量构造为非最小值，反证"取 min"实现）
- B. 行内瞬时 D 与阶段头行 D 不一致（重启期 1631/1808）不影响分组
- C. 空日志 / 无 E0 行 / 无头行 → 空 dict（不静默伪造结果）
- D. 原有 Site (40, 41) 中段值口径行为不变（MID_ENERGY_PATTERN +
  DMRGLogParser 路径）
- E. 真实 8 份 N=200 日志只读验证（数据存在时运行）：末行提取组数与
  收敛阶段"中段 vs 末行"差异

合成日志构造约定（对应真实日志格式）：
    Bond dimension:\t XXXX/XXXX      ← 阶段头行（含制表符）
    sweep N                          ← 每阶段内 1..M 重启
    Site (   i,   j) E0 = ... TruncErr = ... D =  XXXX ...
运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_dmrg_log_analyzer.py
"""
import glob
import importlib.util
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 主脚本文件名含零宽字符 U+200B：用 glob 定位
_SCRIPT = glob.glob(os.path.join(ROOT, 'DMRG_Log_Analyzer*.py'))[0]
_spec = importlib.util.spec_from_file_location('dmrg_log_analyzer', _SCRIPT)
analyzer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyzer)

from Physical_Quantity.DMRG_Log_Parser import DMRGLogParser

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def e0_line(pair, energy, d):
    """合成一条 E0 行（与真实日志字段格式一致）。"""
    i, j = pair
    return (f'Site ({i:4d},{j:4d}) E0 = {energy:.16f} '
            f'TruncErr = 1.00e-09 D = {d:5d} Iter =   2 '
            f'LanczT =     0.20 TotT =     0.30 S =  2.0000000')


def make_log(stages, e0_per_sweep=4, transient=None):
    """合成日志文本。

    stages : [(D, sweep 数), ...]，每阶段内 sweep 编号 1..M 重启
    e0_per_sweep : 每 sweep 的 E0 行数。行能量先随行号递减，末行抬高
      1.5e-8——末行既不是最小值也不是首行值，用于反证错误实现
    transient : 可选 {'key': (D, sweep), 'ds': [瞬时 D 序列]}——注入与
                阶段头行不一致的行内瞬时 D（模拟重启期 1631/1808）
    返回 (text, expected)，expected = {(D, sweep): 末行 E0 精确值}
    """
    lines = []
    expected = {}
    for d, n_sweeps in stages:
        lines.append(f'Bond dimension:\t {d}/{d}')
        for s in range(1, n_sweeps + 1):
            lines.append(f'sweep {s}')
            for k in range(e0_per_sweep):
                line_d = d
                if transient is not None and transient['key'] == (d, s):
                    ds = transient['ds']
                    line_d = ds[min(k, len(ds) - 1)]
                # 能量随行号递减；末行抬高 1.5e-8（非最小值）
                energy = (-29.0 - d * 1e-5 - (s - 1) * 1e-6
                          - (k + 1) * 1e-8)
                if k == e0_per_sweep - 1:
                    energy += 1.5e-8
                lines.append(e0_line((6 + k, 7 + k), energy, line_d))
            expected[(d, s)] = (-29.0 - d * 1e-5 - (s - 1) * 1e-6
                                - e0_per_sweep * 1e-8 + 1.5e-8)
    lines.append('DMRG calculation finished.')
    return '\n'.join(lines) + '\n', expected


# ==========================================================================
# A. 末行提取：多阶段 / sweep 重启 / 末行精确值
# ==========================================================================
print('===== A. 末行提取基本行为 =====')
text_a, exp_a = make_log([(200, 3), (500, 2)], e0_per_sweep=4)
last_a = analyzer.extract_last_e0(text_a)
check('A', '组数 = 阶段 sweep 总数（3+2=5），无多余分组',
      len(last_a) == 5 and set(last_a) == set(exp_a),
      sorted(last_a))
check('A', '每个 (D, sweep) 的末行 E0 精确提取（1e-15 内）',
      all(abs(last_a[k] - v) < 1e-15 for k, v in exp_a.items()))
check('A', 'sweep 编号在不同 D 阶段内重启（(200,1) 与 (500,1) 并存）',
      (200, 1) in last_a and (500, 1) in last_a)
# 末行非最小值：若实现误取 min，会得到 k=e0_per_sweep-2 行的值
text_a2, exp_a2 = make_log([(800, 1)], e0_per_sweep=5)
last_a2 = analyzer.extract_last_e0(text_a2)
check('A', '取最后一条而非最小值/首条（末行能量构造为非最小值）',
      abs(last_a2[(800, 1)] - exp_a2[(800, 1)]) < 1e-15,
      last_a2[(800, 1)])

# ==========================================================================
# B. 行内瞬时 D 与阶段头行 D 不一致
# ==========================================================================
print('===== B. 重启期瞬时 D（1631/1808）=====')
text_b, exp_b = make_log(
    [(2000, 2)], e0_per_sweep=4,
    transient={'key': (2000, 1), 'ds': [1631, 1808, 2000, 2000]})
last_b = analyzer.extract_last_e0(text_b)
check('B', '分组键仍以阶段头行 D=2000 为准（不产生 1631/1808 组）',
      set(last_b) == {(2000, 1), (2000, 2)},
      sorted(last_b))
check('B', '瞬时 D 不影响末行值提取精度',
      abs(last_b[(2000, 1)] - exp_b[(2000, 1)]) < 1e-15)
check('B', '后续 sweep 的末行值不受前一 sweep 瞬时 D 影响',
      abs(last_b[(2000, 2)] - exp_b[(2000, 2)]) < 1e-15)

# ==========================================================================
# C. 空输入 / 无匹配：不伪造结果
# ==========================================================================
print('===== C. 空输入与无匹配 =====')
check('C', '空字符串 → 空 dict',
      analyzer.extract_last_e0('') == {})
check('C', '只有头行/sweep 行、无 E0 行 → 空 dict',
      analyzer.extract_last_e0('Bond dimension:\t 200/200\nsweep 1\n') == {})
check('C', '只有 E0 行、无头行与 sweep 行 → 空 dict（D 未建立不分组）',
      analyzer.extract_last_e0(e0_line((6, 7), -29.1, 200) + '\n') == {})

# ==========================================================================
# D. 原有中段值口径行为不变
# ==========================================================================
print('===== D. 中段值口径（Site 40,41）保留 =====')
text_d, _ = make_log([(200, 2)], e0_per_sweep=4)
out_d = []
mid_expected = {}
pending = None   # 待补 (40,41) 行的 (D, sweep)


def _append_mid_pair():
    d, s = pending
    out_d.append(e0_line((40, 41), -29.5 - (s - 1) * 1e-3, d))
    out_d.append(e0_line((40, 41), -29.5 - (s - 1) * 1e-3 + 5e-8, d))
    mid_expected[(d, s)] = -29.5 - (s - 1) * 1e-3 + 2.5e-8


for line in text_d.splitlines():
    if line.startswith('Bond dimension'):
        if pending is not None:
            _append_mid_pair()
        cur_d = int(line.split()[-1].split('/')[0])
        pending = None
    elif line.startswith('sweep'):
        if pending is not None:
            _append_mid_pair()
        cur_s = int(line.split()[1])
        pending = (cur_d, cur_s)
    out_d.append(line)
if pending is not None:
    _append_mid_pair()
text_mid = '\n'.join(out_d) + '\n'

parser_mid = DMRGLogParser(energy_pattern=analyzer.MID_ENERGY_PATTERN)
parser_mid.parse_from_string(text_mid)
df_mid = parser_mid.get_dataframe()
check('D', '中段值口径：按 (D, sweep) 分组且每组 2 样本（上/下行）',
      len(df_mid) == 2
      and list(df_mid['N_samples']) == [2, 2],
      df_mid[['BondDimension', 'Sweep', 'N_samples']].to_dict('list'))
check('D', '中段值口径：E0_mean 为两条 (40,41) 的均值（1e-12 内，'
      'float64 ulp 级）',
      all(abs(r.E0_mean - mid_expected[(r.BondDimension, r.Sweep)]) < 1e-12
          for r in df_mid.itertuples()))
check('D', 'MID_ENERGY_PATTERN 限定 Site (40, 41)（不匹配其它格点对）',
      re.search(analyzer.MID_ENERGY_PATTERN,
                e0_line((6, 7), -29.0, 200)) is None
      and re.search(analyzer.MID_ENERGY_PATTERN,
                    e0_line((40, 41), -29.0, 200)) is not None)
check('D', 'ALL_ENERGY_PATTERN 匹配任意格点对',
      re.search(analyzer.ALL_ENERGY_PATTERN,
                e0_line((6, 7), -29.0, 200)) is not None)

# ==========================================================================
# E. 真实 8 份 N=200 日志只读验证（数据存在时运行）
# ==========================================================================
print('===== E. 真实日志只读验证 =====')
DATA = os.path.join(ROOT, 'Data_N200', 't_J_3')
REAL = [
    ('d1_10', 'slurm-924148-Zk_tJ.log', {1500: 12}),
    ('d1_10', 'slurm-932873-Zk_tJ.log', {2000: 10, 2500: 10, 3000: 10,
                                         3500: 10, 4000: 10, 4500: 10,
                                         5000: 10}),
    ('d1_100', 'slurm-920743-Zk_tJ.log', {200: 4, 500: 4, 800: 4, 1000: 4,
                                          1200: 4, 1500: 4}),
    ('d1_100', 'slurm-932876-Zk_tJ.log', {2000: 10, 2500: 10, 3000: 10,
                                          3500: 10, 4000: 10, 4500: 10,
                                          5000: 10}),
    ('d1_20', 'slurm-921024-Zk_tJ.log', {200: 4, 500: 4, 800: 4, 1000: 4,
                                         1200: 4, 1500: 4}),
    ('d1_20', 'slurm-932875-Zk_tJ.log', {2000: 10, 2500: 10, 3000: 10,
                                         3500: 10, 4000: 10, 4500: 10,
                                         5000: 10}),
    ('d2_25', 'slurm-924573-Zk_tJ.log', {1500: 10}),
    ('d2_25', 'slurm-932874-Zk_tJ.log', {2000: 10, 2500: 10, 3000: 10,
                                         3500: 10, 4000: 10, 4500: 10,
                                         5000: 10}),
]
if os.path.isdir(DATA):
    ok_all = True
    for doping, fname, expected_counts in REAL:
        path = os.path.join(DATA, doping, fname)
        if not os.path.isfile(path):
            check('E', f'{doping}/{fname} 存在', False, '文件缺失')
            ok_all = False
            continue
        last = analyzer.extract_last_e0(
            open(path, encoding='utf-8', errors='replace').read())
        counts = Counter(d for d, _ in last)
        got_expected = all(counts[d] == n
                           for d, n in expected_counts.items())
        check('E', f'{doping}/{fname}: (D,sweep) 组数与各阶段 sweep 数'
              f'精确匹配（{len(last)} 组）',
              got_expected, dict(counts))
        ok_all = ok_all and got_expected
    # 5000 阶段最终 sweep 的末行值 vs 中段值
    print('  D=5000 最终 sweep：末行值 vs 中段值（Site 40,41 均值）:')
    for doping, fname, _ in REAL:
        if not fname.startswith('slurm-932'):
            continue
        path = os.path.join(DATA, doping, fname)
        text = open(path, encoding='utf-8', errors='replace').read()
        last = analyzer.extract_last_e0(text)
        parser_mid = DMRGLogParser(energy_pattern=analyzer.MID_ENERGY_PATTERN)
        parser_mid.parse_from_string(text)
        df = parser_mid.get_dataframe()
        d_final = int(df.BondDimension.max())
        s_final = int(df[df.BondDimension == d_final].Sweep.max())
        mid_val = float(df[(df.BondDimension == d_final)
                           & (df.Sweep == s_final)].E0_mean.iloc[0])
        last_val = last[(d_final, s_final)]
        diff = abs(mid_val - last_val)
        print(f'    {doping}: mid={mid_val:.12f} last={last_val:.12f} '
              f'|diff|={diff:.3e}')
        check('E', f'{doping} D={d_final} 末 sweep：中段与末行差异 < 1e-5',
              diff < 1e-5, f'{diff:.3e}')
else:
    check('E', '真实日志验证（数据目录不存在，跳过）', True, '跳过')

# ==========================================================================
# 汇总
# ==========================================================================
print()
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in RESULTS:
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    print(f'{cat:6}: {n_pass}/{len(RESULTS[cat])} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)
