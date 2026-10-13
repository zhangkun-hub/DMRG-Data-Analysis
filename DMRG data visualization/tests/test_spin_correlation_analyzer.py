# -*- coding: utf-8 -*-
"""
Spin_Correlation_Analyzer 测试（数据适配 / 校验 / 真实 N200 回归）

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_spin_correlation_analyzer.py

基线说明：baselines/spin_n200_baseline_v1.json 为自旋分析器首次开发完成时
生成的当前版本基线（非任何旧版基准），用于之后的回归比较。

本目录被 .gitignore 的 *test* 规则忽略，纳入版本控制需调整规则。
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Physical_Data_Record import CorrelationRecord
from Physical_Quantity.Correlation_Process import CorrelationProcessor
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer
from Physical_Quantity.Correlation_Decay_Analyzer import AnalyzerConfig

BASELINE = os.path.join(HERE, 'baselines', 'spin_n200_baseline_v1.json')
RESULTS = {'synthetic': [], 'edge_case': [], 'real_data': []}


def check(category, name, cond, detail=''):
    RESULTS[category].append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def diff_paths(a, b, path=''):
    """递归精确比较（数值全等、字符串全等、键集与列表结构一致）"""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            diffs.append(f'{path} 键集不一致: {set(a) ^ set(b)}')
            return diffs
        for k in a:
            diffs.extend(diff_paths(a[k], b[k], f'{path}.{k}'))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path} 长度 {len(a)} vs {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(diff_paths(x, y, f'{path}[{i}]'))
    elif a is None and b is None:
        return diffs
    elif a is None or b is None:
        diffs.append(f'{path}: {a!r} vs {b!r}')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b:
            diffs.append(f'{path}: {a!r} vs {b!r}')
    elif str(a) != str(b):
        diffs.append(f'{path}: {a!r} vs {b!r}')
    return diffs


# ==========================================================================
# A. 合成数据（真实蜂窝格点索引，保证族划分可计算；含天然重复距离）
# ==========================================================================
print('===== A. 合成数据 =====')
g = HoneycombXCGeometry(200, 1.0)
source = 50                     # B 子格源点（50 % 4 == 2）
fields = [j for j in range(2, 200, 4) if j != source]
dists = [g.get_distance(source, j) for j in fields]
uniq_d = len(set(round(d, 9) for d in dists))
print(f'  合成构造：{len(fields)} 条记录，唯一距离 {uniq_d} 个（重复距离 {len(fields)-uniq_d} 对）')

# 1. 幂律合成：C = 0.25·r^(-1.5) + 5% 噪声
rng = np.random.default_rng(7)
base = np.array([0.25 * d ** (-1.5) for d in dists])
C_pow = base + 0.05 * np.abs(base) * rng.standard_normal(len(base))
recs_pow = [CorrelationRecord(source, j, d, v) for j, d, v in zip(fields, dists, C_pow)]
res = SpinCorrDecayAnalyzer.analyze(recs_pow, N=200)
check('synthetic', '幂律合成分类 ∈ {power_law, exponential, non_monotonic, INCONCLUSIVE}',
      res.classification in ('power_law', 'exponential', 'non_monotonic', 'INCONCLUSIVE'),
      res.classification)
check('synthetic', '幂律合成 n_raw 与输入一致', res.n_raw_points == len(recs_pow),
      f"n_raw={res.n_raw_points} vs {len(recs_pow)}")
check('synthetic', '幂律合成距离数组与输入一致',
      res.n_distances == uniq_d, f"n_dist={res.n_distances} vs {uniq_d}")
check('synthetic', '幂律合成族划分可计算（N 提供，全同子格 → 族 0）',
      res.family_diagnostics is not None and set(res.family_diagnostics.family_r.keys()) == {'0'},
      list(res.family_diagnostics.family_r.keys()) if res.family_diagnostics else None)
check('synthetic', '幂律合成 K 接近 1.5（若包络为 power）',
      not ('K' in res.envelope_fit.params and abs(res.envelope_fit.params['K'] - 1.5) > 0.5),
      res.envelope_fit.params)

# 2. 振荡合成：幂律 × cos（符号振荡，检验振荡管线在自旋数据上可用）
C_osc = 0.25 * np.array(dists) ** (-1.5) * np.cos(0.9 * np.array(dists) + 0.3)
recs_osc = [CorrelationRecord(source, j, d, v) for j, d, v in zip(fields, dists, C_osc)]
res = SpinCorrDecayAnalyzer.analyze(recs_osc, N=200)
check('synthetic', '振荡合成不崩溃且分类合法',
      res.classification in ('oscillatory_power_law', 'oscillatory_exponential',
                             'power_law', 'exponential', 'non_monotonic',
                             'complex_decay', 'INCONCLUSIVE'),
      res.classification)

# 3. 重复距离：同距离多值（符号相反、幅度随距离衰减，与真实关联一致）
# → 组内平均 + 近端组 sign_cancel 诊断
d2 = []
v2 = []
for d in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0):
    d2 += [d, d]
    v2 += [0.25 * d ** (-1.5), -0.15 * d ** (-1.5)]
recs_dup = [CorrelationRecord(source, source + 4 + i, d, v) for i, (d, v) in enumerate(zip(d2, v2))]
res = SpinCorrDecayAnalyzer.analyze(recs_dup, N=200)
check('synthetic', '重复距离分组：n_raw=20 → n_dist=10',
      res.n_raw_points == 20 and res.n_distances == 10,
      f"n_raw={res.n_raw_points} n_dist={res.n_distances}")
sw = any('抵消' in w for w in res.warnings)
check('synthetic', '重复距离异号触发 sign_cancel 诊断', sw)

# ==========================================================================
# B. 边界与校验
# ==========================================================================
print('===== B. 边界与校验 =====')
try:
    SpinCorrDecayAnalyzer.analyze([])
    check('edge_case', '空输入 ValueError', False, '未抛异常')
except ValueError:
    check('edge_case', '空输入 ValueError', True)

# 缺字段记录（非 CorrelationRecord 结构）
class FakeRec:
    def __init__(self, d, v):
        self.distance = d
        self.correlation = v
try:
    SpinCorrDecayAnalyzer.analyze([FakeRec(1.0, 0.5), FakeRec(2.0, 0.3)], N=200)
    check('edge_case', '缺 source_point/field_point 字段 → ValueError', False, '未抛异常')
except ValueError:
    check('edge_case', '缺 source_point/field_point 字段 → ValueError', True)

# NaN 关联值
recs_nan = [CorrelationRecord(source, j, d, v) for j, d, v in zip(fields, dists, C_pow)]
recs_nan[5].correlation = np.nan
res = SpinCorrDecayAnalyzer.analyze(recs_nan, N=200)
check('edge_case', 'NaN 关联值不崩溃', res.classification in
      ('INCONCLUSIVE', 'power_law', 'exponential', 'non_monotonic', 'complex_decay'),
      res.classification)

# N=None：族信息不可用降级
res = SpinCorrDecayAnalyzer.analyze(recs_pow, N=None)
nw = any('族划分不可用' in w for w in res.warnings)
check('edge_case', 'N=None 族划分不可用 warning + 族诊断降级',
      nw and res.family_diagnostics is not None and res.family_diagnostics.insufficient,
      [w[:30] for w in res.warnings if '族' in w])

# 数据不足传递（4 个点 → 通用核心 INCONCLUSIVE 早退）
res = SpinCorrDecayAnalyzer.analyze(recs_pow[:4], N=200)
check('edge_case', '4 点数据 → 通用核心数据不足传递（INCONCLUSIVE）',
      res.classification == 'INCONCLUSIVE' and res.evidence_level == 'insufficient',
      f"{res.classification}/{res.evidence_level}")

# 关联定义不混用：Spin 分析器只读 correlation 字段（sup_corr 字段不存在也不报错）
from Physical_Quantity.Physical_Data_Record import SuperCorrRecord
sc_recs = [SuperCorrRecord((50, 51), (60, 61), d, v) for d, v in zip(dists[:6], C_pow[:6])]
try:
    SpinCorrDecayAnalyzer.analyze(sc_recs, N=200)
    check('edge_case', 'SuperCorrRecord 误输入被拦截（缺 correlation 字段）', False, '未抛异常')
except ValueError:
    check('edge_case', 'SuperCorrRecord 误输入被拦截（缺 correlation 字段）', True)

# ==========================================================================
# C. 真实 N200 数据回归
# ==========================================================================
print('===== C. 真实数据（N200 自旋关联） =====')


def load_real_spin(measure, ver):
    base = f'{ROOT}/Data_N200/t_J_3/d1_100/1500_measure_phys_qty/{measure}'
    seq = range(1, 300, 4) if ver == '49' else range(2, 300, 4)
    sz = CorrelationRecord.extract_data(
        open(f'{base}/sz{ver}sz.json', encoding='utf-8').read(), seq, 200, 1.0)
    su = CorrelationRecord.extract_data(
        open(f'{base}/sp{ver}sm.json', encoding='utf-8').read(), seq, 200, 1.0)
    sd = CorrelationRecord.extract_data(
        open(f'{base}/sm{ver}sp.json', encoding='utf-8').read(), seq, 200, 1.0)
    return CorrelationProcessor.Combined_spin_correlation(sz, su, sd)


spin_real = {
    'd1_100_A_sub': load_real_spin('measure_49', '49'),
    'd1_100_B_sub': load_real_spin('measure_50', '50'),
}
results_spin = {}
for name, data in spin_real.items():
    res = SpinCorrDecayAnalyzer.analyze(data, N=200)
    results_spin[name] = res.to_dict()
    d = results_spin[name]
    ok = (d['classification'] and isinstance(d.get('envelope_fit'), dict)
          and isinstance(d.get('oscillation'), dict)
          and isinstance(d.get('stability'), dict))
    check('real_data', f'{name} 结构完整（{len(data)} 条）', ok,
          f"cls={d['classification']}")

# 基线（自旋 v1：当前版本基线，非旧版基准）
if os.path.exists(BASELINE):
    with open(BASELINE, encoding='utf-8') as f:
        base = json.load(f)
    for name in spin_real:
        diffs = diff_paths(base.get(name, {}), results_spin[name])
        check('real_data', f'{name} 与基线精确一致', not diffs,
              f'{len(diffs)} 处差异' if diffs else '逐字段全等')
else:
    payload = dict(results_spin)
    payload['_meta'] = {
        'baseline_kind': 'spin_v1',
        'note': 'Spin_Correlation_Analyzer 首次开发完成时生成；不是任何旧版基准。',
    }
    with open(BASELINE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    check('real_data', '自旋基线已生成（v1）', True, f'-> {BASELINE}')

# ==========================================================================
# 汇总
# ==========================================================================
print()
print('=' * 60)
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in ('synthetic', 'edge_case', 'real_data'):
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    n_tot = len(RESULTS[cat])
    print(f'{cat:12}: {n_pass}/{n_tot} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)
