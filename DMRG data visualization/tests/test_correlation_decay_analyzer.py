# -*- coding: utf-8 -*-
"""
Correlation_Decay_Analyzer 重构后正式回归测试

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_correlation_decay_analyzer.py

测试分三类（报告分别统计）：
  - real_data：真实 N200 超导关联数据（结构完整性 + 当前版本基线对比）
  - synthetic：合成数据分类断言（零背景 / 平台判别 / 纯衰减 / hump 误报率）
  - edge_case：边界与异常输入（空输入 / NaN / Inf / 全零 / 点不足 / 异号 / 单族）

基线说明（诚实声明）：
  baselines/n200_baseline_post_refactor.json 是重构后首次运行生成的
  **当前版本基线**（POST-REFACTOR BASELINE），不是重构前基准——
  重构前基准快照已删除且无法重建。该基线用于之后的回归比较，
  不能用于宣称"重构前后逐字段等价"（该结论由重构当次的 22 组
  一次性比较得出，随临时快照清理后不可重现）。

to_dict 比较口径：
  - 数值字段：默认精确比较（==）；浮点可选容差比较由调用方决定
  - 字符串 / 字典键 / 列表结构：精确比较（不允许任何差异）

注意：本目录被 .gitignore 的 *test* 规则忽略，纳入版本控制需调整规则。
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # DMRG data visualization/
sys.path.insert(0, ROOT)

from Physical_Quantity.Physical_Data_Record import SuperCorrRecord
from Physical_Quantity.Super_Correlation_Process import SuperCorrProcessor
from Physical_Quantity.Super_Correlation_Analyzer import (
    SuperCorrDecayAnalyzer, AnalyzerConfig)

BASELINE = os.path.join(HERE, 'baselines', 'n200_baseline_post_refactor.json')
RESULTS = {'real_data': [], 'synthetic': [], 'edge_case': []}


def recs(r, C):
    return [SuperCorrRecord((0, 1), (2, 3), float(d), float(v)) for d, v in zip(r, C)]


def load_n200(doping, ref, bond, N=200):
    base = f'{ROOT}/Data_N200/t_J_3/{doping}/1500_measure_super_corr/super_corr_{ref}'
    recs4 = {}
    for ch in 'abcd':
        recs4[ch] = SuperCorrRecord.extract_data(
            open(f'{base}/scs{bond}{ch}_fix_refbond_{ref}.json', encoding='utf-8').read(),
            N, 1.0)
    return SuperCorrProcessor.Combined_super_correlation(
        recs4['a'], recs4['b'], recs4['c'], recs4['d'])


def check(category, name, cond, detail=''):
    RESULTS[category].append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def diff_paths(a, b, path=''):
    """递归精确比较：数值全等、字符串全等、键集与列表结构一致。
    返回差异路径列表（空 = 完全一致）。"""
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
# A. 真实数据：N200（结构完整性 + 基线）
# ==========================================================================
print('===== A. 真实数据（N200） =====')
REAL_CASES = [
    # (名字, doping, refbond, bond, config)
    # 覆盖：4 种 doping × 3 种 refbond × A/B/C 方向 × 振荡强/中/无 × 11/12 点 × all_points/both
    ('d1_10_49_50_A', 'd1_10', '49_50', 'A', AnalyzerConfig(envelope_mode='all_points')),   # 强振荡
    ('d1_10_50_54_C', 'd1_10', '50_54', 'C', AnalyzerConfig(envelope_mode='all_points')),   # 11 点强振荡
    ('d1_100_49_50_A', 'd1_100', '49_50', 'A', AnalyzerConfig(envelope_mode='all_points')), # 无振荡强 hump
    ('d1_100_50_51_A', 'd1_100', '50_51', 'A', AnalyzerConfig(envelope_mode='both')),       # 无振荡弱 hump
    ('d1_100_50_51_C', 'd1_100', '50_51', 'C', AnalyzerConfig(envelope_mode='both')),       # 无振荡更弱 hump
    ('d1_100_50_54_C', 'd1_100', '50_54', 'C', AnalyzerConfig(envelope_mode='both')),       # 11 点 hump 卡线场景
    ('d1_20_49_50_A', 'd1_20', '49_50', 'A', AnalyzerConfig(envelope_mode='both')),         # 中振荡 + both 退化
    ('d2_25_50_51_B', 'd2_25', '50_51', 'B', AnalyzerConfig(envelope_mode='all_points')),   # 另一 doping 强振荡
]
results_real = {}
for name, doping, ref, bond, cfg in REAL_CASES:
    data = load_n200(doping, ref, bond)
    res = SuperCorrDecayAnalyzer.analyze(data, cfg)
    results_real[name] = res.to_dict()
    # 结构完整性：关键字段非缺失且数值有限
    d = results_real[name]
    ok_fields = (d['classification'] and isinstance(d.get('envelope_fit'), dict)
                 and isinstance(d.get('oscillation'), dict)
                 and isinstance(d.get('stability'), dict)
                 and d.get('n_points', 0) > 0)
    aicc = d.get('envelope_fit', {}).get('aicc')
    finite = np.isfinite(aicc) if isinstance(aicc, (int, float)) else True
    check('real_data', f'{name} 结构完整+包络有限',
          ok_fields and finite,
          f"cls={d['classification']} n={d.get('n_points')}")

# 基线处理：首次运行生成 POST-REFACTOR BASELINE，之后精确比较
if os.path.exists(BASELINE):
    with open(BASELINE, encoding='utf-8') as f:
        base = json.load(f)
    if base.get('_meta', {}).get('baseline_kind') != 'post_refactor':
        check('real_data', '基线元数据正确', False, '缺少 post_refactor 标注')
    for name in REAL_CASES:
        n = name[0]
        diffs = diff_paths(base.get(n, {}), results_real[n])
        check('real_data', f'{n} 与基线精确一致', not diffs,
              f'{len(diffs)} 处差异' if diffs else '逐字段全等')
else:
    print('  基线文件不存在：生成 POST-REFACTOR BASELINE（非重构前基准）')
    payload = {c[0]: results_real[c[0]] for c in REAL_CASES}
    payload['_meta'] = {
        'baseline_kind': 'post_refactor',
        'note': '重构后首次运行生成。重构前基准快照已删除、无法重建；'
                '本基线用于之后回归，不构成重构前后等价性证明。',
        'cases': [c[0] for c in REAL_CASES],
    }
    with open(BASELINE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    check('real_data', '基线已生成（POST-REFACTOR）', True, f'-> {BASELINE}')

# ==========================================================================
# B. 合成数据：分类断言（重构前的合成矩阵固化为正式断言）
# ==========================================================================
print('===== B. 合成数据 =====')
r = np.arange(3.0, 76.0, 3.0)
for kind, expect in [('constant', 'constant'), ('power', 'power_law'),
                     ('exp', 'exponential'), ('osc_power', 'oscillatory_power_law'),
                     ('osc_exp', 'oscillatory_exponential')]:
    gg = np.random.default_rng(42)
    if kind == 'constant':
        base = np.full_like(r, 1e-4)
    elif kind == 'power':
        base = 1e-4 * r ** (-2.0)
    elif kind == 'exp':
        base = 1e-4 * np.exp(-r / 15.0)
    elif kind == 'osc_power':
        base = 1e-4 * r ** (-2.0) * np.cos(0.86 * r + 0.3)
    else:
        base = 1e-4 * np.exp(-r / 15.0) * np.cos(0.86 * r + 0.3)
    C = base + 0.05 * np.abs(base) * gg.standard_normal(len(r))
    res = SuperCorrDecayAnalyzer.analyze(recs(r, C))
    check('synthetic', f'零背景 {kind} -> {expect}', res.classification == expect,
          res.classification)

K, A = 2.0, 1e-4
tail = A * 75.0 ** (-K)
det, wrong = 0, 0
for ratio in (1.0, 3.0):
    for seed in range(5):
        gg = np.random.default_rng(100 + seed)
        noise = 0.05 + 0.10 * (seed % 5) / 4
        C = ratio * tail + A * r ** (-K) + noise * np.abs(A * r ** (-K)) * gg.standard_normal(len(r))
        res = SuperCorrDecayAnalyzer.analyze(recs(r, C))
        bw = [w for w in res.warnings if 'C_inf' in w and '胜出' in w and '与零一致' not in w]
        if bw:
            det += 1
        elif res.classification in ('power_law', 'exponential'):
            wrong += 1
check('synthetic', '平台判别 检出≥8/10 且明确误判 0', det >= 8 and wrong == 0,
      f'检出 {det}/10 误判 {wrong}')

fp = 0
for seed in range(5):
    gg = np.random.default_rng(200 + seed)
    C = A * r ** (-K) + 0.08 * np.abs(A * r ** (-K)) * gg.standard_normal(len(r))
    res = SuperCorrDecayAnalyzer.analyze(recs(r, C))
    bw = [w for w in res.warnings if 'C_inf' in w and '胜出' in w and '与零一致' not in w]
    if bw:
        fp += 1
check('synthetic', '纯衰减无背景假阳性', fp == 0, f'{fp}/5')

# hump 误报率矩阵（power/exp/振荡 × 噪声 0-15% × 种子）
false_hump = 0
total = 0
for kind, base in [('power', 1e-4 * r ** (-2.0)), ('exp', 1e-4 * np.exp(-r / 15.0))]:
    for noise in (0.0, 0.05, 0.10, 0.15):
        for seed in range(6):
            gg = np.random.default_rng(seed + 100)
            C = base * (1 + noise * gg.standard_normal(len(r)))
            h, _ = SuperCorrDecayAnalyzer._envelope_hump(r, C)
            false_hump += h
            total += 1
for Q, phi in ((0.5, 0.2), (0.86, 0.3), (1.2, 1.0), (2.0, 0.0)):
    for noise in (0.0, 0.05, 0.10, 0.15):
        for seed in range(6):
            gg = np.random.default_rng(seed + 200)
            C = np.abs(1e-4 * r ** (-2.0) * np.cos(Q * r + phi)) * (1 + noise * gg.standard_normal(len(r)))
            h, _ = SuperCorrDecayAnalyzer._envelope_hump(r, C)
            false_hump += h
            total += 1
check('synthetic', 'hump 误报率 = 0', false_hump == 0, f'{false_hump}/{total}')

# 峰后不足路径（显式形态构造：满足 v3 hump 判据——上升段 3 段连续正斜率、
# 峰后 3 点全负且 |斜率| 严格递增、相对谷值上升 ≥5%、峰 ≥50% 全局最大；
# 但峰后仅 3 点 < n_min_hump_window=6 → insufficient）
r12 = np.arange(1.5, 36.0, 3.0)
C_hump = np.array([5.0, 4.4, 3.8, 3.2, 2.6, 1.9, 2.1, 2.3, 2.7, 2.2, 1.75, 1.35]) * 1e-6
h, p = SuperCorrDecayAnalyzer._envelope_hump(r12, C_hump)
res = SuperCorrDecayAnalyzer.analyze(recs(r12, C_hump))
check('synthetic', '峰后不足 -> hump 检出 + verdict=insufficient + 推荐 NaN',
      h and res.stability.verdict == 'insufficient'
      and not np.isfinite(res.stability.recommended_r_min),
      f"hump={h} peak={p} verdict={res.stability.verdict}")

# ==========================================================================
# C. 边界与异常输入
# ==========================================================================
print('===== C. 边界与异常输入 =====')
try:
    SuperCorrDecayAnalyzer.analyze([])
    check('edge_case', '空输入 ValueError', False, '未抛异常')
except ValueError:
    check('edge_case', '空输入 ValueError', True)

C_nan = 1e-4 * r ** (-2.0)
C_nan = C_nan.copy()
C_nan[3] = np.nan
res = SuperCorrDecayAnalyzer.analyze(recs(r, C_nan))
check('edge_case', 'NaN 数据不崩溃且 INCONCLUSIVE', res.classification == 'INCONCLUSIVE',
      res.classification)

C_inf = 1e-4 * r ** (-2.0)
C_inf = C_inf.copy()
C_inf[3] = np.inf
res = SuperCorrDecayAnalyzer.analyze(recs(r, C_inf))
check('edge_case', 'Inf 数据不崩溃', res.classification in
      ('INCONCLUSIVE', 'non_monotonic', 'complex_decay'), res.classification)

res = SuperCorrDecayAnalyzer.analyze(recs(r, np.zeros_like(r)))
check('edge_case', '全零数据不崩溃且 INCONCLUSIVE（P2 修复）',
      res.classification == 'INCONCLUSIVE', res.classification)

res = SuperCorrDecayAnalyzer.analyze(recs(r, np.full_like(r, np.nan)))
check('edge_case', '全 NaN 数据不崩溃且 INCONCLUSIVE',
      res.classification == 'INCONCLUSIVE', res.classification)

res = SuperCorrDecayAnalyzer.analyze(recs(r[:3], 1e-4 * r[:3] ** (-2.0)))
check('edge_case', '3 点数据 -> INCONCLUSIVE', res.classification == 'INCONCLUSIVE',
      res.classification)

base = 1e-4 * r ** (-2.0)
rr = np.concatenate([r, r])
CC = np.concatenate([base, -0.5 * base])
res = SuperCorrDecayAnalyzer.analyze(recs(rr, CC))
sw = any('抵消' in w for w in res.warnings)
check('edge_case', '同距离异号触发 sign_cancel 诊断', sw)

res = SuperCorrDecayAnalyzer.analyze(recs(rr, CC))
fd = res.family_diagnostics
check('edge_case', '单族数据族诊断 insufficient 降级', fd is not None and fd.insufficient)

# P1：旧模块路径 re-export
from Physical_Quantity.Super_Correlation_Analyzer import (  # noqa: E402
    _fmt_sci, _rel_mad, _mad_abs, _mad_circular, _max_circular,
    _r2_in_space, _scaled_fit_problem, _aicc_bic_core, _logspace_lstsq,
    _local_slopes, _spearman)
check('edge_case', 'P1 旧路径私有 helper re-export', True)

# ==========================================================================
# 汇总
# ==========================================================================
print()
print('=' * 60)
total_pass = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if ok)
total_fail = sum(1 for cat in RESULTS for _, ok in RESULTS[cat] if not ok)
for cat in ('real_data', 'synthetic', 'edge_case'):
    n_pass = sum(1 for _, ok in RESULTS[cat] if ok)
    n_tot = len(RESULTS[cat])
    print(f'{cat:12}: {n_pass}/{n_tot} 通过')
print('=' * 60)
print(f'总计: {total_pass} 通过 / {total_fail} 失败')
sys.exit(1 if total_fail else 0)
