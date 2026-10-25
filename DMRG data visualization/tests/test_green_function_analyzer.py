# -*- coding: utf-8 -*-
"""
GreenFunctionDecayAnalyzer 定向测试（Phase B2：单粒子关联衰减薄适配层）

覆盖：
- A. 合成数据（真实几何距离序列，模拟 j>i 单向结构）：
  幂律/指数/振荡×幂律/振荡×指数/常数背景/长距噪声/正负翻转/复数输入
- B. 解析异常：空数据/结构错误/越界/NaN/重复键/源点不一致
- C. 通道等价性回归（B1.1 审计规则）：真实数据 ch1==ch2、ch3==ch4
  逐记录一致；ch1≠ch3
- D. 真实数据回归：N200 d1_100/d1_10 × {1500, 5000}、N400 d1_10
  × 1500 的四通道解析与全管线执行
- E. 统一入口路由与严格 JSON

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_green_function_analyzer.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Physical_Data_Record import SingleParticleRecord
from Physical_Quantity.Green_Function_Analyzer import GreenFunctionDecayAnalyzer
from Physical_Quantity.Correlation_Decay_Analyzer import AnalyzerConfig
from Physical_Quantity.Analysis_Entry import run_analysis
from Physical_Quantity.Analysis_Contract import (
    OBSERVABLE_SINGLE_PARTICLE_CORRELATION, STATUS_OK)

RESULTS = {}

G200 = HoneycombXCGeometry(200, 1.0)
G400 = HoneycombXCGeometry(400, 1.0)
# 真实数据同款距离序列：源点 49 → 目标 50..199（N200）、97 → 98..399（N400）
R200 = np.array([G200.get_distance(49, j) for j in range(50, 200)])
R400 = np.array([G400.get_distance(97, j) for j in range(98, 400)])


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def make_recs(R, fn, src=49, imag_fn=None):
    recs = []
    for k, r in enumerate(R):
        j = src + 1 + k
        v = float(fn(r))
        if imag_fn is not None:
            v = complex(v, float(imag_fn(r)))
        recs.append(SingleParticleRecord(src, j, float(r), v))
    return recs


# ======================================================================
# A. 合成数据（真实几何距离序列）
# ======================================================================
print('===== A. 合成数据 =====')
cfg = AnalyzerConfig()

# A1 幂律
recs_pow = make_recs(R200, lambda r: 1e-3 * r ** (-2.0))
res_pow = GreenFunctionDecayAnalyzer.analyze(recs_pow, cfg, N=200)
check('A', '幂律合成：分析成功、envelope_fit 存在且形式为 K',
      res_pow.envelope_fit is not None
      and 'K' in res_pow.envelope_fit.params,
      f"{res_pow.classification}")

# A2 指数
recs_exp = make_recs(R200, lambda r: 1e-3 * np.exp(-r / 8.0))
res_exp = GreenFunctionDecayAnalyzer.analyze(recs_exp, cfg, N=200)
check('A', '指数合成：分析成功、envelope_fit 形式为 xi',
      res_exp.envelope_fit is not None
      and 'xi' in res_exp.envelope_fit.params,
      f"{res_exp.classification}")

# A3 振荡×幂律
recs_osc_p = make_recs(R200, lambda r: 1e-3 * r ** (-1.5)
                       * np.cos(1.2 * r + 0.3))
res_osc_p = GreenFunctionDecayAnalyzer.analyze(recs_osc_p, cfg, N=200)
check('A', '振荡×幂律：振荡检测运行（Q 有限或过零诊断存在）',
      np.isfinite(res_osc_p.oscillation.Q)
      or res_osc_p.oscillation.n_cross > 0,
      f"ev={res_osc_p.oscillation.evidence} Q={res_osc_p.oscillation.Q}")

# A4 振荡×指数
recs_osc_e = make_recs(R200, lambda r: 1e-3 * np.exp(-r / 12.0)
                       * np.cos(0.9 * r))
res_osc_e = GreenFunctionDecayAnalyzer.analyze(recs_osc_e, cfg, N=200)
check('A', '振荡×指数：分析完成、分类输出（不预设具体分类）',
      res_osc_e.classification in (
          'oscillatory_power_law', 'oscillatory_exponential',
          'non_monotonic', 'complex_decay', 'INCONCLUSIVE'),
      f"{res_osc_e.classification}")

# A5 加性常数背景
recs_bg = make_recs(R200, lambda r: 1e-3 * r ** (-2.0) + 2e-5)
res_bg = GreenFunctionDecayAnalyzer.analyze(recs_bg, cfg, N=200)
check('A', '常数背景：分析完成、噪声底分量记录（tail_median 可区分背景）',
      res_bg.initial_noise_floor is not None
      and res_bg.initial_noise_floor_components is not None,
      f"floor={res_bg.initial_noise_floor:.2e}")

# A6 长距噪声底
def pow_with_noise(r):
    base = 1e-3 * r ** (-2.0)
    if r > 40:
        return base + np.sin(13.7 * r) * 1e-8   # 尾部扰动
    return base
recs_noise = make_recs(R200, pow_with_noise)
res_noise = GreenFunctionDecayAnalyzer.analyze(recs_noise, cfg, N=200)
check('A', '长距噪声底：分析完成、r_max 截断机制可运行（r_max_used ≤ 全距）',
      res_noise.r_max <= np.max(R200) + 1e-9
      and res_noise.n_points > 0,
      f"r_max={res_noise.r_max:.1f}")

# A7 正负翻转与近零
recs_flip = make_recs(R200, lambda r: 1e-3 * r ** (-2.0)
                      * (1.0 if r < 25 else -1.0))
res_flip = GreenFunctionDecayAnalyzer.analyze(recs_flip, cfg, N=200)
check('A', '正负翻转：分析完成、|C| 包络路径可运行（符号由原始值处理）',
      res_flip.envelope_fit is not None
      or res_flip.classification == 'INCONCLUSIVE',
      f"{res_flip.classification}")

# A8 不规则距离与几何分组（N400 距离序列；圆柱 y 周期使不同 j 可
# 同距离 → 302 条记录对应 226 个唯一距离）
recs_geo = make_recs(R400, lambda r: 1e-3 * r ** (-2.0), src=97)
res_geo = GreenFunctionDecayAnalyzer.analyze(recs_geo, cfg, N=400)
check('A', 'N400 几何距离序列：分组/分析正常（302 记录 → 226 唯一距离）',
      res_geo.n_distances > 200,
      f"n_distances={res_geo.n_distances}")

# A9 复数值输入（虚部保留在 .value，分析消费实部）
recs_cplx = make_recs(R200, lambda r: 1e-3 * r ** (-2.0),
                      imag_fn=lambda r: 0.5e-3 * r ** (-2.0))
check('A', '复数值输入：value 保留虚部、correlation 提供实部',
      isinstance(recs_cplx[0].value, complex)
      and recs_cplx[0].value.imag > 0
      and abs(recs_cplx[0].correlation - recs_cplx[0].value.real) < 1e-30)
res_cplx = GreenFunctionDecayAnalyzer.analyze(recs_cplx, cfg, N=200)
check('A', '复数值输入：分析管线消费实部正常执行',
      res_cplx.envelope_fit is not None
      and np.isfinite(res_cplx.envelope_fit.params.get('A', np.nan)))

# ======================================================================
# B. 解析异常（严格拒绝语义）
# ======================================================================
print('===== B. 解析异常 =====')


def parse(text, N=200):
    return SingleParticleRecord.extract_data(text, N)


check('B', '正常解析：记录数/字段/距离正确（j>i 单向全量）',
      (lambda recs: len(recs) == 3
       and recs[0].source_point == 49 and recs[0].field_point == 50
       and recs[0].distance == G200.get_distance(49, 50)
       and recs[0].value == complex(0.5, 0.0))(
          parse('[[[49, 50], [0.5, 0]], [[49, 51], [0.2, 0]], '
                '[[49, 52], [0.1, 0]]]')))
try:
    parse('')
    check('B', '空数据 → ValueError', False, '未抛异常')
except ValueError:
    check('B', '空数据 → ValueError', True)
try:
    parse('[[[49], [0.5, 0]]]')
    check('B', '结构错误 → ValueError（源点提取失败）', False, '未抛异常')
except (ValueError, NameError):
    check('B', '结构错误 → ValueError/NameError（源点提取失败）', True)
try:
    parse('[[[49, 50, 1], [0.5, 0]]]')
    check('B', '条目形状错误 → NameError（既有风格）', False, '未抛异常')
except NameError:
    check('B', '条目形状错误 → NameError（既有风格）', True)
try:
    parse('[[[49, 50], [0.5, 0]], [[48, 51], [0.2, 0]]]')
    check('B', '源点不一致 → ValueError（含预期/实际源点）', False, '未抛异常')
except ValueError as e:
    check('B', '源点不一致 → ValueError（含预期/实际源点）',
          '49' in str(e) and '48' in str(e))
try:
    parse('[[[49, 200], [0.5, 0]]]')
    check('B', '索引越界 → ValueError', False, '未抛异常')
except ValueError as e:
    check('B', '索引越界 → ValueError', '越界' in str(e))
try:
    parse('[[[49, 50], [0.5, 0]], [[49, 50], [0.2, 0]]]')
    check('B', '重复键 → ValueError', False, '未抛异常')
except ValueError as e:
    check('B', '重复键 → ValueError', '重复' in str(e))
try:
    parse('[[[49, 50], [1e400, 0]]]')
    check('B', '非有限值 → ValueError（不静默替换）', False, '未抛异常')
except ValueError as e:
    check('B', '非有限值 → ValueError（不静默替换）', '非有限' in str(e))

# ======================================================================
# C. 通道等价性回归（B1.1 审计规则：绝对 1e-14 / 相对 1e-12）
# ======================================================================
print('===== C. 通道等价性回归 =====')
DATA = os.path.join(ROOT, 'Data_N200', 't_J_3', 'd1_100',
                    '1500_measure_phys_qty', 'measure_49')
ch = {}
for c in range(1, 5):
    p = os.path.join(DATA, f'single_particle_corr{c}.json')
    with open(p, encoding='utf-8') as f:
        raw = f.read()
    recs = SingleParticleRecord.extract_data(raw, 200)
    ch[c] = {tuple([r.source_point, r.field_point]): r.value
             for r in recs}
TOL_ABS, TOL_REL = 1e-14, 1e-12


def pair_equal(a, b):
    if set(ch[a]) != set(ch[b]):
        return False
    for k in ch[a]:
        va, vb = ch[a][k], ch[b][k]
        d = abs(va - vb)
        rel = d / max(abs(va), abs(vb)) if max(abs(va), abs(vb)) > 0 else 0
        if d > TOL_ABS and rel > TOL_REL:
            return False
    return True


check('C', 'ch1==ch2 键集合与逐记录一致（经验事实，不编码为恒等式）',
      pair_equal(1, 2))
check('C', 'ch3==ch4 键集合与逐记录一致',
      pair_equal(3, 4))
check('C', 'ch1≠ch3（上下自旋通道物理不同，不错误等同）',
      not pair_equal(1, 3))

# ======================================================================
# D. 真实数据回归（代表性参数点 × 四通道独立分析）
# ======================================================================
print('===== D. 真实数据回归 =====')
POINTS = [('Data_N200', 'd1_100', '1500', 'measure_49', 200),
          ('Data_N200', 'd1_100', '5000', 'measure_49', 200),
          ('Data_N200', 'd1_10', '1500', 'measure_49', 200),
          ('Data_N200', 'd1_10', '5000', 'measure_49', 200),
          ('Data_N400', 'd1_10', '1500', 'measure_97', 400)]
for base, dop, bd, ver, N in POINTS:
    for c in range(1, 5):
        p = os.path.join(ROOT, base, 't_J_3', dop,
                         f'{bd}_measure_phys_qty', ver,
                         f'single_particle_corr{c}.json')
        with open(p, encoding='utf-8') as f:
            raw = f.read()
        recs = SingleParticleRecord.extract_data(raw, N)
        res = GreenFunctionDecayAnalyzer.analyze(recs, AnalyzerConfig(), N=N)
        ok = (len(recs) > 0 and res.n_distances > 0
              and res.envelope_fit is not None
              and np.isfinite(res.r_min) and np.isfinite(res.r_max))
        check('D', f'{dop}/{bd}/{ver} ch{c}：解析 {len(recs)} 条、'
              '全管线执行（窗口/包络有限）', ok,
              f"n_dist={res.n_distances} rwin=({res.r_min:.1f},"
              f"{res.r_max:.1f}) cls={res.classification}")

# ======================================================================
# E. 统一入口路由 + 严格 JSON
# ======================================================================
print('===== E. 统一入口 =====')
p0 = os.path.join(ROOT, 'Data_N200', 't_J_3', 'd1_100',
                  '1500_measure_phys_qty', 'measure_49',
                  'single_particle_corr1.json')
with open(p0, encoding='utf-8') as f:
    raw0 = f.read()
recs0 = SingleParticleRecord.extract_data(raw0, 200)
rep = run_analysis(OBSERVABLE_SINGLE_PARTICLE_CORRELATION,
                   record_list=recs0, N=200)
check('E', '统一入口路由：status=ok、observable 正确',
      rep.status == STATUS_OK
      and rep.observable == OBSERVABLE_SINGLE_PARTICLE_CORRELATION
      and rep.classification is not None,
      f"{rep.status} {rep.classification}")
js = json.dumps(rep.to_dict(), ensure_ascii=False, default=str)
check('E', 'AnalysisReport 严格 JSON 可序列化且读回一致',
      'NaN' not in js and 'Infinity' not in js
      and json.loads(js)['status'] == STATUS_OK)

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
