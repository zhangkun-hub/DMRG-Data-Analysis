# -*- coding: utf-8 -*-
"""
EnergyProcessor 单元测试（Phase B3-A2/B3-A7：主结果全点拟合 +
MAD 诊断分离）

覆盖：
- A. 主结果全点拟合（B3-A7）：明显离群点场景主结果 E_inf 与全部点
  np.polyfit 截距一致（数值断言）、离群点不被静默剔除；诊断 stdout
  明确"候选异常点"；干净线性数据不误报且 e_inf 与 np.polyfit 精确
  一致；完美共线（零残差）不误报且精确恢复真值；正常噪声全点保留；
  双离群点单轮 MAD 已知限制记录
- B. 外推失败路径：<3 记录 / 空列表 → ValueError（当前接口承诺）
- C. doping 依赖拟合：长度不匹配 → ValueError；正常数据 slope/
  intercept 与 np.polyfit 一致（数值断言，非仅调用成功）
- D. 端到端合成日志：仓库内自包含 fixture（5 个 N 合成日志文本）→
  extract_last_e0 → EnergyRecord.extract_data → extrapolate →
  e_inf 有限且符合构造值；空日志保守路径
- E. diagnose_outliers 诊断分离（B3-A7）：候选异常点识别与敏感性
  重拟合（数值级断言，非 stdout）、退化状态与比值不可定义约定、
  threshold 只影响诊断不影响主结果、字段完整性与输入校验契约

不重复 test_energy_contract_integration.py 的包装/契约断言。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_energy_process.py
"""
import contextlib
import glob
import importlib.util
import io
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Energy_Process import EnergyRecord, EnergyProcessor

# 零宽字符文件名：glob 定位后 importlib 加载（与日志分析器测试同款）
_SCRIPT = glob.glob(os.path.join(ROOT, 'DMRG_Log_Analyzer*.py'))[0]
_spec = importlib.util.spec_from_file_location('dmrg_log_analyzer', _SCRIPT)
_analyzer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyzer)
extract_last_e0 = _analyzer.extract_last_e0

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def make_recs(es_values, ns=(80, 160, 240, 320, 400), J=0.2):
    """按单格点能量 Es 与尺寸 N 构造 EnergyRecord 列表。"""
    return [EnergyRecord(N=n, E_min=es * n, E_avg=es * n, Es=es,
                         Es_J=es / J)
            for n, es in zip(ns, es_values)]


def run_extrapolate(recs, power=1, threshold=2.0):
    """运行 extrapolate 并捕获 stdout（plt.show 已 Agg 化）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e_inf = EnergyProcessor.extrapolate(recs, power, threshold)
    return e_inf, buf.getvalue()


# ======================================================================
# A. 主结果全点拟合（含 MAD 诊断输出）
# ======================================================================
print('===== A. 主结果全点拟合 =====')

# A1 注入明显离群点（Es 在 N=400 处偏离线性趋势 0.01，远超噪声）
# B3-A7：主结果恒为全点拟合——离群点只进入 MAD 诊断，不改变 E_inf
es_clean = [0.1 + 0.05 / n for n in (80, 160, 240, 320, 400)]
es_dirty = list(es_clean)
es_dirty[4] += 0.01   # N=400 离群
recs_clean = make_recs(es_clean)
recs_dirty = make_recs(es_dirty)
e_clean, out_clean = run_extrapolate(recs_clean)
e_dirty, out_dirty = run_extrapolate(recs_dirty)
# 主结果数值锚点：全部点 np.polyfit 截距（不依赖 stdout）
x_dirty = 1 / np.array([80, 160, 240, 320, 400], dtype=float)
y_dirty = np.array([r.Es for r in sorted(recs_dirty, key=lambda r: r.N)])
c_dirty_all = np.polyfit(x_dirty, y_dirty, 1)
check('A', '明显离群点：主结果 E_inf 与全部点 np.polyfit 截距一致'
      '（数值断言）',
      abs(e_dirty - c_dirty_all[1]) < 1e-12,
      f'{e_dirty:.8f} vs {c_dirty_all[1]:.8f}')
check('A', '明显离群点：离群点保留在主结果中（e_inf 相对干净值偏移'
      ' ~5e-3，未静默剔除）',
      abs(es_dirty[4] - es_clean[4]) > 1e-3
      and abs(e_dirty - e_clean) > 1e-3,
      f'{e_dirty:.8f} vs {e_clean:.8f}')
check('A', '明显离群点：诊断 stdout 明确"候选异常点"且含 N=400'
      '（诊断输出，非剔除声明）',
      '检测到候选异常点' in out_dirty and '400' in out_dirty,
      out_dirty.strip().splitlines()[0])

# A2 干净近线性数据：不误报候选点 + 主结果与全点拟合精确一致
check('A', '干净近线性数据：退化保护生效，不误报候选点'
      '（stdout 未检测到）',
      '未检测到显著候选异常点' in out_clean,
      out_clean.strip().splitlines()[0])
ns = np.array([80, 160, 240, 320, 400])
x = 1.0 / ns
y = np.array([r.Es for r in sorted(recs_clean, key=lambda r: r.N)])
c_full = np.polyfit(x, y, 1)
check('A', '干净近线性数据：e_inf 与 np.polyfit 全点结果精确一致',
      abs(e_clean - c_full[1]) < 1e-12,
      f'{e_clean:.12f} vs {c_full[1]:.12f}')

# A3 完美共线（零残差）：不崩溃 + 精确恢复真值
es_exact = [0.5 + 3.0 / n for n in (80, 160, 240, 320, 400)]
recs_exact = make_recs(es_exact)
e_exact, out_exact = run_extrapolate(recs_exact)
check('A', '完美共线：不崩溃、不误报候选点（stdout 未检测到）',
      '未检测到显著候选异常点' in out_exact)
check('A', '完美共线：e_inf 精确恢复真值 E∞=0.5（|Δ| < 1e-10）',
      abs(e_exact - 0.5) < 1e-10, f'{e_exact:.12f}')

# A4 无异常但带正常噪声的数据：不强行剔除（全点保留）
rng = np.random.default_rng(42)
es_noisy = [0.3 + 2.0 / n + rng.normal(0, 1e-6) for n in ns]
recs_noisy = make_recs(es_noisy)
e_noisy, out_noisy = run_extrapolate(recs_noisy)
check('A', '正常噪声无离群：不误报候选点（stdout 未检测到）且 e_inf 有限',
      '未检测到显著候选异常点' in out_noisy and np.isfinite(e_noisy),
      f'e_inf={e_noisy:.6f}')

# A5 多离群点（2/5）：单轮 MAD 的已知限制记录
# 40% 单侧离群使最小二乘线被拉偏、3 个正常点残差高度一致 → MAD≈0
# → 退化保护触发 → 不剔除。单轮稳健判据对 2/5 污染超出能力范围；
# 真实数据（5 点、30/30 无异常点）不受影响。如实记录当前行为。
es_two = list(es_clean)
es_two[3] += 0.005
es_two[4] += 0.008
recs_two = make_recs(es_two)
e_two, out_two = run_extrapolate(recs_two)
check('A', '双离群点（2/5）：单轮 MAD 已知限制——不报候选且主结果'
      ' e_inf 偏离干净值（限制记录，非本修复承诺范围）',
      '未检测到显著候选异常点' in out_two and abs(e_two - e_clean) > 1e-3,
      f'{e_two:.8f} vs {e_clean:.8f}')

# ======================================================================
# B. 外推失败路径
# ======================================================================
print('===== B. 外推失败路径 =====')
try:
    EnergyProcessor.extrapolate(make_recs([0.1, 0.11], ns=(80, 160)))
    check('B', '记录数 < 3 → ValueError（当前接口承诺）', False, '未抛异常')
except ValueError:
    check('B', '记录数 < 3 → ValueError（当前接口承诺）', True)
try:
    EnergyProcessor.extrapolate([])
    check('B', '空列表 → ValueError（同一路径显式验证）', False, '未抛异常')
except ValueError:
    check('B', '空列表 → ValueError（同一路径显式验证）', True)

# ======================================================================
# C. doping 依赖拟合
# ======================================================================
print('===== C. doping 依赖拟合 =====')
try:
    EnergyProcessor.fit_doping_dependence([0.1, 0.05], [-0.15])
    check('C', '长度不匹配 → ValueError（当前接口承诺）', False, '未抛异常')
except ValueError:
    check('C', '长度不匹配 → ValueError（当前接口承诺）', True)
deltas = [0.0, 0.0125, 0.025, 0.05, 0.075, 0.1]
e_infs = [-0.0937, -0.1004, -0.1072, -0.1207, -0.1339, -0.1465]
slope, intercept = EnergyProcessor.fit_doping_dependence(deltas, e_infs)
c_np = np.polyfit(np.array(deltas), np.array(e_infs), 1)
check('C', '正常数据：slope/intercept 与 np.polyfit 一致（数值断言）',
      abs(slope - c_np[0]) < 1e-12 and abs(intercept - c_np[1]) < 1e-12,
      f'slope={slope:.6f} intercept={intercept:.6f}')

# ======================================================================
# D. 端到端合成日志（仓库内自包含 fixture）
# ======================================================================
print('===== D. 端到端合成日志 =====')


def e0_line(pair, energy, d):
    i, j = pair
    return (f'Site ({i:4d},{j:4d}) E0 = {energy:.16f} '
            f'TruncErr = 1.00e-09 D = {d:5d} Iter =   2 '
            f'LanczT =     0.20 TotT =     0.30 S =  2.0000000')


def make_energy_log(N, d, es_true):
    """合成一个 N 尺寸的收敛日志：单一 (D, sweep) 组、4 条 E0 行，
    末行 E0 = es_true * N（末行是最后一次更新的全体系能量）。"""
    lines = [f'Bond dimension:\t {d}/{d}', 'sweep 1']
    for k in range(4):
        e = (es_true * N) * (1.0 + 1e-6 * (3 - k))   # 行内递减，末行=真值
        lines.append(e0_line((40 + k, 41 + k), e, d))
    lines.append('DMRG calculation finished.')
    return '\n'.join(lines) + '\n'


# 5 个 N 的合成日志 → 完整链路（模拟 Data_Energy 结构，不依赖外部数据）
J = 0.2
es_true_by_N = {n: 0.3 + 2.0 / n for n in (80, 160, 240, 320, 400)}
recs_e2e = []
for n, es_true in es_true_by_N.items():
    text = make_energy_log(n, 1500, es_true)
    last = extract_last_e0(text)
    check('D', f'N={n}：末行提取成功且等于构造真值（组末行语义）',
          last and len(last) == 1
          and abs(next(iter(last.values())) - es_true * n) < 1e-9,
          str(next(iter(last.values())) if last else None))
    recs_e2e.append(EnergyRecord.extract_data(text, n, J))
e_e2e, out_e2e = run_extrapolate(recs_e2e)
check('D', '端到端：合成日志 → 外推 e_inf 有限且接近构造 E∞=0.3',
      np.isfinite(e_e2e) and abs(e_e2e - 0.3) < 1e-3,
      f'e_inf={e_e2e:.6f}')

# 空日志保守路径
check('D', '空日志：extract_last_e0 返回空 dict（不伪造）',
      extract_last_e0('') == {})
try:
    EnergyRecord.extract_data('', 200, J)
    check('D', '无 E0 文本：EnergyRecord → ValueError（不伪造）',
          False, '未抛异常')
except ValueError:
    check('D', '无 E0 文本：EnergyRecord → ValueError（不伪造）', True)

# ======================================================================
# E. diagnose_outliers 诊断分离（B3-A7）
# ======================================================================
print('===== E. diagnose_outliers 诊断分离 =====')

# E1 单离群点：候选识别 + 敏感性重拟合（数值级，非 stdout）
d1 = EnergyProcessor.diagnose_outliers(recs_dirty)
check('E', '单离群点：masked_N 含 N=400（候选识别）',
      d1['masked_N'] == [400], str(d1['masked_N']))
check('E', '单离群点：E_inf_all_points 与 extrapolate 主结果一致',
      abs(d1['E_inf_all_points'] - e_dirty) < 1e-12,
      f"{d1['E_inf_all_points']:.8f} vs {e_dirty:.8f}")
check('E', '单离群点：E_inf_refit 收敛到干净数据真值（|Δ|<1e-6，'
      '诊断重拟合有效）',
      abs(d1['E_inf_refit'] - e_clean) < 1e-6,
      f"{d1['E_inf_refit']:.8f} vs {e_clean:.8f}")
check('E', '单离群点：sensitivity_delta 有限、自洽且方向正确'
      '（refit − all_points < 0）',
      np.isfinite(d1['sensitivity_delta'])
      and abs(d1['sensitivity_delta']
              - (d1['E_inf_refit'] - d1['E_inf_all_points'])) < 1e-15
      and d1['sensitivity_delta'] < 0,
      f"{d1['sensitivity_delta']:.3e}")

# E2 干净近线性：退化状态 + 比值不可定义约定
d2 = EnergyProcessor.diagnose_outliers(recs_clean)
check('E', '干净近线性：degenerate=True、masked_N 空、sensitivity_delta=0',
      d2['degenerate'] is True and d2['masked_N'] == []
      and d2['sensitivity_delta'] == 0.0,
      f"deg={d2['degenerate']} masked={d2['masked_N']}")
check('E', '干净近线性：尺度退化时 residual_ratios 全 None'
      '（不伪造有限比值）',
      all(v is None for v in d2['residual_ratios'].values()),
      str(d2['residual_ratios']))

# E3 完美共线：诊断数值精确恢复真值
d3 = EnergyProcessor.diagnose_outliers(recs_exact)
check('E', '完美共线：E_inf_all_points 精确恢复 E∞=0.5（|Δ|<1e-10）',
      abs(d3['E_inf_all_points'] - 0.5) < 1e-10,
      f"{d3['E_inf_all_points']:.12f}")

# E4 正常噪声：有限比值 + 无候选
d4 = EnergyProcessor.diagnose_outliers(recs_noisy)
check('E', '正常噪声：degenerate=False、无候选、比值全有限',
      d4['degenerate'] is False and d4['masked_N'] == []
      and all(np.isfinite(v) for v in d4['residual_ratios'].values()),
      str(d4['residual_ratios']))

# E5 双离群点已知限制：单轮 MAD 不报候选（与 A5 一致）
d5 = EnergyProcessor.diagnose_outliers(recs_two)
check('E', '双离群点：单轮 MAD 已知限制——masked_N 空、'
      'refit == all_points',
      d5['masked_N'] == [] and d5['sensitivity_delta'] == 0.0,
      str(d5['masked_N']))

# E6 threshold 只影响诊断、不影响主结果
e_dirty_th3, _ = run_extrapolate(recs_dirty, threshold=3.0)
d_th3 = EnergyProcessor.diagnose_outliers(recs_dirty, threshold=3.0)
check('E', 'threshold 2.0→3.0：主结果 E_inf 完全不变'
      '（拟合输入不受诊断影响）',
      e_dirty_th3 == e_dirty,
      f'{e_dirty_th3:.10f} vs {e_dirty:.10f}')
check('E', 'threshold 2.0→3.0：诊断候选列表变化（2.0 报 [400]，'
      '3.0 为空——已知灵敏度边界）',
      d1['masked_N'] == [400] and d_th3['masked_N'] == [],
      f"th2.0={d1['masked_N']} th3.0={d_th3['masked_N']}")

# E7 字段完整性与类型
d_fields = {'masked_N': list, 'residual_ratios': dict, 'scale': float,
            'degenerate': bool, 'E_inf_all_points': float,
            'E_inf_refit': float, 'sensitivity_delta': float,
            'mask': list}
check('E', '诊断 dict 字段完整且类型正确',
      all(k in d1 and isinstance(d1[k], t) for k, t in d_fields.items()),
      str(sorted(d1.keys())))
check('E', 'mask 长度与输入一致且与 masked_N 互补',
      len(d1['mask']) == 5
      and sum(d1['mask']) == 5 - len(d1['masked_N']),
      f"len={len(d1['mask'])} sum={sum(d1['mask'])}")

# E8 输入校验契约（与 extrapolate 一致）
try:
    EnergyProcessor.diagnose_outliers(make_recs([0.1, 0.11], ns=(80, 160)))
    check('E', '诊断记录数 < 3 → ValueError（与 extrapolate 同一契约）',
          False, '未抛异常')
except ValueError:
    check('E', '诊断记录数 < 3 → ValueError（与 extrapolate 同一契约）',
          True)

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
