# -*- coding: utf-8 -*-
"""
Spin_Structure_Factor_Analyzer 峰值分析测试

核对具体数值与状态：峰位（原始/细化）、峰高、FWHM、分辨率限制、
边界/近邻峰标记、平坦/无效信号、元数据保留、细化回退、网格校验。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_spin_structure_factor_analyzer.py
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Spin_Structure_Factor_Analyzer import (
    SpinStructureFactorPeakAnalyzer)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


def make_sf(qx, qy, S2d, **meta):
    """构造 sf_result-like dict（q 嵌套排序：qx 外层）"""
    q = np.array([(x, y) for x in qx for y in qy])
    S = S2d.reshape(-1)
    d = {'q': q, 'S_q': S.astype(complex), 'definition': 'test', 'N': 8,
         'is_restricted': meta.get('is_restricted', False)}
    for k in ('pair_coverage', 'source_coverage', 'expected_pair_count'):
        if k in meta:
            d[k] = meta[k]
    return d


def gauss2d(qx, qy, A, x0, y0, sx, sy):
    XX, YY = np.meshgrid(qx, qy, indexing='ij')
    return A * np.exp(-((XX - x0) ** 2 / (2 * sx ** 2)
                        + (YY - y0) ** 2 / (2 * sy ** 2)))


# ==========================================================================
# 1. 已知单峰高斯（数值对照）
# ==========================================================================
print('===== A. 单峰高斯 =====')
qx = np.linspace(-3.0, 3.0, 41)
qy = np.linspace(-2.0, 2.0, 41)
dq_x = qx[1] - qx[0]
S2d = gauss2d(qx, qy, 3.0, -0.2, 0.4, 0.8, 0.7)   # σ 足够小，半高不触网格边界
sf = make_sf(qx, qy, S2d)
rep = SpinStructureFactorPeakAnalyzer.find_peaks(sf)
p = rep['peaks'][0]
FWHM_EX = 2 * np.sqrt(2 * np.log(2))   # 2.3548
# 原始网格峰高 = 最近网格点的采样值（中心 -0.2 → 最近网格 -0.15，偏移 0.05）
exp_height = 3.0 * np.exp(-0.05 ** 2 / (2 * 0.8 ** 2))
check('A', '单峰 n_peaks=1', rep['n_peaks'] == 1, rep['n_peaks'])
check('A', '峰高 = 网格采样值（非真峰值）', abs(p['height'] - exp_height) < 1e-9,
      f"{p['height']} vs {exp_height}")
check('A', '原始峰位 = 最近网格点',
      abs(p['q_original'][0] - (-0.15)) < 1e-9
      and abs(p['q_original'][1] - 0.4) < 1e-9,
      p['q_original'])
check('A', '细化峰位接近真值（<0.2 步长）',
      p['refine_status'] == 'refined'
      and abs(p['q_refined'][0] - (-0.2)) < 0.2 * dq_x
      and abs(p['q_refined'][1] - 0.4) < 0.2 * dq_x,
      f"{p['q_refined']} vs (-0.2, 0.4)")
check('A', 'fwhm_qx ≈ 2.3548×0.8（±1 步长）',
      p['fwhm_qx'] is not None and abs(p['fwhm_qx'] - FWHM_EX * 0.8) < dq_x,
      p['fwhm_qx'])
check('A', 'fwhm_qy ≈ 2.3548×0.7（±1 步长）',
      p['fwhm_qy'] is not None and abs(p['fwhm_qy'] - FWHM_EX * 0.7) < dq_x,
      p['fwhm_qy'])
check('A', '非边界峰', not p['boundary'])

# ==========================================================================
# 2. 双峰与近邻峰风险
# ==========================================================================
print('===== B. 双峰 =====')
qx = np.linspace(-5.0, 5.0, 101)
qy = np.linspace(-3.0, 3.0, 61)
S2d = gauss2d(qx, qy, 2.0, -2.0, 0.0, 0.6, 0.6) + gauss2d(qx, qy, 2.0, 2.0, 0.0, 0.6, 0.6)
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, S2d))
check('B', '双峰分离 4.0 → 2 个峰', rep['n_peaks'] == 2, rep['n_peaks'])
check('B', '分离远 → 无 merged_risk',
      rep['n_peaks'] == 2 and all(not p['merged_risk'] for p in rep['peaks']))
# 近邻峰：两可分辨局部极大（σ=0.06）分离 0.2（2 步长）< 2×对角步长 0.283
S2d_c = gauss2d(qx, qy, 2.0, -0.1, 0.0, 0.06, 0.06) + gauss2d(qx, qy, 2.0, 0.1, 0.0, 0.06, 0.06)
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, S2d_c))
risk = any(p['merged_risk'] for p in rep['peaks'])
check('B', '可分辨双峰距离 < 2×对角步长 → merged_risk 报告',
      rep['n_peaks'] == 2 and risk, f"n={rep['n_peaks']} risk={risk}")

# ==========================================================================
# 3. 边界峰
# ==========================================================================
print('===== C. 边界峰 =====')
qx = np.linspace(-3.0, 3.0, 41)
qy = np.linspace(-2.0, 2.0, 41)
S2d = gauss2d(qx, qy, 2.0, -3.0, 0.0, 0.8, 0.8)   # 中心在 qx 左边界
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, S2d))
p = rep['peaks'][0]
check('C', '边界峰 boundary=True', p['boundary'], p['boundary'])
check('C', '边界峰跳过细化（skipped_boundary）', p['refine_status'] == 'skipped_boundary',
      p['refine_status'])
check('C', '细化回退为原始峰位', p['q_refined'] == p['q_original'])

# ==========================================================================
# 4. 平坦 / 全零 / 近零 / NaN
# ==========================================================================
print('===== D. 平坦与无效信号 =====')
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, np.zeros((41, 41))))
check('D', '全零 → status=all_zero 且 0 峰', rep['status'] == 'all_zero' and rep['n_peaks'] == 0,
      rep['status'])
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, np.full((41, 41), 1.0)))
check('D', '平坦 → status=flat 且 0 峰', rep['status'] == 'flat' and rep['n_peaks'] == 0,
      rep['status'])
S2d = gauss2d(qx, qy, 3.0, 0.0, 0.0, 1.0, 1.0)
S2d[10, 10] = np.nan   # NaN 在远离峰处
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, S2d))
p = rep['peaks'][0]
check('D', '部分 NaN → 不产生虚假峰（峰位仍在真峰附近）且 invalid 计数=1',
      rep['n_peaks'] == 1 and rep['invalid_signal_points'] == 1
      and abs(p['q_original'][0]) < 0.5 and abs(p['q_original'][1]) < 0.5,
      f"n={rep['n_peaks']} invalid={rep['invalid_signal_points']} peak={p['q_original']}")
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, np.full((41, 41), np.nan)))
check('D', '全 NaN → status=all_invalid 且 0 峰',
      rep['status'] == 'all_invalid' and rep['n_peaks'] == 0, rep['status'])

# ==========================================================================
# 5. 不同分辨率
# ==========================================================================
print('===== E. 分辨率 =====')
qx_c = np.linspace(-3.0, 3.0, 13)   # 步长 0.5
qy_c = np.linspace(-2.0, 2.0, 9)
S2d = gauss2d(qx_c, qy_c, 2.0, 0.1, 0.0, 0.25, 0.25)   # FWHM ≈ 0.59 略大于步长
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx_c, qy_c, S2d))
p = rep['peaks'][0]
check('E', '粗网格细化峰位在步长内', p['refine_status'] == 'refined'
      and abs(p['q_refined'][0] - 0.1) < 0.5, p['q_refined'])
qx_f = np.linspace(-3.0, 3.0, 61)   # 步长 0.1
qy_f = np.linspace(-2.0, 2.0, 41)
S2d_f = gauss2d(qx_f, qy_f, 2.0, 0.1, 0.0, 0.25, 0.25)
rep_f = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx_f, qy_f, S2d_f))
p_f = rep_f['peaks'][0]
check('E', '细网格细化峰位更准（<0.1 步长）',
      abs(p_f['q_refined'][0] - 0.1) < 0.1, p_f['q_refined'])
# 峰宽小于采样步长 → resolution_limited
qx_vc = np.linspace(-3.0, 3.0, 13)
S2d_t = gauss2d(qx_vc, qy_c, 2.0, 0.0, 0.0, 0.15, 0.15)   # FWHM≈0.35 < 0.5
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx_vc, qy_c, S2d_t))
check('E', '峰宽小于步长 → resolution_limited=True',
      rep['peaks'][0]['resolution_limited'], rep['peaks'][0]['fwhm_qx'])

# ==========================================================================
# 6. 复数输入：abs vs real
# ==========================================================================
print('===== F. 复数信号 =====')
S2d = gauss2d(qx, qy, 3.0, 0.0, 0.0, 1.0, 1.0)
S_complex = S2d * (1 + 0.5j)
sf = make_sf(qx, qy, np.real(S_complex))
sf['S_q'] = S_complex.reshape(-1)
rep_abs = SpinStructureFactorPeakAnalyzer.find_peaks(sf, signal='abs')
rep_real = SpinStructureFactorPeakAnalyzer.find_peaks(sf, signal='real')
check('F', 'abs 峰高 = |1+0.5i|×3 = √1.25×3',
      abs(rep_abs['peaks'][0]['height'] - np.sqrt(1.25) * 3.0) < 1e-9,
      rep_abs['peaks'][0]['height'])
check('F', 'real 峰高 = 3.0',
      abs(rep_real['peaks'][0]['height'] - 3.0) < 1e-9, rep_real['peaks'][0]['height'])
check('F', 'signal_used 分别记录', rep_abs['signal_used'] == 'abs'
      and rep_real['signal_used'] == 'real')

# ==========================================================================
# 7. 受限元数据保留
# ==========================================================================
print('===== G. 元数据 =====')
sf = make_sf(qx, qy, S2d, is_restricted=True, source_coverage=0.005)
rep = SpinStructureFactorPeakAnalyzer.find_peaks(sf)
check('G', 'is_restricted=True 原样保留', rep['is_restricted'] is True)
check('G', 'source_coverage 透传', abs(rep['source_coverage'] - 0.005) < 1e-15)
check('G', 'pair_coverage 缺失保持缺失（无该键）', 'pair_coverage' not in rep)
check('G', 'definition 透传', rep['definition'] == 'test')

# ==========================================================================
# 8. 细化失败回退
# ==========================================================================
print('===== H. 细化回退 =====')
# 顶点越界：手工构造强不对称邻域——中心为局部极大，但抛物面顶点被
# 单侧强拉升推离中心超过 0.75 步长
S2d_n = np.full((41, 41), 0.1)
i0, j0 = 20, 30
S2d_n[i0, j0] = 3.0
S2d_n[i0 - 1, j0] = 2.9
S2d_n[i0 + 1, j0] = 2.9
S2d_n[i0 - 1, j0 - 1] = 2.8
S2d_n[i0 - 1, j0 + 1] = 2.95
S2d_n[i0 + 1, j0 - 1] = 2.95
S2d_n[i0 + 1, j0 + 1] = 2.85
S2d_n[i0, j0 - 1] = 2.98
S2d_n[i0, j0 + 1] = 1.2   # 单侧大幅压低 → 顶点偏移
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, S2d_n))
bad = [p for p in rep['peaks']
       if p['refine_status'] in ('vertex_out_of_range', 'fit_failed', 'not_concave')]
check('H', '顶点越界峰正确回退（回退值为原始峰位）',
      len(bad) >= 1 and all(p['q_refined'] == p['q_original'] for p in bad),
      [p['refine_status'] for p in rep['peaks']])
# 对照：正常高斯峰细化成功（A 节已覆盖，此处补一个同网格对照）
S2d_ok = gauss2d(qx, qy, 2.0, 0.0, 0.0, 0.8, 0.8)
rep_ok = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx, qy, S2d_ok))
check('H', '正常高斯峰细化成功（对照）',
      rep_ok['peaks'][0]['refine_status'] == 'refined',
      rep_ok['peaks'][0]['refine_status'])

# ==========================================================================
# 9. 不规则网格明确失败
# ==========================================================================
print('===== I. 网格校验 =====')
q_bad = np.array([(x, y) for x in qx for y in qy])[:-1]   # 删一点 → 无法构成规则网格
sf_bad = {'q': q_bad, 'S_q': S2d.reshape(-1)[:-1].astype(complex),
          'definition': 'test', 'N': 8, 'is_restricted': False}
try:
    SpinStructureFactorPeakAnalyzer.find_peaks(sf_bad)
    check('I', '不规则网格抛 ValueError', False, '未抛异常')
except ValueError:
    check('I', '不规则网格抛 ValueError', True)
# 非等距矩形网格：规则但步长不均 → 跳过细化并保留原始峰位
qx_ne = np.array(sorted(set(np.concatenate([np.linspace(-3, 0, 16), np.linspace(0.4, 3, 14)]))))
qy_ne = np.linspace(-2, 2, 31)
XX, YY = np.meshgrid(qx_ne, qy_ne, indexing='ij')
S_ne = np.exp(-(XX ** 2 + YY ** 2) / 0.5)
q_ne = np.array([(x, y) for x in qx_ne for y in qy_ne])
sf_ne = {'q': q_ne, 'S_q': S_ne.reshape(-1).astype(complex),
         'definition': 'test', 'N': 8, 'is_restricted': False}
rep_ne = SpinStructureFactorPeakAnalyzer.find_peaks(sf_ne)
p_ne = rep_ne['peaks'][0]
check('I', '非等距网格 → refine_status=skipped_nonuniform 且回退原始峰位',
      p_ne['refine_status'] == 'skipped_nonuniform'
      and p_ne['q_refined'] == p_ne['q_original'],
      p_ne['refine_status'])
check('I', '非等距网格 → resolution.grid_uniform=False',
      rep_ne['resolution']['grid_uniform'] is False)
check('I', '等距网格 → resolution.grid_uniform=True（对照）',
      SpinStructureFactorPeakAnalyzer.find_peaks(
          make_sf(qx, qy, gauss2d(qx, qy, 2.0, 0.0, 0.0, 0.8, 0.8))
      )['resolution']['grid_uniform'] is True)

# ==========================================================================
# 10. FWHM 无法解析
# ==========================================================================
print('===== J. FWHM 边界 =====')
qx_s = np.linspace(-1.0, 1.0, 11)
qy_s = np.linspace(-1.0, 1.0, 11)
S2d_e = gauss2d(qx_s, qy_s, 2.0, -1.0, 0.0, 0.4, 0.4)   # 中心贴左边界 → 半高触边
rep = SpinStructureFactorPeakAnalyzer.find_peaks(make_sf(qx_s, qy_s, S2d_e))
p = rep['peaks'][0]
check('J', '半高触边 → fwhm_qx=None 且记录原因',
      p['fwhm_qx'] is None and '边界' in p['fwhm_note'],
      f"fwhm_qx={p['fwhm_qx']} note={p['fwhm_note']}")

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
