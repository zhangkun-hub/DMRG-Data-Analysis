# -*- coding: utf-8 -*-
"""
自旋/超导关联绘图布局测试（美观任务：n×3 泛化与新标题格式）。

覆盖（Agg 后端 smoke，合成数据）：
- A. 自旋数据图：单组 dict → 1×3；两组 dict → 2×3；
  行标题 'A sublattice (ref-site:49)' 格式；空 dict → ValueError
- B. 自旋误差图：旧格式 {组: list} → 1×3（总标题含子格与 ref-site）；
  嵌套格式 {组: {行: list}} → 2×3；行标签不一致 → ValueError
- C. 超导图：标题 '(reference A: (49, 50))'（无 'reference bond'）；
  difference 3 键方向 → 3×3，行标签 target A/B/C；键集合不一致 → ValueError
- D. helper：_sublattice_suffix（49→A、50→B、其他→None）

运行方式（在 DMRG data visualization/ 目录下）：
    python -X utf8 tests/test_corr_plot_layout.py
"""
import os
import sys
import types
import numpy as np
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')   # Agg 后端 plt.show() 的 UserWarning 属预期

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from Physical_Quantity.physical_analysis.Correlation_Process import (  # noqa: E402
    CorrelationProcessor, _sublattice_suffix)
from Physical_Quantity.physical_analysis.Super_Correlation_Process import (  # noqa: E402
    SuperCorrProcessor)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def spin_recs(src, seed=1.0):
    """合成自旋关联记录：源点 src，距离 1..8，值符号交替衰减。"""
    out = []
    for d in range(1, 9):
        out.append(types.SimpleNamespace(
            source_point=src, field_point=src + d,
            distance=float(d), correlation=seed * (-1.0) ** d / d**2))
    return out


def super_recs(ref=(49, 50), seed=1.0):
    """合成超导关联记录。"""
    out = []
    for d in range(1, 8):
        out.append(types.SimpleNamespace(
            ref_bond=ref, target_bond=(ref[0] + d, ref[1] + d),
            distance=float(d), sup_corr=seed * (-1.0) ** d / d**2))
    return out


def close():
    plt.close('all')


# ======================================================================
# A. 自旋数据图（n 组 → n×3）
# ======================================================================
print('===== A. 自旋数据图 =====')
CorrelationProcessor.Spin_Corr_Visualization({'A sublattice': spin_recs(49)})
fig = plt.gcf()
ylab = fig.axes[0].get_ylabel()
check('A', '单组 dict → 1 行 3 列', np.array(fig.axes).reshape(1, 3).shape == (1, 3))
check('A', "行标题 'A sublattice (ref-site:49)'",
      'A sublattice (ref-site:49)' in ylab, ylab.split('\n')[0])
close()

CorrelationProcessor.Spin_Corr_Visualization(
    {'A sublattice': spin_recs(49), 'B sublattice': spin_recs(50)})
fig = plt.gcf()
check('A', '两组 dict → 2 行 3 列', np.array(fig.axes).reshape(2, 3).shape == (2, 3))
ax2 = np.array(fig.axes).reshape(2, 3)
labels = [ax2[i, 0].get_ylabel().split('\n')[0] for i in range(2)]
check('A', '两行标题各含 ref-site',
      labels == ['A sublattice (ref-site:49)',
                 'B sublattice (ref-site:50)'], str(labels))
check('A', 'B 行 y-log 轴为对数刻度', ax2[1, 1].get_yscale() == 'log')
close()

try:
    CorrelationProcessor.Spin_Corr_Visualization({})
    check('A', '空 dict → ValueError', False, '未抛异常')
except ValueError:
    check('A', '空 dict → ValueError', True)

# ======================================================================
# B. 自旋误差图（旧格式 + 嵌套格式）
# ======================================================================
print('===== B. 自旋误差图 =====')
CorrelationProcessor.Spin_Corr_Visualization_difference(
    {'UltraDMRG': spin_recs(49), 'Itensor': spin_recs(49, 1.01)})
fig = plt.gcf()
check('B', '旧格式 → 1 行 3 列', np.array(fig.axes).reshape(1, 3).shape == (1, 3))
check('B', "旧格式总标题含 'A sublattice (ref-site:49)'",
      'A sublattice (ref-site:49)' in fig._suptitle.get_text())
close()

CorrelationProcessor.Spin_Corr_Visualization_difference(
    {'UltraDMRG': {'A sublattice': spin_recs(49),
                   'B sublattice': spin_recs(50)},
     'Itensor': {'A sublattice': spin_recs(49, 1.01),
                 'B sublattice': spin_recs(50, 1.01)}})
fig = plt.gcf()
check('B', '嵌套格式 → 2 行 3 列', np.array(fig.axes).reshape(2, 3).shape == (2, 3))
axb = np.array(fig.axes).reshape(2, 3)
l0 = axb[0, 0].get_ylabel()
check('B', "嵌套格式行标题含 'A sublattice (ref-site:49)'",
      'A sublattice (ref-site:49)' in l0)
check('B', '嵌套格式列标题为两组名 + 误差',
      axb[0, 0].get_title() == 'UltraDMRG'
      and axb[0, 1].get_title() == 'Itensor'
      and 'Absolute Error' in axb[0, 2].get_title())
close()

try:
    CorrelationProcessor.Spin_Corr_Visualization_difference(
        {'UltraDMRG': {'A': spin_recs(49), 'B': spin_recs(50)},
         'Itensor': {'A': spin_recs(49, 1.01)}})
    check('B', '行标签集合不一致 → ValueError', False, '未抛异常')
except ValueError:
    check('B', '行标签集合不一致 → ValueError', True)

# ======================================================================
# C. 超导图标题与 difference 3×3
# ======================================================================
print('===== C. 超导图 =====')
SuperCorrProcessor.Super_Corr_Visualization(super_recs(), 'A key (blue bond)')
fig = plt.gcf()
t = fig._suptitle.get_text()
check('C', "单键标题 '(reference A: (49, 50))' 且无 'reference bond'",
      '(reference A: (49, 50))' in t and 'reference bond' not in t, t)
close()

d3 = {'A key (blue bond)': super_recs(), 'B key (green bond)': super_recs(),
      'C key (orange bond)': super_recs()}
SuperCorrProcessor.Super_Corr_Visualization_three(d3)
fig = plt.gcf()
check('C', "three 标题 '(reference A: (49, 50))'",
      '(reference A: (49, 50))' in fig._suptitle.get_text())
close()

SuperCorrProcessor.Super_Corr_Visualization_difference(
    {'UltraDMRG': d3, 'Itensor': d3})
fig = plt.gcf()
check('C', 'difference 3 键方向 → 3 行 3 列', np.array(fig.axes).reshape(3, 3).shape == (3, 3))
axc = np.array(fig.axes).reshape(3, 3)
row_tags = [axc[i, 0].get_ylabel().split('\n')[0] for i in range(3)]
check('C', '行标签 target A/B/C',
      row_tags == ['target A', 'target B', 'target C'], str(row_tags))
check('C', '列标题为两组名 + 误差',
      axc[0, 0].get_title() == 'UltraDMRG'
      and axc[0, 1].get_title() == 'Itensor'
      and 'Absolute Error' in axc[0, 2].get_title())
close()

SuperCorrProcessor.Super_Corr_Visualization_difference(
    {'UltraDMRG': {'A key (blue bond)': super_recs()},
     'Itensor': {'A key (blue bond)': super_recs()}})
fig = plt.gcf()
check('C', 'difference 单键方向 → 1 行 3 列', np.array(fig.axes).reshape(1, 3).shape == (1, 3))
close()

try:
    SuperCorrProcessor.Super_Corr_Visualization_difference(
        {'UltraDMRG': d3,
         'Itensor': {'A key (blue bond)': super_recs()}})
    check('C', '键方向集合不一致 → ValueError', False, '未抛异常')
except ValueError:
    check('C', '键方向集合不一致 → ValueError', True)

# ======================================================================
# D. helper
# ======================================================================
print('===== D. 子格 helper =====')
check('D', '_sublattice_suffix(49) → A', _sublattice_suffix(49) == 'A')
check('D', '_sublattice_suffix(50) → B', _sublattice_suffix(50) == 'B')
check('D', '_sublattice_suffix(51) → None', _sublattice_suffix(51) is None)

from Physical_Quantity.physical_analysis.Super_Correlation_Process import (  # noqa: E402
    _refbond_direction)
check('D', '_refbond_direction(49, 50) → A', _refbond_direction(49, 50) == 'A')
check('D', '_refbond_direction(50, 51) → B', _refbond_direction(50, 51) == 'B')
check('D', '_refbond_direction(50, 54) → C', _refbond_direction(50, 54) == 'C')
check('D', '_refbond_direction(50, 49) → A（与顺序无关）',
      _refbond_direction(50, 49) == 'A')

# ======================================================================
# E. 超导 difference_refbond（行 = refbond 键方向）
# ======================================================================
print('===== E. 超导 difference_refbond =====')
refs = {
    'reference A: (49, 50)': {'A key (blue bond)': super_recs(),
                              'B key (green bond)': super_recs(),
                              'C key (orange bond)': super_recs()},
    'reference B: (50, 51)': {'A key (blue bond)': super_recs((50, 51)),
                              'B key (green bond)': super_recs((50, 51)),
                              'C key (orange bond)': super_recs((50, 51))},
    'reference C: (50, 54)': {'A key (blue bond)': super_recs((50, 54)),
                              'B key (green bond)': super_recs((50, 54)),
                              'C key (orange bond)': super_recs((50, 54))},
}
SuperCorrProcessor.Super_Corr_Visualization_difference_refbond(
    {'D = 1500': refs, 'D = 5000': refs})
fig = plt.gcf()
axe = np.array(fig.axes).reshape(3, 3)
check('E', '3 refbond 行 × 3 列', axe.shape == (3, 3))
check('E', "列标题 'D = 1500' / 'D = 5000' / 误差",
      axe[0, 0].get_title() == 'D = 1500'
      and axe[0, 1].get_title() == 'D = 5000'
      and 'Absolute Error' in axe[0, 2].get_title())
check('E', "行标签 'reference A: (49, 50)' 等",
      axe[0, 0].get_ylabel().split('\n')[0] == 'reference A: (49, 50)'
      and axe[1, 0].get_ylabel().split('\n')[0] == 'reference B: (50, 51)'
      and axe[2, 0].get_ylabel().split('\n')[0] == 'reference C: (50, 54)')
leg = [t.get_text() for t in axe[0, 0].get_legend().texts]
check('E', '第一列图例 target A/B/C',
      leg == ['target A', 'target B', 'target C'], str(leg))
check('E', '每格内 3 键折线（get_lines 含 1 条 axhline 零线，共 4）',
      len(axe[0, 0].get_lines()) == 4 and len(axe[0, 2].get_lines()) == 4)
close()

try:
    SuperCorrProcessor.Super_Corr_Visualization_difference_refbond(
        {'D = 1500': refs,
         'D = 5000': {'reference A: (49, 50)': refs['reference A: (49, 50)']}})
    check('E', 'refbond 集合不一致 → ValueError', False, '未抛异常')
except ValueError:
    check('E', 'refbond 集合不一致 → ValueError', True)

# ======================================================================
# F. 连线顺序（按 distance 排序，首尾不相连）
# ======================================================================
print('===== F. 连线顺序 =====')
shuffled = spin_recs(49)[::-1]   # 倒序输入
CorrelationProcessor.Spin_Corr_Visualization_difference(
    {'UltraDMRG': {'A sublattice': shuffled},
     'Itensor': {'A sublattice': spin_recs(49, 1.01)}})
fig = plt.gcf()
line = fig.axes[0].get_lines()[0]
xd = line.get_xdata()
check('F', '倒序输入连线按 distance 升序（首尾不相连）',
      np.all(np.diff(xd) > 0), str(list(xd)))
check('F', '连线不闭合（首尾 x 不重叠）',
      len(xd) == len(np.unique(xd)))
close()

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
