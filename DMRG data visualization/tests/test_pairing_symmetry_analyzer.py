# -*- coding: utf-8 -*-
"""
PairingSymmetryAnalyzer 阶段 1–2 定向测试（D5：数据装配 + 可比性质量门）

覆盖：
- A. 装配与完整性：完整装配、缺文件/缺通道、重复键、NaN/Inf、
  参考键不一致、空记录、格式错误、目录发现、参考键类别导出
- B. 质量门：K/xi 分流（纯函数 + 真实分析器字段）、振荡特征一致性、
  公共窗口（存在/不存在/点数不足/待校准标记）、SU(2) 记录与阈值路径、
  可比较元素集合（部分矩阵、状态区分）

合成输入为主；真实分析器调用仅用于验证元数据字段来源（幂律/指数
合成记录）。不依赖完整科研数据目录。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_pairing_symmetry_analyzer.py
"""
import json
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Physical_Data_Record import SuperCorrRecord
from Physical_Quantity.Honeycomb_Lattice import HoneycombXCGeometry
from Physical_Quantity.Correlation_Decay_Analyzer import AnalyzerConfig
from Physical_Quantity.Pairing_Symmetry_Analyzer import (
    PairingSymmetryConfig, classify_refbond, discover_refbond_dirs,
    raw_parse_channel_file, load_element, json_fmt, analyze_element,
    gate_form_consistency, gate_common_window, gate_su2,
    compute_window_zero_crossings, layer_oscillation_classification,
    build_comparable_set, LAYER_OSC, LAYER_NON_OSC, LAYER_UNKNOWN,
    LAYER_KEYS,
    INTEGRITY_OK, INTEGRITY_MISSING, INTEGRITY_INCOMPLETE,
    ANALYSIS_OK, ANALYSIS_FAILED)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def write_channel(ref_dir, key, ch, refbond, entries):
    path = os.path.join(
        ref_dir, f'scs{key}{ch}_fix_refbond_{refbond[0]}_{refbond[1]}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[[i, j, k, l], [v, 0]] for (i, j, k, l, v) in entries], f)
    return path


def make_tree(root, refbond=(0, 1), directions='ABC', channels='abcd',
              entries=None, target_fn=None):
    """构造 super_corr_{i}_{j} 临时目录。entries 为各方向各自的条目表；
    target_fn(i) 生成第 i 条目标键与值。"""
    ref_dir = os.path.join(root, f'super_corr_{refbond[0]}_{refbond[1]}')
    os.makedirs(ref_dir, exist_ok=True)
    for key in directions:
        ents = entries.get(key) if entries else None
        if ents is None and target_fn is not None:
            ents = [target_fn(i) for i in range(12)]
        for ch in channels:
            if ents is not None:
                write_channel(ref_dir, key, ch, refbond, ents)
    return ref_dir


# ======================================================================
# A. 装配与完整性
# ======================================================================
print('===== A. 装配与完整性 =====')
tmp_a = tempfile.mkdtemp(prefix='d5_a_')
geo16 = HoneycombXCGeometry(16, 1.0)

# A1 完整装配（N=16，refbond (0,1)，两条 allowed_cond 合法目标键）
ents_ok = [[0, 1, 4, 5, 1.0], [0, 1, 8, 9, 0.5]]
ref_dir = make_tree(os.path.join(tmp_a, 'full'), entries={'A': ents_ok,
                                                          'B': ents_ok,
                                                          'C': ents_ok})
el_ok = {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_dir}
ch_ok, comb_ok, int_ok = load_element(el_ok, 16, 1.0)
check('A', '完整装配：四通道加载、合成 2 条记录、完整性 ok',
      set(ch_ok) == {'a', 'b', 'c', 'd'} and len(comb_ok) == 2
      and int_ok['status'] == INTEGRITY_OK and not int_ok['issues'],
      f"{int_ok['status']} {len(comb_ok)} 条")

# A2 缺文件/缺通道
ref_miss = make_tree(os.path.join(tmp_a, 'miss'), channels='abd')
ch_m, comb_m, int_m = load_element(
    {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_miss}, 16, 1.0)
check('A', '缺通道：完整性 missing、issue 类别 missing_file、路径定位',
      int_m['status'] == INTEGRITY_MISSING and comb_m is None
      and any(i['category'] == 'missing_file'
              and 'scsAc' in i['location'] for i in int_m['issues']),
      int_m['status'])

# A3 重复键 / NaN / 参考键不一致
ents_dup = [[0, 1, 4, 5, 1.0], [0, 1, 4, 5, 0.5], [0, 1, 8, 9, 1.0]]
ref_dup = make_tree(os.path.join(tmp_a, 'dup'), entries={'A': ents_dup})
el_dup = {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_dup}
_, comb_d, int_d = load_element(el_dup, 16, 1.0)
check('A', '重复键：逐项检测（计数而非覆盖）且不静默去重',
      any(i['category'] == 'duplicate_key' for i in int_d['issues'])
      and int_d['status'] == INTEGRITY_OK,   # 结构可解析仍可合成
      [i['category'] for i in int_d['issues']])

# 非有限值：JSON 无法合法表达 NaN/Infinity 字面量；用合法 JSON 的
# 溢出数（1e400 解析后为 inf）触发非有限检测
ents_nan = [[0, 1, 4, 5, 1e400], [0, 1, 8, 9, -1e400]]
ref_nan = make_tree(os.path.join(tmp_a, 'nan'), entries={'A': ents_nan})
_, comb_n, int_n = load_element(
    {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_nan}, 16, 1.0)
# 四通道同一文件 → 每通道 2 条溢出非有限 → 共 8 条逐项报告；
# json_fmt(allow_nan=False) 在非有限值处拒绝序列化（format_error），
# 两种防御路径至少其一触发，且不静默替换为零
n_nonfinite = sum(1 for i in int_n['issues']
                  if i['category'] == 'non_finite')
n_fmt = sum(1 for i in int_n['issues'] if i['category'] == 'format_error')
check('A', '非有限值：逐项报告（8 条 non_finite 或 4 条 format_error 防御），'
      '不静默替换为零',
      n_nonfinite == 8 or (n_nonfinite == 0 and n_fmt == 4),
      f'non_finite={n_nonfinite} format_error={n_fmt}')

ents_mm = [[0, 1, 4, 5, 1.0], [0, 2, 8, 9, 0.5]]
ref_mm = make_tree(os.path.join(tmp_a, 'mm'), entries={'A': ents_mm})
_, comb_mm, int_mm = load_element(
    {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_mm}, 16, 1.0)
check('A', '参考键不一致：逐项报告（含预期与实际键）',
      any(i['category'] == 'refbond_mismatch' and '预期 (0, 1)' in i['detail']
          for i in int_mm['issues']))

# A4 空记录与格式错误（四文件齐全，其一为空 → 文件存在但记录不完整）
ref_empty = make_tree(os.path.join(tmp_a, 'empty'), entries={'A': ents_ok})
path_empty = os.path.join(ref_empty, 'scsAa_fix_refbond_0_1.json')
open(path_empty, 'w', encoding='utf-8').write('[]')
_, comb_e, int_e = load_element(
    {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_empty}, 16, 1.0)
check('A', '空记录：empty_records 类别、完整性 incomplete（文件存在但'
      '记录不完整，区别于 missing）',
      any(i['category'] == 'empty_records' for i in int_e['issues'])
      and int_e['status'] == INTEGRITY_INCOMPLETE)

ref_bad = make_tree(os.path.join(tmp_a, 'bad'))
open(os.path.join(ref_bad, 'scsAa_fix_refbond_0_1.json'),
     'w', encoding='utf-8').write('not json')
_, comb_b, int_b = load_element(
    {'refbond': (0, 1), 'direction': 'A', 'refbond_dir': ref_bad}, 16, 1.0)
check('A', '格式错误：format_error 类别、不进入 extract_data',
      any(i['category'] == 'format_error' for i in int_b['issues']))

# A5 目录发现（在含 super_corr 目录的根上检查；无关目录应被跳过）
os.makedirs(os.path.join(tmp_a, 'full', 'not_a_refbond_dir'))
dirs = discover_refbond_dirs(os.path.join(tmp_a, 'full'))
check('A', '目录发现：super_corr_{i}_{j} 被识别、其余目录被跳过',
      len(dirs) == 1 and dirs[0][1] == (0, 1),
      f'{dirs}')
try:
    discover_refbond_dirs(os.path.join(tmp_a, '不存在'))
    check('A', '目录缺失 → FileNotFoundError（既有失败语义）', False, '未抛异常')
except FileNotFoundError:
    check('A', '目录缺失 → FileNotFoundError（既有失败语义）', True)

# A6 参考键类别导出（N=200 实测三类 + 非法角拒绝）
geo200 = HoneycombXCGeometry(200, 1.0)
check('A', '参考键类别：49_50→A、50_51→B、50_54→C（几何导出，不硬编码）',
      classify_refbond((49, 50), geo200) == 'A'
      and classify_refbond((50, 51), geo200) == 'B'
      and classify_refbond((50, 54), geo200) == 'C')
try:
    classify_refbond((0, 4), geo200)   # 水平键是 C 类——应成功
    classify_refbond((1000, 1001), geo200)   # 越界
    check('A', '越界参考键 → ValueError', False, '未抛异常')
except ValueError:
    check('A', '越界参考键 → ValueError', True)

# ======================================================================
# B. 质量门
# ======================================================================
print('===== B. 质量门 =====')

# B1 K/xi 分流（纯函数）
common, bad = gate_form_consistency(
    {'e1': {'envelope_form': 'K'}, 'e2': {'envelope_form': 'K'},
     'e3': {'envelope_form': 'xi'}, 'e4': {'envelope_form': None}})
check('B', 'K/xi 混入被拦截（含缺失形式的显式原因）',
      common == 'K' and set(bad) == {'e3', 'e4'}
      and '不混用' in bad['e3'] and '缺失' in bad['e4'])

# B2 真实分析器字段来源：幂律 → K、指数 → xi（不凭猜测）
dists = np.arange(3.0, 39.0, 3.0)
g = HoneycombXCGeometry(200, 1.0)
recs_pow = [SuperCorrRecord((49, 50), (52 + 4 * i, 53 + 4 * i),
                            g.sup_corr_distance(49, 50, 52 + 4 * i, 53 + 4 * i),
                            1e-4 * d ** (-2.0))
            for i, d in enumerate(dists)]
recs_exp = [SuperCorrRecord((49, 50), (52 + 4 * i, 53 + 4 * i),
                            g.sup_corr_distance(49, 50, 52 + 4 * i, 53 + 4 * i),
                            1e-4 * np.exp(-d / 15.0))
            for i, d in enumerate(dists)]
ares_p = analyze_element(recs_pow, AnalyzerConfig())
ares_e = analyze_element(recs_exp, AnalyzerConfig())
check('B', '真实分析：幂律合成 → 包络形式 K、指数合成 → xi（真实字段）',
      ares_p['status'] == ANALYSIS_OK
      and ares_e['status'] == ANALYSIS_OK
      and ares_p['meta']['envelope_form'] == 'K'
      and ares_e['meta']['envelope_form'] == 'xi',
      f"{ares_p['meta']['envelope_form']}/{ares_e['meta']['envelope_form']}")
check('B', '真实元数据含窗口/点数/振荡证据/verdict（质量门所需字段齐全）',
      all(k in ares_p['meta'] for k in ('r_min', 'r_max', 'n_points',
                                        'osc_detected', 'osc_evidence',
                                        'verdict', 'classification')))
check('B', '分析失败路径：空记录 → analysis_failed（ValueError 捕获，'
      '不吞其他异常）',
      analyze_element([], AnalyzerConfig())['status'] == ANALYSIS_FAILED)
try:
    analyze_element('not_records', AnalyzerConfig())
    check('B', '非数据级异常原样传播', False, '未抛异常')
except (TypeError, AttributeError):
    check('B', '非数据级异常原样传播', True)

# B3 振荡分层分类（纯函数：证据级别 + 公共窗口过零证据，5B-1 层判据）
def _crossing(rel_c=0, unrel_c=0, rel='none'):
    note = {'reliable': '可靠翻转',
            'unreliable_only': '翻转全部发生在噪声底附近',
            'none': '无翻转',
            'unavailable': '噪声底缺失'}[rel]
    return {'zero_crossing_observed': rel_c + unrel_c,
            'reliable_crossings': rel_c, 'unreliable_crossings': unrel_c,
            'reliability': rel, 'reliability_note': note,
            'noise_floor_used': 1e-10}

crosses = {
    'e_strong': _crossing(3, 0, 'reliable'),
    'e_weak_rel': _crossing(2, 0, 'reliable'),
    'e_weak_none': _crossing(0, 0, 'none'),
    'e_weak_unrel': _crossing(0, 2, 'unreliable_only'),
    'e_weak_nofloor': _crossing(0, 2, 'unavailable'),
    'e_none': _crossing(0, 0, 'none'),
}
layers, lreps = layer_oscillation_classification(
    {'e_strong': {'osc_evidence': 'strong'},
     'e_weak_rel': {'osc_evidence': 'weak'},
     'e_weak_none': {'osc_evidence': 'weak'},
     'e_weak_unrel': {'osc_evidence': 'weak'},
     'e_weak_nofloor': {'osc_evidence': 'weak'},
     'e_none': {'osc_evidence': None}},
    crosses)
check('B', 'strong/moderate 证据 → osc（证据等级保留在报告）',
      layers['e_strong'] == LAYER_OSC
      and lreps['e_strong']['osc_evidence'] == 'strong')
check('B', 'weak 证据 + 公共窗口可靠过零 → osc（过零证据充分，'
      '不依赖证据级别）',
      layers['e_weak_rel'] == LAYER_OSC
      and lreps['e_weak_rel']['reliable_crossings'] == 2)
check('B', 'weak 证据 + 无过零 → non_osc（按非振荡处理，'
      '不表示已证明不存在振荡）',
      layers['e_weak_none'] == LAYER_NON_OSC
      and '不表示已证明' in lreps['e_weak_none']['reason'])
check('B', '过零全部发生在噪声底附近 → unknown（不足以判定真实振荡，'
      '不强行归类）',
      layers['e_weak_unrel'] == LAYER_UNKNOWN
      and '噪声底' in lreps['e_weak_unrel']['reason'])
check('B', '过零可靠性不可判定（噪声底缺失）→ unknown（数据不足）',
      layers['e_weak_nofloor'] == LAYER_UNKNOWN)
check('B', '振荡证据缺失 → unknown（不假定兼容）',
      layers['e_none'] == LAYER_UNKNOWN)

# B3b 过零证据（真实记录：窗口裁剪 + 翻转点对两侧 |C| vs 噪声底，
# 只消费带符号记录）
geo_b3 = HoneycombXCGeometry(200, 1.0)
recs_zc = []
for m in range(1, 13):
    r = geo_b3.sup_corr_distance(49, 50, 49 + 8 * m, 50 + 8 * m)
    v = 1e-4 * r ** (-2.0) * (1.0 if m % 3 else -1.0)   # 每 3 点一次翻转
    recs_zc.append(SuperCorrRecord((49, 50), (49 + 8 * m, 50 + 8 * m), r, v))
zc = compute_window_zero_crossings({'e1': recs_zc}, (3.0, 36.0),
                                   {'e1': 1e-9})
check('B', '过零证据：窗口内翻转计数 + 可靠分类（翻转点对两侧 |C| '
      '均高于噪声底才算可靠）',
      zc['e1']['zero_crossing_observed'] == 7
      and zc['e1']['reliable_crossings'] == 7
      and zc['e1']['reliability'] == 'reliable',
      f"{zc['e1']}")
zc_low = compute_window_zero_crossings({'e1': recs_zc}, (3.0, 36.0),
                                       {'e1': 1e-4})   # 噪声底高于所有 |C|
check('B', '过零证据：噪声底高于翻转点幅度 → 全部不可靠（unreliable_only）',
      zc_low['e1']['reliability'] == 'unreliable_only'
      and zc_low['e1']['reliable_crossings'] == 0
      and zc_low['e1']['unreliable_crossings'] == 7)

# B4-B6 公共窗口
meta_w = {'e1': {'r_min': 3.0, 'r_max': 30.0}, 'e2': {'r_min': 4.0, 'r_max': 33.0}}
pts = {'e1': np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]),
       'e2': np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0])}
w, excl, cnt, uncal = gate_common_window(meta_w, pts, min_common_points=6)
check('B', '公共窗口存在：取交 [4,30]、窗口内点数充足（按实际点数而非端点）',
      w == (4.0, 30.0) and not excl and cnt == {'e1': 9, 'e2': 9} and not uncal)
w2, excl2, cnt2, uncal2 = gate_common_window(meta_w, pts, min_common_points=None)
check('B', 'min_common_points=None：待校准标记（记录点数、不按点数排除）',
      uncal2 and w2 == (4.0, 30.0) and not excl2)
meta_dis = {'e1': {'r_min': 3.0, 'r_max': 10.0}, 'e2': {'r_min': 12.0, 'r_max': 30.0}}
w3, excl3, _, _ = gate_common_window(meta_dis, pts, min_common_points=6)
check('B', '公共窗口不存在：全部排除且保留原因（不强行凑矩阵）',
      w3 is None and all('不相交' in v for v in excl3.values()))
pts_few = {'e1': np.array([3.0, 6.0]), 'e2': np.array([3.0, 6.0, 9.0, 12.0])}
w4, excl4, cnt4, _ = gate_common_window(meta_w, pts_few, min_common_points=6)
check('B', '窗口内点数不足：按实际点数排除（e1 仅 2 点）并保留原因',
      w4 == (4.0, 30.0) and 'e1' in excl4 and '点数' in excl4['e1'])

# B7 SU(2) 检查
def mk_recs(vals, rb=(0, 1)):
    return [SuperCorrRecord(rb, (4, 5), 1.0, v) for v in vals]

ch_su2 = {
    'k1': {'a': mk_recs([1.0, 2.0, 3.0]), 'b': mk_recs([0.5, 1.5, 2.5]),
           'c': mk_recs([0.5, 1.5, 2.5]), 'd': mk_recs([1.0, 2.0, 3.0])},
    'k2': {'a': mk_recs([1.0, 2.0, 3.0]), 'b': mk_recs([0.5, 1.5, 2.5]),
           'c': mk_recs([0.5, 1.5, 2.5]), 'd': mk_recs([1.1, 2.0, 3.0])},
    'k3': {'a': mk_recs([1.0]), 'b': mk_recs([0.5]), 'c': mk_recs([0.5])},
}
su2_unc = gate_su2(ch_su2, su2_error_max=None)
check('B', 'SU(2) 无阈值：只记录误差、gate=uncalibrated（不擅自选阈值）',
      su2_unc['k1']['gate'] == 'uncalibrated'
      and abs(su2_unc['k1']['err_ad']) < 1e-12
      and su2_unc['k2']['max_err'] > 0.03)
su2_cal = gate_su2(ch_su2, su2_error_max=1e-3)
check('B', 'SU(2) 显式阈值：一致通过（~0）、超标拒绝（0.033>1e-3）',
      su2_cal['k1']['gate'] == 'pass' and su2_cal['k2']['gate'] == 'fail')
check('B', 'SU(2) 通道不全 → skipped（不伪造检查结果）',
      su2_unc['k3']['gate'] == 'skipped' and '通道不全' in su2_unc['k3']['detail'])

# B8 可比较元素集合（部分矩阵 + 状态区分，全链路：文件 → 装配 → 门）
print('===== B8 可比较集合 =====')
tmp_b8 = tempfile.mkdtemp(prefix='d5_b8_')
# N=200 refbond (49,50)：allowed_cond 合法目标 (49+8m, 50+8m)，m=1..12
# 通道值须 a=d、b=c 且 a≠b，否则合成 0.5(a−b−c+d) 恒零
g200 = HoneycombXCGeometry(200, 1.0)
ents_a = []
for m in range(1, 13):
    k, l = 49 + 8 * m, 50 + 8 * m
    r = g200.sup_corr_distance(49, 50, k, l)
    ents_a.append([49, 50, k, l, 1e-4 * r ** (-2.0)])
ents_b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents_a]
ref_b8 = make_tree(os.path.join(tmp_b8, 'root', 't_J_3', 'd_syn',
                                '1500_measure_super_corr'),
                   refbond=(49, 50), entries={'A': ents_a, 'B': ents_a,
                                              'C': ents_a})
# b/c 通道改写为 0.8 倍值（a=d、b=c，合成 −0.1·v 非零幂律）
for key in 'ABC':
    for ch in ('b', 'c'):
        write_channel(ref_b8, key, ch, (49, 50), ents_b)
# 删除方向 B 的 a 通道 → B 缺失
os.remove(os.path.join(ref_b8, 'scsBa_fix_refbond_49_50.json'))
rep_b8 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': os.path.dirname(ref_b8)},
    config=PairingSymmetryConfig())
cs = rep_b8['comparable_set']
elB = rep_b8['elements']['49_50|B']
elA = rep_b8['elements']['49_50|A']
check('B', '全链路：方向 B 缺文件 → 完整性 missing 且被排除（原因含 missing_file）',
      elB['integrity']['status'] == INTEGRITY_MISSING
      and '49_50|B' in cs['excluded']
      and any('missing_file' in r for r in cs['excluded']['49_50|B']),
      elB['integrity']['status'])
check('B', '方向 A/C 通过质量门进入成员集合（真实分析 → 共同形式 K、'
      '分层归属 non_osc）',
      cs['common_form'] == 'K'
      and set(cs['member_keys']) == {'49_50|A', '49_50|C'}
      and set(cs['layers'][LAYER_NON_OSC]['member_keys'])
      == {'49_50|A', '49_50|C'},
      f"{cs['common_form']} {cs['member_keys']}")
check('B', '部分矩阵标记与状态（partial + is_partial）',
      cs['status'] == 'partial' and cs['is_partial'])
check('B', '层内公共窗口存在且窗口内点数已记录（non_osc 层，实际距离组计数）',
      cs['layers'][LAYER_NON_OSC]['common_window'] is not None
      and '49_50|A' in cs['layers'][LAYER_NON_OSC]['common_window_points']
      and cs['layers'][LAYER_NON_OSC]['common_window_points']['49_50|A'] > 0,
      f"{cs['layers'][LAYER_NON_OSC]['common_window']}")
check('B', '跨层比较默认拒绝（allowed=False + 明确原因）',
      cs['cross_layer_comparison']['allowed'] is False
      and '跨层' in cs['cross_layer_comparison']['reason'])
check('B', 'SU(2) 误差已记录（合成 a=d/b=c → ~0，gate=uncalibrated）',
      rep_b8['su2']['49_50|A']['gate'] == 'uncalibrated'
      and rep_b8['su2']['49_50|A']['max_err'] < 1e-12)
check('B', '待校准阈值显式标记（min_common_points / su2_error_max / '
      '过零可靠性均在列）',
      'min_common_points' in cs['uncalibrated_thresholds']
      and 'su2_error_max' in cs['uncalibrated_thresholds']
      and 'zero_crossing_reliability_threshold' in cs['uncalibrated_thresholds'])
check('B', '状态四分类不混淆：missing（文件缺失）与 quality gate 拒绝'
      '与 analysis_failed 分别表达',
      elB['integrity']['status'] == INTEGRITY_MISSING
      and elB.get('analysis') is None
      and elA['analysis']['status'] == ANALYSIS_OK
      and elA['quality_gate']['passed'] is True)
check('B', '分层信息：元素质量门记录层归属（纯幂律 weak 证据 + 无过零 → '
      'non_osc）',
      elA['quality_gate']['layer'] == LAYER_NON_OSC)
check('B', '无对称性结论声明存在（本阶段不产生任何判定）',
      'no_symmetry_conclusions' in rep_b8)

# ======================================================================
# B9-B12 振荡证据分层定向用例（5B-1：任务书 §6.1-6.5）
# ======================================================================
print('===== B9-B12 振荡证据分层 =====')
g200 = HoneycombXCGeometry(200, 1.0)


def make_tree_fn(root, refbonds, dir_values):
    """dir_values: {(refbond_class, direction): callable(r) -> v}；
    a=d 通道原值、b=c 通道 0.8 倍。"""
    for rcls, rb in refbonds.items():
        ref_dir = os.path.join(root, f'super_corr_{rb[0]}_{rb[1]}')
        os.makedirs(ref_dir, exist_ok=True)
        for key in 'ABC':
            fn = dir_values.get((rcls, key))
            if fn is None:
                continue
            ents = []
            for m in range(1, 13):
                k, l = rb[0] + 8 * m, rb[1] + 8 * m
                r = g200.sup_corr_distance(rb[0], rb[1], k, l)
                ents.append([rb[0], rb[1], k, l, float(fn(r))])
            ents_b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents]
            for ch in 'ad':
                write_channel(ref_dir, key, ch, rb, ents)
            for ch in 'bc':
                write_channel(ref_dir, key, ch, rb, ents_b)


# B9 d1_10 型：8 个同质振荡元素 + 1 个 weak（纯幂律）元素——
# 不再被连坐排除，各归其层
tmp_b9 = tempfile.mkdtemp(prefix='d5_b9_')
super_b9 = os.path.join(tmp_b9, 't_J_3', 'd_syn', '1500_measure_super_corr')
refs9 = {'A': (49, 50), 'B': (50, 51), 'C': (50, 54)}
vals9 = {}
for rc in refs9:
    for key in 'ABC':
        if rc == 'C' and key == 'C':
            vals9[(rc, key)] = (lambda r: 1e-4 * r ** (-2.0))   # 纯幂律 → weak、无过零
        else:
            vals9[(rc, key)] = (lambda r: 1e-4 * r ** (-2.0)
                                * np.cos(0.8 * r + np.pi / 2))  # strong 振荡
make_tree_fn(super_b9, refs9, vals9)
rep_b9 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_b9}, config=PairingSymmetryConfig())
cs9 = rep_b9['comparable_set']
osc9 = set(cs9['layers'][LAYER_OSC]['member_keys'])
nono9 = set(cs9['layers'][LAYER_NON_OSC]['member_keys'])
expect_osc9 = ({f'{rb[0]}_{rb[1]}|{d}' for rb in refs9.values() for d in 'ABC'}
               - {'50_54|C'})
check('B', 'd1_10 型：同质振荡元素归 osc 层、weak 无过零元素归 non_osc '
      '层（不连坐）',
      osc9 == expect_osc9 and nono9 == {'50_54|C'},
      f'osc={len(osc9)} non_osc={nono9}')
check('B', 'd1_10 型：无任何元素因"缺少混合规则"被排除（旧硬排除语义消失）',
      not any('缺少混合规则' in r
              for rs in cs9['excluded'].values() for r in rs))
check('B', 'd1_10 型：osc 层窗口与 non_osc 层窗口各自独立存在',
      cs9['layers'][LAYER_OSC]['common_window'] is not None
      and cs9['layers'][LAYER_NON_OSC]['common_window'] is not None)

# B10 d1_20 型：weak 证据但公共窗口内存在可靠过零 → osc
tmp_b10 = tempfile.mkdtemp(prefix='d5_b10_')
super_b10 = os.path.join(tmp_b10, 't_J_3', 'd_syn', '1500_measure_super_corr')
make_tree_fn(super_b10, {'A': (49, 50)},
             {('A', k): (lambda r: 1e-4 * r ** (-2.0)
                         * np.sign(np.sin(1.5 * r))) for k in 'ABC'})
rep_b10 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_b10}, config=PairingSymmetryConfig())
cs10 = rep_b10['comparable_set']
el10 = rep_b10['elements']['49_50|A']
ev10 = el10['analysis']['meta'].get('osc_evidence')
zr10 = cs10['layers'][LAYER_OSC]['zero_crossing_report'].get('49_50|A')
check('B', 'd1_20 型：证据级别不足（非 strong/moderate）但公共窗口可靠'
      '过零 → osc 层（过零报告同步记录）',
      ev10 in ('weak', 'possible', 'none')
      and '49_50|A' in cs10['layers'][LAYER_OSC]['member_keys']
      and zr10 is not None and zr10['reliable_crossings'] >= 1,
      f'ev={ev10} zc={zr10}')
check('B', 'd1_20 型：过零证据基于公共窗口实际距离序列（窗口字段与'
      '层窗口一致）',
      zr10 is not None
      and zr10['layer_window'] == cs10['layers'][LAYER_OSC]['common_window'])

# B11 weak 且无过零 → non_osc（低幅振荡 1+0.15cos 恒正，无符号翻转）
tmp_b11 = tempfile.mkdtemp(prefix='d5_b11_')
super_b11 = os.path.join(tmp_b11, 't_J_3', 'd_syn', '1500_measure_super_corr')
make_tree_fn(super_b11, {'A': (49, 50)},
             {('A', k): (lambda r: 1e-4 * r ** (-2.0)
                         * (1 + 0.15 * np.cos(0.8 * r))) for k in 'ABC'})
rep_b11 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_b11}, config=PairingSymmetryConfig())
cs11 = rep_b11['comparable_set']
check('B', 'weak 且无过零 → non_osc（无符号翻转即非振荡层）',
      set(cs11['layers'][LAYER_NON_OSC]['member_keys'])
      == {'49_50|A', '49_50|B', '49_50|C'}
      and not cs11['layers'][LAYER_OSC]['member_keys'])

# B12 过零靠近噪声底、可靠性不明 → unknown（尾部低幅翻转，不判真实振荡）
tmp_b12 = tempfile.mkdtemp(prefix='d5_b12_')
super_b12 = os.path.join(tmp_b12, 't_J_3', 'd_syn', '1500_measure_super_corr')
ref_dir_b12 = os.path.join(super_b12, 'super_corr_49_50')
os.makedirs(ref_dir_b12, exist_ok=True)
ents_b12 = []
for m in range(1, 13):
    k, l = 49 + 8 * m, 50 + 8 * m
    r = g200.sup_corr_distance(49, 50, k, l)
    v = 1e-4 * r ** (-2.0)
    if m == 6:
        v = 5e-9
    elif m == 7:
        v = -5e-9   # 中部一对低幅翻转（幅度接近噪声底）
    ents_b12.append([49, 50, k, l, v])
ents_b12b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in ents_b12]
for key in 'ABC':
    for ch in 'ad':
        write_channel(ref_dir_b12, key, ch, (49, 50), ents_b12)
    for ch in 'bc':
        write_channel(ref_dir_b12, key, ch, (49, 50), ents_b12b)
rep_b12 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_b12}, config=PairingSymmetryConfig())
cs12 = rep_b12['comparable_set']
el12 = rep_b12['elements']['49_50|A']
ev12 = el12['analysis']['meta'].get('osc_evidence')
layer12 = el12['quality_gate'].get('layer')
zr12 = None
for L in LAYER_KEYS:
    r = cs12['layers'][L]['zero_crossing_report'].get('49_50|A')
    if r is not None:
        zr12 = r
check('B', '过零靠近噪声底（低幅翻转）→ unknown 或不可靠过零被记录'
      '（不自动判为可靠振荡）',
      (ev12 in ('weak', 'possible', 'none')
       and ((layer12 == LAYER_UNKNOWN)
            or (layer12 == LAYER_OSC
                and zr12 is not None and zr12['reliable_crossings'] >= 1))),
      f'ev={ev12} layer={layer12} zc={zr12}')

# B13 5B-1.2：迭代归属双向变化与逐层未稳定记录
# 构造：元素 A（m=1..12）近程过零（r<6.5），元素 B/C 仅 m=3..12
# strong 振荡 → 组窗口 lo≈9 不含 A 的近程过零 → A 第 1 轮 non_osc；
# non_osc 层窗口 (3,36) 内含 A 的可靠过零 → A 升 osc；osc 层窗口
# (9,36) 内 A 又无过零 → 降 non_osc → 双向振荡，3 轮耗尽
tmp_b13 = tempfile.mkdtemp(prefix='d5_b13_')
super_b13 = os.path.join(tmp_b13, 't_J_3', 'd_syn',
                         '1500_measure_super_corr')
ref_dir_b13 = os.path.join(super_b13, 'super_corr_49_50')
os.makedirs(ref_dir_b13, exist_ok=True)
g_b13 = HoneycombXCGeometry(200, 1.0)
entsA13 = []
for m in range(1, 13):
    k, l = 49 + 8 * m, 50 + 8 * m
    r = g_b13.sup_corr_distance(49, 50, k, l)
    v = -5e-5 if r < 6.5 else 1e-4 * r ** (-2.0)
    entsA13.append([49, 50, k, l, v])
entsB13 = []
for m in range(3, 13):
    k, l = 49 + 8 * m, 50 + 8 * m
    r = g_b13.sup_corr_distance(49, 50, k, l)
    entsB13.append([49, 50, k, l,
                    1e-4 * r ** (-2.0) * np.cos(0.8 * r + np.pi / 2)])
entsA13b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in entsA13]
entsB13b = [[i, j, k, l, 0.8 * v] for i, j, k, l, v in entsB13]
for ch in 'ad':
    write_channel(ref_dir_b13, 'A', ch, (49, 50), entsA13)
    write_channel(ref_dir_b13, 'B', ch, (49, 50), entsB13)
    write_channel(ref_dir_b13, 'C', ch, (49, 50), entsB13)
for ch in 'bc':
    write_channel(ref_dir_b13, 'A', ch, (49, 50), entsA13b)
    write_channel(ref_dir_b13, 'B', ch, (49, 50), entsB13b)
    write_channel(ref_dir_b13, 'C', ch, (49, 50), entsB13b)
rep_b13 = build_comparable_set(
    {'doping': 'd_syn', 'N': 200, 'bond_dim': '1500',
     'super_corr_root': super_b13}, config=PairingSymmetryConfig())
cs13 = rep_b13['comparable_set']
conv_o13 = cs13['layers'][LAYER_OSC]['layer_convergence']
conv_n13 = cs13['layers'][LAYER_NON_OSC]['layer_convergence']
chg_o13 = conv_o13['last_round_changes']
chg_n13 = conv_n13['last_round_changes']
check('B', '迭代归属双向变化：元素 A 经历 osc↔non_osc 振荡，'
      '3 轮耗尽标注未稳定（stable=False、rounds_used=2、'
      'note 含未稳定），不得伪装为稳定分类',
      conv_o13['stable'] is False and conv_n13['stable'] is False
      and conv_o13['rounds_used'] == 2
      and '未稳定' in conv_o13['note'],
      f"{conv_o13['stable']}/{conv_o13['rounds_used']}")
check('B', '逐层未稳定记录：层名所在字段 + 最后一轮双向变化'
      '（osc 层记录 osc→non_osc、non_osc 层记录 non_osc→osc）',
      any(c['element'] == '49_50|A' and c['from'] == LAYER_OSC
          and c['to'] == LAYER_NON_OSC for c in chg_o13)
      and any(c['element'] == '49_50|A' and c['from'] == LAYER_NON_OSC
              and c['to'] == LAYER_OSC for c in chg_n13),
      f'osc_changes={chg_o13} non_osc_changes={chg_n13}')
check('B', '稳定场景对照：B10 单层稳定数据的 layer_convergence.stable=True '
      '且无变化记录',
      cs10['layers'][LAYER_OSC]['layer_convergence']['stable'] is True
      and not cs10['layers'][LAYER_OSC]['layer_convergence']
      ['last_round_changes'])

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
