# -*- coding: utf-8 -*-
"""
PairingSymmetryAnalyzer 阶段 5B-3 定向测试：跨元素相对符号证据
（r_αβ = s_α·s_β，纯函数 relative_sign_pair_reports）

覆盖（任务书 5B-3 §7）：
1. 同层同号 → product=+1
2. 同层异号 → product=−1
3. 任一 insufficient → unavailable + product=None
4. 任一 unavailable → unavailable + product=None
5. 符号缺失/非法/非有限 → 不得产生有效乘积
6. 跨层配对被拒绝 + 原因
7. unknown 层与缺失层保守处理
8. (i,j)/(j,i) 不重复、无自配对
9. 空层/缺 per_layer/malformed 不崩溃不伪造
10. provenance 与实际输入一致
11. 只消费原始符号状态，不混入振幅符号
12. 输出无候选结论字段（5B-2 supported_candidate 语义保持）

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_pairing_symmetry_relative_sign.py
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

from Physical_Quantity.Pairing_Symmetry_Analyzer import (
    relative_sign_pair_reports, RELPAIR_AVAILABLE, RELPAIR_UNAVAILABLE,
    SIGN_EXTRACTED, SIGN_INSUFFICIENT, SIGN_UNAVAILABLE,
    LAYER_OSC, LAYER_NON_OSC, LAYER_UNKNOWN)

RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ''))


def mk_sign(entries):
    """entries: {key: dict(state, sign, reason?, n_pts?, floor?,
    extra?)} → analyze_sign_channel 形状。"""
    pe = {}
    rel = []
    for k, v in entries.items():
        item = {'state': v.get('state'),
                'sign': v.get('sign'),
                'reason': v.get('reason', ''),
                'n_points_above_floor': v.get('n_pts', 4),
                'noise_floor_used': v.get('floor', 1e-6),
                'crossings': v.get('crossings', 0)}
        item.update(v.get('extra', {}))
        pe[k] = item
        if v.get('state') == SIGN_EXTRACTED:
            rel.append(k)
    return {'state': 'extracted', 'per_element': pe, 'per_layer': {},
            'relative_signs_available': rel}


def mk_report(layers):
    return {'elements': {k: {'quality_gate': {'layer': L, 'passed': True,
                                             'reasons': []}}
                         for k, L in layers.items()}}


def ext(sign):
    return {'state': SIGN_EXTRACTED, 'sign': sign}


print('===== 相对符号证据 =====')

# 1/2 同层同号/异号
s12 = mk_sign({'k1': ext(1), 'k2': ext(1)})
r12 = relative_sign_pair_reports(s12, mk_report({'k1': LAYER_NON_OSC,
                                                 'k2': LAYER_NON_OSC}))
p12 = r12['pairs'][0]
check('T1', '同层同号 → available + product=+1 + 层标注',
      p12['state'] == RELPAIR_AVAILABLE and p12['product'] == 1
      and p12['layer'] == LAYER_NON_OSC
      and r12['n_available'] == 1,
      f"{p12['product']} {p12['layer']}")
s12b = mk_sign({'k1': ext(1), 'k2': ext(-1)})
r12b = relative_sign_pair_reports(s12b, mk_report(
    {'k1': LAYER_NON_OSC, 'k2': LAYER_NON_OSC}))
check('T2', '同层异号 → available + product=−1',
      r12b['pairs'][0]['state'] == RELPAIR_AVAILABLE
      and r12b['pairs'][0]['product'] == -1)

# 3/4 任一 insufficient / unavailable
for bad_state, tag in ((SIGN_INSUFFICIENT, 'T3'),
                       (SIGN_UNAVAILABLE, 'T4')):
    s_bad = mk_sign({'k1': ext(1),
                     'k2': {'state': bad_state, 'sign': None,
                            'reason': f'{bad_state} 原因'}})
    r_bad = relative_sign_pair_reports(s_bad, mk_report(
        {'k1': LAYER_NON_OSC, 'k2': LAYER_NON_OSC}))
    p_bad = r_bad['pairs'][0]
    check(tag, f'任一 {bad_state} → unavailable + product=None + 原因'
          '含状态',
          p_bad['state'] == RELPAIR_UNAVAILABLE
          and p_bad['product'] is None
          and any(bad_state in r for r in p_bad['reason']),
          str(p_bad['reason']))

# 5 符号缺失/非法/非有限
for bad_sign, tag in ((None, '缺失'), (0, '零'), (float('nan'), 'NaN'),
                      (True, 'bool')):
    s5 = mk_sign({'k1': ext(1), 'k2': {'state': SIGN_EXTRACTED,
                                       'sign': bad_sign}})
    r5 = relative_sign_pair_reports(s5, mk_report(
        {'k1': LAYER_NON_OSC, 'k2': LAYER_NON_OSC}))
    p5 = r5['pairs'][0]
    check('T5', f'符号{tag} → 无有效乘积（unavailable + product None）',
          p5['state'] == RELPAIR_UNAVAILABLE
          and p5['product'] is None
          and any('符号值' in r for r in p5['reason']),
          str(p5['reason']))

# 6 跨层配对拒绝
s6 = mk_sign({'k1': ext(1), 'k2': ext(1)})
r6 = relative_sign_pair_reports(s6, mk_report({'k1': LAYER_NON_OSC,
                                               'k2': LAYER_OSC}))
p6 = r6['pairs'][0]
check('T6', '跨层配对被拒绝（unavailable + 原因含跨层），即使双方符号'
      '均有效',
      p6['state'] == RELPAIR_UNAVAILABLE and p6['product'] is None
      and any('跨层' in r for r in p6['reason']),
      str(p6['reason']))

# 7 unknown 层与缺失层
s7a = mk_sign({'k1': ext(1), 'k2': ext(1)})
r7a = relative_sign_pair_reports(s7a, mk_report({'k1': LAYER_NON_OSC,
                                                 'k2': LAYER_UNKNOWN}))
s7b = mk_sign({'k1': ext(1), 'k2': ext(1)})
r7b = relative_sign_pair_reports(s7b, mk_report({'k1': LAYER_NON_OSC,
                                                 'k2': None}))
check('T7', 'unknown 层 / 缺失层 → 保守拒绝（不依据符号值推断）',
      r7a['pairs'][0]['state'] == RELPAIR_UNAVAILABLE
      and any('层归属' in r for r in r7a['pairs'][0]['reason'])
      and r7b['pairs'][0]['state'] == RELPAIR_UNAVAILABLE
      and r7b['pairs'][0]['product'] is None,
      f"{r7a['pairs'][0]['reason']} / {r7b['pairs'][0]['reason']}")

# 8 规范顺序/无自配对
s8 = mk_sign({'c': ext(1), 'a': ext(1), 'b': ext(-1)})
r8 = relative_sign_pair_reports(s8, mk_report(
    {'a': LAYER_NON_OSC, 'b': LAYER_NON_OSC, 'c': LAYER_NON_OSC}))
check('T8', '三元素 → 3 对（C(3,2)）、规范顺序、无自配对、无 (i,j)/(j,i) '
      '重复',
      r8['n_pairs'] == 3
      and all(p['pair'][0] < p['pair'][1] for p in r8['pairs'])
      and len({tuple(p['pair']) for p in r8['pairs']}) == 3
      and {tuple(p['pair']) for p in r8['pairs']}
      == {('a', 'b'), ('a', 'c'), ('b', 'c')},
      str([p['pair'] for p in r8['pairs']]))

# 9 空/缺/malformed
r9a = relative_sign_pair_reports(None, mk_report({'k1': LAYER_NON_OSC}))
r9b = relative_sign_pair_reports({'per_element': {}, 'per_layer': {}},
                                 mk_report({}))
r9c = relative_sign_pair_reports(mk_sign({'k1': ext(1), 'k2': ext(1)}),
                                 None)
r9d = relative_sign_pair_reports(
    {'per_element': {'k1': 'not_a_dict'}}, mk_report({'k1': LAYER_NON_OSC}))
check('T9', 'None 输入/空 per_element/report 缺失/malformed 条目 → '
      '不崩溃不伪造（空结果或全部 unavailable）',
      r9a['pairs'] == [] and r9b['pairs'] == []
      and r9c['pairs'][0]['state'] == RELPAIR_UNAVAILABLE
      and any('层归属' in r for r in r9c['pairs'][0]['reason'])
      and r9d['pairs'] == [],
      f"{len(r9a['pairs'])}/{len(r9b['pairs'])}/"
      f"{len(r9c['pairs'])}/{len(r9d['pairs'])}")

# 13 非 dict 输入 → ValueError（数据级异常语义，不伪装成空配对结果）
for bad_input, arg_name in (('not_a_dict', 'sign_channel'),
                            (['list'], 'sign_channel'),
                            ('not_a_dict', 'report'),
                            (123, 'report')):
    try:
        if arg_name == 'sign_channel':
            relative_sign_pair_reports(bad_input, mk_report(
                {'k1': LAYER_NON_OSC}))
        else:
            relative_sign_pair_reports(mk_sign({'k1': ext(1)}), bad_input)
        check('T13', f'{arg_name} 非 dict（{type(bad_input).__name__}）→ '
              '应抛 ValueError', False, '未抛异常')
    except ValueError as e:
        check('T13', f'{arg_name} 非 dict（{type(bad_input).__name__}）→ '
              'ValueError + 可追溯消息（类型名与参数名）',
              arg_name in str(e) and type(bad_input).__name__ in str(e),
              str(e)[:70])
try:
    relative_sign_pair_reports(
        {'per_element': ['not_a_dict']},
        mk_report({'k1': LAYER_NON_OSC}))
    check('T13', 'per_element 非 dict → 应抛 ValueError', False, '未抛异常')
except ValueError as e:
    check('T13', 'per_element 非 dict → ValueError + 可追溯消息',
          'per_element' in str(e), str(e)[:70])

# 14 非有限 provenance → None 化 + 严格 JSON 可序列化
for bad_val, tag in ((float('nan'), 'NaN'), (float('inf'), '+inf'),
                     (float('-inf'), '-inf')):
    s14 = mk_sign({'k1': {'state': SIGN_EXTRACTED, 'sign': 1,
                          'floor': bad_val, 'n_pts': bad_val},
                   'k2': ext(-1)})
    r14 = relative_sign_pair_reports(s14, mk_report(
        {'k1': LAYER_NON_OSC, 'k2': LAYER_NON_OSC}))
    p14 = r14['pairs'][0]
    check('T14', f'provenance 非有限（{tag}）→ noise_floor_used 与 '
          'n_points_above_floor 转 None、符号判定不受影响',
          p14['provenance']['k1']['noise_floor_used'] is None
          and p14['provenance']['k1']['n_points_above_floor'] is None
          and p14['state'] == RELPAIR_AVAILABLE
          and p14['product'] == -1,
          str(p14['provenance']['k1']))
    try:
        json.dumps(r14, allow_nan=False)
        check('T14', f'provenance 非有限（{tag}）→ '
              'json.dumps(allow_nan=False) 成功', True)
    except ValueError:
        check('T14', f'provenance 非有限（{tag}）→ '
              'json.dumps(allow_nan=False) 成功', False, '序列化失败')

# 10 provenance 与实际输入一致
s10 = mk_sign({'k1': {'state': SIGN_EXTRACTED, 'sign': 1, 'n_pts': 7,
                      'floor': 2.5e-6},
               'k2': ext(-1)})
r10 = relative_sign_pair_reports(s10, mk_report(
    {'k1': LAYER_NON_OSC, 'k2': LAYER_NON_OSC}))
prov10 = r10['pairs'][0]['provenance']
check('T10', 'provenance 与实际输入一致（sign/state/layer/n_pts/floor）',
      prov10['k1']['sign'] == 1
      and prov10['k1']['state'] == SIGN_EXTRACTED
      and prov10['k1']['layer'] == LAYER_NON_OSC
      and prov10['k1']['n_points_above_floor'] == 7
      and prov10['k1']['noise_floor_used'] == 2.5e-6
      and prov10['k2']['sign'] == -1,
      str(prov10))

# 11 只消费原始符号状态，不混入振幅符号（伪造振幅符号字段，输出不跟随）
s11 = mk_sign({'k1': ext(1), 'k2': {'state': SIGN_EXTRACTED, 'sign': 1,
                                    'extra': {'amplitude_sign': -1}}})
r11 = relative_sign_pair_reports(s11, mk_report(
    {'k1': LAYER_NON_OSC, 'k2': LAYER_NON_OSC}))
check('T11', '伪造 amplitude_sign 字段不被消费：product 跟随原始符号 '
      'sign 而非振幅符号',
      r11['pairs'][0]['product'] == 1
      and r11['pairs'][0]['state'] == RELPAIR_AVAILABLE,
      str(r11['pairs'][0]['provenance']))

# 12 输出无候选结论字段（5B-2 语义保持）
r12 = r12b
check('T12', '输出不含 supported_candidate/候选距离字段 + '
      'no_physical_conclusion 声明存在 + 严格 JSON',
      'supported_candidate' not in r12
      and 'candidate_distances' not in r12
      and 'no_physical_conclusion' in r12
      and json.dumps(r12, allow_nan=False, default=str))

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
