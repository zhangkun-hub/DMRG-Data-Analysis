# -*- coding: utf-8 -*-
"""
N400 真实数据回归测试（SC 3 refbond + Spin 2 源点）

只读已存在基线 tests/baselines/n400_baseline_v1.json；基线缺失或
结构不完整直接 FAIL，不自动生成（防止测试静默重写基线掩盖回归）。

比较策略分层：
- 离散字段精确一致（int/bool/str 全等：classification、verdict、
  evidence、analysis_version、点数、警告列表等）
- 连续数值 np.isclose(rtol=1e-8, atol=1e-12)——不要求浮点逐位相等
  （to_dict 已将 NaN 转 null：基线为 null 的字段允许仍为 null，
  基线有值的字段出现 NaN/Inf 时 isclose 自然失败）
- 字典键集与列表结构全等（缺失字段不被静默忽略）
- SC 案例额外做 SU(2) 误差容差比较 + err < 0.2 完整性哨兵
  （0.2 是防止数据损坏/解析异常的宽松上界，不是 SU(2) 物理可接受阈值，
  不据此判断数据物理可信）
- 序列化 JSON 读回后结构仍与基线一致（NaN→null 规范）

物理结论声明：本测试只验证真实数据分析流程的回归稳定性，
不据此判定任何相态。

运行方式（在 DMRG data visualization/ 目录下）：
    python tests/test_n400_regression.py
"""
import sys
import os
import json
import copy

sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Physical_Quantity.Physical_Data_Record import SuperCorrRecord, CorrelationRecord
from Physical_Quantity.Super_Correlation_Process import SuperCorrProcessor
from Physical_Quantity.Correlation_Process import CorrelationProcessor
from Physical_Quantity.Super_Correlation_Analyzer import SuperCorrDecayAnalyzer
from Physical_Quantity.Spin_Correlation_Analyzer import SpinCorrDecayAnalyzer

BASELINE = os.path.join(HERE, 'baselines', 'n400_baseline_v1.json')
RESULTS = {}


def check(cat, name, cond, detail=''):
    RESULTS.setdefault(cat, []).append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))


# ==========================================================================
# 数据加载（与现有测试同款加载逻辑的最小副本；不 import 其他测试模块——
# 那些模块无 __main__ 守卫，导入会执行全套测试）
# ==========================================================================
def load_sc(ref, bond, doping='d1_10', N=400):
    """读取 N400 超导关联 4 通道并合成 <Δ†Δ>；返回 (记录列表, (err_ad, err_bc))"""
    base = f'{ROOT}/Data_N400/t_J_3/{doping}/1500_measure_super_corr/super_corr_{ref}'
    recs4 = {}
    for ch in 'abcd':
        recs4[ch] = SuperCorrRecord.extract_data(
            open(f'{base}/scs{bond}{ch}_fix_refbond_{ref}.json', encoding='utf-8').read(),
            N, 1.0)
    err_ad, err_bc = SuperCorrProcessor.symmetry_check(
        recs4['a'], recs4['b'], recs4['c'], recs4['d'])
    comb = SuperCorrProcessor.Combined_super_correlation(
        recs4['a'], recs4['b'], recs4['c'], recs4['d'])
    return comb, (err_ad, err_bc)


def load_spin(measure, ver, N=400):
    """读取 N400 自旋关联三分量并合成 <S_i·S_j>"""
    base = f'{ROOT}/Data_N400/t_J_3/d1_10/1500_measure_phys_qty/{measure}'
    seq = range(1, N + 100, 4) if ver == '97' else range(2, N + 100, 4)
    sz = CorrelationRecord.extract_data(
        open(f'{base}/sz{ver}sz.json', encoding='utf-8').read(), seq, N, 1.0)
    su = CorrelationRecord.extract_data(
        open(f'{base}/sp{ver}sm.json', encoding='utf-8').read(), seq, N, 1.0)
    sd = CorrelationRecord.extract_data(
        open(f'{base}/sm{ver}sp.json', encoding='utf-8').read(), seq, N, 1.0)
    return CorrelationProcessor.Combined_spin_correlation(sz, su, sd)


# ==========================================================================
# 分层比较：离散字段精确、连续数值容差、键集/结构全等
# ==========================================================================
def compare_tolerant(a, b, path=''):
    """递归比较（a=基线，b=重算）。返回差异路径列表（空 = 一致）。
    - dict：键集必须全等（缺失字段即差异）
    - list：长度必须全等
    - int/bool/str：精确全等（离散字段）
    - float：np.isclose(rtol=1e-8, atol=1e-12)（连续数值，不逐位相等；
      NaN/Inf 与任何有限值都不 isclose，自动判差异）
    - None 对 None 一致；None 对非 None 为差异"""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            diffs.append(f'{path} 键集不一致: {set(a) ^ set(b)}')
            return diffs
        for k in a:
            diffs.extend(compare_tolerant(a[k], b[k], f'{path}.{k}'))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path} 长度 {len(a)} vs {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(compare_tolerant(x, y, f'{path}[{i}]'))
    elif a is None and b is None:
        return diffs
    elif a is None or b is None:
        diffs.append(f'{path}: {a!r} vs {b!r}')
    elif isinstance(a, bool) or isinstance(b, bool):
        # bool 与 int/float 跨类型必须报告差异（True 与 1 不视为一致）；
        # 双双 bool 且相等才一致
        if not (type(a) is type(b) and a == b):
            diffs.append(f'{path}: {a!r} vs {b!r}')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                diffs.append(f'{path}: {a!r} vs {b!r}')
        elif not np.isclose(float(a), float(b), rtol=1e-8, atol=1e-12):
            # int/float 混合与 float/float 均走数值容差比较
            diffs.append(f'{path}: {a!r} vs {b!r}')
    elif str(a) != str(b):
        diffs.append(f'{path}: {a!r} vs {b!r}')
    return diffs


if __name__ == '__main__':
    # ======================================================================
    # A. 基线加载与结构（正式测试只读基线；缺失/错配即 FAIL，不自动生成）
    # ======================================================================
    print('===== A. 基线加载与结构 =====')
    if not os.path.exists(BASELINE):
        check('A', '基线文件存在', False,
              f'缺少 {BASELINE}——正式测试不自动生成基线，请先人工生成并审查后入库')
        print('总计: 0 通过 / 1 失败')
        sys.exit(1)
    with open(BASELINE, encoding='utf-8') as f:
        base = json.load(f)
    check('A', 'baseline_kind = n400_v1',
          base.get('_meta', {}).get('baseline_kind') == 'n400_v1',
          base.get('_meta', {}).get('baseline_kind'))
    expected_cases = ['n400_sc_97_98_A', 'n400_sc_98_99_A', 'n400_sc_98_102_A',
                      'n400_spin_97', 'n400_spin_98']
    check('A', '基线含 5 个预期案例键',
          all(c in base for c in expected_cases) and '_meta' in base,
          [c for c in expected_cases if c not in base] or '齐全')
    check('A', '_meta.cases 与预期一致',
          base.get('_meta', {}).get('cases') == expected_cases,
          base.get('_meta', {}).get('cases'))

    # 基线条目结构检查（访问嵌套字段前防御；结构损坏用 check 报明确 FAIL，
    # 不依赖 KeyError 崩溃）；仅检查，不生成/覆盖/修复基线
    struct_bad = set()
    for c in expected_cases:
        entry = base.get(c)
        problems = []
        if not isinstance(entry, dict):
            problems.append('条目不是 dict')
        else:
            if 'result' not in entry:
                problems.append('缺 result 字段')
            if c.startswith('n400_sc_'):
                sym = entry.get('symmetry')
                if not (isinstance(sym, dict) and 'err_ad' in sym and 'err_bc' in sym):
                    problems.append('缺 symmetry/err_ad/err_bc 字段')
        check('A', f'{c} 基线结构完整', not problems, '; '.join(problems) or '完整')
        if problems:
            struct_bad.add(c)

    # compare_tolerant 自检（内存构造，不依赖正式基线、不写文件；
    # 固化"测试能捕获回归"为可重复自动化覆盖）
    print('===== A2. compare_tolerant 自检 =====')
    tpl = {'classification': 'power_law', 'n': 3, 'none_field': None,
           'params': {'A': 1.0, 'K': 2.5}, 'nested': {'x': 1e-6},
           'flags': [True, False], 'items': ['a', 'b']}
    b = copy.deepcopy(tpl)
    b['classification'] = 'exponential'   # 1. 离散字段改错
    check('A2', '离散字段改错报告差异', bool(compare_tolerant(tpl, b)),
          f'{len(compare_tolerant(tpl, b))} 处')
    b = copy.deepcopy(tpl)
    b['params']['K'] = 3.0                # 2. 连续数值超容差（2.5 → 3.0）
    check('A2', '连续数值超容差报告差异', bool(compare_tolerant(tpl, b)),
          f'{len(compare_tolerant(tpl, b))} 处')
    b = copy.deepcopy(tpl)
    del b['params']['A']                  # 3. 字典缺键
    check('A2', '字典缺键报告差异', bool(compare_tolerant(tpl, b)),
          f'{len(compare_tolerant(tpl, b))} 处')
    b = copy.deepcopy(tpl)
    b['nested']['x'] *= (1.0 + 1e-12)     # 4. 容差内微扰
    check('A2', '容差内微扰不误报', not compare_tolerant(tpl, b),
          f'{len(compare_tolerant(tpl, b))} 处')
    check('A2', '对象与自身比较通过', not compare_tolerant(tpl, tpl),
          f'{len(compare_tolerant(tpl, tpl))} 处')
    check('A2', 'bool vs int 跨类型报告差异（True vs 1）',
          bool(compare_tolerant({'f': True}, {'f': 1})))
    check('A2', 'int 与 float 同值通过数值容差（1 vs 1.0）',
          not compare_tolerant({'f': 1}, {'f': 1.0}))
    check('A2', 'float 超容差报告差异（1.0 vs 1.0+1e-6）',
          bool(compare_tolerant({'f': 1.0}, {'f': 1.0 + 1e-6})))
    check('A2', 'None 与 None 一致、None 与数值报告差异',
          not compare_tolerant({'f': None}, {'f': None})
          and bool(compare_tolerant({'f': None}, {'f': 1.0})))

    # ======================================================================
    # B. SC 案例回归（3 个 refbond × A 键）
    # ======================================================================
    print('===== B. SC 案例（N400，d1_10，bond_dim 1500） =====')
    SU2_SENTINEL = 0.2   # 完整性哨兵上界：防数据损坏/解析异常，非 SU(2) 物理可接受阈值
    for name, ref, bond in [('n400_sc_97_98_A', '97_98', 'A'),
                            ('n400_sc_98_99_A', '98_99', 'A'),
                            ('n400_sc_98_102_A', '98_102', 'A')]:
        if name in struct_bad:
            check('B', f'{name} 跳过（基线结构损坏，A 节已 FAIL）', False)
            continue
        comb, (err_ad, err_bc) = load_sc(ref, bond)
        res = SuperCorrDecayAnalyzer.analyze(comb)
        entry = base[name]

        # 输入点数一致（防止数据文件被替换/截断而未被察觉）
        check('B', f'{name} 输入记录数与基线一致',
              len(comb) == entry['input_records'],
              f"{len(comb)} vs {entry['input_records']}")

        # 结构完整性（沿用现有测试模式）
        d = res.to_dict()
        ok = (d['classification'] and isinstance(d.get('envelope_fit'), dict)
              and isinstance(d.get('oscillation'), dict)
              and isinstance(d.get('stability'), dict))
        check('B', f'{name} 结构完整',
              ok, f"cls={d['classification']}")

        # 分层比较：离散精确 + 连续容差 + 键集/结构全等
        diffs = compare_tolerant(entry['result'], d)
        check('B', f'{name} 与基线分层比较一致', not diffs,
              f'{len(diffs)} 处差异' if diffs else '一致')

        # SU(2) 误差：容差比较 + 完整性哨兵（0.2 非物理可接受阈值）
        sym = entry['symmetry']
        ok_sym = (np.isclose(err_ad, sym['err_ad'], rtol=1e-8, atol=1e-12)
                  and np.isclose(err_bc, sym['err_bc'], rtol=1e-8, atol=1e-12))
        check('B', f'{name} SU(2) 误差与基线一致', ok_sym,
              f"err_ad={err_ad:.4f} err_bc={err_bc:.4f}")
        check('B', f'{name} SU(2) 完整性哨兵（err < {SU2_SENTINEL}，'
                   f'防数据损坏/解析异常，非物理可接受阈值）',
              err_ad < SU2_SENTINEL and err_bc < SU2_SENTINEL,
              f"err_ad={err_ad:.4f} err_bc={err_bc:.4f}")

        # 序列化 JSON 读回：NaN→null 规范，读回后结构与基线仍一致
        back = json.loads(json.dumps(d, ensure_ascii=False))
        diffs_back = compare_tolerant(entry['result'], back)
        check('B', f'{name} JSON 序列化读回一致', not diffs_back,
              f'{len(diffs_back)} 处差异' if diffs_back else '一致')

    # ======================================================================
    # C. Spin 案例回归（2 个源点）
    # ======================================================================
    print('===== C. Spin 案例（N400，d1_10） =====')
    for name, measure, ver in [('n400_spin_97', 'measure_97', '97'),
                               ('n400_spin_98', 'measure_98', '98')]:
        if name in struct_bad:
            check('C', f'{name} 跳过（基线结构损坏，A 节已 FAIL）', False)
            continue
        comb = load_spin(measure, ver)
        res = SpinCorrDecayAnalyzer.analyze(comb, N=400)
        entry = base[name]

        check('C', f'{name} 输入记录数与基线一致',
              len(comb) == entry['input_records'],
              f"{len(comb)} vs {entry['input_records']}")
        d = res.to_dict()
        ok = (d['classification'] and isinstance(d.get('envelope_fit'), dict)
              and isinstance(d.get('oscillation'), dict)
              and isinstance(d.get('stability'), dict))
        check('C', f'{name} 结构完整', ok, f"cls={d['classification']}")
        diffs = compare_tolerant(entry['result'], d)
        check('C', f'{name} 与基线分层比较一致', not diffs,
              f'{len(diffs)} 处差异' if diffs else '一致')
        back = json.loads(json.dumps(d, ensure_ascii=False))
        diffs_back = compare_tolerant(entry['result'], back)
        check('C', f'{name} JSON 序列化读回一致', not diffs_back,
              f'{len(diffs_back)} 处差异' if diffs_back else '一致')

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
