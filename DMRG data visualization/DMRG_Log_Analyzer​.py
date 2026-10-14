# -*- coding: utf-8 -*-
"""
DMRG 优化日志能量提取脚本（P2 修订：中段值口径保留 + 末行值口径新增）

两种口径（均为程序报告的体系变分能量估计，不宣称必然是真实基态能量或
绝对最准确的能量）：
- 中段值（保留原行为）：限定 Site (40, 41) 的 E0 行，按 (D, sweep) 平均——
  sweep 中段的中间估计，用于收敛性交叉核验
- 末行值（P2 新增）：匹配全部 E0 行，取每个 (D, sweep) 分组的最后一条——
  该 sweep 最后一次局域更新后的能量终值，作为该 sweep 的能量口径

分组规则：D 阶段以 `Bond dimension: X/X` 头行为准；sweep 编号允许在不同
D 阶段内重启（每阶段 1..N）；不使用 E0 行内的瞬时 D（如重启期 1631/1808）
覆盖阶段 D。

运行方式（交互）：python DMRG_Log_Analyzer.py < log 文件
导入方式（测试用）：模块级 import 不会读 stdin；extract_last_e0 可直接调用。
"""
import sys

from Physical_Quantity.DMRG_Log_Parser import DMRGLogParser

# 中段值口径（保留，交叉核验用）：限定 Site (40, 41) 的 E0 行
MID_ENERGY_PATTERN = (
    r'^Site\s*\(\s*40,\s*41\).*?'
    r'E0\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)')
# 末行值口径（P2 新增）：匹配全部 E0 行，再取每组最后一条
ALL_ENERGY_PATTERN = (
    r'^Site\s*\(\s*\d+,\s*\d+\).*?'
    r'E0\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)')


def extract_last_e0(data):
    """提取每个 (D, sweep) 分组的末行 E0：{(D, sweep): 末行 E0}。

    末行是该 sweep 最后一次局域更新后程序报告的全体系变分能量估计
    （不宣称必然是真实基态能量或绝对最准确的能量）。
    空输入 / 无匹配 E0 行 / 无有效分组 → 返回空 dict（不静默伪造结果）。
    """
    parser = DMRGLogParser(energy_pattern=ALL_ENERGY_PATTERN)
    parser.parse_from_string(data)
    return {key: values[-1] for key, values in parser.get_raw_dict().items()}


def main():
    """stdin 读日志：先按原中段值口径打印收敛表并绘图，再打印末行值表。"""
    print("请输入能量数据（Ctrl+Z(windows) 或 Ctrl+Z(Linux/Mac) 结束输入）:")
    data = sys.stdin.read()

    # 1. 中段值口径（保留原行为：解析 + 收敛表 + 收敛图）
    parser = DMRGLogParser(energy_pattern=MID_ENERGY_PATTERN)
    parser.parse_from_string(data)
    print("中段值口径（Site 40,41，交叉核验用）:")
    print(parser.get_dataframe().head())
    parser.plot_DMRG_energy_convergence_analysis()

    # 2. 末行值口径（P2 新增）
    last = extract_last_e0(data)
    if not last:
        print("末行值口径：未提取到任何 (D, sweep) 组的 E0"
              "（空输入或无匹配行），不伪造结果")
        return
    print("末行值口径（每 (D,sweep) 终值，按 D、sweep 排序）:")
    for key in sorted(last):
        print(f"  D={key[0]:5d} sweep={key[1]:3d}  E0_last = {last[key]:.12f}")


if __name__ == '__main__':
    main()
