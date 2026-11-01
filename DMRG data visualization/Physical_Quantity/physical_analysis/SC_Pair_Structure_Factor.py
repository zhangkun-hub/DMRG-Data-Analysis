# -*- coding: utf-8 -*-
"""
超导配对结构因子模块（受限估计，bond 中点表示，Phase C 第一阶段）

定义（固定参考键、单方向目标键集合的受限配对结构因子）：
    P_X(q) = (1/N) Σ_{t ∈ B_X} exp[i q·(R_ref − R_t)] · <Δ†_ref Δ_t>

- t = (k, l)：方向 X 文件内的全部实测目标键；R_t = (r_k + r_l)/2 为 bond
  中点坐标（由 HoneycombXCGeometry.coords 精确计算）；R_ref 为参考键中点
- 相位因子 exp[i q·(r_i − r_j)]、坐标差顺序 r_i − r_j、归一化 1/N、
  NaN 跳过与计数、复数 float() 强转行为均继承公共核
  compute_structure_factor，并经独立探针实证（Phase C Step A：格点对
  与整数 ID 键两种场景逐位一致，非仅依据注释推断）
- 公共核 C 键仅接受 int（int() 强转），本模块在模块侧做 bond → 唯一
  整数 ID（i*N + j，0≤i,j<N 保证唯一、可逆）编码，公共核零修改
- 关联量 <Δ†Δ> = 0.5(a − b − c + d)：通道定义与合成公式沿用
  Super_Correlation_Process（费米子符号已与 Δ†_ij=(c†_i↑c†_j↓−c†_i↓c†_j↑)/√2、
  Δ_kl=(c_k↑c_l↓−c_k↓c_l↑)/√2 逐项核对一致，不重新推导）

受限语义（诚实表达，不虚构覆盖率）：
- 输出是"固定参考键 × 单方向目标键集合"的**一行受限结构因子**，不是
  完整配对结构因子矩阵（每目录仅 1 个参考键，完整矩阵数据不存在）
- expected_pair_count / pair_coverage 恒为 None：方向 A/B/C 的几何规则
  未在代码中权威确认（extract_data 的 allowed_cond 是衰减提取过滤，
  非方向定义），预期目标键集合不可计算
- is_restricted 恒为 True；source_coverage 置 None（"源点数/N"口径对
  bond 源不适用，由 ref_bond / target_bond_count 表达）
- P(−q) = P(q)* 是**需验证的性质而非假设**：受限单参考键数据共轭对
  缺失，虚部非零是数据不完整性的直接反映（与电荷模块同语义），
  imaginary_residual 如实报告，不取实部、不强制对称化

解析口径：
- estimate_from_files 全量解析方向文件（无 allowed_cond / 右边界过滤）——
  与 SuperCorrRecord.extract_data（衰减分析的过滤子集口径）不同，属设计
  决策；record_source 字段如实标注
- 四通道合成仅在目标键集合、坐标、相位权重与缺失数据处理一致时执行：
  集合级校验（本模块）+ 逐元素校验（Combined_super_correlation）；
  任一不一致 → ValueError（禁止未经支持的完整跨通道合成），逐通道结果
  仍可用 estimate_channel_from_records 单独获得
- 两条计算路径（先合成再算 SF vs 通道 SF 线性组合 0.5(P_a−P_b−P_c+P_d)）
  仅在上述一致性条件下等价，由合成测试显式验证，不默认成立

本模块不判定配对对称性、不判定 d+id、不判定任何物理相态。
"""

import ast
import os

import numpy as np

from ..data_records.Honeycomb_Lattice import HoneycombXCGeometry
from ..data_records.Physical_Data_Record import SuperCorrRecord
from .Spin_Structure_Factor import (
    compute_structure_factor, _xc_geometry_meta, _q_grid_stats)
from .Super_Correlation_Process import SuperCorrProcessor

# 关联量定义标签（与自旋/电荷结构因子的标签严格区分）
DEF_SC_PAIR = 'sc_pair'   # 合成 <Δ†Δ> = 0.5(a − b − c + d)
DEF_SC_PAIR_CHANNELS = {
    'a': 'sc_pair_channel_a',   # <c†_i↑ c†_j↓ c_k↑ c_l↓>
    'b': 'sc_pair_channel_b',   # <c†_i↓ c†_j↑ c_k↑ c_l↓>
    'c': 'sc_pair_channel_c',   # <c†_i↑ c†_j↓ c_k↓ c_l↑>
    'd': 'sc_pair_channel_d',   # <c†_i↓ c†_j↑ c_k↓ c_l↑>
}
VALID_DIRECTIONS = ('A', 'B', 'C')


class SCPairStructureFactorEstimator:
    """超导配对结构因子受限估计（固定参考键 × 单方向）。

    输出与自旋/电荷 estimator 同构的 dict（q/S_q/definition/N/…/meta/
    q_grid），另加 SC 专属键：direction / channel / ref_bond /
    target_bond_count / target_bonds / record_source /
    channel_target_consistency（仅四通道入口）/ channel_sf（可选）。
    """

    # ------------------------------------------------------------------
    @staticmethod
    def parse_direction_file(path, N):
        """全量解析方向文件（[[i,j,k,l],[v,0]]），无 allowed_cond / 右边界
        过滤（那是 SuperCorrRecord.extract_data 的衰减分析口径）。

        校验：参考键全文件一致、索引 0≤idx<N、值非复数（复数在现有测量
        语义下不存在，明确拒绝而非静默实部化）。NaN/Inf 保留（公共核
        跳过并计数）。返回 [(i, j, k, l, value), ...]（保持文件顺序）。
        """
        with open(path, encoding='utf-8') as f:
            try:
                data = ast.literal_eval(f.read())
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"输入文件 {path} 格式错误，无法解析: {e}")
        if not data:
            raise ValueError(f"输入文件 {path} 内容为空")
        elements = []
        first_bond = None
        for m, element in enumerate(data):
            try:
                [i, j, k, l], [value, *_] = element
            except (TypeError, ValueError):
                raise ValueError(
                    f"第 {m} 个元素格式不符合 [[i,j,k,l],[v,0]]: {element}")
            bond = (int(i), int(j))
            if first_bond is None:
                first_bond = bond
            elif bond != first_bond:
                raise ValueError(
                    f"数据不一致：第 {m} 个元素参考键 {bond}，预期 {first_bond}")
            for idx, name in ((i, 'i'), (j, 'j'), (k, 'k'), (l, 'l')):
                if not (0 <= int(idx) < N):
                    raise ValueError(
                        f"第 {m} 个元素索引 {name}={idx} 超出 [0,{N})")
            if isinstance(value, complex):
                raise ValueError(
                    f"第 {m} 个元素为复数关联值，不支持（现有测量数据全实）: {value}")
            elements.append((int(i), int(j), int(k), int(l), float(value)))
        return elements

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_single(recs, direction, q_list, N, a, definition, channel):
        """单记录列表 → 公共核（bond 中点经整数 ID 编码，公共核零修改）。

        record 校验：参考键一致、目标键索引在 [0,N)、值非复数。
        相位约定与公共核一致：C 键 (ref_ID, tar_ID) → exp[i q·(R_ref − R_t)]。
        """
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction 必须为 {VALID_DIRECTIONS}，得到 {direction!r}")
        if not recs:
            raise ValueError("输入 record 列表为空，无法计算配对结构因子")
        geometry = HoneycombXCGeometry(N, a)
        ref_bond = tuple(int(x) for x in recs[0].ref_bond)
        if not all(0 <= x < N for x in ref_bond):
            raise ValueError(f"参考键 {ref_bond} 超出 [0,{N})")
        pairs = []   # (k, l, v)：目标键 + 关联值
        for r in recs:
            if isinstance(r.sup_corr, complex):
                raise ValueError(
                    "复数关联输入不支持：现有测量数据全实，公共核按 float 处理")
            rb = tuple(int(x) for x in r.ref_bond)
            if rb != ref_bond:
                raise ValueError(
                    f"record 参考键 {rb} 与首元素 {ref_bond} 不一致")
            k, l = int(r.target_bond[0]), int(r.target_bond[1])
            if not (0 <= k < N and 0 <= l < N):
                raise ValueError(f"record 目标键 {(k, l)} 超出 [0,{N})")
            pairs.append((k, l, float(r.sup_corr)))

        # bond → 整数 ID（i*N + j）：公共核 C 键仅接受 int（已实证），
        # 0≤idx<N 保证唯一且可逆（i = ID//N, j = ID%N）
        bond_ids = {}
        def _bond_id(b):
            if b not in bond_ids:
                bond_ids[b] = b[0] * N + b[1]
            return bond_ids[b]

        coords = {}
        for b in [ref_bond] + [(k, l) for (k, l, _) in pairs]:
            (x1, y1) = geometry.coords[b[0]]
            (x2, y2) = geometry.coords[b[1]]
            coords[_bond_id(b)] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        C = [(_bond_id(ref_bond), _bond_id((k, l)), v) for (k, l, v) in pairs]

        result = compute_structure_factor(coords, C, q_list, N, definition,
                                          expected_pairs=None)
        # bond 源语义修正：公共核 source_coverage = 源点数/N，对 bond 源
        # 不适用 → 置 None，由 ref_bond / target_bond_count 表达
        result['source_coverage'] = None
        # skipped_no_coords 的整数 ID 映射回 bond 元组（正常情况下为空：
        # 索引已校验且坐标字典已覆盖全部键）
        id2bond = {bid: b for b, bid in bond_ids.items()}
        result['skipped_no_coords'] = [
            tuple(id2bond.get(int(idx), f'id_{idx}') for idx in pair)
            for pair in result['skipped_no_coords']]
        result['direction'] = direction
        result['channel'] = channel
        result['ref_bond'] = list(ref_bond)
        result['target_bond_count'] = int(result['measured_pair_count'])
        result['target_bonds'] = [[k, l] for (k, l, _) in pairs]
        result['meta'] = _xc_geometry_meta(geometry)
        result['meta']['position_basis'] = 'bond_midpoint'
        result['q_grid'] = _q_grid_stats(q_list)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_from_records(recs_combined, direction, q_list, N, a=1.0):
        """从合成 <Δ†Δ>（list[SuperCorrRecord]，如
        SuperCorrProcessor.Combined_super_correlation 的输出）计算受限
        配对结构因子。definition='sc_pair'，channel='combined'。
        """
        return SCPairStructureFactorEstimator._estimate_single(
            recs_combined, direction, q_list, N, a, DEF_SC_PAIR, 'combined')

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_channel_from_records(recs_channel, channel, direction,
                                      q_list, N, a=1.0):
        """单通道 S(q)（不合成）；channel ∈ {'a','b','c','d'}。
        供通道目标键集合不一致等场景单独取得逐通道结果。
        """
        if channel not in DEF_SC_PAIR_CHANNELS:
            raise ValueError(
                f"channel 必须为 {sorted(DEF_SC_PAIR_CHANNELS)}，得到 {channel!r}")
        return SCPairStructureFactorEstimator._estimate_single(
            recs_channel, direction, q_list, N, a,
            DEF_SC_PAIR_CHANNELS[channel], channel)

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_from_channels(recs_a, recs_b, recs_c, recs_d, direction,
                               q_list, N, a=1.0, include_channels=True):
        """四通道 → 先逐通道 S(q)（可选），再合成主结果。

        合成仅在目标键集合、坐标、相位权重与缺失数据处理一致时执行：
        - 集合级校验：四通道目标键 frozenset 相等（本方法）
        - 逐元素校验：长度、参考键、逐元素目标键（Combined_super_correlation）
        任一不一致 → ValueError（禁止未经支持的完整跨通道合成）；此时
        逐通道结果仍可用 estimate_channel_from_records 单独获得。
        include_channels=True 时主结果附带 channel_sf（a/b/c/d 各自独立
        S(q)，不提前平均，供后续对称性分析使用）。
        """
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction 必须为 {VALID_DIRECTIONS}，得到 {direction!r}")
        recs = {'a': recs_a, 'b': recs_b, 'c': recs_c, 'd': recs_d}
        sets = [frozenset((int(r.target_bond[0]), int(r.target_bond[1]))
                          for r in recs[c]) for c in 'abcd']
        if len(set(sets)) != 1:
            counts = {c: len(recs[c]) for c in 'abcd'}
            raise ValueError(
                "四个通道的目标键集合不一致，禁止未经支持的完整跨通道合成 "
                f"（逐通道计数: {counts}）；逐通道结果可用 "
                "estimate_channel_from_records 单独计算")
        # 合成（逐元素校验在 Combined 内部完成，不一致同样抛 ValueError）
        combined = SuperCorrProcessor.Combined_super_correlation(
            recs_a, recs_b, recs_c, recs_d)
        result = SCPairStructureFactorEstimator._estimate_single(
            combined, direction, q_list, N, a, DEF_SC_PAIR, 'combined')
        result['channel_target_consistency'] = True
        if include_channels:
            result['channel_sf'] = {
                c: SCPairStructureFactorEstimator.estimate_channel_from_records(
                    recs[c], c, direction, q_list, N, a) for c in 'abcd'}
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_from_files(direction, refbond_dir, q_list, N, a=1.0,
                            include_channels=True):
        """便捷入口：读 super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json
        四个通道文件（全量解析）→ estimate_from_channels。

        参考键取自目录名 super_corr_{i}_{j}，与文件内参考键交叉校验。
        record_source='full_direction_file'（与衰减分析 extract_data 的
        过滤子集口径不同的设计决策，如实标注）。
        """
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction 必须为 {VALID_DIRECTIONS}，得到 {direction!r}")
        if not os.path.isdir(refbond_dir):
            raise FileNotFoundError(f"参考键目录不存在: {refbond_dir}")
        base = os.path.basename(os.path.normpath(refbond_dir))
        parts = base.split('_')
        if len(parts) < 4 or parts[0] != 'super' or parts[1] != 'corr':
            raise ValueError(
                f"目录名不符合 super_corr_{{i}}_{{j}} 约定: {refbond_dir}")
        try:
            ref_i, ref_j = int(parts[2]), int(parts[3])
        except ValueError:
            raise ValueError(f"目录名参考键无法解析为整数: {base}")

        geometry = HoneycombXCGeometry(N, a)
        channels = {}
        for ch in 'abcd':
            path = os.path.join(
                refbond_dir,
                f'scs{direction}{ch}_fix_refbond_{ref_i}_{ref_j}.json')
            elements = SCPairStructureFactorEstimator.parse_direction_file(
                path, N)
            rb = (elements[0][0], elements[0][1])
            if rb != (ref_i, ref_j):
                raise ValueError(
                    f"文件 {os.path.basename(path)} 参考键 {rb} 与目录名 "
                    f"{ref_i}_{ref_j} 不一致")
            channels[ch] = [
                SuperCorrRecord(rb, (k, l),
                                geometry.sup_corr_distance(rb[0], rb[1], k, l),
                                v)
                for (i, j, k, l, v) in elements]
        result = SCPairStructureFactorEstimator.estimate_from_channels(
            channels['a'], channels['b'], channels['c'], channels['d'],
            direction, q_list, N, a, include_channels)
        result['record_source'] = 'full_direction_file'
        return result
