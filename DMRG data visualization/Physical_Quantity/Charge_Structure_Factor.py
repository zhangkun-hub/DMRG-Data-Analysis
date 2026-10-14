# -*- coding: utf-8 -*-
"""
受限电荷结构因子估计模块（蜂窝 XC 圆柱 t-J 模型，D4.2）

定义（电子密度连通关联，D4.1 已确认）：
    C_ij = <n_i n_j> - <n_i><n_j>        (i ≠ j：两点裸关联减平均)
    C_ii = <n_i>(1 - <n_i>)             (i = j：占据数幂等 n_i² = n_i)

    S_c(q) = (1/N) Σ_{(i,j)} exp[i q·(r_i - r_j)] C_ij

复用：通用核 compute_structure_factor（归一化 1/N、有序对求和、
expected_pairs=None → pair_coverage 不可完整评估）；xc_momentum_grid；
DensityRecord / CorrelationRecord 解析；几何元信息 _xc_geometry_meta。

限制（受限语义与自旋结构因子一致）：
- 现有数据仅含源点 49/50 的两点关联（nf49nf / nf50nf），完整
  200×200 关联矩阵不存在 → 输出一律 is_restricted=True 的受限估计，
  不补零、不插值、不冒充完整电荷结构因子
- 对角项仅对实测源点按幂等关系严格补全（物理恒等式，非拟合近似）；
  不扩展为全格点对角项（避免把 source_coverage 口径从 2/200 扭曲成 1.0）
- source_coverage = 实测源点数 / N；pair_coverage 保持 None（未知）
- 本模块不根据峰位或峰高判定 CDW 或其他物理相

虚部语义说明：
- 完整实数对称关联矩阵（C_ij = C_ji 且 (i,j)、(j,i) 均有实测数据）的
  求和会产生实数 S(q)：共轭成对的复指数虚部相互抵消
- 受限数据只有单向对（如仅 (49,j)、(50,j) 而无反向对），共轭对缺失，
  S(q) 虚部非零是**输入关联数据不完整性的直接反映**——不应解释为
  完整结构因子的物理虚部，也不能仅凭该现象认定实现错误
- imaginary_residual 仅如实报告该数值；实现不取实部、不强制对称化
  （那会把推导数据伪装成实测）

受限结果比较限制：
- 不同源点集合或不同覆盖率得到的受限结构因子，不应未经验证就直接
  比较；当前不提供归一化补偿、不进行反向关联推导，也不据此判断
  CDW 或其他物理相态
"""

import numpy as np

from .Honeycomb_Lattice import HoneycombXCGeometry
from .Physical_Data_Record import DensityRecord, CorrelationRecord
from .Spin_Structure_Factor import (
    compute_structure_factor, _xc_geometry_meta)

# 关联量定义标签（与自旋结构因子的标签严格区分）
DEF_CHARGE_DENSITY_CONNECTED = 'charge_density_connected'


class ChargeStructureFactorEstimator:
    """受限电荷结构因子估计（连通形式）。

    输入：nf.json（全格点电子密度 <n_i>）+ nf{ver}nf.json（两点裸关联
    <n_i n_j>，每文件一个源点）。输出 compute_structure_factor 的 dict
    （附加 meta 与 diagonal_source_sites 键）。
    """

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_from_files(nf_path, corr_paths, q_list, N, a=1.0,
                            sequences=None) -> dict:
        """从文件估计受限电荷结构因子。

        参数
        ----
        nf_path   : nf.json 路径（全格点单点电子密度，[[i],[v,0]] 格式）
        corr_paths: list[nf{ver}nf.json 路径]；每个文件一个源点，
                    两文件间源点不同、field 无对角项
        q_list    : 动量点（复用 xc_momentum_grid 的输出）
        N, a      : 系统格点数与晶格常数
        sequences : list[range]，解析各 corr 文件的 field 筛选序列；
                    None → 全部 range(0, N, 1)（保留连续 field 全量）

        连通化规则：
        - 每个两点记录：C = <n_i n_j> - <n_i><n_j>
        - 每个实测源点 i 的对角项：C_ii = <n_i>(1 - <n_i>)（幂等补全）
        """
        if not corr_paths:
            raise ValueError("corr_paths 为空，无法计算电荷结构因子")
        if sequences is None:
            sequences = [range(0, N, 1)] * len(corr_paths)
        if len(sequences) != len(corr_paths):
            raise ValueError("sequences 与 corr_paths 长度不一致")

        geometry = HoneycombXCGeometry(N, a)
        dens_recs = DensityRecord.extract_data(
            open(nf_path, encoding='utf-8').read(), N, a)
        density = {int(r.lattice_point): float(r.density) for r in dens_recs}

        pairs = []
        sources = set()
        for path, seq in zip(corr_paths, sequences):
            recs = CorrelationRecord.extract_data(
                open(path, encoding='utf-8').read(), seq, N, a)
            for r in recs:
                i = int(r.source_point)
                j = int(r.field_point)
                if i not in density or j not in density:
                    raise ValueError(
                        f"格点 {i if i not in density else j} 缺少单点密度，"
                        f"无法连通化（nf 与 nf{{ver}}nf 应来自同一 measure）")
                # 连通化：减平均 <n_i><n_j>
                pairs.append((i, j, float(r.correlation)
                              - density[i] * density[j]))
                sources.add(i)

        # 对角项：仅实测源点，占据数幂等（n_i² = n_i）→ C_ii = <n_i>(1-<n_i>)
        for i in sorted(sources):
            pairs.append((i, i, density[i] * (1.0 - density[i])))

        result = compute_structure_factor(
            geometry.coords, pairs, q_list, N,
            definition=DEF_CHARGE_DENSITY_CONNECTED, expected_pairs=None)
        result['meta'] = _xc_geometry_meta(geometry)
        # 记录对角补全的源点（可追溯；不改变受限语义）
        result['diagonal_source_sites'] = sorted(sources)
        # 实际 q 网格统计（只描述输入 q_list 的事实，不参与计算、
        # 不排序/去重计算输入；不宣称这些值是规则网格生成参数——
        # n_qx/n_qy 是唯一坐标数量，qx_min/qx_max 是实际端点，
        # 不保证等于原始 xc_momentum_grid 调用参数）。
        # 仅统计两个坐标分量均为有限值的动量点：非有限坐标不计入
        # 数量与端点（避免"数量有效、端点为 null"的误导性元数据）；
        # 全部非有限 → 与空输入同语义（n=0、端点 None）。
        q_arr = np.asarray(q_list, dtype=float)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, 2)
        q_arr = q_arr[np.isfinite(q_arr).all(axis=1)]
        qx_vals = np.unique(np.round(q_arr[:, 0], 12))
        qy_vals = np.unique(np.round(q_arr[:, 1], 12))
        result['q_grid'] = {
            'n_qx': int(len(qx_vals)),
            'n_qy': int(len(qy_vals)),
            'qx_min': float(qx_vals.min()) if len(qx_vals) else None,
            'qx_max': float(qx_vals.max()) if len(qx_vals) else None,
            'qy_min': float(qy_vals.min()) if len(qy_vals) else None,
            'qy_max': float(qy_vals.max()) if len(qy_vals) else None,
        }
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_batch(measure_specs, q_list, N, a=1.0,
                       sequences=None) -> dict:
        """批量估计：逐组调用 estimate_from_files + 契约包装。

        参数
        ----
        measure_specs : list[dict]，每组含必需键
            'name'（唯一标识）、'nf_path'、'corr_paths'（list）；
            可选键 'sequences'（覆盖全局 sequences）
        q_list, N, a, sequences : 与 estimate_from_files 一致

        行为约定：
        - 一组失败不阻止其他独立组：FileNotFoundError / ValueError
          （数据失败）被记录进 failed，其余组继续
        - TypeError / KeyError / AttributeError 等潜在程序缺陷**不捕获**，
          原样向上传播（不静默转换为数据失败）
        - 重复 name：后出现的同名组跳过并记入 failed（DuplicateName），
          不覆盖先前结果——策略明确且可测试
        - 空 measure_specs → ValueError（与 estimate_from_files 的
          空输入语义一致）
        - 返回 {'results': dict[name → AnalysisReport],
                 'failed': dict[name → {'type', 'message'}]}

        契约包装经函数级延迟导入（Analysis_Integration 依赖本模块，
        模块级导入会形成循环）。
        """
        if not measure_specs:
            raise ValueError("measure_specs 为空，无法批量估计")
        from .Analysis_Integration import charge_structure_factor_report

        results = {}
        failed = {}
        seen = set()
        for spec in measure_specs:
            name = spec['name']   # 缺键 → KeyError 传播（不捕获）
            if name in seen:
                if name not in failed:
                    # 先失败后重复：保留原始失败记录，不覆盖
                    failed[name] = {
                        'type': 'DuplicateName',
                        'message': f'重复的 name: {name!r}，已跳过（不覆盖先前结果）'}
                continue
            seen.add(name)
            group_sequences = spec.get('sequences', sequences)
            try:
                results[name] = charge_structure_factor_report(
                    spec['nf_path'], spec['corr_paths'], q_list, N, a,
                    group_sequences)
            except (FileNotFoundError, ValueError) as e:
                # 仅数据级失败：文件缺失/格式错误/数值校验失败
                failed[name] = {'type': type(e).__name__, 'message': str(e)}
        return {'results': results, 'failed': failed}
