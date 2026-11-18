# SC_Phase_Analyzer 设计文档

> 实现口径与 `docs/DPlusID_PhaseAnalyzer_Physics_Audit.md`（2026-11-18）
> 物理审计的对应关系。阈值均为候选默认值（defaults_not_authorized，
> 未获物理授权）。

## 一、物理口径（审计 §2）

- 键方向 (A, B, C)：HoneycombXCCylinder::ClassifyNearestNeighbor；
  XC 圆柱下 C 键沿圆柱轴（θ=0°），A/B 键 θ=±120°。
- d-wave 基函数（键序 (C, A, B)，审计 §2.1）：
  - d1 = (2Δ_C − Δ_A − Δ_B)/√6（d_{x²−y²} 取向）
  - d2 = (Δ_A − Δ_B)/√2（d_{xy} 取向）
- 判据（审计 §2.2）：arg C_12(r) 的 ±90° 平台 + C_11≈C_22 一致性；
  手性指标 χ_d = 2 Im C_12/(C_11+C_22)（辅助）；**弃用 R_12**
  （d2 受抑制时数值发散，审计 §2.2）。
- 四通道合成：⟨Δ†Δ⟩ = 0.5(a − b − c + d)（复值，审计 §1.1）。

## 二、数据装配（审计 §3）

- 三 refbond（键方向 A/B/C 各一）× 12 文件**全量解析**（不经衰减
  分析的纵向通道过滤）。
- 距离对齐必须用粗 bin（容差 ~2.0）：逐键/细 bin 下 9 个 (α,β)
  组合公共距离为 0（审计 §3.1）。bin 内保留：复数平均、平均 |C|、
  复数离散度、相消比（|复均值|/平均|C|）、原始点数、距离均值/标准差。

## 三、管线（审计 §4）

1. 装配 → 2. bin（可配置容差，多容差 1.5/2.0/2.5 稳定性）→
3. 逐 bin 3×3 矩阵 → 4. d-wave 投影 C_11/C_22/C_12/C_21 →
5. 共轭诊断（仅噪声以上点；无此类点 → not_applicable）→
6. 幅值门控（|C12| > 噪声底×3；噪声底=非零值远距尾部中位数，
   零值不进尾部统计）→ 7. 圆周统计（圆周均值/圆周标准差）→
8. 平台检测（连续 ≥3 点、偏差 ≤15°、圆周 std ≤15°、±90° 分别）→
9. d2 诊断（C11 门控可靠点上中位 |C22|/|C11| 双阈值：
   < 1e-8 数值不可分辨 / [1e-8, 1e-6) 可分辨抑制 / ≥ 1e-6 无抑制）→
10. 六态分类（分类优先级见 §八 P2-2）。

## 四、分类语义（修正后）

| 分类 | 条件 |
|---|---|
| chiral_phase_candidate | ±90° 平台 + 幅值门控 + 多容差稳定 |
| no_chiral_signal | 有效数据充足但无稳定 ±90° 平台 |
| d_component_suppressed | d2 高于数值分辨比（比值 ∈ [1e-8, 1e-6)）：物理可分辨的通道抑制，无法支持双分量共存 |
| d_component_unresolved | d2 比值 < 1e-8：低于数值分辨能力，物理抑制与数值零无法区分 |
| insufficient | C12 有效点不足（相位证据不足，不输出手性结论） |
| inconclusive | 多容差分类不一致（先于物理分类） |

注：`1e-8` 为经验相对幅值分辨阈值（非机器精度/独立数值误差/普适
物理阈值）；比值 ≥ 1e-6 的语义为"未满足当前分析定义的 d2 相对
弱化判据"（详见 §八 P1-2）。

## 五、模块结构

- `physical_analysis/SC_Phase_Analyzer.py`（叶子模块）：几何、
  解析、装配、管线、分类、print_summary。
- 统一入口：`run_analysis(OBSERVABLE_SC_PAIRING_PHASE)` →
  `phase_analysis_report(datasets, N, config)` → `from_phase_analysis`。
- 验证脚本：`exploration/SC_Phase_Analysis.py`（自动发现 doping×D
  组合，产出 phase_analysis_report.md：分类总表 + 详报 + D 档一致性
  + bin 容差稳定性 + 结论边界）。

## 六、测试（tests/test_sc_phase_analyzer.py，20 项断言/15 组）

合成数据只验证算法，不构成物理证据；真实数据结论以 DMRG 数据为准。

## 七、真实数据验证结论（2026-11-18）

12 组合（4 doping × 3 档 D）全部 `d_component_suppressed`：
- d2 通道中位 |C22|/|C11| ~ 1e-7（1500 档）→ 1e-11（8000 档），
  随 D 增大 d2 更精确为零（A/B 键严格简并）；
- 无 ±90° 平台；C12 微弱值噪声主导，圆周均值 0°/±180° 随机散布；
- D 档分类一致（三档全部 d_component_suppressed）。
**物理证据层面：当前参数（N200、Ly=4、t/J=3、4 doping）未发现**
**d+id 手性候选证据；结论限于数值诊断，不推广到二维极限。**

---

## 八、独立审计修正记录（2026-11-19）

审计报告：`SC_Phase_Analyzer_Independent_Audit`（用户对话内交付）。
以下修正均已实施：

### P1-1 同 bin 跨通道偏差不再作为分类阻断

`conjugate_diagnostics` 保留为描述性诊断（含 semantics 字段说明），
`_classify` 中已无任何基于它的分支。有限 XC 圆柱（开放边界、
refbond 几何中心错位）下同一距离 bin 的 C12 与 conj(C21) 没有
同距离共轭恒等式依据；该指标不构成手性判据，也不证明数据正确性。

### P1-2 新增 `d_component_unresolved` 分类

d2 通道分辨能力三层（双阈值比值口径，阈值可配置、未获物理授权）：
- 中位 |C22|/|C11| < 1e-8（d2_unresolved_ratio）→
  `d_component_unresolved`：低于数值分辨能力，物理抑制与数值零
  无法区分
- 1e-8 ≤ 比值 < 1e-6（d2_suppression_ratio）→
  `d_component_suppressed`：物理可分辨的通道抑制
- 比值 ≥ 1e-6 → 无抑制

**判据性质**：双阈值均为相对幅值工程判据（|C22|/|C11| 比值口径），
**不是独立测得的 C22 数值误差**；`1e-8` 为**经验相对幅值分辨阈值**
（参考双精度浮点量级选取、含 bin 平均与代数运算放大的经验安全边际），
**不是 IEEE 双精度机器精度本身、不是独立测得的数值误差、也不是
普适物理阈值**；`1e-6` 的语义为"未满足当前分析定义的 d2 相对弱化
判据"（即比值高于 1e-6 时，当前分析不认定 d2 相对弱化），不构成
对 d2 通道物理属性的进一步断言。两者均未获物理授权。

### P2-1 圆周均值有效性（resultant length）

圆周统计输出 (mean, std, R, n)；R < circ_mean_min_R（默认 0.5，
可配置）时均值显示 n/a 并记录无效原因——弥散分布（如 {0°,180°}
对极混合）不再输出伪 90°。

### P2-2 分类优先级（审计修正版）

insufficient（数据不足）→ inconclusive（多容差不一致，先于物理
分类）→ d_component_unresolved → d_component_suppressed →
chiral_phase_candidate → no_chiral_signal。

### P2-3 跨 D 档一致性层级

单档 analyzer 的分类不代表跨 D 验证；跨 D 一致性在报告汇总层
评估（12 组分类对照表）。未来若某档出现 chiral_phase_candidate，
须多档一致才可作为 doping 级手性候选，否则标 provisional。

### P2-4 cancellation_min_ratio 接入有效点判定

bin 级 min_cancellation（9 组合中最严重相消）低于阈值且
min_n_raw ≥ 2 时该 bin 无效（n_raw=1 的 bin 无相消信息，不适用）；
无效 bin 在 bins_detail 中可定位（min_cancellation/min_n_raw 列）。

### P2-5 噪声底语义

noise_floor 是逐组合独立估计的经验工程门槛（非零值远距尾部
中位数），不是严格数值收敛误差；不同 doping/D/refbond 的
noise_floor 不可直接比较，禁止据其差异推断物理规律。

### P3-1 噪声样本计数

结果含 noise_counts（total/zero/nonzero）；非零样本过少时噪声底
估计不可靠，报告中如实呈现。

### P3-2 bin 几何近似

同一距离 bin 可能合并"距离相同但相对 refbond 的方向矢量不同"的
target 键（窄圆柱有限几何下 sup_corr_distance 的近似）；相同距离
不等于相同空间方向；三个 refbond 几何中心不重合；bin 容差稳定性
检查只能评估部分数值敏感性，不能证明几何近似严格成立。
