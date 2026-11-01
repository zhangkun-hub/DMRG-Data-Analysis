# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 项目概述

本项目是 t-J 模型在 honeycomb（蜂窝）晶格上 DMRG 计算结果的**数据分析与可视化**工具集。核心代码位于 `DMRG data visualization/`，主要工作是：

1. 解析 UltraDMRG 集群产生的测量 JSON 数据，及集群 DMRG 优化日志（slurm log）
2. 将密度、自旋关联、超导关联、电荷关联、能量等物理量封装为记录对象并作图
3. 对比 UltraDMRG 与 Itensor 两套计算结果的绝对/相对误差
4. 热力学极限外推（能量随 1/N 外推，含异常点剔除）
5. 统一分析契约与入口：`AnalysisReport` 统一结果外壳 + `run_analysis()` 统一分派（衰减分析 / 结构因子 / 峰值分析 / 能量外推）及 PhaseDiagnosis 规则引擎（候选级 verdict，能力边界与观测规则）
6. 跨 doping 受控数值汇总（P4.3）：`Cross_Doping_Summary.py` 逐 doping 运行统一入口，按可比性契约分组输出 CSV/JSON（见"跨 doping 汇总"一节）

# 权限约定

- 本项目目录 `C:\Users\ZhangKun\Desktop\DMRG-data-processing` 内的各种操作**无需请示用户**（读文件、grep、运行测试/脚本等被权限系统提示的操作——提示时直接 Yes）
- 仓库外的测量源码文件夹 `C:\Users\ZhangKun\Desktop\DMRGtJModel` 与 `C:\Users\ZhangKun\Desktop\UltraDMRG`：只读访问/查看无需请示；增、删、改等改变程序内容的操作才需先请求（见"数据目录约定"）

# 运行方式

无 build/lint 体系。本项目是一个 Python 分析工具包 + Jupyter notebook 工作流，测试为 `tests/` 下可直接运行的脚本套件：

- 依赖：`numpy`、`pandas`、`matplotlib`，装在 Anaconda 虚拟环境 **Calculation** 中（无 requirements.txt）。运行前先 `conda activate Calculation` 再启动 Jupyter 或运行脚本；该环境下 numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7（较新，注意部分旧 API 已移除）
- **测试**：`tests/` 下 35 套脚本（含 N200/N400 回归基线、合成数据、边界用例与规则引擎专项测试）。在 `DMRG data visualization/` 下运行：`python -X utf8 tests/test_*.py`（Calculation 环境，exit 0 = 全过）。回归基线在 `tests/baselines/`（n200 post_refactor / spin v1 / n400 v1），**不得重新生成或覆盖**；`tests/artifacts/` 是运行产物（被 .gitignore 忽略）
- notebook 内用 `from Physical_Quantity.xxx import ...` 的相对导入，**必须在 `DMRG data visualization/` 目录下启动 Jupyter**，否则导入失败
- 直接运行脚本：`cd "DMRG data visualization" && python "DMRG_Log_Analyzer.py"`（文件名含零宽字符，见下方"已知坑"）
- `Energy_Process.py` 等模块自带 `if __name__ == "__main__"` 的模拟数据演示，可直接 `python Physical_Quantity/Energy_Process.py` 运行验证

# 架构

`DMRG data visualization/Physical_Quantity/` 是核心包，采用「Record 数据类 + Processor 处理类」模式：

- **`Physical_Data_Record.py`** — 数据记录层。抽象基类 `Record`；子类 `DensityRecord`（格点密度）、`CorrelationRecord`（两点关联函数）、`SuperCorrRecord`（四点超导关联，含 reference bond / target bond）。各子类的 `extract_data()` 类方法用 `ast.literal_eval` 解析 DMRG 输出的字符串数据（格式 `[[i, j], [value, 0]]`）并封装为对象列表，封装时用 `HoneycombXCGeometry` 计算格点间距离。
- **`Honeycomb_Lattice.py`** — 晶格几何。抽象基类 `HoneycombLattice`；`HoneycombXCGeometry` 定义 XC 构型：格点数 N 必须是 8 的倍数（基元含 8 格点），提供格点坐标 `coords`、欧氏距离 `get_distance`、超导关联用的 bond 中点距离 `sup_corr_distance`、最近邻索引表 `_nearest_neighbor_grid_index`。
- **`Density_Process.py`** — `DensityProcessor`：密度一维折线图（按 `horizontal_starting_point = i % 4` 分成 4 条水平行曲线）及两组数据绝对误差对比图。
- **`Correlation_Process.py`** — `CorrelationProcessor`：由 `SzSz + 1/2(SuSd + SdSu)` 合成 `<S_i·S_j>`；作图固定 1×3 子图（线性 / y-log / log-log）分别用于判断指数衰减与幂律衰减。
- **`Super_Correlation_Process.py`** — `SuperCorrProcessor`：超导关联四通道 a/b/c/d 的 SU(2) 对称性检验（a↔d、b↔c 相对误差，经验阈值 ~1e-4）、按 `<Δ†Δ> = 0.5(a - b - c + d)` 合成、A/B/C 三个键方向可视化及绝对误差对比。
- **`Energy_Process.py`** — `EnergyRecord`（N、E_min、E_avg、单格点能量 Es、Es/J）与 `EnergyProcessor`：`extrapolate()` 按 `Es(N) = E∞ + a/N^power` 做热力学极限外推（先线性拟合，残差超 threshold×std 判为异常点剔除后重新拟合）；`fit_doping_dependence()` 拟合 E∞ 随空穴率 δ 的线性关系。
- **`DMRG_Log_Parser.py`** — `DMRGLogParser`：解析 DMRG 优化日志中的 Bond dimension、sweep、E0（集群上的优化日志，不是本仓库 Data 目录下的测量日志），按 (D, sweep) 分组自动求平均，输出 DataFrame（BondDimension / Sweep / E0_mean / E0_std / N_samples），并绘制能量收敛图与 |ΔE| 半对数收敛图。主脚本 `DMRG_Log_Analyzer.py` 通过 `sys.stdin` 读入日志文本并用自定义正则（限定 `Site (40, 41)`）解析。
- **`Correlation_Decay_Analyzer.py`** — 通用关联衰减分析核心（叶子模块，不依赖本包其他代码）。`CorrelationDecayAnalyzer`：噪声底 → hump 检测 → r_min 全网格扫描（verdict 4 级：stable/moderate/unstable/insufficient）→ 包络 log 空间拟合（含峰值法/全点模式）→ 族分解 → 振荡诊断（8 项证据聚合）→ 9 候选模型 AICc/BIC 比较 → 决策树分类 8 类（constant / power_law / exponential / oscillatory_power_law / oscillatory_exponential / non_monotonic / complex_decay / INCONCLUSIVE）。`DecayAnalysisResult` 含 config 快照与 analysis_version，可 to_dict/to_json；`analyze_generic` 是各适配层共用的管线主体。
- **`Super_Correlation_Analyzer.py`** — `SuperCorrDecayAnalyzer(CorrelationDecayAnalyzer)`：超导专用薄适配层（输入 list[SuperCorrRecord]、bond 中点两族、超导文案）。算法全部继承通用核心。
- **`Spin_Correlation_Analyzer.py`** — `SpinCorrDecayAnalyzer(CorrelationDecayAnalyzer)`：自旋专用薄适配层（输入 list[CorrelationRecord]、格点子格族、`analyze(recs, config, N, a)` 签名）。
- **`Charge_Correlation_Analyzer.py`** — `ChargeCorrDecayAnalyzer(CorrelationDecayAnalyzer)`：电荷关联薄适配层（复用 Spin 适配层的分组/预处理与子格几何规则，仅文案不同；输入密度-密度关联记录）。
- **`Green_Function_Analyzer.py`** — `GreenFunctionDecayAnalyzer(CorrelationDecayAnalyzer)`：单粒子格林函数衰减分析薄适配层（输入单粒子关联记录，`single_particle_corr{1..4}` 数据），经统一入口（observable `single_particle_correlation`，operation `decay`）路由。
- **`Spin_Structure_Factor.py`** — `compute_structure_factor` 通用结构因子核（S(q)=(1/N)Σ e^{iq·r} C_ij，is_restricted 受限语义）+ `xc_momentum_grid`（XC 圆柱动量网格）+ `SpinStructureFactorEstimator`（自旋受限估计入口）+ 几何元信息。
- **`Spin_Structure_Factor_Analyzer.py`** — `SpinStructureFactorPeakAnalyzer`：数值峰值分析（8 邻域严格局部极大、抛物面细化、FWHM、等距网格门控），只输出数值信息，不判定物理相。
- **`Charge_Structure_Factor.py`** — `ChargeStructureFactorEstimator`：受限电荷结构因子（电子密度连通关联 C_ij = <n_i n_j>−<n_i><n_j>，对角项按占据数幂等补全），含 `estimate_from_files` 与 `estimate_batch`（批量聚合 results/failed，不吞异常）。
- **`SC_Pair_Structure_Factor.py`** — `SCPairStructureFactorEstimator`：超导配对结构因子受限估计（固定参考键 × 单方向目标键集合，bond 中点坐标表示，P(q) = (1/N)Σ exp[iq·(R_ref−R_t)]<Δ†_ref Δ_t>）。bond → 唯一整数 ID（i*N+j）编码复用 `compute_structure_factor` 公共核（零修改）。入口：`estimate_from_files`（全量解析方向文件）/ `estimate_from_records`（合成 <Δ†Δ> 输入）/ `estimate_channel_from_records`（单通道）/ `estimate_from_channels`（四通道，先做集合级 + 逐元素一致性校验，不一致抛 ValueError）。受限语义：is_restricted 恒 True、expected_pair_count/pair_coverage/source_coverage 为 None、record_source='full_direction_file'（全量解析，区别于衰减分析 extract_data 的过滤口径）。已接入统一入口（observable `sc_pair_structure_factor`，operation `estimate`，经 `sc_pair_structure_factor_report` 包装）与跨 doping 汇总（每 refbond × 键方向一行，身份标记 refbond_dir）；P2 可比性审计对其实施 q_grid 摘要逐对检查。测试 `tests/test_sc_pair_structure_factor.py`（36 项，含真实 N200 d1_100/49_50 回归）+ `tests/test_sc_pair_contract_integration.py`（20 项路由/契约集成）。
- **`Pairing_Symmetry_Analyzer.py`**（D5）— `PairingSymmetryAnalyzer`：SC 配对对称性分析（阶段 1 数据装配 / 2 可比性质量门 / 3 公共窗口振幅重拟合 + rank-1 矩阵模型比较 / 4 有符号证据通道 + 三通道状态机 / 5B 候选观测报告，模板 s 波、d_real 六种定向、magnitude_degenerate），经统一入口（observable `sc_pairing_symmetry`，operation `analyze`）路由。**明确不实现**：记录级 bootstrap 与联合协方差——候选支持判定恒 `not_evaluated`（禁止点估计冒充统计支持）。设计文档 `docs/PairingSymmetryAnalyzer_Design.md`。
- **`Analysis_Contract.py`** — 统一分析结果契约：`AnalysisReport`（status/classification/evidence/data_restricted 语义正交、payload 引用、严格 JSON 序列化）+ 4 个纯函数适配器（from_decay_result / from_structure_factor / from_peak_report / from_energy_extrapolation）+ observable/analyzer 常量。
- **`Analysis_Integration.py`** — 集成层：`analyze_with_report`（SC/Spin/Charge 衰减包装，误差棒与拟合优度独立存放）、`structure_factor_report`、`peak_analysis_report`、`charge_structure_factor_report`、`energy_extrapolation_report`。
- **`Phase_Diagnosis.py`** — 多观测量证据汇总契约（P1/P2/P3）：`collect_evidence` 将多份 AnalysisReport 汇总为 `PhaseDiagnosisReport`（结论仅 `evidence_collected`，无物理判据）；`audit_comparability` 结构因子可比性审计（q_grid 摘要逐对覆盖自旋/电荷/SC 配对三个 SF 估计器；source_set 仅电荷；四态 consistent/inconsistent/unknown/not_applicable）；`evidence_sufficiency_check` 证据充分性检查（保守语义，rules=None → overall 恒 unknown）。另含规则引擎 `run_candidate_rules`（候选级九态 verdict，规则须注册并授权才执行）与注册守卫（P3-1 显式替换 / P3-2 幽灵 ID 拒绝 / 候选级冲突检查）。P1-G.28 新增两条**有限尺寸分析能力边界规则**（能力诊断对象，**不是物理相候选**）：`RULE_GAP_V1`（candidate=`gap_scaling`）与 `RULE_WIDTH_V1`（candidate=`width_scaling`），经独立 `GAPW_RULES` 元组独立注册（**不属于 `P1F3_RULES`**），verdict 恒 `not_measurable`——Gap：Data_Energy 仅有单扇区基态日志（无激发态能量/粒子数扇区能量差，禁止从基态总能随 doping 或 N 的变化构造激发态能隙）；Width：XC 构型 y 方向恒 2 胞宽（N 变化是长度方向，不构成多宽度扫描）。`not_measurable` 只表示证据不足，不表示能隙/宽度依赖为零、物理行为不存在或任何物理相被排除。**仅注册、未接入生产执行链路**：`run_analysis`/`Cross_Doping_Summary` 不执行 `GAPW_RULES`，生产报告不产生 Gap/Width verdict；GapScaling/WidthScaling 算法**仍未实现**。专项测试 `tests/test_phase_diagnosis_rules_gap_width.py`（34 项）。
- **`Analysis_Entry.py`** — 统一入口：`run_analysis(observable, *, operation=None, config=None, **kwargs)` 按 (observable, operation) 路由到集成层包装函数（13 条路由：衰减 ×4 / SF 估计 ×3 / 峰值 ×2 / 能量外推 / 配对对称 / 密度 profile / 峰位对齐），未知路由抛 ValueError，不吞底层异常。
- **`Cross_Doping_Summary.py`**（根目录独立脚本，P4.3）— 跨 doping 受控数值汇总：`analyze_doping()` 逐 doping 经 run_analysis 运行 SC 衰减/SC 配对 SF/自旋 SF+峰值/电荷 SF（`only_observables=None` 可选筛选，筛选发生在分析调用之前，未知 observable 抛 ValueError）；`build_rows()` 固定列序行构建（含 source_identity_kind / peak_estimate_entry_index）；`assign_comparability()` 可比性四档分组（硬性限制优先，P2 审计只读引用）；`summarize()` 串联 P1/P2/P3；`write_outputs()` CSV+JSON 覆盖写。细节见"跨 doping 汇总"一节。
- **`SC_Exponent_Summary.py`**（根目录独立脚本，D1）— SC 指数跨数据汇总：经 `analyze_doping(only_observables={superconducting_correlation})` 复用 refbond 发现/失败机制跑 SC 衰减，**不重新拟合**——指数直接取自 `report.payload`（envelope_fit 的 K/xi ± 误差、StabilityReport 的 rel_mad_K/model_consistency/spearman_rho、候选幂律族模型的 K+AICc 证据）。指数状态四分类（exponent_ok/unstable/data_insufficient/input_missing，moderate≠失败）；K 与 xi 严格分列分组建；分组按身份全等（N/bond_dim/refbond/方向/指数类型），组间不合并趋势；comparability_bucket 复用 Cross_Doping_Summary 机械规则。输出 CSV+严格 JSON，不判定物理相态/d+id。注意其默认输出目录未被 .gitignore 覆盖，正式运行建议 `--outdir tests/artifacts/<name>`。

Notebook 位于 `DMRG data visualization/` 根目录，是实际使用入口（每个 cell 顶部有"需要修改的部分"参数块）：`Density.ipynb`、`Energy.ipynb`、`Spin_Correlation.ipynb`、`Superconductivity_Correlation.ipynb`（1500 后缀版对比不同 bond dimension）。`Superconductivity_Correlation.ipynb` 在可视化 cell 之后接有 `SuperCorrDecayAnalyzer` 衰减分析 cell（A/C 键方向循环，title=label）。Itensor 数据没有文件，通过 `getpass.getpass()` 粘贴输入。

# Skills

项目级 skills 位于 `.claude/skills/`，处理相应物理量任务时优先调用：

- `dmrg-density-plot` — 密度数据（nf.json）提取与作图、误差对比
- `dmrg-spin-corr-plot` — 自旋关联三分量提取、合成 `<S_i·S_j>` 与作图
- `dmrg-supercorr-plot` — 超导关联 12 文件提取、SU(2) 对称性检验、四通道合成与作图；衰减类型分析与拟合用 `SuperCorrDecayAnalyzer`（`Super_Correlation_Analyzer.py`，见架构一节）或统一入口 `run_analysis`（`Analysis_Entry.py`）

每个 skill 内含环境准备、数据路径约定、完整调用链与该物理量的专属坑。

# 数据目录约定

UltraDMRG 集群计算结果存放在 `DMRG data visualization/Data_N200/`、`Data_N400/` 与 `Data_Energy/`（**没有 Data_N96**，旧文档引用已过时），层级：

```
Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/*.json
Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_super_corr/super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json
```

实际有数据的组合（其余目录为空骨架）：
- **Data_N200/t_J_3**：doping 4 种（d1_10、d1_20、d1_100、d2_25）；**d1_25、d3_20、d3_25 已因数据问题删除**，不属于当前可用数据，不得假定存在或恢复；bond_dim 1500 与 5000 四组全有；measure 版本 49/50；超导 refbond 49_50/50_51/50_54
- **Data_N400/t_J_3**：doping 5 种（d1_10、d1_20、d1_40、d1_80、d2_25）；bond_dim 1500 与 5000 全有；measure 版本 97/98；超导 refbond 97_98/98_99/98_102；每个 doping 目录内另有 2 份 DMRG 优化日志（低 D 500/1000/1500 + 高 D 2000→5000，均正常结束，N=400）
- 其他 t_J 值（t_J_5/10/15/20）目前只有空目录；注意 J=0.2 对应 t/J=20 是在 notebook 里写死的

- `doping`：空穴率（d1_100 即 δ = 1/100）
- 物理量 JSON 文件：`nf`（单点电子密度）、`sz{ver}sz` / `sp{ver}sm` / `sm{ver}sp`（自旋关联三分量）、`nf{ver}nf`（密度-密度关联，**field 连续**，须用 `range(0, N, 1)` 序列解析而非自旋的 step-4 序列）、`single_particle_corr{1..4}`、源点 0 的旧版 `sz.json`
- 超导关联：bond 方向 A/B/C × 通道 a/b/c/d 共 12 个文件，文件名 `scsAa_fix_refbond_49_50.json` 等；目录名 `super_corr_{refbond1}_{refbond2}` 指明 reference bond
- 每个数据目录下的 `slurm-*.log` 是**测量作业日志**（"Running measurement using existing MPS"），不含 `Bond dimension`/`sweep`/`E0` 行——`DMRGLogParser` 用于解析集群 DMRG 优化日志，与这些文件无关
- **DMRG 优化日志（能量数据）**：N200 的 4 个 doping 目录内各 2 份 `slurm-*-Zk_tJ.log`（共 8 份，1500 与 2000→5000 两档调度，均正常结束）；N400 的 5 个 doping 目录内同样各 2 份（低 D 500/1000/1500 + 高 D 2000→5000，均正常结束）
- **Data_Energy（热力学外推数据）**：层级 `Data_Energy/t_J_3/{doping}/N{n}/slurm-*.log`；6 个 doping（d0、d1_10、d1_20、d1_40、d1_80、d3_40）× 5 个 N（80/160/240/320/400）共 30 份优化日志（D 调度 500,1000,1500，全部正常结束，格式与 N200/N400 日志同构，可用 `extract_last_e0` 解析）——每 doping 5 个 N，**满足热力学极限外推条件（≥3 个 N）**；目录内仅有 *.log 文件，被全局 git.ignore 的 `*.log` 规则覆盖（Git 不可见，不入版本控制）
- 已知数据质量事实：N400 超导数据的 SU(2) 对称性误差实测 0.008~0.144（远超 ~1e-4 经验阈值），合成关联值的物理解释需谨慎；N400 电荷/自旋关联均只有源点 97/98 两组

JSON 格式统一为 `[[索引], [value, 0]]` 的嵌套列表字符串结构（如 `[[[0], [0.9998, 0]], ...]`）。

## 测量源码位置（仓库外，只读参照）

- `Data_Energy`、`Data_N200`、`Data_N400` 三个数据目录的计算源码均来自 **`C:\Users\ZhangKun\Desktop\DMRGtJModel`**（桌面兄弟目录，不在本仓库内）：测量程序如 `src/measure1.cpp`（自旋/密度/单粒子测量）、`src/measureSC.cpp` 与 `src/measureSC_fix_ref_extension.cpp`（超导关联）、`src/dmrg.cpp`、`src/honeycomb_lattice.h` 等
- 该程序的库函数引用来自 **`C:\Users\ZhangKun\Desktop\UltraDMRG`**（如 `MeasureTwoSiteOpGroup`/`DumpMeasuRes` 在 `include/qlmps/...` 下）；另有一份更完整的副本在 `D:\Microsoft VS Code project\DMRG\`（含 `qlmps/model_relevant/operators/fermion/tJ_operators.h` 算符定义——桌面版 UltraDMRG 可能缺部分文件，D 盘版更全）
- **权限约定**：访问、阅读、查看上述两个文件夹**无需请求用户**；只有增、删、改等改变程序内容的操作才需要先请求
- 测量语义已核实（B1.1）：t-J 算子 `bupc/bupa/bdnc/bdna` = c↑†/c↑/c↓†/c↓，弦因子 `f`=(-1)^N 仅插中间点，单粒子测量为 j>源点 的单项采样；`CMakeLists.txt` 有 `-DUSE_REAL=1`（实值版），数据文件 `[re, im]` 格式对应复值序列化路径（数据虚部恒 0 与实值基态一致）

# 跨 doping 汇总（P4.3）

独立脚本 `Cross_Doping_Summary.py`（测试 `tests/test_cross_doping_summary.py`，68 项断言）：

- **输入范围**：默认 `Data_N200/t_J_3` × bond_dim=1500、N=200；**当前实际有数据 4 个 doping**（d1_10/d1_20/d1_100/d2_25；d1_25、d3_20、d3_25 已因数据问题删除）——正式运行用 `--dopings d1_10,d1_20,d1_100,d2_25`（脚本 `DEFAULT_DOPINGS` 保留 7 组默认值，对缺失 doping 会显式记录失败行而非静默跳过）。每 doping 逐条运行 23 项分析——SC 衰减（参考键目录发现式枚举 × A/B/C 键）、SC 配对结构因子（同 refbond × A/B/C 受限估计，含逐通道 channel_sf）、自旋 SF（sz_sz 源点 0 + spin_dot 源点 49/50，各带峰值分析）、电荷 SF（nf + nf49nf + nf50nf 连通形式）。N=400 或 bond_dim=5000 须单独运行（`--N 400` / `--bond-dim 5000`），绝不与 N=200 混组
- **运行方式**：`cd "DMRG data visualization" && python Cross_Doping_Summary.py --outdir cross_doping_summary_output`（Calculation 环境）；CLI 含 `--root/--t-j/--dopings/--N/--bond-dim/--refbonds/--n-qx/--n-qy/--outdir`
- **输出文件**：`cross_doping_summary.csv`（一行一条分析含失败行，固定列序）+ `cross_doping_summary.json`（config/rows/comparability_groups/phase_report/semantics）。输出目录已被 .gitignore 忽略——属可重复生成的派生数据（固定随机种子 42、整体覆盖写不追加、严格 JSON），不入版本控制
- **可比性契约**（四档，机械条件检查，无物理判据；硬性限制优先于宽松分支）：
  - `direct`：身份（N/bond_dim/参考键或源点/键方向或定义）全等且附加检查通过——SF 估计行 P2 q_grid 逐对 consistent（仅摘要级）+ 电荷 P2 source_set 集合相等；峰值行行内 q 摘要一致且对应 estimate 的 P2 审计一致；仅表示分析条件一致，不表示 DMRG 收敛精度相同
  - `cautious`：已知差异（SC 参考键 / 键方向或定义不同），只能谨慎比较
  - `not_direct`：硬性限制（N/bond_dim 不一致、源点身份不同或未知、P2 状态 inconsistent/unknown、峰值 q 摘要不一致或缺失、分析失败）；**不同源点集合不得默认可比**
  - `sole`：无同类条目（无并列比较对象），不是比较结论
  - `source_identity_kind` 三值：`refbond_dir`（SC，目录名）/ `payload_field`（电荷，payload 的 diagonal_source_sites——仅对角补全源点，不等同于完整测量源点集合）/ `script_convention`（自旋，脚本输入约定，未经报告字段验证，不同源点身份不判 direct）
  - `peak_estimate_entry_index`：峰值行唯一关联的前置 estimate 条目索引（要求 doping/N/bond_dim/observable/源点/定义全等；零个或多个候选 → 留空，q 摘要 unknown，不用全局 q_list）
  - q_grid 仅摘要级（n_qx/n_qy/端点），完整坐标未验证
  - 复用 PhaseDiagnosis P1/P2/P3：conclusion 仅 `evidence_collected`（非物理结论）；sufficiency 以 rules=None 调用 → overall 恒为 `unknown`（未提供充分性标准，不编造置信度）
- **局限**：不判定物理趋势/单调性/相态/趋势显著性，数值随 doping 变化仅作观测呈现；SC 衰减分类（non_monotonic 等）是数值拟合分类，不作物理解读；复现性只抽检过 d1_10 单 doping（其余依赖固定种子设计）

# 已知坑

- **`DMRG_Log_Analyzer.py` 文件名含零宽字符 U+200B**：正常拼写路径会 FileNotFoundError。用 `glob.glob('DMRG_Log_Analyzer*.py')` 或按原名复制粘贴路径定位。建议未来 rename（需先征得用户同意）。
- **Windows 控制台默认 GBK 编码**：脚本/notebook 内大量中文注释与 `†`、`Δ` 等 Unicode 字符，print 时可能 `UnicodeEncodeError`。在 bash 里跑 Python 处理本项目文件时用 `python -X utf8` 并 `sys.stdout.reconfigure(encoding='utf-8')`。
- **bash 里 `python` 默认指向 Anaconda base 环境**（numpy 1.21.5），不是 Calculation（numpy 2.2.6）。跑测试/脚本必须先 `conda activate Calculation` 或用全路径 `/d/Anaconda/envs/Calculation/python.exe`——跨环境跑会因浮点尾数差异导致 N200 基线逐字段精确比较失败（实测现象，非代码回归）。
- Notebook 中的参数块是示例值，与数据目录实际参数不一定对应（实际有数据的组合见"数据目录约定"），运行前需核对参数块与数据目录是否匹配。
- `Energy_Process.py` 的 `__main__` 演示代码调用了不存在的 `EnergyRecord.from_raw_data`，仅演示用途，实际使用时用 `extract_data`。
- `DMRGLogParser.get_dataframe()` 在日志无匹配行（空解析）时 `sort_values(["BondDimension", "Sweep"])` 曾抛 `KeyError`（历史记录）——**已修复（P1-G.27）**：空解析路径返回带契约 5 列（BondDimension/Sweep/E0_mean/E0_std/N_samples）与对应 dtype 的空表，不再因排序列缺失抛异常；回归测试 `tests/test_dmrg_log_analyzer.py` F 组覆盖（空字符串/无 E0 行/格式错误日志/正常日志排序与数值不变）。验收时点结果：专项 45/45、全量 34 套通过；P1-G.28-B 审计时点全量 35 套通过（时点结果，非永久保证）。此修复**不表示**所有异常格式输入均已彻底处理——格式错误日志按既有设计静默不匹配，行为未变。
- `DensityProcessor.Line_Chart_Visualization` 要求四组水平行数据长度一致，数据不完整会抛 ValueError。
- **电荷关联（nf{ver}nf）的 field 是连续编号**：必须用 `range(0, N, 1)` 解析，用自旋的 step-4 序列会静默丢弃 3/4 数据（仅剩同子格对）。
- **回归基线不可重新生成**：`tests/baselines/` 的 3 个基线（n200 post_refactor / spin v1 / n400 v1）是逐字段回归锚点，N200 基线对跨环境浮点尾数敏感，任何修改分析器输出键集的改动都会触发"键集不一致"失败。
