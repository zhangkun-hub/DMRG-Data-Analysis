# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 项目概述

本项目是 t-J 模型在 honeycomb（蜂窝）晶格上 DMRG 计算结果的**数据分析与可视化**工具集。核心代码位于 `DMRG data visualization/`，主要工作是：

1. 解析 UltraDMRG 集群产生的测量 JSON 数据，及集群 DMRG 优化日志（slurm log）
2. 将密度、自旋关联、超导关联、电荷关联、能量等物理量封装为记录对象并作图
3. 对比 UltraDMRG 与 Itensor 两套计算结果的绝对/相对误差
4. 热力学极限外推（能量随 1/N 外推，含异常点剔除）
5. 统一分析契约与入口：`AnalysisReport` 统一结果外壳 + `run_analysis()` 统一分派（衰减分析 / 结构因子 / 峰值分析 / 能量外推）

# 运行方式

无 build/lint 体系。本项目是一个 Python 分析工具包 + Jupyter notebook 工作流，测试为 `tests/` 下可直接运行的脚本套件：

- 依赖：`numpy`、`pandas`、`matplotlib`，装在 Anaconda 虚拟环境 **Calculation** 中（无 requirements.txt）。运行前先 `conda activate Calculation` 再启动 Jupyter 或运行脚本；该环境下 numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7（较新，注意部分旧 API 已移除）
- **测试**：`tests/` 下 13 套脚本（当前共 369 项断言，含 N200/N400 回归基线、合成数据与边界用例）。在 `DMRG data visualization/` 下运行：`python -X utf8 tests/test_*.py`（Calculation 环境，exit 0 = 全过）。回归基线在 `tests/baselines/`（n200 post_refactor / spin v1 / n400 v1），**不得重新生成或覆盖**；`tests/artifacts/` 是运行产物（被 .gitignore 忽略）
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
- **`Spin_Structure_Factor.py`** — `compute_structure_factor` 通用结构因子核（S(q)=(1/N)Σ e^{iq·r} C_ij，is_restricted 受限语义）+ `xc_momentum_grid`（XC 圆柱动量网格）+ `SpinStructureFactorEstimator`（自旋受限估计入口）+ 几何元信息。
- **`Spin_Structure_Factor_Analyzer.py`** — `SpinStructureFactorPeakAnalyzer`：数值峰值分析（8 邻域严格局部极大、抛物面细化、FWHM、等距网格门控），只输出数值信息，不判定物理相。
- **`Charge_Structure_Factor.py`** — `ChargeStructureFactorEstimator`：受限电荷结构因子（电子密度连通关联 C_ij = <n_i n_j>−<n_i><n_j>，对角项按占据数幂等补全），含 `estimate_from_files` 与 `estimate_batch`（批量聚合 results/failed，不吞异常）。
- **`Analysis_Contract.py`** — 统一分析结果契约：`AnalysisReport`（status/classification/evidence/data_restricted 语义正交、payload 引用、严格 JSON 序列化）+ 4 个纯函数适配器（from_decay_result / from_structure_factor / from_peak_report / from_energy_extrapolation）+ observable/analyzer 常量。
- **`Analysis_Integration.py`** — 集成层：`analyze_with_report`（SC/Spin/Charge 衰减包装，误差棒与拟合优度独立存放）、`structure_factor_report`、`peak_analysis_report`、`charge_structure_factor_report`、`energy_extrapolation_report`。
- **`Analysis_Entry.py`** — 统一入口：`run_analysis(observable, *, operation=None, config=None, **kwargs)` 按 (observable, operation) 路由到集成层包装函数（7 条链路），未知路由抛 ValueError，不吞底层异常。

Notebook 位于 `DMRG data visualization/` 根目录，是实际使用入口（每个 cell 顶部有"需要修改的部分"参数块）：`Density.ipynb`、`Energy.ipynb`、`Spin_Correlation.ipynb`、`Superconductivity_Correlation.ipynb`（1500 后缀版对比不同 bond dimension）。`Superconductivity_Correlation.ipynb` 在可视化 cell 之后接有 `SuperCorrDecayAnalyzer` 衰减分析 cell（A/C 键方向循环，title=label）。Itensor 数据没有文件，通过 `getpass.getpass()` 粘贴输入。

# Skills

项目级 skills 位于 `.claude/skills/`，处理相应物理量任务时优先调用：

- `dmrg-density-plot` — 密度数据（nf.json）提取与作图、误差对比
- `dmrg-spin-corr-plot` — 自旋关联三分量提取、合成 `<S_i·S_j>` 与作图
- `dmrg-supercorr-plot` — 超导关联 12 文件提取、SU(2) 对称性检验、四通道合成与作图；衰减类型分析与拟合用 `SuperCorrDecayAnalyzer`（`Super_Correlation_Analyzer.py`，见架构一节）或统一入口 `run_analysis`（`Analysis_Entry.py`）

每个 skill 内含环境准备、数据路径约定、完整调用链与该物理量的专属坑。

# 数据目录约定

UltraDMRG 集群计算结果存放在 `DMRG data visualization/Data_N200/` 与 `Data_N400/`（**没有 Data_N96**，旧文档引用已过时），层级：

```
Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/*.json
Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_super_corr/super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json
```

实际有数据的组合（其余目录为空骨架）：
- **Data_N200/t_J_3**：doping 7 种（d1_10、d1_20、d1_25、d1_100、d2_25、d3_20、d3_25）；bond_dim 1500 全部有，5000 仅 d1_10/d1_100/d1_20/d2_25；measure 版本 49/50；超导 refbond 49_50/50_51/50_54
- **Data_N400/t_J_3**：仅 d1_10、bond_dim 1500；measure 版本 97/98；超导 refbond 97_98/98_99/98_102
- 其他 t_J 值（t_J_5/10/15/20）目前只有空目录；注意 J=0.2 对应 t/J=20 是在 notebook 里写死的

- `doping`：空穴率（d1_100 即 δ = 1/100）
- 物理量 JSON 文件：`nf`（单点电子密度）、`sz{ver}sz` / `sp{ver}sm` / `sm{ver}sp`（自旋关联三分量）、`nf{ver}nf`（密度-密度关联，**field 连续**，须用 `range(0, N, 1)` 序列解析而非自旋的 step-4 序列）、`single_particle_corr{1..4}`、源点 0 的旧版 `sz.json`
- 超导关联：bond 方向 A/B/C × 通道 a/b/c/d 共 12 个文件，文件名 `scsAa_fix_refbond_49_50.json` 等；目录名 `super_corr_{refbond1}_{refbond2}` 指明 reference bond
- 每个数据目录下的 `slurm-*.log` 是**测量作业日志**（"Running measurement using existing MPS"），不含 `Bond dimension`/`sweep`/`E0` 行——`DMRGLogParser` 用于解析集群 DMRG 优化日志，与这些文件无关
- 已知数据质量事实：N400 超导数据的 SU(2) 对称性误差实测 0.008~0.144（远超 ~1e-4 经验阈值），合成关联值的物理解释需谨慎；N400 电荷/自旋关联均只有源点 97/98 两组

JSON 格式统一为 `[[索引], [value, 0]]` 的嵌套列表字符串结构（如 `[[[0], [0.9998, 0]], ...]`）。

# 已知坑

- **`DMRG_Log_Analyzer.py` 文件名含零宽字符 U+200B**：正常拼写路径会 FileNotFoundError。用 `glob.glob('DMRG_Log_Analyzer*.py')` 或按原名复制粘贴路径定位。建议未来 rename（需先征得用户同意）。
- **Windows 控制台默认 GBK 编码**：脚本/notebook 内大量中文注释与 `†`、`Δ` 等 Unicode 字符，print 时可能 `UnicodeEncodeError`。在 bash 里跑 Python 处理本项目文件时用 `python -X utf8` 并 `sys.stdout.reconfigure(encoding='utf-8')`。
- **bash 里 `python` 默认指向 Anaconda base 环境**（numpy 1.21.5），不是 Calculation（numpy 2.2.6）。跑测试/脚本必须先 `conda activate Calculation` 或用全路径 `/d/Anaconda/envs/Calculation/python.exe`——跨环境跑会因浮点尾数差异导致 N200 基线逐字段精确比较失败（实测现象，非代码回归）。
- Notebook 中的参数块是示例值，与数据目录实际参数不一定对应（实际有数据的组合见"数据目录约定"），运行前需核对参数块与数据目录是否匹配。
- `Energy_Process.py` 的 `__main__` 演示代码调用了不存在的 `EnergyRecord.from_raw_data`，仅演示用途，实际使用时用 `extract_data`。
- `DMRGLogParser.get_dataframe()` 在日志无匹配行（空解析）时 `sort_values(["BondDimension", "Sweep"])` 会抛 `KeyError`，需先处理空结果或补空表守卫。
- `DensityProcessor.Line_Chart_Visualization` 要求四组水平行数据长度一致，数据不完整会抛 ValueError。
- **电荷关联（nf{ver}nf）的 field 是连续编号**：必须用 `range(0, N, 1)` 解析，用自旋的 step-4 序列会静默丢弃 3/4 数据（仅剩同子格对）。
- **回归基线不可重新生成**：`tests/baselines/` 的 3 个基线（n200 post_refactor / spin v1 / n400 v1）是逐字段回归锚点，N200 基线对跨环境浮点尾数敏感，任何修改分析器输出键集的改动都会触发"键集不一致"失败。
