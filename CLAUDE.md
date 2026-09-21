# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 项目概述

本项目是 t-J 模型在 honeycomb（蜂窝）晶格上 DMRG 计算结果的**数据分析与可视化**工具集。核心代码位于 `DMRG data visualization/`，主要工作是：

1. 解析 UltraDMRG 集群产生的测量 JSON 数据，及集群 DMRG 优化日志（slurm log）
2. 将密度、自旋关联、超导关联、能量等物理量封装为记录对象并作图
3. 对比 UltraDMRG 与 Itensor 两套计算结果的绝对/相对误差
4. 热力学极限外推（能量随 1/N 外推，含异常点剔除）

# 运行方式

无 build/lint/test 体系。本项目是一个 Python 分析工具包 + Jupyter notebook 工作流：

- 依赖：`numpy`、`pandas`、`matplotlib`，装在 Anaconda 虚拟环境 **Calculation** 中（无 requirements.txt）。运行前先 `conda activate Calculation` 再启动 Jupyter 或运行脚本；该环境下 numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7（较新，注意部分旧 API 已移除）
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
- **`Super_Correlation_Analyzer.py`** — `SuperCorrDecayAnalyzer`：超导关联衰减类型分析（新增，独立于既有代码）。输入已合成的 `<Δ†Δ>`（list[SuperCorrRecord]），流程：同距离分组取平均（保留组内原始值）→ effective noise floor（initial/final 两阶段，无循环依赖）→ r_min 全网格扫描（每窗口重跑包络+振荡诊断，推荐值语义为"推荐分析窗口起点"非物理 crossover）→ 包络 log 空间线性拟合 → 去包络 cosine 振荡诊断（8 项 diagnostic checks 证据聚合，无单项硬门槛）→ 5 候选模型同空间同窗口组内 AICc/BIC 比较 → 决策树分类 8 类（constant / power_law / exponential / oscillatory_power_law / oscillatory_exponential / non_monotonic / complex_decay / INCONCLUSIVE）。`DecayAnalysisResult` 含完整 config 快照（analysis_version=1.0）可 `to_dict`/`to_json`；配套 `print_summary` / `Decay_Visualization`（8 面板）/ `analyze_batch`（不自动做跨 refbond 物理结论）。

Notebook 位于 `DMRG data visualization/` 根目录，是实际使用入口（每个 cell 顶部有"需要修改的部分"参数块）：`Density.ipynb`、`Spin_Correlation.ipynb`、`Superconductivity_Correlation.ipynb`（1500 后缀版对比不同 bond dimension）。`Superconductivity_Correlation.ipynb` 在可视化 cell 之后接有 `SuperCorrDecayAnalyzer` 衰减分析 cell；`Superconductivity_Correlation_Analysis.ipynb` 是衰减分析专用入口（N400/N96 真实两路径 + 合成分类测试 + 批量对比）。Itensor 数据没有文件，通过 `getpass.getpass()` 粘贴输入。

# Skills

项目级 skills 位于 `.claude/skills/`，处理相应物理量任务时优先调用：

- `dmrg-density-plot` — 密度数据（nf.json）提取与作图、误差对比
- `dmrg-spin-corr-plot` — 自旋关联三分量提取、合成 `<S_i·S_j>` 与作图
- `dmrg-supercorr-plot` — 超导关联 12 文件提取、SU(2) 对称性检验、四通道合成与作图；衰减类型分析与拟合用 `SuperCorrDecayAnalyzer`（`Super_Correlation_Analyzer.py`，见架构一节）

每个 skill 内含环境准备、数据路径约定、完整调用链与该物理量的专属坑。

# 数据目录约定

`DMRG data visualization/Data_N96/` 存放 UltraDMRG 集群计算结果（当前只有 N=96 一组），层级：

```
Data_N96/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/*.json
Data_N96/{t_J}/{doping}/{bond_dim}_measure_super_corr/super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json
```

- `t_J`：t/J 比值（t_J_3、t_J_5、t_J_10 等，注意 J=0.2 对应 t/J=20 是在 notebook 里写死的）
- `doping`：空穴率（d0、d1_12、d1_16、d1_24、d1_48 = δ = 1/48）
- `bond_dim`：bond dimension（如 1500）；有 `measure_*` 与 `1500_measure_*` 两种前缀
- 物理量 JSON 文件：`nf`（密度）、`sz{ver}sz` / `sp{ver}sm` / `sm{ver}sp`（自旋关联三分量）、`nf{ver}nf`、`single_particle_corr*`
- 超导关联：bond 方向 A/B/C × 通道 a/b/c/d 共 12 个文件，文件名 `scsAa_fix_refbond_25_26.json` 等；目录名 `super_corr_{refbond1}_{refbond2}` 指明 reference bond
- 每个数据目录下的 `slurm-*.log` 是**测量作业日志**（"Running measurement using existing MPS"），不含 `Bond dimension`/`sweep`/`E0` 行——`DMRGLogParser` 用于解析集群 DMRG 优化日志，与这些文件无关

JSON 格式统一为 `[[索引], [value, 0]]` 的嵌套列表字符串结构（如 `[[[0], [0.9998, 0]], ...]`）。

# 已知坑

- **`DMRG_Log_Analyzer.py` 文件名含零宽字符 U+200B**：正常拼写路径会 FileNotFoundError。用 `glob.glob('DMRG_Log_Analyzer*.py')` 或按原名复制粘贴路径定位。建议未来 rename（需先征得用户同意）。
- **Windows 控制台默认 GBK 编码**：脚本/notebook 内大量中文注释与 `†`、`Δ` 等 Unicode 字符，print 时可能 `UnicodeEncodeError`。在 bash 里跑 Python 处理本项目文件时用 `python -X utf8` 并 `sys.stdout.reconfigure(encoding='utf-8')`。
- Notebook 中的 N=200、t_J_3、d1_10 等参数是示例值，与 Data_N96 的实际数据不一定对应，运行前需核对参数块与数据目录是否匹配。
- `Energy_Process.py` 的 `__main__` 演示代码调用了不存在的 `EnergyRecord.from_raw_data`，仅演示用途，实际使用时用 `extract_data`。
- `DMRGLogParser.get_dataframe()` 在日志无匹配行（空解析）时 `sort_values(["BondDimension", "Sweep"])` 会抛 `KeyError`，需先处理空结果或补空表守卫。
- `DensityProcessor.Line_Chart_Visualization` 要求四组水平行数据长度一致，数据不完整会抛 ValueError。
