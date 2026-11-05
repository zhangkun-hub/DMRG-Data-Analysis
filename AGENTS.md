# AGENTS.md

# 用户背景

项目作者是凝聚态计算物理专业的硕士研究生，本项目直接服务于毕业论文。与用户讨论问题、给出方案、编写代码注释和文档时：

- 保持硕士课程和教材能覆盖到的知识水平，用直白的物理语言解释
- 避免不必要的过度抽象或软件工程术语堆砌
- 方案以"能用、好懂、论文里说得清楚"为优先，不追求过度工程化

# 项目概述

本项目是 t-J 模型在 honeycomb（蜂窝）晶格上 DMRG 计算结果的**数据分析与可视化**工具集。核心代码位于 `DMRG data visualization/`，主要工作是：

1. 解析 UltraDMRG 集群产生的测量 JSON 数据，及集群 DMRG 优化日志（slurm log）
2. 将密度、自旋关联、超导关联、电荷关联、能量等物理量封装为记录对象并作图
3. 对比 UltraDMRG 与 Itensor 两套计算结果的绝对/相对误差
4. 热力学极限外推（能量随 1/N 外推，含异常点剔除）
5. 统一分析契约与入口：`AnalysisReport` 统一结果外壳 + `run_analysis()` 统一分派（衰减分析 / 结构因子 / 峰值分析 / 能量外推）及 PhaseDiagnosis 规则引擎（候选级 verdict，能力边界与观测规则）
6. 跨 doping 受控数值汇总（P4.3）：`Cross_Doping_Summary.py` 逐 doping 运行统一入口，按可比性契约分组输出 CSV/JSON（见"跨 doping 汇总"一节）

# 权限约定

- 本项目目录 `C:\Users\ZhangKun\Desktop\DMRG-data-processing` 内的各种操作（读文件、grep、运行测试/脚本等）可直接执行，无需额外请示用户
- 仓库外的测量源码文件夹 `C:\Users\ZhangKun\Desktop\DMRGtJModel` 与 `C:\Users\ZhangKun\Desktop\UltraDMRG`：只读访问/查看可直接执行；增、删、改等改变程序内容的操作须先请求用户确认（见"数据目录约定"）

# 运行方式

无 build/lint 体系。本项目是一个 Python 分析工具包 + Jupyter notebook 工作流，测试为 `tests/` 下可直接运行的脚本套件：

- 依赖：`numpy`、`pandas`、`matplotlib`，装在 Anaconda 虚拟环境 **Calculation** 中（无 requirements.txt）。运行前先 `conda activate Calculation` 再启动 Jupyter 或运行脚本；该环境下 numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7（较新，注意部分旧 API 已移除）
- **测试**：`tests/` 下 35 套脚本（含 N200/N400 回归基线、合成数据、边界用例与规则引擎专项测试）。在 `DMRG data visualization/` 下运行：`python -X utf8 tests/test_*.py`（Calculation 环境，exit 0 = 全过）。回归基线在 `tests/baselines/`（n200 post_refactor / spin v1 / n400 v1），**不得重新生成或覆盖**；`tests/artifacts/` 是运行产物（被 .gitignore 忽略）
- notebook 内用 `from Physical_Quantity.子包名.xxx import ...` 导入，**必须在 `DMRG data visualization/` 目录下启动 Jupyter**，否则导入失败
- 直接运行脚本：`cd "DMRG data visualization" && python "DMRG_Log_Analyzer.py"`（文件名含零宽字符，见下方"已知坑"）
- `physical_analysis/Energy_Process.py` 等模块自带 `if __name__ == "__main__"` 的模拟数据演示，可直接 `python Physical_Quantity/physical_analysis/Energy_Process.py` 运行验证

# 架构

`DMRG data visualization/Physical_Quantity/` 是核心包，按四层架构分子包组织：`data_records/`（数据与记录层）、`physical_analysis/`（物理量分析层）、`evidence_integration/`（证据整合层）、`phase_diagnosis/`（相态诊断层）。导入一律使用子包路径 `from Physical_Quantity.子包名.模块名 import ...`（`__init__.py` 为空、无兼容层）；包内模块间用相对导入（同子包 `from .X`，跨子包 `from ..X`），跨层上溯只允许函数内延迟 import。采用「Record 数据类 + Processor 处理类」模式。

各模块的详细说明见 `DMRG data visualization/AGENTS.md`（子目录级，含完整调用链与物理量处理工作流）。

关键模块概览：

- **`data_records/Physical_Data_Record.py`** — `Record` 抽象基类；子类 `DensityRecord`、`CorrelationRecord`、`SuperCorrRecord`。`extract_data()` 类方法用 `ast.literal_eval` 解析 JSON 数据并封装为对象列表
- **`data_records/Honeycomb_Lattice.py`** — `HoneycombXCGeometry`：XC 构型晶格几何（N 必须是 8 的倍数），提供格点坐标、距离计算、bond 中点距离
- **`physical_analysis/`** — 各物理量处理器（Density/Correlation/SuperCorrelation/Energy Processor）及通用衰减分析核心 `CorrelationDecayAnalyzer`（9 候选模型 AICc/BIC 比较，8 类分类）+ 四个薄适配层（Spin/Super/Charge/GreenFunction）+ 结构因子估计与峰值分析
- **`evidence_integration/`** — `AnalysisReport` 统一结果契约 + `run_analysis()` 统一入口（15 条路由）
- **`phase_diagnosis/`** — `PhaseDiagnosisReport` 多观测量证据汇总 + 可比性审计 + 规则引擎
- **`Cross_Doping_Summary.py`** / **`SC_Exponent_Summary.py`** — 根目录独立脚本，跨 doping 汇总

# 数据目录约定

UltraDMRG 集群计算结果存放在 `DMRG data visualization/Data/Data_N200/`、`Data/Data_N400/` 与 `Data/Data_Energy/`（**没有 Data_N96**，旧文档引用已过时），层级：

```
Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/*.json
Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_super_corr/super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json
```

实际有数据的组合（其余目录为空骨架）：
- **Data_N200/t_J_3**：doping 4 种（d1_10、d1_20、d1_100、d2_25）；**d1_25、d3_20、d3_25 已因数据问题删除**，不得假定存在或恢复；bond_dim 1500 与 5000；measure 版本 49/50；超导 refbond 49_50/50_51/50_54
- **Data_N400/t_J_3**：doping 5 种（d1_10、d1_20、d1_40、d1_80、d2_25）；bond_dim 1500 与 5000；measure 版本 97/98；超导 refbond 97_98/98_99/98_102；每个 doping 目录内另有 2 份 DMRG 优化日志
- **Data_Energy/t_J_3**：6 个 doping（d0、d1_10、d1_20、d1_40、d1_80、d3_40）× 5 个 N（80/160/240/320/400）共 30 份优化日志
- 其他 t_J 值（t_J_5/10/15/20）目前只有空目录

物理量 JSON 文件：`nf`（密度）、`sz{ver}sz` / `sp{ver}sm` / `sm{ver}sp`（自旋关联三分量）、`nf{ver}nf`（密度-密度关联，**field 连续**，须用 `range(0, N, 1)` 解析）、`single_particle_corr{1..4}`、`sz.json`（局域自旋）。超导关联：bond 方向 A/B/C × 通道 a/b/c/d 共 12 个文件。

JSON 格式统一为 `[[索引], [value, 0]]` 的嵌套列表字符串结构。

已知数据质量事实：N400 超导数据 SU(2) 对称性误差 0.008~0.144（远超 ~1e-4 经验阈值），合成关联值物理解释需谨慎。

## 测量源码位置（仓库外，只读参照）

- 计算源码：`C:\Users\ZhangKun\Desktop\DMRGtJModel`（`src/measure1.cpp`、`src/measureSC.cpp` 等）
- 库函数：`C:\Users\ZhangKun\Desktop\UltraDMRG`；更全副本在 `D:\Microsoft VS Code project\DMRG\`

# 跨 doping 汇总（P4.3）

独立脚本 `Cross_Doping_Summary.py`（测试 `tests/test_cross_doping_summary.py`，68 项断言）：

- **输入范围**：默认 `Data/Data_N200/t_J_3` × bond_dim=1500、N=200；当前实际有数据 4 个 doping（d1_10/d1_20/d1_100/d2_25）
- **运行方式**：`cd "DMRG data visualization" && python Cross_Doping_Summary.py`（Calculation 环境）
- **输出**：`cross_doping_summary.csv` + `cross_doping_summary.json`，输出目录被 .gitignore 忽略
- **可比性契约**（四档）：`direct` / `cautious` / `not_direct` / `sole`（机械条件检查，无物理判据）
- **局限**：不判定物理趋势/相态/趋势显著性，数值变化仅作观测呈现

# 已知坑

- **`DMRG_Log_Analyzer.py` 文件名含零宽字符 U+200B**：正常拼写路径会 FileNotFoundError。用 `glob.glob('DMRG_Log_Analyzer*.py')` 定位
- **Windows 控制台默认 GBK 编码**：运行时用 `python -X utf8` 并 `sys.stdout.reconfigure(encoding='utf-8')`
- **bash 里 `python` 默认指向 Anaconda base 环境**（numpy 1.21.5），不是 Calculation（numpy 2.2.6）。跑测试/脚本必须先 `conda activate Calculation`
- Notebook 中的参数块是示例值，运行前需核对参数块与数据目录是否匹配
- `physical_analysis/Energy_Process.py` 的 `__main__` 演示代码调用了不存在的 `EnergyRecord.from_raw_data`，仅演示用途
- `DensityProcessor.Line_Chart_Visualization` 要求四组水平行数据长度一致，数据不完整会抛 ValueError
- **电荷关联（nf{ver}nf）的 field 是连续编号**：必须用 `range(0, N, 1)` 解析，用自旋的 step-4 序列会静默丢弃 3/4 数据
- **回归基线不可重新生成**：`tests/baselines/` 的 3 个基线是逐字段回归锚点，任何修改分析器输出键集的改动都会触发失败
