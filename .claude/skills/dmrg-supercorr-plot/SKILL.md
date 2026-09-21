---
name: dmrg-supercorr-plot
description: 处理 honeycomb 晶格 t-J 模型 DMRG 计算的超导关联数据：解析 Data_N96 下 measure_super_corr 的 12 个 scs{X}{ch}_fix_refbond_{i}_{j}.json 文件、用 SuperCorrRecord.extract_data 封装、做四通道 a/b/c/d 的 SU(2) 对称性检验（a↔d、b↔c）、按 0.5(a-b-c+d) 合成 <Δ†Δ>、用 SuperCorrProcessor 画 A/B/C 三个键方向图及两组数据绝对误差对比。当用户提到超导关联、superconducting correlation、Δ†Δ、SU(2) 对称性检验、scs 文件、reference bond、A/B/C 键方向，或要在本 DMRG 仓库处理超导关联数据时使用——只要涉及本仓库超导关联的提取、对称性检验、合成或作图就应触发，即使用户没提"DMRG"。
---

# DMRG 超导关联数据处理与作图

本 skill 覆盖超导关联 12 文件提取、SU(2) 对称性检验、四通道合成 `<Δ_ij^† Δ_kl>` 并作图的完整链路。核心类：`Physical_Quantity/Physical_Data_Record.py` 的 `SuperCorrRecord` 与 `Physical_Quantity/Super_Correlation_Process.py` 的 `SuperCorrProcessor`。

## 1. 环境准备（每次运行前）

- 用 Anaconda 环境 **Calculation** 运行：`conda activate Calculation`（numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7，勿用旧版 API）
- 包导入 `from Physical_Quantity...` 依赖 cwd 在 `sys.path`，因此**必须在 `DMRG data visualization/` 目录下运行**脚本或启动 Jupyter
- Windows 控制台默认 GBK：用 `python -X utf8` 运行，脚本内加 `sys.stdout.reconfigure(encoding='utf-8')`；JSON 统一 `encoding='utf-8'` 读取
- 批处理/无显示环境作图：先 `import matplotlib; matplotlib.use("Agg")` 再 `plt.savefig(...)`

## 2. 数据定位

超导关联数据位于 `Data_N96/{t_J}/{doping}/{bond_dim}_measure_super_corr/`，每个 reference bond 一个子目录，内含 **12 个文件**：

```
measure_super_corr/super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json
```

- `{i}_{j}`：reference bond 的两个格点，如 `super_corr_25_26`、`super_corr_26_27`（notebook 里 version 参数 `"50_54"` 也是这种 refbond 对，N=200 示例值）
- `X`：键方向 A / B / C；`ch`：通道 a / b / c / d，共 3×4=12 个文件
- 通道物理含义：a = `<c_i↑† c_j↓† c_k↑ c_l↓>`，b = `<c_i↓† c_j↑† c_k↑ c_l↓>`，c = `<c_i↑† c_j↓† c_k↓ c_l↑>`，d = `<c_i↓† c_j↑† c_k↓ c_l↑>`
- `t_J` / `doping` / `bond_dim` 取值同密度 skill

**运行前核对参数与目录一致**：notebook 里 N=200、`d1_10`、`50_54` 是示例值，实际数据只有 N=96，refbond 目录以磁盘为准（可用 `glob` 列 `measure_super_corr/*`）。

## 3. 提取：SuperCorrRecord

```python
from pathlib import Path
from Physical_Quantity.Physical_Data_Record import SuperCorrRecord

base = Path(f'Data_N96/t_J_3/d1_48/measure_super_corr/super_corr_25_26')
records = {}
for bond in 'ABC':
    for channel in 'abcd':
        key = f'scs{bond}{channel}'
        fname = f'{key}_fix_refbond_25_26.json'
        records[key] = SuperCorrRecord.extract_data(
            (base / fname).read_text(encoding='utf-8'), N=96, a=1.0
        )
```

- 返回 `list[SuperCorrRecord]`，属性：`ref_bond`、`target_bond`、`distance`（两个 bond 中点的距离）、`sup_corr`
- extract_data 内部有强筛选：reference bond 必须位于**左边 1/4 边界**（`left_bound_p = int(N/8/4)*8`）；target bond 不超右边界（`l <= int(N*3/4)-1`）且满足 `((k-i)%8, (l-j)%8) ∈ {(0,0),(4,4),(4,7),(5,0),(7,4),(0,5)}`（保持水平方向的 target bond）
- refbond 不在左边界、数据格式错、数据为空都会抛 ValueError

## 4. SU(2) 对称性检验与合成

```python
from Physical_Quantity.Super_Correlation_Process import SuperCorrProcessor

for bond in 'ABC':
    recs = [records[f'scs{bond}{ch}'] for ch in 'abcd']
    err_ad, err_bc = SuperCorrProcessor.symmetry_check(*recs)
    # SU(2) 对称性要求 a==d、b==c；经验判断：~1e-6 极好，~1e-4 可接受，>1e-3 警惕（截断误差/未收敛）
    combined = SuperCorrProcessor.Combined_super_correlation(*recs)
    # <Δ†Δ> = 0.5 (a - b - c + d)
```

- `symmetry_check` 返回 (err_ad, err_bc)：max|a-d|/max|a| 与 max|b-c|/max|b|
- 合成要求四通道长度、ref_bond、target_bond 全部一致，否则 ValueError

## 5. 作图：SuperCorrProcessor

```python
bond_labels = {
    'A': "A key (blue bond)",
    'B': "B key (green bond)",
    'C': "C key (orange bond)",
}
super_corr_dict = {label: SuperCorrProcessor.Combined_super_correlation(*[records[f'scs{b}{ch}'] for ch in 'abcd'])
                   for b, label in bond_labels.items()}

SuperCorrProcessor.Super_Corr_Visualization_three(super_corr_dict)   # 三键方向 1×3 子图
```

- 单键方向用 `Super_Corr_Visualization(record_list, label)`；1×3 子图固定为线性 / y-log / log-log
- **label 格式不能改**：颜色是从 label 里解析的（`label.split("(")[1].split(")")[0].split()[0]`），必须保持 `"X key (颜色 bond)"` 结构
- 两组数据对比：`Super_Corr_Visualization_difference({'UltraDMRG': dict1, 'Itensor': dict2})`——注意是**嵌套字典**（外层两组数据，内层三键方向），取外层前两组

## 6. 已知坑

- 12 个文件缺一不可：文件名由 `scs{X}{ch}_fix_refbond_{i}_{j}.json` 模板拼出，refbond 目录名与文件名中的 `{i}_{j}` 要一致
- `SuperCorrRecord.extract_data` 对 refbond 位置有强校验（左 1/4 边界），换了 refbond 目录后先确认它满足该边界条件
- 不同 bond_dim 对比时目录前缀不同（`measure_super_corr` vs `1500_measure_super_corr`），注意拼路径
- 仓库根目录跑脚本 `import Physical_Quantity` 会 ModuleNotFoundError——从 `DMRG data visualization/` 目录执行

## 7. 衰减类型分析与拟合（SuperCorrDecayAnalyzer）

作图之外需要做衰减类型分类/拟合时，使用同目录 `Super_Correlation_Analyzer.py` 的 `SuperCorrDecayAnalyzer`（本 skill 覆盖的代码均不被其修改）：

- `SuperCorrDecayAnalyzer.analyze(record_list, config=None)` → `DecayAnalysisResult`：输入已合成的 `<Δ†Δ>`（本 skill 第 4 节合成的 `super_corr_dict_UltraDMRG` 值即为合法输入），输出 8 类分类（constant / power_law / exponential / oscillatory_* / non_monotonic / complex_decay / INCONCLUSIVE）+ evidence level
- 配套：`print_summary(result)` 打印完整报告（含 8 项振荡诊断逐项表、r_min 扫描表、噪声底来源）；`Decay_Visualization(result, r, C)` 画 8 面板综合图（r/C 用 `group_by_distance` 的输出）；`analyze_batch({name: records})` 批量分析（不自动做跨 refbond 物理结论）
- 结果含 `analysis_version` 与完整 `AnalyzerConfig` 快照，`to_dict()/to_json(path)` 可追溯
- 该分析器对"合成后"数据工作；SU(2) 对称性误差大（>1e-3）时合成结果本身存疑，先核对本 skill 第 4 节的 symmetry_check 输出
