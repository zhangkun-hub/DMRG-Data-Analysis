---
name: dmrg-spin-corr-plot
description: 处理 honeycomb 晶格 t-J 模型 DMRG 计算的自旋关联数据：解析 Data_N200/Data_N400 下 sz{ver}sz / sp{ver}sm / sm{ver}sp 三分量 JSON、用 CorrelationRecord.extract_data 按 A/B 子格序列封装、由 SzSz + 1/2(SuSd + SdSu) 合成 <S_i·S_j>、用 CorrelationProcessor 画 1×3 子图（线性/y-log/log-log 判断指数或幂律衰减）及两组数据绝对误差对比。当用户提到自旋关联、spin correlation、SzSz、SuSd、SdSu、指数衰减、幂律衰减，或要在本 DMRG 仓库处理自旋关联数据时使用——只要涉及本仓库自旋关联的提取、合成或作图就应触发，即使用户没提"DMRG"。
---

# DMRG 自旋关联数据处理与作图

本 skill 覆盖自旋关联三分量提取、合成 `<S_i·S_j>` 并作图的完整链路。核心类：`Physical_Quantity/Physical_Data_Record.py` 的 `CorrelationRecord` 与 `Physical_Quantity/Correlation_Process.py` 的 `CorrelationProcessor`。

## 1. 环境准备（每次运行前）

- 用 Anaconda 环境 **Calculation** 运行：`conda activate Calculation`（numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7，勿用旧版 API）
- 包导入 `from Physical_Quantity...` 依赖 cwd 在 `sys.path`，因此**必须在 `DMRG data visualization/` 目录下运行**脚本或启动 Jupyter
- Windows 控制台默认 GBK：用 `python -X utf8` 运行，脚本内加 `sys.stdout.reconfigure(encoding='utf-8')`；JSON 统一 `encoding='utf-8'` 读取
- 批处理/无显示环境作图：先 `import matplotlib; matplotlib.use("Agg")` 再 `plt.savefig(...)`

## 2. 数据定位

自旋关联有三个分量文件，位于 `Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/`：

- `sz{version}sz.json` → `<S^z_i S^z_j>`（SzSz）
- `sp{version}sm.json`、`sm{version}sp.json` → 横向两分量
- 实际有数据的组合：`Data_N200/t_J_3` 7 种 doping（d1_10/d1_20/d1_25/d1_100/d2_25/d3_20/d3_25）、1500 全部 + 5000 部分、measure 版本 49/50；`Data_N400/t_J_3` 仅 d1_10/1500、版本 97/98（**仓库没有 Data_N96**）

**关键约定——文件交叉对应**：notebook 中实际是 `SuSd ← sp{ver}sm.json`、`SdSu ← sm{ver}sp.json`（notebook 里变量名与文件名的对应是反的，代码注释标了"注意交叉对应"）。写新脚本时按此映射读取，不要凭文件名直觉对应。

**运行前核对参数与目录一致**：notebook 里的参数块是示例值；参数不匹配会 FileNotFoundError。可用 `glob` 列实际目录。

## 3. 提取：CorrelationRecord

`extract_data` 需要 A/B 子格序列 `sequence`，用于筛选有效源点/场点：

```python
from Physical_Quantity.Physical_Data_Record import CorrelationRecord

N = 200
src_off, max_off = 1, -3        # A 子格；B 子格用 (2, -2)
source_point = int(N / 8 / 4) * 8 + src_off
field_point_max = int(N / 4 * 3) + max_off
sequence = range(source_point, field_point_max + 1, 4)   # 步长 4：同子格才有意义

text_szsz = open(".../sz25sz.json", encoding="utf-8").read()
SzSz = CorrelationRecord.extract_data(text_szsz, sequence, N, a=1.0)
```

- 返回 `list[CorrelationRecord]`，属性：`source_point`、`field_point`、`distance`（由 `HoneycombXCGeometry.get_distance` 计算）、`correlation`
- 只保留场点在 `sequence` 内的条目；数据第一个源点不在 `sequence` 内会直接抛 ValueError
- 三分量用**同一个 sequence** 提取，源点偏移不同时 A/B 子格分别构造

## 4. 合成与作图：CorrelationProcessor

```python
from Physical_Quantity.Correlation_Process import CorrelationProcessor

SiSj = CorrelationProcessor.Combined_spin_correlation(SzSz, SuSd, SdSu)
# 即 <S_i·S_j> = <SzSz> + 1/2 (<SuSd> + <SdSu>)

CorrelationProcessor.Spin_Corr_Visualization(SiSj)
```

- 合成前会校验三分量列表长度、源点、场点一致，不一致抛 ValueError（说明数据版本/序列不配套）
- 作图固定 1×3 子图：线性（原始值）/ y-log（|值|，判断指数衰减）/ log-log（|值|，判断幂律衰减）；负值取对数前必须取绝对值
- 两组数据对比用 `Spin_Corr_Visualization_difference({'UltraDMRG': r1, 'Itensor': r2})`，取字典前两组，要求长度与源点一致

## 5. 已知坑

- 三分量文件与变量的**交叉对应**（见第 2 节）最容易搞错；搞反后合成值物理上不对但不会报错，务必核对
- A/B 子格序列参数：A 子格 `src_off=1, max_off=-3`；B 子格 `src_off=2, max_off=-2`；步长恒为 4
- 仓库根目录跑脚本 `import Physical_Quantity` 会 ModuleNotFoundError——从 `DMRG data visualization/` 目录执行
- 数据源点必须落在 `sequence` 里（extract_data 强校验），换个 version 的数据如果源点变了要先重算 sequence
