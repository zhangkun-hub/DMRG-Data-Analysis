# AGENTS.md — DMRG data visualization

本目录是项目的核心工作目录。所有脚本和 notebook **必须在本目录下运行**（`sys.path` 依赖 cwd）。

# 通用环境准备

- Anaconda 环境 **Calculation**：`conda activate Calculation`（numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7）
- bash 里 `python` 默认指向 base 环境（numpy 1.21.5），必须先激活 Calculation 或用全路径 `D:\Anaconda\envs\Calculation\python.exe`
- Windows GBK 编码问题：用 `python -X utf8` 运行，脚本内加 `sys.stdout.reconfigure(encoding='utf-8')`；JSON 统一 `encoding='utf-8'` 读取
- 批处理/无显示环境作图：先 `import matplotlib; matplotlib.use("Agg")` 再 `plt.savefig(...)`，避免 `plt.show()` 阻塞
- 导入格式：`from Physical_Quantity.子包名.模块名 import ...`

# 物理量处理工作流

以下三节覆盖密度、自旋关联、超导关联的完整调用链（从数据文件到出图），处理对应物理量时应参照执行。

## 一、格点密度（nf.json）

### 数据定位

`Data_N{200|400}/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/nf.json`

JSON 格式：`[[[0], [0.9998, 0]], [[1], [0.9999, 0]], ...]`，每元素 `[格点索引, [值, 0]]`。

### 提取：DensityRecord

```python
from Physical_Quantity.data_records.Physical_Data_Record import DensityRecord

text = open(".../nf.json", encoding="utf-8").read()
records = DensityRecord.extract_data(text, N=200, a=1.0)  # N 必须是 8 的倍数
```

返回 `list[DensityRecord]`，属性：`lattice_point`、`coords`、`density`、`horizontal_starting_point`（= i % 4，分 4 条水平行）。

### 作图：DensityProcessor

```python
from Physical_Quantity.physical_analysis.Density_Process import DensityProcessor

DensityProcessor.Line_Chart_Visualization(records)          # 单组：4 条水平行折线图
summary = DensityProcessor.profile_summary(records)         # 数值汇总
DensityProcessor.Line_Chart_Visualization_difference({'UltraDMRG': r1, 'Itensor': r2})  # 误差对比
```

- 四行长度必须一致，否则 ValueError
- 统一入口：`run_analysis(OBSERVABLE_LOCAL_DENSITY, operation='profile', records=...)`

## 二、自旋关联（sz{ver}sz / sp{ver}sm / sm{ver}sp）

### 数据定位

三分量文件位于 `measure_{version}/` 下：`sz{ver}sz.json`、`sp{ver}sm.json`、`sm{ver}sp.json`。

**关键：文件交叉对应** — `SuSd ← sp{ver}sm.json`、`SdSu ← sm{ver}sp.json`（变量名与文件名反的，代码注释标了"注意交叉对应"）。

### 提取：CorrelationRecord

```python
from Physical_Quantity.data_records.Physical_Data_Record import CorrelationRecord

N = 200
# A 子格：src_off=1, max_off=-3；B 子格：src_off=2, max_off=-2
src_off, max_off = 1, -3
source_point = int(N / 8 / 4) * 8 + src_off
field_point_max = int(N / 4 * 3) + max_off
sequence = range(source_point, field_point_max + 1, 4)  # 步长 4：同子格

SzSz = CorrelationRecord.extract_data(text_szsz, sequence, N, a=1.0)
```

- 只保留场点在 `sequence` 内的条目
- 三分量用同一个 sequence 提取

### 合成与作图：CorrelationProcessor

```python
from Physical_Quantity.physical_analysis.Correlation_Process import CorrelationProcessor

SiSj = CorrelationProcessor.Combined_spin_correlation(SzSz, SuSd, SdSu)
# <S_i·S_j> = <SzSz> + 1/2(<SuSd> + <SdSu>)

CorrelationProcessor.Spin_Corr_Visualization({'A sublattice': SiSj})  # n×3 子图
CorrelationProcessor.Spin_Corr_Visualization_difference(...)           # 误差对比
```

- 合成前校验三分量长度、源点、场点一致
- `Spin_Corr_Visualization`：行 = 组数，列 = 线性 / y-log / log-log

### 衰减分析与结构因子

- `SpinCorrDecayAnalyzer.analyze(recs, config, N, a)` → `DecayAnalysisResult`（8 类分类）
- `SpinStructureFactorEstimator`（受限估计）+ 峰值分析
- 统一入口：`run_analysis(OBSERVABLE_SPIN_CORRELATION, operation='decay', ...)`

## 三、超导关联（scs{X}{ch}_fix_refbond_{i}_{j}.json）

### 数据定位

`{bond_dim}_measure_super_corr/super_corr_{i}_{j}/scs{X}{ch}_fix_refbond_{i}_{j}.json`

- X = 键方向 A/B/C；ch = 通道 a/b/c/d，共 12 个文件
- N200 refbond：49_50、50_51、50_54；N400 refbond：97_98、98_99、98_102
- **N400 SU(2) 误差 0.008~0.144**，合成结果需谨慎

### 提取：SuperCorrRecord

```python
from Physical_Quantity.data_records.Physical_Data_Record import SuperCorrRecord

records = {}
for bond in 'ABC':
    for channel in 'abcd':
        key = f'scs{bond}{channel}'
        fname = f'{key}_fix_refbond_49_50.json'
        records[key] = SuperCorrRecord.extract_data(
            (base / fname).read_text(encoding='utf-8'), N=200, a=1.0
        )
```

- refbond 必须位于左边 1/4 边界（`left_bound_p = int(N/8/4)*8`）
- 返回属性：`ref_bond`、`target_bond`、`distance`（bond 中点距离）、`sup_corr`

### SU(2) 对称性检验与合成

```python
from Physical_Quantity.physical_analysis.Super_Correlation_Process import SuperCorrProcessor

for bond in 'ABC':
    recs = [records[f'scs{bond}{ch}'] for ch in 'abcd']
    err_ad, err_bc = SuperCorrProcessor.symmetry_check(*recs)  # ~1e-6 极好，~1e-4 可接受，>1e-3 警惕
    combined = SuperCorrProcessor.Combined_super_correlation(*recs)  # <Δ†Δ> = 0.5(a-b-c+d)
```

### 作图

```python
bond_labels = {'A': "A key (blue bond)", 'B': "B key (green bond)", 'C': "C key (orange bond)"}
super_corr_dict = {label: SuperCorrProcessor.Combined_super_correlation(
    *[records[f'scs{b}{ch}'] for ch in 'abcd']) for b, label in bond_labels.items()}

SuperCorrProcessor.Super_Corr_Visualization_three(super_corr_dict)  # 三键 1×3 子图
SuperCorrProcessor.Super_Corr_Visualization_difference(...)          # 误差对比（嵌套字典）
SuperCorrProcessor.Super_Corr_Visualization_difference_refbond(...)  # 收敛诊断布局
```

- **label 格式固定** `"X key (颜色 bond)"`：颜色与键方向名从 label 解析，不能改

### 衰减分析

- `SuperCorrDecayAnalyzer.analyze(record_list, config=None)` → `DecayAnalysisResult`
- 统一入口：`run_analysis(OBSERVABLE_SUPERCONDUCTING_CORRELATION, operation='decay', ...)`
- 配套：SC 配对结构因子（`OBSERVABLE_SC_PAIR_STRUCTURE_FACTOR`）、配对对称（`OBSERVABLE_SC_PAIRING_SYMMETRY`）

# 通用已知坑

- 仓库根目录跑脚本 `import Physical_Quantity` 会 ModuleNotFoundError — 必须从本目录执行
- 不同 bond_dim 对比时目录前缀不同（`measure_super_corr` vs `1500_measure_super_corr`），注意拼路径
- 12 个超导文件缺一不可，refbond 目录名与文件名中 `{i}_{j}` 要一致
- A/B 子格序列参数：A 子格 `src_off=1, max_off=-3`；B 子格 `src_off=2, max_off=-2`；步长恒为 4
- 密度分布具有蜂窝子格交错特征（如 0.9998 / 0.9999 交替），不要按 i 顺序连成单条曲线
- Itensor 数据没有文件，notebook 用 `getpass.getpass()` 粘贴输入
