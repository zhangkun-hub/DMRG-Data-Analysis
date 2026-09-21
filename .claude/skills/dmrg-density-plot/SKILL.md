---
name: dmrg-density-plot
description: 处理 honeycomb 晶格 t-J 模型 DMRG 计算的格点密度数据：解析 Data_N96 下 nf.json 测量文件、用 DensityRecord.extract_data 封装、用 DensityProcessor 画密度分布折线图（按 i%4 分成 4 条水平行）或两组数据（UltraDMRG vs Itensor）的绝对误差对比图。当用户提到密度/电子密度/电荷密度分布、nf.json、DensityRecord、DensityProcessor、密度误差对比，或要在本 DMRG 仓库里处理格点密度数据时使用——只要涉及本仓库密度数据的提取或作图就应触发，即使用户没提"DMRG"。
---

# DMRG 格点密度数据处理与作图

本 skill 覆盖从 UltraDMRG 测量 JSON 提取格点密度并作图的完整链路。核心类：`Physical_Quantity/Physical_Data_Record.py` 的 `DensityRecord` 与 `Physical_Quantity/Density_Process.py` 的 `DensityProcessor`。

## 1. 环境准备（每次运行前）

- 用 Anaconda 环境 **Calculation** 运行：`conda activate Calculation`（numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.7，勿用旧版 API）
- 包导入 `from Physical_Quantity...` 依赖 cwd 在 `sys.path`，因此**必须在 `DMRG data visualization/` 目录下运行**脚本或启动 Jupyter
- Windows 控制台默认 GBK，代码里有中文/Unicode 字符时用 `python -X utf8` 运行，脚本内加 `sys.stdout.reconfigure(encoding='utf-8')`；JSON 文件统一用 `encoding='utf-8'` 读取
- 批处理/无显示环境作图：先 `import matplotlib; matplotlib.use("Agg")` 再 `plt.savefig(...)`，避免 `plt.show()` 阻塞

## 2. 数据定位

密度数据文件是 `Data_N96/{t_J}/{doping}/{bond_dim}_measure_phys_qty/measure_{version}/nf.json`：

- `t_J`：t/J 比值，现有 `t_J_3`、`t_J_5`、`t_J_10`、`t_J_15`、`t_J_20`
- `doping`：空穴率，现有 `d0`、`d1_12`、`d1_16`、`d1_24`、`d1_48`
- `bond_dim`：bond dimension 前缀，有 `measure_*` 与 `1500_measure_*` 两种（notebook 里 `bond_dim = "1500"` 拼出 `1500_measure_phys_qty`）
- `version`：测量版本号，如 `25`、`26`

**运行前先核对参数与目录一致**：notebook 里的 `N = 200`、`d1_10` 是示例值，实际数据只有 N=96 一组；参数不匹配会 FileNotFoundError 或解析失败。可用 `glob` 列出实际目录再取数。

JSON 格式统一为嵌套列表：`[[[0], [0.9998, 0]], [[1], [0.9999, 0]], ...]`，第一层元素是 `[格点索引, [值, 0]]`。

## 3. 提取：DensityRecord

```python
from Physical_Quantity.Physical_Data_Record import DensityRecord

text = open(".../nf.json", encoding="utf-8").read()   # 或 Path(...).read_text(encoding='utf-8')
records = DensityRecord.extract_data(text, N=96, a=1.0)   # a 是晶格常数，默认 1.0
```

- 返回 `list[DensityRecord]`，每个对象有属性：`lattice_point`（格点 i）、`coords`、`density`、`horizontal_starting_point`（= i % 4，用于分 4 条水平行）
- `N` 必须是 8 的倍数（`HoneycombXCGeometry` 校验，否则 ValueError）
- 空数据或格式错误会抛 ValueError（extract_data 内部用 `ast.literal_eval` 解析）

## 4. 作图：DensityProcessor

```python
from Physical_Quantity.Density_Process import DensityProcessor

DensityProcessor.Line_Chart_Visualization(records)   # 单组数据：4 条水平行折线图
```

- 按 `horizontal_starting_point`（0/1/2/3）把格点分成 4 条水平行曲线
- **四行长度必须完全一致**，数据文件缺格点会抛 ValueError（数据不完整时先检查 nf.json 是否有 96 个元素）

## 5. 两组数据绝对误差对比（UltraDMRG vs Itensor）

```python
directions = {
    'UltraDMRG': records_ultra,   # 上面第 3 步得到的对象列表
    'Itensor': records_itensor,   # Itensor 数据同样用 extract_data 封装
}
DensityProcessor.Line_Chart_Visualization_difference(directions)
```

- 取字典**前两组**做对比（保持插入顺序），要求两组长度一致
- Itensor 数据没有文件：notebook 里用 `getpass.getpass()` 粘贴输入；写脚本时改成直接读字符串变量即可

## 6. 已知坑

- `DensityProcessor.Line_Chart_Visualization` 对四行数据长度做 `set` 去重校验，不完整数据抛 `ValueError: 数据长度不一致`
- 仓库根目录跑脚本时 `sys.path[0]` 是根目录，`import Physical_Quantity` 会 ModuleNotFoundError——脚本要放在 `DMRG data visualization/` 里，或从该目录执行
- 密度分布具有蜂窝子格交错特征（如 0.9998 / 0.9999 交替），作图时按 4 条水平行观察即可，不要按 i 顺序连成单条曲线误读物理
