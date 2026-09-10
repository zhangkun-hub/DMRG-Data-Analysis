import sys
from Physical_Quantity.DMRG_Log_Parser import DMRGLogParser


### 1.输入日志内容并转换成字符串
print("请输入能量数据（Ctrl+Z(windows) 或 Ctrl+Z(Linux/Mac) 结束输入）:") 
data = sys.stdin.read()  # 当所有内容输入完毕后，先回车，再按Ctrl+Z，再回车，从而结束输入

### 2.提取数据的正则表达式
energy_pattern = r'^Site\s*\(\s*40,\s*41\).*?E0\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)' 

### 3.建立通用DMRG日志Log解析器对象并传入日志字符串数据
parser = DMRGLogParser(energy_pattern = energy_pattern)
parser.parse_from_string(data)   # data 是日志字符串

### 4.将解析的数据存储为pandas DataFrame对象（以二维表格形式展现）
# 其中表格的列分别为：BondDimension，Sweep，E0_mean(平均值)，E0_std(标准差)，N_samples(每个sweep下能量数目)
df = parser.get_dataframe()
print(df.head())

### 5.可视化
parser.plot_DMRG_energy_convergence_analysis()
