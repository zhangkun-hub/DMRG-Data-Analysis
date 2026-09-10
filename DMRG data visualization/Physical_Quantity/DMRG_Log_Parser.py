import re
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DMRGLogParser:
    """
    通用DMRG日志Log解析器

    解析信息：
    - Bond dimension
    - Sweep index
    - Ground-state energy E0

    支持：
    - 字符串 / 文件输入
    - 多次 E0 自动平均
    """
    # 初始化方法，接受三个可选参数，每个参数都是正则表达式模式字符串(有默认值)
    def __init__(
        self,
        bond_pattern: str = r'^Bond dimension:\s*(\d+)/\d+$',
        sweep_pattern: str = r'^sweep\s+(\d+)$',
        energy_pattern: str = r'E0\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)'
    ):
        # 编译正则表达式
        self.re_bond = re.compile(bond_pattern)
        self.re_sweep = re.compile(sweep_pattern)
        self.re_energy = re.compile(energy_pattern)

        # 原始数据容器，键值对格式为：(D, sweep) -> list of E0(即value为存储E0的值的列表)
        self._raw = defaultdict(list) # defaultdict(type)创建字典，并为访问不存在的键时提供一个空值(由type决定，比如list就是空列表)，避免KeyError异常

        # 初始化两个变量，用于跟踪当前解析到的Bond和sweep索引(因为日志中Bond和sweep索引通常出现在能量值之前)
        # 类型提示：Optional[int]表示可以是int或None
        self._current_D: Optional[int] = None     # 记录Bond dimension的值
        self._current_sweep: Optional[int] = None  # 记录sweep的值

        # 解析完成标志(完成则为True)
        self._parsed = False

        
    # =========================
    # 读取接口
    # =========================
    def parse_from_string(self, data: str):
        """日志Log以字符串形式进行解析，其中data为字符串数据"""
        self._reset()                # 首先调用_reset()重置内部状态
        self._parse_lines(data.splitlines()) # 将字符串分割成行，传递给_parse_lines()进行实际解析
        self._parsed = True            # 解析完成后将_parsed设为True

        
    def parse_from_file(self, filepath: str):
        """日志Log以文件形式进行解析，其中filepath为文件路径"""
        self._reset()
        with open(filepath, "r") as f:
            self._parse_lines(f)
        self._parsed = True

        
    # =========================
    # 内部解析逻辑
    # =========================
    def _reset(self):
        """重置解析器状态"""
        self._raw.clear()  # _raw.clear()：清空字典_raw中所有的key-value对
        self._current_D = None
        self._current_sweep = None
        self._parsed = False

        
    def _parse_lines(self, lines):
        """解析日志的每行数据，其中lines可以是字符串数据data也可以是文件对象f"""
        for line in lines:
            # 检测 Bond dimension
            mD = self.re_bond.search(line) 
            if mD:
                self._current_D = int(mD.group(1)) # 捕获第一个括号内容(即Bond dimension的值)，并转换成int
                continue

            # 检测 Sweep
            ms = self.re_sweep.search(line)
            if ms:
                self._current_sweep = int(ms.group(1)) # 捕获sweep的值，并转换成int
                continue

            # 检测 Energy
            mE = self.re_energy.search(line)
            if mE and self._current_D is not None and self._current_sweep is not None:
                E0 = float(mE.group(1))  # 捕获能量E0的值，并转换成float
                key = (self._current_D, self._current_sweep)
                self._raw[key].append(E0)

                
    # =========================
    # 数据导出
    # =========================
    def get_raw_dict(self)-> dict[tuple[int, int], list[float]]:
        """返回字典 (D, sweep) -> [E0, ...]"""
        # 首先检查是否已解析数据
        self._require_parsed()  # 内部方法_require_parsed()见最下面工具部分 
            
        # 将defaultdict转换为普通字典返回
        return dict(self._raw)  

    
    def get_dataframe(self) -> pd.DataFrame:
        """返回平均后的 DataFrame"""
        # 首先检查是否已解析数据
        self._require_parsed()  # 内部方法_require_parsed()见最下面工具部分

        # 将提取的数据整理记录并存储在records列表中
        records = []
        for (D, sweep), values in self._raw.items():
            records.append({
                "BondDimension": D,
                "Sweep": sweep,
                "E0_mean": np.mean(values), # np.mean()为平均值计算
                "E0_std": np.std(values),  # np.std()为标准差计算
                "N_samples": len(values)   
            })

        # 将列表records转换为pandas DataFrame对象（以二维表格形式展现）
        df = pd.DataFrame(records) 
        
        # 排序并重置索引：先按BondDimension升序排序，相同BondDimension时，再按Sweep升序排序。排序好后重新生成从0开始的连续索引并丢弃旧索引(drop决定)
        return df.sort_values(["BondDimension", "Sweep"]).reset_index(drop=True)

    
    # =========================
    # 可视化接口
    # =========================
    def plot_E0(self):
        """能量E0随着D增大与相同D下sweep增大所呈现变化规律"""
        df = self.get_dataframe()
        
        # ========= 1. 构造连续x轴并记录每个BondDimension区域的信息 =========
        x_vals = []    # 存储为每个数据点分配的连续x坐标
        self.D_regions = {} # 字典，记录每个BondDimension区域的信息
        x_counter = 0  # 计数器，用于分配连续的x坐标

        # .groupby("BondDimension"): 按照名为"BondDimension" 的列进行分组(即值相同的行分到同一组)
        # D为"BondDimension" 对应的值，g为子DataFrame(子表格)，包含"BondDimension" 等于当前D的所有行数据信息
        for D, g in df.groupby("BondDimension"):
            n = len(g)   # len(g)为子表格的行数，即每个Bond下sweep的数量
            xs = list(range(x_counter, x_counter + n)) # 生成连续整数序列，并以列表形式存储
            x_vals.extend(xs)

            self.D_regions[D] = {
                "start": x_counter,
                "end": x_counter + n - 1,
                "center": x_counter + n/2 - 0.5, # 计算每个Bond区域的中心点
                "count": n
            }
            x_counter += n # 更新计数器

        df["x"] = x_vals  # 将构造的连续x坐标添加到DataFrame中

        # ========= 2. 绘图 =========
        fig, ax = plt.subplots(figsize=(12, 6))

        # ----- 绘制主折线 -----
        ax.plot(df["x"], df["E0_mean"], color="black", linewidth=1.5, zorder=1) # 图层顺序，确保线条在底层

        # ---- 绘制散点与边界线 ----
        colors = plt.cm.tab20.colors   # 使用tab20颜色映射(这是Matplotlib内置的20种区分度好的颜色)来构造颜色列表
        for i, (D, info) in enumerate(self.D_regions.items()):
            mask = df["BondDimension"] == D  # 创建属于D值区域的布尔掩码数组(要求等号成立的均为True，等号不成立的为False)

            # 绘制散点
            ax.scatter(
                df.loc[mask, "x"],             # 取出属于当前D值区域的连续坐标x(即只选择mask为True的行中的"x"列)
                df.loc[mask, "E0_mean"],       # 注：df.loc[]表示从[]内的标签要求来获取数据
                color=colors[i % len(colors)], # 使用取模运算确保索引不会超出颜色列表长度
                s=25, alpha=0.9, label=f"{D}", zorder=2
            )

            # 绘制边界线
            if i > 0:   # 排除最开头的边界
                ax.axvline(         # ax.axvline：绘制单条从底部到顶部的垂直线(注意: 一次只能画一条线)
                    info["start"],  # 边界线位置
                    color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.6, zorder=0
                )

        # ========= 3. x 轴刻度 =========
        # 主刻度（底部 x 轴）
        ax.set_xticks([info["start"] for info in self.D_regions.values()]) # 从D_regions字典的每个值中提取"start"键对应的值，组成一个列表，作为刻度位置
        ax.set_xticklabels([str(D) for D in self.D_regions.keys()], fontsize=12) # 设置主 x 轴的刻度标签
        ax.set_xlabel("Bond Dimension", fontsize=16)

        # 次刻度（顶部 x 轴）
        secax = ax.secondary_xaxis("top") # 创建次 x 轴，并设置在图表顶部(top)
        secax.set_xticks([info["center"] for info in self.D_regions.values()])
        secax.set_xticklabels([str(info["count"]) for info in self.D_regions.values()], fontsize=10)
        secax.set_xlabel("Number of sweeps", fontsize=13)

        # ========= 4. 其它美化 =========
        ax.set_ylabel("Ground-state energy $E_0$", fontsize=16)
        ax.set_title(
            "DMRG Energy Convergence with Bond Dimension", 
            fontsize=20, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(title="Bond Dimension", fontsize=11, title_fontsize=12, ncol=2) # ncol表示设置图例项的列数，比如2表示设置为两列图例项

        plt.tight_layout()
        plt.show()

        
        
    def plot_absolute_energy_differences(self):
        """绝对能量差|ΔE|随着D增大与相同D下sweep增大所呈现的收敛性（半对数坐标系）"""
        df = self.get_dataframe()

        # ========= 1. 构造 |ΔE| 的数据表 df_dE =========
        # 构造 |ΔE| 数据表 df_dE
        # 注意要求：不同bond的分界线选择在bond第一个能量与上一个bond的最后一个能量差值处，所以|ΔE|数据序列是原序列的切片[1:]
        E = df["E0_mean"].values     # 提取能量E0并转为np数组。注：.values就是将提取的df列数据转换为NumPy数组  
        dE = np.abs(np.diff(E))      # |E_n - E_{n-1}|，其中np.diff(E)表示计算数组E的一阶差分(相邻元素的差值)
        df_dE = df.iloc[1:].copy()   # 将df数据表按行进行切片[1:]操作(索引index同样也进行了切片操作)，然后拷贝给df_dE
        df_dE["dE"] = dE             # 创建差值数据列"dE"

        # ========= 2. 绘图 =========
        fig, ax = plt.subplots(figsize=(12, 6))

        # ---- 主折线（连续）----
        ax.plot(df_dE["x"], df_dE["dE"], color="black", linewidth=1.5, zorder=1)

        # ---- 绘制散点与边界线 ----
        colors = plt.cm.tab20.colors
        for i, (D, info) in enumerate(self.D_regions.items()):
            mask = df_dE["BondDimension"] == D  # 创建属于D值区域的布尔掩码数组(要求等号成立的均为True，等号不成立的为False)

            # 绘制散点
            ax.scatter(
                df_dE.loc[mask, "x"],  # 使用布尔索引选择当前Bond对应的x值(即只选择mask为True的行中的"x"列)
                df_dE.loc[mask, "dE"],
                color=colors[i % len(colors)], # 使用取模运算确保索引不会超出颜色列表长度
                s=25, alpha=0.9, label=f"{D}", zorder=2
            )

            # 绘制边界线
            if i > 0:   # 排除最开头的边界
                ax.axvline(         # ax.axvline：绘制单条从底部到顶部的垂直线(注意: 一次只能画一条线)
                    info["start"],  # 边界线位置
                    color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.6, zorder=0
                )

        # ========= 3. x 轴刻度（与能量图一致） =========
        # 主刻度（底部 x 轴）
        ax.set_xticks([info["start"] for info in self.D_regions.values()]) # 从D_regions字典的每个值中提取"start"键对应的值，组成一个列表，作为刻度位置
        ax.set_xticklabels([str(D) for D in self.D_regions.keys()], fontsize=12) # 设置主 x 轴的刻度标签
        ax.set_xlabel("Bond Dimension", fontsize=16)

        # 次刻度（顶部 x 轴）
        secax = ax.secondary_xaxis("top") # 创建次 x 轴，并设置在图表顶部(top)
        secax.set_xticks([info["center"] for info in self.D_regions.values()])
        secax.set_xticklabels([str(info["count"]) for info in self.D_regions.values()], fontsize=10)
        secax.set_xlabel("Number of sweeps", fontsize=13)

        # ========= 4. 其它美化 =========
        # 设置y轴
        ax.set_ylabel(r"$|E_n - E_{n-1}|$", fontsize=16)
        ax.set_yscale("log")  # ★ 设置y轴为对数刻度：看收敛阶数

        # 其他样式美化
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
        ax.set_title(
            "DMRG Convergence Rate: Absolute Energy Differences",
            fontsize=20, fontweight="bold"
        )
        ax.legend(title="Bond Dimension", fontsize=11, title_fontsize=12, ncol=2)

        plt.tight_layout()
        plt.show()
        
        
        
    def plot_DMRG_energy_convergence_analysis(self):
        """能量E0与绝对能量差|ΔE|绘制在一起的两个子图"""
        df = self.get_dataframe()
        
        # ========= 1. 构造连续x轴并记录每个BondDimension区域的信息 =========
        x_vals = []    # 存储为每个数据点分配的连续x坐标
        D_regions = {} # 字典，记录每个BondDimension区域的信息
        x_counter = 0  # 计数器，用于分配连续的x坐标

        # .groupby("BondDimension"): 按照名为"BondDimension" 的列进行分组(即值相同的行分到同一组)
        # D为"BondDimension" 对应的值，g为子DataFrame(子表格)，包含"BondDimension" 等于当前D的所有行数据信息
        for D, g in df.groupby("BondDimension"):
            n = len(g)   # len(g)为子表格的行数，即每个Bond下sweep的数量
            xs = list(range(x_counter, x_counter + n)) # 生成连续整数序列，并以列表形式存储
            x_vals.extend(xs)

            D_regions[D] = {
                "start": x_counter,
                "end": x_counter + n - 1,
                "center": x_counter + n/2 - 0.5, # 计算每个Bond区域的中心点
                "count": n
            }
            x_counter += n # 更新计数器

        df["x"] = x_vals  # 将构造的连续x坐标添加到DataFrame中


        # ========= 2. 构造 |ΔE| 的数据表 df_dE =========
        # 注意要求：不同bond的分界线选择在bond第一个能量与上一个bond的最后一个能量差值处，所以|ΔE|数据序列是原序列的切片[1:]
        E = df["E0_mean"].values     # 提取能量E0并转为np数组。注：.values就是将提取的df列数据转换为NumPy数组  
        dE = np.abs(np.diff(E))      # |E_n - E_{n-1}|，其中np.diff(E)表示计算数组E的一阶差分(相邻元素的差值)
        df_dE = df.iloc[1:].copy()   # 将df数据表按行进行切片[1:]操作(索引index同样也进行了切片操作)，然后拷贝给df_dE
        df_dE["dE"] = dE             # 创建差值数据列"dE"


        # ========= 3. 创建子图与颜色 =========
        # sharex=True：让两个子图共享x轴；'hspace': 0.1：设置垂直子图间距为子图高度的10% 
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'hspace': 0.1}) 

        # 上方的子图ax1是能量图，下方的子图ax2是能量差图
        ax1, ax2 = axes  

        # 设置颜色
        colors = plt.cm.tab20.colors # 使用tab20颜色映射(这是Matplotlib内置的20种区分度好的颜色)来构造颜色列表


        # ========= 4. 绘制第一个子图：能量E0 =========
        # 主折线（连续）
        ax1.plot(df["x"], df["E0_mean"], color="black", linewidth=1.5, zorder=1)

        # 绘制散点与边界线
        for i, (D, info) in enumerate(D_regions.items()):
            mask = df["BondDimension"] == D  # 创建属于D值区域的布尔掩码数组(要求等号成立的均为True，等号不成立的为False)

            # 绘制散点
            ax1.scatter(
                df.loc[mask, "x"],             # 取出属于当前D值区域的连续坐标x(即只选择mask为True的行中的"x"列)
                df.loc[mask, "E0_mean"],       # 注：df.loc[]表示从[]内的标签要求来获取数据
                color=colors[i % len(colors)], # 使用取模运算确保索引不会超出颜色列表长度
                s=25, alpha=0.9, label=f"{D}", zorder=2
            )

            # 绘制边界线
            if i > 0:    # 排除最开头的边界
                ax1.axvline(        # ax.axvline：绘制单条从底部到顶部的垂直线(注意: 一次只能画一条线)
                    info["start"],  # 边界线位置
                    color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.6, zorder=0
                )

        # 第一个子图的y轴标签
        ax1.set_ylabel("Ground-state energy $E_0$", fontsize=16)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # 第一个子图的顶部x轴（次刻度）
        secax_top = ax1.secondary_xaxis("top") # 创建次 x 轴，并设置在图表顶部(top)
        secax_top.set_xticks([info["center"] for info in D_regions.values()]) # 从D_regions字典的每个值中提取"center"键对应的值组成列表作为刻度位置
        secax_top.set_xticklabels([str(info["count"]) for info in D_regions.values()], fontsize=10) # 设置次 x 轴的刻度标签
        secax_top.set_xlabel("Number of sweeps", fontsize=13)

        # 隐藏第一个子图的x轴标签
        ax1.tick_params(axis='x', labelbottom=False)

        # 在第一个子图中添加图例
        ax1.legend(title="Bond Dimension", fontsize=11, title_fontsize=12, ncol=2)


        # ========= 5. 绘制第二个子图：能量差dE =========
        # 主折线
        ax2.plot(df_dE["x"], df_dE["dE"], color="black", linewidth=1.5, zorder=1)

        # 绘制散点与边界线
        for i, (D, info) in enumerate(D_regions.items()):
            mask = df_dE["BondDimension"] == D  # 创建属于D值区域的布尔掩码数组(要求等号成立的均为True，等号不成立的为False)

            # 绘制散点
            ax2.scatter(
                df_dE.loc[mask, "x"],  # 使用布尔索引选择当前Bond对应的x值(即只选择mask为True的行中的"x"列)
                df_dE.loc[mask, "dE"],
                color=colors[i % len(colors)], # 使用取模运算确保索引不会超出颜色列表长度
                s=25, alpha=0.9, label=f"{D}", zorder=2
            )

            # 绘制边界线
            if i > 0:
                ax2.axvline(
                    info["start"],
                    color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.6, zorder=0
                )

        # 第二个子图的y轴设置为对数刻度
        ax2.set_yscale("log")  # ★ 设置y轴为对数刻度：看收敛阶数
        ax2.set_ylabel(r"$|E_n - E_{n-1}|$", fontsize=16)
        ax2.grid(True, which="both", alpha=0.3, linestyle="--")

        # 设置x轴刻度（共享x轴，只在下图显示标签）
        ax2.set_xticks([info["start"] for info in D_regions.values()]) # 从D_regions字典的每个值中提取"start"键对应的值组成一个列表，作为刻度位置
        ax2.set_xticklabels([str(D) for D in D_regions.keys()], fontsize=12) # 设置主 x 轴的刻度标签
        ax2.set_xlabel("Bond Dimension", fontsize=16)


        # ========= 6. 全局设置 =========
        # 设置总标题
        fig.suptitle("DMRG Energy Convergence Analysis", fontsize=20, fontweight="bold")

        plt.tight_layout()
        plt.show()

        
    # =========================
    # 工具
    # =========================
    def _require_parsed(self):
        if not self._parsed:
            raise RuntimeError("日志尚未解析，请先调用 parse_from_string 或 parse_from_file")
            
            
            
            
            
#-----------------------------------------------------------------------------------
# 使用示例（模拟完整工作流）
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    # 建立通用DMRG日志Log解析器对象并传入日志字符串数据
    data = ""
    parser = DMRGLogParser()
    parser.parse_from_string(data)   # data 是日志字符串

    # 将解析的数据存储为pandas DataFrame对象（以二维表格形式展现）
    # 其中表格的列分别为：BondDimension，Sweep，E0_mean(平均值)，E0_std(标准差)，N_samples(每个sweep下能量数目)
    df = parser.get_dataframe()
    print(df.head())

    # 可视化
    parser.plot_E0()
    parser.plot_absolute_energy_differences()
    parser.plot_DMRG_energy_convergence_analysis()
