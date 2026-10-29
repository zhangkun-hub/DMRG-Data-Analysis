import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List


#-----------------------------------------------------------------------------------
# 1. 数据记录类
#-----------------------------------------------------------------------------------       
@dataclass  # @dataclass 装饰器，可以自动生成 __init__、__repr__ 等方法，从而省去大量样板代码，让代码更简洁。
class EnergyRecord:
    """能量的数据记录器，存储物理系统能量相关的4个实例属性"""
    N: int          # 系统格点数N
    E_min: float      # 提取能量数据的最小值
    E_avg: float      # 提取能量数据的平均值
    Es: float        # 单格点能量Es
    Es_J: float       # Es与J的比值

    def __str__(self):
        return (
            f"系统格点数: N = {self.N}\n"
            f"能量数据的最小值: E_min = {self.E_min}\n"
            f"能量数据的平均值: E_avg = {self.E_avg}\n"
            f"单格点能量: Es = {self.Es}\n"
            f"Es与J的比值: Es_J = {self.Es_J}"
        )

    
    # 单次sweep后数据提取的类方法(类方法作用：只需调用EnergyRecord.extract_data(...)即可获得完整的EnergyRecord对象)
    @classmethod
    def extract_data(cls, data: str, N: int, J: float) -> 'EnergyRecord':
        """
        将单次sweep后数据data的所有E0值提取出来，进行数据处理，然后封装为EnergyRecord对象，
        最后返回该EnergyRecord对象。
        
        :param data: 输入的原始数据字符串
        :param N: 系统格点数
        :param J: t-J模型的系数J
        
        :return cls: 封装好的EnergyRecord对象，这是类方法的特性
        """
        if N <= 0:
            raise ValueError("N 必须为正整数")
        if J == 0:
            raise ValueError("J 不能为 0")
            
        pattern = r'E0\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)'
        matches = re.findall(pattern, data)
        if not matches:
            raise ValueError(f"N={N} 的数据中未找到 E0")
        
        # 以列表形式存储所有能量
        energies = [float(e) for e in matches]
        
        # 计算列表数据
        E_min = min(energies)
        E_avg = sum(energies) / len(matches)
        Es = E_avg / N
        
        return cls(N = N, E_min = E_min, E_avg = E_avg, Es = Es, Es_J = Es / J)
    
    

#-----------------------------------------------------------------------------------
# 2. 能量处理器
#-----------------------------------------------------------------------------------
class EnergyProcessor:
    ### MAD 异常点诊断核心（共用私有实现：extrapolate 与
    ### diagnose_outliers 共用，零复制）
    @staticmethod
    def _outlier_diagnosis(ns, y, power, threshold):
        """MAD 稳健尺度候选异常点诊断核心（B3-A3 算法，原样保留）。

        输入 ns/y 为按 N 升序的格点数数组与单格点能量数组。
        :return dict：诊断字段语义见 diagnose_outliers 的文档。
        """
        x = 1 / (np.asarray(ns, dtype=float) ** power)

        ## 初始拟合（诊断用全点拟合）
        coeffs_init = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs_init, x)
        residuals = y - y_pred  # 残差：衡量拟合的好坏程度

        ## 残差尺度：稳健 MAD（中位数绝对偏差）× 1.4826
        # 1.4826 为正态分布一致性常数（正态数据下等价于标准差）；
        # MAD 对离群点自举污染稳健（B3-A2 实测：std 判据在 5 点小样本
        # 下因离群点污染尺度而漏检）
        mad_res = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad_res

        ## 退化保护：残差尺度接近浮点噪声量级（完美共线/近线性
        # 数据残差 ~1e-15×|y|）时跳过阈值比较，比值按不可定义处理
        # （None，不伪造有限比值）——避免把所有点误判为候选异常、
        # 避免空数组传入 polyfit（B3-A2 实测崩溃）
        y_scale = np.median(np.abs(y))
        if y_scale > 0 and scale <= 1e-12 * y_scale:
            degenerate = True
            mask = np.ones(len(x), dtype=bool)   # 全部保留，不比较阈值
            residual_ratios = {str(int(n)): None for n in ns}
        else:
            degenerate = False
            ## 判断准则：|残差| ≤ threshold × 稳健尺度 → 正常点
            # 默认threshold=2.0，即保留残差在2个稳健尺度内的点
            mask = np.abs(residuals) <= threshold * scale
            # 最小样本保护：剔除后剩余 < 3（契约最低拟合点数）→ 按
            # 残差降序恢复点至 3（不得盲目继续剔除）
            if np.sum(mask) < 3:
                order = np.argsort(np.abs(residuals))[::-1]  # 残差降序
                for idx in order:
                    if np.sum(mask) >= 3:
                        break
                    mask[idx] = True
            residual_ratios = {str(int(n)): float(r / scale)
                               for n, r in zip(ns, residuals)}

        ## 敏感性参照：全点拟合截距（正式主结果）与剔除候选点后的
        # 重拟合截距
        e_inf_all = np.polyfit(x, y, 1)[1]
        e_inf_refit = np.polyfit(x[mask], y[mask], 1)[1]
        return {
            'masked_N': [int(n) for n in ns[~mask]],
            'residual_ratios': residual_ratios,
            'scale': float(scale),
            'degenerate': bool(degenerate),
            'E_inf_all_points': float(e_inf_all),
            'E_inf_refit': float(e_inf_refit),
            'sensitivity_delta': float(e_inf_refit - e_inf_all),
            'mask': [bool(m) for m in mask],
        }

    ### MAD 候选异常点诊断（公开入口，纯数值）
    @staticmethod
    def diagnose_outliers(records: List[EnergyRecord], power: int = 1,
                          threshold: float = 2.0):
        """MAD 稳健尺度候选异常点诊断（纯数值，不绘图、不打印）。

        与 extrapolate 共用同一诊断算法（_outlier_diagnosis），仅返回
        结构化诊断信息供敏感性分析使用；不修改、不决定任何主结果。
        数据口径与 extrapolate 一致：x = 1/N^power，y = Es。

        :param records: 数据列表，元素为封装的EnergyRecord对象。
        :param power: 外推幂次（同 extrapolate）。
        :param threshold: 诊断阈值（|残差|/稳健尺度 ≤ threshold 视为
            正常点）。仅影响诊断候选列表，不影响任何主拟合。

        :return dict，字段语义：
            masked_N (list[int])：候选异常点的 N 列表（|残差|/稳健
                尺度超过 threshold；最小样本保护保证正常点至少 3 个）。
                无候选点时为空列表 []。
            residual_ratios (dict[str→float|None])：各 N 的残差/稳健
                尺度比（有符号，与残差同号；键为 N 的十进制字符串）。
                尺度退化（degenerate=True）时比值不可定义，统一为
                None——不伪造有限比值。
            scale (float)：稳健尺度 = 1.4826 × MAD（残差的中位数绝对
                偏差），与 |残差| 同单位（Es 单位）。
            degenerate (bool)：尺度退化标记（scale ≤ 1e-12 × median|y|，
                完美共线/近线性数据的浮点噪声量级）。True 时不执行
                阈值比较：masked_N 恒为空、residual_ratios 全 None。
            E_inf_all_points (float)：全部数据点的线性拟合截距（正式
                主结果，与 extrapolate 返回值一致）。
            E_inf_refit (float)：剔除候选异常点后的重拟合截距（敏感
                性参照，不进入主结果）。
            sensitivity_delta (float)：E_inf_refit − E_inf_all_points。
                无候选点时（含退化情况）为 0.0。
            mask (list[bool])：按 N 升序与输入一一对应的正常点标记
                （True=正常点），extrapolate 绘图共用。

        记录少于 3 条时抛 ValueError（与 extrapolate 同一契约）。
        """
        if len(records) < 3:
            raise ValueError("检测异常点至少需要 3 组数据")
        sorted_records = sorted(records, key=lambda r: r.N)
        ns = np.array([r.N for r in sorted_records])
        y = np.array([r.Es for r in sorted_records])
        return EnergyProcessor._outlier_diagnosis(ns, y, power, threshold)

    ### 热力学极限外推方法（全点主结果 + MAD 诊断）
    @staticmethod
    def extrapolate(records: List[EnergyRecord], power: int = 1, threshold: float = 2.0):
        """
        进行热力学极限外推。
        :param records: 数据列表，元素为封装的EnergyRecord对象。
        :param power: 表示是随1/N^power进行热力学极限的外推，也就是Es(N) = E∞ + a/N^power。默认为1（常用N→∞的渐近形式）。
        :param threshold: 候选异常点诊断阈值（|残差|与稳健尺度的比值），
            仅用于诊断输出与绘图标注——不改变主结果拟合输入；正式
            E_inf 恒由全部有效数据点拟合得到。诊断数值细节见
            diagnose_outliers。

        :return e_inf: 热力学极限下能量值（全部数据点的线性拟合截距）。
        """
        if len(records) < 3:
            raise ValueError("检测异常点至少需要 3 组数据")

        ### 1. 数据准备
        sorted_records = sorted(records, key=lambda r: r.N)  # 将数据按粒子数N升序排序，其中key=lambda r: r.N是提取每个对象的N值
        ns = np.array([r.N for r in sorted_records])
        x = 1 / (ns ** power)
        y = np.array([r.Es for r in sorted_records])

        
        ### 2. MAD 候选异常点诊断（仅输出诊断信息，不决定拟合输入；
        # 算法主体在 _outlier_diagnosis，与 diagnose_outliers 共用）
        diag = EnergyProcessor._outlier_diagnosis(ns, y, power, threshold)
        mask = np.array(diag['mask'], dtype=bool)
        outlier_mask = ~mask  # ~按位取反运算符：对于布尔类型，会将True变False，False变True

        ### 3. 打印诊断信息（候选异常点，非已确认数据错误）
        if np.any(outlier_mask):  # np.any()：检查数组中是否有True值，如果有一个True就返回True，如果全为False就返回false
            print(f"检测到候选异常点 (N): {diag['masked_N']}（仅诊断——主结果使用全部数据点，不剔除）")
        else:
            print("未检测到显著候选异常点")

        ### 4. 最终拟合（恒使用全部有效数据点——MAD 诊断不改变拟合输入）
        coeffs_final = np.polyfit(x, y, 1)
        slope, e_inf = coeffs_final

        # 构造拟合函数字符串
        func_str = f"$Es(N) = {e_inf:.6f} + \\frac{{{slope:.4f}}}{{N^{power}}}$"
        

        ### 4. 绘图
        plt.figure(figsize=(10, 8))
        
        ## 4.1 绘制正常点和候选异常点（候选点仅诊断标注，不影响拟合）
        plt.scatter(x[mask], y[mask], color='blue', label='Valid Data', zorder=5)
        plt.scatter(x[outlier_mask], y[outlier_mask], color='red', marker='x', s=100, label='Candidate Outliers (Diagnostic Only)', zorder=5)

        ## 4.2 绘制外推拟合线
        x_range = np.linspace(0, np.max(x) * 1.1, 100) # 生成等距的100个x坐标点，范围从0到最大x值的1.1倍
        plt.plot(x_range, np.polyval(coeffs_final, x_range), 'r-', alpha=0.7, label='Linear Fit')

        ## 4.3 添加热力学极限横虚线
        plt.axhline(y=e_inf, color='green', linestyle=':', linewidth=1.5, label=f'Limit ($E_\infty$)')
        
        ## 4.4 在纵轴标注外推值
        plt.plot(0, e_inf, 'go', markersize=8)
        
        # plt.annotate(text, xy, xytext, arrowprops)：在图形xy位置上添加带箭头的文本text标注，xytext为文本起始位置，arrowprops为箭头设置选项
        plt.annotate(f'{e_inf:.6f}', xy=(0, e_inf), xytext=(np.max(x)*0.05, e_inf + (np.max(y)-np.min(y))*0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        ## 4.5 图形修饰
        plt.title("Thermodynamic Limit Extrapolation with Outlier Detection", fontsize=20)
        plt.xlabel(f"$1 / N^{power}$", fontsize=15)
        plt.ylabel("$E_s$ (Energy per site)", fontsize=15)
        plt.grid(True, which='both', linestyle='--', alpha=0.5) # 添加网格
        plt.legend(fontsize=15)
        
        print(f"--- 最终结果 ---")
        print(f"外推公式: Es(N) = {e_inf:.6f} + {slope:.4f} * (1/N^{power})")
        print(f"热力学极限 E_s(∞) = {e_inf:.6f}")
        print(f"主结果基于全部 {len(x)} 个数据点（MAD 诊断不影响拟合输入）")

        plt.show()
        
        return e_inf

    
    
    ### 将不同空穴率δ下与热力学极限下的能量进行线性拟合
    @staticmethod
    def fit_doping_dependence(delta_list: List[float], e_inf_list: List[float]):
        """
        对空穴率 delta 与 热力学极限能量 E_inf 进行线性拟合与绘图
        :param delta_list: 空穴率数据列表，作为x轴坐标。
        :param e_inf_list: 热力学极限下能量数据列表，作为y轴坐标。

        
        :return slope, intercept: 线性拟合的斜率slope和截距intercept。
        """
        if len(delta_list) != len(e_inf_list):
            raise ValueError("空穴率列表与能量列表长度必须一致")
        
        x = np.array(delta_list)
        y = np.array(e_inf_list)

        # 线性拟合 y = k * x + b
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        poly_func = np.poly1d(coeffs)

        # 构造公式字符串
        func_str = f"$E_\\infty(\\delta) = {slope:.4f}\\delta + ({intercept:.4f})$"

        # 绘图
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, color='blue', s=80, edgecolors='black', label='Extrapolated $E_\infty$', zorder=3)
        
        # 拟合线
        x_fit = np.linspace(min(x)*0.9, max(x)*1.1, 100)
        plt.plot(x_fit, poly_func(x_fit), 'r-', linewidth=2, label='Linear Fit', zorder=2)

        # 图形美化
        plt.title("Energy at Thermodynamic Limit vs. Hole Doping $\delta$", fontsize=20)
        plt.xlabel("Hole Doping $\delta$", fontsize=15)
        plt.ylabel("Energy per site $E_\infty$", fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=15)
        
        plt.show()

        print(f"--- 掺杂依赖拟合结果 ---")
        print(f"拟合公式: E_∞(δ) = {slope:.4f}δ + ({intercept:.4f})")
        
        return slope, intercept
    
    
    
#-----------------------------------------------------------------------------------
# 3. 使用示例（模拟完整工作流）
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    ############### 热力学极限外推的使用 ##############
    # 构造模拟数据（人为加入一个异常点）
    # 理论值应在 -1.5 左右
    raw_data_sets = {
        10:  "E0 = -14.2",  # Es = -1.42
        20:  "E0 = -29.6",  # Es = -1.48
        30:  "E0 = -44.7",  # Es = -1.49
        40:  "E0 = -40.0",  # Es = -1.00 <-- 这是一个明显的异常点
        60:  "E0 = -89.7",  # Es = -1.495
        100: "E0 = -149.8" # Es = -1.498
    }
    
    records = []
    for n, data in raw_data_sets.items():
        records.append(EnergyRecord.from_raw_data(data, N=n, J=1.0))

    # 执行外推（设定 threshold=2.0，如果点偏离超过 2 倍标准差则剔除）
    extrapolator = EnergyProcessor()
    extrapolator.extrapolate(records, power=1, threshold=1.5)
    
    
    
    ################# 线性拟合的使用 ######################
    # 假设我们有三个不同的空穴率 delta
    dopings = [0.05, 0.10, 0.15]
    
    # 模拟每个 delta 下不同格点数 N 的实验数据
    # delta = 0.05
    data_delta_05 = {10: "E0=-15.1", 20: "E0=-30.8", 40: "E0=-62.1", 80: "E0=-124.5"}
    # delta = 0.10
    data_delta_10 = {10: "E0=-14.5", 20: "E0=-29.8", 40: "E0=-60.2", 80: "E0=-121.0"}
    # delta = 0.15
    data_delta_15 = {10: "E0=-13.8", 20: "E0=-28.5", 40: "E0=-57.8", 80: "E0=-116.5"}
    
    raw_experimental_data = [data_delta_05, data_delta_10, data_delta_15]
    
    extrapolated_energies = []
    extrapolator = EnergyProcessor()

    # 第一步：遍历每个空穴率，求出其热力学极限能量
    for i, data_dict in enumerate(raw_experimental_data):
        print(f"\n处理空穴率 δ = {dopings[i]} 的数据...")
        records = [EnergyRecord.from_raw_data(v, n, 1.0) for n, v in data_dict.items()]
        e_limit = extrapolator.extrapolate(records, power=1)
        extrapolated_energies.append(e_limit)

    # 第二步：将求得的极限能量随空穴率的变化进行拟合
    print("\n进行空穴率依赖性外推...")
    extrapolator.fit_doping_dependence(dopings, extrapolated_energies)