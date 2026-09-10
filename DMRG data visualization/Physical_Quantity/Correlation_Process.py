import numpy as np
import matplotlib.pyplot as plt
from .Physical_Data_Record import CorrelationRecord


#-----------------------------------------------------------------------------------
# 关联函数的数据处理器
#----------------------------------------------------------------------------------- 
class CorrelationProcessor:   
    #-----------------------------------------------------------------
    #---------------------- 自旋关联函数的合并方法 ---------------------
    #----------------------------------------------------------------- 
    @staticmethod
    def Combined_spin_correlation(
            SzSz: list[CorrelationRecord],
            SuSd: list[CorrelationRecord],
            SdSu: list[CorrelationRecord]
    ) -> list[CorrelationRecord]:
        """
        <S_i·S_j> = <S^z_i·S^z_j> + 1/2 * (<S^+_i·S^-_j> + <S^-_i·S^+_j>)
        通过上述公式，将输入的三个独立自旋关联函数的CorrelationRecord对象列表进行合并

        Input:
        :SzSz: 存储自旋关联函数<S^z_i·S^z_j>的CorrelationRecord对象列表
        :SuSd: 存储自旋关联函数<S^+_i·S^-_j>的CorrelationRecord对象列表
        :SdSu: 存储自旋关联函数<S^-_i·S^+_j>的CorrelationRecord对象列表

        Return:
        :SiSj: 存储自旋关联函数<S_i·S_j>的CorrelationRecord对象列表
        """
        ### 1.检查三个关联函数长度是否一致
        if len(SzSz) != len(SuSd) or len(SzSz) != len(SdSu):
            raise ValueError("三个自旋关联函数的列表长度不一致")

        ### 2.检查源点是否完全相同
        SzSz_i = SzSz[0].source_point
        SuSd_i = SuSd[0].source_point
        SdSu_i = SdSu[0].source_point
        if SzSz_i != SuSd_i or SzSz_i != SdSu_i:
            raise ValueError("三个自旋关联函数的源点并不都相同")

        ### 3.遍历列表所有元素进行判断与合并
        SiSj = []
        for k in range(len(SzSz)):
            ## 3.1 获取每个列表当前元素(是个CorrelationRecord对象)
            SzSz_Record = SzSz[k]
            SuSd_Record = SuSd[k]
            SdSu_Record = SdSu[k]

            ## 3.2 判断三个自旋关联函数的场点是否一致
            if (SzSz_Record.field_point != SuSd_Record.field_point or 
                SzSz_Record.field_point != SdSu_Record.field_point):
                raise ValueError(f"三个自旋关联函数第{k}个元素的场点并不都相同")

            ## 3.3 获取元素对象的各个属性
            j = SzSz_Record.field_point           # 场点
            r = SzSz_Record.distance              # 距离
            SzSz_value = SzSz_Record.correlation  # <S^z_i·S^z_j>
            SuSd_value = SuSd_Record.correlation  # <S^+_i·S^-_j>
            SdSu_value = SdSu_Record.correlation  # <S^-_i·S^+_j>

            ## 3.4 求出<S_i·S_j>的值，并生成CorrelationRecord对象
            SiSj_value = SzSz_value + 0.5 * (SuSd_value + SdSu_value)
            record = CorrelationRecord(SzSz_i, j, r, SiSj_value)

            ## 3.5 将record添加到SiSj列表中
            SiSj.append(record)

        return SiSj


    #-----------------------------------------------------------------
    #--------------------- 自旋关联函数的数据可视化 --------------------
    #-----------------------------------------------------------------
    @staticmethod
    def Spin_Corr_Visualization(record_list: list[CorrelationRecord]):
        """
        将输入的自旋关联函数对象列表元素进行可视化
        :param record_list: 关联函数对象列表。
        """
        ## 1.获取自旋关联函数的源点
        source_point = record_list[0].source_point

        ## 2. 将数据存储到x轴和y轴列表中
        x_data = [] # 创建存储两格点之间的距离数据
        y_data = [] # 创建存储在x_data的距离下对应的自旋关联函数数据
        for element in record_list:
            x_data.append(element.distance)
            y_data.append(element.correlation)

        ## 3. 创建1行3列的子图布局
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        ## 4. 添加总标题
        fig.suptitle(f'Spin-Spin Correlation Function $\\langle S_i \\cdot S_j \\rangle$ (reference site: {source_point})', 
                     fontsize=20, fontweight='bold', y=1.0)
        

        ## 5. 线性坐标图（原始）
        axs[0].scatter(x_data, y_data, color='blue', s=60, zorder=3)
        axs[0].plot(x_data, y_data, color='red', alpha=0.5, linewidth=2)
        axs[0].set_title(f'Linear Scale', fontsize=16)
        axs[0].set_xlabel('distance r', fontsize=14)
        axs[0].set_ylabel(f'$\\langle S_i \\cdot S_j \\rangle$', fontsize=14)
        axs[0].axhline(y=0, color='black', linewidth=3.5, linestyle='-', alpha=0.8, zorder=1)
        axs[0].grid(True, alpha=0.3, linestyle='--')

        ## 6. 半对数坐标图（y轴对数）- 用于判断指数衰减
        # 注意：如果y_data有负值，对数坐标会有问题，这里取绝对值
        y_abs = np.abs(y_data)
        axs[1].scatter(x_data, y_abs, color='green', s=60, zorder=3)
        axs[1].plot(x_data, y_abs, color='orange', alpha=0.5, linewidth=2)
        axs[1].set_yscale('log')
        axs[1].set_title(f'Log-Linear Scale (y-log)\nExponential decay check', fontsize=16)
        axs[1].set_xlabel('distance r', fontsize=14)
        axs[1].set_ylabel(f'$|\\langle S_i \\cdot S_j \\rangle|$ (log scale)', fontsize=14)
        axs[1].grid(True, alpha=0.3, linestyle='--', which='both')

        ## 7. 双对数坐标图 - 用于判断幂律衰减
        axs[2].scatter(x_data, y_abs, color='purple', s=60, zorder=3)
        axs[2].plot(x_data, y_abs, color='brown', alpha=0.5, linewidth=2)
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_title(f'Log-Log Scale\nPower-law decay check', fontsize=16)
        axs[2].set_xlabel('distance r (log scale)', fontsize=14)
        axs[2].set_ylabel(f'$|\\langle S_i \\cdot S_j \\rangle|$ (log scale)', fontsize=14)
        axs[2].grid(True, alpha=0.3, linestyle='--', which='both')

        ## 8. 调整布局
        plt.tight_layout()

        ## 9. 显示图表
        plt.show()


    
    
    


    #---------------------------------------------------------------------------
    #--------------------- 自旋关联函数的数据绝对误差可视化 --------------------
    #---------------------------------------------------------------------------
    @staticmethod
    def Spin_Corr_Visualization_difference(record_list_1: list[CorrelationRecord], record_list_2: list[CorrelationRecord]):
        """
        将输入的两组自旋关联函数对象列表元素进行绝对误差的可视化
        """
        ### 1. 检查两个自旋关联函数对象列表的长度是否一致
        if len(record_list_1) != len(record_list_2):
            raise ValueError("输入的两个自旋关联函数对象列表的列表长度不一致")
        
        ### 2. 获取自旋关联函数的源点
        source_point = record_list_1[0].source_point
        if source_point != record_list_2[0].source_point:
            raise ValueError("两个自旋关联函数的源点并不相同")

        ### 3. 将数据存储到x轴和y轴列表中
        x_data = [] # 创建存储两格点之间的距离数据
        y1_data = [] # 存储record_list_1的自旋关联函数数据
        y2_data = [] # 存储record_list_2的自旋关联函数数据
        y3_data = [] # 存储两个自旋关联函数之差的数据
        for i, element in enumerate(record_list_1):
            x_data.append(element.distance)
            y1_data.append(element.correlation)
            y2_data.append(record_list_2[i].correlation)
            y3_data.append(abs(element.correlation - record_list_2[i].correlation))

        ### 4. 可视化
        ## 4.1 创建1行3列的子图布局
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        ## 4.2 添加总标题
        fig.suptitle(f'Spin-Spin Correlation Function $\\langle S_i \\cdot S_j \\rangle$ (reference site: {source_point})', 
                     fontsize=20, fontweight='bold', y=1.0)
        

        ## 4.3 record_list_1数据的可视化
        axs[0].scatter(x_data, y1_data, color='blue', s=60, zorder=3)
        axs[0].plot(x_data, y1_data, color='red', alpha=0.5, linewidth=2)
        axs[0].set_title(f'UltraDMRG', fontsize=16)
        axs[0].set_xlabel('distance r', fontsize=14)
        axs[0].set_ylabel(f'$\\langle S_i \\cdot S_j \\rangle$', fontsize=14)
        axs[0].grid(True, alpha=0.3, linestyle='--')

        ## 4.4 record_list_2数据的可视化
        axs[1].scatter(x_data, y2_data, color='green', s=60, zorder=3)
        axs[1].plot(x_data, y2_data, color='orange', alpha=0.5, linewidth=2)
        axs[1].set_title(f'Itensor', fontsize=16)
        axs[1].set_xlabel('distance r', fontsize=14)
        axs[1].set_ylabel(f'$\\langle S_i \\cdot S_j \\rangle$', fontsize=14)
        axs[1].grid(True, alpha=0.3, linestyle='--')

        ## 4.5 record_list_1与record_list_2之差的数据可视化
        axs[2].scatter(x_data, y3_data, color='purple', s=60, zorder=3)
        axs[2].plot(x_data, y3_data, color='brown', alpha=0.5, linewidth=2)
        axs[2].set_title(f'Absolute Error of Spin-Spin Correlation', fontsize=16)
        axs[2].set_xlabel('distance r', fontsize=14)
        axs[2].set_ylabel(f'Absolute Error', fontsize=14)
        axs[2].grid(True, alpha=0.3, linestyle='--')

        ## 4.6 调整布局
        plt.tight_layout()

        ## 4.7 显示图表
        plt.show()



