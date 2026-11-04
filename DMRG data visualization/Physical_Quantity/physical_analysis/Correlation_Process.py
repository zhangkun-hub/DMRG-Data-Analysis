import numpy as np
import matplotlib.pyplot as plt
from ..data_records.Physical_Data_Record import CorrelationRecord


def _sublattice_suffix(source_point):
    """测量序列约定的源点子格：i%4==1 → 'A'，i%4==2 → 'B'；其他 → None。

    与数据提取时 A/B 子格序列约定同源（A: src_off=1、B: src_off=2，步长 4）；
    注意 HoneycombXCGeometry.sub_lattice_partition 的 BFS 二部染色是另一套
    命名，不得混用。
    """
    return {1: 'A', 2: 'B'}.get(int(source_point) % 4)


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
    def Spin_Corr_Visualization(
            records_dict: dict[str, list[CorrelationRecord]]):
        """
        将输入的自旋关联函数对象列表进行可视化。

        :param records_dict: 组名 → 关联函数对象列表。传入 n 组数据即画
            n 行 × 3 列（列：线性 / y-log / log-log）。组名建议带子格语义
            （如 'A sublattice'、'B sublattice'），第一列 y 轴标签会自动补
            源点：'A sublattice (ref-site:49)'。
        """
        if not records_dict:
            raise ValueError("records_dict 不能为空")

        ## 1. 创建 n 行 3 列的子图布局
        n_rows = len(records_dict)
        fig, axs = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
        axs = axs.reshape(n_rows, 3)
        row_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        line_colors = ['red', 'orange', 'brown']

        ## 2. 添加总标题
        fig.suptitle(
            r'Spin-Spin Correlation Function $\langle S_i \cdot S_j \rangle$',
            fontsize=20, fontweight='bold', y=1.0)

        ## 3. 逐组作图（一行一组）
        for i, (label, record_list) in enumerate(records_dict.items()):
            if not record_list:
                raise ValueError(f"组 {label!r} 的记录列表为空")
            source_point = record_list[0].source_point
            x_data = [e.distance for e in record_list]
            y_data = [e.correlation for e in record_list]
            y_abs = np.abs(y_data)
            c = row_colors[i % len(row_colors)]
            row_title = f'{label} (ref-site:{source_point})'

            ## 3.1 线性坐标图（原始）
            axs[i, 0].scatter(x_data, y_data, color=c, s=60, zorder=3)
            axs[i, 0].plot(x_data, y_data, color=line_colors[0], alpha=0.5,
                           linewidth=2)
            axs[i, 0].set_title('Linear Scale', fontsize=16)
            axs[i, 0].set_xlabel('distance r', fontsize=14)
            axs[i, 0].set_ylabel(
                f'{row_title}\n$\\langle S_i \\cdot S_j \\rangle$', fontsize=14)
            axs[i, 0].axhline(y=0, color='black', linewidth=3.5, linestyle='-',
                              alpha=0.8, zorder=1)
            axs[i, 0].grid(True, alpha=0.3, linestyle='--')

            ## 3.2 半对数坐标图（y轴对数）- 用于判断指数衰减
            # 注意：如果y_data有负值，对数坐标会有问题，这里取绝对值
            axs[i, 1].scatter(x_data, y_abs, color=c, s=60, zorder=3)
            axs[i, 1].plot(x_data, y_abs, color=line_colors[1], alpha=0.5,
                           linewidth=2)
            axs[i, 1].set_yscale('log')
            axs[i, 1].set_title('Log-Linear Scale (y-log)\n'
                                'Exponential decay check', fontsize=16)
            axs[i, 1].set_xlabel('distance r', fontsize=14)
            axs[i, 1].set_ylabel(
                f'$|\\langle S_i \\cdot S_j \\rangle|$ (log scale)', fontsize=14)
            axs[i, 1].grid(True, alpha=0.3, linestyle='--', which='both')

            ## 3.3 双对数坐标图 - 用于判断幂律衰减
            axs[i, 2].scatter(x_data, y_abs, color=c, s=60, zorder=3)
            axs[i, 2].plot(x_data, y_abs, color=line_colors[2], alpha=0.5,
                           linewidth=2)
            axs[i, 2].set_xscale('log')
            axs[i, 2].set_yscale('log')
            axs[i, 2].set_title('Log-Log Scale\nPower-law decay check',
                                fontsize=16)
            axs[i, 2].set_xlabel('distance r (log scale)', fontsize=14)
            axs[i, 2].set_ylabel(
                f'$|\\langle S_i \\cdot S_j \\rangle|$ (log scale)', fontsize=14)
            axs[i, 2].grid(True, alpha=0.3, linestyle='--', which='both')

        ## 4. 调整布局
        plt.tight_layout()

        ## 5. 显示图表
        plt.show()


    
    
    


    #---------------------------------------------------------------------------
    #--------------------- 自旋关联函数的数据绝对误差可视化 --------------------
    #---------------------------------------------------------------------------
    @staticmethod
    def Spin_Corr_Visualization_difference(
        corr_dict: dict,
    ):
        """
        将输入的自旋关联函数字典中的前两组数据进行可视化，并绘制两组数据之间的绝对误差。

        Input:
        :corr_dict: 两种结构——
            1) 旧格式 dict[str, list[CorrelationRecord]]：
               key  -> 子图标题，value -> 对应的自旋关联函数对象列表
               （如 {'UltraDMRG': records_1, 'Itensor': records_2}），
               画 1 行 3 列（组1 / 组2 / 绝对误差）
            2) 嵌套格式 dict[str, dict[str, list[CorrelationRecord]]]：
               外层 key -> 组名，内层 key -> 行标签（如 A/B 子格），
               画 n 行 3 列（行 = 行标签，列 = 组1 / 组2 / 绝对误差），
               行标签与源点标在第一列 y 轴（如 'A sublattice (ref-site:49)'）
        """
        ### 检查输入数据并获取字典中的前两组数据
        ## 至少需要两组数据，才能计算绝对误差
        if len(corr_dict) < 2:
            raise ValueError(
                "corr_dict 至少需要包含两组自旋关联函数数据，才能计算绝对误差"
            )

        ## 获取字典中的前两组数据
        # 注：Python 3.7+ 中字典保持插入顺序，因此这里按照输入顺序取前两组
        titles = list(corr_dict.keys())
        title_1, title_2 = titles[0], titles[1]
        val_1, val_2 = corr_dict[title_1], corr_dict[title_2]
        nested = isinstance(val_1, dict)

        if nested:
            ### ---- 嵌套格式：n 行 × 3 列 ----
            if not isinstance(val_2, dict):
                raise ValueError("两组数据的输入结构不一致（一组为嵌套字典、"
                                 "另一组为列表）")
            if set(val_1) != set(val_2):
                raise ValueError("两组数据的行标签集合不一致")
            labels = list(val_1.keys())
            n_rows = len(labels)
            fig, axs = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
            axs = axs.reshape(n_rows, 3)
            fig.suptitle(
                r'Spin-Spin Correlation Function '
                r'$\langle S_i \cdot S_j \rangle$',
                fontsize=20, fontweight='bold', y=1.0)
            row_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']

            for i, label in enumerate(labels):
                record_list_1 = val_1[label]
                record_list_2 = val_2[label]
                if len(record_list_1) != len(record_list_2):
                    raise ValueError(
                        f"行 {label!r} 的两个自旋关联函数列表长度不一致")
                source_point = record_list_1[0].source_point
                if source_point != record_list_2[0].source_point:
                    raise ValueError(
                        f"行 {label!r} 的两个自旋关联函数源点并不相同")

                # 按距离排序后再连线（首尾不相连，折线不闭合）
                record_list_1 = sorted(record_list_1,
                                       key=lambda e: e.distance)
                record_list_2 = sorted(record_list_2,
                                       key=lambda e: e.distance)
                x_data = [e.distance for e in record_list_1]
                y1_data = [e.correlation for e in record_list_1]
                y2_data = [e.correlation for e in record_list_2]
                y3_data = [abs(a - b) for a, b in zip(y1_data, y2_data)]
                c = row_colors[i % len(row_colors)]
                row_title = f'{label} (ref-site:{source_point})'

                ## 组1数据的可视化
                axs[i, 0].scatter(x_data, y1_data, color=c, s=60, zorder=3)
                axs[i, 0].plot(x_data, y1_data, color='red', alpha=0.5,
                               linewidth=2)
                axs[i, 0].set_title(title_1, fontsize=16)
                axs[i, 0].set_xlabel('distance r', fontsize=14)
                axs[i, 0].set_ylabel(
                    f'{row_title}\n$\\langle S_i \\cdot S_j \\rangle$',
                    fontsize=14)
                axs[i, 0].grid(True, alpha=0.3, linestyle='--')

                ## 组2数据的可视化
                axs[i, 1].scatter(x_data, y2_data, color=c, s=60, zorder=3)
                axs[i, 1].plot(x_data, y2_data, color='orange', alpha=0.5,
                               linewidth=2)
                axs[i, 1].set_title(title_2, fontsize=16)
                axs[i, 1].set_xlabel('distance r', fontsize=14)
                axs[i, 1].set_ylabel(f'$\\langle S_i \\cdot S_j \\rangle$',
                                     fontsize=14)
                axs[i, 1].grid(True, alpha=0.3, linestyle='--')

                ## 组1与组2之差的数据可视化
                axs[i, 2].scatter(x_data, y3_data, color=c, s=60, zorder=3)
                axs[i, 2].plot(x_data, y3_data, color='brown', alpha=0.5,
                               linewidth=2)
                axs[i, 2].set_title('Absolute Error of Spin-Spin Correlation',
                                    fontsize=16)
                axs[i, 2].set_xlabel('distance r', fontsize=14)
                axs[i, 2].set_ylabel('Absolute Error', fontsize=14)
                axs[i, 2].grid(True, alpha=0.3, linestyle='--')
        else:
            ### ---- 旧格式：1 行 × 3 列 ----
            record_list_1 = val_1
            record_list_2 = val_2

            ### 检查两个自旋关联函数对象列表的长度是否一致
            if len(record_list_1) != len(record_list_2):
                raise ValueError("输入的两个自旋关联函数对象列表的列表长度不一致")

            ### 获取自旋关联函数的源点
            source_point = record_list_1[0].source_point
            if source_point != record_list_2[0].source_point:
                raise ValueError("两个自旋关联函数的源点并不相同")

            ### 将数据存储到x轴和y轴列表中
            # 按距离排序后再连线（首尾不相连，折线不闭合）
            record_list_1 = sorted(record_list_1, key=lambda e: e.distance)
            record_list_2 = sorted(record_list_2, key=lambda e: e.distance)
            x_data = [e.distance for e in record_list_1]
            y1_data = [e.correlation for e in record_list_1]
            y2_data = [e.correlation for e in record_list_2]
            y3_data = [abs(a - b) for a, b in zip(y1_data, y2_data)]

            ### 可视化
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))

            ## 添加总标题（源点子格名按测量序列约定）
            sub = _sublattice_suffix(source_point)
            ref_tag = (f'{sub} sublattice (ref-site:{source_point})'
                       if sub else f'(ref-site:{source_point})')
            fig.suptitle(
                f'Spin-Spin Correlation Function $\\langle S_i \\cdot S_j '
                f'\\rangle$ {ref_tag}',
                fontsize=20, fontweight='bold', y=1.0)

            ## record_list_1数据的可视化
            axs[0].scatter(x_data, y1_data, color='blue', s=60, zorder=3)
            axs[0].plot(x_data, y1_data, color='red', alpha=0.5, linewidth=2)
            axs[0].set_title(title_1, fontsize=16)
            axs[0].set_xlabel('distance r', fontsize=14)
            axs[0].set_ylabel(f'$\\langle S_i \\cdot S_j \\rangle$', fontsize=14)
            axs[0].grid(True, alpha=0.3, linestyle='--')

            ## record_list_2数据的可视化
            axs[1].scatter(x_data, y2_data, color='green', s=60, zorder=3)
            axs[1].plot(x_data, y2_data, color='orange', alpha=0.5, linewidth=2)
            axs[1].set_title(title_2, fontsize=16)
            axs[1].set_xlabel('distance r', fontsize=14)
            axs[1].set_ylabel(f'$\\langle S_i \\cdot S_j \\rangle$', fontsize=14)
            axs[1].grid(True, alpha=0.3, linestyle='--')

            ## record_list_1与record_list_2之差的数据可视化
            axs[2].scatter(x_data, y3_data, color='purple', s=60, zorder=3)
            axs[2].plot(x_data, y3_data, color='brown', alpha=0.5, linewidth=2)
            axs[2].set_title('Absolute Error of Spin-Spin Correlation',
                             fontsize=16)
            axs[2].set_xlabel('distance r', fontsize=14)
            axs[2].set_ylabel('Absolute Error', fontsize=14)
            axs[2].grid(True, alpha=0.3, linestyle='--')

        ## 调整布局
        plt.tight_layout()

        ## 显示图表
        plt.show()



