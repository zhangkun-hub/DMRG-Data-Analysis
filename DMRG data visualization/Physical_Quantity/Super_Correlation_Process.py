import numpy as np
import matplotlib.pyplot as plt
from .Physical_Data_Record import SuperCorrRecord


class SuperCorrProcessor:
    #---------------------------------------------------------------------
    #----------------------- 超导关联函数的对称性检测 ----------------------
    #---------------------------------------------------------------------
    @staticmethod
    def symmetry_check(scs_a: list[SuperCorrRecord],
            scs_b: list[SuperCorrRecord],
            scs_c: list[SuperCorrRecord],
            scs_d: list[SuperCorrRecord]
    ) -> tuple[float, float]:
        """
            如果体系具有SU(2)对称性，那么将体系整体在自旋空间中旋转结果不变，而已知scs_a中所有算符
        自旋反转的结果为scs_d，scs_b中所有算符自旋反转的结果为scs_c，所以在SU(2)对称性下，有：
                scs_a = scs_d，scs_b = scs_c (注意是列表所有对应元素的值都要相等)

        误差经验判断：~1e-6：极好；~1e-4：可接受；>1e-3：要警惕（可能截断误差 / 未收敛）

        Input:
        :scs_a: 超导关联函数的通道a：scs_a == <c_i↑^†·c_j↓^†·c_k↑·c_l↓>
        :scs_b: 超导关联函数的通道b：scs_b == <c_i↓^†·c_j↑^†·c_k↑·c_l↓>
        :scs_c: 超导关联函数的通道c：scs_c == <c_i↑^†·c_j↓^†·c_k↓·c_l↑>
        :scs_d: 超导关联函数的通道d：scs_d == <c_i↓^†·c_j↑^†·c_k↓·c_l↑>

        Return:
        :err_ad: scs_a与scs_d的相对误差
        :err_bc: scs_b与scs_c的相对误差
        """
        ## 1.提取出四种超导关联函数通道的数值列表
        a_vals = [record.sup_corr for record in scs_a]
        b_vals = [record.sup_corr for record in scs_b]
        c_vals = [record.sup_corr for record in scs_c]
        d_vals = [record.sup_corr for record in scs_d]

        ## 2.计算相对误差
        err_ad = np.max(np.abs(np.array(a_vals) - np.array(d_vals))) / np.max(np.abs(a_vals))
        err_bc = np.max(np.abs(np.array(b_vals) - np.array(c_vals))) / np.max(np.abs(b_vals))

        return err_ad, err_bc
    


    #-------------------------------------------------------------------
    #----------------------- 超导关联函数的合并方法 ----------------------
    #-------------------------------------------------------------------
    @staticmethod
    def Combined_super_correlation(
            scs_a: list[SuperCorrRecord],
            scs_b: list[SuperCorrRecord],
            scs_c: list[SuperCorrRecord],
            scs_d: list[SuperCorrRecord]
    ) -> list[SuperCorrRecord]:
        """
        超导关联函数：<Δ_ij^†·Δ_kl> = 0.5 * (scs_a - scs_b - scs_c + scs_d)
        通过上述公式，将输入的四个超导关联函数通道的SuperCorrRecord对象列表进行合并

        Input:
        :scs_a: 超导关联函数的通道a：scs_a == <c_i↑^†·c_j↓^†·c_k↑·c_l↓>
        :scs_b: 超导关联函数的通道b：scs_b == <c_i↓^†·c_j↑^†·c_k↑·c_l↓>
        :scs_c: 超导关联函数的通道c：scs_c == <c_i↑^†·c_j↓^†·c_k↓·c_l↑>
        :scs_d: 超导关联函数的通道d：scs_d == <c_i↓^†·c_j↑^†·c_k↓·c_l↑>

        Return:
        :super_corr: 存储超导关联函数<Δ_ij^†·Δ_kl>的SuperCorrRecord对象列表
        """
        ### 1.检查四个超导关联函数通道长度是否一致
        if len(scs_a) != len(scs_b) or len(scs_a) != len(scs_c) or len(scs_a) != len(scs_d):
            raise ValueError("四个超导关联函数通道的列表长度存在不一致情形")

        ### 2.检查reference bond(i, j)是否完全相同
        scs_a_ij = scs_a[0].ref_bond
        scs_b_ij = scs_b[0].ref_bond
        scs_c_ij = scs_c[0].ref_bond
        scs_d_ij = scs_d[0].ref_bond
        if scs_a_ij != scs_b_ij or scs_a_ij != scs_c_ij or scs_a_ij != scs_d_ij:
            raise ValueError("四个超导关联函数通道的reference bond(i, j)存在不相同情形")

        ### 3.遍历列表所有元素进行判断与合并
        super_corr = []
        for m in range(len(scs_a)):
            ## 3.1 获取每个列表当前元素(是个SuperCorrRecord对象)
            scs_a_Record = scs_a[m]
            scs_b_Record = scs_b[m]
            scs_c_Record = scs_c[m]
            scs_d_Record = scs_d[m]

            ## 3.2 判断四个超导关联函数通道的target bond(k, l)是否一致
            if (scs_a_Record.target_bond != scs_b_Record.target_bond or
                scs_a_Record.target_bond != scs_c_Record.target_bond or
                scs_a_Record.target_bond != scs_d_Record.target_bond):
                raise ValueError(f"四个超导关联函数通道的第{m}个元素的target bond(k, l)存在不相同情形")


            ## 3.3 获取元素对象的各个属性
            target_bond_kl = scs_a_Record.target_bond # target bond(k, l)
            r = scs_a_Record.distance           # reference bond(i, j)与target bond(k, l)的距离
            scs_a_value = scs_a_Record.sup_corr    # <c_i​↑^†·c_j↓^†·c_k​↑·c_l↓>
            scs_b_value = scs_b_Record.sup_corr    # <c_i↓^†·c_j​↑^†·c_k​↑·c_l↓>
            scs_c_value = scs_c_Record.sup_corr    # <c_i​↑^†·c_j↓^†·c_k↓·c_l​↑>
            scs_d_value = scs_d_Record.sup_corr    # <c_i↓^†·c_j​↑^†·c_k↓·c_l​↑>

            ## 3.4 求出<Δ_ij^†·Δ_kl>的值，并生成SuperCorrRecord对象
            super_corr_value = 0.5 * (scs_a_value - scs_b_value - scs_c_value + scs_d_value)
            record = SuperCorrRecord(scs_a_ij, target_bond_kl, r, super_corr_value)

            ## 3.5 将record添加到super_corr列表中
            super_corr.append(record)

        return super_corr



    #-------------------------------------------------------------------------------
    #--------------------- 超导关联函数的数据可视化（一组数据） --------------------
    #-------------------------------------------------------------------------------
    @staticmethod
    def Super_Corr_Visualization(record_list: list[SuperCorrRecord]):
        """
        将输入的超导关联函数对象列表元素进行可视化
        :param record_list: 超导关联函数对象列表。
        """
        ## 1. 提取出超导关联函数的reference bond(i, j)
        ref_bond = record_list[0].ref_bond

        ## 2. 将数据存储到x轴和y轴列表中
        x_data = [] # 创建存储两格点之间的距离数据
        y_data = [] # 创建存储在x_data的距离下对应的超导关联函数数据
        for element in record_list:
            x_data.append(element.distance)
            y_data.append(element.sup_corr)

        ## 3. 创建1行3列的子图布局
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        ## 4. 添加总标题
        fig.suptitle(f'Superconducting Correlation Function $\\langle \\Delta_{{ij}}^\\dagger \\Delta_{{kl}} \\rangle$ (reference bond: {ref_bond})', 
                     fontsize=20, fontweight='bold', y=1.0)
        

        ## 5. 线性坐标图（原始）
        axs[0].scatter(x_data, y_data, color='blue', s=60, zorder=3)
        axs[0].plot(x_data, y_data, color='red', alpha=0.5, linewidth=2)
        axs[0].set_title(f'Linear Scale', fontsize=16)
        axs[0].set_xlabel('distance r', fontsize=14)
        axs[0].set_ylabel(f'$\\langle \\Delta_ij^\\dagger \\Delta_kl \\rangle$', fontsize=14)
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
        axs[1].set_ylabel(f'$|\\langle \\Delta_ij^\\dagger \\Delta_kl \\rangle|$ (log scale)', fontsize=14)
        axs[1].grid(True, alpha=0.3, linestyle='--', which='both')

        ## 7. 双对数坐标图 - 用于判断幂律衰减
        axs[2].scatter(x_data, y_abs, color='purple', s=60, zorder=3)
        axs[2].plot(x_data, y_abs, color='brown', alpha=0.5, linewidth=2)
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_title(f'Log-Log Scale\nPower-law decay check', fontsize=16)
        axs[2].set_xlabel('distance r (log scale)', fontsize=14)
        axs[2].set_ylabel(f'$|\\langle \\Delta_ij^\\dagger \\Delta_kl \\rangle|$ (log scale)', fontsize=14)
        axs[2].grid(True, alpha=0.3, linestyle='--', which='both')

        ## 8. 调整布局
        plt.tight_layout()

        ## 9. 显示图表
        plt.show()
    
    
    
    
    #---------------------------------------------------------------------------------------
    #--------------------- 超导关联函数的数据可视化（三个方向键的数据可视化） --------------------
    #---------------------------------------------------------------------------------------
    @staticmethod
    def Super_Corr_Visualization_three(
        records_dict: dict[str, list[SuperCorrRecord]],
    ):
        """
        支持三个键方向的超导关联函数的可视化。
        :param records_dict: 字典，key为标签（如 'A key (blue bond)'），value为对应的一组 SuperCorrRecord 列表。
        """
        if not records_dict:
            raise ValueError("records_dict cannot be empty")

        ## 1. 获取reference bond（取第一组第一条）
        first_key = next(iter(records_dict)) # 获取字典的第一个键(字典是无序的，但在Python 3.7+中插入顺序保留，这里取第一个插入的键)
        ref_bond = records_dict[first_key][0].ref_bond

        ## 2. 给三种键方向准备相应的颜色与形状进行区分
        colors = ['blue', 'green', 'orange']
        markers = ['o', 's', 'D']

        ## 3. 创建子图
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f'Superconducting Correlation Function $\\langle \\Delta_{{ij}}^\\dagger \\Delta_{{kl}} \\rangle$ (reference bond: {ref_bond})',
            fontsize=20, fontweight='bold', y=1.0
        )

        ## 4. 遍历每一组数据并作图
        for idx, (label, rec_list) in enumerate(records_dict.items()):
            x_data = [r.distance for r in rec_list]
            y_data = [r.sup_corr for r in rec_list]
            y_abs = np.abs(y_data)

            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]

            # 4.1 --- 线性图 --- (原始数据图)
            axs[0].scatter(x_data, y_data, color=c, marker=m, s=60, label=label, zorder=3)
            axs[0].plot(x_data, y_data, color=c, alpha=0.5, linewidth=2)

            # 4.2 --- 半对数图（y轴对数）--- (用于判断指数衰减)
            axs[1].scatter(x_data, y_abs, color=c, marker=m, s=60, label=label, zorder=3)
            axs[1].plot(x_data, y_abs, color=c, alpha=0.5, linewidth=2)

            # 4.3 --- 双对数图 --- (用于判断幂律衰减)
            axs[2].scatter(x_data, y_abs, color=c, marker=m, s=60, label=label, zorder=3)
            axs[2].plot(x_data, y_abs, color=c, alpha=0.5, linewidth=2)

        ## 5. 设置各子图属性
        titles = [
            'Linear Scale',
            'Log-Linear Scale (y-log)\nExponential decay check',
            'Log-Log Scale\nPower-law decay check'
        ]
        ylabels = [
            '$\\langle \\Delta^\\dagger \\Delta \\rangle$',
            '$|\\langle \\Delta^\\dagger \\Delta \\rangle|$ (log scale)',
            '$|\\langle \\Delta^\\dagger \\Delta \\rangle|$ (log scale)'
        ]

        for i, ax in enumerate(axs):
            ax.set_title(titles[i], fontsize=16)
            ax.set_xlabel('distance r', fontsize=14)
            ax.set_ylabel(ylabels[i], fontsize=14)
            ax.axhline(y=0, color='black', linewidth=3.5, linestyle='-', alpha=0.8, zorder=1)
            ax.grid(True, alpha=0.3, linestyle='--')

            # 对于i ≥ 1的子图（即第二个和第三个），设置 y 轴为对数刻度
            if i >= 1:
                ax.set_yscale('log')
            
            # 对于i == 2的子图（第三个），再额外设置 x 轴为对数刻度
            if i == 2:
                ax.set_xscale('log')

            ax.legend(fontsize=14, loc='best')

        plt.tight_layout()
        plt.show()

        
        
        
        
    #-------------------------------------------------------------------------------
    #---------------------- 超导关联函数的数据的绝对误差可视化 ---------------------
    #-------------------------------------------------------------------------------
    @staticmethod
    def Super_Corr_Visualization_difference(
        records_dict_1: dict[str, list[SuperCorrRecord]],
        records_dict_2: dict[str, list[SuperCorrRecord]]
    ):
        """
        将输入的两组超导关联函数字典里的各组元素进行绝对误差的可视化
        """
        ### 2. 获取reference bond（取第一组第一条）
        first_key_1 = next(iter(records_dict_1)) # 获取字典的第一个键(字典是无序的，但在Python 3.7+中插入顺序保留，这里取第一个插入的键)
        ref_bond = records_dict_1[first_key_1][0].ref_bond

        ### 2. 给三种键方向准备相应的颜色与形状进行区分
        colors = ['blue', 'green', 'orange']
        markers = ['o', 's', 'D']

        ### 3. 创建子图
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f'Superconducting Correlation Function $\\langle \\Delta_{{ij}}^\\dagger \\Delta_{{kl}} \\rangle$ (reference bond: {ref_bond})',
            fontsize=20, fontweight='bold', y=1.0
        )

        
        ### 4. 遍历每一组数据并作图
        for idx, (label, rec_list) in enumerate(records_dict_1.items()):
            ### 4.1 检查两个超导关联函数对象列表的长度是否一致
            if len(rec_list) != len(records_dict_2[label]):
                raise ValueError(f"输入的两个超导关联函数字典在键({label})对应的数据列表的长度不一致")
            
            ## 4.2 输入x轴与y轴数据
            x_data = [r.distance for r in rec_list]
            y_data_1 = [r.sup_corr for r in rec_list]
            y_data_2 = [r.sup_corr for r in records_dict_2[label]]
            y_data_3 = [abs(a - b) for a, b in zip(y_data_1, y_data_2)]

            ## 4.3 输入颜色与数据形状
            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]

            # 4.4 record_dict_1数据的可视化
            axs[0].scatter(x_data, y_data_1, color=c, marker=m, s=60, label=label, zorder=3)
            axs[0].plot(x_data, y_data_1, color=c, alpha=0.5, linewidth=2)
            
            # 4.5 record_dict_2数据的可视化
            axs[1].scatter(x_data, y_data_2, color=c, marker=m, s=60, label=label, zorder=3)
            axs[1].plot(x_data, y_data_2, color=c, alpha=0.5, linewidth=2)

            # 4.6 record_dict_1与record_dict_2之差的数据可视化
            axs[2].scatter(x_data, y_data_3, color=c, marker=m, s=60, label=label, zorder=3)
            axs[2].plot(x_data, y_data_3, color=c, alpha=0.5, linewidth=2)

            
        ### 5. 设置各子图属性
        titles = [
            'UltraDMRG',
            'Itensor',
            'Absolute Error of Superconducting Correlation'
        ]
        ylabels = [
            '$\\langle \\Delta^\\dagger \\Delta \\rangle$',
            '$\\langle \\Delta^\\dagger \\Delta \\rangle$',
            'Absolute Error'
        ]

        for i, ax in enumerate(axs):
            ax.set_title(titles[i], fontsize=16)
            ax.set_xlabel('distance r', fontsize=14)
            ax.set_ylabel(ylabels[i], fontsize=14)
            ax.axhline(y=0, color='black', linewidth=3.5, linestyle='-', alpha=0.8, zorder=1)
            ax.grid(True, alpha=0.3, linestyle='--')

        axs[0].legend(fontsize=14, loc='best')
        axs[1].legend(fontsize=14, loc='best')
        plt.tight_layout()
        plt.show()
    