import numpy as np
import matplotlib.pyplot as plt
from ..data_records.Physical_Data_Record import SuperCorrRecord


def _refbond_direction(i: int, j: int, ly: int = 4) -> str:
    """XC 圆柱构型最近邻键方向分类（对应 DMRGtJModel
    honeycomb_lattice.h 的 HoneycombXCCylinder::ClassifyNearestNeighbor，
    Ly=4）：同列（x 相同）竖直键按 (y_min + x) % 2 分 A/B，同行水平键为 C。

    验证：N200 数据 refbond 49_50→A、50_51→B、50_54→C。
    """
    if i > j:
        i, j = j, i
    x0, y0 = divmod(i, ly)
    x1, y1 = divmod(j, ly)
    if x0 == x1:
        y_min = y0 if (y0 + ly - y1) % ly == 1 else y1
        return 'A' if (y_min + x0) % 2 == 0 else 'B'
    return 'C'


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
    def Super_Corr_Visualization(record_list: list[SuperCorrRecord], label):
        """
        将输入的超导关联函数对象列表元素进行可视化
        :param record_list: 超导关联函数对象列表。
        :label: 图例，用于表示超导关联函数的哪个键方向(固定格式: "A key (blue bond)")
        """
        ## 1. 提取出超导关联函数的reference bond(i, j)与键方向
        ref_bond = record_list[0].ref_bond
        bond_key = label.split()[0]     # 键方向名（如 'A'）；颜色解析仍依赖原始 label 格式

        ## 2. 将数据存储到x轴和y轴列表中
        x_data = [] # 存储两格点之间的距离数据
        y_data = [] # 存储在x_data的距离下对应的超导关联函数数据
        for element in record_list:
            x_data.append(element.distance)
            y_data.append(element.sup_corr)

        ## 3. 创建1行3列的子图布局
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        ## 4. 添加总标题和颜色
        fig.suptitle(f'Superconducting Correlation Function $\\langle \\Delta_{{ij}}^\\dagger \\Delta_{{kl}} \\rangle$ (reference {bond_key}: {ref_bond})',
                     fontsize=20, fontweight='bold', y=1.0)
        color = label.split("(")[1].split(")")[0].split()[0]
        
        ## 5. 线性坐标图（原始）
        axs[0].scatter(x_data, y_data, color=color, s=60, label=f'target {bond_key}', zorder=3)
        axs[0].plot(x_data, y_data, color='red', alpha=0.5, linewidth=2)
        axs[0].set_title(f'Linear Scale', fontsize=16)
        axs[0].set_xlabel('distance r', fontsize=14)
        axs[0].set_ylabel(f'$\\langle \\Delta_ij^\\dagger \\Delta_kl \\rangle$', fontsize=14)
        axs[0].axhline(y=0, color='black', linewidth=3.5, linestyle='-', alpha=0.8, zorder=1)
        axs[0].grid(True, alpha=0.3, linestyle='--')
        axs[0].legend(fontsize=14, loc='best')

        ## 6. 半对数坐标图（y轴对数）- 用于判断指数衰减
        # 注意：如果y_data有负值，对数坐标会有问题，这里取绝对值
        y_abs = np.abs(y_data)
        axs[1].scatter(x_data, y_abs, color=color, s=60, label=f'target {bond_key}', zorder=3)
        axs[1].plot(x_data, y_abs, color='orange', alpha=0.5, linewidth=2)
        axs[1].set_yscale('log')
        axs[1].set_title(f'Log-Linear Scale (y-log)\nExponential decay check', fontsize=16)
        axs[1].set_xlabel('distance r', fontsize=14)
        axs[1].set_ylabel(f'$|\\langle \\Delta_ij^\\dagger \\Delta_kl \\rangle|$ (log scale)', fontsize=14)
        axs[1].grid(True, alpha=0.3, linestyle='--', which='both')
        axs[1].legend(fontsize=14, loc='best')

        ## 7. 双对数坐标图 - 用于判断幂律衰减
        axs[2].scatter(x_data, y_abs, color=color, s=60, label=f'target {bond_key}', zorder=3)
        axs[2].plot(x_data, y_abs, color='brown', alpha=0.5, linewidth=2)
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_title(f'Log-Log Scale\nPower-law decay check', fontsize=16)
        axs[2].set_xlabel('distance r (log scale)', fontsize=14)
        axs[2].set_ylabel(f'$|\\langle \\Delta_ij^\\dagger \\Delta_kl \\rangle|$ (log scale)', fontsize=14)
        axs[2].grid(True, alpha=0.3, linestyle='--', which='both')
        axs[2].legend(fontsize=14, loc='best')

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

        ## 1. 获取reference bond（取第一组第一条）与键方向
        first_key = next(iter(records_dict)) # 获取字典的第一个键(字典是无序的，但在Python 3.7+中插入顺序保留，这里取第一个插入的键)
        ref_bond = records_dict[first_key][0].ref_bond
        ref_key = first_key.split()[0]

        ## 2. 给三种键方向准备相应的颜色与形状进行区分
        colors = ['blue', 'green', 'orange']
        markers = ['o', 's', 'D']

        ## 3. 创建子图
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f'Superconducting Correlation Function $\\langle \\Delta_{{ij}}^\\dagger \\Delta_{{kl}} \\rangle$ (reference {ref_key}: {ref_bond})',
            fontsize=20, fontweight='bold', y=1.0
        )

        ## 4. 遍历每一组数据并作图
        for idx, (label, rec_list) in enumerate(records_dict.items()):
            x_data = [r.distance for r in rec_list]
            y_data = [r.sup_corr for r in rec_list]
            y_abs = np.abs(y_data)

            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]
            legend_label = f'target {label.split()[0]}'

            # 4.1 --- 线性图 --- (原始数据图)
            axs[0].scatter(x_data, y_data, color=c, marker=m, s=60, label=legend_label, zorder=3)
            axs[0].plot(x_data, y_data, color=c, alpha=0.5, linewidth=2)

            # 4.2 --- 半对数图（y轴对数）--- (用于判断指数衰减)
            axs[1].scatter(x_data, y_abs, color=c, marker=m, s=60, label=legend_label, zorder=3)
            axs[1].plot(x_data, y_abs, color=c, alpha=0.5, linewidth=2)

            # 4.3 --- 双对数图 --- (用于判断幂律衰减)
            axs[2].scatter(x_data, y_abs, color=c, marker=m, s=60, label=legend_label, zorder=3)
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
        super_corr_dict: dict[str, dict[str, list[SuperCorrRecord]]]
    ):
        """
        将输入的超导关联函数字典中的前两组数据进行可视化，并绘制两组数据之间的绝对误差。

        子图布局：行 = 键方向（内层字典的键，如 A/B/C 三键 → 3 行），
        列 = 组1 / 组2 / 绝对误差。行标签 'target A' 等标在第一列 y 轴。

        Input:
        :super_corr_dict: dict[str, dict[str, list[SuperCorrRecord]]]
            外层字典的结构：
                    key   -> 子图标题
                    value -> 对应的超导关联函数字典

            例如：{'UltraDMRG': records_dict_1, 'Itensor': records_dict_2}
                其中 records_dict_1 和 records_dict_2 的结构为：
                    dict[str, list[SuperCorrRecord]]
        """
        ### 检查输入数据并获取外层字典中的前两组数据
        ## 至少需要两组数据，才能计算绝对误差
        if len(super_corr_dict) < 2:
            raise ValueError(
                "super_corr_dict 至少需要包含两组超导关联函数数据，才能计算绝对误差"
            )

        ## 获取外层字典中的前两组数据
        # 注：Python 3.7+ 中字典保持插入顺序，因此这里按照输入顺序取前两组
        titles = list(super_corr_dict.keys())

        title_1 = titles[0]
        title_2 = titles[1]

        records_dict_1 = super_corr_dict[title_1]
        records_dict_2 = super_corr_dict[title_2]


        ### 获取reference bond（取第一组第一条）与键方向
        first_key_1 = next(iter(records_dict_1)) # 获取字典的第一个键(字典是无序的，但在Python 3.7+中插入顺序保留，这里取第一个插入的键)
        ref_bond = records_dict_1[first_key_1][0].ref_bond
        ref_key = first_key_1.split()[0]

        ### 行 = 键方向（内层字典的键），列 = 组1 / 组2 / 绝对误差
        labels = list(records_dict_1.keys())
        if set(labels) != set(records_dict_2.keys()):
            raise ValueError("两个超导关联函数字典的键方向集合不一致")
        n_rows = len(labels)
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        markers = ['o', 's', 'D', '^', 'v', '*']

        ### 创建子图（n 行 × 3 列）
        fig, axs = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
        axs = axs.reshape(n_rows, 3)
        fig.suptitle(
            f'Superconducting Correlation Function $\\langle \\Delta_{{ij}}^\\dagger \\Delta_{{kl}} \\rangle$ (reference {ref_key}: {ref_bond})',
            fontsize=20, fontweight='bold', y=1.0
        )

        ### 逐键方向作图（一行一个键方向）
        for idx, label in enumerate(labels):
            rec_list = records_dict_1[label]
            ### 检查两个超导关联函数对象列表的长度是否一致
            if len(rec_list) != len(records_dict_2[label]):
                raise ValueError(f"输入的两个超导关联函数字典在键({label})对应的数据列表的长度不一致")

            ## 输入x轴与y轴数据
            x_data = [r.distance for r in rec_list]
            y_data_1 = [r.sup_corr for r in rec_list]
            y_data_2 = [r.sup_corr for r in records_dict_2[label]]
            y_data_3 = [abs(a - b) for a, b in zip(y_data_1, y_data_2)]

            ## 输入颜色与数据形状
            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]
            row_tag = f'target {label.split()[0]}'

            # record_dict_1数据的可视化
            axs[idx, 0].scatter(x_data, y_data_1, color=c, marker=m, s=60,
                                zorder=3)
            axs[idx, 0].plot(x_data, y_data_1, color=c, alpha=0.5, linewidth=2)
            axs[idx, 0].set_title(title_1, fontsize=16)
            axs[idx, 0].set_xlabel('distance r', fontsize=14)
            axs[idx, 0].set_ylabel(
                f'{row_tag}\n$\\langle \\Delta^\\dagger \\Delta \\rangle$',
                fontsize=14)

            # record_dict_2数据的可视化
            axs[idx, 1].scatter(x_data, y_data_2, color=c, marker=m, s=60,
                                zorder=3)
            axs[idx, 1].plot(x_data, y_data_2, color=c, alpha=0.5, linewidth=2)
            axs[idx, 1].set_title(title_2, fontsize=16)
            axs[idx, 1].set_xlabel('distance r', fontsize=14)
            axs[idx, 1].set_ylabel(
                '$\\langle \\Delta^\\dagger \\Delta \\rangle$', fontsize=14)

            # record_dict_1与record_dict_2之差的数据可视化
            axs[idx, 2].scatter(x_data, y_data_3, color=c, marker=m, s=60,
                                zorder=3)
            axs[idx, 2].plot(x_data, y_data_3, color=c, alpha=0.5, linewidth=2)
            axs[idx, 2].set_title('Absolute Error of Superconducting '
                                  'Correlation', fontsize=16)
            axs[idx, 2].set_xlabel('distance r', fontsize=14)
            axs[idx, 2].set_ylabel('Absolute Error', fontsize=14)

        ### 设置各子图公共属性
        for ax in axs.flat:
            ax.axhline(y=0, color='black', linewidth=3.5, linestyle='-',
                       alpha=0.8, zorder=1)
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

    #---------------------------------------------------------------------------------------
    #------------ 超导关联函数绝对误差可视化（行 = reference bond 键方向） ----------------
    #---------------------------------------------------------------------------------------
    @staticmethod
    def Super_Corr_Visualization_difference_refbond(
        super_corr_dict: dict[str, dict[str, dict[str, list[SuperCorrRecord]]]]
    ):
        """
        收敛诊断用两组数据绝对误差对比图：行 = reference bond（键方向），
        列 = 组1 / 组2 / 绝对误差，每个子图内该 refbond 的 target A/B/C
        三键同图（颜色/标记区分 + 图例 target X）。

        Input:
        :super_corr_dict: 外层 key -> 组名（如 'D = 1500'，用作列标题），
            value -> {ref_tag: {label: list[SuperCorrRecord]}}；
            ref_tag 形如 'reference A: (49, 50)'（调用方用
            _refbond_direction 构造）。取外层前两组。
        """
        if len(super_corr_dict) < 2:
            raise ValueError(
                "super_corr_dict 至少需要包含两组超导关联函数数据，"
                "才能计算绝对误差")
        titles = list(super_corr_dict.keys())
        title_1, title_2 = titles[0], titles[1]
        refs_1 = super_corr_dict[title_1]
        refs_2 = super_corr_dict[title_2]
        if set(refs_1) != set(refs_2):
            raise ValueError("两组数据的 reference bond 集合不一致")

        ref_tags = list(refs_1.keys())
        n_rows = len(ref_tags)
        colors = ['blue', 'green', 'orange']
        markers = ['o', 's', 'D']

        ### 创建子图（n 个 refbond 行 × 3 列）
        fig, axs = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
        axs = axs.reshape(n_rows, 3)
        fig.suptitle(
            r'Superconducting Correlation Function '
            r'$\langle \Delta_{ij}^\dagger \Delta_{kl} \rangle$',
            fontsize=20, fontweight='bold', y=1.0)

        ### 逐 refbond 行作图，行内逐 target 键（同格内多键叠加）
        for idx, ref_tag in enumerate(ref_tags):
            d1, d2 = refs_1[ref_tag], refs_2[ref_tag]
            if set(d1) != set(d2):
                raise ValueError(
                    f"reference {ref_tag} 的两组键方向集合不一致")
            for k, label in enumerate(d1):
                rec_list = sorted(d1[label], key=lambda r: r.distance)
                rec_list_2 = sorted(d2[label], key=lambda r: r.distance)
                if len(rec_list) != len(rec_list_2):
                    raise ValueError(
                        f"reference {ref_tag} 键({label})的两组数据列表"
                        "长度不一致")
                x_data = [r.distance for r in rec_list]
                y1 = [r.sup_corr for r in rec_list]
                y2 = [r.sup_corr for r in rec_list_2]
                y3 = [abs(a - b) for a, b in zip(y1, y2)]
                c = colors[k % len(colors)]
                m = markers[k % len(markers)]
                legend_tag = f'target {label.split()[0]}'

                # 组1数据
                axs[idx, 0].scatter(x_data, y1, color=c, marker=m, s=60,
                                    label=legend_tag, zorder=3)
                axs[idx, 0].plot(x_data, y1, color=c, alpha=0.5, linewidth=2)
                # 组2数据
                axs[idx, 1].scatter(x_data, y2, color=c, marker=m, s=60,
                                    label=legend_tag, zorder=3)
                axs[idx, 1].plot(x_data, y2, color=c, alpha=0.5, linewidth=2)
                # 绝对误差
                axs[idx, 2].scatter(x_data, y3, color=c, marker=m, s=60,
                                    label=legend_tag, zorder=3)
                axs[idx, 2].plot(x_data, y3, color=c, alpha=0.5, linewidth=2)

            axs[idx, 0].set_title(title_1, fontsize=16)
            axs[idx, 1].set_title(title_2, fontsize=16)
            axs[idx, 2].set_title('Absolute Error of Superconducting '
                                  'Correlation', fontsize=16)
            axs[idx, 0].set_ylabel(
                f'{ref_tag}\n$\\langle \\Delta^\\dagger \\Delta \\rangle$',
                fontsize=14)
            axs[idx, 1].set_ylabel('$\\langle \\Delta^\\dagger \\Delta '
                                   '\\rangle$', fontsize=14)
            axs[idx, 2].set_ylabel('Absolute Error', fontsize=14)
            axs[idx, 0].legend(fontsize=12, loc='best')

        ### 设置各子图公共属性
        for ax in axs.flat:
            ax.set_xlabel('distance r', fontsize=14)
            ax.axhline(y=0, color='black', linewidth=3.5, linestyle='-',
                       alpha=0.8, zorder=1)
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()