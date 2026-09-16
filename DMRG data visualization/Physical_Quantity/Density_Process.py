import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .Physical_Data_Record import DensityRecord
from .Honeycomb_Lattice import HoneycombXCGeometry


#-----------------------------------------------------------------------------------
# honeycomb格点的电子密度n的数据处理器
#----------------------------------------------------------------------------------- 
class DensityProcessor: 
    #------------------------------------------------------------------- 
    #------------------- 电子密度的一维折线图可视化 --------------------
    #------------------------------------------------------------------- 
    @staticmethod
    def Line_Chart_Visualization(record_list: list[DensityRecord]):
        """
        电子密度的一维折线图可视化
        :param record_list: 格点电子密度对象列表。
        """
        ## 将数据存储到y轴列表中
        y0_data = []    # 创建存储水平起始点为0的水平行的格点密度数据
        y1_data = []    # 创建存储水平起始点为1的水平行的格点密度数据
        y2_data = []    # 创建存储水平起始点为2的水平行的格点密度数据
        y3_data = []    # 创建存储水平起始点为3的水平行的格点密度数据
        
        for element in record_list:
            if element.horizontal_starting_point == 0:
                y0_data.append(element.density)
            elif element.horizontal_starting_point == 1:
                y1_data.append(element.density)
            elif element.horizontal_starting_point == 2:
                y2_data.append(element.density)
            else:
                y3_data.append(element.density)
                
        ## 判断四组数据的长度是否完全一致
        # 判断原理：通过集合set的自动去重功能，可以判断如果集合长度不为1，说明元素不完全相同
        lengths = [len(y0_data), len(y1_data), len(y2_data), len(y3_data)]
        if len(set(lengths)) != 1:      
            raise ValueError(f"数据长度不一致: {lengths}") 

        ## 绘制折线图
        plt.figure(figsize=(10, 6))  # 设置图形大小
            
        ## 创建x轴数据（索引）
        max_len = max(len(y0_data), len(y1_data), len(y2_data), len(y3_data))
        x = range(len(y0_data))

        ## 绘制四条折线
        plt.plot(x, y0_data, label='Horizontal Start 0', marker='o', linestyle='-', linewidth=2)
        plt.plot(x, y1_data, label='Horizontal Start 1', marker='s', linestyle='--', linewidth=2)
        plt.plot(x, y2_data, label='Horizontal Start 2', marker='^', linestyle='-.', linewidth=2)
        plt.plot(x, y3_data, label='Horizontal Start 3', marker='d', linestyle=':', linewidth=2)
        # 注：marker(数据点标记): o圆形，s方形，^三角形，d菱形

        ## 添加标题和标签
        plt.title('Electron Density Distribution in Honeycomb Lattice', fontsize=20, fontweight='bold')
        plt.xlabel('Index', fontsize=16)
        plt.ylabel('Density $\\langle n \\rangle$', fontsize=16)

        ## 添加图例
        plt.legend(loc='best', fontsize=14)

        ## 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')

        ## 自动调整布局
        plt.tight_layout()

        ## 显示图形
        plt.show()

        
        
    
    #----------------------------------------------------------------------------- 
    #------------------- 电子密度的绝对误差的一维折线图可视化 --------------------
    #----------------------------------------------------------------------------- 
    @staticmethod
    def Line_Chart_Visualization_difference(
        density_dict: dict[str, list[DensityRecord]]
    ):
        """
        将输入的电子密度字典中的前两组数据进行可视化，并绘制两组数据之间的绝对误差的一维折线图可视化。

        Input:
        :density_dict: dict[str, list[DensityRecord]]
            字典的结构：
                    key   -> 子图标题
                    value -> 对应的自旋关联函数对象列表

            例如：{'UltraDMRG': records_list_1, 'Itensor': records_list_2}
        """
        # -----------------------------------------------------------------------
        # ---------------------------- 检查输入数据 ------------------------------
        ## 检查输入数据并获取字典中的前两组数据
        # 至少需要两组数据，才能计算绝对误差
        if len(density_dict) < 2:
            raise ValueError(
                "density_dict 至少需要包含两组电子密度数据，才能计算绝对误差"
            )

        # 获取字典中的前两组数据
        # 注：Python 3.7+ 中字典保持插入顺序，因此这里按照输入顺序取前两组
        titles = list(density_dict.keys())

        title_1 = titles[0]
        title_2 = titles[1]

        record_list_1 = density_dict[title_1]
        record_list_2 = density_dict[title_2]

        ## 检查两个密度对象长度是否一致
        if len(record_list_1) != len(record_list_2):
            raise ValueError("输入的两个密度对象列表的列表长度不一致")


        # ------------------------------------------------------------------------
        # ----------------------- 按水平起点分类数据并检查 ------------------------- 
        ## 创建四组数据：
        # index = 0, 1, 2, 3 分别对应 horizontal_starting_point
        data_1 = [[], [], [], []]           # 记录record_list_1的数据
        data_2 = [[], [], [], []]           # 记录record_list_1的数据
        data_difference = [[], [], [], []]  # 记录data1与data2的绝对误差

        ## 根据 horizontal_starting_point 将数据分别存入四组列表
        for i, element in enumerate(record_list_1):
            start_p = element.horizontal_starting_point

            ## 检查 horizontal_starting_point 是否在 0~3 范围内
            if start_p not in range(4):
                raise ValueError(f"发现非法的 horizontal_starting_point: {start_p}")

            density_1 = element.density
            density_2 = record_list_2[i].density

            data_1[start_p].append(density_1)
            data_2[start_p].append(density_2)
            data_difference[start_p].append(abs(density_1 - density_2))

        ## 分别检查三组数据中的四个水平行长度是否一致
        lengths_1 = [len(data) for data in data_1]
        lengths_2 = [len(data) for data in data_2]
        lengths_difference = [len(data) for data in data_difference]

        if len(set(lengths_1)) != 1:
            raise ValueError(f"输入的record_list_1的四组数据长度不一致: {lengths_1}")

        if len(set(lengths_2)) != 1:
            raise ValueError(f"输入的record_list_2的四组数据长度不一致: {lengths_2}")

        if len(set(lengths_difference)) != 1:
            raise ValueError(f"两列表数据之差对应的四组数据长度不一致: {lengths_difference}")


        # --------------------------------------------------------------------
        # ------------------------- 准备绘图数据 ------------------------------
        ## x轴数据
        x = range(len(data_1[0]))

        ## 四个水平起点对应的标签
        labels = [
            'Horizontal Start 0',
            'Horizontal Start 1',
            'Horizontal Start 2',
            'Horizontal Start 3'
        ]

        ## 绘图参数
        colors = ['blue', 'orange', 'green', 'red']
        markers = ['o', 's', '^', 'd']
        linestyles = ['-', '--', '-', ':']

        ## 将三组数据与对应标题统一进行保存
        plot_data = [
            ("record_list_1", data_1, title_1),
            ("record_list_2", data_2, title_2),
            ("difference", data_difference, "Absolute Error of Electron Density")
        ]


        #------------------------------------------------------------------------
        #--------------------- 创建三个子图进行可视化 -----------------------------
        ## 创建1行3列的子图布局
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        ## 添加总标题
        fig.suptitle(
            'Electron Density Distribution in Honeycomb Lattice',
            fontsize=20,
            fontweight='bold',
            y=1.0
        )

        ### 给每个子图进行对应数据的可视化
        for ax, (_, data, title) in zip(axs, plot_data):
            ## 绘制四条水平行数据
            for i in range(4):
                ax.scatter(x, data[i], color=colors[i], s=60, zorder=3, label=labels[i])
                ax.plot(x, data[i], color=colors[i], marker=markers[i],
                    linestyle=linestyles[i], alpha=0.5, linewidth=2
                )

            ## 设置标题
            ax.set_title(title, fontsize=16)

            ## 设置x轴
            ax.set_xlabel('Index', fontsize=14)

            ## 设置y轴
            if title == "Absolute Error of Electron Density":
                ax.set_ylabel('Absolute Error', fontsize=14)
            else:
                ax.set_ylabel(r'Density $\langle n \rangle$', fontsize=14)

            ## 网格
            ax.grid(True, alpha=0.3, linestyle='--')

            ## 前两个子图显示图例
            if title != "Absolute Error of Electron Density":
                ax.legend(fontsize=14, loc='best')

        ## 调整布局
        plt.tight_layout()

        ## 显示图表
        plt.show()
    
    


        
        
    #------------------------------------------------------------------- 
    #-------------------- 电子密度的二维平面可视化 ---------------------
    #------------------------------------------------------------------- 
    @staticmethod
    def Two_Dimensional_Visualization(record_list: list[DensityRecord], N:int, a: float = 1.0):
        """
        电子密度的二维平面可视化
        :param record_list: 格点电子密度对象列表。
        :param N: 系统格点数。
        :param a: honeycomb晶格的晶格常数（默认为a = 1.0）。
        """
        ### 1.建立格点坐标、最近邻格点列表和密度列表
        honey_latt = HoneycombXCGeometry(N, a) # honeycomb晶格结构的对象honey_latt
        coords = honey_latt.coords  # 格点坐标
        lattice_nn_list = honey_latt._nearest_neighbor_grid_index() # 最近邻格点列表
        densities = []
        for element in record_list:
            densities.append(element.density)
            
        
        ### 2.创建图形
        fig, ax = plt.subplots(figsize=(50, 10)) # plt.subplots()：Matplotlib的核心函数，用于创建图形窗口fig(可以包含多个坐标轴)和坐标轴ax

        ### 3.绘制honeycomb晶格连接线
        for i, j in lattice_nn_list:     
            x1, y1 = coords[i] # 点1的横纵坐标
            x2, y2 = coords[j] # 点2的横纵坐标
            ax.plot([x1, x2], [y1, y2],  # 在坐标轴ax上绘制连接点1到点2的线段
                    color='gray',        # 灰色
                    linewidth=2,         # 线宽2点
                    alpha=0.7,           # 透明度
                    zorder=1)            # zorder的值为图层顺序，数值越大，绘制在越上层，此处1为在底层绘制

        ### 4.绘制格点
        node_colors = [] # 创建列表存储每个格点的颜色值

        ## 4.1 通过循环遍历所有格点来绘制
        for i in range(N): 
            x, y = coords[i] # 从坐标字典中获取当前格点的(x, y)坐标

            # 4.1.1 使用密度值映射颜色
            color_val = (densities[i] - min(densities)) / (max(densities) - min(densities)) # color_val计算原因：将密度值归一化到[0, 1]区间
            node_color = plt.cm.RdYlBu_r(color_val) # 使用颜色映射函数plt.cm.RdYlBu_r()获取颜色(_r表示反向),即color_val值从小到大对应颜色为蓝-黄-红
            node_colors.append(node_color) # 将当前格点的颜色添加到列表中

            # 4.1.2 绘制圆形格点
            circle = patches.Circle(
                (x, y),               # patches.Circle()用于在图表上创建一个圆形的图形元素，其中(x, y)代表圆心的坐标
                radius=0.12,           # 圆半径
                facecolor=node_color,     # 圆形填充颜色
                edgecolor='black',       # 圆形边框颜色选为black(黑色)
                linewidth=2,           # 圆形边框宽度
                zorder=3             # 绘制图层级数为3，确保格点绘制在连接线（zorder=1）之上，但低于文本标签（zorder=4）
            )
            ax.add_patch(circle) # 将圆形添加到坐标轴

            # 4.1.3 添加格点编号
            ax.text(x, y, str(i),        # ax.text(x,y,str())是 Matplotlib 中用于在图表位置(x, y)中添加str()括号里的文本的核心函数。
                    ha='center',         # 水平对齐位置(center为居中)
                    va='center',         # 垂直对齐位置(center为居中)
                    fontsize=15,         # 字体大小
                    fontweight='heavy',  # 字体粗细(heavy为很粗的字体)
                    color='black',       # 字体颜色
                    zorder=4)

            # 4.1.4 给每个格点添加密度值标签
            # 根据节点位置调整标签位置
            if i in [0, 4, 1, 5]:    # 中下方的4个格点
                label_y = y - 0.3    # 标签位置的纵坐标为：y - 0.3
            else:                    # 上方的4个格点
                label_y = y + 0.3    # 标签位置的纵坐标为：y + 0.3

            ax.text(x, label_y, f'{densities[i]:.5f}', 
                    ha='center', va='center', fontsize=15, 
                    bbox=dict(        # bbox=dict(...)：背景框样式,即为文本添加背景框
                        boxstyle="round,pad=0.2",   # 框样式，此处为圆角矩形，内边距0.2
                        facecolor="lightyellow",    # 背景框填充颜色，此处为浅黄色
                        edgecolor="gray",           # 边框颜色，此处为灰色
                        alpha=0.8),                 # 透明度
                    zorder=2)

        ## 4.2 设置坐标轴
        all_x = [coords[i][0] for i in range(N)]    # 获取所有格点的x坐标
        all_y = [coords[i][1] for i in range(N)]    # 获取所有格点的y坐标
        ax.set_xlim(min(all_x)-0.5, max(all_x)+0.5) # 设置x轴的显示范围
        ax.set_ylim(min(all_y)-0.7, max(all_y)+0.7) # 设置y轴的显示范围
        ax.set_aspect('equal') # set_aspect()函数用于设置坐标轴的纵横比（宽高比）,而'equal'为1:1比例(为了保持honeycomb形状)
        ax.axis('off') # 完全关闭坐标轴的显示

        ## 4.3 添加标题
        ax.set_title(f'Electron Density Distribution in Honeycomb Lattice', fontsize=25, pad=8) # pad：标题与图形的间距

        ## 4.4 创建颜色条
        sm = plt.cm.ScalarMappable(  # plt.cm.ScalarMappable()：创建可映射到颜色的对象，用于连接颜色映射和数值范围与生成颜色条提供数据
            cmap=plt.cm.RdYlBu_r,    # 表示从红(低值) → 黄(中值) → 蓝(高值)的颜色渐变
            norm=plt.Normalize(     # Normalizer(a,b)为归一化器,将处于[a,b]区间的原始数据值线性映射到[0, 1]区间,计算方法为上面变量color_val的计算
                min(densities), max(densities))
        )
        # set_array()步骤必须存在，要给ScalarMappable提供实际数据！！！
        sm.set_array([]) # 此处设置空数组是只需要颜色条的显示功能，不需要实际的数据映射，从而避免无意义的数据复制
        cbar = plt.colorbar(          # 创建颜色条
            sm,                       # ScalarMappable对象，用于提供颜色映射数据
            ax=ax,                    # 确定颜色条关联的坐标轴，并将颜色条显示在指定坐标轴旁
            orientation='vertical',   # 颜色条方向，此处'vertical'为垂直
            fraction=0.046,           # 颜色条粗细：占坐标轴高度的比例
            pad=0.04,                 # 间距：颜色条与坐标轴的间距
            location='left'           # 将颜色条放在左侧
        )
        cbar.ax.tick_params(    # cbar.ax.tick_params()：精细控制颜色条的刻度样式(其中cbar.ax是颜色条内部的坐标轴对象)
            axis='y',           # 设置y轴刻度(对应垂直颜色条)
            labelsize=15,       # 刻度字体大小
            length=6,           # 刻度线长度
            width=1,            # 刻度线宽度
            pad=8               # 标签与刻度线的距离
        )
        cbar.set_label('Density Value', fontsize=18, labelpad=15) # 设置颜色条标签，其中labelpad为标签与颜色条的距离

        plt.tight_layout()
        plt.show()
