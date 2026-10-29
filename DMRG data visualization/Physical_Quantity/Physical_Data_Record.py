import ast
import numpy as np
from .Honeycomb_Lattice import HoneycombXCGeometry



#-----------------------------------------------------------------------------------
# 数据记录器的抽象基类
#-----------------------------------------------------------------------------------
class Record:
    # 初始化方法
    def __init__(self):
        raise NotImplementedError("子类必须实现 __init__ 方法")  # NotImplementedError表示未实现错误    
    
    # 字符串方法__str__用于实现对象信息的输出
    def __str__(self):
        raise NotImplementedError("子类必须实现 __str__ 方法")  


    # 定义数据提取的类方法
    @classmethod
    def extract_data(cls, data: str, **kwargs):
        raise NotImplementedError("子类必须实现 extract_data 类方法")
   



#-----------------------------------------------------------------------------------
# honeycomb的每个格点密度n的数据记录器
#-----------------------------------------------------------------------------------  
class DensityRecord(Record):
    def __init__(self, lattice_point: int, coords: tuple, density: float, horizontal_starting_point: int):
        self.lattice_point = lattice_point                  # honeycomb晶格点i
        self.coords = coords                            # 格点i对应的坐标
        self.density = density                          # 格点i对应的电子密度
        self.horizontal_starting_point = horizontal_starting_point  # 格点i对应的水平起始点(分别为0，1，2，3)

        
    def __str__(self):
        return f"honeycomb的晶格点{self.lattice_point}, 坐标: {self.coords}, 对应的电子密度n = {self.density:.6f}, 其水平起始点为: {self.horizontal_starting_point}"
    
    
    
    # 数据提取的类方法(类方法作用：只需调用DensityRecord.extract_data(...)即可获得完整的DensityRecord对象)
    @classmethod
    def extract_data(cls, data: str, N:int, a: float = 1.0) -> list['DensityRecord']:
        """
        将输入的字符串数据解析并封装为 DensityRecord 对象列表。
        
        :param data: 计算的关联函数的数据(是列表构成的字符串，每个元素的格式为: [[i, j], [val, 0]])。
        :param N: 系统格点数。
        :param a: honeycomb晶格的晶格常数（默认为a = 1.0）。
                    
        :return record_list: 列表，存储关联函数<i·j>所封装好的DensityRecord对象。
        """
        ### 1.提取数据
        try:
            data_list = ast.literal_eval(data)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"输入数据格式错误，无法解析: {e}")

        if not data_list:
            raise ValueError(f"输入的数据{data}内容为空")
            
        ### 2.创建存储列表与honeycomb晶格结构的对象honey_latt
        record_list = []
        honey_latt = HoneycombXCGeometry(N, a)

        ### 3.遍历数据并封装成DensityRecord对象
        for k, element in enumerate(data_list):
            ## 4.1 解包数据 (格式为: [[i], [val, ...]])
            try:
                [i], [value, *_] = element  # *星号操作符：用于收集剩余的元素到一个列表中；_表示"不关心的变量"或"占位符"
            except (TypeError, ValueError):
                raise NameError(f"Error：第 {k} 个元素格式不正确")
               
            # 获取格点i的坐标与对应水平起始点j
            i_coords = honey_latt.coords[i]
            j = i % 4     # 通过模4取余，便得到格点i对应的水平起始点值

            ## 4.2 实例化类对象
            record_obj = cls(
                lattice_point=i, 
                coords = i_coords,
                density=value,
                horizontal_starting_point=j
            )
            record_list.append(record_obj)

            
        return record_list


#-----------------------------------------------------------------------------------
# 关联函数<i·j>的数据记录器
#-----------------------------------------------------------------------------------  
class CorrelationRecord(Record):
    def __init__(self, source_point: int, field_point: int, distance: float, correlation: float):
        self.source_point = source_point     # 关联函数的源点i
        self.field_point = field_point      # 关联函数的场点j
        self.distance = distance          # 源点i与场点j之间的距离
        self.correlation = correlation      # 关联函数<i·j>的值

        
    def __str__(self):
        return f"关联函数的源点i = {self.source_point}, 场点j = {self.field_point}, 距离r = {self.distance}, 对应关联函数 = {self.correlation:.6f}"
    
    
    
    # 数据提取的类方法(类方法作用：只需调用CorrelationRecord.extract_data(...)即可获得完整的CorrelationRecord对象)
    @classmethod
    def extract_data(cls, data: str, sequence: range, N: int, a: float = 1.0) -> list['CorrelationRecord']:
        """
        将输入的字符串数据解析并封装为 CorrelationRecord 对象列表。
        
        :param data: 计算的关联函数的数据(是列表构成的字符串，每个元素的格式为: [[i, j], [val, 0]])。
        :param sequence: honeycomb晶格下有意义的关联场点序列（源点所在
                    y 行的场点集约定）：i % 4 是 y 行索引（不是蜂窝
                    A/B 子晶格划分）。例如：源点 49 所在 y 行
                    （i % 4 == 1）：range(1, 300, 4)；源点 50 所在
                    y 行（i % 4 == 2）：range(2, 300, 4)。子晶格划分
                    见 HoneycombXCGeometry.sub_lattice_partition
                    （近邻表二部染色）。
        :param N: 系统格点数(要求：N/8可以整除)。
        :param a: honeycomb晶格的晶格常数（默认为a = 1.0）。
                    
        :return record_list: 列表，存储关联函数<i·j>所封装好的CorrelationRecord对象。
        """
        ### 1.提取数据
        try:
            data_list = ast.literal_eval(data)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"输入数据格式错误，无法解析: {e}")

        if not data_list:
            raise ValueError(f"输入的数据{data}内容为空")

        ### 2.获取预期的源点并校验
        ## 2.1 获取源点
        try:
            first_i = data_list[0][0][0]
        except (IndexError, TypeError):
            raise ValueError("数据结构不符合预期格式 [[i, j], [val, 0]]")
        
        ## 2.2 校验：源点是否在有效序列中
        if first_i not in sequence:
            raise ValueError(f"初始源点 {first_i} 不在给定的晶格序列 sequence 中")

        ### 3.创建存储坐标列表与honeycomb晶格结构的对象honey_latt
        record_list = []
        honey_latt = HoneycombXCGeometry(N, a)
        
        ### 4.遍历数据并封装成CorrelationRecord对象
        for k, element in enumerate(data_list):
            ## 4.1 解包数据 (格式为: [[i, j], [val, ...]])
            try:
                [i, j], [value, *_] = element  # *星号操作符：用于收集剩余的元素到一个列表中；_表示"不关心的变量"或"占位符"
            except (TypeError, ValueError):
                raise NameError(f"Error：第 {k} 个元素格式不正确")

            ## 4.2 校验源点一致性
            if i != first_i:
                raise ValueError(f"数据不一致：第 {k} 个元素的源点为 {i}，预期为 {first_i}")

            ## 4.3 校验场点并提取
            if j in sequence:
                # 4.3.1 Honeycomb 晶格距离计算
                r = honey_latt.get_distance(i, j)
                
                # 4.3.2 实例化类对象
                record_obj = cls(
                    source_point=i, 
                    field_point=j, 
                    distance=r, 
                    correlation=value
                )
                record_list.append(record_obj)

        return record_list      
        
        
        
        
        
#-----------------------------------------------------------------------------------
# 超导超导关联函数<i·j·k·l>的数据记录器
#-----------------------------------------------------------------------------------  
class SuperCorrRecord(Record):
    def __init__(self, ref_bond: tuple, target_bond: tuple, distance: float, sup_corr: float):
        self.ref_bond = ref_bond        # 超导关联函数的reference bond构成的点(i, j)
        self.target_bond = target_bond  # 超导关联函数的target bond构成的点(k, l)
        self.distance = distance        # reference bond(i, j)的中心与target bond(k, l)的中心之间的距离
        self.sup_corr = sup_corr        # 超导关联函数<i·j·k·l>的值

        
    def __str__(self):
        return f"超导关联函数的reference bond: {self.ref_bond}, target bond: {self.target_bond}, 距离r: {self.distance}, 超导关联函数: {self.sup_corr:.6f}"
    
    
    
    # 数据提取的类方法(类方法作用：只需调用SuperCorrRecord.extract_data(...)即可获得完整的SuperCorrRecord对象)
    @classmethod
    def extract_data(cls, data: str, N: int, a: float = 1.0) -> list['SuperCorrRecord']:
        """
        将输入的字符串数据解析并封装为SuperCorrRecord对象列表。
        
        :param data: 计算的超导关联函数的数据(是列表构成的字符串，每个元素的格式为: [[i, j, k, l], [val, 0]])。
        :param N: 系统格点数(要求：N/8可以整除)。
        :param a: honeycomb晶格的晶格常数（默认为a = 1.0）。
                    
        :return record_list: 列表，存储超导关联函数<i·j·k·l>所封装好的SuperCorrRecord对象。
        """
        ### 1.提取数据
        try:
            data_list = ast.literal_eval(data)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"输入数据格式错误，无法解析: {e}")

        if not data_list:
            raise ValueError(f"输入的数据{data}内容为空")

        ### 2.获取预期的reference bond并校验
        ## 2.1 获取reference bond(这里用first_bond暂时储存)
        try:
            first_bond = tuple(data_list[0][0][0:2])
        except (IndexError, TypeError):
            raise ValueError("数据结构不符合预期格式 [[i, j, k, l], [val, 0]]")
        
        ## 2.2 校验：reference bond是否左边1/4边界处
        left_bound_p = int(N / 8 / 4) * 8
        left_boundary = [left_bound_p, left_bound_p + 1, left_bound_p + 2, left_bound_p + 3]
        if first_bond[0] not in left_boundary:
            raise ValueError(f"初始reference bond {first_bond[0]} 不在左边1/4边界处")

        ### 3.创建存储坐标列表与honeycomb晶格结构的对象honey_latt
        record_list = []
        honey_latt = HoneycombXCGeometry(N, a)
        
        ### 4.遍历数据，将满足条件的数据提取并封装成SuperCorrRecord对象
        ## 提取tar-bond的条件1：将与ref-bond尽可能保持水平的tar-bond进行提取
        allowed_cond = {(0,0), (4,4), (4,7), (5,0), (7,4), (0,5)}

        for m, element in enumerate(data_list):
            ## 解包数据 (格式为: [[i, j, k, l], [val, 0]])
            try:
                [i, j, k, l], [value, *_] = element  # *星号操作符：用于收集剩余的元素到一个列表中；_表示"不关心的变量"或"占位符"
            except (TypeError, ValueError):
                raise NameError(f"Error：第 {m} 个元素格式不正确")

            ## 校验reference bond一致性
            if (i, j) != first_bond:
                raise ValueError(f"数据不一致：第 {m} 个元素的源点为 {(i, j)}，预期为 {first_bond}")

            ## 提取tar-bond的条件2：要求target bond不超过右边界
            right_bound_p = int(N * 3/4) - 1
            if l <= right_bound_p:
                d1 = (k - i) % 8
                d2 = (l - j) % 8
                if (d1, d2) in allowed_cond:
                    # 超导关联函数的距离计算
                    r = honey_latt.sup_corr_distance(i, j, k, l)                  
                    # 实例化类对象
                    record_obj = cls(
                        ref_bond=(i, j), 
                        target_bond=(k, l), 
                        distance=r, 
                        sup_corr=value
                    )
                    # 添加到record_list中
                    record_list.append(record_obj)  

        return record_list  
        


#-----------------------------------------------------------------------------------
# 单粒子关联函数<i·j>的数据记录器（B2：GreenFunction 输入）
#-----------------------------------------------------------------------------------
class SingleParticleRecord(Record):
    """单粒子关联记录（j > 源点 的单向有序对，B1.1 已核实测量源码）。

    value 保留复数表达能力（当前数据虚部恒 0，B1.1 CONFIRMED 实值
    基态；不因当前数据实值而将接口限定为实数）。分析管线（距离分组/
    衰减核心）消费实部——经 correlation 属性提供，与 CorrelationRecord
    的分析字段约定一致，便于复用既有预处理。
    """
    def __init__(self, source_point: int, field_point: int,
                 distance: float, value: complex):
        self.source_point = source_point     # 关联函数的源点 i
        self.field_point = field_point       # 关联函数的场点 j（j > i）
        self.distance = distance             # 源点与场点之间的欧氏距离
        self.value = value                   # 关联值（复数；当前数据虚部为 0）

    @property
    def correlation(self):
        """分析管线使用的实部（虚部保留在 .value，不丢弃）。"""
        return float(self.value.real)

    def __str__(self):
        return (f"单粒子关联的源点 i = {self.source_point}, "
                f"场点 j = {self.field_point}, 距离 r = {self.distance}, "
                f"关联值 = {self.value:.6f}")

    # 数据提取的类方法
    @classmethod
    def extract_data(cls, data: str, N: int, a: float = 1.0) \
            -> list['SingleParticleRecord']:
        """
        将输入的字符串数据解析并封装为 SingleParticleRecord 对象列表。

        :param data: 单粒子关联数据（列表构成的字符串，元素格式
                     [[i, j], [re, im]]；j > i 单向，全量解析，无
                     sequence/step 过滤——B1 实测目标集合为 ref_site+1
                     至 N-1）。
        :param N: 系统格点数（要求：N/8 可以整除）。
        :param a: 晶格常数（默认为 a = 1.0）。
        :return record_list: SingleParticleRecord 对象列表。

        校验（遵循项目既有提取风格）：空数据/结构错误/源点不一致 →
        ValueError；条目格式错误 → NameError（与 CorrelationRecord
        一致的既有风格）；索引越界/重复键/非有限值 → ValueError
        （严格拒绝，不静默跳过）。
        """
        ### 1.提取数据
        try:
            data_list = ast.literal_eval(data)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"输入数据格式错误，无法解析: {e}")

        if not data_list:
            raise ValueError(f"输入的数据{data}内容为空")

        ### 2.获取预期的源点并校验
        try:
            first_i = data_list[0][0][0]
        except (IndexError, TypeError):
            raise ValueError("数据结构不符合预期格式 [[i, j], [re, im]]")

        ### 3.创建晶格几何对象
        honey_latt = HoneycombXCGeometry(N, a)

        ### 4.遍历数据并封装
        record_list = []
        seen_pairs = set()
        for k, element in enumerate(data_list):
            ## 4.1 解包数据（格式: [[i, j], [re, im]]）
            try:
                [i, j], [re, im] = element
            except (TypeError, ValueError):
                raise NameError(f"Error：第 {k} 个元素格式不正确")

            ## 4.2 校验源点一致性
            if i != first_i:
                raise ValueError(
                    f"数据不一致：第 {k} 个元素的源点为 {i}，"
                    f"预期为 {first_i}")

            ## 4.3 校验索引范围
            if not (0 <= i < N and 0 <= j < N):
                raise ValueError(
                    f"索引越界：第 {k} 个元素 ({i}, {j}) 超出 N={N}")

            ## 4.4 校验重复键
            if (i, j) in seen_pairs:
                raise ValueError(f"重复键：第 {k} 个元素 ({i}, {j}) 已存在")
            seen_pairs.add((i, j))

            ## 4.5 校验有限值并构造复数
            re_f, im_f = float(re), float(im)
            if not (np.isfinite(re_f) and np.isfinite(im_f)):
                raise ValueError(
                    f"非有限值：第 {k} 个元素 ({i}, {j}) 值为 ({re}, {im})")
            value = complex(re_f, im_f)

            ## 4.6 Honeycomb 晶格距离计算与封装
            r = honey_latt.get_distance(i, j)
            record_obj = cls(source_point=i, field_point=j,
                             distance=r, value=value)
            record_list.append(record_obj)

        return record_list
