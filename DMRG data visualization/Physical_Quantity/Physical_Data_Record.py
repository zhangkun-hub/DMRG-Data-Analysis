import ast
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
        :param sequence: honeycomb晶格下有意义的关联格点序列。由源点位置而分为两种情形：源点处于A子格的序列/源点处于B子格的序列。
                    例如：A子格序列：range(1, 300, 4)；B子格序列：range(2, 300, 4)。
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
        left_bound_p = int(N / 4 / 4) * 4
        left_boundary = [left_bound_p, left_bound_p + 1, left_bound_p + 2, left_bound_p + 3]
        if first_bond[0] not in left_boundary:
            raise ValueError(f"初始reference bond {first_bond[0]} 不在左边1/4边界处")

        ### 3.创建存储坐标列表与honeycomb晶格结构的对象honey_latt
        record_list = []
        honey_latt = HoneycombXCGeometry(N, a)
        
        ### 4.遍历数据并封装成SuperCorrRecord对象
        for m, element in enumerate(data_list):
            ## 4.1 解包数据 (格式为: [[i, j, k, l], [val, 0]])
            try:
                [i, j, k, l], [value, *_] = element  # *星号操作符：用于收集剩余的元素到一个列表中；_表示"不关心的变量"或"占位符"
            except (TypeError, ValueError):
                raise NameError(f"Error：第 {m} 个元素格式不正确")

            ## 4.2 校验reference bond一致性
            if (i, j) != first_bond:
                raise ValueError(f"数据不一致：第 {m} 个元素的源点为 {(i, j)}，预期为 {first_bond}")

            ## 4.3 校验target bond是否在右边3/4边界处之内，满足条件的提取
            right_bound_p = int(N * 3/4) - 1
            if l <= right_bound_p:
                # 4.3.1 超导关联函数的距离计算
                r = honey_latt.sup_corr_distance(i, j, k, l)
                
                # 4.3.2 实例化类对象
                record_obj = cls(
                    ref_bond=(i, j), 
                    target_bond=(k, l), 
                    distance=r, 
                    sup_corr=value
                )
                record_list.append(record_obj)

        return record_list  
        
        