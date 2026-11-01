import numpy as np


#-----------------------------------------------------------------------------------
# 抽象基类
#-----------------------------------------------------------------------------------
class HoneycombLattice:
    # 初始化方法
    def __init__(self):
        raise NotImplementedError("子类必须实现 __init__ 方法")  # NotImplementedError表示未实现错误
    
    
    # 字符串方法__str__用于实现对象信息的输出
    def __str__(self):
        raise NotImplementedError("子类必须实现 __str__ 方法")  


    # 存储honeycomb晶格的格点坐标的方法
    def _generate_coords(self):
        raise NotImplementedError("子类必须实现 _generate_coords 方法")
        
        
    # 求距离的方法
    def get_distance(self):
        raise NotImplementedError("子类必须实现 get_distance 方法")
        
        
    # 存储honeycomb晶格的最近邻格点索引的方法
    def _nearest_neighbor_grid_index(self):
        raise NotImplementedError("子类必须实现 _nearest_neighbor_grid_index 方法")
        
        
        
        
#-----------------------------------------------------------------------------------
# XC构型下的格点坐标管理类
#-----------------------------------------------------------------------------------   
class HoneycombXCGeometry(HoneycombLattice):
    """Honeycomb 晶格 XC 构型坐标管理类"""
    def __init__(self, N: int, a: float = 1.0):
        """
        :param N: 格点数。
        :param a: 晶格常数（默认为1.0）。
        """
        # 判断格点数是否是最小基元格点数8的倍数
        if N % 8 != 0:
            raise ValueError(f"格点数 N={N} 必须是最小基元 8 的倍数")
        
        self.N = N                     # 格点数
        self.a = a                     # 最近邻边长
        self.hx = 3.0 * a              # 同一水平面上相邻六边形中心的水平距离
        self.hy = np.sqrt(3) * a       # 六边形高度
        self.coords = self._generate_coords()   # honeycomb晶格坐标存储字典（由方法_generate_coords输出）

        
        
    def _generate_coords(self) -> dict:
        """生成 XC 构型的所有格点坐标，并返回每个格点对应坐标的字典coords"""
        # 基元 8 格点相对坐标
        basis = [
            # 第1列
            (0.0, 0.0),                      # 格点0
            (-0.5 * self.a, 0.5 * self.hy),  # 格点1
            (0.0, self.hy),                  # 格点2
            (-0.5 * self.a, 1.5 * self.hy),  # 格点3
            
            # 第2列
            (self.a, 0.0),                   # 格点4
            (1.5 * self.a, 0.5 * self.hy),   # 格点5
            (self.a, self.hy),               # 格点6
            (1.5 * self.a, 1.5 * self.hy)    # 格点7
        ]
        
        coords = {}
        for l in range(self.N // 8):
            for i in range(8):
                point = l * 8 + i
                x0, y0 = basis[i]
                coords[point] = (x0 + l * self.hx, y0)
                
        return coords

    
    
    def get_distance(self, i: int, j: int) -> float:
        """计算并返回honeycomb晶格点 i 和 j 之间的欧几里得距离"""
        # 转换成np.array数组
        pos_i = np.array(self.coords[i])
        pos_j = np.array(self.coords[j])
        
        return np.linalg.norm(pos_i - pos_j)  # 返回欧几里得距离
    


    def sup_corr_distance(self, i: int, j: int, k: int, l: int) -> float:
        """计算超导关联函数的源点(i, j)与场点(k, l)之间的欧几里得距离"""
        # 获取四个点的坐标
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        x3, y3 = self.coords[k]
        x4, y4 = self.coords[l]

        # 计算源点(i, j)与场点(k, l)的中点坐标
        source_point = ((x1 + x2) / 2, (y1 + y2) / 2)
        field_point = ((x3 + x4) / 2, (y3 + y4) / 2)

        # 转换成np数组并返回两点间距离(即超导关联函数的距离)
        pos_ij = np.array(source_point)
        pos_kl = np.array(field_point)
        
        return np.linalg.norm(pos_ij - pos_kl)
        
    
    
    def _nearest_neighbor_grid_index(self) -> list[tuple]:
        """生成 XC 构型的所有最近邻格点的索引，并用元组存储，然后将所有最近邻元组索引存储到列表中"""
        ### 构建honeycomb的最近邻格点指标i、j的列表
        lattice_nn_list = []
        
        ## honeycomb竖直方向(即y方向)指标列表的构建
        for k in range(0, self.N - 3, 4):
            list_y = [(i, i+1) for i in range(k, k+3)] # 最下边x指标为k时对应的y方向指标列表
            lattice_nn_list.extend(list_y)
            periodic = (k+3, k)   # 注：y方向为周期边界
            lattice_nn_list.append(periodic)

        ## honeycomb水平方向(即x方向)最近邻指标列表的构建
        # 第一列水平方向的最近邻指标构建
        list_x = [(0, 4), (2, 6)]
        
        # 之后所有列水平方向的最近邻指标构建
        if self.N > 8:
            for k in range(5, self.N - 10, 8):
                list_x_primitive = [(k, k+4), (k+2, k+6), (k+3, k+7), (k+5, k+9)]
                list_x.extend(list_x_primitive)
                
        ## 添加到最近邻格点列表中
        lattice_nn_list.extend(list_x)
        

        return lattice_nn_list


    def sub_lattice_partition(self) -> dict:
        """基于本构型最近邻表的二部染色 A/B 子晶格划分。

        honeycomb 格是二部图：任意最近邻边连接 A/B 两套三角子晶格。
        从格点 0 出发按 `_nearest_neighbor_grid_index` 做 BFS 染色，
        并验证：
        - 二部性：每条最近邻边两端必须异色（违反 → ValueError，
          不伪造划分）
        - 完整性：全部 N 个格点均被染色（图不连通或近邻表缺边 →
          ValueError，不输出残缺划分）

        返回 dict：'derivation'（推导方式，可追溯）与 'site_labels'
        （索引 0..N-1 的 'A'/'B' 列表，格点 0 所在套命名为 'A'）。
        划分由本构型的近邻表推导，不由调用方硬编码；注意该划分是
        二部图意义下的子晶格，与 CorrelationRecord docstring 中
        "A 子格序列 range(1,N,4)"（i%4==1）的测量源点序列约定
        是两套不同命名，不得混用。
        """
        nn = self._nearest_neighbor_grid_index()
        adj = {i: [] for i in range(self.N)}
        for i, j in nn:
            adj[i].append(j)
            adj[j].append(i)
        labels = {}
        labels[0] = 'A'
        stack = [0]
        while stack:
            u = stack.pop()
            cu = labels[u]
            for v in adj[u]:
                if v in labels:
                    if labels[v] == cu:
                        raise ValueError(
                            f'最近邻表不是二部图：边 ({u}, {v}) 两端同色，'
                            '无法划分 A/B 子晶格')
                else:
                    labels[v] = 'B' if cu == 'A' else 'A'
                    stack.append(v)
        if len(labels) != self.N:
            missing = [i for i in range(self.N) if i not in labels]
            raise ValueError(
                f'最近邻图不连通或覆盖不完整（未染色格点: {missing[:8]}'
                f'{"…" if len(missing) > 8 else ""}，共 {len(missing)} 个），'
                '子晶格划分不完整')
        return {'derivation':
                'bipartite_coloring_from_nearest_neighbor_table',
                'site_labels': [labels[i] for i in range(self.N)]}
        
        