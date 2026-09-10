from quspin.operators import hamiltonian # 用于在给定的基（basis）上构建哈密顿量算符(或其他物理观测量)；
from quspin.basis import spin_basis_1d # 用于创建一维自旋-1/2链的希尔伯特空间基；
from quspin.basis import spinless_fermion_basis_1d # 用于创建一维无自旋费米子链的希尔伯特空间基；
from quspin.basis import spinful_fermion_basis_1d # 用于创建一维1/2自旋费米子链的希尔伯特空间基；
import numpy as np 
import matplotlib.pyplot as plt  # 用于结果可视化


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------           
def tJ_Model(L, hole_doping, t1, J1, t2=0, J2=0, boundary='periodic', kblock=None, pblock=None, sblock=None, a=1, dtype=np.complex128):
    """
    一维t-J模型（强关联费米子模型，有相互作用的自旋1/2费米子，并且强制要求无双占据）
    
    Input:
    L (int): 一维链长度即格点数(注意：在考虑自旋情形下，一个格点最多可以填充两个电子)；
    hole_doping (float): 空穴率，即空穴数与格点数的比值；
    t1 (float,默认为1.0): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
    boundary (字符串类型，取值只有'periodic'(默认)、'open'): boundary='periodic'表示周期边界条件；boundary ='open'表示开放边界条件；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    sblock (int,取值1或-1,默认为None): 将体系自旋向上与自旋向下费米子发生互换,操作记为Π,若Π|Ψ> = +|Ψ>,则sblock=1;若Π|Ψ> = -|Ψ>,则sblock=-1；
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.complex128): 默认设置矩阵的数据类型为128位复数,因为费米子算符可能涉及复数；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """   
    #### 计算粒子数（根据空穴浓度）
    N_total = L * (1 - hole_doping)     # 总电子数；
    N_up = int(np.ceil(N_total / 2))    # 上自旋电子数(np.ceil()：向上取整函数)；
    N_down = int(np.floor(N_total / 2))  # 下自旋电子数(np.floor()：向下取整函数)；
    
    ####### 周期性边界条件
    if boundary=='periodic': 

        #### 构建基矢：禁止双占据（每个格点最多一个电子）
        basis = spinful_fermion_basis_1d(
            L, 
            Nf=(N_up, N_down), # 可选(默认None)：指定上、下自旋的粒子数,形式为(N_up, N_down),其中N_up上自旋粒子数,N_down下自旋粒子数；
            double_occupancy=False, # 表示是否存在双重占用位点，double_occupancy=True(默认)表示允许双占据；double_occupancy=False则禁止双占据；
            kblock=kblock, 
            pblock=pblock, 
            sblock=sblock,
            a=a
        )

        #### 定义site-coupling lists（周期边界条件下）
        ### 最近邻hopping项
        hop_nn_left = [[-t1, i, (i+1)%L] for i in range(L)]  # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_𝑖+1,𝜎)；
        hop_nn_right = [[t1, i, (i+1)%L] for i in range(L)]  # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_𝑖+1,𝜎)；
        ### 次近邻hopping项
        hop_nnn_left = [[-t2, i, (i+2)%L] for i in range(L)]    # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_𝑖+1,𝜎)；
        hop_nnn_right = [[t2, i, (i+2)%L] for i in range(L)]  # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_𝑖+1,𝜎)；

        ### 最近邻相互作用项(自旋相互作用项+密度相互作用项): S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
        ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
        int_ss = [[J1/2, i, (i+1)%L, i, (i+1)%L] for i in range(L)] 
        ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
        int_nn_ij = [[-J1/2, i, (i+1)%L] for i in range(L)] # n_i↑·n_i↓项；
        int_nn_ji = [[-J1/2, (i+1)%L, i] for i in range(L)] # n_j↑·n_i↓项；

        ### 次近邻相互作用项(自旋相互作用项+密度相互作用项): S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
        ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
        int_sss = [[J2/2, i, (i+2)%L, i, (i+2)%L] for i in range(L)] 
        ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
        int_nnn_ij = [[-J2/2, i, (i+2)%L] for i in range(L)] # n_i↑·n_i↓项；
        int_nnn_ji = [[-J2/2, (i+2)%L, i] for i in range(L)] # n_j↑·n_i↓项；

        # 构建static list
        static = [
            ### 最近邻hopping项
            # 上自旋
            ["+-|", hop_nn_left],   # 右向hopping；
            ["-+|", hop_nn_right],  # 左向hopping(厄米共轭)；
            # 下自旋
            ["|+-", hop_nn_left],   # 右向hopping；
            ["|-+", hop_nn_right],  # 左向hopping(厄米共轭)；

            ### 次近邻hopping项
            # 上自旋
            ["+-|", hop_nnn_left],   # 右向hopping；
            ["-+|", hop_nnn_right],  # 左向hopping(厄米共轭)；
            # 下自旋
            ["|+-", hop_nnn_left],   # 右向hopping；
            ["|-+", hop_nnn_right],  # 左向hopping(厄米共轭)；

            ### 最近邻相互作用项
            ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
            ["+-|-+", int_ss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_𝑖+1↑·𝑐_𝑖↓·𝑐†_𝑖+1↓；
            ["-+|+-", int_ss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_𝑖+1↑·𝑐†_𝑖↓·𝑐_𝑖+1↓；
            ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
            ["n|n", int_nn_ij],  # n_i↑·n_j↓；
            ["n|n", int_nn_ji],  # n_j↑·n_i↓；

            ### 次近邻相互作用项
            ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
            ["+-|-+", int_sss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_𝑖+1↑·𝑐_𝑖↓·𝑐†_𝑖+1↓；
            ["-+|+-", int_sss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_𝑖+1↑·𝑐†_𝑖↓·𝑐_𝑖+1↓；
            ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
            ["n|n", int_nnn_ij],  # n_i↑·n_j↓；
            ["n|n", int_nnn_ji]   # n_j↑·n_i↓；
        ]

        dynamic = []  # 无时间依赖项

        # 构建哈密顿量
        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=np.complex128, 
            check_symm=True, 
            check_pcon=True, 
            check_herm=True
        )
        
    ###### 开放性边界条件
    else:
        #### 构建基矢：禁止双占据（每个格点最多一个电子）
        basis = spinful_fermion_basis_1d(
            L, 
            Nf=(N_up, N_down), 
            double_occupancy=False,
            kblock=None, # 在开放边界条件下无平移对称性，故kblock=None
            pblock=pblock, 
            sblock=sblock,
            a=a
        )

        #### 定义site-coupling lists（开放边界条件下）
        ### 最近邻hopping项
        hop_nn_left = [[-t1, i, (i+1)] for i in range(L-1)]  # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_𝑖+1,𝜎)；
        hop_nn_right = [[t1, i, (i+1)] for i in range(L-1)]   # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_𝑖+1,𝜎)；
        ### 次近邻hopping项
        hop_nnn_left = [[-t2, i, (i+2)] for i in range(L-2)]  # 直接项：从右向左跃迁项(𝑐†𝑖,𝜎·𝑐𝑖+1,𝜎)；
        hop_nnn_right = [[t2, i, (i+2)] for i in range(L-2)]   # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_𝑖+1,𝜎)；

        ### 最近邻相互作用项(自旋相互作用项+密度相互作用项): S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
        ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
        int_ss = [[J1/2, i, (i+1), i, (i+1)] for i in range(L-1)] 
        ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
        int_nn_ij = [[-J1/2, i, (i+1)] for i in range(L-1)] # n_i↑·n_i↓项；
        int_nn_ji = [[-J1/2, (i+1), i] for i in range(L-1)] # n_j↑·n_i↓项；

        ### 次近邻相互作用项(自旋相互作用项+密度相互作用项): S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
        ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
        int_sss = [[J2/2, i, (i+2), i, (i+2)] for i in range(L-2)] 
        ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
        int_nnn_ij = [[-J2/2, i, (i+2)] for i in range(L-2)] # n_i↑·n_i↓项；
        int_nnn_ji = [[-J2/2, (i+2), i] for i in range(L-2)] # n_j↑·n_i↓项；

        # 构建static list
        static = [
            ### 最近邻hopping项
            # 上自旋
            ["+-|", hop_nn_left],   # 右向hopping；
            ["-+|", hop_nn_right],  # 左向hopping(厄米共轭)；
            # 下自旋
            ["|+-", hop_nn_left],   # 右向hopping；
            ["|-+", hop_nn_right],  # 左向hopping(厄米共轭)；

            ### 次近邻hopping项
            # 上自旋
            ["+-|", hop_nnn_left],   # 右向hopping；
            ["-+|", hop_nnn_right],  # 左向hopping(厄米共轭)；
            # 下自旋
            ["|+-", hop_nnn_left],   # 右向hopping；
            ["|-+", hop_nnn_right],  # 左向hopping(厄米共轭)；

            ### 最近邻相互作用项
            ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
            ["+-|-+", int_ss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_𝑖+1↑·𝑐_𝑖↓·𝑐†_𝑖+1↓；
            ["-+|+-", int_ss],  # S^-_i·S^+_j = 𝑐𝑖↑·𝑐†_𝑖+1↑·𝑐†_𝑖↓·𝑐_𝑖+1↓；
            ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
            ["n|n", int_nn_ij],  # n_i↑·n_j↓；
            ["n|n", int_nn_ji],  # n_j↑·n_i↓；

            ### 次近邻相互作用项
            ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
            ["+-|-+", int_sss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_𝑖+1↑·𝑐_𝑖↓·𝑐†_𝑖+1↓；
            ["-+|+-", int_sss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_𝑖+1↑·𝑐†_𝑖↓·𝑐_𝑖+1↓；
            ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
            ["n|n", int_nnn_ij],  # n_i↑·n_j↓；
            ["n|n", int_nnn_ji]   # n_j↑·n_i↓；
        ]

        dynamic = []  # 无时间依赖项

        # 构建哈密顿量
        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=np.complex128, 
            check_symm=True, 
            check_pcon=True, 
            check_herm=True
        )

    return H, basis





#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------           
def Hubbard_Model(L, t1, U, Nf, t2=0, boundary='periodic', kblock=None, pblock=None, sblock=None, a=1, dtype=np.complex128):
    """
    一维Hubbard模型（有相互作用自旋1/2费米子）
    
    Input:
    L (int): 一维链长度即格点数(注意：在考虑自旋情形下，一个格点最多可以填充两个电子)；
    t1 (float): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    U(float): 相互作用系数；
    Nf (元组): 形式为元组Nf=(N_up,N_down),其中N_up为上自旋粒子数,N_down为下自旋粒子数;一般取Nf=(L//2,L//2)为半填充;若Nf=None则表示不限制粒子数；
    boundary (字符串类型，取值只有'periodic'(默认)、'open'): boundary='periodic'表示周期边界条件；boundary ='open'表示开放边界条件；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    sblock (int,取值1或-1,默认为None): 将体系自旋向上与自旋向下费米子发生互换,操作记为Π,若Π|Ψ> = +|Ψ>,则sblock=1;若Π|Ψ> = -|Ψ>,则sblock=-1；
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.complex128): 默认设置矩阵的数据类型为128位复数,因为费米子算符可能涉及复数；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """
    ####### 周期性边界条件
    if boundary=='periodic': 
        
        #### 定义基(函数spinful_fermion_basis_1d()定义1/2自旋费米子空间)
        ## 不限制上、下自旋粒子的填充数(注意：由于指数爆炸问题，一般格点数L<=12才可以设置，其中L=12需要内存大约1 GB，而L=14为16 GB)
        if Nf is None: 
            basis = spinful_fermion_basis_1d(
                L, 
                double_occupancy=True, # 表示是否存在双重占用位点，double_occupancy=True(默认)表示允许双占据；double_occ=False则禁止双占据；
                kblock=kblock, 
                pblock=pblock, 
                sblock=sblock,
                a=a
            )
            
        ## 限制上、下自旋粒子的填充数(一般为半填充,即N_up=L//2,N_down=L//2)
        else:
            basis = spinful_fermion_basis_1d(
                L, 
                Nf=Nf, # 可选：形式为元组Nf=(N_up, N_down),其中N_up为上自旋粒子数,N_down为下自旋粒子数；Nf=(L//2,L//2)为半填充；不写则表示不限制；
                double_occupancy=True, # 表示是否存在双重占用位点，double_occupancy=True(默认)表示允许双占据；double_occ=False则禁止双占据；
                kblock=kblock, 
                pblock=pblock, 
                sblock=sblock,
                a=a
            )

        #### 构建耦合列表（周期边界条件下）
        ### hopping项
        # 最近邻项
        hop_nn_left = [[-t1, i, (i+1)%L] for i in range(L)]   # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_𝑖+1,𝜎)；
        hop_nn_right = [[t1, i, (i+1)%L] for i in range(L)]   # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·𝑐†_𝑖+1,𝜎)；
        # 次近邻项
        hop_nnn_left = [[-t2, i, (i+2)%L] for i in range(L)]  # 直接项：从右向左跃迁项；
        hop_nnn_right = [[t2, i, (i+2)%L] for i in range(L)]  # 厄密共轭项：从左向右跃迁项；

        # 相互作用项（同一格点上、下自旋）
        int_list = [[U, i, i] for i in range(L)]  # U * n_i↑·n_i↓

        #### 构建哈密顿量
        static = [
            # 最近邻hopping项
            ["+-|", hop_nn_left], # 上自旋("+-|"表示𝑐†_𝑖,↑·𝑐_𝑖+1,↑,而|为分隔上下自旋空间的分隔符(左上右下),此处省略右自旋空间恒等算符I)；
            ["|+-", hop_nn_left], # 下自旋("|+-"表示𝑐†_𝑖,↓·𝑐_𝑖+1,↓)；
            ["-+|", hop_nn_right], # 上自旋厄密共轭项(𝑐†_𝑖+1,↑·𝑐_𝑖,↑= - 𝑐_𝑖,↑·𝑐†_𝑖+1,↑)；
            ["|-+", hop_nn_right], # 下自旋厄密共轭项(𝑐†_𝑖+1,↓·𝑐_𝑖,↓= - 𝑐_𝑖,↓·𝑐†_𝑖+1,↓)；
            # 次近邻hopping项
            ["+-|", hop_nnn_left], # 上自旋；
            ["|+-", hop_nnn_left], # 下自旋；
            ["-+|", hop_nnn_right], # 上自旋厄密共轭项；
            ["|-+", hop_nnn_right], # 下自旋厄密共轭项；
            # 相互作用项
            ["n|n", int_list]    # 上、下自旋密度乘积("n|n"表示n_i↑·n_i↓)；
        ]
        dynamic = []
        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype, # 费米子算符可能涉及复数
            check_symm=True,
            check_pcon=True,
            check_herm=True
        )

    ###### 开放性边界条件
    else:
        #### 定义基(函数spinful_fermion_basis_1d()定义1/2自旋费米子空间)
        ## 不限制上、下自旋粒子的填充数(注意：由于指数爆炸问题，一般格点数L<=12才可以设置，其中L=12需要内存大约1 GB，而L=14为16 GB)
        if Nf is None: 
            basis = spinful_fermion_basis_1d(
                L, 
                double_occupancy=True, # 表示是否存在双重占用位点，double_occupancy=True(默认)表示允许双占据；double_occ=False则禁止双占据；
                kblock=None, 
                pblock=pblock, 
                sblock=sblock,
                a=a
            )
            
        ## 限制上、下自旋粒子的填充数(一般为半填充,即N_up=L//2,N_down=L//2)
        else:
            basis = spinful_fermion_basis_1d(
                L, 
                Nf=Nf, # 可选：形式为元组Nf=(N_up, N_down),其中N_up为上自旋粒子数,N_down为下自旋粒子数；Nf=(L//2,L//2)为半填充；不写则表示不限制；
                double_occupancy=True, # 表示是否存在双重占用位点，double_occupancy=True(默认)表示允许双占据；double_occ=False则禁止双占据；
                kblock=None, 
                pblock=pblock, 
                sblock=sblock,
                a=a
            )

        #### 构建耦合列表（开放边界条件下）
        ### hopping项
        # 最近邻项
        hop_nn_left = [[-t1, i, (i+1)] for i in range(L-1)]   # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_𝑖+1,𝜎)；
        hop_nn_right = [[t1, i, (i+1)] for i in range(L-1)]   # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·𝑐†_𝑖+1,𝜎)；
        # 次近邻项
        hop_nnn_left = [[-t2, i, (i+2)] for i in range(L-2)]  # 直接项：从右向左跃迁项；
        hop_nnn_right = [[t2, i, (i+2)] for i in range(L-2)]  # 厄密共轭项：从左向右跃迁项；

        # 在位相互作用项（同一格点上、下自旋）
        int_list = [[U, i, i] for i in range(L)]  # U * n_i↑·n_i↓

        #### 构建哈密顿量
        static = [
            # 最近邻hopping项
            ["+-|", hop_nn_left],     
            ["|+-", hop_nn_left],    
            ["-+|", hop_nn_right],     
            ["|-+", hop_nn_right],    
            # 次近邻hopping项
            ["+-|", hop_nnn_left],    
            ["|+-", hop_nnn_left],   
            ["-+|", hop_nnn_right],    
            ["|-+", hop_nnn_right],   
            # 相互作用项
            ["n|n", int_list]      
        ]
        dynamic = []
        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype, # 费米子算符可能涉及复数
            check_symm=True, 
            check_pcon=True,
            check_herm=True
        )
    
    return H, basis
    
    
    
    

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------  
def Heisenberg_Model(L, J1, J2=0, boundary='periodic', Nup=None, kblock=None, pblock=None, zblock=None, a=1, dtype=np.float64):
    """
    一维海森堡XXX模型,选择自旋表象(即不选pauli表象)
    
    Input:
    L (int): 一维链长度即格点数；
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
    boundary (字符串类型，取值只有'periodic'(默认)、'open'): boundary='periodic'表示周期边界条件；boundary ='open'表示开放边界条件；
    Nup (int,默认为None,一般取L//2): 系统自旋向上的总数。U1对称性下,系统自旋向上的总数Nup可以确定总Sz量子数子空间(比如L//2，表示处在总自旋Sz=0的子空间)；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    zblock (int,取值1或-1,默认为None): 自旋反演对称性的量子数,1对应偶对称性、-1对应对称性； 
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.float64): 默认设置矩阵的数据类型为64位浮点数,这确保了数值计算的精度和效率,特别适用于实对称哈密顿量(如自旋)；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """   
    ######## 周期性边界条件
    if boundary=='periodic':
    
        #### 定义基(包括各种对称性的考虑)(函数spin_basis_1d()定义自旋空间)
        basis = spin_basis_1d(
            L=L,        # 必填(int)：一维链长度；
            S="1/2",     # 可选(str 或 float)：每个格点上的自旋量子数，默认自旋-1/2；
            pauli=False,  # 可选(bool)：pauli=False(默认):使用物理自旋算符表象,例如,"x"对应 Sx; pauli=True:使用Pauli表象,此时"x"对应σx=2Sx；
            Nup=Nup, # 可选(int 或 list of int)：系统自旋向上的总数。U1对称性下,系统自旋向上的总数Nup可以确定总Sz量子数子空间(比如L//2对应总Sz=0)；
            kblock=kblock, # 可选(int)：指定动量块即波矢k的取值；
            pblock=pblock, # 可选(int,取值1或-1)：指定空间反射(宇称)对称性的量子数； 
            zblock=zblock, # 可选(int,取值1或-1)：表示指定自旋反演对称性的量子数； 
            a=a        # 可选(int,默认为1)：平移对称性下原胞格点数；  
        ) 

        #### 哈密顿量的构建（周期边界条件下）
        ## 构建耦合列表(其中[J,i,j]为每个格点i生成一个包含三个元素的子列表,而J表示耦合强度,i、j表示当前格点与相邻格点索引)：
        # 最近邻耦合列表（i和i+1）
        nn_bond_list_xy = [[J1/2, i, (i+1)%L] for i in range(L)] # 自旋xy对应的耦合列表(将Sx与Sy用S+与S-表示则所有矩阵都是实矩阵，但会多出现1/2)；
        nn_bond_list_zz = [[J1, i, (i+1)%L] for i in range(L)] # 自旋z对应的耦合列表；
        # 次近邻耦合列表（i和i+2）
        nnn_bond_list_xy = [[J2/2, i, (i+2)%L] for i in range(L)] # 自旋xy对应的耦合列表(将Sx与Sy用S+与S-表示则所有矩阵都是实矩阵，但会多出现1/2)；
        nnn_bond_list_zz = [[J2, i, (i+2)%L] for i in range(L)] # 自旋z对应的耦合列表；
        # 注意：j=(i+1)%L(%为取余)为周期边界条件(因为对最后格点i=L-1经过该操作得到j=0,即L-1与0格点相邻)。
        
        ## 构建哈密顿量
        # 静态部分(不随时间变化的算符)
        static = [
            # 最近邻海森堡相互作用(注意："x"对应 Sx、"y"对应 Sy、"z"对应 Sz、"+"对应 S+、"-"对应 S-)
            ["+-", nn_bond_list_xy], 
            ["-+", nn_bond_list_xy], 
            ["zz", nn_bond_list_zz],
            # 次近邻海森堡相互作用
            ["+-", nnn_bond_list_xy],
            ["-+", nnn_bond_list_xy], 
            ["zz", nnn_bond_list_zz]
        ]
        # 动态部分(随时间变化的算符)
        dynamic = [] 
        # 使用QuSpin的hamiltonian构造函数来得到哈密顿量对象，其矩阵形式是自动使用稀疏矩阵格式存储哈密顿量(但H不只是矩阵)
        H = hamiltonian(  
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype,     
            check_symm=True,  # 可选(bool布尔类型)：check_symm=True(默认)表示验证运算符是否与基底对称性兼容；check_symm=False表示禁用；
            check_pcon=True,  # 可选(bool布尔类型)：check_pcon=True(默认)表示检查运算符是否保持粒子数守恒；check_pcon=False表示禁用；
            check_herm=True   # 可选(bool布尔类型)：check_herm=True(默认)表示验证运算符是否为厄米算符；check_herm=False表示禁用；
        ) 
    
    ######## 开放性边界条件
    else: 
        #### 定义基(包括各种对称性的考虑)(函数spin_basis_1d()定义自旋空间)
        basis = spin_basis_1d(
            L=L,        
            S="1/2",     
            pauli=False,  
            Nup=Nup, 
            kblock=None, # 在开放边界条件下无平移对称性，故kblock=None
            pblock=pblock, 
            zblock=zblock,  
            a=a          
        ) 
        
        #### 哈密顿量的构建（开放边界条件下）
        ## 构建耦合列表：
        nn_bond_list_xy = [[J1/2, i, (i+1)] for i in range(L-1)] # 自旋xy对应的耦合列表(将Sx与Sy用S+与S-表示则所有矩阵都是实矩阵，但会多出现1/2)；
        nn_bond_list_zz = [[J1, i, (i+1)] for i in range(L-1)] # 自旋z对应的耦合列表；
        nnn_bond_list_xy = [[J2/2, i, (i+2)] for i in range(L-2)] # 次近邻耦合列表（i和i+2）
        nnn_bond_list_zz = [[J2, i, (i+2)] for i in range(L-2)] # 自旋z对应的耦合列表；

        ## 构建哈密顿量
        static = [
            ["+-", nn_bond_list_xy], # 最近邻海森堡相互作用(注意："x"对应 Sx、"y"对应 Sy、"z"对应 Sz、"+"对应 S+、"-"对应 S-)
            ["-+", nn_bond_list_xy], 
            ["zz", nn_bond_list_zz],
            ["+-", nnn_bond_list_xy],# 次近邻海森堡相互作用
            ["-+", nnn_bond_list_xy], 
            ["zz", nnn_bond_list_zz]
        ]
        dynamic = [] 
        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype,
            check_symm=True,
            check_pcon=True,
            check_herm=True
        ) 

    return H, basis



#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------           
def Ising_Model(L, J1, J2=0, h=0, boundary='periodic', Nup=None, kblock=None, pblock=None, zblock=None, a=1, dtype=np.float64):
    """
    一维横场Ising模型
    
    Input:
    L (int): 一维链长度即格点数；
    J1 (floats): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
    h (float,默认为0): 横场强度；
    boundary (字符串类型，取值只有'periodic'(默认)、'open'): boundary='periodic'表示周期边界条件；boundary ='open'表示开放边界条件；
    Nup (int,默认为None,一般取L//2): 系统自旋向上的总数。U1对称性下,系统自旋向上的总数Nup可以确定总Sz量子数子空间(比如L//2，表示系统处在Sz=0的子空间)；
    kblock (int,默认为None,一般取L//2): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    zblock(int,取值1或-1,默认为None): 自旋反演对称性的量子数,1对应偶对称性、-1对应对称性； 
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.float64): 默认设置矩阵的数据类型为64位浮点数,这确保了数值计算的精度和效率,特别适用于实对称哈密顿量(如自旋)；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """
    ####### 周期性边界条件
    if boundary=='periodic':     
    
        #### 定义基(包括各种对称性的考虑)(函数spin_basis_1d()定义自旋空间)
        basis = spin_basis_1d(
            L=L,
            S="1/2",
            pauli=False,
            Nup=Nup,      
            kblock=kblock,  
            pblock=pblock,  
            zblock=zblock,   
            a=a          
        )
    
        ## 构建耦合列表（周期边界条件下）
        nn_bond_list = [[J1, i, (i+1)%L] for i in range(L)] # 最近邻耦合（zz相互作用）
        nnn_bond_list = [[J2, i, (i+2)%L] for i in range(L)] # 次近邻耦合（zz相互作用）
        h_field = [[-h, i] for i in range(L)] # 横场耦合（x方向）

        ## 构建哈密顿量
        static = [
            ["zz", nn_bond_list],   # 最近邻zz相互作用
            ["zz", nnn_bond_list],  # 次近邻zz相互作用
            ["x", h_field]        # 横场
        ]
        dynamic = []

        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype,
            check_symm=True,
            check_pcon=True,
            check_herm=True
        )
    
    ###### 开放性边界条件
    else: 
        #### 定义基(包括各种对称性的考虑)(函数spin_basis_1d()定义自旋空间)
        basis = spin_basis_1d(
            L=L,
            S="1/2",
            pauli=False,
            Nup=Nup,      
            kblock=None, # 在开放边界条件下无平移对称性，故kblock=None
            pblock=pblock,  
            zblock=zblock,   
            a=a          
        )
    
        ## 构建耦合列表（开放边界条件下）
        nn_bond_list = [[J1, i, (i+1)] for i in range(L-1)] # 最近邻耦合（zz相互作用）
        nnn_bond_list = [[J2, i, (i+2)] for i in range(L-2)] # 次近邻耦合（zz相互作用）
        h_field = [[-h, i] for i in range(L)] # 横场耦合（x方向）

        ## 构建哈密顿量
        static = [
            ["zz", nn_bond_list],   # 最近邻zz相互作用
            ["zz", nnn_bond_list],  # 次近邻zz相互作用
            ["x", h_field]        # 横场
        ]
        dynamic = []
        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype,
            check_symm=True,
            check_pcon=True,
            check_herm=True
        )

    return H, basis 




    
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------           
def Tight_Binding_Model(L, t1, t2=0, boundary='periodic', Nf=None, kblock=None, pblock=None, a=1, dtype=np.float64):
    """
    一维Tight Binding模型(无自旋自由费米子体系)；
    注：有自旋的Tight Binding模型本质就是Hubbard模型中相互作用项U=0的情形，相比无自旋只是单纯多个对自旋的求和号，而各种计算结果一般也就多乘个2(自旋1/2)
    
    Input:
    L (int): 一维链长度即格点数；
    t1 (float): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    boundary (字符串类型，取值只有'periodic'(默认)、'open'): boundary='periodic'表示周期边界条件；boundary ='open'表示开放边界条件；
    Nf (int,默认为None,一般取L//2): 指定希尔伯特空间的粒子数子空间(U1对称性导致),可以是单个整数(如Nf=L//2表示半填充粒子数)，或一个整数列表(如Nf=[0,1,2]表                           示取多个粒子数子空间)。如果为None，表示填充所有可能的粒子数空间；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称； 
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.complex128): 默认设置矩阵的数据类型为128位复数,因为费米子算符可能涉及复数；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """
    ####### 周期性边界条件
    if boundary=='periodic': 
        
        #### 定义基(函数spinless_fermion_basis_1d()定义无自旋费米子空间)
        basis = spinless_fermion_basis_1d(
            L,             # 必填(int)：一维链长度；
            Nf=Nf,         # 可选(int或list of int,默认值为None)：指定希尔伯特空间的粒子数子空间(U1对称性导致)。可以是单个整数(如Nf=L//2表示半                               填充粒子数),或一个整数列表(如Nf=[0,1,2]表示取多个粒子数子空间)。如果为None,表示填充所有可能的粒子数空间；
            kblock=kblock,      # 可选(int)：指定动量块即波矢k的取值；
            pblock=pblock,      # 可选(int,取值1或-1)：指定空间反射(宇称)对称性的量子数； 
            a=a            # 可选(int,默认为1)：平移对称性下原胞格点数。
        )

        ## 构建耦合列表（周期边界条件下）
        # 最近邻项
        hop_nn_left = [[-t1, i, (i+1)%L] for i in range(L)]   # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_𝑖+1,𝜎)；
        hop_nn_right = [[t1, i, (i+1)%L] for i in range(L)]   # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·𝑐†_𝑖+1,𝜎)；
        # 次近邻项
        hop_nnn_left = [[-t2, i, (i+2)%L] for i in range(L)]  # 直接项：从右向左跃迁项；
        hop_nnn_right = [[t2, i, (i+2)%L] for i in range(L)]  # 厄密共轭项：从左向右跃迁项；

        ## 构建哈密顿量
        static = [
            ["+-", hop_nn_left],   # 最近邻hopping (𝑐†_𝑖·𝑐_𝑖+1)
            ["-+", hop_nn_right],   # 厄米共轭项 (𝑐†_𝑖+1·𝑐_𝑖= - 𝑐_𝑖·𝑐†_𝑖+1)
            ["+-", hop_nnn_left],  # 次近邻hopping
            ["-+", hop_nnn_right]   # 厄米共轭项
        ]
        dynamic = []

        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=dtype, # 费米子算符可能涉及复数
            check_symm=True,
            check_pcon=True,
            check_herm=True
        )

    ###### 开放性边界条件
    else:
        #### 定义基(函数spinless_fermion_basis_1d()定义无自旋费米子空间)
        basis = spinless_fermion_basis_1d(
            L,             
            Nf=Nf,         
            kblock=None, # 在开放边界条件下无平移对称性，故kblock=None     
            pblock=pblock,      
            a=a            
        )

        ## 构建耦合列表（开放边界条件下）
        # 最近邻项
        hop_nn_left = [[-t1, i, (i+1)] for i in range(L-1)]   # 直接项：从右向左跃迁项(𝑐†_𝑖·𝑐_𝑖+1)；
        hop_nn_right = [[t1, i, (i+1)] for i in range(L-1)]   # 厄密共轭项：从左向右跃迁项(𝑐†_𝑖+1·𝑐_𝑖= - 𝑐_𝑖·𝑐†_𝑖+1)；
        # 次近邻项
        hop_nnn_left = [[-t2, i, (i+2)] for i in range(L-2)]  # 直接项：从右向左跃迁项；
        hop_nnn_right = [[t2, i, (i+2)] for i in range(L-2)]  # 厄密共轭项：从左向右跃迁项；
        
        hop_nnn = [[-t2, i, (i+2)] for i in range(L-2)] # 次近邻hopping

        ## 构建哈密顿量
        static = [
            ["+-", hop_nn_left],   # 最近邻hopping (𝑐†_𝑖·𝑐_𝑖+1)
            ["-+", hop_nn_right],   # 厄米共轭项 (𝑐†_𝑖+1·𝑐_𝑖= - 𝑐_𝑖·𝑐†_𝑖+1)
            ["+-", hop_nnn_left],  # 次近邻hopping
            ["-+", hop_nnn_right]   # 厄米共轭项
        ]
        dynamic = []

        H = hamiltonian(
            static, 
            dynamic, 
            basis=basis, 
            dtype=np.complex128, # 费米子算符可能涉及复数
            check_symm=True,
            check_pcon=True,
            check_herm=True
        )

    return H, basis