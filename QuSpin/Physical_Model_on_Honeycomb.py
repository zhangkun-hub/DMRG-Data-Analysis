from quspin.operators import hamiltonian # 用于在给定的基（basis）上构建哈密顿量算符(或其他物理观测量)；
from quspin.basis import spin_basis_1d # 用于创建一维自旋-1/2链的希尔伯特空间基；
from quspin.basis import spinless_fermion_basis_1d # 用于创建一维无自旋费米子链的希尔伯特空间基；
from quspin.basis import spinful_fermion_basis_1d # 用于创建一维1/2自旋费米子链的希尔伯特空间基；
import numpy as np 
import matplotlib.pyplot as plt  # 用于结果可视化

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------   
def tJ_Model_honeycomb_16site(hole_doping, t1, J1, t2=0, J2=0, kblock=None, pblock=None, sblock=None, a=1, dtype=np.complex128):
    """
    Honeycomb上16个格点的t-J模型（强关联费米子模型，有相互作用的自旋1/2费米子，并且强制要求无双占据）
    （其中上下格点数为4，左右格点数为4，且上下边界为周期边界，左右边界为开放边界）
    
    Input:
    hole_doping (float): 空穴率，即空穴数与格点数的比值；
    t1 (float,默认为1.0): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    sblock (int,取值1或-1,默认为None): 将体系自旋向上与自旋向下费米子发生互换,操作记为Π,若Π|Ψ> = +|Ψ>,则sblock=1;若Π|Ψ> = -|Ψ>,则sblock=-1；
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.complex128): 默认设置矩阵的数据类型为128位复数,因为费米子算符可能涉及复数；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """   
    L = 16 # 格点数
    
    #### 计算粒子数（根据空穴浓度）
    N_total = L * (1 - hole_doping)     # 总电子数；
    N_up = int(np.ceil(N_total / 2))    # 上自旋电子数(np.ceil()：向上取整函数)；
    N_down = int(np.floor(N_total / 2))  # 下自旋电子数(np.floor()：向下取整函数)；

    #### 构建基矢：禁止双占据（每个格点最多一个电子）
    basis = spinful_fermion_basis_1d(
        L, 
        Nf=(N_up, N_down), 
        double_occupancy=False,
        kblock=kblock, 
        pblock=pblock, 
        sblock=sblock,
        a=a
    )
    
    ### 构建honeycomb的最近邻格点指标i、j的列表
    lattice_nn_list = []
    # honeycomb竖直方向(即y方向)指标列表的构建
    for k in range(0, 13, 4):
        list_y = [(i,(i+1)) for i in range(k, k+3)] # x指标为k时对应的y方向指标列表
        lattice_nn_list.extend(list_y)
        periodic = (k+3,k) # 注：y方向为周期边界
        lattice_nn_list.append(periodic)
    
    # honeycomb水平方向(即x方向)指标列表的构建
    list_x = [(0,4), (2,6), (5,9), (7,11), (8,12), (10,14)] # 注：x方向为开放边界
    lattice_nn_list.extend(list_x)
    
    ### 构建honeycomb的次近邻格点指标i、j的列表
    lattice_nnn_list = []
    # honeycomb竖直方向(即y方向)指标列表的构建
    for k in range(0, 13, 4):
        list_nnn_y = [(i,(i+2)) for i in range(k, k+2)] # x指标为k时对应的y方向指标列表
        lattice_nnn_list.extend(list_y)
        
        # y方向为周期边界
        periodic_1 = (k+2,k) 
        periodic_2 = (k+3,k+1)
        lattice_nnn_list.append(periodic_1)
        lattice_nnn_list.append(periodic_2)
    
    # honeycomb水平方向(即x方向)指标列表的构建
    list_nnn_x = [
        (1,4), (1,6), (3,4), (3,6), (0,5), (0,7), (4,9), (4,11), (2,7), (2,5), (6,11), (6,9), 
              (5,10), (5,8), (9,14), (9,12), (7,10), (7,8), (11,14), (11,12), (8,13), (8,15), (10,15), (10,13)
    ] # 注：x方向为开放边界
    lattice_nnn_list.extend(list_x)

    #### 定义site-coupling lists
    ### 最近邻hopping项
    hop_nn_left = [[-t1, i, j] for i,j in lattice_nn_list]  # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_j,𝜎)；
    hop_nn_right = [[t1, i, j] for i,j in lattice_nn_list]  # 厄密共轭项：从左向右跃迁项(𝑐†_j,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_j,𝜎)；
    ### 次近邻hopping项
    hop_nnn_left = [[-t2, i, j] for i,j in lattice_nnn_list]  # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_j,𝜎)；
    hop_nnn_right = [[t2, i, j] for i,j in lattice_nnn_list]  # 厄密共轭项：从左向右跃迁项(𝑐†_j,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_j,𝜎)；

    ### 最近邻相互作用项(自旋相互作用项 + 密度相互作用项)： S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
    ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
    int_ss = [[J1/2, i, j, i, j] for i,j in lattice_nn_list]
    ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
    int_nn_ij = [[-J1/2, i, j] for i,j in lattice_nn_list] # n_i↑·n_i↓项；
    int_nn_ji = [[-J1/2, j, i] for i,j in lattice_nn_list] # n_j↑·n_i↓项；

    ### 次近邻相互作用项(自旋相互作用项 + 密度相互作用项)： S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
    ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
    int_sss = [[J2/2, i, j, i, j] for i,j in lattice_nnn_list] 
    ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
    int_nnn_ij = [[-J2/2, i, j] for i,j in lattice_nnn_list] # n_i↑·n_i↓项；
    int_nnn_ji = [[-J2/2, j, i] for i,j in lattice_nnn_list] # n_j↑·n_i↓项；

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
        ["+-|-+", int_ss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_j↑·𝑐_𝑖↓·𝑐†_j↓；
        ["-+|+-", int_ss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_j↑·𝑐†_𝑖↓·𝑐_j↓；
        ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
        ["n|n", int_nn_ij],  # n_i↑·n_j↓；
        ["n|n", int_nn_ji],  # n_j↑·n_i↓；

        ### 次近邻相互作用项
        ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
        ["+-|-+", int_sss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_j↑·𝑐_𝑖↓·𝑐†_j↓；
        ["-+|+-", int_sss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_j↑·𝑐†_𝑖↓·𝑐_j↓；
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
        dtype=dtype, 
        check_symm=True, 
        check_pcon=True, 
        check_herm=True
    )
    
    return H, basis




#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------   
def tJ_Model_honeycomb_Average_value_16site(hole_doping, t1, J1, t2=0, J2=0, kblock=None, pblock=None, sblock=None, a=1, dtype=np.complex128):
    """
    Honeycomb上16个格点的t-J模型的各格点的自旋Sz算符与密度算符n的平均值
    
    Input:(与定义的tJ_Model_honeycomb函数的参数完全一样)
    hole_doping (float): 空穴率，即空穴数与格点数的比值；
    t1 (float,默认为1.0): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    sblock (int,取值1或-1,默认为None): 将体系自旋向上与自旋向下费米子发生互换,操作记为Π,若Π|Ψ> = +|Ψ>,则sblock=1;若Π|Ψ> = -|Ψ>,则sblock=-1；
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.complex128): 默认设置矩阵的数据类型为128位复数,因为费米子算符可能涉及复数；

    Return:
    Sz_avg: 储存每个格点自旋Sz算符平均值的列表；
    n_avg: 储存每个格点密度算符平均值的列表。
    """  
    L = 16 # 格点数
    
    # 基与哈密顿量的建立
    H, basis = tJ_Model_honeycomb_16site(hole_doping, t1, J1, t2, J2, kblock, pblock, sblock, a, dtype)
    
    # 基态与基态能量
    E_gs, V_gs = H.eigsh(k=1, which='SA') 
    E_gs = E_gs[0] # 基态能量
    V_gs = V_gs[:, 0] # 基态
    
    #### 一、 自旋Sz算符平均值
    ## 计算每个格点的自旋基态平均值<S^z_i>，其中第i格点的Sz_i = 1/2 * (𝑐†_𝑖,↑·𝑐_𝑖,↑ - 𝑐†_𝑖,↓·𝑐_𝑖,↓)
    Sz_avg = []  # 创建存储每个格点的S_z平均值

    for i in range(L):
        # 构建第i个格点的Sz算符(与哈密顿量的构建完全类似)
        static_Sz = [
            ["+-|", [[0.5, i, i]]],  # 1/2 * 𝑐†_𝑖,↑·𝑐_𝑖,↑
            ["|+-", [[-0.5, i, i]]] # -1/2 * 𝑐†_𝑖,↓·𝑐_𝑖,↓

        ] 
        dynamic_Sz = [] 
        S_z_i = hamiltonian(static_Sz, dynamic_Sz, basis=basis, dtype=dtype, check_symm=False, check_pcon=False, check_herm=False)

        # 计算基态期望值，并转换为自旋算符S_z（除以2）
        Sz_i = S_z_i.expt_value(V_gs).real  # S_z_i.expt_value为S_z_i的期望值计算函数,V_gs为上面得到的基态,real表示期望值为实数
        Sz_avg.append(Sz_i)

        
    #### 二、密度算符平均值
    ## 计算每个格点的密度基态平均值<n_i>，其中n_i = n_i,↑ + n_i,↓
    n_avg = []  # 创建存储每个格点的S_z平均值

    for i in range(L):
        # 构建第i个格点的Sz算符(与哈密顿量的构建完全类似)
        static_n = [
            ["n|", [[1.0, i]]],  # n_i,↑
            ["|n", [[1.0, i]]]   # n_i,↓
        ] 
        dynamic_n = [] 
        ni = hamiltonian(static_n, dynamic_n, basis=basis, dtype=dtype, check_symm=False, check_pcon=False, check_herm=False)

        # 计算基态期望值，并转换为自旋算符S_z（除以2）
        n_i = ni.expt_value(V_gs).real  # S_z_i.expt_value为S_z_i的期望值计算函数,V_gs为上面得到的基态,real表示期望值为实数
        n_avg.append(n_i)

        
        
    return Sz_avg, n_avg





#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------   
def tJ_Model_honeycomb_Correlation_function_16site(ref_site, hole_doping, t1, J1, t2=0, J2=0, kblock=None, pblock=None, sblock=None, a=1, dtype=np.complex128):
    """
    Honeycomb上16个格点的t-J模型的自旋关联函数<S^+_i·S^-_j>、<S^-_i·S^+_j>、<S^z_i·S^z_j>和密度关联函数<n_i·n_j>
    
    Input:
    ref_site: 关联函数的起始格点位置；
    hole_doping (float): 空穴率，即空穴数与格点数的比值；
    t1 (float,默认为1.0): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
    Nf (元组,默认为None): 形式为元组Nf=(N_up, N_down),其中N_up为上自旋粒子数,N_down为下自旋粒子数；Nf=(L//2,L//2)为半填充；若Nf=None则表示不限制粒子数；
    kblock (int,默认为None,一般取0): 动量块即波矢k的取值(平移对称性的量子数，对应动量2πk/L),要求系统具有周期性边界即平移对称性(a=1且pblock=1)；
    pblock (int,取值1或-1,默认为None): 空间反射(宇称)对称性的量子数，1对应偶宇称、-1对应奇宇称；
    sblock (int,取值1或-1,默认为None): 将体系自旋向上与自旋向下费米子发生互换,操作记为Π,若Π|Ψ> = +|Ψ>,则sblock=1;若Π|Ψ> = -|Ψ>,则sblock=-1；
    a=1 (int,默认为1): 原胞大小(即原胞格点数)。当 a>1 时，系统被视为具有 a-site 的晶胞结构；
    dtype (默认np.complex128): 默认设置矩阵的数据类型为128位复数,因为费米子算符可能涉及复数；

    Return:
    Su_Sd_correlations, distances_Su_Sd: 储存自旋关联函数<S^+_i·S^-_j>与i、j之间距离的列表；
    Sd_Su_correlations, distances_Sd_Su: 储存自旋关联函数<S^-_i·S^+_j>与i、j之间距离的列表；
    Sz_Sz_correlations, distances_Sz_Sz: 储存自旋关联函数<S^z_i·S^z_j>与i、j之间距离的列表；
    nn_correlations, distances_nn: 储存自旋关联函数<n_i·n_j>与i、j之间距离的列表。
    """  
    L = 16 # 格点数
    ref_site_Su = ref_site_Sd = ref_site_Sz = ref_site_nn = ref_site # 关联函数的起始格点
    distances = []    # 存储两格点之间的距离列表
    
    # 基与哈密顿量的建立
    H, basis = tJ_Model_honeycomb_16site(hole_doping, t1, J1, t2, J2, kblock, pblock, sblock, a, dtype)
    
    # 基态与基态能量
    E_gs, V_gs = H.eigsh(k=1, which='SA') 
    E_gs = E_gs[0] # 基态能量
    V_gs = V_gs[:, 0] # 基态
    
    #### 一、自旋关联函数<S^+_i·S^-_j>。其中S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_j↑·𝑐_𝑖↓·𝑐†_j↓
    Su_Sd_correlations = [] # 存储关联值

    # 平移对称性下关联函数只是相对距离的函数，而相对距离d从1到L-1
    for r in range(1, L-ref_site_Su): 
        j = (ref_site_Su + r) # 格点j取值

        # 构建S^+_i·S^-_j算符(与哈密顿量的构建完全类似)
        static_Su_Sd = [
            ["+-|-+", [[1.0, ref_site_Su, j, ref_site_Su, j]]]  # 𝑐†_𝑖↑·𝑐_j↑·𝑐_𝑖↓·𝑐†_j↓
        ]
        dynamic_Su_Sd = [] 
        S_ud_ij = hamiltonian(static_Su_Sd, dynamic_Su_Sd, basis=basis, dtype=dtype, check_symm=False, check_pcon=False, check_herm=False)

        # 计算基态期望值，即基态关联函数<S^+_i·S^-_j>
        S_uS_d = S_ud_ij.expt_value(V_gs).real # # S_ud_ij.expt_value为S_ud_ij的期望值计算函数,V_gs为上面得到的基态,real表示期望值为实数
        Su_Sd_correlations.append(S_uS_d)
        distances.append(r)


    #### 二、自旋关联函数<S^-_i·S^+_j>。其中S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_j↑·𝑐†_𝑖↓·𝑐_j↓
    Sd_Su_correlations = [] # 存储关联值

    # 平移对称性下关联函数只是相对距离的函数，而相对距离d从1到L-1
    for r in range(1, L-ref_site_Sd): 
        j = (ref_site_Sd + r) # 格点j取值

        # 构建S^-_i·S^+_j算符(与哈密顿量的构建完全类似)
        static_Sd_Su = [
            ["-+|+-", [[1.0, ref_site_Sd, j, ref_site_Sd, j]]]  # 𝑐_𝑖↑·𝑐†_j↑·𝑐†_𝑖↓·𝑐_j↓
        ]
        dynamic_Sd_Su = [] 
        S_du_ij = hamiltonian(static_Sd_Su, dynamic_Sd_Su, basis=basis, dtype=dtype, check_symm=False, check_pcon=False, check_herm=False)

        # 计算基态期望值，即基态关联函数<S^-_i·S^+_j>
        S_dS_u = S_du_ij.expt_value(V_gs).real # S_du_ij.expt_value为S_du_ij的期望值计算函数,V_gs为上面得到的基态,real表示期望值为实数
        Sd_Su_correlations.append(S_dS_u)
        
        
    ### 三、自旋关联函数<S^z_i·S^z_j>。其中S^z_i·S^z_j = 1/4 * (n_i↑·n_j↑ + n_i↓·n_j↓ - n_i↑·n_j↓ - n_j↑·n_i↓)
    Sz_Sz_correlations = []  # 存储关联值

    # 平移对称性下关联函数只是相对距离的函数，而相对距离d从1到L-1
    for r in range(1, L-ref_site_Sz): 
        j = (ref_site_Sz + r) # 格点j取值

        # 构建S^z_i·S^z_j算符(与哈密顿量的构建完全类似)
        static_Sz_Sz = [
            ["nn|", [[0.25, ref_site_Sz, j]]],  # 1/4 * n_i↑·n_j↑
            ["|nn", [[0.25, ref_site_Sz, j]]],  # 1/4 * n_i↓·n_j↓
            ["n|n", [[-0.25, ref_site_Sz, j]]],  # -1/4 * n_i↑·n_j↓
            ["n|n", [[-0.25, j, ref_site_Sz]]]   # -1/4 * n_j↑·n_i↓
        ]
        dynamic_Sz_Sz = [] 
        S_zz_ij = hamiltonian(static_Sz_Sz, dynamic_Sz_Sz, basis=basis, dtype=dtype, check_symm=False, check_pcon=False, check_herm=False)

        # 计算基态期望值，即基态关联函数<S^z_i·S^z_j>
        S_zS_z = S_zz_ij.expt_value(V_gs).real # # S_zz_ij.expt_value为S_zz_ij的期望值计算函数,V_gs为上面得到的基态,real表示期望值为实数
        Sz_Sz_correlations.append(S_zS_z)


    ### 四、密度关联函数<n_i·n_j>。其中n_i·n_j = n_i↑·n_j↑ + n_i↓·n_j↓ + n_i↑·n_j↓ + n_j↑·n_i↓
    nn_correlations = []  # 存储关联值

    # 平移对称性下关联函数只是相对距离的函数，而相对距离d从1到L-1
    for r in range(1, L-ref_site_nn): 
        j = (ref_site_nn + r) # 格点j取值

        # 构建n_i·n_j算符(与哈密顿量的构建完全类似)
        static_nn = [
            ["nn|", [[1.0, ref_site_nn, j]]],  # n_i↑·n_j↑
            ["|nn", [[1.0, ref_site_nn, j]]],  # n_i↓·n_j↓
            ["n|n", [[1.0, ref_site_nn, j]]],  # n_i↑·n_j↓
            ["n|n", [[1.0, j, ref_site_nn]]]   # n_j↑·n_i↓
        ]
        dynamic_nn = [] 
        S_nn_ij = hamiltonian(static_nn, dynamic_nn, basis=basis, dtype=dtype, check_symm=False, check_pcon=False, check_herm=False)

        # 计算基态期望值，即基态关联函数<S^z_i·S^z_j>
        S_nS_n = S_nn_ij.expt_value(V_gs).real # # S_zz_ij.expt_value为S_zz_ij的期望值计算函数,V_gs为上面得到的基态,real表示期望值为实数
        nn_correlations.append(S_nS_n)

        
        
    return Su_Sd_correlations, Sd_Su_correlations, Sz_Sz_correlations, nn_correlations, distances





#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------           
def Heisenberg_Model_honeycomb(J1, J2=0, Nup=None, kblock=None, pblock=None, zblock=None, a=1, dtype=np.float64):
    """
    Honeycomb上16个格点的Heisenberg Model模型（主要用于检验t-J模型半满是否与其对应）
    （其中上下格点数为4，左右格点数为4，且上下边界为周期边界，左右边界为开放边界）
    
    Input:
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；
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
    L = 16 # 格点数

    #### 构建自旋基矢
    basis = spin_basis_1d(
        L, 
        S="1/2",
        pauli=False,
        Nup=Nup,
        kblock=kblock,
        pblock=pblock, 
        zblock=zblock,
        a=a
    )
    
    ### 构建honeycomb的最近邻格点指标i、j的列表
    lattice_nn_list = []
    # honeycomb竖直方向(即y方向)指标列表的构建
    for k in range(0, 13, 4):
        list_y = [(i,(i+1)) for i in range(k, k+3)] # x指标为k时对应的y方向指标列表
        lattice_nn_list.extend(list_y)
        periodic = (k+3,k) # 注：y方向为周期边界
        lattice_nn_list.append(periodic)
    
    # honeycomb水平方向(即x方向)指标列表的构建
    list_x = [(0,4), (2,6), (5,9), (7,11), (8,12), (10,14)] # 注：x方向为开放边界
    lattice_nn_list.extend(list_x)
    
    ### 构建honeycomb的次近邻格点指标i、j的列表
    lattice_nnn_list = []
    # honeycomb竖直方向(即y方向)指标列表的构建
    for k in range(0, 13, 4):
        list_nnn_y = [(i,(i+2)) for i in range(k, k+2)] # x指标为k时对应的y方向指标列表
        lattice_nnn_list.extend(list_y)
        
        # y方向为周期边界
        periodic_1 = (k+2,k) 
        periodic_2 = (k+3,k+1)
        lattice_nnn_list.append(periodic_1)
        lattice_nnn_list.append(periodic_2)
    
    # honeycomb水平方向(即x方向)指标列表的构建
    list_nnn_x = [
        (1,4), (1,6), (3,4), (3,6), (0,5), (0,7), (4,9), (4,11), (2,7), (2,5), (6,11), (6,9), 
              (5,10), (5,8), (9,14), (9,12), (7,10), (7,8), (11,14), (11,12), (8,13), (8,15), (10,15), (10,13)
    ] # 注：x方向为开放边界
    lattice_nnn_list.extend(list_x)

    #### 定义site-coupling lists
    # 最近邻耦合列表
    nn_bond_list_xy = [[J1/2, i, j] for i,j in lattice_nn_list] # 自旋xy对应的耦合列表(将Sx与Sy用S+与S-表示则所有矩阵都是实矩阵，但会多出现1/2)；
    nn_bond_list_zz = [[J1, i, j] for i,j in lattice_nn_list] # 自旋z对应的耦合列表；
    # 次近邻耦合列表
    nnn_bond_list_xy = [[J2/2, i, j] for i,j in lattice_nnn_list] # 自旋xy对应的耦合列表(将Sx与Sy用S+与S-表示则所有矩阵都是实矩阵，但会多出现1/2)；
    nnn_bond_list_zz = [[J2, i, j] for i,j in lattice_nnn_list] # 自旋z对应的耦合列表；

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





def tJ_Model_honeycomb_8site(hole_doping, t1, J1, t2=0, J2=0):
    """
    Honeycomb上8个格点的t-J模型（强关联费米子模型，有相互作用的自旋1/2费米子，并且强制要求无双占据）
    （其中上下格点数为4，左右格点数为4，且上下边界为周期边界，左右边界为开放边界）
    
    Input:
    hole_doping (float): 空穴率，即空穴数与格点数的比值；
    t1 (float,默认为1.0): 最近邻跃迁项(hopping)系数；
    t2 (float,默认为0): 次近邻跃迁项(hopping)系数；
    J1 (float): 最近邻相互作用；
    J2 (float,默认为0): 次近邻相互作用；

    Return:
    H: 返回H是一个智能的、功能齐全的哈密顿量计算引擎(包含了数据(矩阵)和操作(方法)的完整包),能直接执行几乎所有常见的量子多体计算任务；
    basis: 返回basis是一个基对象,包含了所有必要信息的容器(不是单纯基向量列表)，比如格点数N、每个格点维数sps、在给定对称性约束下当前基的维度Ns。
    """
    L = 8 # 格点数
    
    #### 计算粒子数（根据空穴浓度）
    N_total = int(L * (1 - hole_doping))     # 总电子数；
    N_up = int(np.ceil(N_total / 2))    # 上自旋电子数(np.ceil()：向上取整函数)；
    N_down = int(np.floor(N_total / 2))  # 下自旋电子数(np.floor()：向下取整函数)；

    #### 构建基矢：禁止双占据（每个格点最多一个电子）
    basis = spinful_fermion_basis_1d(
        L, 
        Nf=(N_up, N_down), 
        double_occupancy=False
    )

    ### 构建honeycomb的近邻格点指标i、j的列表
    # 最近邻列表
    lattice_nn_list = []
    # honeycomb竖直方向(即y方向)指标列表的构建
    for k in range(0, 5, 4):
        list_y = [(i,(i+1)) for i in range(k, k+3)] # x指标为k时对应的y方向指标列表
        lattice_nn_list.extend(list_y)
        periodic = (k+3,k) # 注：y方向为周期边界
        lattice_nn_list.append(periodic)

    # honeycomb水平方向(即x方向)指标列表的构建
    list_x = [(0,4), (2,6)] # 注：x方向为开放边界
    lattice_nn_list.extend(list_x)

    # 次近邻列表
    lattice_nnn_list = [(0,2), (0,5), (1,3), (1,6), (1,4), (2,0), (2,7), (2,5), (3,6), (3,1), (4,6), (5,7), (6,4), (7,5)]

    #### 定义site-coupling lists
    ### 最近邻hopping项
    hop_nn_left = [[-t1, i, j] for i,j in lattice_nn_list]  # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_j,𝜎)；
    hop_nn_right = [[t1, i, j] for i,j in lattice_nn_list]  # 厄密共轭项：从左向右跃迁项(𝑐†_j,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_j,𝜎)；
    ### 次近邻hopping项
    hop_nnn_left = [[-t2, i, j] for i,j in lattice_nnn_list]  # 直接项：从右向左跃迁项(𝑐†_𝑖,𝜎·𝑐_j,𝜎)；
    hop_nnn_right = [[t2, i, j] for i,j in lattice_nnn_list]  # 厄密共轭项：从左向右跃迁项(𝑐†_j,𝜎·𝑐_𝑖,𝜎= - 𝑐_𝑖,𝜎·c†_j,𝜎)；

    ### 最近邻相互作用项(自旋相互作用项 + 密度相互作用项)：S_i·S_j-1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
    ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
    int_ss = [[J1/2, i, j, i, j] for i,j in lattice_nn_list]
    ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
    int_nn_ij = [[-J1/2, i, j] for i,j in lattice_nn_list] # n_i↑·n_i↓项；
    int_nn_ji = [[-J1/2, j, i] for i,j in lattice_nn_list] # n_j↑·n_i↓项；

    ### 次近邻相互作用项(自旋相互作用项 + 密度相互作用项)S_i·S_j - 1/4 n_i·n_j = 1/2(S^+_i·S^-_j + S^-_i·S^+_j) + S^z_i·S^z_j - 1/4 n_i·n_j
    ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
    int_sss = [[J2/2, i, j, i, j] for i,j in lattice_nnn_list] 
    ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
    int_nnn_ij = [[-J2/2, i, j] for i,j in lattice_nnn_list] # n_i↑·n_i↓项；
    int_nnn_ji = [[-J2/2, j, i] for i,j in lattice_nnn_list] # n_j↑·n_i↓项；

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
        ["+-|-+", int_ss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_j↑·𝑐_𝑖↓·𝑐†_j↓；
        ["-+|+-", int_ss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_j↑·𝑐†_𝑖↓·𝑐_j↓；
        ## J * (S^z_i·S^z_j - 1/4 n_i·n_j) = -J * 1/2(n_i↑·n_j↓ + n_j↑·n_i↓)项
        ["n|n", int_nn_ij],  # n_i↑·n_j↓；
        ["n|n", int_nn_ji],  # n_j↑·n_i↓；

        ### 次近邻相互作用项
        ## J * 1/2(S^+_i·S^-_j + S^-_i·S^+_j)项
        ["+-|-+", int_sss],  # S^+_i·S^-_j = 𝑐†_𝑖↑·𝑐_j↑·𝑐_𝑖↓·𝑐†_j↓；
        ["-+|+-", int_sss],  # S^-_i·S^+_j = 𝑐_𝑖↑·𝑐†_j↑·𝑐†_𝑖↓·𝑐_j↓；
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

