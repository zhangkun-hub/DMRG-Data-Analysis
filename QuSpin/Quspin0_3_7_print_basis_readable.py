from quspin.basis import spinful_fermion_basis_1d

# ====================================================================================================================================
#  QuSpin 0.3.7：basis的量子态整数编码格式：
#    state_int = (spin_up_bits << L) + spin_down_bits  注：spin_up_bits << L表示将spin_up_bits的二进制向前移动L位，而后面则跟着L个0
#  
#    即state_int表示将L格点的有1/2自旋费米子量子态按二进制比特分成两部分(共2L长度比特)：
#        高位 L 比特(前L位比特) = ↑ 自旋
#        低位 L 比特(后L位比特) = ↓ 自旋
#    
#    比如：L=2的量子态，其state_int = 1100对应的真实物理量子态为：|↑,↑>
# =====================================================================================================================================

def decode_spinful_state_v037(state_int, L):
    """
    从 QuSpin 0.3.7 的整数编码解析 ↑ / ↓ 占据
    注意：1. QuSpin输出的basis的量子态state_int = (spin_up_bits << L) + spin_down_bits;
        2. state_int共 2L 位
    """
    ### 获得QuSpin输出的basis的量子态state_int的高、低L比特
    # 低位 L 比特
    down_bits = state_int & ((1 << L) - 1) # (1<<L)-1表示将1前移L位(后面为L个0),然后减1得到后L位全为1的掩码,再通过 & 就可以提取到state_int的后L位
    # 高位 L 比特
    up_bits = state_int >> L # state_int >> L表示将state_int向后移L位，相当于丢弃最低的L位，提取剩余的，即state_int的前L位部分

    ### 将up_bits与down_bits的二进制比特转换成列表储存
    up = [(up_bits >> j) & 1 for j in range(L)] # j从0取到L-1，而up列表的第j元素是up_bits向后移动j位再通过与1进行 & 操作，从而将up_bits的比特按列表储存
    up.reverse()
    down = [(down_bits >> j) & 1 for j in range(L)]
    down.reverse()
    
    return up, down


def occupation_to_symbol(up, down):
    """把 (up_i, down_i) 转成 |↑↓>, |↑>, |↓>, |0>"""
    if up == 1 and down == 1:
        return "↑↓"
    elif up == 1 and down == 0:
        return "↑"
    elif up == 0 and down == 1:
        return "↓"
    else:
        return "0"


def print_basis_readable(basis):
    """打印体系basis的完整基矢，以可读格式显示，如 |↑↓>|0>"""
    L = basis.L
    print("基矢列表: ")
    for i, state_int in enumerate(basis.states): # enumerate()函数为可迭代对象的每个元素添加索引，返回一个生成器，产生(索引, 元素)的元组
        up, down = decode_spinful_state_v037(state_int, L)
        
        symbols = [occupation_to_symbol(up[j], down[j]) for j in range(L)]
        ket = ">|".join(symbols) # ">|".join(symbols)：用">|"作为分隔符连接所有符号。比如symbols = ["0", "1", "0"]，则结果："0>|1>|0"

        print(f"state {i+1}: int={state_int:0{2*L}b}, |{ket}>") # :0{2*L}b：b表示将state_int格式化为二进制; 2*L表示总宽度为2L位; 0表示用0填充空白

        
def basis_vector_readable(basis, i):
    """打印第i个基向量，以可读格式显示，如 |↑↓>|0>"""
    L = basis.L
    state_int = basis.states[i]
    up, down = decode_spinful_state_v037(state_int, L)
    symbols = [occupation_to_symbol(up[j], down[j]) for j in range(L)]
    ket = ">|".join(symbols) # ">|".join(symbols)：用">|"作为分隔符连接所有符号。比如symbols = ["0", "1", "0"]，则结果："0>|1>|0"
    state = f"|{ket}>"

    return state 

        
        
# ================== 测试 ==================
if __name__ == "__main__":
    L = 2
    Nf = [(0,0), (2,0)]
    basis = spinful_fermion_basis_1d(L=L, Nf=Nf, double_occupancy=False)
    
    print("QuSpin 内部 states =", basis.states)
    print_basis_readable(basis)
