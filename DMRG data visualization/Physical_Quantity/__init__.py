# -*- coding: utf-8 -*-
"""
Physical_Quantity 包兼容层（P1 目录重组：四层架构子包化）。

本包 21 个模块已按四层架构分入 4 个子包：
- data_records/        数据与记录层（Honeycomb_Lattice、Physical_Data_Record、DMRG_Log_Parser）
- physical_analysis/   物理量分析层（Process/Analyzer/Structure_Factor 共 14 个）
- evidence_integration/ 证据整合层（Analysis_Contract、Analysis_Integration、Analysis_Entry）
- phase_diagnosis/     相态诊断层（Phase_Diagnosis）

旧路径导入（`from Physical_Quantity.<模块名> import ...`）由本文件经
sys.modules 别名注册继续支持——下游（35 套测试、根脚本、notebooks）
零改动。语义说明：
- 别名模块的 __name__ 保持真实路径（如
  Physical_Quantity.data_records.Physical_Data_Record）；全仓库无
  pickle / __module__ 断言依赖，两种路径 import 得到同一个模块对象
  （A is B，由 sys.modules 单对象保证）。
- 代价：import 本包任意模块会热导入全部 21 个模块（单进程约 1-2 秒）。
  备选（meta path finder 懒别名）记录为 fallback，不默认采用。
- 铁律：跨层上溯（physical_analysis → evidence_integration）只允许
  函数内延迟 import；模块级依赖必须保持 DAG。
"""
import sys

# ---- 1. 子包导入（子包 __init__ 为空，无副作用） ----
from . import data_records, physical_analysis, evidence_integration, \
    phase_diagnosis

# ---- 2. 真实模块导入（严格拓扑序，防循环 import；延迟 import 不执行） ----
from .evidence_integration import Analysis_Contract as _Analysis_Contract
from .data_records import Honeycomb_Lattice as _Honeycomb_Lattice
from .data_records import DMRG_Log_Parser as _DMRG_Log_Parser
from .physical_analysis import Energy_Process as _Energy_Process
from .physical_analysis import Correlation_Decay_Analyzer as _Correlation_Decay_Analyzer
from .physical_analysis import Spin_Structure_Factor_Analyzer as _Spin_Structure_Factor_Analyzer
from .data_records import Physical_Data_Record as _Physical_Data_Record
from .physical_analysis import Correlation_Process as _Correlation_Process
from .physical_analysis import Super_Correlation_Process as _Super_Correlation_Process
from .physical_analysis import Density_Process as _Density_Process
from .physical_analysis import Spin_Structure_Factor as _Spin_Structure_Factor
from .physical_analysis import Spin_Correlation_Analyzer as _Spin_Correlation_Analyzer
from .physical_analysis import Super_Correlation_Analyzer as _Super_Correlation_Analyzer
from .physical_analysis import Charge_Correlation_Analyzer as _Charge_Correlation_Analyzer
from .physical_analysis import Green_Function_Analyzer as _Green_Function_Analyzer
from .physical_analysis import Charge_Structure_Factor as _Charge_Structure_Factor
from .physical_analysis import SC_Pair_Structure_Factor as _SC_Pair_Structure_Factor
from .physical_analysis import Pairing_Symmetry_Analyzer as _Pairing_Symmetry_Analyzer
from .evidence_integration import Analysis_Integration as _Analysis_Integration
from .evidence_integration import Analysis_Entry as _Analysis_Entry
from .phase_diagnosis import Phase_Diagnosis as _Phase_Diagnosis

# ---- 3. 旧路径别名一次性注册（无"半注册"窗口） ----
_ALIASES = {
    'Analysis_Contract': _Analysis_Contract,
    'Honeycomb_Lattice': _Honeycomb_Lattice,
    'DMRG_Log_Parser': _DMRG_Log_Parser,
    'Energy_Process': _Energy_Process,
    'Correlation_Decay_Analyzer': _Correlation_Decay_Analyzer,
    'Spin_Structure_Factor_Analyzer': _Spin_Structure_Factor_Analyzer,
    'Physical_Data_Record': _Physical_Data_Record,
    'Correlation_Process': _Correlation_Process,
    'Super_Correlation_Process': _Super_Correlation_Process,
    'Density_Process': _Density_Process,
    'Spin_Structure_Factor': _Spin_Structure_Factor,
    'Spin_Correlation_Analyzer': _Spin_Correlation_Analyzer,
    'Super_Correlation_Analyzer': _Super_Correlation_Analyzer,
    'Charge_Correlation_Analyzer': _Charge_Correlation_Analyzer,
    'Green_Function_Analyzer': _Green_Function_Analyzer,
    'Charge_Structure_Factor': _Charge_Structure_Factor,
    'SC_Pair_Structure_Factor': _SC_Pair_Structure_Factor,
    'Pairing_Symmetry_Analyzer': _Pairing_Symmetry_Analyzer,
    'Analysis_Integration': _Analysis_Integration,
    'Analysis_Entry': _Analysis_Entry,
    'Phase_Diagnosis': _Phase_Diagnosis,
}
for _name, _module in _ALIASES.items():
    sys.modules[f'{__name__}.{_name}'] = _module
    setattr(sys.modules[__name__], _name, _module)   # 兜底：from Physical_Quantity import X
del _name, _module, _ALIASES
