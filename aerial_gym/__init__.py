import os
import sys

import numpy as np

# 全局修复：部分第三方库仍引用已弃用的 NumPy 标量别名。
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

# 如果早前导入失败留下了不完整的 networkx 模块，这里清除以便重新加载。
sys.modules.pop("networkx", None)

import isaacgym

AERIAL_GYM_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

from .task import *
from .env_manager import *
from .robots import *
from .control import *
from .registry import *
from .utils import *
from .config import *
