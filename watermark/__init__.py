"""
3D网格水印系统模块
模块化的3D网格水印嵌入和提取系统

主要功能模块：
- file_operations: 文件读取和保存操作
- utils: 工具函数（坐标转换、分bin、水印生成等）
- watermark_core: 水印嵌入和提取核心功能
- robustness: 鲁棒性测试和攻击模拟
- main: 主函数和示例调用
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "3D Watermark System"
__description__ = "模块化的3D网格水印嵌入和提取系统"

# --- 从子模块导入所有需要的函数和类 ---

# 从 watermark_core 导入核心功能
from .watermark_core import watermark_embedding, watermark_extraction

# 从 file_operations 导入文件操作
from .file_operations import read_obj_vertices, save_obj_with_new_vertices, create_obj_str_with_new_vertices

# 从 utils 导入工具函数
from .utils import generate_fixed_seed_watermark, read_watermark_from_bin, calculate_rmse, string_to_binary, binary_to_string

# 从 robustness 导入鲁棒性测试
from .robustness import attack_rotation, attack_scaling, attack_translation, attack_clipping, attack_noise, attack_smoothing, attack_simplification, attack_vertex_reordering, test_multiple_attacks


# --- 定义 `__all__` 以明确哪些符号应该被 `from watermark import *` 导入 ---
__all__ = [
    # 核心功能
    'watermark_embedding',
    'watermark_extraction',
    'test_multiple_attacks',
    
    # 文件操作
    'read_obj_vertices',
    'save_obj_with_new_vertices',
    'calculate_rmse',
    
    # 工具函数
    'generate_fixed_seed_watermark',
    'read_watermark_from_bin',
    'string_to_binary',
    'binary_to_string',
    
    # 主函数
    'main',
    'run_single_test',
    
    # 元信息
    '__version__',
    '__author__',
    '__description__'
]
