"""
网格简化模块
提供3D网格简化和映射功能
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

def simplify_mesh(vertices, faces, target_ratio=0.125, min_vertices=2000):
    """
    使用Open3D的Quadric Edge Collapse简化网格
    """
    # 验证参数
    if target_ratio <= 0 or target_ratio > 1:
        print(f"警告：target_ratio={target_ratio}超出有效范围(0,1]，已调整为0.125")
        target_ratio = 0.125
    
    # 原始面片数
    original_face_count = len(faces)
    
    # 计算目标面片数 (通常简化目标是面片数，而不是顶点数)
    # 计算目标顶点数
    target_vertices = max(int(len(vertices) * target_ratio), min_vertices)
    # 粗略估计目标面片数 (假设面片数与顶点数比例不变)
    target_faces = int(original_face_count * (target_vertices / len(vertices)))
    
    # 转换为 Open3D Mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # 简化网格
    # Open3D的simplify_quadric_decimation需要目标面片数
    coarse_o3d_mesh = o3d_mesh.simplify_quadric_decimation(
        target_faces
    )
    
    # 获取简化后的顶点和面片
    coarse_vertices = np.asarray(coarse_o3d_mesh.vertices)
    coarse_faces = np.asarray(coarse_o3d_mesh.triangles)
    
    # 创建原始顶点到简化顶点的映射
    # ... (保持 create_vertex_mapping 函数不变) ...
    vertex_mapping = create_vertex_mapping(vertices, coarse_vertices, coarse_faces)
    
    print(f"网格简化: 原始顶点数={len(vertices)}, 简化后顶点数={len(coarse_vertices)}, 比例={len(coarse_vertices)/len(vertices):.4f}")
    
    return coarse_vertices, coarse_faces, vertex_mapping

def create_vertex_mapping(original_vertices, coarse_vertices, coarse_faces):
    """
    创建原始顶点到简化顶点的映射关系
    
    参数:
        original_vertices: np.array, 原始顶点坐标
        coarse_vertices: np.array, 简化后的顶点坐标
        coarse_faces: np.array, 简化后的面片索引
        
    返回:
        vertex_mapping: dict, 原始顶点到简化顶点的映射关系
    """
    # 构建KD树用于最近邻搜索
    kdtree = KDTree(coarse_vertices)
    
    # 为每个原始顶点找到最近的简化顶点
    distances, indices = kdtree.query(original_vertices)
    
    # 创建映射字典
    vertex_mapping = {i: int(indices[i]) for i in range(len(original_vertices))}
    
    return vertex_mapping

def propagate_modifications(original_vertices, coarse_vertices_original, coarse_vertices_modified, vertex_mapping):
    """
    将简化网格上的修改传播回原始网格
    
    参数:
        original_vertices: np.array, 原始顶点坐标
        coarse_vertices_original: np.array, 简化前的顶点坐标
        coarse_vertices_modified: np.array, 修改后的简化顶点坐标
        vertex_mapping: dict, 原始顶点到简化顶点的映射关系
        
    返回:
        modified_vertices: np.array, 修改后的原始顶点坐标
    """
    # 计算简化网格上的偏移量
    coarse_offsets = coarse_vertices_modified - coarse_vertices_original
    
    # 创建修改后的原始顶点数组
    modified_vertices = original_vertices.copy()
    
    # 将偏移量应用到原始顶点
    for i in range(len(original_vertices)):
        coarse_idx = vertex_mapping[i]
        modified_vertices[i] = original_vertices[i] + coarse_offsets[coarse_idx]
    
    return modified_vertices