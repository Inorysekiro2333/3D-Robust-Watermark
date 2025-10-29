"""
文件读取和保存操作模块
包含OBJ文件的读取、保存、顶点操作等功能
"""

import numpy as np
import os


def read_obj_vertices(obj_path):
    """直接从OBJ文件解析顶点，确保与原始顶点数完全一致（不合并重复顶点）"""
    vertices = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):  # 只处理顶点行
                parts = line.split()
                if len(parts) >= 4:  # 确保有x,y,z坐标
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    vertices.append([x, y, z])
    vertices_np = np.array(vertices, dtype=np.float64)
    file_name = os.path.basename(obj_path)    
    return vertices_np, len(vertices_np) ,file_name


def save_obj_with_new_vertices(original_obj_path, new_vertices, output_obj_path):
    """保留原始OBJ拓扑，替换顶点坐标并保存（严格匹配顶点数）"""
    with open(original_obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    vertex_idx = 0
    original_vertex_count = sum(1 for line in lines if line.startswith("v "))
    
    # 校验新顶点数是否匹配
    if len(new_vertices) != original_vertex_count:
        raise ValueError(f"新顶点数({len(new_vertices)})与原始顶点数({original_vertex_count})不匹配")
    
    for line in lines:
        if line.startswith("v "):
            if vertex_idx >= len(new_vertices):
                raise IndexError(f"顶点索引{vertex_idx}超出新顶点数组长度{len(new_vertices)}")
            x, y, z = new_vertices[vertex_idx]
            new_lines.append(f"v {x:.10f} {y:.10f} {z:.10f}\n")
            vertex_idx += 1
        else:
            new_lines.append(line)
    
    with open(output_obj_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)



def read_obj_faces(obj_path):
    """读取OBJ文件中的面信息"""
    faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("f "):
                parts = line.split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # 转换为0索引
                faces.append(face)
    
    # 不直接转换为NumPy数组，而是返回列表，避免不同长度面片导致的错误
    return faces


def create_obj_str_with_new_vertices(original_obj_path, new_vertices):
    """保留原始OBJ拓扑，替换顶点坐标并返回OBJ文件内容的字符串（严格匹配顶点数）"""
    with open(original_obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    vertex_idx = 0
    original_vertex_count = sum(1 for line in lines if line.startswith("v "))
    
    # 校验新顶点数是否匹配
    if len(new_vertices) != original_vertex_count:
        raise ValueError(f"新顶点数({len(new_vertices)})与原始顶点数({original_vertex_count})不匹配")
    
    for line in lines:
        if line.startswith("v "):
            if vertex_idx >= len(new_vertices):
                raise IndexError(f"顶点索引{vertex_idx}超出新顶点数组长度{len(new_vertices)}")
            x, y, z = new_vertices[vertex_idx]
            new_lines.append(f"v {x:.10f} {y:.10f} {z:.10f}\n")
            vertex_idx += 1
        else:
            new_lines.append(line)
    
    return "".join(new_lines)
