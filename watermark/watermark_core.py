"""
水印嵌入和提取核心模块
包含水印嵌入和提取的主要功能
"""

import time
import numpy as np
from .file_operations import read_obj_vertices, save_obj_with_new_vertices, create_obj_str_with_new_vertices
from .utils import (
    calculate_rmse, cartesian_to_spherical, spherical_to_cartesian, adaptive_partition_bins,
    normalize_bin_rho, denormalize_bin_rho, calculate_k_coefficient,
    generate_fixed_seed_watermark, read_watermark_from_bin, string_to_binary
)


def watermark_embedding(original_obj_path,watermark,N=256):
    """水印嵌入主函数"""
    start_time = time.time()
    print(f"开始水印嵌入...")
    
    vertices, original_vertex_count ,file_name= read_obj_vertices(original_obj_path)
    L = original_vertex_count
    
    if L < N:
        print(f"错误：顶点数{L} < 水印位数{N}")
        exit(1)
    
    centroid = np.mean(vertices, axis=0)
    rho, theta, phi = cartesian_to_spherical(vertices, centroid)
    try:
        rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(rho, N)
    except Exception as e:
        print(f"自适应分bin失败：{str(e)}")
        exit(1)
    
    watermark = watermark
    alpha, delta_k = 0.05, 0.001
    rho_modified = rho_adjusted.copy()
    
    for n in range(N):
        idx_list = bin_indices[n]
        rho_bin = rho_adjusted[idx_list]
        b_min, b_max = bin_rho_min[n], bin_rho_max[n]
        
        rho_norm = normalize_bin_rho(rho_bin, b_min, b_max)
        k, final_mean = calculate_k_coefficient(rho_norm, watermark[n], alpha, delta_k)
        rho_mapped = rho_norm ** k
        rho_denorm = denormalize_bin_rho(rho_mapped, b_min, b_max)
        rho_modified[idx_list] = rho_denorm
        # print(f"第{n}位水印（{watermark[n]}）：k={k:.4f}，最终均值={final_mean:.4f}（顶点数：{len(idx_list)}）")
    
    # 转换回笛卡尔坐标并校验数量
    new_vertices = spherical_to_cartesian(rho_modified, theta, phi, centroid)
    if len(new_vertices) != original_vertex_count:
        raise ValueError(f"坐标转换后顶点数({len(new_vertices)})与原始不符({original_vertex_count})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    rmse=calculate_rmse(vertices, new_vertices)
    file_name_new=file_name.replace(".obj","_watermarked.obj")
    
    return file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse

def watermark_extraction(watermarked_obj_path, N=256):
    """水印提取主函数"""
    start_time = time.time()
    print(f"开始水印提取...")
    
    vertices, _ ,_= read_obj_vertices(watermarked_obj_path)
    centroid = np.mean(vertices, axis=0)
    rho, _, _ = cartesian_to_spherical(vertices, centroid)
    
    try:
        rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(rho, N)
    except Exception as e:
        print(f"提取时自适应分bin失败：{str(e)}")
        return np.array([]), 0.0
    
    extracted = np.zeros(N, dtype=np.uint8)
    for n in range(N):
        idx_list = bin_indices[n]
        rho_bin = rho_adjusted[idx_list]
        b_min, b_max = bin_rho_min[n], bin_rho_max[n]
        
        rho_norm = normalize_bin_rho(rho_bin, b_min, b_max)
        mu = np.mean(rho_norm)
        extracted[n] = 1 if mu >= 0.5 else 0
        # print(f"第{n}位提取：均值={mu:.4f} → 水印位={extracted[n]}（顶点数：{len(idx_list)}）")
    
    # original_watermark=watermark_orginal
    # accuracy = np.mean(extracted == original_watermark) * 100 if len(extracted) == len(original_watermark) else 0.0
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"水印提取完成，耗时: {elapsed_time:.4f} 秒")
    
    return extracted

if __name__ == "__main__":
    # 测试水印嵌入
    watermark = string_to_binary("我直接哈哈哈哈哈哈哈哈哈哈", target_length=256)
    file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse = watermark_embedding("3D_models/aobin.obj", watermark)
    
    print(f"原始文件名: {file_name}")
    print(f"水印后文件名: {file_name_new}")
    print(f"嵌入时间: {elapsed_time:.4f} 秒")
    print(f"RMSE: {rmse:.6f}")
    print(f"水印长度: {len(watermark)} 位")