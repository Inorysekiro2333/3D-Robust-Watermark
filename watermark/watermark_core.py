"""
水印嵌入和提取核心模块
包含水印嵌入和提取的主要功能
"""

import time
import numpy as np
from .file_operations import read_obj_vertices, read_obj_faces, save_obj_with_new_vertices, create_obj_str_with_new_vertices
from .utils import (
    calculate_rmse, cartesian_to_spherical, spherical_to_cartesian, adaptive_partition_bins,
    normalize_bin_rho, denormalize_bin_rho, calculate_k_coefficient,
    generate_fixed_seed_watermark, read_watermark_from_bin, string_to_binary
)
from .mesh_simplification import simplify_mesh, propagate_modifications


def watermark_embedding(original_obj_path, watermark, N=256, use_multiresolution=True, target_ratio=0.125, min_vertices=2000):
    """
    水印嵌入主函数
    
    参数:
        original_obj_path: str, 原始3D模型路径
        watermark: np.array, 水印比特序列
        N: int, 水印比特数
        use_multiresolution: bool, 是否使用多分辨率方案
        target_ratio: float, 简化网格的目标比例 (0.125 = 1/8)
        min_vertices: int, 简化网格的最小顶点数
    """
    start_time = time.time()
    
    # 读取原始模型
    vertices, original_vertex_count, file_name = read_obj_vertices(original_obj_path)
    L = original_vertex_count
    
    if L < N:
        print(f"错误：顶点数{L} < 水印位数{N}")
        exit(1)
    
    if use_multiresolution:
        print(f"开始水印嵌入，使用多分辨率+均分bin方案 (简化比例={target_ratio}, 最小顶点数={min_vertices})...")
        
        # 读取面片信息
        faces = read_obj_faces(original_obj_path)
        
        # 简化网格
        coarse_vertices, coarse_faces, vertex_mapping = simplify_mesh(
            vertices, faces, target_ratio, min_vertices
        )
        
        # 在简化网格上嵌入水印
        coarse_centroid = np.mean(coarse_vertices, axis=0)
        coarse_rho, coarse_theta, coarse_phi = cartesian_to_spherical(coarse_vertices, coarse_centroid)
        
        try:
            coarse_rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(
                coarse_rho, N, method="equal_width"
            )
            
            # 打印每个bin的顶点数统计
            bin_counts = [len(indices) for indices in bin_indices]
            print(f"粗网格分bin统计: 最小={min(bin_counts)}顶点, 最大={max(bin_counts)}顶点, 平均={np.mean(bin_counts):.1f}顶点")
        except Exception as e:
            print(f"粗网格自适应分bin失败：{str(e)}")
            exit(1)
        
        # 嵌入水印
        alpha, delta_k = 0.05, 0.001
        coarse_rho_modified = coarse_rho_adjusted.copy()
        
        for n in range(N):
            idx_list = bin_indices[n]
            rho_bin = coarse_rho_adjusted[idx_list]
            b_min, b_max = bin_rho_min[n], bin_rho_max[n]
            
            rho_norm = normalize_bin_rho(rho_bin, b_min, b_max)
            k, final_mean = calculate_k_coefficient(rho_norm, watermark[n], alpha, delta_k)
            rho_mapped = rho_norm ** k
            rho_denorm = denormalize_bin_rho(rho_mapped, b_min, b_max)
            coarse_rho_modified[idx_list] = rho_denorm
        
        # 转换回笛卡尔坐标
        coarse_new_vertices = spherical_to_cartesian(coarse_rho_modified, coarse_theta, coarse_phi, coarse_centroid)
        
        # 将修改传播回原始网格
        new_vertices = propagate_modifications(vertices, coarse_vertices, coarse_new_vertices, vertex_mapping)
        
    else:
        print("开始水印嵌入，使用equal_width分bin方法...")
        
        # 原始方法：直接在原始网格上嵌入
        centroid = np.mean(vertices, axis=0)
        rho, theta, phi = cartesian_to_spherical(vertices, centroid)
        
        try:
            rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(
                rho, N, method="equal_width"
            )
            
            # 打印每个bin的顶点数统计，用于评估分bin效果
            bin_counts = [len(indices) for indices in bin_indices]
            print(f"分bin统计: 最小={min(bin_counts)}顶点, 最大={max(bin_counts)}顶点, 平均={np.mean(bin_counts):.1f}顶点")
        except Exception as e:
            print(f"自适应分bin失败：{str(e)}")
            exit(1)
        
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
        
        # 转换回笛卡尔坐标
        new_vertices = spherical_to_cartesian(rho_modified, theta, phi, centroid)
    
    # 校验顶点数量
    if len(new_vertices) != original_vertex_count:
        raise ValueError(f"坐标转换后顶点数({len(new_vertices)})与原始不符({original_vertex_count})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    rmse = calculate_rmse(vertices, new_vertices)
    file_name_new = file_name.replace(".obj", "_watermarked.obj")
    
    return file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse

def watermark_extraction(watermarked_obj_path, N=256, use_multiresolution=True, target_ratio=0.125, min_vertices=2000):
    """
    水印提取主函数
    
    参数:
        watermarked_obj_path: str, 带水印3D模型路径
        N: int, 水印比特数
        use_multiresolution: bool, 是否使用多分辨率方案
        target_ratio: float, 简化网格的目标比例 (0.125 = 1/8)
        min_vertices: int, 简化网格的最小顶点数
    """
    start_time = time.time()
    
    # 读取模型
    vertices, vertex_count, _ = read_obj_vertices(watermarked_obj_path)
    L = vertex_count
    
    if L < N:
        print(f"错误：顶点数{L} < 水印位数{N}")
        return np.array([]), 0.0
    
    if use_multiresolution:
        print(f"开始水印提取，使用多分辨率+均分bin方案 (简化比例={target_ratio}, 最小顶点数={min_vertices})...")
        
        # 读取面片信息
        faces = read_obj_faces(watermarked_obj_path)
        
        # 简化网格
        coarse_vertices, coarse_faces, _ = simplify_mesh(
            vertices, faces, target_ratio, min_vertices
        )
        
        # 在简化网格上提取水印
        coarse_centroid = np.mean(coarse_vertices, axis=0)
        coarse_rho, _, _ = cartesian_to_spherical(coarse_vertices, coarse_centroid)
        
        try:
            coarse_rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(
                coarse_rho, N, method="equal_width"
            )
            
            # 打印每个bin的顶点数统计
            bin_counts = [len(indices) for indices in bin_indices]
            print(f"粗网格分bin统计: 最小={min(bin_counts)}顶点, 最大={max(bin_counts)}顶点, 平均={np.mean(bin_counts):.1f}顶点")
        except Exception as e:
            print(f"粗网格自适应分bin失败：{str(e)}")
            return np.array([]), 0.0
        
        extracted = np.zeros(N, dtype=np.uint8)
        
        for n in range(N):
            idx_list = bin_indices[n]
            rho_bin = coarse_rho_adjusted[idx_list]
            b_min, b_max = bin_rho_min[n], bin_rho_max[n]
            
            rho_norm = normalize_bin_rho(rho_bin, b_min, b_max)
            mu = np.mean(rho_norm)
            extracted[n] = 1 if mu >= 0.5 else 0
    
    else:
        print("开始水印提取，使用equal_width分bin方法...")
        
        # 原始方法：直接在原始网格上提取
        centroid = np.mean(vertices, axis=0)
        rho, _, _ = cartesian_to_spherical(vertices, centroid)
        
        try:
            rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(
                rho, N, method="equal_width"
            )
            
            # 打印每个bin的顶点数统计，用于评估分bin效果
            bin_counts = [len(indices) for indices in bin_indices]
            print(f"提取时分bin统计: 最小={min(bin_counts)}顶点, 最大={max(bin_counts)}顶点, 平均={np.mean(bin_counts):.1f}顶点")
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
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"水印提取完成，耗时: {elapsed_time:.4f} 秒")
    
    return extracted

if __name__ == "__main__":
    # 测试水印嵌入
    watermark = string_to_binary("我直接哈哈哈哈哈哈哈哈哈哈", target_length=256)
    
    # 设置参数
    original_obj_path = "static/uploads/aobin.obj"  # 原始模型路径
    N = 256  # 水印比特数
    use_multiresolution = True  # 是否使用多分辨率方案
    target_ratio = 0.125  # 简化网格的目标比例 (1/8)
    min_vertices = 2000  # 简化网格的最小顶点数
    
    # 使用多分辨率方案嵌入水印
    file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse = watermark_embedding(
        original_obj_path, watermark, N, use_multiresolution, target_ratio, min_vertices
    )
    
    # 保存带水印模型
    save_obj_with_new_vertices(f"static/results/{file_name_new}", f"static/uploads/{file_name}", new_vertices)
    print(f"水印嵌入完成，耗时: {elapsed_time:.4f} 秒, RMSE: {rmse:.8f}")
    
    # 测试水印提取
    extracted = watermark_extraction(
        f"static/results/{file_name_new}", N, use_multiresolution, target_ratio, min_vertices
    )
    
    # 计算准确率
    accuracy = np.mean(extracted == watermark) * 100
    print(f"水印提取准确率: {accuracy:.2f}%")
    
    print(f"原始文件名: {file_name}")
    print(f"水印后文件名: {file_name_new}")
    print(f"嵌入时间: {elapsed_time:.4f} 秒")
    print(f"RMSE: {rmse:.6f}")
    print(f"水印长度: {len(watermark)} 位")

    # 测试不同简化比例的效果
    print("\n测试不同简化比例的效果:")
    for ratio in [0.25, 0.125, 0.0625]:
        print(f"\n简化比例: {ratio} (1/{int(1/ratio)})")
        extracted = watermark_extraction(
            f"static/results/{file_name_new}", N, True, ratio, min_vertices
        )
        correct = np.sum(extracted == watermark)
        accuracy = correct / N * 100
        print(f"准确率: {accuracy:.2f}% ({correct}/{N})")
        
    # 测试不使用多分辨率方案的效果
    print("\n不使用多分辨率方案的效果:")
    extracted = watermark_extraction(
        f"static/results/{file_name_new}", N, False
    )
    correct = np.sum(extracted == watermark)
    accuracy = correct / N * 100
    print(f"准确率: {accuracy:.2f}% ({correct}/{N})")