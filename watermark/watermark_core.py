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
    generate_fixed_seed_watermark, read_watermark_from_bin, string_to_binary,
    geometric_median
)


def watermark_embedding(original_obj_path, watermark, N=256, binning_method="equal_frequency"):
    """
    水印嵌入主函数
    
    参数:
        original_obj_path: str, 原始3D模型路径
        watermark: np.array, 水印比特序列
        N: int, 水印比特数
        binning_method: str, 分bin方法，可选 "equal_width"(等宽), "equal_frequency"(等频), "kde"(核密度估计)
    """
    start_time = time.time()
    print(f"开始水印嵌入，使用{binning_method}分bin方法...")
    
    vertices, original_vertex_count, file_name = read_obj_vertices(original_obj_path)
    L = original_vertex_count
    
    if L < N:
        print(f"错误：顶点数{L} < 水印位数{N}")
        exit(1)
    
    # centroid = np.mean(vertices, axis=0)
    centroid = geometric_median(vertices)
    rho, theta, phi = cartesian_to_spherical(vertices, centroid)
    try:
        rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(rho, N, method=binning_method)
        
        # 打印每个bin的顶点数统计，用于评估分bin效果
        bin_counts = [len(indices) for indices in bin_indices]
        print(f"分bin统计: 最小={min(bin_counts)}顶点, 最大={max(bin_counts)}顶点, 平均={np.mean(bin_counts):.1f}顶点")
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
    rmse = calculate_rmse(vertices, new_vertices)
    file_name_new = file_name.replace(".obj", f"_watermarked_{binning_method}.obj")
    
    return file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse

def watermark_extraction(watermarked_obj_path, N=256, binning_method="equal_frequency"):
    """
    水印提取主函数
    
    参数:
        watermarked_obj_path: str, 带水印3D模型路径
        N: int, 水印比特数
        binning_method: str, 分bin方法，可选 "equal_width"(等宽), "equal_frequency"(等频), "kde"(核密度估计)
    """
    start_time = time.time()
    print(f"开始水印提取，使用{binning_method}分bin方法...")
    
    vertices, _, _ = read_obj_vertices(watermarked_obj_path)
    # centroid = np.mean(vertices, axis=0)
    centroid = geometric_median(vertices)
    rho, _, _ = cartesian_to_spherical(vertices, centroid)
    
    try:
        rho_adjusted, bin_indices, bin_rho_min, bin_rho_max = adaptive_partition_bins(rho, N, method=binning_method)
        
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
    
    # 使用等频分bin方法（默认）
    file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse = watermark_embedding(
        "static/uploads/aobin.obj", watermark, binning_method="equal_frequency"
    )
    
    print(f"原始文件名: {file_name}")
    print(f"水印后文件名: {file_name_new}")
    print(f"嵌入时间: {elapsed_time:.4f} 秒")
    print(f"RMSE: {rmse:.6f}")
    print(f"水印长度: {len(watermark)} 位")

    # 测试水印提取
    extracted = watermark_extraction(file_name_new, N=256, binning_method="equal_frequency")
    print(f"提取水印: {extracted}")
    accuracy = np.mean(extracted == watermark) * 100 if len(extracted) == len(watermark) else 0.0
    print(f"提取准确率: {accuracy:.2f}%")
    
    # 也可以尝试KDE自适应分bin
    # file_name, file_name_new, new_vertices, watermark, elapsed_time, rmse = watermark_embedding(
    #     "3D_models/aobin.obj", watermark, binning_method="kde"
    # )
    # print(f"原始文件名: {file_name}")
    # print(f"水印后文件名: {file_name_new}")
    # print(f"嵌入时间: {elapsed_time:.4f} 秒")
    # print(f"RMSE: {rmse:.6f}")
    # print(f"水印长度: {len(watermark)} 位")