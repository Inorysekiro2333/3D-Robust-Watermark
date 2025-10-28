"""
工具函数模块
包含坐标转换、分bin、水印生成等基础功能
"""

import numpy as np


def cartesian_to_spherical(vertices, centroid):
    """笛卡尔坐标转球坐标（ρ, θ, φ）"""
    dx = vertices[:, 0] - centroid[0]
    dy = vertices[:, 1] - centroid[1]
    dz = vertices[:, 2] - centroid[2]
    
    rho = np.sqrt(dx**2 + dy**2 + dz**2)
    theta = np.arctan2(dy, dx)  # 范围(-π, π)
    rho_safe = np.where(rho < 1e-12, 1e-12, rho)
    phi = np.arccos(dz / rho_safe)  # 范围(0, π)
    
    return rho, theta, phi


def spherical_to_cartesian(rho, theta, phi, centroid):
    """球坐标转笛卡尔坐标（确保输出顶点数与输入一致）"""
    if not (len(rho) == len(theta) == len(phi)):
        raise ValueError("rho, theta, phi数组长度必须一致")
    
    x = rho * np.cos(theta) * np.sin(phi) + centroid[0]
    y = rho * np.sin(theta) * np.sin(phi) + centroid[1]
    z = rho * np.cos(phi) + centroid[2]
    return np.column_stack((x, y, z))


def adaptive_partition_bins(rho, N=256, low_q=0.05, high_q=0.95):
    """
    自适应等量分 bin（基于分位数密集区间）
    
    参数:
        rho: np.array，顶点球坐标模长
        N: int，目标分 bin 数
        low_q, high_q: float，去除尾部稀疏点的分位数
    
    返回:
        final_bin: list of list，每个 bin 内的原始索引
        bin_rho_min: list，每个 bin 的最小 rho
        bin_rho_max: list，每个 bin 的最大 rho
    """
    rho = np.asarray(rho, dtype=np.float64)
    L = len(rho)
    if L < N:
        raise ValueError(f"顶点数 {L}={L} < 分 bin 数 N={N}，无法分箱")
    
    # 1️⃣ 计算密集区间
    dense_r_min, dense_r_max = np.quantile(rho, [low_q, high_q])
    mask_dense = (rho >= dense_r_min) & (rho <= dense_r_max)
    r_dense = rho[mask_dense]
    idx_dense = np.where(mask_dense)[0]

    # 2️⃣ 排序，等量分 bin
    sorted_order = np.argsort(r_dense)
    sorted_r = r_dense[sorted_order]
    sorted_idx = idx_dense[sorted_order]

    pts_per_bin = len(sorted_r) // N
    remainder = len(sorted_r) % N

    final_bin = []
    bin_rho_min = []
    bin_rho_max = []

    start = 0
    for i in range(N):
        end = start + pts_per_bin + (1 if i < remainder else 0)
        indices = sorted_idx[start:end].tolist()
        if len(indices) == 0:
            # 安全兜底
            indices = []
        final_bin.append(indices)
        if len(indices) > 0:
            bin_rho_min.append(float(np.min(rho[indices])))
            bin_rho_max.append(float(np.max(rho[indices])))
        else:
            bin_rho_min.append(float(sorted_r[start]))
            bin_rho_max.append(float(sorted_r[start]))
        start = end

    return rho, final_bin, bin_rho_min, bin_rho_max


def normalize_bin_rho(rho_bin, bin_min, bin_max):
    """归一化bin内rho到[0,1]"""
    if bin_max - bin_min < 1e-12:
        return np.zeros_like(rho_bin, dtype=np.float64)
    return (rho_bin - bin_min) / (bin_max - bin_min)


def denormalize_bin_rho(rho_norm, bin_min, bin_max):
    """逆归一化并校验边界"""
    rho_denorm = rho_norm * (bin_max - bin_min) + bin_min
    return np.clip(rho_denorm, bin_min, bin_max)


def calculate_k_coefficient(rho_norm_bin, watermark_bit, alpha=0.05, delta_k=0.001):
    """迭代计算k值"""
    k = 1.0
    target_mean = 0.5 + alpha if watermark_bit == 1 else 0.5 - alpha
    
    while True:
        rho_mapped = rho_norm_bin ** k
        current_mean = np.mean(rho_mapped)
        
        if watermark_bit == 1:
            if current_mean >= target_mean:
                break
            k -= delta_k
        else:
            if current_mean <= target_mean:
                break
            k += delta_k
        
        if k <= 0.001 or k > 1000:
            break
    
    return k, current_mean


def generate_fixed_seed_watermark(N, seed=42):
    """生成固定种子的N bits水印，返回数组并保存为二进制文件"""
    np.random.seed(seed)
    watermark = np.random.randint(0, 2, size=N, dtype=np.uint8)
    
    byte_len = (N + 7) // 8
    watermark_bytes = bytearray(byte_len)
    for i in range(N):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        watermark_bytes[byte_idx] |= (watermark[i] << bit_idx)
    
    with open("original_watermark.bin", "wb") as f:
        f.write(watermark_bytes)
    return watermark


def read_watermark_from_bin(N, bin_path="original_watermark.bin"):
    """从二进制文件读取N bits水印"""
    with open(bin_path, "rb") as f:
        watermark_bytes = f.read()
    
    watermark = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        watermark[i] = (watermark_bytes[byte_idx] >> bit_idx) & 1
    return watermark

def string_to_binary(input_string, target_length=256):
    """
    将字符串转换为固定长度的二进制数组（256位）
    
    参数:
        input_string: str，输入字符串
        target_length: int，目标二进制长度，默认256
    
    返回:
        binary_array: np.array，256位二进制数组
    """
    # 将字符串转换为字节
    string_bytes = input_string.encode('utf-8')
    
    # 转换为二进制数组
    binary_list = []
    for byte in string_bytes:
        # 将每个字节转换为8位二进制
        for i in range(7, -1, -1):  # 从高位到低位
            binary_list.append((byte >> i) & 1)
    
    original_length = len(binary_list)
    
    # 处理长度：超过256位则截取前256位，不足则补0
    if len(binary_list) > target_length:
        print(f"警告：字符串转换后长度为 {original_length} 位，超过目标长度 {target_length} 位，将截取前 {target_length} 位")
        binary_list = binary_list[:target_length]
    elif len(binary_list) < target_length:
        print(f"提示：字符串转换后长度为 {original_length} 位，少于目标长度 {target_length} 位，将在末尾补 {target_length - original_length} 个0")
        # 补0到256位
        binary_list.extend([0] * (target_length - len(binary_list)))
    else:
        print(f"字符串转换后长度正好为 {target_length} 位，无需调整")
    
    binary_array = np.array(binary_list, dtype=np.uint8)
    return binary_array

def binary_to_string(binary_array):
    """
    将256位二进制数组转换为字符串
    
    参数:
        binary_array: np.array，256位二进制数组（0和1组成）
    
    返回:
        result_string: str，转换后的字符串
    """
    if len(binary_array) == 0:
        return ""
    
    # 检查输入长度是否为256位
    if len(binary_array) != 256:
        raise ValueError(f"输入长度错误：期望256位，实际输入{len(binary_array)}位")
    
    # 将二进制数组转换为字节
    byte_list = []
    for i in range(0, len(binary_array), 8):
        byte_bits = binary_array[i:i+8]
        # 将8位二进制转换为字节
        byte_value = 0
        for j, bit in enumerate(byte_bits):
            byte_value |= (bit << (7 - j))
        byte_list.append(byte_value)
    
    # 使用替换策略进行 UTF-8 解码（非法序列以 替代）
    result_string = bytes(byte_list).decode('utf-8', errors='replace')
    # 去除末尾的null字符（补0产生的）
    result_string = result_string.rstrip('\x00')
    return result_string


def calculate_rmse(vertices_original, vertices_watermarked):
    """计算原始模型与含水印模型的顶点坐标RMSE"""
    if len(vertices_original) != len(vertices_watermarked):
        raise ValueError("原始模型与含水印模型顶点数必须一致")
    squared_errors = np.sum((vertices_original - vertices_watermarked) **2, axis=1)
    return np.sqrt(np.mean(squared_errors))

def bit_accuracy(w_in: str, w_out: str) -> float:

    if len(w_in) != len(w_out):
        raise ValueError("输入串长度必须相同")
    matches = sum(a == b for a, b in zip(w_in, w_out))
    return matches / len(w_in)