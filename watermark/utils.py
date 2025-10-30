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


def adaptive_partition_bins(rho, N=256, low_q=0.05, high_q=0.95, method="equal_width"):
    """
    自适应分 bin 方法
    
    参数:
        rho: np.array，顶点球坐标模长
        N: int，目标分 bin 数
        low_q, high_q: float，去除尾部稀疏点的分位数
        method: str，分bin方法，可选 "equal_width"(等宽), "equal_frequency"(等频), "kde"(核密度估计)
    
    返回:
        final_bin: list of list，每个 bin 内的原始索引
        bin_rho_min: list，每个 bin 的最小 rho
        bin_rho_max: list，每个 bin 的最大 rho
    """
    rho = np.asarray(rho, dtype=np.float64)
    L = len(rho)
    if L < N:
        raise ValueError(f"顶点数 {L} < 分 bin 数 N={N}，无法分箱")
    
    if method == "equal_width":
        return equal_width_binning(rho, N, low_q, high_q)
    elif method == "equal_frequency":
        return equal_frequency_binning(rho, N, low_q, high_q)
    elif method == "kde":
        return kde_adaptive_binning(rho, N, low_q, high_q)
    else:
        raise ValueError(f"不支持的分bin方法: {method}，可选: equal_width, equal_frequency, kde")


def equal_width_binning(rho, N=256, low_q=0.05, high_q=0.95):
    """
    等宽分 bin（基于分位数密集区间）
    
    参数:
        rho: np.array，顶点球坐标模长
        N: int，目标分 bin 数
        low_q, high_q: float，去除尾部稀疏点的分位数
    
    返回:
        final_bin: list of list，每个 bin 内的原始索引
        bin_rho_min: list，每个 bin 的最小 rho
        bin_rho_max: list，每个 bin 的最大 rho
    """
    # 1️⃣ 计算密集区间
    dense_r_min, dense_r_max = np.quantile(rho, [low_q, high_q])
    mask_dense = (rho >= dense_r_min) & (rho <= dense_r_max)
    r_dense = rho[mask_dense]
    idx_dense = np.where(mask_dense)[0]

    # 2️⃣ 等宽分 bin
    bin_width = (dense_r_max - dense_r_min) / N
    final_bin = [[] for _ in range(N)]
    bin_rho_min = []
    bin_rho_max = []

    # 为每个点分配bin
    for i, (r, idx) in enumerate(zip(r_dense, idx_dense)):
        bin_idx = min(int((r - dense_r_min) / bin_width), N-1)
        final_bin[bin_idx].append(idx)
    
    # 计算每个bin的最小最大值
    for i in range(N):
        indices = final_bin[i]
        if len(indices) > 0:
            bin_rho_min.append(float(np.min(rho[indices])))
            bin_rho_max.append(float(np.max(rho[indices])))
        else:
            # 空bin处理
            bin_min = dense_r_min + i * bin_width
            bin_max = dense_r_min + (i + 1) * bin_width
            bin_rho_min.append(float(bin_min))
            bin_rho_max.append(float(bin_max))
    
    return rho, final_bin, bin_rho_min, bin_rho_max


def equal_frequency_binning(rho, N=256, low_q=0.05, high_q=0.95):
    """
    等频分 bin（每个bin包含相近数量的顶点）
    
    参数:
        rho: np.array，顶点球坐标模长
        N: int，目标分 bin 数
        low_q, high_q: float，去除尾部稀疏点的分位数
    
    返回:
        final_bin: list of list，每个 bin 内的原始索引
        bin_rho_min: list，每个 bin 的最小 rho
        bin_rho_max: list，每个 bin 的最大 rho
    """
    # 1️⃣ 计算密集区间
    dense_r_min, dense_r_max = np.quantile(rho, [low_q, high_q])
    mask_dense = (rho >= dense_r_min) & (rho <= dense_r_max)
    r_dense = rho[mask_dense]
    idx_dense = np.where(mask_dense)[0]

    # 2️⃣ 排序，等频分 bin
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
            bin_rho_min.append(float(sorted_r[start]) if start < len(sorted_r) else dense_r_min)
            bin_rho_max.append(float(sorted_r[start]) if start < len(sorted_r) else dense_r_max)
        start = end

    return rho, final_bin, bin_rho_min, bin_rho_max


def kde_adaptive_binning(rho, N=256, low_q=0.05, high_q=0.95):
    """
    基于核密度估计(KDE)的自适应分bin
    
    参数:
        rho: np.array，顶点球坐标模长
        N: int，目标分 bin 数
        low_q, high_q: float，去除尾部稀疏点的分位数
    
    返回:
        final_bin: list of list，每个 bin 内的原始索引
        bin_rho_min: list，每个 bin 的最小 rho
        bin_rho_max: list，每个 bin 的最大 rho
    """
    from scipy.stats import gaussian_kde
    
    # 1️⃣ 计算密集区间
    dense_r_min, dense_r_max = np.quantile(rho, [low_q, high_q])
    mask_dense = (rho >= dense_r_min) & (rho <= dense_r_max)
    r_dense = rho[mask_dense]
    idx_dense = np.where(mask_dense)[0]
    
    if len(r_dense) < 10:  # KDE需要足够的样本
        return equal_frequency_binning(rho, N, low_q, high_q)
    
    # 2️⃣ 计算KDE
    try:
        kde = gaussian_kde(r_dense)
        
        # 3️⃣ 根据密度定义bin边界
        # 在密集区间内均匀采样点
        x_eval = np.linspace(dense_r_min, dense_r_max, 1000)
        density = kde(x_eval)
        
        # 计算累积密度
        cum_density = np.cumsum(density)
        cum_density = cum_density / cum_density[-1]  # 归一化到[0,1]
        
        # 根据累积密度等分N个bin
        bin_edges = np.zeros(N+1)
        bin_edges[0] = dense_r_min
        bin_edges[-1] = dense_r_max
        
        for i in range(1, N):
            # 找到累积密度为i/N的点
            target = i / N
            idx = np.argmin(np.abs(cum_density - target))
            bin_edges[i] = x_eval[idx]
    except Exception as e:
        print(f"KDE计算失败: {str(e)}，回退到等频分bin")
        return equal_frequency_binning(rho, N, low_q, high_q)
    
    # 4️⃣ 根据bin边界分配点
    final_bin = [[] for _ in range(N)]
    for i, (r, idx) in enumerate(zip(r_dense, idx_dense)):
        # 找到点所在的bin
        bin_idx = np.searchsorted(bin_edges, r) - 1
        bin_idx = max(0, min(bin_idx, N-1))  # 确保索引在有效范围内
        final_bin[bin_idx].append(idx)
    
    # 5️⃣ 计算每个bin的最小最大值
    bin_rho_min = []
    bin_rho_max = []
    for i in range(N):
        indices = final_bin[i]
        if len(indices) > 0:
            bin_rho_min.append(float(np.min(rho[indices])))
            bin_rho_max.append(float(np.max(rho[indices])))
        else:
            # 空bin处理
            bin_rho_min.append(float(bin_edges[i]))
            bin_rho_max.append(float(bin_edges[i+1]))
    
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

def caculate_psnr(vertices_original, vertices_watermarked):
    """计算原始模型与含水印模型的顶点坐标PSNR"""
    if len(vertices_original) != len(vertices_watermarked):
        raise ValueError("原始模型与含水印模型顶点数必须一致")
    mse = np.mean((vertices_original - vertices_watermarked) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同
    max_pixel = np.max(vertices_original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def caculate_hausdorff(vertices_original, vertices_watermarked):
    """计算原始模型与含水印模型的顶点坐标Hausdorff距离"""
    from scipy.spatial import cKDTree

    tree_original = cKDTree(vertices_original)
    tree_watermarked = cKDTree(vertices_watermarked)

    distances_original_to_watermarked, _ = tree_original.query(vertices_watermarked)
    distances_watermarked_to_original, _ = tree_watermarked.query(vertices_original)

    hausdorff_distance = max(np.max(distances_original_to_watermarked), np.max(distances_watermarked_to_original))
    return hausdorff_distance

def geometric_median(vertices, weights=None, eps=1e-6, max_iter=200, tol=1e-6):
    """
    Compute geometric median (Weiszfeld algorithm) for points `vertices`.
    - vertices: np.ndarray, shape (N, d)
    - weights: optional array shape (N,), non-negative; if None, equal weights used
    - eps: small regularization to avoid division by zero (used only to detect exact matches)
    - max_iter: maximum iterations
    - tol: convergence tolerance on change in position (L2 norm)
    Returns: np.ndarray shape (d,)
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D array shape (N,d)")

    N, d = vertices.shape
    if N == 0:
        raise ValueError("empty vertex set")

    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != N:
            raise ValueError("weights length must equal number of vertices")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    # Good initialization: use weighted mean (fast) -- better than random
    y = np.sum(vertices * w[:, None], axis=0) / np.sum(w)

    for it in range(max_iter):
        # distances from current estimate to each point
        diff = vertices - y  # (N, d)
        dist = np.linalg.norm(diff, axis=1)  # (N,)

        # Check if y coincides (within eps) with one of the vertices
        zero_mask = dist < eps
        if np.any(zero_mask):
            # If y is exactly at a data point v_j, that data point is the geometric median
            # for positive weights and all others nonnegative; return it (weighted choice)
            # But safer: return the weighted average of coincident points
            return np.sum(vertices[zero_mask] * w[zero_mask, None], axis=0) / np.sum(w[zero_mask])

        # Compute Weiszfeld numerator and denominator: vectorized, guarded against zeros
        inv_dist = w / dist  # (N,)
        numerator = np.sum(vertices * inv_dist[:, None], axis=0)  # (d,)
        denominator = np.sum(inv_dist)

        y_new = numerator / denominator

        # Convergence check
        if np.linalg.norm(y_new - y) < tol:
            return y_new

        y = y_new

    # If not converged, return last estimate (optionally fallback to mean)
    # You may want to log a warning in production code
    return y