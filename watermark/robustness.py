"""
鲁棒性测试模块 - 新版本
包含独立的攻击函数和批量测试功能
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from .file_operations import read_obj_vertices, save_obj_with_new_vertices, read_obj_faces
from .watermark_core import watermark_extraction
from typing import Callable, Dict


def attack_rotation(obj_path, angles=30, output_path=None):
    """旋转攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_rotated_{angles}.obj"
    
    vertices, _, _ = read_obj_vertices(obj_path)
    ang = (angles, angles, angles) if not isinstance(angles, (list, tuple, np.ndarray)) else angles
    r = R.from_euler('xyz', ang, degrees=True)
    transformed_vertices = r.apply(vertices)
    
    save_obj_with_new_vertices(obj_path, transformed_vertices, output_path)
    return output_path


def attack_scaling(obj_path, scale=1.2, output_path=None):
    """缩放攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_scaled_{scale}.obj"
    
    vertices, _, _ = read_obj_vertices(obj_path)
    transformed_vertices = vertices * scale
    
    save_obj_with_new_vertices(obj_path, transformed_vertices, output_path)
    return output_path


def attack_translation(obj_path, offset=(0.5, -0.3, 0.2), output_path=None):
    """平移攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_translated.obj"
    
    vertices, _, _ = read_obj_vertices(obj_path)
    transformed_vertices = vertices + np.array(offset)
    
    save_obj_with_new_vertices(obj_path, transformed_vertices, output_path)
    return output_path


def attack_clipping(obj_path, ratio=0.1, output_path=None):
    """裁剪攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_clipped_{int(ratio*100)}.obj"
    
    vertices, _, _ = read_obj_vertices(obj_path)
    if ratio <= 0:
        transformed_vertices = vertices
    else:
        ratio = min(max(ratio, 0.0), 1.0)
        z_min = np.min(vertices[:, 2])
        z_max = np.max(vertices[:, 2])
        threshold = z_min + (1.0 - ratio) * (z_max - z_min)
        clipped = vertices.copy()
        mask = clipped[:, 2] > threshold
        clipped[mask, 2] = threshold
        transformed_vertices = clipped
    
    save_obj_with_new_vertices(obj_path, transformed_vertices, output_path)
    return output_path


def attack_noise(obj_path, sigma_ratio=0.001, output_path=None):
    """噪声攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_noise_{sigma_ratio}.obj"
    
    vertices, _, _ = read_obj_vertices(obj_path)
    if sigma_ratio <= 0:
        transformed_vertices = vertices
    else:
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        sigma = sigma_ratio * diag
        np.random.seed(42)
        noise = np.random.normal(loc=0.0, scale=sigma, size=vertices.shape)
        transformed_vertices = vertices + noise
    
    save_obj_with_new_vertices(obj_path, transformed_vertices, output_path)
    return output_path


def attack_smoothing(obj_path, iterations=5, lambda_=0.03, output_path=None):
    """平滑攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_smoothed_{iterations}.obj"
    
    vertices, _, _ = read_obj_vertices(obj_path)
    faces = read_obj_faces(obj_path)
    
    n = len(vertices)
    adjacency = [[] for _ in range(n)]
    for face in faces:
        face = [v - 1 for v in face]
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            if v2 not in adjacency[v1]:
                adjacency[v1].append(v2)
            if v1 not in adjacency[v2]:
                adjacency[v2].append(v1)
    
    for _ in range(iterations):
        new_vertices = vertices.copy()
        for i in range(n):
            if adjacency[i]:
                neighbor_mean = np.mean(vertices[adjacency[i]], axis=0)
                new_vertices[i] = vertices[i] * (1 - lambda_) + neighbor_mean * lambda_
        vertices = new_vertices
    
    save_obj_with_new_vertices(obj_path, vertices, output_path)
    return output_path


def attack_simplification(obj_path, reduction=0.5, output_path=None):
    """简化攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_simplified_{int(reduction*100)}.obj"
    
    with open(obj_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    vertices, faces = [], []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == 'f':
            face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
            faces.append(face)
    
    triangles = []
    for f in faces:
        if len(f) == 3:
            triangles.append(f)
        elif len(f) == 4:
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
    
    triangles = np.array(triangles, dtype=np.int32)
    total_faces = len(triangles)
    target_faces = max(1, int(total_faces * reduction))
    
    np.random.seed(42)
    keep = np.random.choice(total_faces, target_faces, replace=False)
    simplified_tri = triangles[keep]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for v in vertices:
            f.write(f'v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n')
        for line in lines:
            if not line.startswith(('v ', 'f ')):
                f.write(line)
        for tri in simplified_tri:
            f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
    
    return output_path


def attack_vertex_reordering(obj_path, output_path=None):
    """顶点重排序攻击"""
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = f"{base_name}_reordered.obj"
    
    with open(obj_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    vertex_lines = []
    face_indices = []
    other_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            other_lines.append(line)
            continue
        if parts[0] == 'v' and len(parts) >= 4:
            vertex_lines.append(line)
        elif parts[0] == 'f' and len(parts) >= 4:
            indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
            face_indices.append(indices)
        else:
            other_lines.append(line)
    
    num_vertices = len(vertex_lines)
    if num_vertices == 0:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return output_path
    
    np.random.seed(42)
    permutation = np.random.permutation(num_vertices)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(num_vertices)
    
    reordered_vertex_lines = [vertex_lines[i] for i in permutation]
    
    remapped_faces = []
    for idx_list in face_indices:
        remapped = [int(inverse_permutation[i]) + 1 for i in idx_list]
        remapped_faces.append(remapped)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for vline in reordered_vertex_lines:
            f.write(vline)
        for oline in other_lines:
            f.write(oline)
        for face in remapped_faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")
    
    return output_path


class AttackStrategy:
    """
    策略模式实现攻击方法的统一接口
    """
    def __init__(self):
        self._strategies: Dict[str, Callable[[str, float], str]] = {}

    def register_strategy(self, name: str, strategy: Callable[[str, float], str]):
        self._strategies[name] = strategy

    def execute(self, attack_type: str, obj_path: str, strength: float) -> str:
        if attack_type not in self._strategies:
            raise ValueError(f"未注册的攻击类型: {attack_type}")
        return self._strategies[attack_type](obj_path, strength)

# 创建策略实例
attack_strategy = AttackStrategy()

# 注册攻击方法
def rotation_attack(obj_path: str, strength: float) -> str:
    return attack_rotation(obj_path, angles=strength)
def scaling_attack(obj_path: str, strength: float) -> str:
    return attack_scaling(obj_path, scale=strength)
def translation_attack(obj_path: str, strength: float) -> str:
    return attack_translation(obj_path, offset=(strength, strength, strength))
def clipping_attack(obj_path: str, strength: float) -> str:
    return attack_clipping(obj_path, ratio=strength)
def noise_attack(obj_path: str, strength: float) -> str:
    return attack_noise(obj_path, sigma_ratio=strength)
def smoothing_attack(obj_path: str, strength: float) -> str:
    return attack_smoothing(obj_path, iterations=int(strength), lambda_=0.03)
def simplification_attack(obj_path: str, strength: float) -> str:
    return attack_simplification(obj_path, reduction=strength)
def vertex_reordering_attack(obj_path: str, strength: float) -> str:
    return attack_vertex_reordering(obj_path)

attack_strategy.register_strategy("rotation", rotation_attack)
attack_strategy.register_strategy("scaling", scaling_attack)
attack_strategy.register_strategy("translation", translation_attack)
attack_strategy.register_strategy("clipping", clipping_attack)
attack_strategy.register_strategy("noise", noise_attack)
attack_strategy.register_strategy("smoothing", smoothing_attack)
attack_strategy.register_strategy("simplification", simplification_attack)
attack_strategy.register_strategy("vertex_reordering", vertex_reordering_attack)

# 提供统一的攻击接口
def apply_attack(obj_path: str, attack_type: str, strength: float) -> str:
    """
    应用指定的攻击方法

    Args:
        obj_path (str): 模型路径
        attack_type (str): 攻击类型
        strength (float): 攻击强度

    Returns:
        str: 攻击后的模型路径
    """
    return attack_strategy.execute(attack_type, obj_path, strength)


def test_multiple_attacks(attacks_list, watermarked_obj, output_path=""):
    """测试多个攻击"""
    model_name = os.path.splitext(os.path.basename(watermarked_obj))[0]
    
    print(f"开始测试 {len(attacks_list)} 个攻击...")
    
    attacked_files = []
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    for i, attack_info in enumerate(attacks_list):
        attack_func = attack_info["func"]
        attack_params = attack_info.get("params", {})
        
        attack_type = getattr(attack_func, "__name__", "attack")
        
        def _param_val_to_str(val):
            if isinstance(val, (list, tuple)):
                return "x".join(str(x) for x in val)
            if isinstance(val, float):
                s = f"{val:.6f}".rstrip("0").rstrip(".")
                return s if s else "0"
            return str(val)
        
        param_parts = [f"{k}-{_param_val_to_str(v)}" for k, v in attack_params.items()]
        params_str = "_".join(param_parts) if param_parts else "default"
        
        filename = f"{model_name}_{attack_type}_{params_str}.obj"
        output_file = os.path.join(output_path, filename) if output_path else filename
        
        attacked_obj = attack_func(watermarked_obj, output_path=output_file, **attack_params)
        attacked_files.append(attacked_obj)
    
    print(f"测试完成，已生成 {len(attacked_files)} 个攻击文件。")
    return attacked_files


if __name__ == "__main__":
    attacks_list = [
        # #{"func": attack_rotation, "params": {"angles": 30}},
        # {"func": attack_rotation, "params": {"angles": 45}},
        # # {"func": attack_scaling, "params": {"scale": 1.5}},
        # # {"func": attack_translation, "params": {"offset": (0.5, -0.3, 0.2)}},
        # # {"func": attack_clipping, "params": {"ratio": 0.1}},
        # # {"func": attack_noise, "params": {"sigma_ratio": 0.001}},
        # # {"func": attack_smoothing, "params": {"iterations": 5, "lambda_": 0.03}},
        # # {"func": attack_simplification, "params": {"reduction": 0.5}},
        # # {"func": attack_vertex_reordering, "params": {}}
    ]
    
    watermarked_obj = "3D_models_with_watermarking/Airplane_w.obj"
    output_path = "attacks_model"
    
    attacked_files = test_multiple_attacks(attacks_list, watermarked_obj, output_path)