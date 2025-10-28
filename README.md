# 3D鲁棒水印系统

3D鲁棒水印系统是一个专为三维模型设计的版权保护解决方案，通过将原始单文件代码重构为模块化架构，提供了高效、灵活且鲁棒性强的水印嵌入与提取功能。本系统支持多种几何攻击下的水印保持能力，适用于3D模型数字版权保护领域。

## 项目结构

```
3D-Robust-Watermark/
├── src/                      # 源代码目录
│   ├── __init__.py          # 模块初始化文件
│   ├── file_operations.py   # 文件读取和保存操作
│   ├── utils.py             # 工具函数（坐标转换、分bin、水印生成等）
│   ├── watermark_core.py    # 水印嵌入和提取核心功能
│   ├── robustness.py        # 鲁棒性测试和攻击模拟
│   ├── main.py              # 主函数和示例调用
│   ├── example_usage.py     # 使用示例
│   └── api/                 # API接口模块
│       ├── __init__.py
│       ├── app.py          # FastAPI应用
│       └── routes.py       # API路由
├── static/                  # 静态资源
│   ├── uploads/            # 上传文件存储
│   └── results/            # 结果文件存储
├── tests/                  # 测试用例
├── data/                   # 示例数据
├── docker/                 # Docker相关文件
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt        # 项目依赖
├── README.md              # 说明文档
└── LICENSE                # 许可证文件
```

## 核心功能模块

### 1. 文件操作模块 (file_operations.py)
- `read_obj_vertices()`: 高效读取OBJ文件顶点数据
- `save_obj_with_new_vertices()`: 保存修改后的顶点到OBJ文件
- `calculate_rmse()`: 计算原始模型与水印模型间的均方根误差
- `read_obj_faces()`: 读取OBJ文件的面信息
- `load_model()`: 通用3D模型加载接口（支持OBJ, STL, PLY等格式）
- `save_model()`: 通用3D模型保存接口

### 2. 工具函数模块 (utils.py)
- `cartesian_to_spherical()`: 笛卡尔坐标到球坐标的转换
- `spherical_to_cartesian()`: 球坐标到笛卡尔坐标的转换
- `adaptive_partition_bins()`: 基于模型特征的自适应分bin策略
- `normalize_bin_rho()`: 对bin的径向分量进行归一化处理
- `denormalize_bin_rho()`: 逆归一化操作
- `calculate_k_coefficient()`: 计算水印强度系数
- `generate_fixed_seed_watermark()`: 基于固定种子生成可重复水印
- `read_watermark_from_bin()`: 从bin中提取水印信息
- `calculate_model_stats()`: 计算模型统计特征（顶点数、面数、边界框等）
- `visualize_watermark_diff()`: 可视化水印嵌入前后的差异

### 3. 水印核心模块 (watermark_core.py)
- `watermark_embedding()`: 水印嵌入主函数，支持多种嵌入策略
- `watermark_extraction()`: 水印提取主函数，支持盲检测
- `calculate_watermark_capacity()`: 计算模型可承载的水印容量
- `adjust_watermark_strength()`: 根据模型特征动态调整水印强度

### 4. 鲁棒性测试模块 (robustness.py)
- `apply_rotation1/2/3()`: 多轴旋转攻击测试
- `apply_scaling()`: 非均匀缩放攻击测试
- `apply_translation()`: 平移攻击测试
- `apply_clipping()`: 部分裁剪攻击测试
- `apply_noise()`: 随机噪声攻击测试
- `apply_smoothing()`: 表面平滑攻击测试
- `apply_simplification()`: 网格简化攻击测试
- `apply_vertex_reordering()`: 顶点重排序攻击测试
- `apply_mesh_retopology()`: 网格重拓扑攻击测试
- `evaluate_robustness()`: 全面的鲁棒性评估框架
- `generate_attack_report()`: 生成详细的攻击测试报告

### 5. API接口模块 (api/)
- `app.py`: FastAPI应用主程序
- `routes.py`: RESTful API路由定义
- `models.py`: API请求/响应数据模型
- `dependencies.py`: API依赖项和中间件
- `watermark_api.py`: 水印相关API实现

### 6. 主程序模块 (main.py)
- `main()`: 程序入口点，支持命令行参数
- `run_single_test()`: 单模型测试流程
- `run_batch_test()`: 批量模型测试流程
- `generate_test_report()`: 生成测试报告

## 使用方法

### 方法1：直接导入使用（推荐用于开发）
```python
from src import watermark_embedding, watermark_extraction, evaluate_robustness

# 基本水印嵌入
watermark_embedding("input.obj", "output.obj", watermark_length=256, strength=0.1)

# 带参数的水印嵌入
params = {
    "watermark_length": 256,
    "strength": 0.15,
    "method": "spherical",
    "seed": 42
}
watermark_embedding("input.obj", "output.obj", **params)

# 水印提取
_, accuracy = watermark_extraction("output.obj", watermark_length=256)
print(f"提取准确率: {accuracy:.2f}%")

# 鲁棒性测试
report = evaluate_robustness("input.obj", "output.obj", watermark_length=256)
print(f"鲁棒性评分: {report["overall_score"]:.2f}")
```

### 方法2：运行主函数（适合快速测试）
```python
from src.main import main

# 使用默认参数运行
main()

# 自定义参数运行
main(
    input_dir="data/models",
    output_dir="results",
    watermark_length=512,
    test_attacks=["rotation", "noise", "simplification"]
)
```

### 方法3：运行示例（学习参考）
```python
python src/example_usage.py
```

### 方法4：使用Web API（适合生产环境）
```python
# 启动Web服务
from src.api.app import app

# 使用uvicorn运行
# uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 或使用Docker（推荐）
# docker-compose up --build
```

### API使用示例
```python
import requests

# 水印嵌入
response = requests.post(
    "http://localhost:8000/api/watermark/v1/embed",
    files={"file": open("model.obj", "rb")},
    data={"watermark_length": 256, "strength": 0.1}
)
with open("watermarked_model.obj", "wb") as f:
    f.write(response.content)

# 水印提取
response = requests.post(
    "http://localhost:8000/api/watermark/v1/extract",
    files={"file": open("watermarked_model.obj", "rb")},
    data={"watermark_length": 256}
)
result = response.json()
print(f"提取水印: {result["watermark"]}, 准确率: {result["accuracy"]}%")

# 鲁棒性测试
response = requests.post(
    "http://localhost:8000/api/watermark/v1/attack",
    files={"file": open("watermarked_model.obj", "rb")},
    data={
        "attacks": ["rotation", "noise", "simplification"],
        "parameters": {"noise_level": 0.01, "simplification_ratio": 0.5}
    }
)
report = response.json()
print(f"攻击测试报告: {report}")
```

## 系统特性

1. **模块化架构**: 清晰的功能分离，便于维护和扩展
   - 核心算法与接口分离
   - 工具函数独立封装
   - 支持插件式攻击方法

2. **高效性能优化**: 
   - 使用NumPy进行向量化计算
   - 并行处理支持
   - 内存优化设计，支持大规模模型

3. **全面的鲁棒性测试**: 
   - 7种标准攻击方法
   - 自定义攻击组合
   - 详细的攻击报告生成

4. **灵活的API设计**: 
   - 命令行接口
   - Python API
   - RESTful Web API
   - 支持批量处理

5. **可扩展性**: 
   - 插件式水印算法
   - 可扩展的攻击方法
   - 模块化的评估指标

6. **可视化工具**: 
   - 水印嵌入前后对比
   - 攻击效果可视化
   - 交互式报告生成

7. **容器化支持**: 
   - Docker部署
   - Docker Compose编排
   - 可扩展的基础镜像

## 系统依赖

### 核心依赖
- **numpy**: 数值计算核心库
- **scipy**: 科学计算与信号处理
- **trimesh**: 3D网格处理（可选，但推荐）
- **open3d**: 3D数据处理与可视化

### API相关依赖
- **fastapi**: Web API框架
- **uvicorn**: ASGI服务器
- **python-multipart**: 文件上传支持

### 可视化依赖
- **matplotlib**: 2D绘图
- **plotly**: 交互式可视化
- **pyvista**: 3D模型可视化

### 开发与测试依赖
- **pytest**: 单元测试框架
- **black**: 代码格式化
- **flake8**: 代码质量检查
- **mypy**: 类型检查

安装所有依赖：
```bash
pip install -r requirements.txt
```

## 版本信息

- **当前版本**: 1.0.0
- **发布日期**: 2025-10-28
- **作者**: 3D Watermark System Team
- **维护者**: InorySekiro2333
- **许可证**: MIT

## 项目架构

### 目录结构

```
3D-Robust-Watermark/
├── src/                      # 源代码
│   ├── core/                 # 核心算法实现
│   ├── api/                  # API接口
│   ├── utils/                # 工具函数
│   └── tests/                # 单元测试
├── static/                   # 静态资源
├── data/                     # 示例数据
├── docs/                     # 文档
├── docker/                   # Docker配置
└── scripts/                  # 辅助脚本
```

## 部署指南

### 方法1：使用Docker部署（推荐）

#### 单服务Docker部署
```powershell
# 构建镜像
docker build -t 3d-watermark-web:latest .

# 运行容器
docker run --rm   -p 8000:8000   -v ${PWD}:/app   -v ${PWD}/static/uploads:/app/static/uploads   -v ${PWD}/static/results:/app/static/results   3d-watermark-web:latest
```

#### 使用Docker Compose（推荐）
```powershell
docker-compose up --build
```

### 方法2：本地部署

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 启动Web服务
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### 运行命令行工具
```bash
python src/main.py --help
```

