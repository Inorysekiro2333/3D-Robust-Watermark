# 3D网格水印系统模块

这是一个模块化的3D网格水印嵌入和提取系统，将原始的单文件代码重构为多个功能模块。

## 模块结构

```
xixi/
├── __init__.py              # 模块初始化文件
├── file_operations.py       # 文件读取和保存操作
├── utils.py                 # 工具函数（坐标转换、分bin、水印生成等）
├── watermark_core.py        # 水印嵌入和提取核心功能
├── robustness.py            # 鲁棒性测试和攻击模拟
├── main.py                  # 主函数和示例调用
├── example_usage.py         # 使用示例
└── README.md               # 说明文档
```

## 模块功能

### 1. file_operations.py
- `read_obj_vertices()`: 读取OBJ文件顶点
- `save_obj_with_new_vertices()`: 保存修改后的顶点
- `calculate_rmse()`: 计算RMSE误差
- `read_obj_faces()`: 读取面信息

### 2. utils.py
- `cartesian_to_spherical()`: 笛卡尔坐标转球坐标
- `spherical_to_cartesian()`: 球坐标转笛卡尔坐标
- `adaptive_partition_bins()`: 自适应分bin
- `normalize_bin_rho()`: 归一化
- `denormalize_bin_rho()`: 逆归一化
- `calculate_k_coefficient()`: 计算k系数
- `generate_fixed_seed_watermark()`: 生成水印
- `read_watermark_from_bin()`: 读取水印

### 3. watermark_core.py
- `watermark_embedding()`: 水印嵌入主函数
- `watermark_extraction()`: 水印提取主函数

### 4. robustness.py
- `apply_rotation1/2/3()`: 旋转攻击
- `apply_scaling()`: 缩放攻击
- `apply_translation()`: 平移攻击
- `apply_clipping()`: 裁剪攻击
- `apply_noise()`: 噪声攻击
- `apply_smoothing()`: 平滑攻击
- `apply_simplification()`: 简化攻击
- `apply_vertex_reordering()`: 顶点重排序攻击
- `evaluate_robustness()`: 鲁棒性评估

### 5. main.py
- `main()`: 主函数
- `run_single_test()`: 单模型测试

## 使用方法

### 方法1：直接导入使用
```python
from xixi import watermark_embedding, watermark_extraction, evaluate_robustness

# 水印嵌入
watermark_embedding("input.obj", "output.obj", 256)

# 水印提取
_, accuracy = watermark_extraction("output.obj", 256)
print(f"准确率: {accuracy:.2f}%")

# 鲁棒性测试
evaluate_robustness("input.obj", "output.obj", 256)
```

### 方法2：运行主函数
```python
from xixi.main import main
main()
```

### 方法3：运行示例
```python
python xixi/example_usage.py
```

## 特性

1. **模块化设计**: 功能分离，便于维护和扩展
2. **时间记录**: 自动记录嵌入和提取时间
3. **鲁棒性测试**: 包含多种攻击测试
4. **易于使用**: 简单的API接口
5. **可扩展性**: 易于添加新功能

## 依赖

- numpy
- scipy
- trimesh (可选)

## 版本

- 版本: 1.0.0
- 作者: 3D Watermark System
# xixi

## 使用 Docker 运行 (快速开始)

下面是使用 Docker 和 docker-compose 在本地启动并测试 API 的最小说明（针对 Windows PowerShell）。

1) 使用 docker 构建镜像并启动（单服务）:

```powershell
# 在项目根目录运行（包含 Dockerfile 的目录）
docker build -t 3d-watermark-web:latest .
docker run --rm -p 8000:8000 -v ${PWD}:/app -v ${PWD}\static\uploads:/app/static/uploads -v ${PWD}\static\results:/app/static/results 3d-watermark-web:latest
```

2) 使用 docker-compose（推荐 — 会自动挂载 uploads/results）:

```powershell
docker-compose up --build
```

### 使用阿里云镜像 / 指定基础镜像

如果你在国内网络环境下访问 Docker Hub 有问题，可以：

- 用阿里云镜像服务上已有的 Python 镜像（如果你在阿里云容器镜像服务中有公开或私有仓库托管了 Python 镜像）；
- 或使用 Docker 加速器（registry mirror）来加速从 Docker Hub 拉取镜像。

示例：在构建时通过 `--build-arg` 指定基础镜像为阿里云仓库中的镜像（请把 `<your_namespace>` 与镜像名替换为你在阿里云容器镜像服务上的命名空间/镜像）：

```powershell
# 把 BASE_IMAGE 替换为你在阿里云上的镜像，如 registry.cn-hangzhou.aliyuncs.com/<your_namespace>/python:3.9-slim
docker build --build-arg BASE_IMAGE=registry.cn-hangzhou.aliyuncs.com/<your_namespace>/python:3.9-slim -t 3d-watermark-web:latest .

# 运行
docker run --rm -p 8000:8000 -v ${PWD}:/app -v ${PWD}\static\uploads:/app/static/uploads -v ${PWD}\static\results:/app/static/results 3d-watermark-web:latest
```

如果你没有把镜像推到阿里云镜像仓库，也可以配置 Docker 的镜像加速器（registry mirror），常见做法是在 Docker 的 daemon 配置中添加阿里云提供的镜像加速地址（阿里云会为你的账户生成一个专属加速器地址）。

在 Windows 上（Docker Desktop / Docker Engine），可以修改或创建 `C:\ProgramData\docker\config\daemon.json`（需要管理员权限），添加类似内容：

```json
{
	"registry-mirrors": ["https://<your-aliyun-mirror>.mirror.aliyuncs.com"]
}
```

保存后重启 Docker 服务，再次运行 `docker-compose up --build`，这样 Docker 在拉取 `python:3.9-slim` 等镜像时会走加速器。

注意：镜像加速器地址通常需要你在阿里云容器镜像服务控制台查看并启用。

3) 测试 API:

打开浏览器或使用 curl/postman 访问接口，默认服务运行在 http://localhost:8000

- 文档（自动生成）：http://localhost:8000/docs
- 嵌入接口：POST http://localhost:8000/api/watermark/v1/embed
- 提取接口：POST http://localhost:8000/api/watermark/v1/extract
- 攻击接口：POST http://localhost:8000/api/watermark/v1/attack

注意事项:
- 本镜像默认使用 uvicorn 启动（2 个 worker）。如果你在低资源机器上运行，可把 `--workers` 调小到 1。
- 如果 pip 安装 scipy/numpy 时遇到编译问题，可以在 Dockerfile 中添加更多系统依赖或使用包含科学计算库的基础镜像。
