from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
import os
import time
from typing import Optional

from watermark import watermark_embedding, watermark_extraction, save_obj_with_new_vertices
from watermark.utils import string_to_binary, binary_to_string

app = FastAPI()

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# storage dirs
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
WATERMARKED_DIR = os.path.join(BASE_DIR, 'static', 'results', 'watermarked_models')
ATTACKED_DIR = os.path.join(BASE_DIR, 'static', 'results', 'attacted_models')

# 创建必要的目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WATERMARKED_DIR, exist_ok=True)
os.makedirs(ATTACKED_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'obj'}

def allowed_file(filename: str) -> bool:
    """
    判断文件扩展名是否允许
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========================
# 嵌入水印接口
# ========================
@app.post('/api/watermark/v1/embed')
async def embed_watermark(file: UploadFile = File(...), watermark_text: Optional[str] = Form('')):
    """
    上传OBJ模型并嵌入水印，返回水印文件下载链接
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='未选择文件')
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail='不支持的文件类型')

    original_filename = secure_filename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, original_filename)

    # 保存上传文件
    with open(saved_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    watermark_bits = string_to_binary(watermark_text, target_length=256)
    result = watermark_embedding(saved_path, watermark_bits)

    if isinstance(result, dict):
        file_name_new = result.get('file_name_new') or f"{os.path.splitext(original_filename)[0]}_watermarked.obj"
        new_vertices = result.get('new_vertices')
        embed_time = result.get('elapsed_time') or result.get('embed_time')
        rmse = result.get('rmse')
    else:
        file_name, file_name_new, new_vertices, _, embed_time, rmse = result

    out_path = os.path.join(WATERMARKED_DIR, file_name_new)
    save_obj_with_new_vertices(saved_path, new_vertices, out_path)

    download_url = f"/api/watermark/v1/download/{file_name_new}"
    embed_time_str = f"{embed_time:.2f}s"
    rmse_str = f"{rmse:.3e}"

    resp = {
        'code': 0,
        'content': 'Success',
        'extra': {
            'watermark_text': watermark_text,
            'embed_time': embed_time_str,
            'rmse': rmse_str,
            'download_url': download_url,
            'original_filename': original_filename,
            'watermarked_filename': file_name_new,
        }
    }
    return JSONResponse(content=resp)

# ========================
# 提取水印接口
# ========================
@app.post('/api/watermark/v1/extract')
async def extract_watermark(file: UploadFile = File(...), watermark_length: Optional[int] = Form(256), watermark_text: Optional[str] = Form('')):
    """
    上传OBJ模型并提取水印
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='未选择文件')
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail='不支持的文件类型')

    saved_name = secure_filename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    # 保存上传文件
    with open(saved_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    start_time = time.time()
    extracted_bits = watermark_extraction(saved_path)
    elapsed = time.time() - start_time
    extracted_text = binary_to_string(extracted_bits)

    # 如果有传原始水印，计算准确率
    if watermark_text:
        original_bits = string_to_binary(watermark_text, target_length=watermark_length)
        accuracy = (extracted_bits == original_bits).mean() * 100 if len(extracted_bits) == len(original_bits) else 0.0
    response = {
        'code': 0,
        'content': 'Success',
        'extra': {
            'extracted_watermark': extracted_text,
            'extract_time': f"{elapsed:.2f}s",
            'extract_accuracy': f"{accuracy:.2f}%" if watermark_text else "N/A"
        }
    }
    return JSONResponse(content=response)

# ========================
# 攻击接口
# ========================
@app.post('/api/watermark/v1/attack')
async def attack_model(file: UploadFile = File(...), attack_type: str = Form(...), strength: float = Form(...)):
    """
    上传OBJ模型并对模型进行攻击，返回攻击后模型下载链接
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='未选择文件')
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail='不支持的文件类型')

    original_filename = secure_filename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, original_filename)

    # 保存上传文件
    with open(saved_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    # 动态导入攻击方法和基础攻击函数
    from watermark.robustness import apply_attack
    import shutil
    
    # 先生成临时攻击文件
    temp_attacked_path = apply_attack(saved_path, attack_type, strength)
    
    # 获取攻击后的文件名
    attacked_filename = os.path.basename(temp_attacked_path)
    
    # 移动文件到ATTACKED_DIR目录
    final_attacked_path = os.path.join(ATTACKED_DIR, attacked_filename)
    
    # 如果临时文件不在目标目录，则移动过去
    if temp_attacked_path != final_attacked_path:
        if os.path.exists(temp_attacked_path):
            shutil.move(temp_attacked_path, final_attacked_path)
        else:
            # 如果临时文件不存在，可能是生成在当前目录
            # 尝试从UPLOAD_DIR查找
            temp_file_in_upload = os.path.join(UPLOAD_DIR, attacked_filename)
            if os.path.exists(temp_file_in_upload):
                shutil.move(temp_file_in_upload, final_attacked_path)

    download_url = f"/api/watermark/v1/download/{attacked_filename}"

    response = {
        'code': 0,
        'content': 'Success',
        'extra': {
            'download_url': download_url,
            'original_filename': original_filename,
            'attacked_filename': attacked_filename,
        }
    }
    return JSONResponse(content=response)

# ========================
# 下载接口
# ========================
@app.get('/api/watermark/v1/download/{filename}')
def download_result(filename: str):
    """
    下载指定文件（仅支持results目录下的obj文件）
    """
    # 优先从watermarked_models查找，如果不存在则从attacted_models查找
    file_path = os.path.join(WATERMARKED_DIR, filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(ATTACKED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='文件不存在')
    headers = {
        "Cache-Control": "public, max-age=600",
        "Pragma": "public"
    }
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream', headers=headers)


# If run directly, start uvicorn (useful for local dev)

# ========================
# 本地开发入口
# ========================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
