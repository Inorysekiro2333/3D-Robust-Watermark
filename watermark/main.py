"""
主函数模块
包含程序的主要执行逻辑和示例调用
"""

from .watermark_core import watermark_embedding, watermark_extraction
from .utils import string_to_binary, binary_to_string
import os


def single_test(original_obj, watermark_text, output_obj, N=256):
    """
    单个水印嵌入测试功能
    
    参数:
        original_obj: str,原始模型路径
        watermark_text: str,水印文本
        output_obj: str,输出模型路径
        N: int,水印位数,默认256
    
    返回:
        dict,包含测试结果的字典
    """
    
    print(f"原始模型: {original_obj}")
    print(f"水印文本: {watermark_text}")
    print(f"输出模型: {output_obj}")
    print(f"水印位数: {N}")
    
    # 检查原始模型是否存在
    if not os.path.exists(original_obj):
        error_msg = f"错误：原始模型文件不存在: {original_obj}"
        print(error_msg)
        return {"success": False, "error": error_msg}
    
    # 1. 将水印文本转换为二进制
    print(f"\n1. 转换水印文本为二进制...")
    watermark_binary = string_to_binary(watermark_text, target_length=N)
    print(f"水印二进制长度: {len(watermark_binary)} 位")
    
    # 2. 执行水印嵌入
    print(f"\n2. 开始水印嵌入...")
    file_name, file_name_new, new_vertices, embedded_watermark, elapsed_time, rmse = watermark_embedding(
        original_obj, watermark_binary, N
    )
    print(f"水印嵌入完成！")
    print(f"RMSE: {rmse:.6f}")
    
    # 保存带水印的模型
    print(f"保存水印模型到: {output_obj}")
    from .file_operations import save_obj_with_new_vertices
    save_obj_with_new_vertices(original_obj, new_vertices, output_obj)
    
    # 3. 执行水印提取
    print(f"\n3. 开始水印提取...")
    extracted_watermark = watermark_extraction(output_obj, N)
    
    # 4. 计算准确率
    if len(extracted_watermark) == len(embedded_watermark):
        accuracy = (extracted_watermark == embedded_watermark).mean() * 100
        print(f"提取准确率: {accuracy:.2f}%")
    else:
        error_msg = "错误：提取的水印长度与嵌入的不匹配"
        print(error_msg)
        return {"success": False, "error": error_msg}
    
    # 5. 将提取的水印转换回文本
    print(f"\n4. 将提取的水印转换回文本...")
    extracted_text = binary_to_string(extracted_watermark)
    print(f"提取的水印文本: {extracted_text}")
    
    # 6. 显示结果对比
    print(f"\n5. 结果对比:")
    print(f"原始水印: {watermark_text}")
    print(f"提取水印: {extracted_text}")
    is_match = watermark_text == extracted_text
    
    # 返回结果
    return {
        "success": True,
        "original_text": watermark_text,
        "extracted_text": extracted_text,
        "accuracy": accuracy,
        "rmse": rmse
    }


def multiple_test(test_cases):
    """
    多个水印嵌入测试功能
    
    参数:
        test_cases: list,测试用例列表，每个元素为字典格式：
            {
                "original_obj": str,原始模型路径,
                "watermark_text": str,水印文本,
                "output_obj": str,输出模型路径,
                "N": int,水印位数,默认256
            }
    
    返回:
        list，每个测试用例的结果列表
    """
    print("="*60)
    print("多个水印嵌入测试")
    print("="*60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        
        original_obj = test_case["original_obj"]
        watermark_text = test_case["watermark_text"]
        output_obj = test_case["output_obj"]
        N = test_case.get("N", 256)
        
        # 调用单个测试
        result = single_test(original_obj, watermark_text, output_obj, N)
        result["test_case"] = i
        results.append(result)
        
        if result["success"]:
            print(f"测试用例 {i} 成功！准确率: {result['accuracy']:.2f}%")
        else:
            print(f"测试用例 {i} 失败: {result['error']}")
    
    # 显示汇总结果
    print(f"\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    print(f"总测试用例: {total_count}")
    print(f"成功: {success_count}")
    print(f"失败: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    return results


if __name__ == "__main__":
    print("请选择要执行的功能:")
    print("1. 单个水印嵌入测试 (single_test)")
    print("2. 多个水印嵌入测试 (multiple_test)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    ###单模型测试参数配置
    if choice == "1":
        # 单个测试参数
        original_obj = "static/uploads/aobin.obj"
        watermark_text = "测试水印1"
        output_obj = "static/results/watermarked_models/aobin.obj"
        N = 256
        
        print(f"\n=== 执行单个测试 ===")
        result = single_test(original_obj, watermark_text, output_obj, N)
        
        if result["success"]:
            print(f"\n测试成功！")
            print(f"准确率: {result['accuracy']:.2f}%")
            print(f"RMSE: {result['rmse']:.6f}")
        else:
            print(f"\n测试失败: {result['error']}")
     ###多模型测试参数配置       
    elif choice == "2":
        # 多个测试参数
        test_cases = [
            {
                "original_obj": "static/uploads/aobin.obj",
                "watermark_text": "测试水印1",
                "output_obj": "static/results/watermarked_models/aobin.obj",
                "N": 256
            },
            {
                "original_obj": "static/uploads/aobing.obj",
                "watermark_text": "测试水印2中文测试",
                "output_obj": "static/results/watermarked_models/aobing.obj",
                "N": 256
            },
            {
                "original_obj": "static/uploads/long.obj",
                "watermark_text": "Test Watermark 3",
                "output_obj": "static/results/watermarked_models/long.obj",
                "N": 256
            }

        ]
        
        print(f"\n=== 执行多个测试 ===")
        results = multiple_test(test_cases)
        
        # 显示详细结果
        print(f"\n详细结果:")
        for i, result in enumerate(results, 1):
            if result["success"]:
                print(f"测试 {i}: 成功 - 准确率 {result['accuracy']:.2f}% - RMSE {result['rmse']:.6f}")
            else:
                print(f"测试 {i}: 失败 - {result['error']}")
