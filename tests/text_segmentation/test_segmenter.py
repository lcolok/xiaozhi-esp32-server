#!/usr/bin/env python3
"""
文本分段测试脚本

使用方法：
1. 确保已激活虚拟环境：
   source .venv/bin/activate

2. 运行测试脚本：
   cd /root/xiaozhi-esp32-server  # 确保在项目根目录
   python tests/text_segmentation/test_segmenter.py

脚本会：
1. 在 tmp 目录下读取 article_segment_sample.txt
2. 使用 TextSegmenter 进行分段
3. 将结果保存到 tmp/segment_results.txt
4. 在控制台显示分段结果预览和性能统计
"""

import os
import sys
import yaml
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

from core.utils.text_segmentation import TextSegmenter

def load_config():
    """加载配置文件"""
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def format_time(seconds):
    """格式化时间"""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"

def ensure_tmp_dir():
    """确保 tmp 目录存在"""
    tmp_dir = os.path.join(project_root, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def create_sample_if_not_exists(tmp_dir: str):
    """如果样本文件不存在，创建一个示例文件"""
    sample_file = os.path.join(tmp_dir, 'article_segment_sample.txt')
    if not os.path.exists(sample_file):
        sample_text = """这是一个测试文本。它包含多个句子！让我们看看分段效果如何？这是第二段落。
这个段落有一些感叹句！比如这样！还有这样！
最后一段。这是最后一句话。再见！"""
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"已创建示例文件: {sample_file}")
    return sample_file

def main():
    total_start_time = time.time()
    
    # 确保目录存在
    tmp_dir = ensure_tmp_dir()
    
    # 准备输入输出文件
    input_file = create_sample_if_not_exists(tmp_dir)
    output_file = os.path.join(tmp_dir, 'segment_results.txt')
    
    # 读取输入文件
    file_read_start = time.time()
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    file_read_time = time.time() - file_read_start
    
    # 初始化分段器
    init_start = time.time()
    config = load_config()
    segmenter = TextSegmenter.get_instance(config)
    init_time = time.time() - init_start
    
    # 获取文本统计信息
    char_count = len(text)
    line_count = len(text.splitlines())
    
    # 进行分段
    segment_start = time.time()
    segments = segmenter.segment(text)
    segments = list(segments)  # 确保迭代器被完全消费
    segment_time = time.time() - segment_start
    
    # 写入结果
    write_start = time.time()
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}. {segment}\n")
    write_time = time.time() - write_start
    
    total_time = time.time() - total_start_time
    
    # 打印结果和性能统计
    print(f"\n处理完成！结果已保存到: {output_file}")
    print("\n性能统计:")
    print("-" * 50)
    print(f"输入文本统计:")
    print(f"- 字符数: {char_count}")
    print(f"- 行数: {line_count}")
    print(f"- 分段数: {len(segments)}")
    
    print(f"\n详细初始化耗时:")
    timing_stats = segmenter.get_timing_stats()
    if "total_init" in timing_stats:
        print(f"- 配置加载: {format_time(timing_stats.get('config_load', 0))}")
        print(f"- 模型创建: {format_time(timing_stats.get('model_create', 0))}")
        if "model_to_gpu" in timing_stats:
            print(f"- GPU加载: {format_time(timing_stats.get('model_to_gpu', 0))}")
        print(f"- 模型初始化总计: {format_time(timing_stats.get('model_init', 0))}")
        print(f"- 完整初始化总计: {format_time(timing_stats.get('total_init', 0))}")
    
    print(f"\n文本处理耗时:")
    print(f"- 文件读取: {format_time(file_read_time)}")
    print(f"- 文本分段: {format_time(timing_stats.get('segment_process', 0))}")
    print(f"- 结果写入: {format_time(write_time)}")
    print(f"- 总耗时: {format_time(total_time)}")
    print(f"- 分段处理速度: {char_count/timing_stats.get('segment_process', 0.001):.1f} 字符/秒")
    
    print("\n分段结果预览:")
    print("-" * 50)
    for i, segment in enumerate(segments, 1):
        print(f"{i}. {segment}")
    print("-" * 50)

if __name__ == "__main__":
    main()
