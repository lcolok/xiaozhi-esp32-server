# 文本分段实现文档

## 概述
本文档详细说明了系统中文本分段功能的实现，包括技术选型、架构设计、性能优化等方面。

## 技术栈
- **核心库**: wtpsplit
- **模型**: SaT (Segment any Text)
- **依赖**:
  - torch: 用于模型推理
  - wtpsplit: 文本分段库
  - pandas: wtpsplit的依赖库

## 实现架构

### 1. TextSegmenter 类
位置: `/core/utils/text_segmentation.py`

#### 1.1 设计模式
- 采用单例模式
- 确保模型只加载一次，避免重复加载带来的性能开销
```python
class TextSegmenter:
    _instance: Optional['TextSegmenter'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

#### 1.2 性能优化
- 使用 TimingStats 类跟踪各个阶段的性能数据
- 支持 GPU 加速（如果可用）
- 实现了批处理接口以提高处理效率

#### 1.3 配置项
```yaml
text_segmentation:
  model_name: "sat-3l-sm"  # 使用3层小型模型，平衡性能和效果
  threshold: 0.2  # 分段阈值，越高分段越保守
  style_or_domain: "ud"  # 使用Universal Dependencies风格
  language: "zh"  # 中文
```

### 2. 系统集成

#### 2.1 初始化流程
1. 服务启动时在 ConnectionHandler 初始化阶段预加载模型
```python
def __init__(self, websocket, config):
    try:
        start_time = time.time()
        self.text_segmenter = TextSegmenter.get_instance(config)
        init_time = time.time() - start_time
        self.logger.bind(tag=TAG).info(f"TextSegmenter 预加载完成，耗时：{init_time:.2f}秒")
    except Exception as e:
        self.logger.bind(tag=TAG).error(f"TextSegmenter 预加载失败: {e}")
```

#### 2.2 错误处理
- 实现了 fallback 机制，当模型分段失败时回退到基于规则的分段
- 完整的异常捕获和日志记录

## 性能指标

### 1. 初始化性能
- 模型加载时间: ~3-4秒（首次）
- 配置加载: ~20ms
- GPU加载（如果可用）: ~100ms

### 2. 运行时性能
- 文本分段速度: ~600-700字符/秒
- 内存占用: 
  - 模型加载: ~200-300MB
  - 运行时: 根据输入文本长度动态变化

## 测试工具

### 1. 专用测试脚本
位置: `/tests/text_segmentation/test_segmenter.py`

功能：
- 支持自定义测试文本
- 提供详细的性能统计
- 结果可视化展示

使用方法：
```bash
source .venv/bin/activate
python -m tests.text_segmentation.test_segmenter
```

### 2. 测试用例
示例文本位置: `/tests/text_segmentation/samples/`

## 最佳实践

### 1. 配置调优
- threshold 参数建议范围：0.1-0.3
  - 0.2: 默认值，适合一般场景
  - <0.2: 更激进的分段，段落更短
  - >0.2: 更保守的分段，段落更长

### 2. 模型选择
可选模型：
- sat-3l-sm: 当前使用，平衡性能和效果
- sat-12l-sm: 最佳效果，但速度较慢
- sat-1l-sm: 最快速度，但效果略差

### 3. 性能优化建议
1. 预加载模型避免冷启动延迟
2. 考虑使用批处理接口处理大量文本
3. 如有GPU，建议启用GPU加速

## 未来优化方向

1. **模型优化**
   - 模型量化减小内存占用
   - 探索更轻量级的模型架构

2. **功能增强**
   - 支持更多语言和领域
   - 添加自适应分段阈值

3. **性能提升**
   - 实现异步处理
   - 添加结果缓存机制

## 常见问题

1. **模型加载时间过长**
   - 已通过单例模式解决，确保只加载一次
   - 可考虑使用更小的模型版本

2. **分段结果不理想**
   - 调整 threshold 参数
   - 确认语言和领域设置正确

3. **内存占用过高**
   - 考虑使用更小的模型
   - 确保及时释放不需要的资源

## 参考资料

1. [wtpsplit 官方文档](https://github.com/segment-any-text/wtpsplit)
2. [SaT模型论文](https://arxiv.org/abs/2012.15688)
3. [项目测试报告](/tests/text_segmentation/README.md)
