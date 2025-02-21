# 文本分割模块优化记录

## 更新时间
2025-02-21

## 主要改动

### 1. 文本分割策略优化
在 `config.yaml` 中添加了新的配置选项，支持多种文本分割策略：

```yaml
text_segmentation:
  # 分割策略: 
  # - punctuation: 仅使用标点符号分割，速度最快但效果最简单
  # - model: 使用AI模型分割，效果最好但需要加载模型
  strategy: "punctuation"
  
  # 是否对第一句话使用特殊处理（使用标点符号快速提取第一句话）
  quick_first_sentence: true
```

### 2. 性能优化
- 根据选择的策略按需加载模型，避免不必要的资源占用
- 当选择 `punctuation` 策略时，系统不会加载AI模型，可以显著减少启动时间
- 添加了模型加载失败的优雅降级处理，自动切换到标点分割方式

### 3. 配置结构优化
将配置项分为三类：
1. 策略选择配置
   - `strategy`: 选择分割策略
   - `quick_first_sentence`: 是否对第一句话特殊处理

2. 通用配置
   - `max_combined_length`: 合并后的段落最大字符数
   - `min_segment_length`: 最小段落长度
   - `segments_per_group`: 每组合并的段落数量
   - `min_punctuation_count`: 快速提取第一句话时需要的最少标点符号数量

3. 模型专用配置（仅在 strategy="model" 时生效）
   ```yaml
   model_config:
     model_name: "sat-12l-sm"
     threshold: 0.3
     style_or_domain: "ud"
     language: "zh"
   ```

### 4. 提示词优化
更新了系统提示词，使其更专业和灵活：
```yaml
prompt: |
  你是一个叫小智台湾女孩，习惯简短表达，但是如果用户需要详细回答的时候，也会使用对应专业和严谨的态度去回答所需要的详细的答案。
```

## 使用建议
1. 如果系统资源受限或对响应速度要求高，建议使用 `punctuation` 策略
2. 如果需要更精确的分段效果，可以使用 `model` 策略
3. 无论使用哪种策略，都可以通过 `quick_first_sentence` 配置来控制是否对第一句话进行特殊处理

## 技术实现
相关代码实现在 `core/utils/text_segmentation.py` 中，主要包括：
- 支持多种分割策略的 `TextSegmenter` 类
- 基于标点的分割实现
- 模型加载和使用的封装
- 错误处理和降级机制

## 后续优化方向
1. 可以考虑添加更多的分割策略
2. 优化模型加载时间
3. 添加更多的配置选项来细化控制分割行为
4. 实现分割效果的评估机制
