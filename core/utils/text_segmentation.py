import torch
from wtpsplit import SaT
from typing import List, Iterator, Optional, Dict
import time
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

class TimingStats:
    """用于记录时间统计的辅助类"""
    def __init__(self):
        self.stats: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        """开始计时"""
        self._start_times[name] = time.time()

    def end(self, name: str):
        """结束计时"""
        if name in self._start_times:
            duration = time.time() - self._start_times[name]
            self.stats[name] = self.stats.get(name, 0) + duration
            del self._start_times[name]

    def get_stats(self) -> Dict[str, float]:
        """获取统计结果"""
        return self.stats.copy()

class TextSegmenter:
    _instance: Optional['TextSegmenter'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: dict = None):
        """初始化文本分段器，使用单例模式确保只初始化一次
        
        Args:
            config: 配置字典，包含分段相关的配置。只在第一次初始化时使用。
        """
        # 确保只初始化一次
        if not self._initialized:
            if config is None:
                raise ValueError("Config must be provided for first initialization")
            
            self.logger = setup_logging()
            self.config = config
            
            # 从配置中读取参数
            text_segmentation_config = config.get("text_segmentation", {})
            self.model_name = text_segmentation_config.get("model_name", "sat-12l-sm")
            self.threshold = text_segmentation_config.get("threshold", 0.95)
            self.style_or_domain = text_segmentation_config.get("style_or_domain", "ud")
            self.language = text_segmentation_config.get("language", "zh")
            self.max_combined_length = text_segmentation_config.get("max_combined_length", 100)
            self.min_segment_length = text_segmentation_config.get("min_segment_length", 20)
            self.segments_per_group = text_segmentation_config.get("segments_per_group", 3)
            self.min_punctuation_count = text_segmentation_config.get("min_punctuation_count", 5)
            
            # 初始化计时器
            self._timing_stats = TimingStats()
            self._timing_stats.start("init")
            self._timing_stats.start("config_load")
            
            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._timing_stats.end("config_load")
            
            # 初始化模型
            self._timing_stats.start("model_load")
            self._init_model()
            self._timing_stats.end("model_load")
            self._timing_stats.end("init")
            
            logger.bind(tag=TAG).info(
                f"TextSegmenter singleton initialized with model {self.model_name} "
                f"on {self.device}"
                + (f" (adapted for {self.language} {self.style_or_domain})" 
                   if self.style_or_domain and self.language else "")
            )
            self._initialized = True
        else:
            self.config = config

    @classmethod
    def get_instance(cls, config: dict = None) -> 'TextSegmenter':
        """获取 TextSegmenter 的单例实例
        
        Args:
            config: 配置字典，只在第一次初始化时使用
            
        Returns:
            TextSegmenter 实例
        """
        if cls._instance is None:
            if config is None:
                raise ValueError("Config must be provided for first initialization")
            return cls(config)
        return cls._instance

    @classmethod
    def get_timing_stats(cls) -> Dict[str, float]:
        """获取时间统计信息"""
        return cls._instance._timing_stats.get_stats()

    def _init_model(self):
        try:
            # 初始化模型
            if self.style_or_domain and self.language:
                self.model = SaT(
                    self.model_name,
                    style_or_domain=self.style_or_domain,
                    language=self.language
                )
            else:
                self.model = SaT(self.model_name)
            
            # 如果有GPU，使用半精度和GPU加速
            if self.device == "cuda":
                self.model.half().to(self.device)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to initialize TextSegmenter singleton: {e}")
            raise

    def _quick_first_sentence(self, text: str) -> tuple[str, str]:
        """快速提取第一句话，基于简单的标点符号计数和最小长度要求
        
        Args:
            text: 输入文本
            
        Returns:
            tuple: (第一句话, 剩余文本)
        """
        import re
        
        # 定义标点符号模式
        punct_pattern = r'[。！？!?.,，]'
        
        # 初始化计数器和上一个标点位置
        punct_count = 0
        last_punct_pos = -1
        current_length = 0
        
        # 遍历文本寻找标点
        for match in re.finditer(punct_pattern, text):
            punct_count += 1
            current_pos = match.end()
            current_length = current_pos
            
            # 如果达到所需的标点数量且长度足够
            if punct_count >= self.min_punctuation_count and current_length >= self.min_segment_length:
                last_punct_pos = current_pos
                break
            # 继续寻找直到找到一个长度足够的句子
            last_punct_pos = current_pos
        
        # 如果找到了合适的断句点
        if last_punct_pos > 0:
            first_sentence = text[:last_punct_pos].strip()
            remaining_text = text[last_punct_pos:].strip()
            # 再次检查长度，如果太短就返回空
            if len(first_sentence) < self.min_segment_length:
                return "", text
            return first_sentence, remaining_text
        
        # 如果没有找到合适的断句点，返回空和原文本
        return "", text

    def _find_last_sentence_break(self, text: str) -> int:
        """在文本中找到最后一个完整句子的结束位置
        
        Args:
            text: 输入文本
            
        Returns:
            int: 最后一个完整句子的结束位置，如果没有找到则返回-1
        """
        import re
        
        # 定义标点符号模式
        punct_pattern = r'[。！？!?.,，]'
        
        last_punct_pos = -1
        for match in re.finditer(punct_pattern, text):
            last_punct_pos = match.end()
        
        return last_punct_pos

    def _combine_segments(self, segments: List[str]) -> List[str]:
        """合并短段落，确保每个段落的长度在合适的范围内
        
        Args:
            segments: 原始分段列表
            
        Returns:
            合并后的分段列表
        """
        if not segments:
            return segments
            
        # 如果设置了固定的分组大小
        if self.segments_per_group > 0:
            combined = []
            current_group = []
            
            for segment in segments:
                current_group.append(segment)
                
                # 当达到指定的段落数量或是最后一段时
                if len(current_group) >= self.segments_per_group or segment == segments[-1]:
                    combined_text = "".join(current_group)
                    # 如果合并后的文本超过最大长度，则分开处理
                    if len(combined_text) > self.max_combined_length:
                        # 将超长的组按长度拆分
                        temp = ""
                        for seg in current_group:
                            if len(temp) + len(seg) <= self.max_combined_length:
                                temp += seg
                            else:
                                if temp:
                                    combined.append(temp)
                                temp = seg
                        if temp:
                            combined.append(temp)
                    else:
                        combined.append(combined_text)
                    current_group = []
            
            # 处理剩余的段落
            if current_group:
                combined_text = "".join(current_group)
                if len(combined_text) <= self.max_combined_length:
                    combined.append(combined_text)
                else:
                    combined.extend(current_group)
            
            return combined
        
        # 如果没有设置固定分组大小，使用基于长度的合并逻辑
        combined = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # 如果当前段落加上下一段不超过最大长度，就合并
            if len(current) + len(next_seg) <= self.max_combined_length:
                current = current + next_seg
            else:
                # 如果当前段落长度大于最小长度，就保存当前段落
                if len(current) >= self.min_segment_length:
                    combined.append(current)
                current = next_seg
        
        # 处理最后一个段落
        if current:
            combined.append(current)
            
        return combined

    def segment(self, text: str, quick_first: bool = True) -> List[str]:
        """将文本分段
        
        Args:
            text: 输入文本
            quick_first: 是否使用快速首句提取
            
        Returns:
            分段后的文本列表
        """
        try:
            self._timing_stats.start("segment_process")
            
            # 如果启用快速首句提取
            if quick_first:
                first_sentence, remaining_text = self._quick_first_sentence(text)
                if first_sentence:
                    # 如果有剩余文本，使用模型处理
                    if remaining_text:
                        remaining_segments = self.model.split(remaining_text, threshold=self.threshold)
                        segments = [first_sentence] + [seg.strip() for seg in remaining_segments if seg.strip()]
                    else:
                        segments = [first_sentence]
                else:
                    # 如果没有找到首句，直接使用模型处理全文
                    segments = [seg.strip() for seg in self.model.split(text, threshold=self.threshold) if seg.strip()]
            else:
                # 不使用快速首句提取，直接使用模型处理全文
                segments = [seg.strip() for seg in self.model.split(text, threshold=self.threshold) if seg.strip()]
            
            # 合并短段落
            segments = self._combine_segments(segments)
            
            return segments
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during text segmentation: {e}")
            # 如果分段失败，回退到简单的标点符号分段
            return self._fallback_segment(text)
        finally:
            self._timing_stats.end("segment_process")

    def segment_batch(self, texts: List[str]) -> Iterator[List[str]]:
        """批量处理文本分段
        
        Args:
            texts: 文本列表
            
        Returns:
            分段后的文本列表的迭代器
        """
        try:
            self._timing_stats.start("batch_segment_process")
            return self.model.split(texts, threshold=self.threshold)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during batch text segmentation: {e}")
            return map(self._fallback_segment, texts)
        finally:
            self._timing_stats.end("batch_segment_process")
    
    def segment_stream(self, text: str, is_first_segment: bool = True) -> List[str]:
        """流式分段，用于处理实时生成的文本
        
        Args:
            text: 输入文本
            is_first_segment: 是否是第一次分段
            
        Returns:
            分段后的文本列表
        """
        try:
            # 如果是第一次分段，优先提取第一句话
            if is_first_segment:
                first_sentence, remaining_text = self._quick_first_sentence(text)
                if first_sentence:
                    if not remaining_text:
                        return [first_sentence]
                        
                    # 处理剩余文本
                    remaining_segments = self.model.split(remaining_text, threshold=self.threshold)
                    remaining_segments = [seg.strip() for seg in remaining_segments if seg.strip()]
                    
                    # 对剩余部分进行合并
                    if remaining_segments:
                        combined_remaining = self._combine_segments(remaining_segments)
                        return [first_sentence] + combined_remaining
                    
                    return [first_sentence]
            
            # 非首次分段，使用普通分段然后合并
            segments = self.segment(text, quick_first=False)
            return self._combine_segments(segments)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during stream text segmentation: {e}")
            return [text]

    def _fallback_segment(self, text: str) -> List[str]:
        """简单的基于标点的分段方法，作为后备方案
        
        Args:
            text: 输入文本
            
        Returns:
            分段后的文本列表
        """
        import re
        # 使用常见的中文和英文标点作为分隔符
        segments = re.split(r'([。！？!?])', text)
        # 将分隔符附加到对应的文本后
        result = []
        for i in range(0, len(segments)-1, 2):
            if i+1 < len(segments):
                result.append(segments[i] + segments[i+1])
            else:
                result.append(segments[i])
        return [seg.strip() for seg in result if seg.strip()]
