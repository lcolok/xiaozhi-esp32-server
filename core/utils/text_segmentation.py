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
    _timing_stats = TimingStats()
    
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
            
            self._timing_stats.start("total_init")
            
            self._timing_stats.start("config_load")    
            self.config = config.get("text_segmentation", {})
            self.model_name = self.config.get("model_name", "sat-3l-sm")
            self.threshold = self.config.get("threshold", 0.5)
            self.style_or_domain = self.config.get("style_or_domain")
            self.language = self.config.get("language")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # 添加快速首句提取的配置
            self.min_punctuation_count = self.config.get("min_punctuation_count", 2)
            self._timing_stats.end("config_load")
            
            # 初始化模型
            try:
                self._timing_stats.start("model_init")
                # 如果指定了语言和领域，使用适配的模型
                if self.style_or_domain and self.language:
                    self._timing_stats.start("model_create")
                    self.model = SaT(
                        self.model_name,
                        style_or_domain=self.style_or_domain,
                        language=self.language
                    )
                    self._timing_stats.end("model_create")
                else:
                    self._timing_stats.start("model_create")
                    self.model = SaT(self.model_name)
                    self._timing_stats.end("model_create")
                
                # 如果有GPU，使用半精度和GPU加速
                if self.device == "cuda":
                    self._timing_stats.start("model_to_gpu")
                    self.model.half().to(self.device)
                    self._timing_stats.end("model_to_gpu")
                
                self._timing_stats.end("model_init")
                
                logger.bind(tag=TAG).info(
                    f"TextSegmenter singleton initialized with model {self.model_name} "
                    f"on {self.device}"
                    + (f" (adapted for {self.language} {self.style_or_domain})" 
                       if self.style_or_domain and self.language else "")
                )
                self._initialized = True
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to initialize TextSegmenter singleton: {e}")
                raise
            finally:
                self._timing_stats.end("total_init")

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
        return cls._timing_stats.get_stats()

    def _quick_first_sentence(self, text: str) -> tuple[str, str]:
        """快速提取第一句话，基于简单的标点符号计数
        
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
        
        # 遍历文本寻找标点
        for match in re.finditer(punct_pattern, text):
            punct_count += 1
            last_punct_pos = match.end()
            
            # 如果达到所需的标点数量
            if punct_count >= self.min_punctuation_count:
                break
        
        # 如果找到了足够的标点
        if last_punct_pos > 0:
            first_sentence = text[:last_punct_pos].strip()
            remaining_text = text[last_punct_pos:].strip()
            return first_sentence, remaining_text
        
        # 如果没有找到足够的标点，返回空和原文本
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
                    return segments
            
            # 如果快速提取没有结果或未启用，使用原有逻辑
            segments = self.model.split(text, threshold=self.threshold)
            return [seg.strip() for seg in segments if seg.strip()]
            
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
    
    def segment_stream(self, text: str, is_first: bool = False) -> List[str]:
        """流式文本分段，针对流式返回的文本进行优化的分段方法
        
        Args:
            text: 输入文本
            is_first: 是否是第一段文本
            
        Returns:
            分段后的文本列表，最后一个可能是不完整的句子
        """
        try:
            self._timing_stats.start("segment_stream_process")
            
            # 如果是第一段文本且启用了快速首句提取
            if is_first:
                first_sentence, remaining_text = self._quick_first_sentence(text)
                if first_sentence:
                    if not remaining_text:
                        return [first_sentence]
                    text = remaining_text
                    segments = [first_sentence]
                else:
                    segments = []
            else:
                segments = []
            
            # 找到最后一个句子的结束位置
            last_punct_pos = self._find_last_sentence_break(text)
            
            if last_punct_pos > 0:
                # 处理完整的句子部分
                complete_text = text[:last_punct_pos]
                try:
                    complete_segments = self.model.split(complete_text, threshold=self.threshold)
                    segments.extend([seg.strip() for seg in complete_segments if seg.strip()])
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Error during model split: {e}")
                    # 如果模型分段失败，使用简单的标点分段
                    segments.extend(self._fallback_segment(complete_text))
                
                # 添加未完成的句子部分
                remaining_text = text[last_punct_pos:].strip()
                if remaining_text:
                    segments.append(remaining_text)
            else:
                # 如果没有找到句子结束，整个文本作为未完成句子
                if text.strip():
                    segments.append(text.strip())
            
            return segments
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during stream segmentation: {e}")
            # 如果分段失败，回退到简单的标点符号分段
            return self._fallback_segment(text)
        finally:
            self._timing_stats.end("segment_stream_process")

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
