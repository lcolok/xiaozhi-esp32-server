import time
import wave
import os
from abc import ABC, abstractmethod
from config.logger import setup_logging
from typing import Optional, Tuple, List
import uuid
import requests

import opuslib
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

TAG = __name__
logger = setup_logging()

class ASR(ABC):
    @abstractmethod
    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """解码Opus数据并保存为WAV文件"""
        pass

    @abstractmethod
    def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass


class ASRProviderBase(ABC):
    def __init__(self, config):
        self.config = config
        audio_params = config.get("xiaozhi", {}).get("audio_params", {})
        self.sample_rate = audio_params.get("sample_rate", 24000)
        self.channels = audio_params.get("channels", 1)
        self.frame_duration = audio_params.get("frame_duration", 60)
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将Opus音频数据解码并保存为WAV文件"""
        file_path = f"data/asr_{session_id}.wav"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        decoder = opuslib.Decoder(self.sample_rate, self.channels)
        pcm_data = bytearray()

        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, self.frame_size)
                pcm_data.extend(pcm_frame)
            except opuslib.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus解码错误: {e}", exc_info=True)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)

        return file_path


class FunASR(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__(config)
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")  # 修正配置键名
        self.delete_audio_file = delete_audio_file

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = AutoModel(
            model=self.model_dir,
            vad_kwargs={"max_single_segment_time": 30000},
            disable_update=True,
            hub="hf"
            # device="cuda:0",  # 启用GPU加速
        )

    def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.bind(tag=TAG).debug(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 语音识别
            start_time = time.time()
            result = self.model.generate(
                input=file_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            text = rich_transcription_postprocess(result[0]["text"])
            logger.bind(tag=TAG).debug(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
            return None, None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"文件删除失败: {file_path} | 错误: {e}")


class SiliconflowASR(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__(config)
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY environment variable is not set")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file
        self.model = config.get("model", "FunAudioLLM/SenseVoiceSmall")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.bind(tag=TAG).debug(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 调用Siliconflow ASR API
            start_time = time.time()
            url = "https://api.siliconflow.cn/v1/audio/transcriptions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            files = {
                "file": ("audio.wav", open(file_path, "rb"), "audio/wav"),
                "model": (None, self.model)
            }

            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            
            text = response.json().get("text", "")
            logger.bind(tag=TAG).debug(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
            return None, None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"文件删除失败: {file_path} | 错误: {e}")


def create_instance(class_name: str, *args, **kwargs) -> ASR:
    """工厂方法创建ASR实例"""
    cls_map = {
        "FunASR": FunASR,
        "SiliconflowASR": SiliconflowASR,
        # 可扩展其他ASR实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的ASR类型: {class_name}")
