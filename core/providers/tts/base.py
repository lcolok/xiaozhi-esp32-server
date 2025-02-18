import asyncio
from abc import ABC, abstractmethod
import os
from typing import AsyncGenerator, Optional, List, Tuple
import numpy as np
import opuslib
from pydub import AudioSegment
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProviderBase(ABC):
    def __init__(self, config, delete_audio_file):
        self.delete_audio_file = delete_audio_file
        self.output_file = config.get("output_file")
        self.supports_streaming = False  # 默认不支持流式输出

    @abstractmethod
    def generate_filename(self):
        pass

    def to_tts(self, text):
        """非流式文本转语音的默认实现"""
        tmp_file = self.generate_filename()
        try:
            max_repeat_time = 5
            while not os.path.exists(tmp_file) and max_repeat_time > 0:
                asyncio.run(self.text_to_speak(text, tmp_file))
                if not os.path.exists(tmp_file):
                    max_repeat_time = max_repeat_time - 1
                    logger.bind(tag=TAG).error(f"语音生成失败: {text}:{tmp_file}，再试{max_repeat_time}次")

            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(f"语音生成成功: {text}:{tmp_file}，重试{5 - max_repeat_time}次")

            return tmp_file
        except Exception as e:
            logger.bind(tag=TAG).info(f"Failed to generate TTS file: {e}")
            return None

    @abstractmethod
    async def text_to_speak(self, text: str, output_file: str) -> None:
        """非流式文本转语音接口"""
        pass

    async def stream_text_to_speak(self, text: str) -> AsyncGenerator[Tuple[bytes, float], None]:
        """
        流式文本转语音接口
        返回: 生成器，每次返回 (音频数据, 持续时间)
        """
        raise NotImplementedError("This TTS provider does not support streaming")

    def wav_to_opus_data(self, wav_file_path) -> Tuple[List[bytes], float]:
        """将 WAV 文件转换为 Opus 数据包列表和持续时间"""
        # 使用pydub加载PCM文件
        file_type = os.path.splitext(wav_file_path)[1]
        if file_type:
            file_type = file_type.lstrip('.')
        audio = AudioSegment.from_file(wav_file_path, format=file_type)

        duration = len(audio) / 1000.0

        # 转换为单声道和16kHz采样率
        audio = audio.set_channels(1).set_frame_rate(16000)
        raw_data = audio.raw_data

        # 初始化Opus编码器
        encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)

        # 编码参数
        frame_duration = 60  # 60ms per frame
        frame_size = int(16000 * frame_duration / 1000)  # 960 samples/frame

        opus_datas = []
        # 按帧处理音频数据
        for i in range(0, len(raw_data), frame_size * 2):
            chunk = raw_data[i:i + frame_size * 2]
            if len(chunk) < frame_size * 2:
                chunk += b'\x00' * (frame_size * 2 - len(chunk))
            np_frame = np.frombuffer(chunk, dtype=np.int16)
            opus_data = encoder.encode(np_frame.tobytes(), frame_size)
            opus_datas.append(opus_data)

        return opus_datas, duration

    async def stream_to_opus(self, audio_data: bytes) -> Tuple[List[bytes], float]:
        """将原始音频数据流转换为 Opus 数据包列表和持续时间"""
        # 将字节数据转换为 AudioSegment
        audio = AudioSegment(
            data=audio_data,
            sample_width=2,  # 16-bit
            frame_rate=16000,
            channels=1
        )

        duration = len(audio) / 1000.0
        raw_data = audio.raw_data

        # 初始化Opus编码器
        encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
        frame_duration = 60
        frame_size = int(16000 * frame_duration / 1000)

        opus_datas = []
        for i in range(0, len(raw_data), frame_size * 2):
            chunk = raw_data[i:i + frame_size * 2]
            if len(chunk) < frame_size * 2:
                chunk += b'\x00' * (frame_size * 2 - len(chunk))
            np_frame = np.frombuffer(chunk, dtype=np.int16)
            opus_data = encoder.encode(np_frame.tobytes(), frame_size)
            opus_datas.append(opus_data)

        return opus_datas, duration
