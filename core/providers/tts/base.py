import asyncio
from abc import ABC, abstractmethod
import os
from typing import AsyncGenerator, Optional, List, Tuple
import numpy as np
import opuslib
from pydub import AudioSegment
from config.logger import setup_logging
import math
import librosa
import soundfile as sf
import io

TAG = __name__
logger = setup_logging()


class TTSProviderBase(ABC):
    def __init__(self, config, delete_audio_file):
        """初始化TTS基础类"""
        self.config = config
        self.delete_audio_file = delete_audio_file
        
        # 优先从xiaozhi配置中获取音频参数
        xiaozhi_config = config.get("xiaozhi", {})
        xiaozhi_audio_params = xiaozhi_config.get("audio_params", {})
        
        # 从客户端参数获取基本音频参数，如果没有则使用xiaozhi配置或默认值
        client_audio_params = config.get("audio_params", {})
        self.format = client_audio_params.get("format", xiaozhi_audio_params.get("format", "opus"))
        self.sample_rate = int(client_audio_params.get("sample_rate", xiaozhi_audio_params.get("sample_rate", 24000)))
        self.channels = int(client_audio_params.get("channels", xiaozhi_audio_params.get("channels", 1)))
        self.frame_duration = int(client_audio_params.get("frame_duration", xiaozhi_audio_params.get("frame_duration", 60)))
        
        # 增益相关参数始终从xiaozhi配置中获取
        self.gain = float(xiaozhi_audio_params.get("gain", 1.0))
        self.smart_gain = bool(xiaozhi_audio_params.get("smart_gain", True))
        
        logger.bind(tag=TAG).info(f"初始化音频参数 - 采样率: {self.sample_rate}Hz, 声道: {self.channels}, 帧时长: {self.frame_duration}ms, 增益: {self.gain}倍, 智能增益: {'开启' if self.smart_gain else '关闭'}")

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

    def _apply_gain(self, audio_segment):
        """应用音量增益，支持智能增益控制"""
        # 获取音频统计信息
        rms_before = audio_segment.rms
        dbfs_before = audio_segment.dBFS
        max_amp_before = audio_segment.max
        
        # 检查增益是否合理
        if self.gain <= 0:
            logger.bind(tag=TAG).warning(f"增益值 {self.gain} 不合法，将使用默认值 1.0")
            self.gain = 1.0
        
        # 记录原始音频信息
        logger.bind(tag=TAG).info(f"原始音频 - 格式: {audio_segment.channels}, 采样率: {audio_segment.frame_rate}Hz, 声道: {audio_segment.channels}, RMS: {rms_before}dB, dBFS: {dbfs_before}dB, 最大振幅: {max_amp_before}")
        
        # 根据智能增益开关决定处理方式
        target_gain = self.gain
        if self.smart_gain:
            if max_amp_before > 0:
                max_gain = 32767.0 / max_amp_before  # 计算不会导致截幅的最大增益
                safe_gain = min(self.gain, max_gain * 0.9)  # 留出10%的余量
                
                if safe_gain < self.gain:
                    logger.bind(tag=TAG).warning(f"智能增益：原始增益值 {self.gain} 可能导致失真，已自动调整为 {safe_gain:.2f}")
                    target_gain = safe_gain
        else:
            if self.gain > 10.0:
                logger.bind(tag=TAG).warning(f"增益值 {self.gain} 过大，可能导致音频失真")
            target_gain = self.gain
        
        # 应用增益
        logger.bind(tag=TAG).info(f"应用音量增益 - 目标增益: {target_gain}倍 (智能增益: {'开启' if self.smart_gain else '关闭'})")
        
        try:
            # 使用 pydub 内置的增益功能
            gain_db = 20 * math.log10(target_gain)
            audio_segment = audio_segment.apply_gain(gain_db)
            
            # 记录处理后的音频统计信息
            logger.bind(tag=TAG).info(f"增益后 - RMS: {audio_segment.rms}dB, dBFS: {audio_segment.dBFS}dB, 最大振幅: {audio_segment.max}")
            
            return audio_segment
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"应用音量增益失败: {str(e)}")
            return audio_segment  # 如果处理失败，返回原始音频

    def wav_to_opus_data(self, wav_file_path) -> Tuple[List[bytes], float]:
        """将 WAV 文件转换为 Opus 数据包列表和持续时间"""
        # 使用pydub加载PCM文件
        file_type = os.path.splitext(wav_file_path)[1]
        if file_type:
            file_type = file_type.lstrip('.')
        audio = AudioSegment.from_file(wav_file_path, format=file_type)
        
        # 记录原始音频信息
        logger.bind(tag=TAG).info(f"原始音频 - 格式: {file_type}, 采样率: {audio.frame_rate}Hz, 声道: {audio.channels}, RMS: {audio.rms}dB, dBFS: {audio.dBFS}dB")

        # 应用音量增益
        if abs(self.gain - 1.0) > 0.01:  # 如果增益不等于1.0（考虑浮点数比较）
            logger.bind(tag=TAG).info(f"应用音量增益 - 目标增益: {self.gain}倍")
            try:
                audio = self._apply_gain(audio)
                logger.bind(tag=TAG).info(f"增益后 - RMS: {audio.rms}dB, dBFS: {audio.dBFS}dB")
            except Exception as e:
                logger.bind(tag=TAG).error(f"应用音量增益失败: {str(e)}")

        duration = len(audio) / 1000.0

        # 转换为配置的声道数和采样率
        audio = audio.set_channels(self.channels).set_frame_rate(self.sample_rate)
        raw_data = audio.raw_data

        # 初始化Opus编码器
        encoder = opuslib.Encoder(self.sample_rate, self.channels, opuslib.APPLICATION_AUDIO)

        # 编码参数
        frame_size = int(self.sample_rate * self.frame_duration / 1000)

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
            frame_rate=self.sample_rate,
            channels=self.channels
        )

        # 应用音量增益
        audio = self._apply_gain(audio)

        duration = len(audio) / 1000.0
        raw_data = audio.raw_data

        # 初始化Opus编码器
        encoder = opuslib.Encoder(self.sample_rate, self.channels, opuslib.APPLICATION_AUDIO)

        # 编码参数
        frame_size = int(self.sample_rate * self.frame_duration / 1000)

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
