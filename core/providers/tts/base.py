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
        self.delete_audio_file = delete_audio_file
        self.output_file = config.get("output_file")
        self.supports_streaming = False  # 默认不支持流式输出
        
        # 从配置中获取音频参数
        xiaozhi_config = config.get("xiaozhi", {})
        xiaozhi_audio_params = xiaozhi_config.get("audio_params", {})
        
        # 从客户端参数或默认值获取基本音频参数
        client_audio_params = config.get("audio_params", {})
        self.sample_rate = client_audio_params.get("sample_rate", xiaozhi_audio_params.get("sample_rate", 24000))
        self.channels = client_audio_params.get("channels", xiaozhi_audio_params.get("channels", 1))
        self.frame_duration = client_audio_params.get("frame_duration", xiaozhi_audio_params.get("frame_duration", 60))
        
        # 增益参数始终从xiaozhi配置中获取
        self.gain = float(xiaozhi_audio_params.get("gain", 1.0))
        
        logger.bind(tag=TAG).info(f"初始化音频参数 - 采样率: {self.sample_rate}Hz, 声道: {self.channels}, 帧时长: {self.frame_duration}ms, 增益: {self.gain}倍")

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

    def _apply_audio_gain(self, audio: AudioSegment) -> AudioSegment:
        """应用音量增益，并防止爆音"""
        if self.gain == 1.0:
            return audio
            
        # 记录处理前的音量信息
        logger.bind(tag=TAG).info(f"音量增益处理前 - RMS: {audio.rms}dB, dBFS: {audio.dBFS}dB, Max Amplitude: {audio.max}")
        
        # 将 AudioSegment 转换为 numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # 转换为 float32，范围 [-1, 1]
        samples = samples.astype(np.float32) / 32768.0
        
        # 使用 librosa 的 normalize 函数进行音量增益
        # ref_db 参数控制目标音量大小
        target_db = 20 * math.log10(self.gain)
        processed_samples = librosa.util.normalize(samples, norm=np.inf, ref_db=target_db)
        
        # 应用动态范围压缩，防止爆音
        processed_samples = librosa.effects.preemphasis(processed_samples)
        
        # 转回 int16 范围
        processed_samples = np.clip(processed_samples * 32768.0, -32768, 32767).astype(np.int16)
        
        # 将处理后的数据转回 AudioSegment
        processed_audio = AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=audio.channels
        )
        
        # 记录处理后的音量信息
        logger.bind(tag=TAG).info(f"音量增益处理后 - RMS: {processed_audio.rms}dB, dBFS: {processed_audio.dBFS}dB, Max Amplitude: {processed_audio.max}")
        
        return processed_audio

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
                # 将增益值转换为dB并应用
                gain_db = 20 * math.log10(self.gain)
                audio = audio.apply_gain(gain_db)
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
        audio = self._apply_audio_gain(audio)

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
