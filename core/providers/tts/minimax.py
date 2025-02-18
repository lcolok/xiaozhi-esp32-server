import json
import os
import time
from typing import Dict, Any, Optional, AsyncGenerator, Tuple

import aiohttp
import requests
from loguru import logger

from core.providers.tts.base import TTSProviderBase

TAG = __name__


class TTSProvider(TTSProviderBase):
    def __init__(self, config: Dict[str, Any], delete_audio_file: bool):
        super().__init__(config, delete_audio_file)
        self.base_url = config.get('base_url')
        self.group_id = config.get('group_id')
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'speech-01-turbo')
        self.voice_id = config.get('voice_id', 'male-qn-qingse')
        self.voice_setting = config.get('voice_setting', {
            'speed': 1.0,
            'vol': 1.0,
            'pitch': 0,
            'emotion': 'happy'
        })
        self.audio_setting = config.get('audio_setting', {
            'sample_rate': 32000,
            'bitrate': 128000,
            'format': 'mp3',
            'channel': 1
        })
        self.output_file = config.get('output_file', 'tmp/')
        os.makedirs(self.output_file, exist_ok=True)
        self.supports_streaming = True  # MiniMax 支持流式输出

    def generate_filename(self, extension=None):
        """生成输出文件名"""
        if extension is None:
            extension = f".{self.audio_setting['format']}"
        return os.path.join(self.output_file, f'minimax_tts_{hash(str(time.time()))}{extension}')

    async def text_to_speak(self, text: str, output_file: str) -> None:
        """非流式文本转语音接口"""
        url = f"{self.base_url}?GroupId={self.group_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": self.voice_id,
                **self.voice_setting
            },
            "audio_setting": self.audio_setting
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get('base_resp', {}).get('status_code') != 0:
                error_msg = response_data.get('base_resp', {}).get('status_msg', 'Unknown error')
                logger.bind(tag=TAG).error(f"MiniMax TTS API error: {error_msg}")
                return

            audio_data = response_data.get('data', {}).get('audio')
            if not audio_data:
                logger.bind(tag=TAG).error("No audio data in response")
                return

            with open(output_file, 'wb') as f:
                f.write(bytes.fromhex(audio_data))

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in MiniMax TTS: {str(e)}")
            return

    async def stream_text_to_speak(self, text: str) -> AsyncGenerator[Tuple[bytes, float], None]:
        """流式文本转语音接口"""
        url = f"{self.base_url}?GroupId={self.group_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": self.voice_id,
                **self.voice_setting
            },
            "audio_setting": self.audio_setting
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line.startswith(b'data: '):
                            try:
                                json_data = json.loads(line[6:])
                                if json_data.get('base_resp', {}).get('status_code') != 0:
                                    error_msg = json_data.get('base_resp', {}).get('status_msg', 'Unknown error')
                                    logger.bind(tag=TAG).error(f"MiniMax TTS API error: {error_msg}")
                                    return

                                audio_data = json_data.get('data', {}).get('audio')
                                if audio_data:
                                    audio_bytes = bytes.fromhex(audio_data)
                                    # 将音频数据转换为 Opus 格式
                                    opus_packets, duration = await self.stream_to_opus(audio_bytes)
                                    for packet in opus_packets:
                                        yield packet, duration / len(opus_packets)
                            except json.JSONDecodeError:
                                logger.bind(tag=TAG).error("Failed to decode JSON from stream")
                            except Exception as e:
                                logger.bind(tag=TAG).error(f"Error processing stream chunk: {str(e)}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in MiniMax streaming TTS: {str(e)}")
            return
