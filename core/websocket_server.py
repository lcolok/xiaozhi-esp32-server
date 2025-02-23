import asyncio
import websockets
from config.logger import setup_logging
from core.connection import ConnectionHandler
from core.utils.util import get_local_ip
from core.utils import asr, vad, llm, tts

TAG = __name__

class WebSocketServer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging()
        self.local_ip = get_local_ip()
        
        # 记录配置信息
        audio_params = config.get("xiaozhi", {}).get("audio_params", {})
        self.logger.bind(tag=TAG).info(f"加载配置 - 音频参数: {audio_params}")
        
        # 初始化组件
        self.vad = vad.create_instance(
            self.config["selected_module"]["VAD"],
            self.config["VAD"][self.config["selected_module"]["VAD"]]
        )
        self.asr = asr.create_instance(
            self.config["selected_module"]["ASR"],
            self.config["ASR"][self.config["selected_module"]["ASR"]],
            self.config["delete_audio"]
        )
        self.llm = llm.create_instance(
            self.config["selected_module"]["LLM"]
            if not 'type' in self.config["LLM"][self.config["selected_module"]["LLM"]]
            else
            self.config["LLM"][self.config["selected_module"]["LLM"]]['type'],
            self.config["LLM"][self.config["selected_module"]["LLM"]],
        )
        self.tts = tts.create_instance(
            self.config["selected_module"]["TTS"]
            if not 'type' in self.config["TTS"][self.config["selected_module"]["TTS"]]
            else
            self.config["TTS"][self.config["selected_module"]["TTS"]]["type"],
            self.config["TTS"][self.config["selected_module"]["TTS"]],
            self.config["delete_audio"]
        )

    async def start(self):
        server_config = self.config["server"]
        host = server_config["ip"]
        port = server_config["port"]

        self.logger.bind(tag=TAG).info("Server is running at ws://{}:{}", self.local_ip, port)
        self.logger.bind(tag=TAG).info("=======上面的地址是websocket协议地址，请勿用浏览器访问=======")
        async with websockets.serve(
                self._handle_connection,
                host,
                port
        ):
            await asyncio.Future()

    async def _handle_connection(self, websocket):
        """处理新连接，每次创建独立的ConnectionHandler"""
        handler = ConnectionHandler(self.config, self.vad, self.asr, self.llm, self.tts)
        await handler.handle_connection(websocket)
