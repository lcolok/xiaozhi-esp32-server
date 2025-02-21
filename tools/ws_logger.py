from mitmproxy import ctx
from mitmproxy.websocket import WebSocketMessage
import json
from datetime import datetime

def format_message(content):
    try:
        # 尝试解析并格式化 JSON
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except:
        # 如果不是 JSON，返回原始内容
        return content

def websocket_message(flow):
    # 确保是 WebSocket 连接
    if flow.websocket:
        # 获取最新的消息
        message = flow.websocket.messages[-1]
        
        # 确定消息方向
        direction = "Client → Server" if message.from_client else "Server → Client"
        
        # 格式化时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 尝试解码消息内容
        try:
            content = message.content.decode('utf-8')
            formatted_content = format_message(content)
        except:
            formatted_content = f"Binary message, length: {len(message.content)} bytes"
        
        # 打印带格式的消息
        log_message = f"""
{'=' * 50}
{timestamp} {direction}
{formatted_content}
{'=' * 50}
"""
        ctx.log.info(log_message)

def websocket_connected(flow):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ctx.log.info(f"\n{timestamp} WebSocket connection established")
    ctx.log.info(f"Client address: {flow.client_conn.address}")
    ctx.log.info(f"Server address: {flow.server_conn.address}")
    ctx.log.info(f"Request URL: {flow.request.pretty_url}")
    ctx.log.info("\nRequest headers:")
    for header, value in flow.request.headers.items():
        ctx.log.info(f"  {header}: {value}")
    if hasattr(flow, 'response') and flow.response:
        ctx.log.info("\nResponse headers:")
        for header, value in flow.response.headers.items():
            ctx.log.info(f"  {header}: {value}")

def websocket_error(flow):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ctx.log.error(f"\n{timestamp} WebSocket error occurred")
    if hasattr(flow, 'error'):
        ctx.log.error(f"Error details: {flow.error}")

def websocket_closed(flow):
    if flow.websocket:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        ctx.log.info(f"\n{timestamp} WebSocket connection closed")
        if hasattr(flow.websocket, 'close_code'):
            ctx.log.info(f"Close code: {flow.websocket.close_code}")
        if hasattr(flow.websocket, 'close_reason'):
            ctx.log.info(f"Close reason: {flow.websocket.close_reason}")

def request(flow):
    # 记录所有请求，不仅仅是WebSocket
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ctx.log.info(f"\n{timestamp} New request: {flow.request.method} {flow.request.pretty_url}")
    ctx.log.info("Request headers:")
    for header, value in flow.request.headers.items():
        ctx.log.info(f"  {header}: {value}")

def response(flow):
    # 记录所有响应
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ctx.log.info(f"\n{timestamp} Response: {flow.response.status_code}")
    ctx.log.info("Response headers:")
    for header, value in flow.response.headers.items():
        ctx.log.info(f"  {header}: {value}")
