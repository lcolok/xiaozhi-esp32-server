from mitmproxy import ctx
from mitmproxy.websocket import WebSocketMessage
import json
from datetime import datetime
from collections import defaultdict

# 用于跟踪二进制消息的统计信息
binary_stats = {
    'client_to_server': {'count': 0, 'total_bytes': 0},
    'server_to_client': {'count': 0, 'total_bytes': 0},
    'last_print_time': datetime.now()
}

def format_message(content):
    try:
        # 尝试解析并格式化 JSON
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except:
        return content

def print_binary_stats():
    c2s = binary_stats['client_to_server']
    s2c = binary_stats['server_to_client']
    
    if c2s['count'] > 0 or s2c['count'] > 0:
        stats_message = f"Binary: C→S: {c2s['count']}条 ({c2s['total_bytes']/1024:.1f}KB) | S→C: {s2c['count']}条 ({s2c['total_bytes']/1024:.1f}KB)"
        ctx.log.info(stats_message)
        # 重置统计
        c2s['count'] = 0
        c2s['total_bytes'] = 0
        s2c['count'] = 0
        s2c['total_bytes'] = 0
        binary_stats['last_print_time'] = datetime.now()

def websocket_message(flow):
    if flow.websocket:
        message = flow.websocket.messages[-1]
        direction = "C→S" if message.from_client else "S→C"
        stats_key = 'client_to_server' if message.from_client else 'server_to_client'
        
        try:
            content = message.content.decode('utf-8')
            formatted_content = format_message(content)
            # 只打印非单字符的文本消息
            if len(content.strip()) > 1:
                log_message = f"{direction}\n{formatted_content}"
                ctx.log.info(log_message)
        except:
            # 累计二进制消息统计
            binary_stats[stats_key]['count'] += 1
            binary_stats[stats_key]['total_bytes'] += len(message.content)
            
            time_diff = (datetime.now() - binary_stats['last_print_time']).total_seconds()
            total_messages = (binary_stats['client_to_server']['count'] + 
                            binary_stats['server_to_client']['count'])
            
            if time_diff >= 10 or total_messages >= 100:
                print_binary_stats()

def websocket_connected(flow):
    ctx.log.info(f"WebSocket connected: {flow.request.pretty_url}")

def websocket_error(flow):
    ctx.log.error(f"WebSocket error: {flow.error if hasattr(flow, 'error') else 'Unknown error'}")

def websocket_closed(flow):
    if flow.websocket:
        ctx.log.info("WebSocket closed")

def request(flow):
    ctx.log.info(f"{flow.request.method} {flow.request.pretty_url}")

def response(flow):
    ctx.log.info(f"Response: {flow.response.status_code}")
