#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting WebSocket Proxy...${NC}"

# 确保在tools目录下
cd "$(dirname "$0")"

# 停止已经运行的mitmproxy进程
echo -e "${BLUE}Stopping any running mitmproxy processes...${NC}"
pkill -f "mitm" || true

# 等待端口释放
sleep 1

# 启动mitmproxy
echo -e "${BLUE}Starting mitmproxy with WebSocket logging...${NC}"
echo -e "${GREEN}Proxy will be available at ws://localhost:27182${NC}"
echo -e "${GREEN}Web interface will be available at http://127.0.0.1:8081${NC}"

# 使用虚拟环境中的mitmweb
/root/xiaozhi-esp32-server/.venv/bin/mitmweb \
    -s ws_logger.py \
    --mode reverse:https://api.tenclass.net \
    --listen-port 27182 \
    --ssl-insecure \
    --no-web-open-browser
