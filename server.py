import os
import socket
from io import BytesIO
from typing import Optional

import qrcode
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from core.agent import DateAgent, AgentConfig

# 加载环境变量
load_dotenv()

# =========================
# 配置
# =========================

# Hugging Face Spaces 默认使用 7860 端口
APP_PORT = int(os.getenv("PORT", "7860"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
# Hugging Face 提供的公网域名环境变量
SPACE_HOST = os.getenv("SPACE_HOST") 

# LLM 配置 (优先环境变量)
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL", "deepseek-chat")
KB_PATH = os.getenv("KB_PATH", "data/knowledge_base.md")

templates = Jinja2Templates(directory="templates")
app = FastAPI(title="📅 年月日知识智能专家")

app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化 Agent
agent_config = AgentConfig(
    kb_path=KB_PATH,
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL
)
agent = DateAgent(agent_config)

class ChatRequest(BaseModel):
    message: str
    mode: str = "小学生"  # 小学生 or 家长

# =========================
# 工具函数
# =========================

def get_local_ip() -> str:
    forced = os.getenv("LOCAL_IP")
    if forced: return forced
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def public_base_url() -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.rstrip("/")
    if SPACE_HOST:
        return f"https://{SPACE_HOST}"
    ip = get_local_ip()
    return f"http://{ip}:{APP_PORT}"

# =========================
# 路由
# =========================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/info")
async def api_info():
    return JSONResponse({
        "public_base_url": public_base_url(),
        "qr_url": "/qr.png",
        "has_api_key": bool(API_KEY)
    })

@app.get("/qr.png")
async def qr_png():
    url = public_base_url() + "/"
    qr = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        # 获取流式回答并拼接（FastAPI 简单处理为非流式返回，前端可以根据需要改为流式）
        full_answer = ""
        for chunk in agent.ask(message, role=req.mode):
            full_answer += chunk
        
        return JSONResponse({"answer": full_answer.strip()})
    except Exception as e:
        return JSONResponse({"answer": f"抱歉，处理您的请求时出错了：{str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
