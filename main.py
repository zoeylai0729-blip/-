# -*- coding: utf-8 -*-
import json
import os
import re
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx

# Only load .env in local environment
if os.getenv("VERCEL") is None:
    load_dotenv(override=True)

app = FastAPI(title="副业天赋分析 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    bazi_string: str
    birth_date: str
    gender: str
    birth_time: Optional[str] = None

SYSTEM_PROMPT = """
# Role
你是一位精通中国传统子平八字命理，同时深谙人性与商业逻辑的【实战派财富参谋】。你的任务是根据用户的八字信息，为其提供一份逻辑严密，同时年轻人听得懂，并具有实操意义的《搞钱天赋与副业指南》。

# Tone
- 专业严谨：避免使用「发大财」、「走大运」等江湖气词汇。
- 现代感：将术语转化为商业术语。
- 启发性：不做绝对化的预判，基于命理逻辑提供深度观察。

# Analysis Logic
1. 确定核心动能（十神定性）：食伤、财星、官杀、印星、比劫。
2. 五行映射行业。
3. 2026 流年干扰。

# Output Format
你必须返回纯 JSON，不要包含任何 Markdown 代码块。
{
  "identity_label": "标签",
  "identity_desc": "描述",
  "tags": ["标签1"],
  "wealth_logic": ["逻辑"],
  "plan_a": {"title": "标题", "desc": "描述", "reason": "理由"},
  "plan_b": {"title": "标题", "desc": "描述", "reason": "理由"},
  "warning_reminder": "提醒",
  "warning_advice": "建议"
}
"""

def build_user_message(req: AnalyzeRequest) -> str:
    gender_cn = "男" if req.gender == "male" else "女"
    time_str = req.birth_time if req.birth_time else "不详"
    return f"公历生日：{req.birth_date}\n性别：{gender_cn}\n出生时间：{time_str}\n八字：{req.bazi_string}"

def extract_json_from_content(content: str) -> dict:
    text = content.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)

async def call_llm(user_message: str) -> dict:
    api_key = os.getenv("API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    if not api_key:
        raise HTTPException(status_code=500, detail="Missing API_KEY in environment variables")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.6,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    if base_url and base_url.strip():
        base_url = base_url.strip().rstrip("/")
        url = base_url if "chat/completions" in base_url else f"{base_url}/v1/chat/completions"
    else:
        url = "https://api.openai.com/v1/chat/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    return extract_json_from_content(content)

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        user_message = build_user_message(req)
        result = await call_llm(user_message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/debug/env")
async def debug_env():
    key_exists = (os.getenv("API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")) is not None
    return {
        "API_KEY_SET": key_exists,
        "API_BASE_URL": os.getenv("API_BASE_URL"),
        "VERCEL_ENV": os.getenv("VERCEL")
    }

INDEX_HTML = Path(__file__).resolve().parent / "index.html"

@app.get("/")
async def index():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return JSONResponse(status_code=404, content={"detail": "index.html not found"})

@app.api_route("/{path_name:path}", methods=["GET"])
async def catch_all(request: Request, path_name: str):
    if path_name.strip("/") == "":
        return await index()
    return JSONResponse(status_code=404, content={"detail": f"Not Found: {path_name}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
