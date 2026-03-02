# -*- coding: utf-8 -*-
"""
副业天赋测算 - 后端服务
提供 /analyze 接口，根据八字等信息调用大模型生成分析报告。
"""
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
if not os.getenv("VERCEL"):
    load_dotenv(override=True)

app = FastAPI(title="副业天赋分析 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 请求/响应模型 ----------
class AnalyzeRequest(BaseModel):
    bazi_string: str  # 八字字符串，如 "庚辰年 辛巳月 ..."
    birth_date: str   # 公历生日，如 "2000.5.15"
    gender: str       # "male" | "female"
    birth_time: Optional[str] = None  # 出生时辰描述，如 "午时" 或 "11:00-13:00"

SYSTEM_PROMPT = """
# Role
你是一位精通中国传统子平八字命理，同时深谙人性与商业逻辑的【实战派财富参谋】。你的任务是根据用户的八字信息，为其提供一份逻辑严密，同时年轻人听得懂，并具有实操意义的《搞钱天赋与副业指南》。

# Tone
- 专业严谨：避免使用「发大财」、「走大运」等江湖气词汇。
- 拒绝爹味/班味：不要说“提升管理能力、赋能行业”这种黑话，不要说sop、mvp等互联网用语。要说“把废话整理成文档卖钱”。
- 现代感：将「食神、官杀、比劫」等术语转化为「内容创造力、管理执行力、社交杠杆」等商业术语。
- 启发性：不做绝对化的预判，而是基于命理逻辑提供深度观察。

# Analysis Logic
1. 确定核心动能（十神定性）：
  1. 食伤旺 -> 靠创意、表达、技能变现（创作者经济）。
  2. 财星旺 -> 靠贸易、周转、资源整合（小商业经营，如闲鱼倒货（信息差）、羊毛党）。
  3. 官杀旺 -> 靠体系、规则、秩序（整理师/把关人，如社群纪律管理员、简历修改/面试模拟、资料校对）。
  4. 印星旺 -> 靠知识、输入、滋养（深度内容/咨询，如干货资料打包（卖信息差）、情感咨询/树洞、塔罗占卜）。
  5. 比劫旺 -> 靠团队、社群、体力/耐力（渠道营销，如私域社群运营、同城跑腿/陪诊、地推、需要体力的副业。）。
2. 五行映射行业：结合喜用五行匹配 2026 年的热门赛道。
3. 2026 流年干扰（丙午年）：重点分析「火」元素对用户日主的影响，给出当下的行动基调。

# Constraints (必须遵守的红线)
1. 禁止高门槛：严禁推荐律师、企业战略咨询、风投、全案策划等需要 3 年以上经验的职业。
2. 聚焦轻副业：推荐方向尽量是低成本、低代码、甚至一部手机就能开始的（如：资料打包、PPT 美化、情绪树洞、咸鱼倒货）。

# Output Format
你必须返回纯 JSON，不要包含任何 Markdown 代码块（如 ```json）或其它说明文字。
JSON 结构必须严格如下（字段名与类型不可改）：
{
  "identity_label": "核心搞钱人格的简短标签，如：知识二道贩子、天生销冠体质",
  "identity_desc": "英文或中英混合的副标题，如：The Lord of Copy-Paste",
  "tags": ["标签1", "标签2"],
  "wealth_logic": ["天赋：...", "定位：...", "策略：..."],
  "plan_a": {"title": "推荐方向A标题", "desc": "具体描述（做什么、怎么做，1-2句）", "reason": "命理理由（为什么适合你，1句）"},
  "plan_b": {"title": "推荐方向B标题", "desc": "具体描述（做什么、怎么做，1-2句）", "reason": "命理理由（为什么适合你，1句）"},
  "warning_reminder": "关键提醒（今年最该注意的事，1句）",
  "warning_advice": "行动建议（该进还是该退，1句）"
}
"""

def build_user_message(req: AnalyzeRequest) -> str:
    gender_cn = "男" if req.gender == "male" else "女"
    time_str = req.birth_time if req.birth_time else "不详（按平旦论）"
    return f"""
公历生日：{req.birth_date}
性别：{gender_cn}
出生时间：{time_str}
八字：{req.bazi_string}

请根据以上信息，严格按照 Output Format 中的 JSON 结构返回一份搞钱天赋分析报告。只输出一个 JSON 对象，不要其它内容。
""".strip()

def extract_json_from_content(content: str) -> dict:
    """从模型返回中提取纯 JSON（去除 markdown 代码块等）."""
    text = content.strip()
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)

async def call_llm(user_message: str) -> dict:
    """调用 OpenAI 兼容接口（如 DeepSeek / GPT-4）."""
    api_key = os.getenv("API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="未配置 API_KEY，请在 Vercel Settings 中设置 API_KEY",
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.6,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # 若已配置完整 chat completions 地址（如 OpenAI / DeepSeek），直接使用；否则拼接
    if base_url and base_url.strip():
        base_url = base_url.strip().rstrip("/")
        if "chat/completions" in base_url:
            url = base_url
        else:
            url = f"{base_url}/v1/chat/completions"
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
    """接收八字、生日、性别等，调用大模型返回分析结果 JSON."""
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
    """返回前端页面，便于直接打开 http://127.0.0.1:8000 使用"""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return JSONResponse(status_code=404, content={"detail": "index.html not found"})

@app.api_route("/{path_name:path}", methods=["GET"])
async def catch_all(request: Request, path_name: str):
    if not path_name or path_name.strip("/") == "":
        return await index()
    return JSONResponse(status_code=404, content={"detail": f"Path Not Found: {path_name}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

