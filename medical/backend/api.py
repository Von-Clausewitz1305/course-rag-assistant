import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def ensure_embedding_model() -> None:
    """本地没有 bge-m3 时再下载，避免 milvus 在导入时找不到模型。"""
    import config

    model_dir = Path(config.model_path)
    if (model_dir / "config.json").is_file():
        return

    print(f"正在下载嵌入模型到 {model_dir} ...")
    model_dir.mkdir(parents=True, exist_ok=True)
    if "HF_ENDPOINT" not in os.environ and not _huggingface_reachable():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("无法直接连接 Hugging Face，改用镜像 https://hf-mirror.com")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=str(model_dir),
        ignore_patterns=["onnx/*", "*.onnx", "imgs/*"],
    )


def _huggingface_reachable() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        return True
    except Exception:
        return False


ensure_embedding_model()

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from medical.backend.main import (
    DeepSeekError,
    ensure_medical_index,
    index_medical_to_milvus,
    query_medical,
)

@asynccontextmanager
async def _lifespan(_app):
    try:
        count = ensure_medical_index()
        print(f"医疗索引就绪，当前 {count} 条")
    except Exception as e:
        print(f"启动时构建索引失败: {e}")
    yield


# 创建FastAPI实例
app = FastAPI(lifespan=_lifespan)

# 处理跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源的跨域请求
    allow_methods=["*"], # 允许所有的http方法
    allow_headers=["*"], # 允许所有的请求头
)


# 重建向量索引的接口
@app.post("/api/rebuild-index")
def rebuild_index() -> dict:
    try:
        count = index_medical_to_milvus()
        return {"ok": True, "message": f"索引构建完成，写入 milvus {count} 条数据"}
    except Exception as e:
        print(f"构建索引失败 {e}")
        raise HTTPException(status_code=500, detail="索引构建失败，请重试")


# /api/ask接口的请求实体
class AskRequest(BaseModel):
    question: str
    api_key: str
    model: str = "deepseek-chat"

# 接收用户提问并回答
@app.post("/api/ask")
def ask_question(payload: AskRequest) -> dict:
    q = payload.question.strip()
    api_key = payload.api_key.strip()
    model = (payload.model or "deepseek-chat").strip()
    if not q:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not api_key:
        raise HTTPException(status_code=400, detail="请填写 DeepSeek API Key")
    if model not in {"deepseek-chat", "deepseek-reasoner"}:
        raise HTTPException(status_code=400, detail="不支持的模型")
    try:
        return query_medical(q, api_key=api_key, model=model)
    except DeepSeekError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"查询失败 {type(e).__name__}")
        raise HTTPException(status_code=500, detail="请检查一下向量数据库是否构建成功")



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)
