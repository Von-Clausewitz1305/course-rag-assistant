import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any

import config
import milvus
import ragPromptEnhancer
from ragPromptEnhancer import DB_NAME, COLLECTION, DIMENSION

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ALLOWED_MODELS = {"deepseek-chat", "deepseek-reasoner"}


class DeepSeekError(Exception):
    """页面可直接展示的 DeepSeek 调用错误，不包含密钥。"""

COURSE_PROMPT_TEMPLATE = """【角色】你是课程助教，只根据课程资料回答问题。
【任务】根据下面的课程片段回答用户的问题，禁止编造资料里没有的内容
【课程资料】
{context}
【用户的问题】
{question}
【输出要求】
1. 准确、简洁，分条回答；
2. 资料里没有相关内容时，直接说“课程资料中不存在”；
3. 不要用资料以外的知识回答。
"""

EMBED_BATCH_SIZE=32
INSERT_BATCH_SIZE=200
_db_lock = threading.Lock()

# 1. 读取课程 jsonl，每行是一个课件片段
def load_course_passages(data_dir: str) -> List[Dict[str, str]]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"当前目录 {data_dir} 里面没有 jsonl 文件")

    passages: List[Dict[str, str]] = []
    for path in files:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            passages.append(
                {
                    "doc_id": str(row.get("doc_id") or "").strip(),
                    "chapter": str(row.get("chapter") or "").strip(),
                    "title": str(row.get("title") or "").strip(),
                    "text": text,
                }
            )

    if not passages:
        raise ValueError(f"目录 {data_dir} 里没有有效的课程片段")
    return passages

# 2.1 在进程内只创建一次milvus连接
def _ensure_milvus_connect() -> None:
    if milvus.get_client() is not None:
        return
    milvus.use_db(DB_NAME)

# 2.2 卸载collection的内存索引，还要保持milvus的连接不断开
def _release_collection_only() -> None:
    client = milvus.get_client()
    if client is None:
        return
    try:
        client.release_collection(collection_name=COLLECTION)
    except Exception:
        pass

# 2.3 删除collection中的全部数据，为重建索引提供空间
def _delete_all_entities() -> None:
    client = milvus.get_client()
    try:
        client.delete(collection_name=COLLECTION, filter="id>=0")
        return
    except Exception:
        pass
    # 一次性删除异常，可以考虑分批删除
    batch_size = 1000
    while True:
        rows = client.query(
            collection_name=COLLECTION,
            filter="id>=0",
            output_fields=["id"],
            limit=batch_size,
        )
        if not rows:
            break
        client.delete(collection_name=COLLECTION, ids=[row["id"] for row in rows])
        if len(rows) < batch_size:
            break

# 2.4 连接milvus数据库，把collection加载到内容，供检索使用
def connect_db() -> None:
    _ensure_milvus_connect()
    client = milvus.get_client()
    if client.has_collection(COLLECTION):
        client.load_collection(collection_name=COLLECTION)


# 2. 创建向量数据库
def init_medical_db() -> None:
    # 1. 获取milvus的客户端
    _ensure_milvus_connect()
    client = milvus.get_client()
    # 2. 首次建库
    if not client.has_collection(COLLECTION):
        client.create_collection(collection_name=COLLECTION, dimension=DIMENSION)
        return
    # 3. 重建索引的时候，先把内存中的集合卸载
    _release_collection_only()
    # 4. 加载集合
    try:
        client.load_collection(collection_name=COLLECTION)
    except Exception:
        pass
    # 5. 清空旧的向量数据
    _delete_all_entities()
    _release_collection_only()

# 3. 数据向量化
def build_milvus_rows(
        passages: List[Dict[str, str]],
        embed_bath_size: int=EMBED_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    embed_texts = [
        "\n".join(part for part in (item["chapter"], item["title"], item["text"]) if part)
        for item in passages
    ]
    vectors: List[List[float]] = []

    for start in range(0, len(embed_texts), embed_bath_size):
        batch_texts = embed_texts[start: start+embed_bath_size]
        batch_vectors = milvus.embed_model.get_text_embedding_batch(texts=batch_texts)
        vectors.extend(batch_vectors)
        print(f"已经执行完向量化 {min(start + embed_bath_size, len(embed_texts))} / {len(embed_texts)}条")

    if len(vectors) != len(passages):
        raise ValueError(
            f"向量数 {len(vectors)} 与课程片段数 {len(passages)} 数目不一致"
        )

    rows: List[Dict[str, Any]] = []
    for i, (item, vector, embed_text) in enumerate(zip(passages, vectors, embed_texts), start=1):
        rows.append(
            {
                "id": i,
                "vector": vector,
                "text": embed_text,
                "doc_id": item["doc_id"],
                "chapter": item["chapter"],
                "title": item["title"],
            }
        )
    print(f"构造待插入数据 {len(rows)}条")
    return rows

# 4. 向量化的数据插入向量数据库
def insert_data(
        rows: List[Dict[str, Any]],
        batch_size: int = INSERT_BATCH_SIZE,
) -> int:
    total = len(rows)
    inserted = 0

    for batch_idx, start in enumerate(range(0, total, batch_size)) :
        batch = rows[start: start + batch_size]
        milvus.insert(COLLECTION, batch)
        inserted += len(batch)
        print(f"已经入库 {inserted}/{total} 条")
        if start + batch_size < total:
            time.sleep(0.5)

    print(f"已经完成入库：{inserted} 条， 向量维度 {DIMENSION}")
    return inserted

# 5. 数据检索
def search_medical(question: str, top_k: int = 5) -> List[dict]:
    # 1. 连接数据库
    connect_db()
    # 2. 查询操作
    results = milvus.search_by_text(
        collection_name=COLLECTION,
        text=question,
        limit=top_k,
        fields=["id", "text", "doc_id", "chapter", "title"],
    )
    # 3. 结果返回
    return results[0] if results else []


# 6. 提示词增强
def get_prompt(
        question:str,
        top_k:int=5,
        max_distance:float=0.6,
        max_chars:int=1800,
) -> str:
    connect_db()
    hits = ragPromptEnhancer.retrieve(question, top_k=top_k)
    texts = ragPromptEnhancer.filter_chunks(hits, max_distance=max_distance)
    context = ragPromptEnhancer.compress_context(texts, max_chars=max_chars)
    if not context:
        context = "(无相关的课程片段)"
    return COURSE_PROMPT_TEMPLATE.format(context=context, question=question)

# 7. LLM生成（DeepSeek，密钥只在本次请求中使用）
def get_answer(
        prompt: str,
        api_key: str,
        model: str = "deepseek-chat",
) -> str:
    from openai import APIConnectionError, APIStatusError, OpenAI

    if model not in ALLOWED_MODELS:
        raise DeepSeekError("不支持的模型，请选择 deepseek-chat 或 deepseek-reasoner")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except APIStatusError as e:
        if e.status_code in (401, 403):
            raise DeepSeekError("DeepSeek API Key 无效") from None
        raise DeepSeekError("DeepSeek 调用失败") from None
    except APIConnectionError:
        raise DeepSeekError("连接 DeepSeek 失败，请检查网络") from None
    except Exception:
        raise DeepSeekError("DeepSeek 调用失败") from None

    content = resp.choices[0].message.content if resp.choices else ""
    return (content or "").strip()

# 8. 提供可被api调用的方法 -> 重建索引
def index_medical_to_milvus() -> int:
    with _db_lock:
        _release_collection_only()
        init_medical_db()
        passages = load_course_passages(config.course_data_path)
        rows = build_milvus_rows(passages)
        return insert_data(rows)

def _collection_row_count() -> int:
    _ensure_milvus_connect()
    client = milvus.get_client()
    if not client.has_collection(COLLECTION):
        return 0
    stats = client.get_collection_stats(collection_name=COLLECTION)
    return int(stats.get("row_count", 0))


# 启动时若条数和课程 jsonl 不一致，就重建索引
def ensure_medical_index() -> int:
    expected = len(load_course_passages(config.course_data_path))
    try:
        count = _collection_row_count()
    except Exception as e:
        print(f"读取索引状态失败，将重新构建: {e}")
        count = 0
    if count == expected and count > 0:
        connect_db()
        return count
    return index_medical_to_milvus()


# 9. 提供可被api调用的方法   -> 用户提问
def query_medical(question: str, api_key: str, model: str = "deepseek-chat") -> dict:
    with _db_lock:
        hits = search_medical(question=question)
        rank = 0
        sources =[]
        for hit in hits:
            text = hit.get("entity").get("text")
            rank +=1
            entity = hit.get("entity")
            sources.append(
                {
                    "rank":rank,
                    "score": round(1.0 - hit.get("distance", 1.0), 4),
                    "text":text,
                    "metadata":{
                        "doc_id": entity.get("doc_id"),
                        "chapter": entity.get("chapter"),
                        "title": entity.get("title"),
                        "id": entity.get("id"),
                    }
                }
            )
        prompt = get_prompt(question)
        _release_collection_only()
    answer = get_answer(prompt, api_key=api_key, model=model)
    if answer.strip().rstrip("。") in {"文档中不存在", "课程资料中不存在"}:
        return {"answer":answer, "sources":""}
    return {"answer":answer, "sources":sources}


if __name__ == "__main__":
    # results = search_medical(question="我有点头疼，请问该怎么办？")
    # for result in results:
    #     print(result)

    prompt = get_prompt(question="我有点头疼，请问该怎么办？")
    print(prompt)
    print("请在页面上填写 DeepSeek API Key 后再提问。")