import json
import math
import os
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymilvus import MilvusClient
import config
from typing import List, Dict, Any, Optional

client = None


class LocalVectorClient:
    """Windows 上没有 milvus-lite，用本地 JSON 保存向量，接口与 MilvusClient 对齐。"""

    def __init__(self, dbpath: str):
        self.path = Path(str(dbpath) + ".json")
        self.collections: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            self.collections = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.collections, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def drop_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)
        self._save()

    def create_collection(self, collection_name: str, dimension: int) -> None:
        self.collections[collection_name] = {"dimension": dimension, "rows": []}
        self._save()

    def load_collection(self, collection_name: str) -> None:
        return None

    def release_collection(self, collection_name: str) -> None:
        return None

    def get_collection_stats(self, collection_name: str) -> Dict[str, int]:
        rows = self.collections.get(collection_name, {}).get("rows", [])
        return {"row_count": len(rows)}

    def insert(self, collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, int]:
        bucket = self.collections[collection_name]["rows"]
        for row in data:
            stored = dict(row)
            vector = stored.get("vector")
            if vector is not None and not isinstance(vector, list):
                stored["vector"] = list(vector)
            bucket.append(stored)
        self._save()
        return {"insert_count": len(data)}

    def delete(self, collection_name: str, filter: str = "", ids: Optional[List[int]] = None) -> None:
        rows = self.collections[collection_name]["rows"]
        if ids is not None:
            id_set = set(ids)
            self.collections[collection_name]["rows"] = [row for row in rows if row.get("id") not in id_set]
        elif filter.strip().startswith("id>="):
            minimum = int(filter.split(">=")[1])
            self.collections[collection_name]["rows"] = [row for row in rows if int(row.get("id", -1)) < minimum]
        else:
            raise ValueError(f"不支持的删除条件: {filter}")
        self._save()

    def query(self, collection_name: str, filter: str = "", output_fields: Optional[List[str]] = None, limit: int = 10) -> List[Dict]:
        rows = self.collections[collection_name]["rows"]
        if filter.strip().startswith("id>="):
            minimum = int(filter.split(">=")[1])
            rows = [row for row in rows if int(row.get("id", -1)) >= minimum]
        picked = rows[:limit]
        if not output_fields:
            return [dict(row) for row in picked]
        return [{field: row.get(field) for field in output_fields} for row in picked]

    def get(self, collection_name: str, ids: List[int], output_fields: Optional[List[str]] = None) -> List[Dict]:
        id_set = set(ids)
        rows = [row for row in self.collections[collection_name]["rows"] if row.get("id") in id_set]
        if not output_fields:
            return [dict(row) for row in rows]
        return [{field: row.get(field) for field in output_fields} for row in rows]

    def search(self, collection_name: str, data: List[List[float]], limit: int = 5, output_fields: Optional[List[str]] = None) -> List[List[dict]]:
        rows = self.collections[collection_name]["rows"]
        fields = output_fields or []
        grouped = []
        for query in data:
            hits = []
            for row in rows:
                entity = {field: row.get(field) for field in fields}
                hits.append({
                    "id": row.get("id"),
                    "distance": _cosine_distance(query, row.get("vector") or []),
                    "entity": entity,
                })
            hits.sort(key=lambda item: item["distance"])
            grouped.append(hits[:limit])
        return grouped


def _cosine_distance(left: List[float], right: List[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x, y in zip(left, right):
        dot += x * y
        left_norm += x * x
        right_norm += y * y
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot / math.sqrt(left_norm * right_norm)
    return 1.0 - similarity


def _open_client(dbpath: str):
    path = str(dbpath)
    if os.name == "nt":
        return LocalVectorClient(path)
    try:
        return MilvusClient(path)
    except Exception as exc:
        if "milvus-lite" not in str(exc) and "milvus_lite" not in str(exc):
            raise
        return LocalVectorClient(path)

# 1. 连接milvus
def connect(dbpath : str):
    global client
    client = _open_client(dbpath)
    return client

# 2. 获取client
def get_client():
    return client

# 3. 创建db
def create_db(dbname : str) -> MilvusClient:
    return connect(config.base_path / f"{dbname}.db")

# 4. 切换db  use db
def use_db(dbname : str) -> MilvusClient:
    return connect(config.base_path / f"{dbname}.db")

# 5. 列出所有的db
def list_dbs() -> List[str]:
    return [p.stem for p in config.base_path.glob("*.db")]

# 6. 创建collection
def create_collection(name: str, dimension: int) -> None:
    client = get_client()
    if client.has_collection(name):
        client.drop_collection(collection_name=name)
    client.create_collection(collection_name=name, dimension=dimension)

# 7. 删除collection
def drop_collection(name:str) -> None:
    get_client().drop_collection(collection_name=name)

# 8. 插入数据
def insert(collection_name : str, rows: List[Dict[str, Any]]) -> Dict:
    return get_client().insert(collection_name=collection_name, data=rows)

# 9. 查询数据
def get_by_ids(collection_name: str, ids: List[int], fields: Optional[List[str]] = None) -> List[Dict]:
    return get_client().get(collection_name=collection_name, ids=ids, output_fields=fields)

# 声明向量模型
embed_model = HuggingFaceEmbedding(model_name=config.model_path)

# 10. 向量查询
def search_by_text(
        collection_name:str,
        limit: int=5,
        text: str="",
        fields: Optional[List[str]] = None,
) -> List[List[dict]]:
    vectors = [embed_model.get_text_embedding(text)]
    return get_client().search(
        collection_name=collection_name,
        data=vectors,
        limit=limit,
        output_fields=fields,
    )


if __name__ == "__main__":
    use_db("bit")
    # create_collection("01", 1024)
    # embed_model = HuggingFaceEmbedding(model_name=config.model_path)
    # text = "我是一个小比特，我今年上大一了！"
    # vec = embed_model.get_text_embedding(text)
    #
    # print(insert("01", [{"id": 1, "vector": vec, "text":text}]))
    # print(insert("01", [{"id": 2, "vector": vec, "text":text}]))
    # print(insert("01", [{"id": 3, "vector": vec, "text":text}]))
    # get_client().load_collection(collection_name="01")
    # print(get_by_ids(collection_name="01", ids=[1], fields=["id", "vector", "text"]))
    # datas = ["小明是一个大帅哥", "小明长得很好看", "今天中午吃点什么呢？", "我们正在学习RAG的课程"]
    # i=4
    # for data in datas:
    #     vector = embed_model.get_text_embedding(data)
    #     insert("01", [{"id":i, "vector": vector, "text": data}])
    #     i = i+1
    get_client().load_collection(collection_name="01")
    text = "有很多女孩喜欢小明"
    results = search_by_text("01", text=text, limit=2)
    print(results)