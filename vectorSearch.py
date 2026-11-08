# 向量检索
import shutil
import time

import config
import milvus
from sentenceChunk import sentence_chunk_documents

# 1. 构建向量数据库
DB_NAME = "vectorSearch"
COLLECTION = "vectorSearch"
DIMENSION = 1024

def init():
    # 1. 判断db是否存在
    db_path = config.base_path / f"{DB_NAME}.db"
    if db_path.exists():
        shutil.rmtree(db_path)
        time.sleep(0.5)

    # 2. 创建db
    milvus.create_db(DB_NAME)
    milvus.create_collection(COLLECTION, DIMENSION)


# 2. 文本分块，分块内容逐步向量化入库（文本分块的方案->句子分块）
def index_to_milvus() -> int:
    chunks = sentence_chunk_documents(config.structure_path, 1)
    if not chunks:
        raise ValueError("分块结果为空，请检测数据集内容")
    rows = []

    # 把分好块的内容向量化，拼装成要入库的数据
    for i, chunk in enumerate(chunks):
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append(
            {
                "id": i+1,
                "vector": vec,
                "text": chunk.text
            }
        )
    milvus.insert(COLLECTION, rows)
    print(f"已经完成入库,{len(rows)}个记录, 向量维度{DIMENSION}")
    return len(rows)

# 3. 文本检索
def search_similar(query:str, limit: int =2):
    milvus.get_client().load_collection(collection_name=COLLECTION)
    results = milvus.search_by_text(
        COLLECTION,
        text=query,
        limit=limit,
        fields=["id", "text"],
    )

    hits = results[0] if results else []
    if not hits:
        print("未检索到相关结果")
        return
    print(f"\n用户查询:{query}")
    print(f"共{len(hits)} 条相似的结果")

    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity")
        text = entity.get("text")
        print(f"\n[{rank}] 距离 distance={hit.get('distance'):.4f}")
        print(f"id={entity.get('id', hit.get('id'))}")
        print(text)

# 4. 循环输入用户查询，来获取知识库内容
def search():
    print("\n向量检索就绪，请输入用户查询问题")
    while True:
        query = input("请输入查询问题：").strip()
        if query.lower() in ("q"):
            print("已经退出!")
            break
        if not  query:
            continue
        search_similar(query)

if __name__ == "__main__":
    init()
    index_to_milvus()
    search()

