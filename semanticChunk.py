# 语义分块
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config
import util


# 1. 先考虑加载嵌入模型
# 2. 如何把文本转换成向量
# 3. 计算向量的相似度
# 4. 打印中间的结果

def test_embedding_similarity(
        sentence1: str,
        sentence2: str,
) -> float:

    # 1. 获取嵌入模型对象
    embed_model = HuggingFaceEmbedding(model_name=config.model_path)
    # 2. 如何把文本转换成向量
    vec1 = embed_model.get_text_embedding(sentence1)
    vec2 = embed_model.get_text_embedding(sentence2)

    # 3. 计算向量的相似度
    similarity =embed_model.similarity(vec1, vec2)

    # 4. 打印中间的结果
    print(f"\n句子1：{sentence1}")
    print(f"\n向量1：（维度 {len(vec1)}）: {vec1}")
    print(f"\n句子2：{sentence2}")
    print(f"\n向量2：（维度 {len(vec2)}）: {vec2}")
    print(f"\n句子相似度：{similarity:.4f}")

    return similarity



# 1. 要能够直接操作一个目录
# 2. 接收的是Document列表，返回的是Document列表
def semantic_chunk_documents(
        input_dir : str,
) -> list[Document]:
    # 1. 获取清洗之后的Document列表
    documents = util.clean_all_formats(input_dir=input_dir)
    # 2. 声明出来一个空列表
    chunks = []
    # 3. 加载嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=config.model_path)
    # 4. 语义分割节点解析器对象
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=10,
        breakpoint_percentile_threshold=98,
    )
    # 5. 执行语义分块操作
    nodes = splitter.get_nodes_from_documents(
        documents= documents
    )

    for i, node in enumerate(nodes):
        print(f"第{i+1}个分块")
        print(node.metadata)
        print(node.text)

    # 6. 将得到的结果返回给前面声明好的空列表里面
    for node in nodes:
        chunks.append(Document(text=node.text, metadata=node.metadata))

    return chunks




if __name__ == "__main__":
    # test_embedding_similarity("小明是个大帅哥", "小明长得很帅")
    # test_embedding_similarity("小明长得很帅", "小明长得很丑")
    semantic_chunk_documents(config.document_path)