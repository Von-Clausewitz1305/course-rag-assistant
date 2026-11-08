from llama_index.core.base.embeddings.base import SimilarityMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config

# 计算两段文本的相似度
def embedding_similarity(
        sentence1: str,
        sentence2: str,
        mode: SimilarityMode=SimilarityMode.DEFAULT,
) -> float:

    # 1. 模型加载
    embed_model = HuggingFaceEmbedding(model_name=config.model_path)

    # 2. 计算文本向量
    vec1 = embed_model.get_text_embedding(sentence1)
    vec2 = embed_model.get_text_embedding(sentence2)

    # 3. 计算相似度
    similarity = embed_model.similarity(vec1, vec2, mode=mode)

    return  similarity

if __name__ == "__main__":
    sentence1 = "我是一个小比特，今年上大一了"
    sentence2 = "陕西省的省会城市是西安"
    print(embedding_similarity(sentence1, sentence2))
    # print(embedding_similarity(sentence1, sentence2, SimilarityMode.EUCLIDEAN))
    print(embedding_similarity(sentence1, sentence2, SimilarityMode.DOT_PRODUCT))