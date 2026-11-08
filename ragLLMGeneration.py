# 数据集->数据加载->文本分块（把前面讲到的分块方法都使用一下）->
# 嵌入模型->向量入库->数据检索->提示词增强->LLM生成
from typing import List

from llama_index.core import Document

import config
import fixedSizeChunk
import llmChunk
import ragPromptEnhancer
import recursiveChunk
import semanticChunk
import sentenceChunk
import structureChunk
import util

if __name__ == "__main__":
    # 1. 生成数据库
    ragPromptEnhancer.init()
    # 2. 数据加载
    chunks: List[Document] = []

    # chunks = fixedSizeChunk.fixed_size_chunk_documents(config.structure_path) # 固定分块
    # chunks = sentenceChunk.sentence_chunk_documents(config.structure_path) # 句子分块
    # chunks = semanticChunk.semantic_chunk_documents(config.structure_path) # 语义分块
    # chunks = recursiveChunk.recursive_chunk_documents(config.structure_path) # 递归分块
    # chunks = llmChunk.llm_chunk_documents(config.structure_path) # LLM分块
    chunks = structureChunk.structure_chunk_documents(config.structure_path) #基于文档结构的分块
    # 3. 数据入库
    ragPromptEnhancer.index_to_milvus_by_chunks(chunks)
    # 4. 提示词增强
    while True:
        question = input("请输入用户查询的问题： ").strip()
        if question.lower() in ("q"):
            print("已经退出循环")
            break;
        if not question:
            print("question无效")
            continue
        prompt = ragPromptEnhancer.get_prompt(question)
        print(prompt)
        print("大模型生成的内容：\n")
        for text in util.stream_complete(prompt):
            print(text, end="", flush=True)
        print("\n")