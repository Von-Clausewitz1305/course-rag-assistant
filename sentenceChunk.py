# 句子分块
import re

from llama_index.core import Document

import config
import util


# 1. 要能够直接操作一个目录
# 2. 一个关键参数，多少个句子可以合并成一个分块
# 3. 接收的是Document列表，返回的是Document列表

# 实现按照句子分块的功能
def sentence_chunk_documents(
        input_dir: str,
        max_sentences: int=5,
) -> list[Document]:
    # 1. 准备一个可以接收结果的列表
    chunks = []
    # 2. 清洗数据，获取清洗后的Document列表
    documents = util.clean_all_formats(input_dir=input_dir)
    # 3. 遍历每一个Document对象，进行句子分块的操作
    for d in documents:
        # 3.1 text文本信息
        text = d.text
        path = d.metadata.get("file_path")

        # 3.2 正则表达式来处理句子分块
        sents = [
           s.strip()
           for s in re.split(r"(?<=[。！？！？\.])\s+", text)
           if s.strip()
        ]

        # 3.3 固定个数的句子拼装成一个分块
        for i in range(0, len(sents), max_sentences):

            part = " ".join(sents[i: i+max_sentences]).strip()
            if not part:
                continue


            m = dict(d.metadata)
            m.update({
                "source_file_path": path,
                "chunk_index": i,
                "chunk_start":i,
                "chunk_end": min(i+max_sentences, len(sents))
            })

            chunks.append(Document(text=part, metadata=m))


    # 4. 进行简单打印，预览效果
    # for i, chunk in enumerate(chunks):
    #     print(f"第{i+1}个分块")
    #     print(chunk.metadata)
    #     print(chunk.text)


    # 5. 返回Document列表
    return chunks

if __name__ == "__main__":
    sentence_chunk_documents(config.base_path / "文档")
