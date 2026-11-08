# 固定大小分块
from llama_index.core import Document

import config
import util


# 1. 要能够直接操作一个目录
# 2. 两个关键参数，一个是块大小，一个是重叠区的大小
# 3. 接收的是Document列表，返回的是Document列表

def fixed_size_chunk_documents(
        input_dir: str,
        chunk_size: int =500,
        chunk_overlap: int=50,
)  -> list[Document]:
    # 1. 滑动步长，准备一个可以接收结果的列表
    step = chunk_size - chunk_overlap
    chunks = []
    # 2. 调用清洗方法，拿到目录下面所有清洗后的Document列表
    documents = util.clean_all_formats(input_dir)
    document_index = 1
    # 3. 遍历每一个Document对象，进行分块操作
    for d in documents:
        text = d.text
        path = d.metadata.get("file_path")
        start = 0
    # 4. 利用while循环进行滑动窗口分块
        while start < len(text):
            end = min (start + chunk_size, len(text))
            # 考虑预览效果
            m = dict(d.metadata)
            m.update({
                "path": path,
                "chunk_start": start,
                "chunk_end": end,
                "document_index": document_index,
            })
            chunks.append(Document(text=text[start:end], metadata=m))

            if end >= len(text):
                break

            start = start + step
        document_index = document_index+1

    # 5 预览
    for i, d in enumerate(chunks):
        print(f"第{i+1}个分块")
        metadata = d.metadata
        print(f"文件路径：{metadata.get("file_path")} | 分块的起始位置：{metadata.get("chunk_start")} "
              f"| 分块的终止位置：{metadata.get("chunk_end")}| 分块前的索引位置：{metadata.get("document_index")}")
        print(d.text)

    # 6. 返回结果
    return chunks


if __name__ == "__main__":
    fixed_size_chunk_documents(config.base_path / "文档")