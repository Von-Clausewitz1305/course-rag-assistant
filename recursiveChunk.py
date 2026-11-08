from typing import List
from llama_index.core.schema import Document
from llama_index.core.node_parser import TokenTextSplitter
import util
import config

# 递归分块
def recursive_chunk_documents(
        intput_str :str
) -> List[Document]:
    # 1. 获取清洗后的数据
    documents = util.clean_all_formats(intput_str)

    print(f"清洗后的数据长度{len(documents)}")

    # 2. 根据符号去划分文本
    # 分隔符优先级列表：从前到后依次尝试，越靠前越优先（语义边界越完整）
    # 段落 -> 换行 -> 中英文句末 -> 分号 -> 逗号 -> 空格 -> 单字符
    separators = [
        "\n\n", "\n",
        "。", "！", "？",
        ". ", "! ", "? ",
        "；", ";",
        "，", ",",
        " ", "",
    ]

    # 3. 创建分块器
    spliter = TokenTextSplitter(
        separator=separators[0],    #首选段落标识进行分块
        backup_separators=separators[1:], #备用分隔符
    )

    # 4. 使用分块器来处理清洗好的文档
    chunks = []

    for d in documents:
        text = d.text
        if not text:
            continue

        path = d.metadata.get("file_path")

        nodes = spliter.get_nodes_from_documents(
            [Document(text=text, metadata={"file_path": path})]
        )

        for idx, n in enumerate(nodes):
            m = dict(n.metadata)
            m.update(
                {
                    "source_file_path" : path,
                    "chunk_index": idx,
                }
            )
            chunks.append(Document(text=n.text, metadata=m))

    for i, chunk in enumerate(chunks):
        print(f"第{i+1}个分块")
        print(chunk.metadata)
        print(chunk.text)

    return chunks

if __name__ == "__main__":
    recursive_chunk_documents(config.document_path)