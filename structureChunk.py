import re
import csv
from typing import List
from  llama_index.core.schema import  Document

import util
import config

max_chunk_chars = 1500
# PDF 页内次级结构切分点：编号问题（1.）或项目符号（• ◦ -）前
_PDF_SUBSECTION_RE = re.compile(r"(?=\n(?:\d+\.|•|◦|\-)\s*)")

# 从ppt的解析结果提取文本
def _extract_section_text(section : dict) -> str:
    return section.get("content")

# 处理PPT类型文档分块
def _chunk_ppt_document(
        d: Document
) -> List[Document]:
    sections = d.metadata.get("text_sections")
    slide_title = d.metadata.get("title")

    texts = [_extract_section_text(s) for s in sections]
    texts = [t for t in texts if t]

    full_text = "\n".join(texts)
    if len(full_text) <= max_chunk_chars:
        meta = {
            "chunk_index": 0
        }
        return [Document(text=full_text, metadata=meta)]

    groups = []
    current = []

    for i, text in enumerate(texts):
        current.append(text)

    if current:
        groups.append("\n".join(current))

    # 转换成分块
    chunks: List[Document] = []

    for idx, group in enumerate(groups):
        chunk_text = group
        if slide_title and slide_title not in chunk_text:
            chunk_text = f"{slide_title}\n{chunk_text}"

        meta = {
            "section_index": idx
        }
        chunks.append(Document(text=chunk_text, metadata=meta))

    return chunks

# 按照文档结构对pdf类型文档进行分块
def _chunk_pdf_document(
        d : Document
) -> List[Document]:
    text = d.text
    if not text:
        return []

    page_label = d.metadata.get("page_label")
    base_meta = {
        "page_label":page_label
    }

    if len(text) <= max_chunk_chars:
        meta = dict(base_meta)
        meta["chunk_index"] = 0
        return [Document(text=text, metadata=meta)]

    parts = [p.strip() for p in _PDF_SUBSECTION_RE.split(text) if p.strip()]
    if len(parts) <=1:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 转换成分块
    chunks: List[Document] = []
    for idx, part in enumerate(parts):
        meta = dict(base_meta)
        meta.update(
            {
                "chunk_index": idx
            }
        )
        chunks.append(Document(text=part, metadata=meta))

    return chunks

# 获取csv文件的头
def _read_csv_header(path : str) -> str:
    with open(path, newline="", encoding="utf-8-sig") as f:
        row = next(csv.reader(f), None)
        return ",".join(row)

# 处理csv类型文件
def _chunk_csv_document(
    d: Document
)->List[Document]:
    text = d.text
    if not text:
        return []
    lines = [ line.strip() for line in text.splitlines() if line.strip()]
    path = d.metadata.get("file_path")
    header = _read_csv_header(path)
    data_lines = lines[0:]

    chunks: List[Document] = []
    for idx, line in enumerate(data_lines):
        chunk_text = f"{header}\n{line}"
        meta = {
            "chunk_index": idx
        }
        chunks.append(Document(text=chunk_text, metadata=meta))

    return chunks

# 单个文档分块
def _chunk_single_document(
        d: Document
)->List[Document]:
    # 1. 处理ppt类型的文档
    if d.metadata.get("text_sections"):
        return _chunk_ppt_document(d)
    # 2. 处理pdf类型的文档
    if d.metadata.get("page_label"):
        return  _chunk_pdf_document(d)
    # 3. 处理csv类型的文档
    if d.metadata.get("file_name").endswith(".csv"):
        return _chunk_csv_document(d)
    # 4. txt类型文档
    return [d]

# 基于文档结构的分块
def structure_chunk_documents(
        input_str : str,
) -> List[Document]:
    # 1. 获取清洗之后的数据
    documents = util.clean_all_formats(input_str)

    # 2. 逐一分块
    chunks: List[Document] = []
    for d in documents:
        doc_chunks = _chunk_single_document(d)
        chunks.extend(doc_chunks)

    # 3. 测试输出
    for i, chunk in enumerate(chunks):
        print(f"第{i+1}个分块")
        print(chunk.metadata)
        print(chunk.text)

    return chunks

if __name__ == "__main__":
    structure_chunk_documents(config.structure_path)