# 数据清洗


# 1. 考虑创建一个清洗单个字符串的方法

# 2. 考虑创建一个清洗Document对象的方法

# 3. 考虑创建一个批量处理的方法   处理的是目录下面的所有的文件

import re
from pathlib import Path

import unicodedata
from llama_index.core import Document
from util import parse_all_formats
from config import base_path

# 清洗单个字符串
def clean_text(text: str, suffix: str) -> str:

    # 1. 清洗掉隐形字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\u200b\u200c\u200d]", "", text)
    # 2. 全角转半角
    text = unicodedata.normalize("NFKC", text)

    # 3. 统一换行符
    text = text.replace("\r\n","\n").replace("\r", "\n")

    # 4. 针对特殊类型文件处理
    if suffix.lower() in {".pptx", ".ppt"}:
        text = re.sub(r"^Title:\s*.+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\[Speaker Notes\]:\s*", "", text, flags=re.MULTILINE)

    # 5. 剔除markdown标记
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # 去掉 ** 保留里面的字
    text = text.replace("#", "")  # 简单粗暴去掉所有 #

    # 6. 去除多余的空行
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)  # 第一步：去掉每行首尾的空白
    text = re.sub(r"\n+", "\n", text)  # 第二步：连续空行压成一个换行

    return text.strip()


# 实现清洗一个Document对象
def clean_doc(document: Document) -> Document:
    # 1. 获取文件的后缀
    suffix = Path(document.metadata.get("file_path")).suffix
    text = document.text

    cleaned_text = clean_text(text, suffix)

    # 2. 返回清洗后的Document对象
    return Document(text=cleaned_text, metadata=document.metadata)


# 实现批量处理
def clean_all_formats(input_dir: Path) -> list[Document]:
    # 1. 先把文件从文件夹里面解析出来，得到一个Document对象的列表
    docs = parse_all_formats(input_dir)
    # 2. 逐个清洗Document对象
    cleaned_docs = [clean_doc(doc) for doc in docs]
    # 3. 进行简单的验证
    for i, doc in enumerate(cleaned_docs):
        print(f"\n 第{i+1}个对象 | 路径: {doc.metadata.get('file_path')}")
        print(doc.text[:200])
    # 4. 返回清洗过后的对象列表
    return cleaned_docs

if __name__ == "__main__":
    clean_all_formats(base_path / "文档")