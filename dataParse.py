# 数据解析


# 1. 希望把待处理的数据在同一个目录下面进行处理

# 2. 对数据进行批量读取

# 3. 对拿到的数据进行结果预览

# 4. 结果返回


from llama_index.core import SimpleDirectoryReader
from pathlib import Path


base_path = Path(__file__).parent

# 数据解析函数
def parse_all_formats(input_dir):

    # 1. 创建加载数据的对象
    reader = SimpleDirectoryReader(
        input_dir=input_dir
    )
    # 2. 加载数据
    documents =  reader.load_data()

    # 3. 数据预览
    for i, doc in enumerate(documents):

        file_path = doc.metadata.get("file_path")
        text = doc.text[:200]
        print(f"\n  第{i+1} 个对象 | 文件路径：{file_path}")
        print(text)

    return documents


if __name__ == "__main__":
    parse_all_formats(base_path/ "文档")