import os
from pathlib import  Path
from dotenv import load_dotenv

# 父级目录
base_path = Path(__file__).parent

# 嵌入模型的路径（放在项目内，换电脑不用改盘符）
model_path = str(base_path / "models" / "BAAI" / "bge-m3")

# 文档所在的目录
document_path = base_path / "文档"

# 结构分块目录
structure_path = document_path / "结构分块"

# 解析通义千问大模型秘钥
load_dotenv(base_path/".env")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

# 医疗问答全量数据目录（只给抽样脚本用；目录不存在时不影响问答）
medical_all_data_path = base_path / "medical" / "data_all"

# 医疗问答实际检索用的 CSV 目录
medical_data_path = base_path / "medical" / "data"

# 课程问答语料（jsonl，每行一个课件片段）
course_data_path = base_path / "classdate"