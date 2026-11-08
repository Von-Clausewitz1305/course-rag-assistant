#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download(
    'BAAI/bge-m3',
    cache_dir='D:\\models'
)

print("模型已经下载到 ", model_dir)