# 随机抽样提取全量数据的千分之一
import os

import pandas

import config

# 原始全量数据的目录
source_dir = config.medical_all_data_path
# 简化数据的目录
target_dir = config.medical_data_path
# 随机抽样的比例
sample_ratio = 0.001

# 创建简化数据的目录
os.makedirs(target_dir, exist_ok=True)

# 全量目录不在本机时直接跳过，避免抽样脚本影响问答数据
if not os.path.isdir(source_dir):
    print(f"全量数据目录不存在，已跳过抽样: {source_dir}")
    filenames = []
else:
    filenames = os.listdir(source_dir)

# 列出source_dir目录下面的所有文件
for filename in filenames:
    # 只处理csv
    if filename.lower().endswith(".csv"):
        csv_path = os.path.join(source_dir, filename)
        out_path = os.path.join(target_dir, filename)

        try:
            df = pandas.read_csv(csv_path)
            total_rows = len(df)
            # 假如文档为空，直接跳过
            if total_rows == 0:
                print(f"跳过{filename} 文件为空")
                continue

            # 原始数据的行数小于1000的时候，至少要取1条
            if total_rows * sample_ratio < 1:
                sample_num = 1
            sample_num = int (total_rows * sample_ratio)

            df_sample = df.sample(n=sample_num, random_state=42)

            # 取到数据之后，需要保存到csv
            df_sample.to_csv(out_path, index=False, encoding="utf-8-sig")

            print(f"完成 {filename} 总{total_rows}行， 抽取了{sample_num} 行")

        except Exception as e:
            print(f"简化{filename}失败 {str(e)}")
