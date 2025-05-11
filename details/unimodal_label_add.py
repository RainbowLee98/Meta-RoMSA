import pandas as pd
import pickle
import numpy as np
import csv

from h5py.h5pl import append

# 读取CSV构建标签字典
csv_path = 'dataset/MOSEI_query_unimodallabel.csv'

pkl_path = 'dataset/mosei_unaligned_50_divide.pkl'
new_pkl_path = 'dataset/MOSEI_unaligned_50_meta_updated.pkl'


csv_reader = csv.reader(open(csv_path))
# csv_data = pd.read_csv(csv_path)
csv_dict = []
for row in csv_reader:
    if row[2] == 'labelT':
        continue
    csv_dict.append({
        'id':row[0],
        'regression_labels_T':row[2],
        'regression_labels_A':row[3],
        'regression_labels_V':row[4],

    })

# 转换为目标字典结构
keys = ['id', 'regression_labels_T', 'regression_labels_A', 'regression_labels_V']
result_dict = {key: [item[key] for item in csv_dict] for key in keys}

# 加载PKL文件
with open(pkl_path, 'rb') as f:
    pkl_data = pickle.load(f)

query_data = pkl_data['query']
train_data = pkl_data['train']
valid_data = pkl_data['valid']
test_data = pkl_data['test']

query_data['regression_labels_T'] = []
query_data['regression_labels_A'] = []
query_data['regression_labels_V'] = []

# 构建ID到索引的映射
result_ids = result_dict['id']
id_to_index = {id: idx for idx, id in enumerate(result_ids)}

# 遍历query_data的ID，按顺序提取并添加标签
for sample_id in query_data['id']:
    if sample_id in id_to_index:
        idx = id_to_index[sample_id]
        query_data['regression_labels_T'].append(float(result_dict['regression_labels_T'][idx]))
        query_data['regression_labels_A'].append(float(result_dict['regression_labels_A'][idx]))
        query_data['regression_labels_V'].append(float(result_dict['regression_labels_V'][idx]))
    else:
        raise ValueError(f"ID {sample_id} not found in result_dict")


print(query_data)
merged = {
    "query": query_data,
    "train": train_data,
    "test": test_data,
    "valid": valid_data
}

# # 保存修改后的文件
with open(new_pkl_path, 'wb') as f:
    pickle.dump(merged, f)

print("处理完成，新增labelT/labelA/labelV字段")
