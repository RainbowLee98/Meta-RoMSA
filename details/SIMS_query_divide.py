import pickle
import os
import numpy as np
np.random.seed(42)
ori_pth = 'dataset/SIMSV2/Processed/unaligned_normalized.pkl'
save_pth = 'dataset/SIMSV2/Processed/meta_unaligned_normalized.pkl'
with open(ori_pth, 'rb') as f:
    data = pickle.load(f)

# 提取不同的数据集
train_data = data['train']
test_data = data['test']
valid_data = data['valid']

random_indices = np.random.choice(len(train_data['id']), 128, False)
query = {
    key: np.array(value)[random_indices] if isinstance(value, (list, np.ndarray)) else [value[i] for i in random_indices]
    for key, value in train_data.items()
}

# 生成剩余训练集的索引
remaining_indices = np.setdiff1d(np.arange(len(train_data['id'])), random_indices)
# 创建剩余训练集
remaining_train = {
    key: np.array(value)[remaining_indices] if isinstance(value, (list, np.ndarray)) else [value[i] for i in remaining_indices]
    for key, value in train_data.items()
}

print(query.keys())
print(len(query['id']))
merged = {
    "query": query,
    "train": remaining_train,
    "test": test_data,
    "valid": valid_data
}


# for i in range(len(query['id'])):
#     print(query['id'][i])

with open(save_pth, 'wb') as f:
    pickle.dump(merged, f)