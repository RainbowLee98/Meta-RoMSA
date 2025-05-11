import pickle
def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 提取数据
    query_id = data['query']['id']
    raw_text = data['query']['raw_text']  # 确保键名正确
    labels = data['query']['regression_labels']
    output_file = "mosi_query_raw.txt"
    # 保存为文本文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(128):
            f.write(f"{query_id[i]}^{raw_text[i]}^{labels[i]}\n")  # 用空格分隔


file_path = 'MOSI_unaligned_50_divide.pkl'

print(file_path)


read_pkl_file(file_path)
