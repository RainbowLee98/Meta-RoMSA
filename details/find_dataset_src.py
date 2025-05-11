import shutil
import os
from config.get_data_root import data_root


def process_file(input_file, source_dir, target_dir):
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 分割行内容
            parts = line.split('\t')

            if len(parts) < 2:
                print(f"第{line_num}行格式错误，已跳过")
                continue

            _, name_part = parts[0], parts[1]

            # 提取数字部分
            try:
                _, number = name_part.split('$_$')
            except ValueError:
                print(f"第{line_num}行名称格式错误: {name_part}")
                continue

            # 构造路径
            src_path = os.path.join(source_dir, parts[0])
            src_path = os.path.join(src_path, f'{number}.mp4')
            print(src_path)
            dst_path = os.path.join(target_dir, f'{name_part}.mp4')
            print(dst_path)

            # 执行复制
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"成功复制: {src_path} -> {dst_path}")
            else:
                print(f"文件不存在，无法复制: {src_path}")


if __name__ == "__main__":
    dataset_name = 'MOSEI'
    txt_file = dataset_name.lower()+'_query_ids.txt'

    src_path = os.path.join(dataset_name,'Raw')
    src = os.path.join(data_root, src_path)
    qry_path = os.path.join(dataset_name,'Query')
    dst = os.path.join(data_root, qry_path)

    process_file(txt_file, src, dst)
