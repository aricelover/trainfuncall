import json
import os
from glob import glob
from transformers import AutoTokenizer
from tqdm import tqdm  # 用于显示进度条


def count_tokens_in_jsonl_folder(folder_path, text_field='text', batch_size=1000):
    """
    计算文件夹内所有JSONL文件在Qwen tokenizer下的总token数

    参数:
    folder_path: 包含JSONL文件的文件夹路径
    text_field: JSON对象中包含文本的字段名 (默认为'text')
    batch_size: 批处理大小 (默认1000)
    """
    # 1. 初始化Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)

    total_tokens = 0
    file_count = 0
    processed_files = 0

    # 2. 获取所有JSONL文件
    jsonl_files = glob(os.path.join(folder_path, "*.jsonl"))
    total_files = len(jsonl_files)

    if total_files == 0:
        print(f"在文件夹 {folder_path} 中未找到JSONL文件")
        return 0

    print(f"找到 {total_files} 个JSONL文件，开始处理...")

    # 3. 处理每个文件
    for file_path in tqdm(jsonl_files, desc="处理文件中"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch = []
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get(text_field)
                        if text:
                            batch.append(text)

                            # 批量处理
                            if len(batch) >= batch_size:
                                tokens = tokenizer(batch, add_special_tokens=False)["input_ids"]
                                total_tokens += sum(len(t) for t in tokens)
                                batch = []
                    except json.JSONDecodeError:
                        print(f"警告: 文件 {file_path} 中跳过无效的JSON行")
                        continue

                # 处理剩余文本
                if batch:
                    tokens = tokenizer(batch, add_special_tokens=False)["input_ids"]
                    total_tokens += sum(len(t) for t in tokens)

            processed_files += 1
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    print("\n" + "=" * 50)
    print(f"处理完成: {processed_files}/{total_files} 个文件成功处理")
    print(f"总Token数量: {total_tokens}")
    print("=" * 50)

    return total_tokens


if __name__ == "__main__":
    # 使用示例
    folder_path = "/path/to/your/jsonl/folder"  # 替换为你的文件夹路径
    count_tokens_in_jsonl_folder(folder_path)