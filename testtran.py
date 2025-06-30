import os
import json
from datasets import Dataset, load_dataset
from tqdm import tqdm
import re

dataset = load_dataset('json', data_files='result.jsonl', split='train')

res = []
for data in tqdm(dataset):
    content = data['messages'][-1]['content']
    functions = re.findall(r'<tool_call>(.+?)</tool_call>', content, re.DOTALL)
    toolcall = []
    for function in functions:
        try:
            function = json.loads(function)
        except Exception:
            continue
        toolcall.append(function)
    toolcall = json.dumps(toolcall, ensure_ascii=False)
    res.append({'toolcall': toolcall})
new_dataset = Dataset.from_list(res)
new_dataset.to_json('toolcall.jsonl')

os.rename('toolcall.jsonl', 'result.json')