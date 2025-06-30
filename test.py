import os
from typing import Dict, Any
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import InferArguments, infer_main, register_dataset, DatasetMeta, SubsetDataset

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/function_call_competition',
        subsets=[SubsetDataset('train', split=['train']), SubsetDataset('test', split=['test'])]
    ))

ckpt_dir = 'output/vx-xxx/checkpoint-xxx'  # last_checkpoint
result = infer_main(InferArguments(
    adapters=[ckpt_dir],
    temperature=0,
    val_dataset="swift/function_call_competition:test",
    infer_backend='vllm',
    vllm_max_lora_rank=8,
    result_path='result.jsonl',
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    max_new_tokens=512))