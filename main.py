# 22GiB
import os
from typing import Dict, Any
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    TrainArguments, sft_main, register_dataset, DatasetMeta, ResponsePreprocessor, SubsetDataset
)

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/function_call_competition',
        subsets=[SubsetDataset('train', split=['train']), SubsetDataset('test', split=['test'])]
    ))

if __name__ == '__main__':
    sft_main(TrainArguments(
        model='Qwen/Qwen3-8B',
        dataset=['swift/function_call_competition:train'],
        agent_template='hermes',
        loss_scale='hermes',
        train_type='lora',
        torch_dtype='bfloat16',
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        lora_rank=8,
        lora_alpha=32,
        target_modules=['all-linear'],
        gradient_accumulation_steps=16,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        max_length=2048,
        output_dir='output',
        warmup_ratio=0.05,
        dataset_num_proc=4,
        dataloader_num_workers=4,
        use_liger_kernel=True,
        attn_impl='flash_attn',
        packing=True,
        save_only_model=True,
        acc_strategy='seq',
    ))
