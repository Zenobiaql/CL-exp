"""
script for finetuning on each task separately, and check accuracy on all tasks
"""

import os
import draccus
import tqdm
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.utils.data.distributed import DistributedSampler

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from dataclasses import dataclass
from collections import deque

from dataset import SimplerDataset
import random
from typing import List, Union
from log import ModelLogger, ModuleTracker

from copy import deepcopy
from multiprocessing import cpu_count


# DDP process group setup
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=torch.distributed.timedelta(seconds=3600))


"""
Finetuning configuration, passed with .yaml or .yml file
for reference
"""

class FinetuneConfig:
    
    # model path
    vla_path: str

    "Data Configuration"
    # dataset root directory       
    data_dir: str
    # name of dataset class                        
    dataset_name: str
    # directory for log and checkpoints                            
    run_root_dir: str
    num_workers: int                                    

    "Finetuning Configuration"
    epochs: int                                                
    batch_size: int                                                                                        
    save_steps: int                                            
    learning_rate: float                                       
    grad_accumulation_steps: int                               
    image_aug: bool                                         
                        
    "LoRA Arguments"
    use_lora: bool                                        
    lora_rank: int                                           
    lora_dropout: float
    lora_module: Union[List[str], str]


"""
FinetuneConfig finetune() uses is in run_code.py, FinetuneConfig above is just for reference
"""          

def finetune(cfg: FinetuneConfig)->None:
    
    # set up DDP process group
    ddp_setup()
    
    if dist.get_rank() == 0:
        os.makedirs(cfg.run_root_dir, exist_ok=True)
    
    dataloader_set = {}
    val_dataloader_set = {}

    data_root_dir = Path(cfg.data_dir)
            
    if data_root_dir.is_dir():
    
        for task in tqdm.tqdm(data_root_dir.iterdir(), desc="Tasks"):
            
            if task.is_dir():
                dataloader_set[task.name] = task.name
                val_dataloader_set[task.name] = task.name
                
            else:
                pass
        
    else:
        pass
    
    
    for task_id, value in dataloader_set.items():
        
        task_sub_dir = os.path.join(cfg.run_root_dir, task_id)
        os.makedirs(task_sub_dir, exist_ok=True)
            
        logger_complex = ModelLogger(cfg.vla_path, task_id, cfg.batch_size, cfg.learning_rate, task_sub_dir)
        
        exp_id = (
            f"{task_id}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
            
        logger_complex.log_training_config(exp_id)
    

    device_id = dist.get_rank()
    dist.destroy_process_group()
    print(f"Finished running code on rank {device_id}.")

if __name__ == "__main__":
    finetune()
    
    