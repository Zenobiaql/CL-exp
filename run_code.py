"""
A script to run the CL experiment with different settings
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

from dataset import PizzaDataset
import random
from typing import List, Union
from scripts.sandbox_split import finetune
#from debugging.debug_train import finetune

@dataclass
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
    
@draccus.wrap()
def main(cfg: FinetuneConfig):  
    finetune(cfg)
    
if __name__ == "__main__":
    main()