"""
script for saving full ckpt
"""

import os
import draccus
import tqdm
import logging
import time
from pathlib import Path
import datetime

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

from dataset import SplitDataset
import random
from typing import List, Union
from log import ModelLogger, ModuleTracker

from copy import deepcopy
from multiprocessing import cpu_count
import numpy as np
import shutil


# load configs
class LoadConfigs:
    def __init__(self):
        self.vla_path = "/home/v-qilinzhang/model"
        self.adapter_path = "/home/v-qilinzhang/ckpts/epoch24"

# load model       
def load_model():
    cfg = LoadConfigs()
    processor = AutoProcessor.from_pretrained(cfg.vla_path, 
                                              trust_remote_code=True)
    
    init_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    print("model loaded")
    
    merged_vla = PeftModel.from_pretrained(init_vla, cfg.adapter_path)
    print("model loaded")
    for name, param in merged_vla.named_parameters():
        if "lora" in name:
            print(name)
        else:
            print("not lora")
                            
if __name__ == "__main__":
    load_model()