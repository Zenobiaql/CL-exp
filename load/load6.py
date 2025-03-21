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
from m_peft import LoraConfig, PeftModel, get_peft_model
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
        self.vla_path = "/mnt/data-qilin/mr_sqft/DoRA/google_robot_open_middle_drawer/merged/epoch0"
        self.adapter_path = "/mnt/data-qilin/mr_sqft/DoRA/google_robot_open_top_drawer/raw_adapter/epoch0"
        self.ckpt_path = "/mnt/data-qilin/mr_sqft/DoRA/google_robot_open_top_drawer/merged/epoch0"

# load model       
def load_model(cfg: LoadConfigs):
    processor = AutoProcessor.from_pretrained(cfg.vla_path, 
                                              trust_remote_code=True)
    
    init_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    merged_vla = PeftModel.from_pretrained(init_vla, cfg.adapter_path)
    merged_vla = merged_vla.merge_and_unload()
    
    os.makedirs(cfg.ckpt_path, exist_ok=True)
    processor.save_pretrained(cfg.ckpt_path)
    merged_vla.save_pretrained(cfg.ckpt_path)
    
    if os.path.exists(os.path.join(cfg.ckpt_path, "modeling_prismatic.py")) == False:
        shutil.copy(os.path.join(cfg.vla_path, "modeling_prismatic.py"), os.path.join(cfg.ckpt_path, "modeling_prismatic.py"))
        print("\n----------\ncode added\n----------\n")
    
    print("\n----------\nmodel saved\n----------\n")

