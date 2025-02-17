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

vla = AutoModelForVision2Seq.from_pretrained(
        "/datahdd_8T/zql/CL-Adapter/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

print([key for key, _ in vla.named_modules()])