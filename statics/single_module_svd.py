"""
Script for analyzing the singular values of a single module in a model.
"""
import os
import numpy as np
from tqdm import tqdm
import torch
import logging
import draccus
from dataclasses import dataclass
from transformers import AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model
from typing import List, Union


# Logger setup
def get_logger(name, file_path):
    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file_path is not None: 
        logger.addHandler(f_handler)
    
    return logger

 
"""
We use LoRA configs here with configs which model trained with,
in order to find exact trained parameters.
"""
@dataclass
class SVDConfig:
    
    # model path
    vla_path: str                                         
                        
    # lora config
    use_lora: bool                                        
    lora_rank: int                                           
    lora_dropout: float
    lora_module: Union[List[str], str]
    
    # logger
    logger_path: str
    statics_path: str
    svd_path: str


"""
FinetuneConfig finetune() uses is in run_code.py, FinetuneConfig above is just for reference
"""      
@draccus.wrap()
def svd_analysis(cfg: SVDConfig)->None:
    
    logger = get_logger("SVD", cfg.logger_path)
    
    if not os.path.exists(cfg.vla_path):
        logger.error(f"Model path not found: {cfg.vla_path}")
        raise FileNotFoundError(f"Model path not found: {cfg.vla_path}")
    
    # load target checkpoint
    init_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # we use the same config as training setting to figure out those trained parameters
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_module,
            init_lora_weights="gaussian",
        )
        init_vla = get_peft_model(init_vla, lora_config)
        init_vla.print_trainable_parameters()
        
    device = torch.device("cuda")
    print(f"Running code on {device}.")
    init_vla = init_vla.to(device)
    
    os.makedirs(cfg.svd_path, exist_ok=True)
    
    with open(os.path.join(cfg.statics_path, "names.txt"), "w") as f:
        for name, param in tqdm(init_vla.named_parameters()):
                if param.requires_grad:
                    size = param.numel()
                    f.write(f"name: {name}, size: {size}\n")
                    param = param.detach()
                    u, s, v = torch.svd(param)
                    u = u.cpu().numpy()
                    s = s.cpu().numpy()
                    v = v.cpu().numpy()
                    np.save(os.path.join(cfg.svd_path, f"{name}_u.npy"), u)
                    np.save(os.path.join(cfg.svd_path, f"{name}_s.npy"), s)
                    np.save(os.path.join(cfg.svd_path, f"{name}_v.npy"), v)
                    logger.info(f"Saving {name} SVD decomposition.")       


if __name__ == "__main__":
    svd_analysis()