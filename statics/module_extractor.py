import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import torch
import logging
import draccus
from dataclasses import dataclass
from transformers import AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model
from typing import List, Union, Dict
import shutil


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


# Fixer to add modeling_prismatic.py
# ckpt_path: path to fix, original_path: path to copy from
def fixer(ckpt_path: str, original_path: str, logger)->None:
    
    if not os.path.exists(ckpt_path):
        logger.error(f"File not found: {ckpt_path}")
        raise FileNotFoundError(f"Destination file not found: {ckpt_path}")
    if not os.path.exists(original_path):
        logger.error(f"File not found: {original_path}")
        raise FileNotFoundError(f"Source file not found: {original_path}")
    
    src_path = os.path.join(original_path, "modeling_prismatic.py")
    dst_path = os.path.join(ckpt_path, "modeling_prismatic.py")
    
    if os.path.exists(dst_path):
        logger.info(f"Destination file already exists: {dst_path}")
        return
    else:
        shutil.copy(src_path, dst_path)
        logger.info(f"File has been copied from {src_path} to {dst_path}")
        print(f"File has been copied from {src_path} to {dst_path}")


# Extractor to extract analysis target module and do SVD
# support a .txt file to specify the module name
def extractor(ckpt_path: str, name_file_path: str, root_path: str, device, logger)->Dict:
    
    module_to_fix_path = os.path.join(ckpt_path, "modeling_prismatic.py")
    if not os.path.exists(module_to_fix_path):
        logger.error(f"File not found: {ckpt_path}")
        raise FileNotFoundError(f"To be fixed file not found: {ckpt_path}")
    
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla = vla.to(device)
    logger.info(f"Model has been loaded from {ckpt_path}")
    
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(os.path.join(root_path, "before_svd"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "u_marices"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "s_marices"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "v_marices"), exist_ok=True)

    # extract by names
    with open(name_file_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            for name, param in vla.named_parameters():
                line_in_name = False
                if line in name:
                    line_in_name = True
                    param = param.detach()
                    u, s, v = torch.linalg.svd(param)
                    torch.save(param, os.path.join(root_path, "before_svd", f"{name}_before_svd.pt"))
                    torch.save(u, os.path.join(root_path, "u_marices", f"{name}_u.pt"))
                    torch.save(s, os.path.join(root_path, "s_marices", f"{name}_s.pt"))
                    torch.save(v, os.path.join(root_path, "v_marices", f"{name}_v.pt"))
                    logger.info(f"SVD decomposition has been done for {name}")
                    break
            if line_in_name == False:
                logger.error(f"Module not found: {line}")
                raise ValueError(f"Module not found: {line}")
    
    
    
    for name, param in vla.named_parameters():
        if module_name in name:
            param = param.detach()
            extracted_modules[name] = {}
            logger.info(f"Extracted module: {name}")
            u, s, v = torch.svd(param)
            extracted_modules[name]["u"] = u
            extracted_modules[name]["s"] = s
            extracted_modules[name]["v"] = v
            logger.info(f"SVD decomposition has been done for {name}")
    
    return extracted_modules


# use extractor() and fix() to process a merged ckpt dir
def process_merged(original_path: str,
                   run_root_path: str, 
                   module_name: str,
                   statistics_path: str, 
                   device,
                   )->None:
    
    current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    os.makedirs(statistics_path, exist_ok=True)
    logger = get_logger(f"{run_root_path}-SVD", os.path.join(statistics_path, f"svd-{current_time}.log"))
    logger.info(f"Start analyzing {run_root_path}.")
    
    merged_path = Path(str(os.path.join(run_root_path, "merged")))
    
    for ckpt_path in tqdm(merged_path.iterdir()):
        
        if not ckpt_path.is_dir():
            continue
        
        num_epochs = ckpt_path.name
        stat_path = os.path.join(statistics_path, num_epochs)
        os.makedirs(stat_path, exist_ok=True)
        fixer(str(ckpt_path), original_path, logger)
        extracted_modules = extractor(str(ckpt_path), module_name, device, logger)
        
        for name, usv in extracted_modules.items():
            torch.save(usv["u"], os.path.join(stat_path, f"{name}_u.pt"))
            torch.save(usv["s"], os.path.join(stat_path, f"{name}_s.pt"))
            torch.save(usv["v"], os.path.join(stat_path, f"{name}_v.pt"))
            logger.info(f"Saving {name} SVD decomposition.")
        
        extracted_modules.clear()
    
    logger.info(f"Analysis has been done for {run_root_path}.")
    print(f"Analysis has been done for {run_root_path}.")
    
    return


if __name__ == "__main__":
    process_merged()