"""
Script for analyzing intruder dimension of a single module in a model.
"""
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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


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


# Load usv tensors
def load_uv_tensors(root_path: str, module_name: str, logger)->Dict:
    u_tensor = []
    v_tensor = []
    
    root = Path(root_path)
    for ckpt in tqdm(root.iterdir()):
        if not ckpt.is_dir():
            continue
        logger.info(f"Loading {ckpt.name} tensor:")
        u_matrix = torch.load(ckpt / f"{module_name}_u.pt")
        v_matrix = torch.load(ckpt / f"{module_name}_v.pt")
        
        u_tensor.append(u_matrix)
        v_tensor.append(v_matrix)
            
    logger.info(f"Loaded {len(u_tensor)} u tensors, {len(v_tensor)} v tensors")
    return u_tensor, v_tensor


# Calculate cosine similarity
def cosine_similarity(u_tensor: List[torch.Tensor], v_tensor: List[torch.Tensor], logger)->List:
    ind = []
    
    for i in tqdm(range(len(u_tensor))):
        
            inv = 0
            for k in range(10):
                invade = True
                for t in tqdm(range(len(u_tensor[0]))):
                    if abs(torch.nn.functional.cosine_similarity(u_tensor[i][k], u_tensor[0][t], dim=0)) > 0.1:
                        invade = False
                if invade:
                    inv += 1
            ind.append((0, i, inv))

    
    # 生成热图数据
    heatmap_data = torch.zeros((len(u_tensor), len(u_tensor)))
    for i, j, inv in ind:
        heatmap_data[i, j] = inv
    
    data = heatmap_data.numpy()
    sns.heatmap(data, annot=True, fmt='.1f', cmap='coolwarm')
    plt.show()
    
    return ind


@dataclass
class IDimConfig:
    
    root_dir: str
    module_name: str
    output_path: str
    log_path: str
    
    
@draccus.wrap()    
def main(cfg: IDimConfig):
    
    # Logger setup
    logger = get_logger("IDim", cfg.log_path)
    
    # Load tensors
    u_tensor, v_tensor = load_uv_tensors(cfg.root_dir, cfg.module_name, logger)
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity(u_tensor, v_tensor, logger)
    
    # Save cosine similarity
    print(cos_sim)

if __name__ == "__main__":
    main()