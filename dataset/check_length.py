import os
import logging 
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import random
import draccus
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass


@dataclass
class CheckLengthConfig:
    root_dir: str
    stat_dir: str
    
def get_logger(name, file_path)->logging.Logger:
    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file_path is not None: 
        logger.addHandler(f_handler)
    
    return logger

@draccus.wrap()
def check_data(cfg: CheckLengthConfig)->None:
    
    p = Path(cfg.root_dir)
    
    os.makedirs(cfg.stat_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-", time.localtime())
    
    logger = get_logger("CheckData", os.path.join(cfg.stat_dir, f"{current_time}check_data.log"))
    logger.info(f"Starting data check for {p.name}")
    
    for sub_dir in tqdm(p.iterdir(), desc=f"Processing {p.name}"):
        if sub_dir.is_dir():
            idx = 0
            num_match = 0
            num_standard = 0
            
            for subsub_dir in sub_dir.iterdir():
                if subsub_dir.is_dir():
                    if idx == 0:
                        with open(subsub_dir / "instruction.txt", "r") as f:
                            instruction = f.read()
                    idx += 1
                    
                    with open(subsub_dir / "instruction.txt", "r") as f:
                        instruction_new = f.read()
                        
                    if instruction != instruction_new:
                        logger.error(f"Instruction mismatch for {subsub_dir.name} in {sub_dir.name}")
                    
                    action = np.load(subsub_dir / "action.npy")
                    frames = np.load(subsub_dir / "frames.npy")
                    if len(action) == len(frames):
                        logger.error(f"Action and frames length match for {subsub_dir.name} in {sub_dir.name}")
                        num_match += 1
                    
                    elif len(action)+1 == len(frames):
                        num_standard += 1
                        logger.info(f"Action: {len(action)} and Frames: {len(frames)} for {subsub_dir.name} in {sub_dir.name}")
                    
        logger.info(f"Instruction check for {sub_dir.name} is done.")
        logger.info(f"{sub_dir.name} has {idx} subdirectories.")
        logger.info(f"{num_match} subdirectories have matching action and frames length.")
        logger.info(f"{num_standard} subdirectories have standard action and frames length.")
        
if __name__ == "__main__":
    check_data()