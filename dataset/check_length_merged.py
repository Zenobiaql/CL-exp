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
                    
            action = np.load(sub_dir / "action.npy")
            frames = np.load(sub_dir / "frames.npy")
            logger.info(f"Action length is {len(action)} and frames length is {len(frames)} for {sub_dir.name}.")
            if len(action) != len(frames):
                logger.error(f"Action and frames length mismatch for {sub_dir.name}.")
                    
        logger.info(f"Instruction check for {sub_dir.name} is done.")
        
if __name__ == "__main__":
    check_data()