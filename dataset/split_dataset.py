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
class DataSplitConfig:
    root_dir: str
    tv_rate: float
    output_dir: str

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
def data_spliting(cfg: DataSplitConfig)->None:
    
    p = Path(cfg.root_dir)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    logger = get_logger("DataSplit", os.path.join(cfg.output_dir, "data_split.log"))
    logger.info(f"Starting data spliting for {p.name}")
    
    for sub_dir in tqdm(p.iterdir(), desc=f"Processing {p.name}"):
        
        if sub_dir.is_dir():
            sub_dir_name = sub_dir.name
            action = np.load(sub_dir / "action.npy")
            frames = np.load(sub_dir / "frames.npy")
            
            len_action = len(action)
            len_frames = len(frames)
            if len_action != len_frames:
                logger.error(f"Action length for {sub_dir_name} is {len_action} and frames length is {len_frames}.")
                raise ValueError(f"Action and frames length mismatch for {sub_dir_name}")
            else:
                length = len_action
                logger.info(f"Length of action and frames for {sub_dir_name} is {length}")
                
            action = np.random.shuffle(action)
            frames = np.random.shuffle(frames)
            
            tv_len = int(length * cfg.tv_rate)
            train_action = action[:tv_len]
            train_frames = frames[:tv_len]
            val_action = action[tv_len:]
            val_frames = frames[tv_len:]
            
            logger.info(f"Length of train action and frames for {sub_dir_name} is {len(train_action)}.")
            logger.info(f"Length of val action and frames for {sub_dir_name} is {len(val_action)}.")
            
            train_dir = os.makedirs(cfg.output_dir, "train", sub_dir_name, exist_ok=True)
            val_dir = os.makedirs(cfg.output_dir, "val", sub_dir_name, exist_ok=True)
            
            np.save(os.path.join(train_dir, "action.npy"), train_action)
            np.save(os.path.join(train_dir, "frames.npy"), train_frames)
            np.save(os.path.join(val_dir, "action.npy"), val_action)
            np.save(os.path.join(val_dir, "frames.npy"), val_frames)
            
            logger.info(f"Saved train action and frames for {sub_dir_name}.")
            
            with open(sub_dir / "instruction.txt", "r") as f:
                instruction = f.read()
            
            with open(os.path.join(train_dir, "instruction.txt"), "w") as f:
                f.write(instruction)
                logger.info(f"Saved train instruction for {sub_dir_name}.")
            
            with open(os.path.join(val_dir, "instruction.txt"), "w") as f:
                f.write(instruction)
                logger.info(f"Saved val instruction for {sub_dir_name}.")
                
        else:
            pass    

            
if __name__ == "__main__":
    data_spliting()