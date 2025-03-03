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