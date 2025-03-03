import os
import logging 
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import draccus
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

@dataclass
class DataMergingConfig:
    
    root_dir: str
    output_dir: str

"""
    directory to merge has a double level structure, task/traj/ins&frame&act
    target is to merge all ins&frame&act into one folder
"""

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


def _get_traj_action(data_root_dir: Path, logger)->None:
        
    if data_root_dir.is_dir():
        action_file = data_root_dir / 'action.npy'
        action_data = np.load(action_file)
        
    else:
        logger.info(f"Skipping {str(data_root_dir)}, as it is not a directory.")
        print(f"Skipping {str(data_root_dir)}, as it is not standard.")
    
    return action_data
    
            
    
def _get_traj_frame(data_root_dir: Path, logger)->None:
    
    if data_root_dir.is_dir():
        frame_file = data_root_dir / 'frames.npy'
        frame_data = np.load(frame_file)
        
    else:
        logger.info(f"Skipping {str(data_root_dir)}, as it is not a directory.")
        print(f"Skipping {str(data_root_dir)}, as it is not standard.")
    
    return np.delete(frame_data, -1)
        

def _collect_submerge(data_root_dir: str, output_dir: str, logger)->None:
    
    os.makedirs(output_dir, exist_ok=True)
    
    instruction_path = os.path.join(output_dir, 'instruction.txt')
    action_path = os.path.join(output_dir, 'action.npy')
    frame_path = os.path.join(output_dir, 'frames.npy')
    
    root = Path(data_root_dir)
    
    # add instruction if not exist
    if os.path.exists(instruction_path) == False:
        with open(instruction_path, 'w') as instruction:
            with open(next(root.iterdir()) / 'instruction.txt', 'r') as init_instruction:
                instruction_content = init_instruction.read()
                instruction.write(instruction_content)
    
    with Pool(processes=cpu_count()) as pool:
        action_data = pool.starmap(_get_traj_action, [(subdir, logger) for subdir in root.iterdir()])
        frame_data = pool.starmap(_get_traj_frame, [(subdir, logger) for subdir in root.iterdir()])
        
    if len(action_data) != len(frame_data):
        logger.error(f"Action and frame data length mismatch for {root.name}")
        raise ValueError(f"Action and frame data length mismatch for {root.name}")
    
    np.save(action_path, np.concatenate(action_data, axis=0))
    np.save(frame_path, np.concatenate(frame_data, axis=0))
    
    logger.info(f"Merging {root.name} merged into {output_dir}, with {len(action_data)} trajectories")
    print(f"Merging {root.name} merged into {output_dir}, with {len(action_data)} trajectories")
    

def process_subdir(subdir: Path, output_dir: str, logger):
    if subdir.is_dir():
        logger.info(f"Start to merge {subdir.name}:")
        _collect_submerge(str(subdir), output_dir, logger)
        logger.info(f"Finish merging {subdir.name}")
        logger.info("")
    else:
        logger.info(f"Skipping {subdir.name}, as it is not a directory.")
        logger.info("")
        print(f"Skipping {subdir.name}, as it is not standard.")


@draccus.wrap()
def main(cfg: DataMergingConfig)->None:
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    logger = get_logger('data_merging', os.path.join(cfg.output_dir, f"{current_time}_data_merging.log"))
    
    # root include triple level structure, root/task/traj/ins&frame&act
    p = Path(cfg.root_dir)
    subdirs = [subdir for subdir in p.iterdir() if subdir.is_dir()]
    
    logger.info(f"CPU count: {cpu_count()}")
    logger.info(f"Start to merge {len(subdirs)} directories.")

    for subdir in subdirs:
        process_subdir(subdir, os.path.join(cfg.output_dir, subdir.name), logger)
        
    logger.info(f"Finish merging {len(subdirs)} directories.")
        

if __name__ == "__main__":
    main()
            

    
    