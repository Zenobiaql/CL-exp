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

# compaired with data_root_dir, output_dir has the same directory level to original directory
def merge_traj(data_root_dir: str, output_dir: str, logger)->None:
    
    os.makedirs(output_dir, exist_ok=True)
    
    instruction_path = os.path.join(output_dir, 'instruction.txt')
    action_path = os.path.join(output_dir, 'action.npy')
    frame_path = os.path.join(output_dir, 'frames.npy')
    
    root = Path(data_root_dir)
        
    cpu_id = os.getpid()
    logger.info(f"Task {root.name} is being processed by CPU core {cpu_id}.")
    print(f"Task {root.name} is being processed by CPU core {cpu_id}.")
    
    action_data = []
    frame_data = []
    
    idx = 0
    
    for subdir in tqdm(root.iterdir(), desc=f"Processing {root.name}"):
        
        if subdir.is_dir():
            
            # if it is the first folder, initialize instruction file
            if idx == 0:
                with open(instruction_path, 'w') as instruction:
                    with open(subdir / 'instruction.txt', 'r') as init_instruction:
                        instruction_content = init_instruction.read()
                        instruction.write(instruction_content)
            
            idx += 1
            
            instruction_file = subdir / 'instruction.txt'
            action_file = subdir / 'action.npy'
            frame_file = subdir / 'frames.npy'
            
            with open(instruction_file, 'r') as instruction:
                current_instruction = instruction.read()
            
            if current_instruction == instruction_content:
                action_data.append(np.load(action_file))
                frame_data.append(np.load(frame_file))
                
            else:
                ValueError("Instruction file is not consistent")
        
        else:
            logger.info(f"Skipping {str(subdir)}, as it is not a directory.")
            print(f"Skipping {str(subdir)}, as it is not standard.")
                
    np.save(action_path, np.concatenate(action_data, axis=0))
    np.save(frame_path, np.concatenate(frame_data, axis=0))
    
    logger.info(f"Merging {root.name} merged into {output_dir}, with {idx} trajectories")
    print(f"Merging {root.name} merged into {output_dir}, with {idx} trajectories")
    

def process_subdir(subdir: Path, output_dir: str, logger):
    if subdir.is_dir():
        logger.info(f"Start to merge {subdir.name}:")
        merge_traj(str(subdir), output_dir, logger)
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

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_subdir, [(subdir, os.path.join(cfg.output_dir, subdir.name), logger) for subdir in subdirs])
        

if __name__ == "__main__":
    main()
            

    
    