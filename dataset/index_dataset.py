"""
Dataset initialized with index, read data when getitems
"""
from pathlib import Path
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from prismatic.vla.datasets import RLDSBatchTransform
from tqdm import tqdm
from dataclasses import dataclass

class SimplerDataset(Dataset):
    def __init__(self, data_root_dir: Path, action_tokenizer, processtokenier, image_transform, prompt_builder_fn):
        self.data_dir = data_root_dir
        self.batchTransform = RLDSBatchTransform(
            action_tokenizer,
            processtokenier,
            image_transform, 
            prompt_builder_fn)
               
        action_file  = self.data_dir / 'action.npy'
        actions = np.load(action_file)
        self.indx = [i for i in range(len(actions)//2)]
        
                      
    def __len__(self):
        return len(self.indx)
    
    
    def __getitem__(self, idx):
        frame_file = self.data_dir / 'frames.npy'
        action_file  = self.data_dir / 'action.npy'
        instruction_file = self.data_dir / 'instruction.txt'
        
        frames = np.load(frame_file)
        actions = np.load(action_file)
        
        with open(instruction_file, 'r') as f:
            instruction = f.read()
            data_pack = {}
            data_pack["dataset_name"] = "PIZZADATASET"
            data_pack['action'] = [actions[self.indx[idx]]]
            data_pack["observation"] = {}
            data_pack["observation"]["image_primary"] = [frames[self.indx[idx]]]
            data_pack["task"] = {}
            data_pack["task"]["language_instruction"] = instruction
        
        return self.batchTransform(data_pack)