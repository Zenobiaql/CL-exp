import os
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
from typing import Any, Dict, List, Tuple


# compute cosine similarity
def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    tensor1_flat = tensor1.contiguous().view(-1)
    tensor2_flat = tensor2.contiguous().view(-1)
    dot_product = torch.dot(tensor1_flat, tensor2_flat)
    norm_tensor1 = torch.norm(tensor1_flat)
    norm_tensor2 = torch.norm(tensor2_flat)
    cosine_sim = dot_product / (norm_tensor1 * norm_tensor2)
    
    return cosine_sim.item()

# Load the safetensors file
def load_safetensors(file_path):
    with safe_open(file_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        
    return tensors

# check two ckpt average similarity            
def compare_tensors(tensors_0: Dict, tensors_1: Dict) -> Tuple[float, float]:
    vision_average_cosine_sim = 0
    other_average_cosine_sim = 0
    num_vision_layers = 0
    
    for key in tqdm(tensors_0.keys()):
        if "vision" in key:
            num_vision_layers += 1
            tensor_0 = tensors_0[key].to(torch.float32)
            tensor_1 = tensors_1[key].to(torch.float32)
            if "lora_A" in key:
                u_1, s_1, vh_1 = torch.svd(tensor_0)
                u_2, s_2, vh_2 = torch.svd(tensor_1)
                #print(f"vh_1: {vh_1.shape}, vh_2: {vh_2.shape}")
                cosine_sim = cosine_similarity(vh_1, vh_2)
                vision_average_cosine_sim += cosine_sim
            elif "lora_B" in key:
                u_1, s_1, vh_1 = torch.svd(tensor_0)
                u_2, s_2, vh_2 = torch.svd(tensor_1)
                #print(f"u_1: {u_1.shape}, u_2: {u_2.shape}")
                cosine_sim = cosine_similarity(u_1, u_2)
                vision_average_cosine_sim += cosine_sim
        else:
            tensor_0 = tensors_0[key].to(torch.float32)
            tensor_1 = tensors_1[key].to(torch.float32)
            if "lora_A" in key:
                u_1, s_1, vh_1 = torch.svd(tensor_0)
                u_2, s_2, vh_2 = torch.svd(tensor_1)
                #print(f"vh_1: {vh_1.shape}, vh_2: {vh_2.shape}")
                cosine_sim = cosine_similarity(vh_1, vh_2)
                other_average_cosine_sim += cosine_sim
            elif "lora_B" in key:
                u_1, s_1, vh_1 = torch.svd(tensor_0)
                u_2, s_2, vh_2 = torch.svd(tensor_1)
                #print(f"u_1: {u_1.shape}, u_2: {u_2.shape}")
                cosine_sim = cosine_similarity(u_1, u_2)
                other_average_cosine_sim += cosine_sim
                
    vision_average_cosine_sim = vision_average_cosine_sim / num_vision_layers
    other_average_cosine_sim /= (len(tensors_0)-num_vision_layers)
    
    return other_average_cosine_sim, vision_average_cosine_sim


task = [
    "google_robot_close_bottom_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_top_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_top_drawer",
]
root = "/home/v-qilinzhang/pissa_slsqft_withbug"
vision_sim = []
other_sim = []

for k in tqdm(range(6)):
    file_root = os.path.join(root, task[k])
    if k < 5:
        new_file_root = os.path.join(root, task[k+1])

    for i in tqdm(range(24)):
        file_path_0 = os.path.join(file_root, f"epoch{i}/adapter_model.safetensors")
        file_path_1 = os.path.join(file_root, f"epoch{i+1}/adapter_model.safetensors")
        tensors_0 = load_safetensors(file_path_0)
        tensors_1 = load_safetensors(file_path_1)
        other_ave, vision_ave = compare_tensors(tensors_0, tensors_1)
        other_sim.append(other_ave)
        vision_sim.append(vision_ave)
    
    if k < 5:
        file_path_old = os.path.join(file_root, f"epoch24/adapter_model.safetensors")
        file_path_new = os.path.join(new_file_root, f"epoch0/adapter_model.safetensors")
        tensors_0 = load_safetensors(file_path_0)
        tensors_1 = load_safetensors(file_path_1)
        other_ave, vision_ave = compare_tensors(tensors_0, tensors_1)
        other_sim.append(other_ave)
        vision_sim.append(vision_ave)

epochs = list(range(len(vision_sim))) 
plt.plot(epochs, other_sim, label="other", color='orange')
plt.plot(epochs, vision_sim, label="vision", color='blue')
plt.xlabel('Epoch')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity between Epochs')
plt.legend()
#plt.grid(True)
for i in range(24, len(epochs), 25):
    plt.axvline(x=i, color='red', linestyle='--', linewidth=0.5)

# 保存图像文件
plt.savefig('test_new_line_new_pissa.png')

print("All Done")