"""
Safetensors file contains LoRA tensors.
Doing SVD on each tensor,the U matrix would show some structure information of the tensor.
Comparing consine similarity of U matrix between two epochs.
The cosine similarity of U matrix between two epochs contains information of how similar 
u vectors are and how much their order changes.
"""

import os
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD


# function to calculate cosine similarity
def cosine_similarity(tensor1, tensor2):
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

# Compare tensors from two safetensors files
def compare_tensors(tensors_0, tensors_1):
    svd = TruncatedSVD(n_components=16, algorithm='arpack')
    vision_average_cosine_sim = 0
    other_average_cosine_sim = 0
    num_vision_layers = 0
    key_pairs = []
    for key in tqdm(tensors_0.keys()):
        if "lora" in key and "patch" not in key:
            parts = key.rsplit('.lora', 1)
            #print(f"key: {key}, parts: {parts}")
            new_key = parts[0]
            if new_key not in key_pairs:
                key_pairs.append(new_key)
    for key in tqdm(key_pairs):
        # vision module has "vision" in its name
        if "vision" in key:
            num_vision_layers += 1
            tensor_0 = (torch.t(tensors_0[key+".lora_A.weight"])@torch.t(tensors_0[key+".lora_B.weight"])).to(torch.float32).numpy()
            tensor_1 = (torch.t(tensors_1[key+".lora_A.weight"])@torch.t(tensors_1[key+".lora_B.weight"])).to(torch.float32).numpy()
            #print(f"tensor_0: {tensor_0.shape}, tensor_1: {tensor_1.shape}")
            components_1 = svd.fit_transform(tensor_0)
            components_2 = svd.fit_transform(tensor_1)
            cosine_sim = cosine_similarity(torch.tensor(components_1), torch.tensor(components_2))
            vision_average_cosine_sim += cosine_sim
        else:
            tensor_0 = (torch.t(tensors_0[key+".lora_A.weight"])@torch.t(tensors_0[key+".lora_B.weight"])).to(torch.float32).numpy()
            tensor_1 = (torch.t(tensors_1[key+".lora_A.weight"])@torch.t(tensors_1[key+".lora_B.weight"])).to(torch.float32).numpy()
            #print(f"tensor_0: {tensor_0.shape}, tensor_1: {tensor_1.shape}")
            components_1 = svd.fit_transform(tensor_0)
            components_2 = svd.fit_transform(tensor_1)
            cosine_sim = cosine_similarity(torch.tensor(components_1), torch.tensor(components_2))
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
root = "/home/v-qilinzhang/pissa_slsqft"
vision_sim = []
other_sim = []

for k in tqdm(range(6)):
    file_root = os.path.join(root, task[k], "raw_adapter")
    if k < 5:
        new_file_root = os.path.join(root, task[k+1], "raw_adapter")
    for i in tqdm(range(24)):
        with open("tracker.txt", "a") as f:
            f.write(f"check epoch {i} in task {task[k]}\n")
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
plt.xlabel('Epochs')
plt.ylabel('Cosine Similarity')
for i in range(24, len(epochs), 25):
    plt.axvline(x=i, color='red', linestyle='--', linewidth=2)
plt.title('Cosine Similarity between Epochs, Single Initialized PiSSA')
plt.legend()
plt.grid(True)
plt.savefig('pissa_slsqft_long.png')
print("\nDone\n")