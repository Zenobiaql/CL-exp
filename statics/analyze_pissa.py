import os
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.contiguous().view(-1)
    tensor2_flat = tensor2.contiguous().view(-1)
    dot_product = torch.dot(tensor1_flat, tensor2_flat)
    norm_tensor1 = torch.norm(tensor1_flat)
    norm_tensor2 = torch.norm(tensor2_flat)
    cosine_sim = dot_product / (norm_tensor1 * norm_tensor2)
    
    return cosine_sim.item()

def load_safetensors(file_path):
    with safe_open(file_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    return tensors
            
def compare_tensors(tensors_0, tensors_1):
    average_cosine_sim = 0
    for key in tqdm(tensors_0.keys()):
            tensor_0 = tensors_0[key].to(torch.float32)
            tensor_1 = tensors_1[key].to(torch.float32)
            u_1, s_1, vh_1 = torch.linalg.svd(tensor_0)
            u_2, s_2, vh_2 = torch.svd(tensor_1)
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
root = "/home/v-qilinzhang/pissa_slsqft"
vision_sim = []
other_sim = []

for k in tqdm(range(6)):
    file_root = os.path.join(root, task[k], "raw_adapter")
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
plt.grid(True)

# 保存图像文件
plt.savefig('cosine_similarity_plot_compare_new_pissa.png')

print("图像已保存为 'cosine_similarity_plot_compare_new_pissa.png'.")


