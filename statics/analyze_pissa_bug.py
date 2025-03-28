import os
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def cosine_similarity(tensor1, tensor2):
    
    # 确保张量被展平
    tensor1_flat = tensor1.contiguous().view(-1)
    tensor2_flat = tensor2.contiguous().view(-1)
    
    # 计算点积
    dot_product = torch.dot(tensor1_flat, tensor2_flat)
    
    # 计算张量的范数
    norm_tensor1 = torch.norm(tensor1_flat)
    norm_tensor2 = torch.norm(tensor2_flat)
    
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_tensor1 * norm_tensor2)
    
    return cosine_sim.item()

def load_safetensors(file_path):
    # Load the safetensors file
    with safe_open(file_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    return tensors
            
def compare_tensors(tensors_0, tensors_1):
    vision_average_cosine_sim = 0
    other_average_cosine_sim = 0
    num_vision_layers = 0
    for key in tqdm(tensors_0.keys()):
        if "vision" in key:
            num_vision_layers += 1
            tensor_0 = tensors_0[key].to(torch.float32)
            tensor_1 = tensors_1[key].to(torch.float32)
            u_1, s_1, vh_1 = torch.svd(tensor_0)
            u_2, s_2, vh_2 = torch.svd(tensor_1)
            cosine_sim = cosine_similarity(u_1, u_2)
            vision_average_cosine_sim += cosine_sim
        else:
            tensor_0 = tensors_0[key].to(torch.float32)
            tensor_1 = tensors_1[key].to(torch.float32)
            u_1, s_1, vh_1 = torch.svd(tensor_0)
            u_2, s_2, vh_2 = torch.svd(tensor_1)
            cosine_sim = cosine_similarity(u_1, u_2)
            other_average_cosine_sim += cosine_sim
    vision_average_cosine_sim = vision_average_cosine_sim / num_vision_layers
    other_average_cosine_sim /= (len(tensors_0)-num_vision_layers)
    return other_average_cosine_sim, vision_average_cosine_sim

# Example usage
task = [
    "google_robot_close_bottom_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_top_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_top_drawer",
]
root = "/home/v-qilinzhang/pissa_slsqft_withbug"
for t in task:
    file_root = os.path.join(root, t, "raw_adapter")
vision_sim = []
other_sim = []
for i in tqdm(range(24)):
    file_path_0 = os.path.join(file_root, f"epoch{i}/adapter_model.safetensors")
    file_path_1 = os.path.join(file_root, f"epoch{i+1}/adapter_model.safetensors")
    tensors_0 = load_safetensors(file_path_0)
    tensors_1 = load_safetensors(file_path_1)
    other_ave, vision_ave = compare_tensors(tensors_0, tensors_1)
    other_sim.append(other_ave)
    vision_sim.append(vision_ave)

epochs = list(range(24))  
plt.plot(epochs, other_sim, label="other", color='orange')
plt.plot(epochs, vision_sim, label="vision", color='blue')
plt.xlabel('Epoch')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity between Epochs')
plt.legend()
plt.grid(True)

# 保存图像文件
plt.savefig('cosine_similarity_plot_compare_2.png')

print("图像已保存为 'cosine_similarity_plot_compare_2.png'.")


