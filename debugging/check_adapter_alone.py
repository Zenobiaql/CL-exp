import torch
from safetensors.torch import load_file


# 读取 safetensors 文件
file_path = '/home/v-qilinzhang/runs/0306-2113-Test/google_robot_close_bottom_drawer/raw_adapter/epoch0/adapter_model.safetensors'
tensors = load_file(file_path)

# 打印读取的张量
with open("/home/v-qilinzhang/CL-exp/debugging/tracker0307.txt", 'w') as f:
    for key, tensor in tensors.items():
        print(f"Tensor name: {key}, Tensor shape: {tensor.shape}")
        f.write(f"Tensor name: {key}, Tensor shape: {tensor.shape}\n")


