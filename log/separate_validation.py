import os
import draccus
import tqdm
import logging
import time
from pathlib import Path
import torch.distributed as dist


# Logger setup
def get_logger(name, file_path):
    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file_path is not None: 
        logger.addHandler(f_handler)
    
    return logger


class ModelLogger():
    def __init__(self, vla_path, dataset_name, batch_size, learning_rate, run_root_path):
        current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        self.logger = get_logger(f"log{dataset_name}", os.path.join(run_root_path, f"time{current_time}.log"))
        self.val_logger = get_logger(f"val_log{dataset_name}", os.path.join(run_root_path, f"time{current_time}-validation.log"))

        if dist.get_rank() == 0:      
            self.logger.info(f"Fine-tuning OpenVLA Model `{vla_path}` on `{dataset_name}`")
            self.logger.info(f"Training setting batch size {batch_size}, learning rate {learning_rate}")
            self.val_logger.info(f"Fine-tuning OpenVLA Model `{vla_path}` on `{dataset_name}`")
        
    def log_train_step(self, syn_smoothened_loss, syn_smoothened_action_accuracy, syn_smoothened_l1_loss, gradient_step_idx):
        if dist.get_rank() == 0:
            self.logger.info(f"train_loss: {syn_smoothened_loss:.4f}, action_accuracy: {syn_smoothened_action_accuracy:.4f}, l1_loss: {syn_smoothened_l1_loss:.4f}, step: {gradient_step_idx}")
            
    def log_val_step(self, val_dataset_name, mul_avg_loss, mul_avg_accuracy, mul_avg_action_l1_loss):
        if dist.get_rank() == 0:
            self.val_logger.info(f"On dataset {val_dataset_name}, Loss:{mul_avg_loss:.4f}, Accuracy:{mul_avg_accuracy:.4f}, L1 Loss:{mul_avg_action_l1_loss:.4f}.")
    
    def log_val_start(self, epoch):
        if dist.get_rank() == 0:
            self.val_logger.info(f"Validation after epoch{epoch}:")
    
    def log_val_finish(self):
        if dist.get_rank() == 0:
            self.val_logger.info("Validation finished.")
            self.val_logger.info("")
            
    def log_checkpoint_saved(self, epoch):
        if dist.get_rank() == 0:
            self.logger.info(f"Model Checkpoint for epoch {epoch} saved.")
            self.logger.info("")
            self.val_logger.info(f"Model Checkpoint for epoch {epoch} saved.")
            self.val_logger.info("")
            
    def log_training_config(self, config):
        if dist.get_rank() == 0:
            self.logger.info(f"Training config: {config}")
            self.val_logger.info(f"Training config: {config}")
            

class ModuleTracker():
    def __init__(self, trainable_param, run_root_path):
        self.run_root_path = run_root_path
        self.trainable_params = trainable_param
        
        
    def trainable_module(self):
        
        module_tracker_path = os.path.join(self.run_root_path, "trainable_modules.txt")
        
        if dist.get_rank() == 0:
            
            with open(module_tracker_path, "w") as file:
                num_params = 0
                num_modules = 0
                for param in self.trainable_params:
                    num_params += param.numel()
                    num_modules += 1
                    file.write(f"module_name: {param}\n")
                    file.write(f"module_param: {param.numel()}\n")
                file.write(f"---------------------\n")              
                file.write(f"num_params: {num_params}\n")
                file.write(f"num_modules: {num_modules}\n")


class SandboxModelLogger():
    def __init__(self, vla_path, dataset_name, batch_size, learning_rate, run_root_path):
        current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        self.logger = get_logger(f"log{dataset_name}", os.path.join(run_root_path, f"time{current_time}.log"))
        self.val_logger = get_logger(f"val_log{dataset_name}", os.path.join(run_root_path, f"time{current_time}-validation.log"))
      
        self.logger.info(f"Fine-tuning OpenVLA Model `{vla_path}` on `{dataset_name}`")
        self.logger.info(f"Training setting batch size {batch_size}, learning rate {learning_rate}")
        self.val_logger.info(f"Fine-tuning OpenVLA Model `{vla_path}` on `{dataset_name}`")
        
    def log_train_step(self, smoothened_loss, smoothened_action_accuracy, smoothened_l1_loss, gradient_step_idx):
        self.logger.info(f"train_loss: {smoothened_loss:.4f}, action_accuracy: {smoothened_action_accuracy:.4f}, l1_loss: {smoothened_l1_loss:.4f}, step: {gradient_step_idx}")
            
    def log_val_step(self, val_dataset_name, avg_loss, avg_accuracy, avg_action_l1_loss, epoch):
        self.val_logger.info(f"On dataset {val_dataset_name}, Loss:{avg_loss:.4f}, Accuracy:{avg_accuracy:.4f}, L1 Loss:{avg_action_l1_loss:.4f}, at epoch {epoch}.")
    
    def log_val_start(self, epoch):
        self.val_logger.info(f"Validation after epoch{epoch}:")
    
    def log_val_finish(self):
        self.val_logger.info("Validation finished.")
        self.val_logger.info("")
            
    def log_checkpoint_saved(self, epoch):
        self.logger.info(f"Model Checkpoint for epoch {epoch} saved.")
        self.logger.info("")
        self.val_logger.info(f"Model Checkpoint for epoch {epoch} saved.")
        self.val_logger.info("")
            
    def log_training_config(self, config):
        self.logger.info(f"Training config: {config}")
        self.val_logger.info(f"Training config: {config}")
            

class SandboxModuleTracker():
    def __init__(self, trainable_param, run_root_path):
        self.run_root_path = run_root_path
        self.trainable_params = trainable_param
        
        
    def trainable_module(self):
        
        module_tracker_path = os.path.join(self.run_root_path, "trainable_modules.txt")
            
        with open(module_tracker_path, "w") as file:
            num_params = 0
            num_modules = 0
            for param in self.trainable_params:
                num_params += param.numel()
                num_modules += 1              
            file.write(f"num_params: {num_params}\n")
            file.write(f"num_modules: {num_modules}\n")
        
    
