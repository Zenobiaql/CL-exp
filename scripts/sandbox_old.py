"""
script for finetuning on each task separately, and check accuracy on all tasks
"""

import os
import draccus
import tqdm
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.utils.data.distributed import DistributedSampler

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from dataclasses import dataclass
from collections import deque

from dataset import PizzaDataset
import random
from typing import List, Union
from log import ModelLogger, ModuleTracker

from copy import deepcopy
from multiprocessing import cpu_count
import numpy as np


"""
Finetuning configuration, passed with .yaml or .yml file
for reference
"""

class FinetuneConfig:
    
    # model path
    vla_path: str

    "Data Configuration"
    # dataset root directory       
    data_dir: str
    # name of dataset class                        
    dataset_name: str
    # directory for log and checkpoints                            
    run_root_dir: str
    num_workers: int                                    

    "Finetuning Configuration"
    epochs: int                                                
    batch_size: int                                                                                        
    save_steps: int                                            
    learning_rate: float                                       
    grad_accumulation_steps: int                               
    image_aug: bool                                         
                        
    "LoRA Arguments"
    use_lora: bool                                        
    lora_rank: int                                           
    lora_dropout: float
    lora_module: Union[List[str], str]

    
"""
Model training and validation class, including logging 
"""                                                                                    
                                                                      
# Model training class, adapted for different task and log/file settings    
class Model:
    def __init__(
        self,

        # training loop settings 
        epochs,
        batch_size,
        grad_accumulation_steps,

        # log/file/directory settings
        run_dir,
        logger_complex,

        # model training settings
        optimizer, 
        vla,
        vla_path,
        processor,
        action_tokenizer,
        use_lora,

        # data loader settings
        dataloader,
        val_dataloader_set,
        task_id,
        
        # device settings
        device_id,
        
    ):
        self.epochs = epochs
        self.optimizer = optimizer
        self.vla = vla
        self.dataloader = dataloader
        self.val_dataloader_set = val_dataloader_set
        self.grad_accumulation_steps = grad_accumulation_steps
        self.action_tokenizer = action_tokenizer
        self.run_dir = run_dir
        self.logger_complex = logger_complex
        self.use_lora = use_lora
        self.processor = processor
        self.task_id = task_id
        self.vla_path = vla_path
        self.batch_size = batch_size
        self.device_id = device_id
    
    # val_dataloader_set should be a dictionary of dataloaders of all tasks to be validated
    def _validate_step(self, num_epoch):
        self.vla.eval()
        
        with torch.no_grad():
            for val_dataset_name, val_dataloader in self.val_dataloader_set.items():
                    
                total_loss = 0
                total_accuracy = 0
                total_l1_loss = 0
                    
                for batch_idx, batch in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"{val_dataset_name}"):
                    output: CausalLMOutputWithPast = self.vla(
                        input_ids=batch["input_ids"].to(self.device_id),
                        attention_mask=batch["attention_mask"].to(self.device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss
                    total_loss += loss.item()
                    
                    action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                    action_preds = action_logits.argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = action_gt > self.action_tokenizer.action_token_begin_idx
                    correct_preds = (action_preds == action_gt) & mask
                    action_accuracy = correct_preds.sum().float() / mask.sum().float()
                    total_accuracy += action_accuracy.item()

                    continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    )
                    continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                    total_l1_loss += action_l1_loss.item()
                    
                avg_loss = total_loss / len(val_dataloader)
                avg_accuracy = total_accuracy / len(val_dataloader)
                avg_action_l1_loss = total_l1_loss / len(val_dataloader)

                if dist.get_rank() == 0:
                    self.logger_complex.log_val_step(val_dataset_name, avg_loss, avg_accuracy, avg_action_l1_loss)
                    
            if dist.get_rank() == 0:
                self.logger_complex.log_val_finish()


    # training loop, comprised of training and validation steps
    # validate after each training epoch, on all tasks
    def train_step(self):
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.vla.train()    
            self.optimizer.zero_grad()

            for batch_idx, batch in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = self.vla(
                        input_ids=batch["input_ids"].to(self.device_id),
                        attention_mask=batch["attention_mask"].to(self.device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > self.action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                recent_losses = deque(maxlen=self.grad_accumulation_steps)
                recent_action_accuracies = deque(maxlen=self.grad_accumulation_steps)
                recent_l1_losses = deque(maxlen=self.grad_accumulation_steps)

                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                gradient_step_idx = batch_idx // self.grad_accumulation_steps

                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if batch_idx % 10 == 0:    
                    if dist.get_rank() == 0:
                        self.logger_complex.log_train_step(smoothened_loss, smoothened_action_accuracy, smoothened_l1_loss, gradient_step_idx)
            
            if dist.get_rank() == 0:
                self.logger_complex.log_val_start(epoch)

            # validate after each epoch
            self._validate_step(epoch)

            if self.use_lora:
                #if dist.get_rank() == 0:
                    self.logger_complex.log_checkpoint_saved(epoch)
                    save_dir = self.run_dir
                    self.vla.save_pretrained(os.path.join(save_dir, f'epoch{epoch}'))

            else:
                ValueError("LoRA not used, please check configuration.")       
                

"""
FinetuneConfig finetune() uses is in run_code.py, FinetuneConfig above is just for reference
"""          

def finetune(cfg: FinetuneConfig)->None:
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK'] = '0'
    dist.init_process_group("nccl", rank=0, world_size=1)
    
    if dist.get_rank() == 0:
        os.makedirs(cfg.run_root_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    
    dataloader_set = {}
    val_dataloader_set = {}
    data_root_dir = Path(cfg.data_dir)
            
    # training set for current task
    for sub_dir in data_root_dir.iterdir():
        if sub_dir.is_dir():
        # current task dataset
            task_data = PizzaDataset(
                sub_dir,
                action_tokenizer,
                processor.tokenizer,
                processor.image_processor.apply_transform,
                prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
            )
            
            index = [i for i in range(len(task_data))]
            #random.shuffle(index)
            val_index = index[:len(index)//10]
            train_index = index[len(index)//10:]
            print(len(val_index))
            print(len(train_index))
                    
            dataloader = DataLoader(
                Subset(task_data, train_index),
                batch_size=cfg.batch_size,
                sampler=DistributedSampler(Subset(task_data, train_index)),
                collate_fn=collator,
                num_workers=cfg.num_workers,
            )

            val_dataloader = DataLoader(
                Subset(task_data, val_index),
                batch_size=cfg.batch_size,
                sampler=DistributedSampler(Subset(task_data, val_index)),
                collate_fn=collator,
                num_workers=cfg.num_workers,
            )
            
            print(len(val_dataloader))
            print(len(dataloader))
            
            dataloader_set[sub_dir.name] = dataloader
            val_dataloader_set[sub_dir.name] = val_dataloader
        
    for task_name, dataloader in dataloader_set.items():
        task_sub_dir = os.path.join(cfg.run_root_dir, task_name)
        os.makedirs(task_sub_dir, exist_ok=True)
            
        logger_complex = ModelLogger(cfg.vla_path, task_name, cfg.batch_size, cfg.learning_rate, task_sub_dir)
        
        # get initialized vla
        init_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_module,
                init_lora_weights="gaussian",
            )
            init_vla = get_peft_model(init_vla, lora_config)
            init_vla.print_trainable_parameters()
            
        device_id = int(os.environ["LOCAL_RANK"])
        init_vla = init_vla.to(device_id)

        trainable_params = [param for param in init_vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
        
        module_tracker = ModuleTracker(trainable_params, task_sub_dir)
        module_tracker.trainable_module()
        
        exp_id = (
            f"{data_root_dir.name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
                
        logger_complex.log_training_config(exp_id)
    
        model_train = Model(
            cfg.epochs, 
            cfg.batch_size,
            cfg.grad_accumulation_steps,
                        
            task_sub_dir,
            logger_complex,

            optimizer, 
            init_vla,
            cfg.vla_path,
            processor,
            action_tokenizer,
            cfg.use_lora,

            dataloader,
            val_dataloader_set,
            data_root_dir.name,
                        
            device_id,
        )

        model_train.train_step()


    dist.destroy_process_group()
    print(f"Finished running code on rank {device_id}.")

if __name__ == "__main__":
    finetune()
    
    