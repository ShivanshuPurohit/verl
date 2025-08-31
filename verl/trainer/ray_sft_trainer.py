# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight Ray-based SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os
import logging
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Type, Dict

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict, pad
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_to_local
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto

from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

WorkerType = Type[Worker]

class Role(Enum):
    """Define roles for SFT training"""
    Trainer = 0

@dataclass
class ResourcePoolManager:
    """Manages resource pools for SFT training"""
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool for a role"""
        return self.resource_pool_dict[self.mapping[role]]

def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None

def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj

class RaySFTTrainer(object):
    """
    Ray-based SFT Trainer that runs training on distributed workers
    """

    def __init__(self, 
                config,
                tokenizer, 
                role_worker_mapping: dict[Role, WorkerType], 
                resource_pool_manager: ResourcePoolManager,
                ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        
        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        
        # normalize dp size
        self._normalize_config_bsz()

        # Create dataloader
        self._create_dataloader()

    def _normalize_config_bsz(self):
        """Normalize batch size configuration"""
        # Get total number of GPUs
        n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        
        if n_gpus > 0:
            print(f'Normalize batch size by dp {n_gpus}')
            assert self.config.data.train_batch_size % n_gpus == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {n_gpus}"
            self.config.data.train_batch_size //= n_gpus
            
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0, f"Train batch size {self.config.data.train_batch_size} not divisible by micro batch size {self.config.data.micro_batch_size_per_gpu}"

    def _create_dataloader(self):
        """Create training and validation dataloaders"""
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=self.config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        config=self.config.data)
        self.val_dataset = SFTDataset(parquet_files=self.config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      config=self.config.data)

        # build dataloader with appropriate sampler for distributed training
        # Use data parallel rank and size instead of global rank and world size
        rank = 0  # Will be set by DistributedSampler
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=8,
                                           pin_memory=True,
                                           drop_last=True)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=False,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_batch_size = self.config.data.micro_batch_size_per_gpu * self.config.ulysses_sequence_parallel_size
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.val_batch_size,
                                         sampler=self.val_sampler,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True)
        if len(self.val_dataloader) == 0 and len(self.val_dataset) > 0:
            print(f"Warning: Validation dataloader is empty despite having {len(self.val_dataset)} samples in the dataset. This is due to batch size ({self.config.data.micro_batch_size_per_gpu * world_size}) being larger than dataset size with drop_last=True.")
   
        # Set total steps for training
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            self.total_steps = self.config.trainer.total_training_steps
            
        print("Train dataset len:", len(self.train_dataset))
        print("Val dataset len:", len(self.val_dataset))
        print(f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}')
        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

    def init_workers(self):
        """Initialize Ray worker groups"""
        self.resource_pool_manager.create_resource_pool()
        
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        # Create trainer worker
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Trainer)
        trainer_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Trainer],
                                         config=self.config)
        self.resource_pool_to_cls[resource_pool]['trainer'] = trainer_cls
        
        # Initialize WorkerGroup
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print(f'resource_pool: {resource_pool}')
            print(f'class_dict: {class_dict}')
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            
            # Configure environment variables for distributed setup
            world_size = self.world_size = resource_pool.world_size
            print(f"Setting up worker group with world_size={world_size}")
            
            # Spawn worker
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # Keep the reference of WorkerDict to support ray >= 2.31
            self.wg_dicts.append(wg_dict)
            
        self.trainer_wg = all_wg['trainer']
        
        # Initialize model - this will trigger the distributed setup
        print("Initializing model on worker group...")
        self.trainer_wg.init_model(total_steps=self.total_steps)

    def _save_checkpoint(self):
        """Save checkpoint
        
        Args:
            push_model: Whether to push the model to HuggingFace Hub
        """
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, 
                                              f'global_step_{self.global_steps}')
        trainer_local_path = os.path.join(local_global_step_folder, 'trainer')
        
        trainer_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'trainer')
        
        self.trainer_wg.save_checkpoint(trainer_local_path,
                                      trainer_remote_path,
                                      self.global_steps,
                                      remove_previous_ckpt=self.config.trainer.get('remove_previous_ckpt_in_save', False))
        
        # Save dataloader
        os.makedirs(local_global_step_folder, exist_ok=True)
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        
        # Latest checkpointed iteration tracker
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                         'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """Load checkpoint"""
        if self.config.trainer.resume_mode == 'disable':
            return 0
            
        # Find checkpoint folder
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        
        # Handle resume modes
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        elif self.config.trainer.resume_mode == 'resume_from_path':
            assert self.config.trainer.resume_from_path is not None, "resume_from_path is not set"
            global_step_folder = self.config.trainer.resume_from_path            
            self.global_steps = int(global_step_folder.split('global_step_')[-1])
            print(f'Load from checkpoint folder: {global_step_folder}')
       
        trainer_path = os.path.join(global_step_folder, 'trainer')
        self.trainer_wg.load_checkpoint(trainer_path,
                                      del_local_after_load=self.config.trainer.get('del_local_ckpt_after_load', False))
        
        # Load dataloader
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
            
        return self.global_steps

    def fit(self):
        """Train the model using Ray workers"""
        # Initialize tracking
        tracking = Tracking(project_name=self.config.trainer.project_name,
                            experiment_name=self.config.trainer.experiment_name,
                            default_backend=self.config.trainer.logger)

        self.global_steps = 0
        
        # Load checkpoint if needed
        self._load_checkpoint()
        
        # Training loop
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(self.train_dataloader,
                            total=self.steps_per_epoch,
                            desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}"):
                self.global_steps += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size)
                
                # Send data to trainer workers
                # Convert TensorDict to DataProto before passing to trainer
                from verl import DataProto
                data_proto = DataProto.from_tensordict(data)
                metrics_proto = self.trainer_wg.training_step(data_proto)
                
                # Extract metrics from DataProto's meta_info field
                metrics = metrics_proto.meta_info
                
                # Log metrics
                tracking.log(data=metrics, step=self.global_steps)

                # Validation
                if self.config.trainer.val_freq > 0 and self.global_steps % self.config.trainer.val_freq == 0:
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.val_dataloader.batch_size)
                        
                        # Convert TensorDict to DataProto for validation as well
                        val_data_proto = DataProto.from_tensordict(val_data)
                        val_loss_proto = self.trainer_wg.validation_step(val_data_proto)
                        # Extract loss value from DataProto
                        val_loss = val_loss_proto.meta_info['val/loss']
                        val_losses.append(val_loss)
                    
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    val_metrics = {'val/loss': avg_val_loss}
                    tracking.log(data=val_metrics, step=self.global_steps)
                
                # Save checkpoint
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    # Only push to hub on specific intervals if configured
                    self._save_checkpoint()
                    
                # Early exit check
                if self.global_steps >= self.total_steps:
                    # Final validation and checkpoint
                    if self.config.trainer.val_freq > 0:
                        val_losses = []
                        for val_data in self.val_dataloader:
                            val_data = TensorDict(val_data, batch_size=self.val_dataloader.batch_size)
                            
                            # Convert TensorDict to DataProto for validation as well
                            val_data_proto = DataProto.from_tensordict(val_data)
                            val_loss_proto = self.trainer_wg.validation_step(val_data_proto)
                            # Extract loss value from DataProto
                            val_loss = val_loss_proto.meta_info['val/loss']
                            val_losses.append(val_loss)
                        
                        avg_val_loss = sum(val_losses) / len(val_losses)
                        val_metrics = {'val/loss': avg_val_loss}
                        tracking.log(data=val_metrics, step=self.global_steps)
                    
                    # Always push to hub for the final checkpoint if configured
                    self._save_checkpoint()
                    return


# Update the main function to use Ray
import hydra
import ray

@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    # Import the SFT worker implementation
    from verl.workers.sft_worker import SFTWorker
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(runtime_env={ 'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })
    
    # Define role-worker mapping
    role_worker_mapping = {
        Role.Trainer: SFTWorker
    }
    
    # Define resource pool manager
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={"trainer": config.trainer.process_on_nodes},
        mapping={Role.Trainer: "trainer"}
    )
    
    # Create tokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=config.model.trust_remote_code)
    
    # Initialize trainer
    trainer = RaySFTTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager
    )
    
    # Initialize workers
    trainer.init_workers()
    
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
