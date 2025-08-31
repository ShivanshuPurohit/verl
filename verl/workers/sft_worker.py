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
SFT Worker for Ray-based training
"""

from datetime import timedelta
import os
import logging
import re
from contextlib import nullcontext
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


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


class SFTWorker(Worker):
    """Worker for SFT training using FSDP"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Loads in environment variables from central location - passing in via the driver process doesn't work.
        local_env_path = copy_to_local(config.trainer.get("env", ".env"))
        from dotenv import load_dotenv
        load_dotenv(local_env_path, override=True)

        if "HF_TOKEN" in os.environ:
            from huggingface_hub import login
            login(os.environ["HF_TOKEN"])
        
        # Initialize process group before attempting to get world size
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(timeout=timedelta(seconds=3600))

        
        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
        
        # Setup for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        
        # Only initialize sequence parallel if requested and ensure synchronization
        if self.ulysses_sequence_parallel_size > 1:
            # Make sure all processes have reached this point
            torch.distributed.barrier()
            
            # Calculate DP size safely
            dp_size = world_size // self.ulysses_sequence_parallel_size
            assert dp_size * self.ulysses_sequence_parallel_size == world_size, \
                f"World size {world_size} must be divisible by SP size {self.ulysses_sequence_parallel_size}"
            
            # Create the device mesh with explicit device IDs to avoid conflicts
            rank = torch.distributed.get_rank()
            local_rank = torch.cuda.current_device()
            
            # Initialize ulysses device mesh
            self.ulysses_device_mesh = init_device_mesh(
                device_type='cuda',
                mesh_shape=(dp_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=('dp', 'sp')
            )
            
            # Initialize sequence parallel group
            set_ulysses_sequence_parallel_group(self.ulysses_device_mesh)
            
            # Verify initialization successful
            if rank == 0:
                print(f"Ulysses sequence parallel initialized with size {self.ulysses_sequence_parallel_size}")
                print(f"DP size: {dp_size}, Total world size: {world_size}")
            
            # Ensure all processes have completed initialization
            torch.distributed.barrier()
        else:
            if torch.distributed.get_rank() == 0:
                print("Ulysses sequence parallel disabled")
        
        # Create sharding manager after device mesh is properly initialized
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh) if self.ulysses_device_mesh else None
        
        self.model = None
        self.fsdp_model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tokenizer = None
        
        # HuggingFace Hub integration
        self.push_to_hub = self.config.trainer.get('push_to_hub', False)
        self.hub_model_id = self.config.trainer.get('hub_model_id', None)
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, total_steps):
        """Initialize the model, optimizer, and scheduler"""
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        train_batch_size = self.config.data.train_batch_size
        assert train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        if self.config.data.micro_batch_size_per_gpu is not None:
            assert self.config.data.micro_batch_size_per_gpu * self.ulysses_sequence_parallel_size >= n_gpus
        
        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)
            
        log_gpu_memory_usage('Before model allocation', logger=logger)
        
        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(config.model_type)
            
        if self.use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(config, verbose=True)
            
        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings,
                                                      mesh=self.device_mesh)
        
        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                              config=config,
                                                                              torch_dtype=torch.float32,
                                                                              attn_implementation='flash_attention_2',
                                                                              trust_remote_code=trust_remote_code)
            
            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)
                
            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            
        log_gpu_memory_usage('After model allocation', logger=logger)
        
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                        reduce_dtype=torch.float32,
                                        buffer_dtype=torch.float32)
        
        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                               config=self.config.model.fsdp_config.wrap_policy,
                                               is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.rank == 0:
            print(auto_wrap_policy)
            
        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
            
        self.fsdp_model = FSDP(module=self.model,
                              auto_wrap_policy=auto_wrap_policy,
                              param_init_fn=init_fn,
                              sharding_strategy=ShardingStrategy.FULL_SHARD,
                              mixed_precision=mixed_precision,
                              device_mesh=self.device_mesh,
                              sync_module_states=True,
                              device_id=torch.cuda.current_device(),
                              cpu_offload=cpu_offload,
                              use_orig_params=False,
                              forward_prefetch=False,
        )
        
        log_gpu_memory_usage('After FSDP wrapping', logger=logger)
        
        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                   lr=self.config.optim.lr,
                                   betas=self.config.optim.betas,
                                   weight_decay=self.config.optim.weight_decay)
        
        log_gpu_memory_usage('After initialize optimizer', logger=logger)
        self.total_steps = total_steps
        self.steps_per_epoch = self.total_steps // self.config.trainer.total_epochs
        
        if self.rank == 0:
            print(
                f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

            
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=num_warmup_steps,
                                                           num_training_steps=self.total_steps)
        
        # Build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        
        return True  # Indicate successful initialization
        
    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.ulysses_sequence_parallel_size > 1
        
        # Move inputs to GPU and prepare loss mask
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        position_ids = batch['position_ids'].cuda()
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp and self.sharding_manager is not None else nullcontext()
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids[:, 1:].contiguous()
                    output = self.fsdp_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)
                    logits = output.logits
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels.contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss * loss_mask.to(loss.device)
                else:
                    # Only use sequence parallel if ulysses_device_mesh is properly initialized
                    if self.ulysses_device_mesh is None:
                        raise RuntimeError("Sequence parallel was enabled but ulysses_device_mesh is None")
                        
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler
                    
                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                              attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                    
                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                         indices).transpose(0, 1)
                    
                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                    
                    # Forward pass
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        use_cache=False)
                    
                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    # Gather and unpad for sequence parallelism
                    loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    
                    # This is the loss collected from all ulysses ranks
                    full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                    full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    full_loss = full_loss.reshape(-1)
                    loss_mask = loss_mask.to(full_loss.device)
                    loss = full_loss * loss_mask
                    
                valid_token_this_rank = torch.sum(loss_mask)
                
                if self.config.data.balance_dp_token:
                    torch.distributed.all_reduce(valid_token_this_rank)
                    dp_size = self.ulysses_device_mesh.size('dp') if use_sp and self.ulysses_device_mesh is not None else torch.distributed.get_world_size()
                else:
                    dp_size = 1
                    
                loss = torch.sum(loss) / valid_token_this_rank * dp_size
                
                if do_backward:
                    loss.backward()
                return loss
                
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def training_step(self, batch: DataProto):
        """Execute a single training step"""
        self.fsdp_model.train()
        
        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)
        
        self.optimizer.zero_grad()
        
        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        if self.ulysses_sequence_parallel_size > 1:
            assert isinstance(self.sharding_manager, FSDPUlyssesShardingManager)
            batch = self.sharding_manager.preprocess_data(data=batch)

        # Check if batch is DataProto and extract batch if it is
        if isinstance(batch, DataProto):
            # Extract the TensorDict from DataProto
            batch = batch.batch
        
        # Move data to GPU if needed
        # Use the actual batch size from the data instead of from config
        actual_batch_size = batch.shape[0] if hasattr(batch, 'shape') else len(batch)
        batch = TensorDict(batch, batch_size=[actual_batch_size]).cuda()
        
        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()
            
        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        
        log_gpu_memory_usage('Before optimizer step', logger=logger)
        
        self.optimizer.step()
        
        log_gpu_memory_usage('After optimizer step', logger=logger)
        
        self.lr_scheduler.step()
        
        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]
        
        log_gpu_memory_usage('After offload weights', logger=logger)
        
        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        
        # Convert metrics dictionary to a DataProto object
        metrics = {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}
        metrics_tensor = TensorDict({
            'loss': torch.tensor([metrics['train/loss']]),
            'lr': torch.tensor([metrics['train/lr(1e-3)']])
        }, batch_size=[1])
        
        return DataProto(batch=metrics_tensor, meta_info=metrics)
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def validation_step(self, batch: DataProto):
        """Execute a single validation step"""
        self.fsdp_model.eval()
        
        if self.ulysses_sequence_parallel_size > 1:
            assert isinstance(self.sharding_manager, FSDPUlyssesShardingManager)
            batch = self.sharding_manager.preprocess_data(data=batch)

        # Check if batch is DataProto and extract batch if it is
        if isinstance(batch, DataProto):
            # Extract the TensorDict from DataProto
            batch = batch.batch

        # Move data to GPU if needed
        # Use the actual batch size from the data instead of from config
        actual_batch_size = batch.shape[0] if hasattr(batch, 'shape') else len(batch)
        batch = TensorDict(batch, batch_size=[actual_batch_size]).cuda()
        
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        
        # Convert scalar loss to a DataProto object
        val_loss = loss.item()
        metrics_tensor = TensorDict({
            'loss': torch.tensor([val_loss])
        }, batch_size=[1])
        
        return DataProto(batch=metrics_tensor, meta_info={'val/loss': val_loss})
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, step=0, remove_previous_ckpt=False):
        """Save model checkpoint
        
        Args:
            local_path: Path to save the checkpoint locally
            hdfs_path: Optional path to save the checkpoint to HDFS
            step: Current training step (used for naming in HF Hub)
            remove_previous_ckpt: Whether to remove the previous checkpoint
            push_model: Whether to push the model to the HuggingFace Hub
        """
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()
            
        # save huggingface model
        if self.rank == 0:
            os.makedirs(local_path, exist_ok=True)
            self.model.save_pretrained(local_path, state_dict=state_dict)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(local_path)
            if hdfs_path:
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path, dirs_exist_ok=True)
                
            # Push to HuggingFace Hub if enabled
            if self.push_to_hub:
                if not self.hub_model_id:
                    print("Warning: push_to_hub is enabled but hub_model_id is not set. Skipping push to hub.")
                else:
                    print(f"Pushing model to the HuggingFace Hub as {self.hub_model_id}")
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        if api.repo_exists(self.hub_model_id):
                            print(f"WARNING: repo {self.hub_model_id} already exists.")
                        api.create_repo(self.hub_model_id, private=True, exist_ok=True)
                        # Create a branch named after the step for versioning
                        api.create_branch(repo_id=self.hub_model_id, branch=str(step))
                        # Upload the model files to that branch
                        api.upload_folder(
                            repo_id=self.hub_model_id, 
                            folder_path=local_path,
                            repo_type="model",
                            revision=str(step)
                        )
                        
                        # The tokenizer is already saved to local_path by default in line 477
                        # so it will be included in the uploaded folder
                        print(f"Successfully pushed model to HuggingFace Hub")
                    except Exception as e:
                        print(f"Failed to push model to HuggingFace Hub. Error: {e}")
                
        torch.distributed.barrier()
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):
        """Load model checkpoint"""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        
        if self.rank == 0:
            print(f"Loading checkpoint from {path}")
            
        # Load model weights
        self.model.from_pretrained(path)
        torch.distributed.barrier()
        
        if del_local_after_load and self.rank == 0 and os.path.exists(path):
            import shutil
            shutil.rmtree(path) 
