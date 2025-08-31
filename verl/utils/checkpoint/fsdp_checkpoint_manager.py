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

import os
import warnings
import tempfile
from typing import Optional, Union

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.fs import copy_to_local, is_non_local, copy, makedirs, exists, is_tracto

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        dataloader=None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        # Store checkpoint_contents as instance variable rather than passing to parent
        self.checkpoint_contents = checkpoint_contents
        self.dataloader = dataloader
        
        # Initialize previous_saved_paths
        self.previous_saved_paths = []
        self.previous_global_step = 0
        
        # Call parent init without checkpoint_contents
        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
        )

    def load_checkpoint(self, path: str = None, local_path: str = None, hdfs_path: str = None, del_local_after_load=False, dataloader=None):
        """Load checkpoint from local or remote storage.
        
        Args:
            path: Path to checkpoint directory (for backward compatibility)
            local_path: Path to checkpoint directory
            hdfs_path: HDFS path (unused but kept for API compatibility)
            del_local_after_load: Whether to delete local files after loading
            dataloader: Dataloader to restore state to (overrides the one stored in the class)
        """
        # Handle both path and local_path parameters for backward compatibility
        if path is not None and local_path is None:
            local_path = path
        if local_path is None:
            return

        # Extract the component type (actor or critic) from the path or class name
        component_type = "actor"  # Default to actor
        if "critic" in self.__class__.__name__.lower() or "critic" in str(local_path).lower():
            component_type = "critic"
        
        # Use the same robust path detection logic as save_checkpoint
        path_parts = local_path.strip('/').split('/')
        contains_component = component_type in path_parts
        
        # Check if this is a YT path and component directory isn't already included
        if is_tracto(local_path) and not contains_component:
            component_path = os.path.join(local_path, component_type)
            print(f"[rank-{self.rank}]: Loading from YTsaurus component path {component_path}")
        else:
            component_path = local_path
            print(f"[rank-{self.rank}]: Loading from path {component_path}")
        
        # Every rank downloads its own checkpoint
        remote_model_path = os.path.join(component_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(component_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(component_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        # Use copy_to_local to handle both HDFS and YTsaurus paths
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)
        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(remote_model_path) else None
                os.remove(local_optim_path) if is_non_local(remote_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(remote_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")
        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        # Use dataloader from parameters if provided, otherwise use the one stored in the class
        dataloader_to_load = dataloader if dataloader is not None else self.dataloader
        # Load dataloader state if requested and it exists
        if "dataloader" in self.checkpoint_contents and dataloader_to_load is not None:
            remote_dataloader_path = os.path.join(component_path, "dataloader.pt")
            if exists(remote_dataloader_path):
                try:
                    local_dataloader_path = copy_to_local(remote_dataloader_path)
                    dataloader_state = torch.load(local_dataloader_path, weights_only=False)
                    print(f"[rank-{self.rank}]: Loading dataloader state from {remote_dataloader_path}")
                    # Apply state to dataloader
                    if hasattr(dataloader_to_load, "load_state_dict"):
                        dataloader_to_load.load_state_dict(dataloader_state)
                    elif hasattr(dataloader_to_load.sampler, "load_state_dict") and "sampler" in dataloader_state:
                        dataloader_to_load.sampler.load_state_dict(dataloader_state["sampler"])
                    else:
                        print(f"[rank-{self.rank}]: Dataloader or sampler has no load_state_dict method. State not restored.")
                    # Cleanup temporary file
                    if del_local_after_load and is_non_local(remote_dataloader_path):
                        os.remove(local_dataloader_path)
                except Exception as e:
                    print(f"[rank-{self.rank}]: Failed to load dataloader state: {e}")
        # Load tokenizer from huggingface directory with correct naming
        if ("tokenizer" in self.checkpoint_contents or "hf_model" in self.checkpoint_contents) and self.rank == 0:
            # Use the same path logic for HF directories as in save_checkpoint
            if contains_component:
                remote_hf_path = os.path.join(local_path, f"{component_type}_huggingface")
            else:
                # If component wasn't in path, it's at local_path/component_type/component_type_huggingface
                remote_hf_path = os.path.join(local_path, component_type, f"{component_type}_huggingface")
            
            if exists(remote_hf_path):
                try:
                    local_hf_path = copy_to_local(remote_hf_path)
                    print(f"[rank-{self.rank}]: Loading tokenizer from {local_hf_path}")
                    # Load the tokenizer if processing_class has from_pretrained
                    if hasattr(self.processing_class.__class__, "from_pretrained"):
                        loaded_tokenizer = self.processing_class.__class__.from_pretrained(local_hf_path)
                        # Update the current tokenizer with loaded state
                        for key, value in vars(loaded_tokenizer).items():
                            if not key.startswith("_"):
                                setattr(self.processing_class, key, value)
                        print(f"[rank-{self.rank}]: Successfully loaded tokenizer state")
                    # Cleanup temporary files
                    if del_local_after_load and is_non_local(remote_hf_path):
                        import shutil
                        shutil.rmtree(local_hf_path)
                except Exception as e:
                    print(f"[rank-{self.rank}]: Failed to load tokenizer: {e}")

    def save_checkpoint(self, local_path: str = None, path: str = None, hdfs_path: str = None,
                      global_step: int = 0, max_ckpt_to_keep=None, remove_previous_ckpt=None,
                      push_model=None, dataloader=None, dataloader_state=None, **kwargs):
        """Save checkpoint to local or remote storage.
        
        Args:
            local_path: Path to save checkpoint to
            path: Alternative path parameter (for backward compatibility)
            hdfs_path: HDFS path (unused but kept for API compatibility)
            global_step: Current training step
            max_ckpt_to_keep: Maximum number of checkpoints to keep
            remove_previous_ckpt: Whether to remove previous checkpoints (for backward compatibility)
            push_model: Whether to push model to hub
            dataloader: Dataloader instance to save state (if dataloader_state not provided)
            dataloader_state: Pre-computed dataloader state dictionary (preferred if available)
            **kwargs: Additional keyword arguments for future compatibility
        """
        # Handle both path and local_path parameters for backward compatibility
        if path is not None and local_path is None:
            local_path = path
        if local_path is None:
            return

        # For backward compatibility, use remove_previous_ckpt if provided
        if remove_previous_ckpt is not None and max_ckpt_to_keep is None:
            if remove_previous_ckpt:
                max_ckpt_to_keep = 1
            else:
                max_ckpt_to_keep = 0

        # record the previous global step
        self.previous_global_step = global_step

        # Remove previous checkpoints if needed
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        # Determine if saving to remote storage
        is_remote_save = is_tracto(local_path) or (hdfs_path is not None and hdfs_path != "")
        temp_local_path = None
        
        # Extract the component type (actor or critic) from the path or class name
        component_type = "actor"  # Default to actor
        if "critic" in self.__class__.__name__.lower() or ("critic" in str(local_path).lower() if local_path else False):
            component_type = "critic"
        
        # Check if local_path already contains the component name to avoid duplication
        path_parts = local_path.strip('/').split('/') if local_path else [] 
        contains_component = component_type in path_parts
        
        if is_remote_save and is_tracto(local_path):
            # Create a temporary local directory for saving before uploading to YTsaurus
            temp_local_path = os.path.join(tempfile.gettempdir(), f"ckpt_temp_{global_step}_{component_type}_{self.rank}")
            os.makedirs(temp_local_path, exist_ok=True)
            actual_save_path = temp_local_path
            
            # Determine the correct component directory for YT
            if contains_component:
                remote_component_path = local_path
            else:
                remote_component_path = os.path.join(local_path, component_type)
            
            # Create root checkpoint directory if saving remotely
            base_yt_path = os.path.dirname(remote_component_path)
            if base_yt_path and base_yt_path != "//":
                 makedirs(base_yt_path, exist_ok=True) 
            
            # Create component directory within the step directory
            makedirs(remote_component_path, exist_ok=True)
        else:
            # For local saving, check if we need to create component subdirectory
            if contains_component:
                actual_save_path = self.local_mkdir(local_path)
            else:
                component_dir = os.path.join(local_path, component_type)
                actual_save_path = self.local_mkdir(component_dir)
        
        torch.distributed.barrier()
        # Every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(actual_save_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(actual_save_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(actual_save_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
                print(f"[rank-{self.rank}]: Saving {component_type} model to {os.path.abspath(model_path)}")
                print(f"[rank-{self.rank}]: Saving {component_type} optim to {os.path.abspath(optim_path)}")
                print(f"[rank-{self.rank}]: Saving {component_type} extra_state to {os.path.abspath(extra_path)}")
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)
                torch.save(extra_state_dict, extra_path)
        # Upload to YTsaurus if needed
        if is_remote_save and is_tracto(local_path):
            torch.distributed.barrier()  # Make sure all ranks finished saving locally
            
            # No need to create directories again, just use the paths
            remote_model_path = os.path.join(remote_component_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
            remote_optim_path = os.path.join(remote_component_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            remote_extra_path = os.path.join(remote_component_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            
            copy(model_path, remote_model_path)
            copy(optim_path, remote_optim_path)
            copy(extra_path, remote_extra_path)
            
        # Save dataloader state if it's in checkpoint_contents - only rank 0 saves this
        if "dataloader" in self.checkpoint_contents and self.rank == 0:
            # Prioritize pre-computed dataloader_state if provided
            final_dataloader_state = None
            if dataloader_state is not None:
                 final_dataloader_state = dataloader_state
                 print(f"[rank-{self.rank}]: Using pre-computed dataloader state.")
            else:
                # Fallback to getting state from dataloader instance
                dataloader_to_save = dataloader if dataloader is not None else self.dataloader
                if dataloader_to_save is not None:
                    print(f"[rank-{self.rank}]: Extracting state from dataloader instance.")
                    try:
                        # Extract dataloader state
                        if hasattr(dataloader_to_save, "state_dict"):
                            final_dataloader_state = dataloader_to_save.state_dict()
                        else:
                            # For standard PyTorch DataLoader, we need to save sampler state
                            sampler = dataloader_to_save.sampler
                            if hasattr(sampler, "state_dict"):
                                final_dataloader_state = {"sampler": sampler.state_dict()}
                            else:
                                print(f"[rank-{self.rank}]: Dataloader or sampler has no state_dict method. Saving minimal state.")
                                final_dataloader_state = {"_warning": "Limited state saved"}
                    except Exception as e:
                        print(f"[rank-{self.rank}]: Failed to extract dataloader state from instance: {e}")

            # Proceed only if we have a valid state dictionary
            if final_dataloader_state is not None:
                dataloader_path = os.path.join(actual_save_path, "dataloader.pt")
                print(f"[rank-{self.rank}]: Saving dataloader state to {os.path.abspath(dataloader_path)}")
                try:
                    torch.save(final_dataloader_state, dataloader_path)
                    # Upload to YTsaurus if needed
                    if is_remote_save and is_tracto(local_path):
                        remote_dataloader_path = os.path.join(remote_component_path, "dataloader.pt")
                        print(f"[rank-{self.rank}]: Uploading dataloader state to YTsaurus: {remote_dataloader_path}")
                        copy(dataloader_path, remote_dataloader_path)
                except Exception as e:
                    print(f"[rank-{self.rank}]: Failed to save dataloader state dict: {e}")
            else:
                 print(f"[rank-{self.rank}]: No valid dataloader state available to save.")

        # Handle saving HF model and tokenizer with proper naming
        if "hf_model" in self.checkpoint_contents or "tokenizer" in self.checkpoint_contents:
            # wait for everyone to dump to local
            torch.distributed.barrier()
            if self.rank == 0:
                # Use correct huggingface directory name
                hf_local_path = os.path.join(actual_save_path, f"{component_type}_huggingface")
                os.makedirs(hf_local_path, exist_ok=True)
                # Save model configuration
                if hasattr(self.model, "_fsdp_wrapped_module") and hasattr(self.model._fsdp_wrapped_module, "config"):
                    self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
                    print(f"[rank-{self.rank}]: Saved model config to {hf_local_path}")
                # Save tokenizer files
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(hf_local_path)
                    print(f"[rank-{self.rank}]: Saved tokenizer/processor to {hf_local_path}")
                # If saving to YTsaurus, also copy huggingface folder
                if is_remote_save and is_tracto(local_path):
                    # Create the huggingface directory in YT with correct naming
                    remote_hf_dir = os.path.join(local_path, f"{component_type}_huggingface")
                    makedirs(remote_hf_dir, exist_ok=True)
                    # Copy all files inside the huggingface directory
                    for hf_file in os.listdir(hf_local_path):
                        local_file_path = os.path.join(hf_local_path, hf_file)
                        remote_file_path = os.path.join(remote_hf_dir, hf_file)
                        print(f"[rank-{self.rank}]: Uploading {hf_file} to YTsaurus: {remote_file_path}")
                        copy(local_file_path, remote_file_path)
        # Create latest_checkpointed_iteration.txt if rank 0
        base_save_path = local_path

        if self.rank == 0:
            latest_iter_content = str(global_step)
            latest_iter_filename = "latest_checkpointed_iteration.txt"
            
            # Save locally first, then copy, similar to dataloader
            temp_latest_iter_path = None
            try:
                # Create in temp directory first
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp_file:
                    temp_latest_iter_path = tmp_file.name
                    tmp_file.write(latest_iter_content)
                
                # Determine final destination (should be at the root of the step folder, not component folder)
                final_latest_iter_path = os.path.join(base_save_path, latest_iter_filename)
                
                print(f"[rank-{self.rank}]: Copying latest checkpoint info to: {final_latest_iter_path}")
                copy(temp_latest_iter_path, final_latest_iter_path)

            except Exception as e:
                 print(f"[rank-{self.rank}]: Failed to save/copy latest_checkpointed_iteration.txt: {e}")
            finally:
                 if temp_latest_iter_path and os.path.exists(temp_latest_iter_path):
                     os.remove(temp_latest_iter_path)

        torch.distributed.barrier()
        # Clean up temporary local path if needed
        if temp_local_path is not None:
            import shutil
            try:
                shutil.rmtree(temp_local_path)
            except Exception as e:
                print(f"[rank-{self.rank}]: Failed to clean up temporary checkpoint directory: {e}")

        self.previous_saved_paths.append(local_path)
