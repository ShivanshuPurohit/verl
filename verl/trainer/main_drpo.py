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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import os
import hydra

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer_async import RayTrainerAsync


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='drpo_trainer', version_base=None)
def main(config):
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        from verl.utils.fs import copy_to_local

        # Load in environment variables from central location
        local_env_path = copy_to_local(config.trainer.get("env", ".env"))
        from dotenv import load_dotenv
        load_dotenv(local_env_path, override=True)
        # this is for local ray cluster
        env_vars = {'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_SOCKET_IFNAME': 'eth0',
            'NCCL_IB_SL': '1',
            'NCCL_IB_HCA': 'mlx5',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
            'NCCL_CUMEM_ENABLE': '0',
            'TP_SOCKET_IFNAME': 'eth0',
            'GLOO_SOCKET_IFNAME': 'eth0',
            'CUDA_LAUNCH_BLOCKING': '1',
            'VLLM_ATTENTION_BACKEND': 'XFORMERS'
                    }
        if config.trainer.nnodes == 1:
            env_vars.pop('NCCL_SOCKET_IFNAME')
            env_vars.pop('GLOO_SOCKET_IFNAME')
            env_vars.pop('TP_SOCKET_IFNAME')

        # check for HF_TOkEN or WANDB_API_KEY, if they exist in the driver process, add them to the env_vars
        if 'HF_TOKEN' in os.environ:
            env_vars['HF_TOKEN'] = os.environ['HF_TOKEN']
        if 'WANDB_API_KEY' in os.environ:
            env_vars['WANDB_API_KEY'] = os.environ['WANDB_API_KEY']
        if 'YT_TOKEN' in os.environ:
            env_vars['YT_TOKEN'] = os.environ['YT_TOKEN']
        ray.init(runtime_env={'env_vars': env_vars})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from verl.utils.fs import copy_to_local
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

        # monkey patch the update methods to be non-blocking
        from verl.single_controller.base.decorator import MAGIC_ATTR
        getattr(ActorRolloutRefWorker.update_actor, MAGIC_ATTR)['blocking'] = False
        getattr(CriticWorker.update_critic, MAGIC_ATTR)['blocking'] = False

        if config.actor_rollout_ref.rollout.async_gen:
            # Make generation non-blocking if doing async generation
            getattr(ActorRolloutRefWorker.generate_sequences, MAGIC_ATTR)['blocking'] = False

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.Actor: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        Role.Rollout: ray.remote(ActorRolloutRefWorker),
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }

    # TODO (chase): extend this allocation to multinode
    # should be simple - allocate half the nodes to training and the other half to inference
    if config.actor_rollout_ref.rollout.async_gen:
        actor_ref_pool_id = 'actor_pool'
        rollout_pool_id = 'rollout_pool'

        nnodes = config.trainer.nnodes
        if nnodes > 1:
            # Allocate fraction of the GPUs for training and the other half for inference
            inference_nodes = config.trainer.get("inference_nodes")
            assert inference_nodes is not None, f"trainer.inference nodes is not defined, but trainer.nnodes={nnodes}"

            if nnodes - inference_nodes == 0:
                raise ValueError(f"The total number of nodes {config.trainer.nnodes} must be strictly greater than the number of inference nodes {inference_nodes}")

            # Split the nodes evenly between generation and training
            resource_pool_spec = {
                actor_ref_pool_id: [config.trainer.n_gpus_per_node] * (nnodes - inference_nodes),
                rollout_pool_id: [config.trainer.n_gpus_per_node] * inference_nodes,
            }
        else:
            resource_pool_spec = {
                actor_ref_pool_id: [config.trainer.n_gpus_per_node // 2],
                rollout_pool_id: [config.trainer.n_gpus_per_node // 2],
            }

        mapping = {
            Role.Actor: actor_ref_pool_id,
            Role.Rollout: rollout_pool_id,
        }

        if config.algorithm.adv_estimator == 'ppo':
            # If using PPO, we need to allocate the critic and reference models to the training nodes
            mapping[Role.Critic] = actor_ref_pool_id 
            mapping[Role.RefPolicy] = actor_ref_pool_id 
        elif config.algorithm.adv_estimator == 'grpo':
            # We only need to additionally allocate the reference model
            mapping[Role.RefPolicy] = actor_ref_pool_id 
    else:
        # If we are not using async generation, we can just use the same resource pool for all the roles
        actor_rollout_ref_pool_id = 'actor_rollout_ref_pool'
        mapping = {
            # the actor and rollout are now under the same worker group
            Role.ActorRollout: actor_rollout_ref_pool_id,
            Role.RefPolicy: actor_rollout_ref_pool_id,
        }

        if config.algorithm.adv_estimator == 'ppo':
            critic_pool_id = 'critic_pool'
            mapping[Role.Critic] = critic_pool_id 
            if config.trainer.nnodes > 1:
                resource_pool_spec = {
                    actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes // 2,
                    critic_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes // 2,
                }
            else:
                resource_pool_spec = {
                    actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node // 2],
                    critic_pool_id: [config.trainer.n_gpus_per_node // 2],
                }
        else:
            if config.trainer.nnodes > 1:
                resource_pool_spec = {
                    actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
                }
            else:
                resource_pool_spec = {
                    actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node],
                }

    if config.algorithm.adv_estimator == 'drpo':
        role_worker_mapping.pop(Role.RefPolicy)

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker

        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = critic_pool_id 

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainerAsync(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
