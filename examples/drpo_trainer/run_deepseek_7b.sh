set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_CONFIG_PATH=examples/vllm_logging_config.json

python3 -m verl.trainer.main_drpo \
    --config-name drpo_trainer \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    +data.gen_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.tau=0.01 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.async_gen=True \
    actor_rollout_ref.rollout.replay_buffer_size=1024 \
    actor_rollout_ref.rollout.cleanba_style=True \
    actor_rollout_ref.rollout.update_interval=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    trainer.critic_warmup=0 \
    +trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='drpo_testing' \
    trainer.experiment_name='deepseek_llm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
