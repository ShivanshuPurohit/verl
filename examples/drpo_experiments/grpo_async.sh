set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_CONFIG_PATH=examples/vllm_logging_config.json
SPECIFIC_CHECKPOINT="//home/shiv/test_ckpts/drpo_async/global_step_14"
# trainer.resume_from_path=$SPECIFIC_CHECKPOINT \

python3 -m verl.trainer.main_drpo \
    --config-name grpo_trainer_async \
    data.train_files=//home/chase/data/big-math/train.parquet \
    data.val_files=//home/chase/data/big-math/test.parquet \
    data.train_batch_size=32\
    +data.gen_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.online_prob=0.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.async_gen=True \
    actor_rollout_ref.rollout.replay_buffer_size=50000 \
    actor_rollout_ref.rollout.cleanba_style=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=4096 \
    actor_rollout_ref.rollout.update_interval=10 \
    +actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.val_generations_to_log_to_wandb=32 \
    trainer.critic_warmup=0 \
    +trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='drpo_experiments' \
    trainer.experiment_name='qwen_1.5b_async_grpo' \
    +trainer.remove_previous_ckpt_in_save=False \
    +trainer.del_local_ckpt_after_load=False \
    trainer.resume_mode=disable \
    trainer.default_local_dir=//home/chase/test_ckpts/drpo_async \
    trainer.resume_from_path=$SPECIFIC_CHECKPOINT \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    +trainer.env=//home/chase/.env \
    +actor_rollout_ref.actor.checkpoint.path=$SPECIFIC_CHECKPOINT \
    +actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,tokenizer,dataloader] \
    $@
