# This is a baseline run using GRPO on a qwen-1.5b-r1-distill model.
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_CONFIG_PATH=examples/vllm_logging_config.json

deepscale_train_path=//home/chase/data-instruct-format/data/deepscale/train.parquet
math_test_path=//home/chase/data-instruct-format/data/math/test.parquet
aime24_test_path=//home/chase/data-instruct-format/data/aime-2024/test.parquet
aime25_test_path=//home/chase/data-instruct-format/data/aime-2025/test.parquet

train_files="['$deepscale_train_path']"
test_files="['$math_test_path', '$aime24_test_path', '$aime25_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.actor.log_adam_momentum=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=4096 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    +algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='baseline_debugging' \
    trainer.experiment_name='qwen1.5b_r1_distill_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.default_local_dir=//home/chase/checkpoints/qwen1.5b_r1_distill_grpo_deepscale \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="//home/chase/checkpoints/qwen1.5b_r1_distill_grpo_deepscale/global_step_2" \
    +trainer.env=//home/chase/.env \
    trainer.nnodes=4 \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
