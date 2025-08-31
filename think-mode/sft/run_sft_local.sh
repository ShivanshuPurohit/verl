save_path="checkpoints/think-mode-sft/qwen3_8b-v1"
# Shift the arguments so $@ refers to the rest
shift 2

servicenow_train_path=data/servicenow-think-mode-sft/train.parquet
servicenow_test_path=data/servicenow-think-mode-sft/test.parquet
open_thoughts_train_path=data/open-thoughts-think-mode-sft/train.parquet
open_thoughts_test_path=data/open-thoughts-think-mode-sft/test.parquet

train_files="['$servicenow_train_path', '$open_thoughts_train_path']"
test_files="['$servicenow_test_path', '$open_thoughts_test_path']"

python3 -m verl.trainer.main_sft \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.use_chat_template=False \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['solution'] \
    data.train_batch_size=512 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=20480 \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.enable_gradient_checkpointing=True \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.weight_decay=0.05 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=think-mode-sft \
    trainer.experiment_name=qwen2.5_7b-v1 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null $@ \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    +trainer.env=".env" \
    trainer.save_freq=500 \
    trainer.val_freq=250 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true