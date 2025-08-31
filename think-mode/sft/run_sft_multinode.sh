save_path="/home/violet/checkpoints/think-mode-sft/qwen3_8b-v0"
# Shift the arguments so $@ refers to the rest
shift 2

servicenow_train_path=//home/violet/data/servicenow-think-mode-sft/train.parquet
servicenow_test_path=//home/violet/data/servicenow-think-mode-sft/test.parquet
open_thoughts_train_path=//home/violet/data/open-thoughts-think-mode-sft/train.parquet
open_thoughts_test_path=//home/violet/data/open-thoughts-think-mode-sft/test.parquet

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
    data.train_batch_size=1024 \
    data.micro_batch_size_per_gpu=32 \
    data.max_length=16384 \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.enable_gradient_checkpointing=True \
    optim.lr=5e-6 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=think-mode-sft \
    trainer.experiment_name=qwen2.5_7b-v0 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null $@ \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    +trainer.env="//home/violet/.env" \
    trainer.save_freq=200 \
    trainer.val_freq=20 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
