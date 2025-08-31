save_path=~/checkpoint/tir-sft-sanity/qwen7b-tir-1sol-4o-distill
# Shift the arguments so $@ refers to the rest
shift 2

python -m verl.trainer.main_sft \
    data.train_files=data/tir-sft-sanity/train.parquet \
    data.val_files=data/tir-sft-sanity/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.additional_mask="[['<obs>','</obs>']]" \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['solution'] \
    data.train_batch_size=256 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=8192 \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=tir-sft \
    trainer.experiment_name=qwen7b-tir-1sol-4o-distill \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null $@ \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    +trainer.push_to_hub=True \
    +trainer.hub_model_id="RLAIF/qwen7b-tir-sft" \
    trainer.save_freq=232 \
    trainer.val_freq=29 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
