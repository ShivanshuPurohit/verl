
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=data/math/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=data/math/qwen2.5-1.5b-instruct.parquet \
    data.batch_size=512 \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=4096 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
