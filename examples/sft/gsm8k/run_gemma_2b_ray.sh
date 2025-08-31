#!/bin/bash
# Ray-based SFT trainer for Gemma 2B on GSM8K

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b_ray.sh <num_gpus> <save_path> [other_configs...]"
    exit 1
fi

num_gpus=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Start Ray if not already running
if ! ray status > /dev/null 2>&1; then
    ray start --head --num-gpus=$num_gpus
fi

# Create a properly formatted process_on_nodes array
gpu_array="["
for ((i=0; i<num_gpus; i++)); do
    if [ $i -eq 0 ]; then
        gpu_array+="$i"
    else
        gpu_array+=", $i"
    fi
done
gpu_array+="]"
process_on_nodes="[[$gpu_array]]"

# Define a job configuration JSON for ray job submit
cat > ray_job_config.json << EOF
{
    "runtime_env": {
        "working_dir": "../../..",
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN"
        }
    },
    "entrypoint": "python -m verl.trainer.main_sft"
}
EOF

# Submit the job to Ray
ray job submit \
    -- \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=google/gemma-2b-it \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-2b-it \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    "trainer.process_on_nodes=$process_on_nodes" \
    trainer.save_freq=100 \
    trainer.val_freq=100 \
    $@

# Clean up the temporary config file
rm ray_job_config.json 