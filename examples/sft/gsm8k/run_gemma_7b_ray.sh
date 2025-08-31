#!/bin/bash
# Ray-based SFT trainer for Gemma 7B on GSM8K
# Demonstrates multi-node training with Ray

set -x

if [ "$#" -lt 3 ]; then
    echo "Usage: run_gemma_7b_ray.sh <num_nodes> <gpus_per_node> <save_path> [other_configs...]"
    exit 1
fi

num_nodes=$1
gpus_per_node=$2
save_path=$3

# Shift the arguments so $@ refers to the rest
shift 3

# Start Ray on head node if not already running
if ! ray status > /dev/null 2>&1; then
    # Get the head node IP - in production you would use your actual head node IP
    # For local testing, use 127.0.0.1
    HEAD_IP="127.0.0.1"
    
    # Start the Ray head node
    ray start --head --num-gpus=$gpus_per_node \
        --dashboard-host=0.0.0.0 \
        --include-dashboard=true \
        --port=6379
    
    # In a production environment, you would have commands here to:
    # 1. Start Ray on worker nodes using ray start --address=$HEAD_IP:6379 --num-gpus=$gpus_per_node
    # 2. Verify that all nodes have joined
    
    # For single-node simulation of multiple nodes, we'll just use the process_on_nodes parameter
    echo "Ray started on head node. In production, you would connect worker nodes."
fi

# Create a properly formatted process_on_nodes array
# For multi-node, each inner array represents GPU indices for that node
process_on_nodes="["
for ((n=0; n<num_nodes; n++)); do
    if [ $n -gt 0 ]; then
        process_on_nodes+=", "
    fi
    
    process_on_nodes+="["
    for ((i=0; i<gpus_per_node; i++)); do
        if [ $i -gt 0 ]; then
            process_on_nodes+=", "
        fi
        process_on_nodes+="$i"
    done
    process_on_nodes+="]"
done
process_on_nodes+="]"

# Define a job configuration JSON for ray job submit
cat > ray_job_config.json << EOF
{
    "runtime_env": {
        "working_dir": "../../..",
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN"
        },
        "pip": [
            "flash-attn"
        ]
    },
    "entrypoint": "python -m verl.trainer.main_sft",
    "max_retries": 3,
    "metadata": {
        "name": "Gemma 7B SFT on GSM8K",
        "team": "ML",
        "priority": "high"
    }
}
EOF

# Submit the job to Ray
ray job submit \
    --runtime-env-json-file=ray_job_config.json \
    -- \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=64 \
    model.partial_pretrain=google/gemma-1.1-7b-it \
    model.enable_gradient_checkpointing=True \
    optim.lr=5e-6 \
    optim.warmup_steps_ratio=0.03 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-7b-it \
    trainer.total_epochs=3 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$gpus_per_node \
    trainer.nnodes=$num_nodes \
    "trainer.process_on_nodes=$process_on_nodes" \
    trainer.save_freq=100 \
    trainer.val_freq=100 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=True \
    $@

# Clean up the temporary config file
rm ray_job_config.json

# Print info about accessing Ray Dashboard
echo "==================================================="
echo "Ray Job submitted. Check the Ray Dashboard at:"
echo "http://$HEAD_IP:8265"
echo "===================================================" 