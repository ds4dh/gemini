#!/bin/bash

# Slurm configuration
PARTITION="shared-gpu"
TIME_TO_RUN="0-11:55:00"
NUM_GPUS=1
CONFIG_FILE="./config.yaml"
INFERENCE_BACKEND="vllm-serve-async"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'"
    exit 1
fi

# Node and GPU memory configuration
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *baobab* ]]; then
    echo "Detected Baobab cluster from hostname: $HOSTNAME"
    DEFAULT_NODE_LIST="gpu034,gpu035"                   # 24GB GPUs
    LARGE_MEM_NODE_LIST="gpu020,gpu030,gpu031,gpu028"   # 40GB GPUs
    LARGER_MEM_NODE_LIST="gpu029,gpu032,gpu033,gpu045"  # 80GB GPUs
elif [[ "$HOSTNAME" == *bamboo* ]]; then
    echo "Detected Bamboo cluster from hostname: $HOSTNAME"
    DEFAULT_NODE_LIST="gpu002,gpu003,gpu007"            # 24-97GB GPUs
    LARGE_MEM_NODE_LIST="gpu003,gpu007,gpu005,gpu006"   # 80-141GB GPUs
    LARGER_MEM_NODE_LIST="gpu003,gpu007,gpu005,gpu006"  # 80-141GB GPUs
else
    echo "Warning: Unknown hostname '$HOSTNAME'. Using default node list."
    DEFAULT_NODE_LIST="gpu034,gpu035"
    LARGE_MEM_NODE_LIST="gpu020,gpu030"
    LARGER_MEM_NODE_LIST="gpu029,gpu032"
fi

# Models to test
MODEL_PATHS=(
    "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"
)

# Default quantization schemes
DEFAULT_QUANTS=(
    "Q6_K_XL"
)

# Main Loop
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    CURRENT_QUANTS=("${DEFAULT_QUANTS[@]}")

    for QUANT_SCHEME in "${CURRENT_QUANTS[@]}"; do
        echo "--------------------------------------------------------"
        echo "   Preparing benchmark for:"
        echo "   Model: $MODEL_PATH"
        echo "   Quantization: $QUANT_SCHEME"

        # Determine node list AND memory to use
        NODE_LIST_TO_USE=$DEFAULT_NODE_LIST
        GPU_MEM_UTIL="0.80"
        MEM_TO_USE="64G"

        if [[ "$MODEL_PATH" == *"70B"* ]]; then
            NODE_LIST_TO_USE=$LARGER_MEM_NODE_LIST
            GPU_MEM_UTIL="0.95"
            MEM_TO_USE="200G"
        fi
        
        # Determine reasoning parser
        if [[ "$MODEL_PATH" == *"DeepSeek-R1"* ]]; then
            CURRENT_PARSER="deepseek_r1"
        else
            CURRENT_PARSER="qwen3"
        fi
        
        echo "   Targeting nodes: $NODE_LIST_TO_USE"
        echo "   GPU Memory Utilization: $GPU_MEM_UTIL"
        echo "   Reasoning Parser: $CURRENT_PARSER"
        echo "--------------------------------------------------------"

        # Submit Slurm job with CLI parameters (no sed YAML hacking!)
        ./scripts/run_benchmark.sh \
            -g "$NUM_GPUS" \
            -p "$PARTITION" \
            -m "$MEM_TO_USE" \
            -t "$TIME_TO_RUN" \
            -n "$NODE_LIST_TO_USE" \
            -c "$CONFIG_FILE" \
            --model "$MODEL_PATH" \
            --quant-scheme "$QUANT_SCHEME" \
            --backend "$INFERENCE_BACKEND" \
            --gpu-mem "$GPU_MEM_UTIL" \
            --reasoning-parser "$CURRENT_PARSER"

        echo "Benchmark job submitted for $MODEL_PATH ($QUANT_SCHEME)."
        sleep 1
    done
done

echo "All benchmark jobs submitted successfully!"