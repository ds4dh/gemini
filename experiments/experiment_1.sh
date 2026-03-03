#!/bin/bash

# Slurm configuration
PARTITION="shared-gpu"
TIME_TO_RUN="0-11:55:00"
NUM_GPUS=1
MAX_CONCURRENT_INFS="32"  # todo: check if very large models fail with 256 and would require 64

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
    echo "Error: Could not determine known cluster from hostname '$HOSTNAME'."
    echo "Plase modify the script to define your own node list."
    exit 1
fi

# Benchmark configuration
CONFIG_FILE="./configs/model_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'"
    exit 1
fi
INFERENCE_BACKEND="vllm-serve-async"

# Models to test
MODEL_PATHS=(
    # "unsloth/Qwen3-0.6B-GGUF"
    # "unsloth/Qwen3-1.7B-GGUF"
    # "unsloth/Qwen3-4B-GGUF"
    # "unsloth/Qwen3-8B-GGUF"
    # "unsloth/Qwen3-14B-GGUF"
    # "unsloth/Qwen3-32B-GGUF"
    # "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF"
    "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"
)

# Default quantization schemes (available in most models)
DEFAULT_QUANTS=(
    # "Q8_0"
    "Q6_K_XL"
    # "Q5_K_XL"
    # "Q4_K_XL"
    # "Q3_K_XL"  # llama-70B ran 61% in 6 hours for this quant scheme
    # "Q2_K_XL"
)

# Specific schemes for DeepSeek-R1-Distill-Qwen-32B (restricted schemes)
DEEPSEEK_QWEN_32B_QUANTS=(
    "Q8_0"
    "Q6_K"
    "Q5_K_M"
    "Q4_K_M"
    "Q3_K_M"
    "Q2_K_L"
)

# Map specific models to their custom quantization array name
declare -A MODEL_QUANT_MAP
MODEL_QUANT_MAP["unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF"]="DEEPSEEK_QWEN_32B_QUANTS"

# Node requirements mapping
declare -A NODE_LIST_REQUIREMENTS

NODE_LIST_REQUIREMENTS["unsloth/Qwen3-14B-GGUF:Q8_0"]=$LARGE_MEM_NODE_LIST

NODE_LIST_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q3_K_XL"]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q4_K_XL"]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q5_K_XL"]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q6_K_XL"]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q8_0"]=$LARGER_MEM_NODE_LIST

NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF":Q3_K_M]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF":Q4_K_M]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF":Q5_K_M]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF":Q6_K]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF":Q8_0]=$LARGER_MEM_NODE_LIST

NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF":Q2_K_XL]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF":Q3_K_XL]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF":Q4_K_XL]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF":Q5_K_XL]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF":Q6_K_XL]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF":Q8_0]=$LARGER_MEM_NODE_LIST

NODE_LIST_REQUIREMENTS["unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q3_K_XL"]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q4_K_XL"]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q5_K_XL"]=$LARGE_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q6_K_XL"]=$LARGER_MEM_NODE_LIST
NODE_LIST_REQUIREMENTS["unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q8_0"]=$LARGER_MEM_NODE_LIST

# Main Loop
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    
    # Check if model has a specific quantization mapping
    if [[ -v "MODEL_QUANT_MAP[$MODEL_PATH]" ]]; then
        ARRAY_NAME=${MODEL_QUANT_MAP[$MODEL_PATH]}
        declare -n CURRENT_QUANTS=$ARRAY_NAME
    else
        CURRENT_QUANTS=("${DEFAULT_QUANTS[@]}")  # fallback to default
    fi

    for QUANT_SCHEME in "${CURRENT_QUANTS[@]}"; do
        echo "--------------------------------------------------------"
        echo "   Preparing benchmark for:"
        echo "   Model: $MODEL_PATH"
        echo "   Quantization: $QUANT_SCHEME"

        # Determine node list AND memory to use
        KEY="${MODEL_PATH}:${QUANT_SCHEME}"
        NODE_LIST_TO_USE=""
        GPU_MEM_UTIL="0.80"
        MEM_TO_USE="64G" 

        if [[ -v "NODE_LIST_REQUIREMENTS[$KEY]" ]]; then
            NODE_LIST_TO_USE=${NODE_LIST_REQUIREMENTS[$KEY]}
            
            if [[ "$NODE_LIST_TO_USE" == "$LARGER_MEM_NODE_LIST" ]]; then
                GPU_MEM_UTIL="0.95"
                MEM_TO_USE="200G"   # set high memory for 80GB GPU nodes (70B models)
                echo "   Found custom config: targeting larger-memory nodes: $NODE_LIST_TO_USE"
                
            elif [[ "$NODE_LIST_TO_USE" == "$LARGE_MEM_NODE_LIST" ]]; then
                GPU_MEM_UTIL="0.80"
                MEM_TO_USE="120G"   # set medium memory for 40GB GPU nodes
                echo "   Found custom config: targeting large-memory nodes: $NODE_LIST_TO_USE"
            fi
        else
            NODE_LIST_TO_USE=$DEFAULT_NODE_LIST
            MEM_TO_USE="64G"        # standard memory for smaller models
            echo "   Using default config: targeting standard nodes: $NODE_LIST_TO_USE"
        fi
        
        # Determine reasoning parser
        if [[ "$MODEL_PATH" == *"DeepSeek-R1"* ]]; then
            CURRENT_PARSER="deepseek_r1"
        else
            CURRENT_PARSER="qwen3"
        fi
        
        echo "   Setting GPU memory utilization to $GPU_MEM_UTIL"
        echo "   Using $CURRENT_PARSER reasoning parser."
        echo "--------------------------------------------------------"

        # Modify the YAML file using sed
        sed -i "s|^model_path:.*|model_path: $MODEL_PATH|" "$CONFIG_FILE"
        sed -i "s|^quant_scheme:.*|quant_scheme: $QUANT_SCHEME|" "$CONFIG_FILE"
        sed -i "s|^inference_backend:.*|inference_backend: $INFERENCE_BACKEND|" "$CONFIG_FILE"
        sed -i "s|^gpu_memory_utilization:.*|gpu_memory_utilization: $GPU_MEM_UTIL|" "$CONFIG_FILE"
        sed -i "s|^max_concurrent_inferences:.*|max_concurrent_infs: $MAX_CONCURRENT_INFS|" "$CONFIG_FILE"
        sed -i "s|^reasoning_parser:.*|reasoning_parser: $CURRENT_PARSER|" "$CONFIG_FILE"
        echo "Updated '$CONFIG_FILE'."

        # Submit the job with the specified parameters
        ./scripts/run_benchmark.sh \
            -g "$NUM_GPUS" \
            -p "$PARTITION" \
            -m "$MEM_TO_USE" \
            -t "$TIME_TO_RUN" \
            -n "$NODE_LIST_TO_USE"

        echo "Benchmark job sent for $MODEL_PATH ($QUANT_SCHEME)."
        echo ""

        # Prevent a race condition
        sleep 1
    done
done