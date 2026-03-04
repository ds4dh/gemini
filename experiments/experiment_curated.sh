#!/bin/bash

# Paths
CURATED_DATA_PATH="/data/final/20260302_Letters_Combined.csv"
MODEL_CONFIG_PATH="./configs/model_config.yaml"
DATA_CONFIG_PATH="./configs/data_config.yaml"
PROMPT_CONFIG_PATH="./configs/prompt_config.yaml"

# Configuration
INFERENCE_BACKEND="vllm-serve-async"
GPU_MEM_UTIL="0.80"
MAX_CONCURRENT_INFS="64"

# Check if the model configuration file exists
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at '$MODEL_CONFIG_PATH'"
    exit 1
fi

# Models and quantizations to test (GGUF)
MODEL_PATHS=(
    # "unsloth/Qwen3-0.6B-GGUF"
    # "unsloth/Qwen3-1.7B-GGUF"
    # "unsloth/Qwen3-4B-GGUF"
    # "unsloth/Qwen3-8B-GGUF"
    "unsloth/Qwen3-32B-GGUF"
    # "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
    # "unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF"  # VLLM NOT READY FOR NEXT-GGUF!
    # "unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF"  # NOT ENOUGH MEMORY!
)
QUANT_SCHEMES=(
    # "Q2_K_XL"
    # "Q3_K_XL"
    # "Q4_K_XL"
    # "Q5_K_XL"
    "Q6_K_XL"
    # "Q8_0"
)

# # Models and quantizations to test (AWQ/FP8)
# MODEL_PATHS=(
#     # Qwen/Qwen3-0.6B-FP8  # (no AWQ version available)
#     # Qwen/Qwen3-1.7B-FP8  # (no AWQ version available)
#     # Qwen/Qwen3-4B-AWQ
#     # Qwen/Qwen3-4B-FP8
#     # Qwen/Qwen3-8B-AWQ
#     # Qwen/Qwen3-8B-FP8
#     # cyankiwi/Qwen3-Next-80B-A3B-Thinking-AWQ-4bit
# )
# QUANT_SCHEMES=(
#     "no-quant-scheme"
# )

# Main loop
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for QUANT_SCHEME in "${QUANT_SCHEMES[@]}"; do
        echo "--------------------------------------------------------"
        echo "   Running benchmark for:"
        echo "   Model: $MODEL_PATH"
        echo "   Quantization: $QUANT_SCHEME"
        echo "--------------------------------------------------------"

        # Update the model configuration file
        sed -i "s|^model_path:.*|model_path: $MODEL_PATH|" "$MODEL_CONFIG_PATH"
        sed -i "s|^quant_scheme:.*|quant_scheme: $QUANT_SCHEME|" "$MODEL_CONFIG_PATH"
        sed -i "s|^inference_backend:.*|inference_backend: $INFERENCE_BACKEND|" "$MODEL_CONFIG_PATH"
        sed -i "s|^gpu_memory_utilization:.*|gpu_memory_utilization: $GPU_MEM_UTIL|" "$MODEL_CONFIG_PATH"
        sed -i "s|^max_concurrent_inferences:.*|max_concurrent_infs: $MAX_CONCURRENT_INFS|" "$MODEL_CONFIG_PATH"
        echo "Updated '$MODEL_CONFIG_PATH' for the current run."

        # Run the benchmark with updated configuration
        python -m scripts.run_benchmark \
            --curated-data-path "$CURATED_DATA_PATH" \
            --data-config-path "$DATA_CONFIG_PATH" \
            --model-config-path "$MODEL_CONFIG_PATH" \
            --prompt-config-path "$PROMPT_CONFIG_PATH"

        # Check the exit code of the benchmark script
        if [ $? -ne 0 ]; then
            echo "Error: Benchmark failed for $MODEL_PATH ($QUANT_SCHEME). Aborting."
            echo ""  # exit 1
        else
            echo "Benchmark finished successfully for $MODEL_PATH ($QUANT_SCHEME)."
            echo ""
        fi

        # Small delay between runs
        sleep 1
    done
done

echo "All benchmarks completed successfully!"