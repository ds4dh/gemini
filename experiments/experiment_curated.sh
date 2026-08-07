#!/bin/bash

# Configuration & Paths
CONFIG_FILE="./config.yaml"
CURATED_DATA_PATH="data/synthetic_clinical_notes.csv"  # or /data/final/20260302_Letters_Combined.csv
INFERENCE_BACKEND="vllm"                              # or vllm-serve-async / llama-cpp
GPU_MEM_UTIL="0.80"

# Check baseline config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'"
    exit 1
fi

# Models and quantizations to test
MODEL_PATHS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    # "unsloth/Qwen3-32B-GGUF"
)
QUANT_SCHEMES=(
    "none"
    # "Q6_K_XL"
)

# Main loop
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for QUANT_SCHEME in "${QUANT_SCHEMES[@]}"; do
        echo "--------------------------------------------------------"
        echo "   Running extraction pipeline for:"
        echo "   Model:        $MODEL_PATH"
        echo "   Quantization: $QUANT_SCHEME"
        echo "   Backend:      $INFERENCE_BACKEND"
        echo "--------------------------------------------------------"

        # Execute run_pipeline with dynamic CLI parameters (no sed YAML hacking!)
        python run_pipeline.py \
            --config "$CONFIG_FILE" \
            --input-path "$CURATED_DATA_PATH" \
            --model "$MODEL_PATH" \
            --quant-scheme "$QUANT_SCHEME" \
            --backend "$INFERENCE_BACKEND" \
            --gpu-memory-utilization "$GPU_MEM_UTIL"

        # Check exit code
        if [ $? -ne 0 ]; then
            echo "Error: Extraction failed for $MODEL_PATH ($QUANT_SCHEME)."
        else
            echo "Extraction finished successfully for $MODEL_PATH ($QUANT_SCHEME)."
        fi

        sleep 1
    done
done

echo "All extraction runs completed!"