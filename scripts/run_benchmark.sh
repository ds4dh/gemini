#!/bin/bash

# --- Default values for Slurm arguments ---
PARTITION="shared-gpu"
TIME="0-00:15:00"
GPUS_PER_TASK=1
MEM="128G"
NODE_LIST="gpu034,gpu035"
RUN_CONFIG="configs/run_cfg.yaml"
EXTRACTION_CONFIG="configs/extraction_cfgs/mrs_score.yaml"
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
QUANT_SCHEME="none"
INFERENCE_BACKEND="vllm-serve-async"
GPU_MEM_UTIL="0.85"
REASONING_PARSER="qwen3"

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--partition) PARTITION="$2"; shift 2 ;;
        -t|--time) TIME="$2"; shift 2 ;;
        -g|--gpus-per-task) GPUS_PER_TASK="$2"; shift 2 ;;
        -m|--mem) MEM="$2"; shift 2 ;;
        -n|--nodelist) NODE_LIST="$2"; shift 2 ;;
        -rc|--run-config) RUN_CONFIG="$2"; shift 2 ;;
        -ec|--extraction-config) EXTRACTION_CONFIG="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --quant-scheme) QUANT_SCHEME="$2"; shift 2 ;;
        --backend) INFERENCE_BACKEND="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM_UTIL="$2"; shift 2 ;;
        --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            exit 0
            ;;
        *) shift ;;
    esac
done

# --- Display configuration ---
echo "Submitting Slurm job for:"
echo "  Model:             ${MODEL_PATH}"
echo "  Quantization:      ${QUANT_SCHEME}"
echo "  Backend:           ${INFERENCE_BACKEND}"
echo "  Partition:         ${PARTITION}"
echo "  Time Limit:        ${TIME}"
echo "  System Memory:     ${MEM}"
echo "  Node List:         ${NODE_LIST}"
echo "-----------------------------------------------"

# --- Slurm job execution setup ---
JOB_NAME="gemini-extract"
NODES=1
NTASKS=1
CPUS_PER_TASK=8

SIF_FOLDER="/home/users/b/borneta/sif"
SIF_NAME="gemini-image.sif"
SIF_IMAGE="${SIF_FOLDER}/${SIF_NAME}"

# Load environment variables if present
if [ -f .env ]; then
    export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
fi

# Submit Slurm sbatch job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodelist=${NODE_LIST}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --gpus-per-task=${GPUS_PER_TASK}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=./results/logs/job_%j.txt
#SBATCH --error=./results/logs/job_%j.err

echo "Starting job \$SLURM_JOB_ID on \$(hostname)..."
srun apptainer exec --nv "${SIF_IMAGE}" python scripts/run_pipeline.py \\
    --run-config "${RUN_CONFIG}" \\
    --extraction-config "${EXTRACTION_CONFIG}" \\
    --model "${MODEL_PATH}" \\
    --quant-scheme "${QUANT_SCHEME}" \\
    --backend "${INFERENCE_BACKEND}" \\
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \\
    --reasoning-parser "${REASONING_PARSER}"

echo "Job finished with exit code \$?."
EOF

echo "Job submitted successfully to Slurm."