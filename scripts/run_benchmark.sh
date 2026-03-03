#!/bin/bash

# --- Default values for arguments ---
PARTITION="shared-gpu"
TIME="0-00:15:00"
GPUS_PER_TASK=1
MEM="128G"  # to load big models, must request significant system RAM
NODE_LIST="gpu034,gpu035"  # the ones from private-teodoro-gpu

# --- Help/usage function ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --partition       Slurm partition to use      (Default: ${PARTITION})"
    echo "  -t, --time            Job time limit (D-HH:MM:SS) (Default: ${TIME})"
    echo "  -g, --gpus-per-task   Number of GPUs per task     (Default: ${GPUS_PER_TASK})"
    echo "  -m, --mem             Memory per node (e.g. 64G)  (Default: ${MEM})"
    echo "  -n, --nodelist        Comma-separated node list   (Default: ${NODE_LIST})"
    echo "  -h, --help            Display this help message"
    echo "Example:"
    echo "  $0 -p shared-gpu -t 0-01:00:00 -g 1 -n gpu029,gpu032,gpu033,gpu045"
    exit 1
}

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--partition)
        if [[ -n "$2" && "$2" != -* ]]; then
            PARTITION="$2"
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -t|--time)
        if [[ -n "$2" && "$2" != -* ]]; then
            TIME="$2"
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -g|--gpus-per-task)
        if [[ -n "$2" && "$2" != -* ]]; then
            GPUS_PER_TASK="$2"
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -m|--mem)
        if [[ -n "$2" && "$2" != -* ]]; then
            MEM="$2"
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -n|--nodelist)
        if [[ -n "$2" && "$2" != -* ]]; then
            NODE_LIST="$2"
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -h|--help)
        usage
        ;;
        *)    # unknown option
        echo "Error: Unknown option '$1'"
        usage
        ;;
    esac
done

# --- Display final configuration ---
echo "Submitting job with the following configuration:"
echo "  Partition:         ${PARTITION}"
echo "  Time Limit:        ${TIME}"
echo "  GPUs per Task:     ${GPUS_PER_TASK}"
echo "  System Memory:     ${MEM}"
echo "  Node List:         ${NODE_LIST}"
echo "-----------------------------------------------"

# --- Slurm job configuration ---
JOB_NAME=gemini-inference
NODES=1
NTASKS=1
CPUS_PER_TASK=8
CONSTRAINT=""
# CONSTRAINT="COMPUTE_MODEL_RTX_3090_25G"
# CONSTRAINT="COMPUTE_MODEL_RTX_3090_25G|COMPUTE_MODEL_RTX_4090_25G"

# --- Python environment configuration ---
SIF_FOLDER="/home/users/b/borneta/sif"
SIF_NAME="gemini-image.sif"
SIF_IMAGE="${SIF_FOLDER}/${SIF_NAME}"

# --- Load local variables from .env file
if [ -f .env ]; then
    export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
else
    echo "Error: .env file not found!"
    exit 1
fi

# --- Decryption script configuration ---
PYTHON_WRAPPER_CALL="python -m scripts.run_benchmark"
ENCRYPTED_DATA_PATH="./data/data_2025/processed/dataset.encrypted.csv"
CURATED_DATA_PATH="./data/data_2024/processed/dataset.csv"
REMOTE_ENV_PATH="/home/borneta/Documents/gemini/.env"
KEY_NAME="GEMINI"
OUTPUT_FILE="./results/decrypted_data.csv"

# --- Create a configuration snapshot ---
CONFIG_SNAPSHOT_DIR="./configs/pending/config_$(date +%Y%m%d_%H%M%S)_$$"
echo "Creating configuration snapshot in: ${CONFIG_SNAPSHOT_DIR}"
mkdir -p "${CONFIG_SNAPSHOT_DIR}"
cp ./configs/*.yaml "${CONFIG_SNAPSHOT_DIR}/"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy configuration files. Aborting submission."
    exit 1
fi

# Job submission uses a "here document"
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
#SBATCH --constraint="${CONSTRAINT}"
#SBATCH --time=${TIME}
#SBATCH --output=./results/logs/job_%j.txt
#SBATCH --error=./results/logs/job_%j.err

# The srun command to be executed on the compute node
echo "Starting job \$SLURM_JOB_ID on node \$(hostname)..."
echo "Using configuration snapshot from: ${CONFIG_SNAPSHOT_DIR}"
srun apptainer exec --nv "${SIF_IMAGE}" ${PYTHON_WRAPPER_CALL} \\
    --encrypted-data-path "${ENCRYPTED_DATA_PATH}" \\
    --curated-data-path "${CURATED_DATA_PATH}" \\
    --remote-env-path "${REMOTE_ENV_PATH}" \\
    --key-name "${KEY_NAME}" \\
    --hostname "${REMOTE_HOSTNAME}" \\
    --username "${REMOTE_USERNAME}" \\
    --data-config-path "${CONFIG_SNAPSHOT_DIR}/data_config.yaml" \\
    --model-config-path "${CONFIG_SNAPSHOT_DIR}/model_config.yaml" \\
    --prompt-config-path "${CONFIG_SNAPSHOT_DIR}/prompt_config.yaml"

echo "Job finished with exit code \$?."
echo "Cleaning up temporary config directory: ${CONFIG_SNAPSHOT_DIR}"
rm -rf "${CONFIG_SNAPSHOT_DIR}"

EOF

# Let the user know the job has been submitted
echo "Job has been submitted to Slurm."

# Interactive session command:
# apptainer exec --nv /home/users/b/borneta/sif/gemini-image.sif \
#    python -m scripts.run_benchmark \
#    --encrypted-data-path ./data/data_2025/processed/dataset.encrypted.csv \
#    --curated-data-path ./data/data_2024/processed/dataset.csv \
#    --remote-env-path /home/borneta/Documents/gemini/.env \
#    --key-name "GEMINI" \
#    --hostname "$HOST_IP" \
#    --username "$REMOTE_USER" \
#    --data-config-path ./configs/data_config.yaml \
#    --model-config-path ./configs/model_config.yaml \
#    --prompt-config-path ./configs/prompt_config.yaml

# Desktop run command
# python -m scripts.run_benchmark \
#     --encrypted-data-path "./data/data_2025/processed/dataset.encrypted.csv" \
#     --curated-data-path "./data/data_2024/processed/dataset.csv" \
#     --remote-env-path "/home/shares/ds4dh/gemini_project/gemini/.env" \
#     --key-name "GEMINI" \
#     --hostname "login1.baobab.hpc.unige.ch" \
#     --username "your_user_name" \
#     --data-config-path "./configs/data_config.yaml" \
#     --model-config-path "./configs/model_config.yaml" \
#     --prompt-config-path "./configs/prompt_config.yaml"