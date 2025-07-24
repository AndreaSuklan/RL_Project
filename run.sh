#!/bin/bash
#SBATCH --job-name=reinforcement_learning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
# SBATCH --gpus=1
#SBATCH --mem=256G
# SBATCH --mem-per-cpu=MaxMemperCPU
#SBATCH --partition=EPYC
#SBATCH --account=dssc
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# === Environment setup ===
VENV_DIR=~/rlprojenv
echo "Activating virtualenv..." >&2
source "$VENV_DIR/bin/activate"
# source ~/rlprojenv/bin/activate

# === Environment setup ===

echo "Checking Python virtual environment..." >&2
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[INFO] Virtualenv not found. Creating new environment at $VENV_DIR..." >&2
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Installing requirements..." >&2
    pip install --upgrade pip
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        echo "[WARN] No requirements.txt found. Skipping dependency install." >&2
    fi
else
    echo "[INFO] Activating existing virtualenv..." >&2
    source "$VENV_DIR/bin/activate"
fi


mkdir -p logs
LOGFILE="logs/env_${SLURM_JOB_ID}.log"
PARAMS_FILE="rl_params.txt"

# # === Detect GPU count ===
# if [ -z "$SLURM_GPUS" ]; then
#   echo "[WARN] SLURM_GPUS is not set. Using nvidia-smi to count." >&2
#   SLURM_GPUS=$(nvidia-smi -L | wc -l)
# fi

# === Read parameters ===
RL_ARGS=()
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# || -z "$line" ]] && continue
    line="${line#[[:space:]]*[0-9]*[[:space:]]}"
    line="${line%"${line##*[![:space:]]}"}"
    RL_ARGS+=("$line")
done < "$PARAMS_FILE"

if [[ ${#RL_ARGS[@]} -lt 2 ]]; then
    echo "[WARN] Using defaults: train dqn" >&2
    RL_ARGS+=(
        "train"
        "dqn"
    )
fi

# === Thread settings ===
THREADS_PER_GPU=$(( SLURM_CPUS_PER_TASK / 1))
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_GPU

# === Log environment ===
echo "==== SLURM ENVIRONMENT ===="     > "$LOGFILE"
env | grep SLURM_                   >> "$LOGFILE"
echo ""                             >> "$LOGFILE"

# echo "==== GPU STATUS ===="          >> "$LOGFILE"
# nvidia-smi                          >> "$LOGFILE" 2>/dev/null || echo "nvidia-smi not available" >> "$LOGFILE"

echo "==== CMD ===="        >> "$LOGFILE"
echo "python3 main.py ${RL_ARGS[@]}" >> "$LOGFILE"

# === Launch ===
python3 src/main.py ${RL_ARGS[@]} >> "logs/train_${RL_ARGS[0]}_${RL_ARGS[1]}_${SLURM_JOB_ID}.log" 2>&1

