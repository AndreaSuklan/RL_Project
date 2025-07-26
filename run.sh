#!/bin/bash
#SBATCH --job-name=reinforcement_learning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --partition=THIN
#SBATCH --account=dssc
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# === Environment setup ===
VENV_DIR=~/rlprojenv
echo "Activating virtualenv..." >&2

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

# === Parameters ===
mkdir -p logs
LOGFILE="logs/env_${SLURM_JOB_ID}.log"
# PARAMS_FILE="rl_params.txt"

GRID_FILE="rl_grid.csv"

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID" >> "$LOGFILE"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID + 1))  # skip header

echo "[DEBUG] Line read from CSV (task $SLURM_ARRAY_TASK_ID):" >> "$LOGFILE"
tail -n +$((TASK_INDEX + 1)) "$GRID_FILE" | head -n1 >> "$LOGFILE"

IFS=, read -r model degree verbosity seed run_mode agent < <(tail -n +$((TASK_INDEX + 1)) "$GRID_FILE" | head -n1)
echo "[INFO] Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> model=$model, degree=$degree verbosity=$verbosity seed=$seed run_mode=$run_mode agent=$agent " >> "$LOGFILE"

RL_ARGS+=("--model=$model")
RL_ARGS+=("--degree=$degree")
RL_ARGS+=("-v=$verbosity")
RL_ARGS+=("-s=$seed")
RL_ARGS+=("$run_mode")
RL_ARGS+=("$agent")

# === Final debug print ===
echo "[DEBUG] MODEL:$model DEGREE:$degree VERBOSITY:$verbosity SEED:$seed RUN_MODE:$run_mode AGENT:$agent" >> "$LOGFILE"

# === Compose final log filename ===
LOGNAME="train_${agent}_${model}"
[[ "$model" == "poly" ]] && LOGNAME+="_d${degree}"
[[ -n "$seed" ]] && LOGNAME+="_s${seed}"
LOGNAME+="_${SLURM_JOB_ID}.log"
LOGRUN="logs/${LOGNAME}"

# === Thread settings ===
THREADS_PER_GPU=$(( SLURM_CPUS_PER_TASK / 1))
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_GPU

# === Log environment ===
echo "==== SLURM ENVIRONMENT ===="     >> "$LOGFILE"
env | grep SLURM_                   >> "$LOGFILE"
echo ""                             >> "$LOGFILE"

echo "==== CMD ===="                  >> "$LOGFILE"
echo "python3 main.py ${RL_ARGS[@]}" >> "$LOGFILE"

LOGRUN="logs/${LOGNAME}"
CMD="python3 src/main.py ${RL_ARGS[@]}"

echo "[INFO] Launching: $CMD" >&2
$CMD >> "$LOGRUN" 2>&1
