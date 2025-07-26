num_tasks=$(($(wc -l < rl_grid.csv) - 1))

echo "num_tasks: $num_tasks"
echo "--array=0-$(($num_tasks - 1))"

sbatch --array=0-$(($num_tasks - 1)) run.sh

