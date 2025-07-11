
N=1
CONFIG_LIST="/home/jingqi/DeepLearningWorkshop/recipes/research_on_activation/research_on_activation_recipes.txt"

# 捕获 SIGINT/SIGTERM 并杀死所有子进程
trap 'echo "Killing all child processes..."; kill 0; exit 1' SIGINT SIGTERM

mapfile -t CONFIGS < "$CONFIG_LIST"
TOTAL=${#CONFIGS[@]}

for ((i=0; i<TOTAL; i+=N)); do
    PIDS=()
    for ((j=0; j<N && i+j<TOTAL; j++)); do
        CONFIG_PATH="${CONFIGS[i+j]}"
        echo "Starting: $CONFIG_PATH"
        python -m trainer.train --config_path "$CONFIG_PATH" &
        PIDS+=($!)
    done

    # 等待本批次所有进程结束
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
done

echo "All training jobs finished."