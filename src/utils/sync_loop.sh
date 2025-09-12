LOGDIR="../../logs/experiment1"

function sync_dir() {
  local dir="$1"
  # 경로를 사람이 읽을 수 있게 정리 (experiment1_train 같은 형태)
  local RUN_ID=$(realpath --relative-to="$(realpath "$LOGDIR"/..)" "$dir" \
    | tr '/.' '_')
  
  wandb sync "$dir" \
    --id "$RUN_ID" \
    --project LLM_DAG_ALLIGN \
    --entity bluejun0901-gyeonggi-science-high-school
}

while true; do
  for d in $(find "$LOGDIR" -type f -name "events.out.tfevents.*" -exec dirname {} \; | sort -u); do
    sync_dir "$d"
  done
  sleep 600
done