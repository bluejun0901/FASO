LOGDIR="../../logs/experiment1"

RUN_ID=$(echo -n "$(realpath "$LOGDIR")" | sha1sum | cut -c1-12)

while true; do
  wandb sync "$LOGDIR" \
    --id "$RUN_ID" \
    --project LLM_DAG_ALLIGN \
    --entity bluejun0901-gyeonggi-science-high-school
  sleep 600
done