#!/bin/bash
set -euo pipefail

# (optional) debug
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib,bond,eth0,eno1}"
export NCCL_IB_CUDA_SUPPORT=1
export TOKENIZERS_PARALLELISM=false

learning_rates=(1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4 9e-5 7e-5 5e-5 3e-5 1e-5)
heads=(12 6 2 1)
embeds=(768 1020 1536 2304)
model_paths=(models.NRGPT_H_FF2W models.GPT models.GPT_Rec_parallel)
min_lr_divfactors=(1 1.1 1.15 1.2 1.5 2 5 10)


# DO NOT start these many jobs at a time if your cluster can't handle it, rather try small portion of these configs based on cluster availablity
# These are all the params list that we tried to get the best config for each model

for model_path in "${model_paths[@]}"; do
  for head in "${heads[@]}"; do
    for embed in "${embeds[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for divfactor in "${min_lr_divfactors[@]}"; do
          
          min_lr=$(awk -v lr="$lr" -v div="$divfactor" 'BEGIN {print lr/div}')
          echo "Model: $model_path, Head: $head, Embed: $embed, LR: $lr, MIN_LR: $min_lr (divfactor: $divfactor)"

          LOG_DIR="./logs_owt/nrgpt"
          # Create unique log file names with all parameters
          ERR_LOG="$LOG_DIR/err_${model_path}_h${head}_e${embed}_lr${lr}_div${divfactor}"
          OUT_LOG="$LOG_DIR/out_${model_path}_h${head}_e${embed}_lr${lr}_div${divfactor}"
          mkdir -p "$LOG_DIR"
          rm -f $OUT_LOG
  
          # This is the bsub command specific for our system, please change it accordingly suitable for your cluster
          # Request 2 hosts, 8 GPUs per host
          bsub -n 2 -R "span[ptile=1]" \
                -q normal \
                -G grp_exploratory \
                -M 120G \
                -gpu "num=8:mode=exclusive_process" \
                -J "${embed}_${head}_${lr}_${divfactor}" \
                -eo "$ERR_LOG" \
                -o "$OUT_LOG" \
                bash -c '
set -euo pipefail

# export PYTHONPATH="./NRGPT-ICLR:${PYTHONPATH:-}" #set your pythonpath if needed

# --- Discover allocated hosts (one per slot: host count = NNODES) ---
mapfile -t HOSTS < <(echo "$LSB_MCPU_HOSTS" | awk "{for(i=1;i<=NF;i+=2)print \$i}")
NNODES=${#HOSTS[@]}
if (( NNODES < 2 )); then
  echo "FATAL: expected 2 nodes but got ${NNODES}. Check -n and span." >&2
  exit 1
fi

# Master host + stable port derived from job id
MASTER_ADDR="${HOSTS[0]}"
MASTER_PORT=$((10000 + (${LSB_JOBID:-12345} % 40000)))

# GPUs per node (fallback to 8)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
else
  GPUS_PER_NODE=8
fi

# Export shared env to child shells
export MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE LSB_JOBID
export LR="'"$lr"'"
export MIN_LR="'"$min_lr"'"
export HEAD="'"$head"'"
export EMBED="'"$embed"'"
export MODEL_PATH="'"$model_path"'"
export NCCL_SOCKET_IFNAME="'"${NCCL_SOCKET_IFNAME:-ib,bond,eth0,eno1}"'"
export NCCL_IB_CUDA_SUPPORT="1"
export TOKENIZERS_PARALLELISM="false"

echo "[INFO] Hosts: ${HOSTS[*]}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE}"

# --- Launch on each host: stream the launcher via stdin (no /tmp sharing needed) ---
pids=()
for i in $(seq 0 $((NNODES-1))); do
  host="${HOSTS[$i]}"
  echo "[INFO] launching node_rank=${i} on ${host}"

  lsrun -m "$host" bash -s -- "$i" << "EOS" & pids+=($!)
#!/bin/bash
set -euo pipefail

# activate env if needed:
# source make/activate.sh

# Rebuild PYTHONPATH safely on the remote node
# export PYTHONPATH="./NRGPT-ICLR:${PYTHONPATH:-}" #set your pythonpath if needed

NODE_RANK="$1"
echo "[node_rank=${NODE_RANK}] host=$(hostname) master=${MASTER_ADDR}:${MASTER_PORT} gpus=${GPUS_PER_NODE}"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${LSB_JOBID:-torchjob}" \
  --node_rank="${NODE_RANK}" \
  -- \
  train_nanogpt.py \
  config/train_owt_config.py \
  --learning_rate="${LR}" \
  --min_lr="${MIN_LR}" \
  --n_head="${HEAD}" \
  --n_embed="${EMBED}" \
  --model_path="${MODEL_PATH}"
EOS

done

# Wait for both nodes
for p in "${pids[@]}"; do
  wait "$p"
done
'
        done
      done
    done
  done
done
