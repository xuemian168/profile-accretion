#!/bin/bash
# Runs on the Aqua login node: submits one H100 PBS job that scores the length control for all six
# models in sequence, polls it, and resubmits if the job is system-held.
set -u
S=${SCRATCH:?set SCRATCH to your cluster scratch directory (model cache, container image, python libs)}; W=$PWD
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a wrapper.log; }
jstate(){ qstat -xf "$1" 2>/dev/null | awk '/job_state =/{s=$3} /Exit_status =/{e=$3} /comment =/{c=$0} END{printf "%s|%s|%s\n", s, e, c}'; }
cat > run_length.pbs <<PBS
#PBS -q gpu_batch
#PBS -N len_ctrl
#PBS -l select=1:ncpus=8:ngpus=1:gpu_id=H100:mem=96gb
#PBS -l walltime=06:00:00
#PBS -j oe
cd "\$PBS_O_WORKDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee gpu.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$S/cache/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$S/g2p/pylibs
RUN="apptainer exec --nv --bind $S:$S --bind $W:$W $S/sif/torch-cuda.sif python"
for M in Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-32B mistralai/Mistral-Small-24B-Instruct-2501 allenai/OLMo-2-0325-32B-Instruct; do
  T=\${M##*/}
  if [ -s length_\$T.jsonl ] && grep -q FINISHED log_length_\$T.txt 2>/dev/null; then echo "skip \$T"; continue; fi
  \$RUN score_length.py --items items_all.json --model \$M --out length_\$T.jsonl --max_tokens_batch 10000 > log_length_\$T.txt 2>&1 || echo "FAILED \$M" >> failures.txt
  tail -1 log_length_\$T.txt
done
PBS
for round in $(seq 1 40); do
  J=$(qsub -o "$W/pbs.log" run_length.pbs); log "submitted $J"
  while true; do
    sleep 120
    IFS='|' read -r st ex cm <<< "$(jstate $J)"
    if [ "$st" = "F" ]; then log "job $J finished exit=$ex"; ls -lh length_*.jsonl; cat failures.txt 2>/dev/null; exit 0; fi
    if [ "$st" = "H" ]; then log "job $J held: $cm"; qdel $J >/dev/null 2>&1; sleep 600; break; fi
  done
done
log "gave up"
