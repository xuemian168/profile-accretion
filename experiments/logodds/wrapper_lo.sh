#!/bin/bash
# Aqua login node: two H100 PBS jobs (A: Qwen2.5-32B, Qwen2.5-7B; B: Qwen2.5-14B, Qwen3-32B) rescoring the main
# controlled design with the log-odds readout. Resubmits a job if it is system-held.
set -u
S=${SCRATCH:?set SCRATCH to your cluster scratch directory (model cache, container image, python libs)}; W=$PWD
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a wrapper.log; }
jstate(){ qstat -xf "$1" 2>/dev/null | awk '/job_state =/{s=$3} /Exit_status =/{e=$3} /comment =/{c=$0} END{printf "%s|%s|%s\n", s, e, c}'; }
mk(){ # $1 tag, rest models
  local tag=$1; shift
  cat > run_$tag.pbs <<PBS
#PBS -q gpu_batch
#PBS -N lo_$tag
#PBS -l select=1:ncpus=8:ngpus=1:gpu_id=H100:mem=96gb
#PBS -l walltime=06:00:00
#PBS -j oe
cd "\$PBS_O_WORKDIR"
nvidia-smi --query-gpu=name --format=csv,noheader | tee gpu_$tag.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$S/cache/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$S/g2p/pylibs
RUN="apptainer exec --nv --bind $S:$S --bind $W:$W $S/sif/torch-cuda.sif python"
for M in $*; do
  T=\${M##*/}
  if grep -q FINISHED log_lo_\$T.txt 2>/dev/null; then echo "skip \$T"; continue; fi
  \$RUN score_accretion_lo.py --items items_all.json --model \$M --out lo_\$T.jsonl --max_tokens_batch 10000 > log_lo_\$T.txt 2>&1 || echo "FAILED \$M" >> failures.txt
  tail -1 log_lo_\$T.txt
done
PBS
}
mk A Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-7B-Instruct
mk B Qwen/Qwen2.5-14B-Instruct Qwen/Qwen3-32B
declare -A JOB DONE; DONE[A]=0; DONE[B]=0
for round in $(seq 1 40); do
  for t in A B; do
    if [ "${DONE[$t]}" = "0" ] && [ -z "${JOB[$t]:-}" ]; then JOB[$t]=$(qsub -o "$W/pbs_$t.log" run_$t.pbs); log "submitted $t ${JOB[$t]}"; fi
  done
  while true; do
    sleep 120; held=0
    for t in A B; do
      [ "${DONE[$t]}" = "1" ] && continue
      IFS='|' read -r st ex cm <<< "$(jstate ${JOB[$t]})"
      if [ "$st" = "F" ]; then DONE[$t]=1; log "job $t ${JOB[$t]} finished exit=$ex"
      elif [ "$st" = "H" ]; then log "job $t held: $cm"; qdel ${JOB[$t]} >/dev/null 2>&1; JOB[$t]=""; held=1; fi
    done
    [ "${DONE[A]}" = "1" ] && [ "${DONE[B]}" = "1" ] && { log "all done"; ls -lh lo_*.jsonl; cat failures.txt 2>/dev/null; exit 0; }
    [ "$held" = "1" ] && { sleep 600; break; }
  done
done
log "gave up"
