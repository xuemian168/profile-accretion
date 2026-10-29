#!/bin/bash
# Runs on the Aqua login node. Submits one H100 job per pair of models running e2e_sweep.py
# (16 variants: 5 global profile_t*, 5 global anchor_t*, 5 profile_cal_tau*, 1 profile_audit_soft)
# over blocks_e2e.json. Priority models (A, B) submitted first; C is lower priority.
set -u
S=${SCRATCH:?set SCRATCH to your cluster scratch directory (model cache, container image, python libs)}
W=$PWD
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a wrapper_sweep.log; }

jstate(){ qstat -xf "$1" 2>/dev/null | awk '/job_state =/{s=$3} /Exit_status =/{e=$3} /comment =/{c=$0} END{printf "%s|%s|%s\n", s, e, c}'; }

write_job(){ # $1 tag, $2.. models
  local tag=$1; shift
  cat > run_$tag.pbs <<EOF
#PBS -q gpu_batch
#PBS -N swp_$tag
#PBS -l select=1:ncpus=8:ngpus=1:gpu_id=H100:mem=96gb
#PBS -l walltime=10:00:00
#PBS -j oe
cd "\$PBS_O_WORKDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee gpu_$tag.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$S/cache/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$S/g2p/pylibs
RUN="apptainer exec --nv --bind $S:$S --bind $W:$W $S/sif/torch-cuda.sif python"
for M in $*; do
  T=\$(echo \$M | sed 's#.*/##')
  if [ -s sweep_\$T.jsonl ] && grep -q FINISHED log_sweep_\$T.txt 2>/dev/null; then echo "skip \$T"; continue; fi
  \$RUN e2e_sweep.py --blocks blocks_e2e.json --model \$M --variants variants_\$T.json --out sweep_\$T.jsonl --bs 16 > log_sweep_\$T.txt 2>&1 || echo "FAILED \$M" >> failures.txt
  tail -3 log_sweep_\$T.txt
done
EOF
}

probe_ok(){
  cat > probe.pbs <<'EOF'
#PBS -q gpu_batch
#PBS -N swp_probe
#PBS -l select=1:ncpus=2:ngpus=1:gpu_id=H100:mem=8gb
#PBS -l walltime=00:03:00
#PBS -j oe
nvidia-smi --query-gpu=name --format=csv,noheader
EOF
  local J; J=$(qsub -o "$W/probe.log" probe.pbs) || return 1
  for i in $(seq 1 30); do
    sleep 60
    IFS='|' read -r st ex cm <<< "$(jstate $J)"
    if [ "$st" = "F" ]; then [ "$ex" = "0" ] && { log "probe $J ok"; return 0; }; log "probe $J finished exit=$ex"; return 1; fi
    if [ "$st" = "H" ]; then log "probe $J held: $cm"; qdel $J >/dev/null 2>&1; return 1; fi
  done
  log "probe $J still not finished after 30 min (state $st), deleting"; qdel $J >/dev/null 2>&1; return 1
}

write_job A Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct
write_job B allenai/OLMo-2-0325-32B-Instruct mistralai/Mistral-Small-24B-Instruct-2501
write_job C Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-32B
declare -A JOB DONE
DONE[A]=0; DONE[B]=0; DONE[C]=0
for round in $(seq 1 60); do
  [ "$round" -gt 1 ] && { log "waiting 10 min before resubmitting"; sleep 600; }
  until probe_ok; do log "GPU not available; sleeping 20 min"; sleep 1200; done
  for t in A B C; do
    if [ "${DONE[$t]}" = "0" ] && [ -z "${JOB[$t]:-}" ]; then JOB[$t]=$(qsub -o "$W/pbs_$t.log" run_$t.pbs); log "submitted $t ${JOB[$t]}"; fi
  done
  while true; do
    sleep 120
    held=0
    for t in A B C; do
      [ "${DONE[$t]}" = "1" ] && continue
      IFS='|' read -r st ex cm <<< "$(jstate ${JOB[$t]})"
      if [ "$st" = "F" ]; then DONE[$t]=1; log "job $t ${JOB[$t]} finished exit=$ex"
      elif [ "$st" = "H" ]; then log "job $t held: $cm"; qdel ${JOB[$t]} >/dev/null 2>&1; JOB[$t]=""; held=1; fi
    done
    [ "${DONE[A]}" = "1" ] && [ "${DONE[B]}" = "1" ] && [ "${DONE[C]}" = "1" ] && { log "all done"; ls -lh sweep_*.jsonl || true; cat failures.txt 2>/dev/null || true; exit 0; }
    [ "$held" = "1" ] && break
  done
done
log "gave up after 60 rounds"; ls -lh sweep_*.jsonl || true
