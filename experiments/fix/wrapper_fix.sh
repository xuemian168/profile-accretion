#!/bin/bash
# Runs on the Aqua login node. Submits one H100 job per pair of models running e2e_fix.py
# (anchor, profile, profile_cal, profile_audit, profile_cal_audit) over blocks_e2e.json.
# Finished stages are skipped on resubmission; a held job is deleted and retried.
set -u
S=${SCRATCH:?set SCRATCH to your cluster scratch directory (model cache, container image, python libs)}
W=$PWD
REPS="anchor,profile,profile_cal,profile_audit,profile_cal_audit"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a wrapper_fix.log; }

jstate(){ qstat -xf "$1" 2>/dev/null | awk '/job_state =/{s=$3} /Exit_status =/{e=$3} /comment =/{c=$0} END{printf "%s|%s|%s\n", s, e, c}'; }

write_job(){ # $1 tag, $2.. models
  local tag=$1; shift
  cat > run_$tag.pbs <<EOF
#PBS -q gpu_batch
#PBS -N fix_$tag
#PBS -l select=1:ncpus=8:ngpus=1:gpu_id=H100:mem=96gb
#PBS -l walltime=08:00:00
#PBS -j oe
cd "\$PBS_O_WORKDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee gpu_$tag.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$S/cache/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$S/g2p/pylibs
RUN="apptainer exec --nv --bind $S:$S --bind $W:$W $S/sif/torch-cuda.sif python"
for M in $*; do
  T=\$(echo \$M | sed 's#.*/##')
  if [ -s e2e_fix_\$T.jsonl ] && grep -q FINISHED log_\$T.txt 2>/dev/null; then echo "skip \$T"; continue; fi
  \$RUN e2e_fix.py --blocks blocks_e2e.json --model \$M --thresholds thresholds.json --reps $REPS --out e2e_fix_\$T.jsonl --bs 16 > log_\$T.txt 2>&1 || echo "FAILED \$M" >> failures.txt
  tail -3 log_\$T.txt
done
EOF
}

probe_ok(){
  cat > probe.pbs <<'EOF'
#PBS -q gpu_batch
#PBS -N fix_probe
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

write_job A Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct
write_job B Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3-32B
write_job C allenai/OLMo-2-0325-32B-Instruct mistralai/Mistral-Small-24B-Instruct-2501
declare -A JOB DONE
DONE[A]=0; DONE[B]=0; DONE[C]=0
for round in $(seq 1 60); do
  [ "$round" -gt 1 ] && { log "waiting 10 min before resubmitting"; sleep 600; }
  until probe_ok; do log "GPU not available; sleeping 30 min"; sleep 1800; done
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
    [ "${DONE[A]}" = "1" ] && [ "${DONE[B]}" = "1" ] && [ "${DONE[C]}" = "1" ] && { log "all done"; ls -lh e2e_fix_*.jsonl || true; cat failures.txt 2>/dev/null || true; exit 0; }
    [ "$held" = "1" ] && break
  done
done
log "gave up after 60 rounds"; ls -lh e2e_fix_*.jsonl || true
