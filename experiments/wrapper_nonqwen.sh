#!/bin/bash
# Runs on the Aqua login node. Submits one job per
# model (controlled profiles, mitigation/contamination, online linking) and polls them; a job that
# gets system-held is deleted and resubmitted. Finished stages are skipped on resubmission.
set -u
S=${SCRATCH:?set SCRATCH to your cluster scratch directory (model cache, container image, python libs)}
W=$PWD
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a wrapper.log; }
jstate(){ qstat -xf "$1" 2>/dev/null | awk '/job_state =/{s=$3} /Exit_status =/{e=$3} /comment =/{c=$0} END{printf "%s|%s|%s\n", s, e, c}'; }

write_job(){ # $1 tag, $2 model
  local tag=$1 M=$2; local T=${M##*/}
  cat > run_$tag.pbs <<EOF
#PBS -q gpu_batch
#PBS -N nq_$tag
#PBS -l select=1:ncpus=8:ngpus=2:gpu_id=A100:mem=96gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "\$PBS_O_WORKDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee gpu_$tag.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$S/cache/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$S/g2p/pylibs
RUN="apptainer exec --nv --bind $S:$S --bind $W:$W $S/sif/torch-cuda.sif python"
if [ ! -s smoke_$T.jsonl ]; then \$RUN score_accretion.py --items items_all.json --model $M --out smoke_$T.jsonl --limit 48 --max_tokens_batch 6000 > smoke_$T.txt 2>&1 || echo "FAILED smoke $M" >> failures.txt; fi
grep -E "yes ids|prompts" smoke_$T.txt || true
awk -F'"mass": ' 'NF>1{split(\$2,a,","); s+=a[1]; n++; if(n==1||a[1]<m)m=a[1]} END{print "smoke n",n,"mean mass",s/n,"min mass",m}' smoke_$T.jsonl || true
for st in scores mitig e2e; do
  if [ -s \${st}_$T.jsonl ] && grep -q FINISHED log_\${st}_$T.txt 2>/dev/null; then echo "skip \$st"; continue; fi
  case \$st in
    scores) \$RUN score_accretion.py --items items_all.json --model $M --out scores_$T.jsonl --max_tokens_batch 6000 > log_scores_$T.txt 2>&1 || echo "FAILED scores $M" >> failures.txt ;;
    mitig)  \$RUN score_jobs.py --items jobs_mitig.jsonl --model $M --out mitig_$T.jsonl --max_tokens_batch 6000 > log_mitig_$T.txt 2>&1 || echo "FAILED mitig $M" >> failures.txt ;;
    e2e)    \$RUN e2e_link.py --blocks blocks_e2e.json --model $M --out e2e_$T.jsonl --bs 8 > log_e2e_$T.txt 2>&1 || echo "FAILED e2e $M" >> failures.txt ;;
  esac
  tail -2 log_\${st}_$T.txt
done
EOF
}

probe_ok(){
  cat > probe.pbs <<'EOF'
#PBS -q gpu_batch
#PBS -N nq_probe
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

write_job A allenai/OLMo-2-0325-32B-Instruct
write_job B mistralai/Mistral-Small-24B-Instruct-2501
declare -A JOB DONE
DONE[A]=0; DONE[B]=0
for round in $(seq 1 60); do
  [ "$round" -gt 1 ] && { log "waiting 10 min before resubmitting"; sleep 600; }
  for t in A B; do
    if [ "${DONE[$t]}" = "0" ] && [ -z "${JOB[$t]:-}" ]; then JOB[$t]=$(qsub -o "$W/pbs_$t.log" run_$t.pbs); log "submitted $t ${JOB[$t]}"; fi
  done
  while true; do
    sleep 120
    held=0
    for t in A B; do
      [ "${DONE[$t]}" = "1" ] && continue
      IFS='|' read -r st ex cm <<< "$(jstate ${JOB[$t]})"
      if [ "$st" = "F" ]; then DONE[$t]=1; log "job $t ${JOB[$t]} finished exit=$ex"
      elif [ "$st" = "H" ]; then log "job $t held: $cm"; qdel ${JOB[$t]} >/dev/null 2>&1; JOB[$t]=""; held=1; fi
    done
    [ "${DONE[A]}" = "1" ] && [ "${DONE[B]}" = "1" ] && { log "all done"; ls -lh *.jsonl || true; cat failures.txt 2>/dev/null || true; exit 0; }
    [ "$held" = "1" ] && break
  done
done
log "gave up after 60 rounds"; ls -lh *.jsonl || true
