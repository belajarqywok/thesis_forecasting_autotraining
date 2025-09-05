#!/bin/bash
LOGDIR="logfile"
BASELOG="$LOGDIR/automated_training.log"
mkdir -p "$LOGDIR"

get_logfile() {
  if [ ! -f "$BASELOG" ]; then
    echo "$BASELOG"
  else
    i=1
    while [ -f "$LOGDIR/automated_training_$i.log" ]; do
      i=$((i+1))
    done
    echo "$LOGDIR/automated_training_$i.log"
  fi
}

echo "---------------------------------------------------------------" | tee -a "$LOGFILE"
echo "-------------------- [ Autogen Workflows ] --------------------" | tee -a "$LOGFILE"
echo "---------------------------------------------------------------" | tee -a "$LOGFILE"
python autogen_workflows.py | tee -a "$LOGFILE"
ls -al .github/workflows | tee -a "$LOGFILE"
rm .github/workflows/workflow_*_pipeline.yaml | tee -a "$LOGFILE"

echo "-------------------------------------------------------" | tee -a "$LOGFILE"
echo "-------------------- [ Workloads ] --------------------" | tee -a "$LOGFILE"
echo "-------------------------------------------------------" | tee -a "$LOGFILE"
ls -al indonesia_stocks/workloads | tee -a "$LOGFILE"

LOGFILE=$(get_logfile)
for (( i=1; i<=$1; i++ ))
do
  echo "---------------------------------------------------------" | tee -a "$LOGFILE"
  echo "------------------ [ Iteration - $i ] ------------------"  | tee -a "$LOGFILE"
  echo "---------------------------------------------------------" | tee -a "$LOGFILE"
  python autotraining.py \
    --model_registry=deployment_models \
    --workloads_json=workloads_$i.json 2>&1 | tee -a "$LOGFILE"
done

echo "---------------------------------------------------------------" | tee -a "$LOGFILE"
echo "-------------------- [ Deployment Models ] --------------------" | tee -a "$LOGFILE"
echo "---------------------------------------------------------------" | tee -a "$LOGFILE"
ls -al deployment_models | tee -a "$LOGFILE"
