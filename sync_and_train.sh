#!/bin/bash
# Pi-side orchestration: push crops + labels to the workstation, run a training
# tag there, pull the run dir (model + eval report) back.
#   ./sync_and_train.sh v1 [extra train.py args...]
set -euo pipefail
WS=ian@192.168.0.180
REMOTE=Code/meteorml-uk00dy
TAG=${1:?usage: sync_and_train.sh <tag> [train args]}
shift || true

# GMN auto-labels are written on the Pi -- the manifests here are the source of
# truth, and the rsync --delete below would clobber labels written remotely.
python3 ~/source/meteorml-uk00dy/src/label_gmn.py --data ~/RMS_data/ml_training

rsync -a --delete ~/RMS_data/ml_training/ "$WS:$REMOTE/data/ml_training/"
ssh "$WS" "cd $REMOTE && git pull -q && \
  .venv/bin/python src/build_dataset.py --data data/ml_training && \
  .venv/bin/python src/train.py --tag '$TAG' $* && \
  .venv/bin/python src/evaluate.py --candidate 'runs/$TAG/model.keras'"
mkdir -p ~/source/meteorml-uk00dy/runs
rsync -a "$WS:$REMOTE/runs/$TAG/" ~/source/meteorml-uk00dy/runs/"$TAG"/
echo "run '$TAG' pulled back to ~/source/meteorml-uk00dy/runs/$TAG/"
