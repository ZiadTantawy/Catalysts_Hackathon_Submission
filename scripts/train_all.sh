#!/usr/bin/env bash
# End-to-end IDSSE training on this machine (Apple Silicon friendly).
set -euo pipefail
cd "$(dirname "$0")/.."
# Re-exec under caffeinate so the Mac stays awake for the whole run.
if [[ -z "${TRAIN_ALL_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export TRAIN_ALL_CAFFEINATED=1
  exec caffeinate -dims -- "$0" "$@"
fi
export BARATH_CACHE="${BARATH_CACHE:-/Users/karim/Private/Barca Hackathon/Catalysts_Hackathon_Submission/cache}"
mkdir -p logs cache/idsse
LOG="logs/train_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== train_all start $(date) ==="
echo "BARATH_CACHE=$BARATH_CACHE"
echo "device probe:"
python3 -c "from src.idsse.tune_trees import N_JOBS, _xgb_device; from src.idsse import train_nn; print('N_JOBS', N_JOBS); print('xgb', _xgb_device()); print('nn', train_nn.DEVICE)"

echo "=== 1/7 build tables ==="
python3 -m src.idsse.build_tables

echo "=== 2/7 tune lightgbm ==="
python3 -m src.idsse.tune_trees tune lgbm

echo "=== 3/7 tune xgboost ==="
python3 -m src.idsse.tune_trees tune xgb

echo "=== 4/7 tune random forest ==="
python3 -m src.idsse.tune_trees tune rf

echo "=== 5/7 final tree fits + spellF1 ==="
python3 -m src.idsse.tune_trees final

echo "=== 6/7 neural nets (gru, lstm, tcn) ==="
python3 -m src.idsse.train_nn cv gru
python3 -m src.idsse.train_nn cv lstm
python3 -m src.idsse.train_nn cv tcn

echo "=== train_all done $(date) ==="
echo "log: $LOG"
