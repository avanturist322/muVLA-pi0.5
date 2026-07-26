#!/usr/bin/env bash
# Download one or more MIKASA-Robo tasks from mikasa-robo/mikasa-robo-vla-lerobot.
#
# The HF directory name is snake_case (remember_color_3_vla_v0), NOT the env id
# (RememberColor3-VLA-v0). Each task subdir is a self-contained LeRobot dataset v3
# root (~19 MB). Pass task names as arguments; with no arguments the single-task
# default (remember_color_3_vla_v0) is used.
#
#   ./download_data.sh                                        # default task
#   ./download_data.sh remember_color_3_vla_v0 remember_color_5_vla_v0
#
# After downloading, pre-decode the videos (torchcodec is unusable here):
#   .venv/bin/python scripts/predecode_videos.py data/<task>
set -euo pipefail

ROOT="${PI05_MEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASE="https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot/resolve/main"

TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  TASKS=(remember_color_3_vla_v0)
fi

FILES=(
  "data/chunk-000/file-000.parquet"
  "meta/episodes/chunk-000/file-000.parquet"
  "meta/info.json"
  "meta/stats.json"
  "meta/tasks.parquet"
  "source_rlds_metadata.json"
  "videos/observation.images.top/chunk-000/file-000.mp4"
  "videos/observation.images.wrist/chunk-000/file-000.mp4"
)

for task in "${TASKS[@]}"; do
  DST="$ROOT/data/$task"
  echo "=== $task -> $DST"
  for f in "${FILES[@]}"; do
    out="$DST/$f"
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then
      echo "skip (exists): $f"
      continue
    fi
    echo "get: $f"
    curl -sSfL --retry 3 -o "$out" "$BASE/$task/$f"
  done
  echo "--- tree: $task ---"
  find "$DST" -type f -printf '%10s  %p\n' | sort -k2
done

echo DOWNLOAD_DONE
