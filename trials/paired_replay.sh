#!/usr/bin/env bash
# trials/paired_replay.py inside the container, with the live nodes' environment (ROS, the workspace,
# CUDA's libraries). Arguments go straight through:
#   docker exec remake_grasp bash /opt/remake_grasp/trials/paired_replay.sh RUN... --out DIR
set -uo pipefail
WT="$(cd "$(dirname "$0")/.." && pwd)"
set +u; . "$WT/docker/env.sh"; set -u
export RG_ROOT="$WT" PYTHONPATH="$WT/ros/remake_grasp:$PYTHONPATH"
exec /opt/venv/bin/python3 -W ignore "$WT/trials/paired_replay.py" "$@"
