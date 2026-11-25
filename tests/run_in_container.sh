#!/usr/bin/env bash
# Run one of this repo's python tests INSIDE the container, with the environment the live nodes have:
#   docker exec [-e RG_RUNS=/opt/remake_grasp/runs] remake_grasp bash /opt/remake_grasp/tests/run_in_container.sh test_x.py
set -uo pipefail
WT="$(cd "$(dirname "$0")/.." && pwd)"
set +u; . "$WT/docker/env.sh"; set -u            # the live nodes' shell: ROS, the workspace, CUDA's libraries
export RG_ROOT="$WT" PYTHONPATH="$WT/ros/remake_grasp:$PYTHONPATH"
exec /opt/venv/bin/python3 -W ignore "$WT/tests/$1"
