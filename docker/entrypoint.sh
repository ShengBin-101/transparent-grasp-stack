#!/bin/bash
# Container start-up: ssh keys for the NUC (only with RG_SSH=1 in start.sh), library cache for the
# host's CUDA, output folders.
if [ -d /host-ssh ]; then
    mkdir -p /root/.ssh
    cp -r /host-ssh/. /root/.ssh/
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/* 2>/dev/null
fi
ldconfig 2>/dev/null
mkdir -p /opt/remake_grasp/data/live /opt/remake_grasp/runs
exec "$@"
