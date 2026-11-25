#!/bin/bash
# Stop every pipeline process in the container (tmuxp runs this before starting a session).
# Run inside the container. From the host, ./start.sh --stop stops everything at once.
PATTERN='ros2 launch remake_grasp|/lib/remake_grasp/|realsense2_camera_node|component_container|foxglove_bridge|parameter_bridge'
pgrep -f "$PATTERN" >/dev/null || exit 0
pkill -INT -f "$PATTERN"
for _ in 1 2 3 4 5; do pgrep -f "$PATTERN" >/dev/null || exit 0; sleep 1; done
pkill -TERM -f "$PATTERN"; sleep 2
pkill -KILL -f "$PATTERN" 2>/dev/null
exit 0
