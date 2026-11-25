#!/usr/bin/env bash
# Terminal 3, optional: the websocket bridges, for watching the arm from another machine.
# The pipeline does not need them: the Orin reaches the NUC through ros1_bridge and the master.
#
#     ./run_bridge.sh                 foxglove_bridge + rosbridge + the URDF/mesh server
#     ./run_bridge.sh --no-description  without the URDF/mesh server
#     PUBLIC_HOST=10.0.0.5 ./run_bridge.sh   if clients reach this box by some other address
#
# Foxglove:            ws://<this host>:8765, connection type "Foxglove WebSocket"
# other websocket clients: rosbridge on ws://<this host>:9090; a URDF for them is served at
#                      http://<this host>:8080/robot_description.urdf, because its package://
#                      mesh paths mean nothing to a browser on another machine
#                      (scripts/serve_description.py).
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PORT="${PORT:-9090}"
HTTP_PORT="${HTTP_PORT:-8080}"
PUBLIC_HOST="${PUBLIC_HOST:-}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

DESCRIPTION=true
for a in "$@"; do
    case "$a" in
        --no-description) DESCRIPTION=false ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) die "unknown option '$a' (try --help)" ;;
    esac
done

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"
set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u

ADDR="${PUBLIC_HOST:-${ROS_IP:-${ROS_HOSTNAME:-$(hostname -I | awk '{print $1}')}}}"
say "rosbridge  ws://$ADDR:$PORT        (websocket clients; Foxglove uses foxglove_bridge on 8765)"
if [ "$DESCRIPTION" = true ]; then
    say "URDF       http://$ADDR:$HTTP_PORT/robot_description.urdf"
    say "           Foxglove 3D panel -> add layer -> URDF -> that URL"
    warn "the URDF server waits for /robot_description, which run_controller.sh"
    warn "publishes. Without the controller it will sit and wait."
fi

exec roslaunch fr3_pose_commander bridge.launch \
    port:="$PORT" http_port:="$HTTP_PORT" \
    description:="$DESCRIPTION" public_host:="$PUBLIC_HOST"
