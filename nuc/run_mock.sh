#!/usr/bin/env bash
# The whole stack with NO ROBOT, for building and checking the Foxglove view.
#
#     ./run_mock.sh              mock arm + bridges + commander (dry run)
#     ./run_mock.sh --still      hold a fixed posture instead of sweeping
#     ./run_mock.sh --no-commander   just the arm and the bridges
#
# Touches no hardware: the commander runs in dry run, resolving and checking
# poses without publishing to a controller. The mock arm sweeps gently but does
# not follow commands; the commander's log shows what it would send.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ARM="${ARM:-fr3}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

ANIMATE=true; COMMANDER=true
for a in "$@"; do
    case "$a" in
        --still) ANIMATE=false ;;
        --no-commander) COMMANDER=false ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) die "unknown option '$a' (try --help)" ;;
    esac
done

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"
set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u

if ping -c1 -W1 172.16.0.2 >/dev/null 2>&1; then
    warn "the robot at 172.16.0.2 answers. This script ignores it and publishes"
    warn "a MOCK posture; what you see in Foxglove is not the real arm's pose."
fi

ADDR="${ROS_IP:-${ROS_HOSTNAME:-$(hostname -I | awk '{print $1}')}}"
say "no robot: publishing a mock posture on /joint_states"
say "Foxglove  ->  ws://$ADDR:8765      (connection type: Foxglove WebSocket)"
say "rosbridge ->  ws://$ADDR:9090      (other websocket clients)"
say "the 3D panel resolves the meshes through foxglove_bridge; nothing else needed"

exec roslaunch fr3_pose_commander mock_bringup.launch \
    robot:="$ARM" animate:="$ANIMATE" commander:="$COMMANDER"
