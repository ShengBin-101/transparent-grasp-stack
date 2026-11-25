#!/usr/bin/env bash
# Terminal 1: the cartesian impedance controller. Headless.
#
#     ./run_controller.sh              controller + gripper
#     ./run_controller.sh --state-only franka_states only, no motion controller
#     ./run_controller.sh --check      run the preflight and exit
#
# Run run_home.sh first: the nullspace gain holds whatever posture the arm is in
# when the controller starts.
#
# --state-only is for bench tests with the brakes closed: a motion controller's
# Move command is rejected then and takes the whole launch down, while
# franka_state_controller alone only reads.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
ARM="${ARM:-fr3}"
SPAWN_DELAY="${SPAWN_DELAY:-5}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

MODE=full; CHECK_ONLY=false
for a in "$@"; do
    case "$a" in
        --state-only) MODE=state ;;
        --check) CHECK_ONLY=true ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) die "unknown option '$a' (try --help)" ;;
    esac
done

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"

say "robot at $ROBOT_IP"
if ping -c1 -W2 "$ROBOT_IP" >/dev/null 2>&1; then
    printf '    ok    reachable\n'
else
    die "no ping to $ROBOT_IP.
       The arm is off, or this host is not on 172.16.0.0/24, or the cable is
       out. Nothing in this workspace can talk to the robot until that is
       fixed. The rest of the pipeline can still be exercised without it:
         ./run_bridge.sh                    the Foxglove and websocket bridges
         ./run_commander.sh --dry-run       poses resolved and checked, no motion
       Both need a master, so start one first: roscore &"
fi

# FCI listens on 1337 only once activated in Desk; checked here for a clear error.
if timeout 3 bash -c "</dev/tcp/$ROBOT_IP/1337" 2>/dev/null; then
    printf '    ok    FCI port 1337 open\n'
else
    die "FCI port 1337 refused. Activate FCI in Desk at https://$ROBOT_IP"
fi

# The link must sustain FCI's 1 kHz loop (tools/fci_link.sh); --state-only only warns.
source "$(dirname "${BASH_SOURCE[0]}")/tools/fci_link.sh"
fci_link_check "$ROBOT_IP" "$([ "$MODE" = state ] && echo warn || echo "${FCI_LINK_CHECK:-refuse}")" || exit 1

[ "$CHECK_ONLY" = true ] && { say "preflight only; nothing launched"; exit 0; }

set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u

if [ "$MODE" = state ]; then
    say "franka_states only - no motion controller, safe with the arm locked"
    exec roslaunch franka_control franka_control.launch \
        robot_ip:="$ROBOT_IP" load_gripper:=true robot:="$ARM"
fi

say "cartesian impedance controller + gripper (headless)"
say "stiffness lives in rqt_reconfigure; ./run_bridge.sh then Foxglove to watch it"
exec roslaunch fr3_pose_commander controller.launch \
    robot_ip:="$ROBOT_IP" robot:="$ARM" load_gripper:=true \
    spawn_delay:="$SPAWN_DELAY"
