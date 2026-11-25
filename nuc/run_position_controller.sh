#!/usr/bin/env bash
# Terminal 1, POSITION-CONTROL variant: the Cartesian pose tracking controller. Headless.
#
#     ./run_position_controller.sh           pose controller + gripper
#     ./run_position_controller.sh --check   run the preflight and exit
#
# Use INSTEAD of ./run_controller.sh (never both: each claims the arm), then start the
# commander pointed at it:
#
#     ./run_commander.sh --position --dry-run     rehearse
#     ./run_commander.sh --position               for real
#
# Drives the robot's own Cartesian pose interface, which its stiff internal joint
# impedance tracks to about a millimetre, where a soft impedance controller stalls
# short on joint friction. Trade-offs: src/fr3_position_control/launch/position_controller.launch.
#
# Run run_home.sh first: the controller latches the elbow (joint 3's angle, joint 4's
# flip sign) at start and holds it on every leg. HOLD_ELBOW=false sends the pose only
# and lets the firmware pick joint 3.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
ARM="${ARM:-fr3}"
SPAWN_DELAY="${SPAWN_DELAY:-5}"
HOLD_ELBOW="${HOLD_ELBOW:-true}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

CHECK_ONLY=false
for a in "$@"; do
    case "$a" in
        --check) CHECK_ONLY=true ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) die "unknown option '$a' (try --help)" ;;
    esac
done

case "$HOLD_ELBOW" in
    true|false) ;;
    *) die "HOLD_ELBOW must be true or false, not '$HOLD_ELBOW'" ;;
esac

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"

say "robot at $ROBOT_IP"
if ping -c1 -W2 "$ROBOT_IP" >/dev/null 2>&1; then
    printf '    ok    reachable\n'
else
    die "no ping to $ROBOT_IP. The arm is off, or this host is not on 172.16.0.0/24,
       or the cable is out."
fi
if timeout 3 bash -c "</dev/tcp/$ROBOT_IP/1337" 2>/dev/null; then
    printf '    ok    FCI port 1337 open\n'
else
    die "FCI port 1337 refused. Activate FCI in Desk at https://$ROBOT_IP"
fi

# The link must sustain FCI's 1 kHz loop (tools/fci_link.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/fci_link.sh"
fci_link_check "$ROBOT_IP" || exit 1

set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u

# Without the built plugin the spawner fails 5 s in with an unknown controller type.
if rospack find fr3_position_control >/dev/null 2>&1 \
   && [ -f "$WS/devel/lib/libfr3_position_control.so" ]; then
    printf '    ok    fr3_position_control built\n'
else
    die "fr3_position_control is not built. Run: $WS/build.sh fr3_position_control"
fi

# Two controllers on one arm is the failure to rule out before anything moves.
if timeout 3 rostopic list >/dev/null 2>&1 \
   && rosservice list 2>/dev/null | grep -q '^/controller_manager/list_controllers$'; then
    die "a controller_manager is already running (run_controller.sh?). Stop it first:
       one arm, one controller."
fi

[ "$CHECK_ONLY" = true ] && { say "preflight only; nothing launched"; exit 0; }

say "Cartesian POSE controller + gripper (headless)"
say "command topic: /cartesian_pose_tracking_controller/target_pose"
say "elbow: hold_elbow=$HOLD_ELBOW (the controller logs the latched values at start)"
say "then: ./run_commander.sh --position   (add --dry-run to rehearse)"
exec roslaunch fr3_position_control position_controller.launch \
    robot_ip:="$ROBOT_IP" robot:="$ARM" load_gripper:=true \
    spawn_delay:="$SPAWN_DELAY" hold_elbow:="$HOLD_ELBOW"
