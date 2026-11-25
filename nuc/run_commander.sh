#!/usr/bin/env bash
# Terminal 2: the node that turns a desired pose into arm motion.
#
#     ./run_commander.sh --dry-run    resolve and check everything, move nothing
#     ./run_commander.sh --position   drive fr3_position_control (run_position_controller.sh)
#                                     instead of the impedance controller
#     ./run_commander.sh              for real
#     ./run_commander.sh --no-checks  every refusal becomes a warning and the arm
#                                     goes anyway (the ramp still applies)
#
# Run run_controller.sh (or run_position_controller.sh) first: without its
# franka_states every pose is refused for want of robot state.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASE_FRAME="${BASE_FRAME:-fr3_link0}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

DRY=false; CHECKS=true
COMMAND_TOPIC=/cartesian_impedance_example_controller/equilibrium_pose
Z_MIN=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=true ;;
        --no-checks) CHECKS=false ;;
        # z_min 5 mm above the table (z = -0.041); equal to the config's, kept explicit.
        --position) COMMAND_TOPIC=/cartesian_pose_tracking_controller/target_pose
                    Z_MIN=-0.036 ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) die "unknown option '$a' (try --help)" ;;
    esac
done

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"
set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u

# The planner here and the Orin's sequencer each keep a copy of the hand-eye; they must match.
MINE=$(grep -E '^ee_to_camera:' "$(rospack find fr3_pose_commander)/config/camera_extrinsics.yaml" | tr -d ' ')
ORIN="$WS/../ros/remake_grasp/config/eye_in_hand.yaml"
THEIRS=$( { grep -E '^ee_to_camera:' "$ORIN" 2>/dev/null || true; } | tr -d ' ')
if [ -z "$THEIRS" ]; then
    warn "no ee_to_camera in $ORIN to compare with (not a full transparent-grasp-stack clone?)"
elif [ "$MINE" = "$THEIRS" ]; then
    printf '    ok    ee_to_camera matches the Orin'"'"'s eye_in_hand.yaml\n'
else
    warn "ee_to_camera DIFFERS from the Orin's eye_in_hand.yaml:"
    warn "  here (camera_extrinsics.yaml): $MINE"
    warn "  Orin (eye_in_hand.yaml):       $THEIRS"
    warn "One camera, two calibrations. Fix before moving the arm."
fi

if timeout 3 rostopic list >/dev/null 2>&1; then
    printf '    ok    ROS master reachable\n'
    if timeout 5 rostopic echo -n1 /franka_state_controller/franka_states >/dev/null 2>&1; then
        printf '    ok    franka_states is publishing\n'
    else
        warn "no franka_states. Start run_controller.sh first (--state-only works"
        warn "with the arm locked). Without it every pose refuses: 'no robot state'."
    fi
else
    warn "no ROS master yet - roslaunch will start one. If run_controller.sh is"
    warn "meant to be running, start it first so they share a master."
fi

if [ "$DRY" = true ]; then
    say "DRY RUN: every refusal and every resolved pose is real; nothing moves"
else
    warn "LIVE: this will move the arm. Hand on the stop."
fi

say "setpoints -> $COMMAND_TOPIC${Z_MIN:+   (z_min $Z_MIN)}"
exec roslaunch fr3_pose_commander pose_commander.launch \
    base_frame:="$BASE_FRAME" dry_run:="$DRY" checks:="$CHECKS" \
    command_topic:="$COMMAND_TOPIC" z_min:="$Z_MIN"
