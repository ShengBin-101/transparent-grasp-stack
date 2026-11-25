#!/usr/bin/env bash
# Terminal 1, BEFORE the controller: go to the known capture pose.
#
#     ./run_home.sh --check      preflight only; commands nothing
#     ./run_home.sh              move to the saved capture pose (asks first)
#     ./run_home.sh --yes        the same, without the prompt
#     ./run_home.sh --save       save where the arm is NOW as the capture pose
#     MAX_DQ=0.1 ./run_home.sh   slower (rad/s for the furthest joint)
#
# The pose is joint angles in config/home_pose.yaml, so a hand-found posture can
# be kept with --save.
#
# Run this before run_controller.sh: the controller latches and holds the posture it starts in.
# FCI accepts one connection: stop the controller first (this launch exits when the move ends).
# MOVES THE ARM in joint space with nothing checking the path (straight interpolation).
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
ARM="${ARM:-fr3}"
HOME_YAML="${HOME_YAML:-$WS/src/fr3_pose_commander/config/home_pose.yaml}"
# echo_robot_state comes from the libfranka that setup_franka.sh builds.
LIBFRANKA_PREFIX="${LIBFRANKA_PREFIX:-$WS/local_libfranka_0_10_0}"
ECHO_STATE="${ECHO_STATE:-$LIBFRANKA_PREFIX/usr/bin/echo_robot_state}"
MAX_DQ="${MAX_DQ:-0.2}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

CHECK_ONLY=false; SAVE=false; ASSUME_YES=false
for a in "$@"; do
    case "$a" in
        --check) CHECK_ONLY=true ;;
        --save) SAVE=true ;;
        -y|--yes) ASSUME_YES=true ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) die "unknown option '$a' (try --help)" ;;
    esac
done

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"
[ -f "$HOME_YAML" ] || die "no capture pose at $HOME_YAML"

# The controller that interpolates; load_controller would only say "ok: False".
if [ "$CHECK_ONLY" != true ] && [ "$SAVE" != true ] \
        && [ ! -d /opt/ros/noetic/share/joint_trajectory_controller ]; then
    die "joint_trajectory_controller is not installed, so nothing here can
    interpolate a joint move inside the 1 kHz control loop. Install it:

        sudo apt install ros-noetic-joint-trajectory-controller

    Then this works unchanged. Until then the arm can only be moved by hand
    (enabling device + guiding button) -- and NOT by publishing joint commands
    from a ROS node, which is what jittered the arm on 2026-09-16."
fi

say "robot at $ROBOT_IP"
ping -c1 -W2 "$ROBOT_IP" >/dev/null 2>&1 \
    || die "no ping to $ROBOT_IP. The arm is off, the cable is out, or this
       host is not on 172.16.0.0/24."
printf '    ok    reachable\n'

timeout 3 bash -c "</dev/tcp/$ROBOT_IP/1337" 2>/dev/null \
    || die "FCI port 1337 refused. Activate FCI in Desk at https://$ROBOT_IP"
printf '    ok    FCI port 1337 open\n'

# The link must sustain FCI's 1 kHz loop (tools/fci_link.sh); --save only reads, so it only warns.
source "$(dirname "${BASH_SOURCE[0]}")/tools/fci_link.sh"
fci_link_check "$ROBOT_IP" "$([ "$SAVE" = true ] && echo warn || echo "${FCI_LINK_CHECK:-refuse}")" || exit 1

# ---------------------------------------------------------------- --save
# Reads the arm straight off FCI, not ROS, so it works after hand-guiding with nothing running.
if [ "$SAVE" = true ]; then
    if pgrep -f "[f]ranka_control" >/dev/null 2>&1; then
        die "franka_control holds the FCI connection, so the arm cannot be read
       here. Stop run_controller.sh and try again."
    fi
    [ -x "$ECHO_STATE" ] || die "no echo_robot_state at $ECHO_STATE"
    say "reading the arm"
    STATE_FILE="$(mktemp --suffix=.json)"
    trap 'rm -f "$STATE_FILE"' EXIT
    LD_LIBRARY_PATH="$LIBFRANKA_PREFIX/usr/lib:${LD_LIBRARY_PATH:-}" \
        timeout 5 "$ECHO_STATE" "$ROBOT_IP" 2>/dev/null | head -1 \
        > "$STATE_FILE" || die "could not read the robot state"
    python3 - "$HOME_YAML" "$STATE_FILE" <<'EOF'
import json, re, sys
yaml_path, state_path = sys.argv[1], sys.argv[2]
q = [float(v) for v in json.load(open(state_path))["q"]]
print("    q = [%s]" % ", ".join("%.6f" % v for v in q))
text = open(yaml_path).read()
new = re.sub(r"(?m)^home_joints:\s*\[[^\]]*\]",
             "home_joints: [%s]" % ", ".join("%.9f" % v for v in q), text, count=1)
if new == text:
    raise SystemExit("could not find a home_joints line to replace in " + yaml_path)
open(yaml_path, "w").write(new)
print("    saved to " + yaml_path)
EOF
    warn "THE POSTURE IS NOT CHECKED. The old workspace refused to save a pose"
    warn "the pick would later refuse, by measuring the Jacobian's smallest"
    warn "singular value; that analysis has not been ported here yet. A pose"
    warn "that looks fine can still be one the arm has no authority to work"
    warn "from. Pick it with the scene in view and verify by doing a pick."
    say "capture pose saved. Copy it back to the repo so the next sync keeps it."
    exit 0
fi

# ---------------------------------------------------------------- the target
# Regexes are anchored to line starts so a commented-out value in home_pose.yaml is never read.
say "target, from ${HOME_YAML#$HOME/}"
python3 - "$HOME_YAML" <<'EOF' || die "could not read the capture pose"
import re, sys
text = open(sys.argv[1]).read()
q = [float(v) for v in re.search(r"(?m)^home_joints:\s*\[([^\]]+)\]", text).group(1).split(",")]
if len(q) != 7:
    raise SystemExit("home_joints has %d values, expected 7" % len(q))
print("    q = [%s]" % ", ".join("%.4f" % v for v in q))
EOF

# Read the robot mode up front rather than meet it as a rejected Move command.
ROBOT_MODE=unknown
if [ -x "$ECHO_STATE" ] && ! pgrep -f "[f]ranka_control" >/dev/null 2>&1; then
    ROBOT_MODE=$(LD_LIBRARY_PATH="$LIBFRANKA_PREFIX/usr/lib:${LD_LIBRARY_PATH:-}" \
        timeout 4 "$ECHO_STATE" "$ROBOT_IP" 2>/dev/null | head -1 \
        | grep -o '"robot_mode": "[^"]*"' | cut -d'"' -f4) || true
    [ -n "$ROBOT_MODE" ] || ROBOT_MODE=unknown
    printf '    ok    robot_mode = %s\n' "$ROBOT_MODE"
else
    warn "could not read the robot mode (no echo_robot_state, or franka_control holds FCI)"
fi

case "$ROBOT_MODE" in
    Idle|Move|unknown) ;;
    "User stopped") die "the user stop is engaged. Release it, then run this again." ;;
    *) die "robot_mode is '$ROBOT_MODE', not Idle or Move. Open the joints in
       Desk (https://$ROBOT_IP)." ;;
esac

if pgrep -f "[f]ranka_control" >/dev/null 2>&1; then
    die "franka_control is running and holds the FCI connection. Stop
       run_controller.sh first -- the robot allows one connection."
fi
printf '    ok    nothing else holds FCI\n'

[ "$CHECK_ONLY" = true ] && { say "preflight only; nothing launched"; exit 0; }

set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

if [ "$ASSUME_YES" != true ]; then
    warn "This MOVES THE ARM in joint space, with nothing checking the path."
    warn "Clear the workspace and keep a hand on the stop."
    read -r -p "    type 'move' to continue: " reply
    [ "$reply" = move ] || { say "nothing commanded"; exit 0; }
fi

# move_to_start.py takes its goal as rosparams, so the pose is passed as launch args.
say "homing at ${MAX_DQ} rad/s"
eval "$(python3 - "$HOME_YAML" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
q = [float(v) for v in re.search(r"(?m)^home_joints:\s*\[([^\]]+)\]", text).group(1).split(",")]
print(" ".join("q%d=%.9f" % (i + 1, v) for i, v in enumerate(q)))
EOF
)"
exec roslaunch fr3_pose_commander home.launch \
    robot_ip:="$ROBOT_IP" robot:="$ARM" transmission:=position max_dq:="$MAX_DQ" \
    q1:="$q1" q2:="$q2" q3:="$q3" q4:="$q4" q5:="$q5" q6:="$q6" q7:="$q7"
