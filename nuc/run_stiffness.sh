#!/usr/bin/env bash
# The stiffness, from a terminal. No rqt_reconfigure, no display.
#
#     ./run_stiffness.sh                    show the current gains
#     ./run_stiffness.sh rotational 80      set one (trans | rot | null also work)
#     ./run_stiffness.sh --watch            live tracking error, gains, conditioning
#     ./run_stiffness.sh --reset            back to what controller.launch sets
#
# Stall diagnostic: with --watch in one shell, raise a gain in another. If the
# error shrinks it was gain or friction; if not, the posture lacks authority in
# that direction, so re-home rather than raise the gain further.
#
# Talks to the controller's dynamic_reconfigure server, so it works over SSH.
# Values are clamped to the server's range, not rejected: run with no arguments
# afterwards to see what took. Changes last until the controller restarts.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONTROLLER="${CONTROLLER:-cartesian_impedance_example_controller}"
NODE="/$CONTROLLER/dynamic_reconfigure_compliance_param_node"
TIMEOUT="${TIMEOUT:-10}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"
set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u

timeout 3 rostopic list >/dev/null 2>&1 \
    || die "no ROS master. Start run_controller.sh first."

# Look for the service, not a node: the server lives on the controller's NodeHandle
# and never appears in `rosnode list`.
if ! rosservice list 2>/dev/null | grep -q "^$NODE/set_parameters$"; then
    die "no $NODE.
       The controller is not loaded, so there are no gains to read. Start
       run_controller.sh and give it the spawn delay, then try again."
fi

show() {
    say "gains on $CONTROLLER"
    for n in translational_stiffness rotational_stiffness nullspace_stiffness; do
        printf '    %-24s %s\n' "$n" "$(rosparam get "$NODE/$n" 2>/dev/null || echo '--')"
    done
    printf '    %s\n' "damping is derived: D = 2*sqrt(K), so raising K raises it too"
}

case "${1:-}" in
    "")        show; exit 0 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    --watch)
        say "watching. Ctrl-C to stop; set gains from another shell."
        exec rosrun fr3_pose_commander watch_tracking.py ;;
    --reset)
        say "back to controller.launch's values"
        rosrun dynamic_reconfigure dynparam set --timeout="$TIMEOUT" "$NODE" translational_stiffness 1000
        rosrun dynamic_reconfigure dynparam set --timeout="$TIMEOUT" "$NODE" rotational_stiffness 50
        rosrun dynamic_reconfigure dynparam set --timeout="$TIMEOUT" "$NODE" nullspace_stiffness 2
        show; exit 0 ;;
esac

[ $# -eq 2 ] || die "give a parameter and a value, e.g. ./run_stiffness.sh rotational 80
       (or no arguments to show, --watch to watch, --reset to restore)"

case "$1" in
    trans|translational|translational_stiffness) PARAM=translational_stiffness ;;
    rot|rotational|rotational_stiffness)         PARAM=rotational_stiffness ;;
    null|nullspace|nullspace_stiffness)          PARAM=nullspace_stiffness ;;
    *) die "unknown parameter '$1'. One of: translational, rotational, nullspace." ;;
esac

case "$2" in
    ''|*[!0-9.]*) die "'$2' is not a number" ;;
esac

# The cfg's range is 0-1000 N/m and dynamic_reconfigure clamps silently, so warn.
if [ "$PARAM" = translational_stiffness ] && [ "${2%%.*}" -gt 1000 ] 2>/dev/null; then
    warn "$2 N/m is above the controller's 1000 N/m ceiling; it will run as 1000."
fi

say "$PARAM -> $2"
rosrun dynamic_reconfigure dynparam set --timeout="$TIMEOUT" "$NODE" "$PARAM" "$2"
show
