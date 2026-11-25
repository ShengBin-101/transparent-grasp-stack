#!/usr/bin/env bash
# Move the arm to capture viewpoint 1-5.   MOVES THE ARM
#
#     ./run_viewpoint.sh 1            straight down over the workspace centre
#     ./run_viewpoint.sh 2            near side (between the base and the centre)
#     ./run_viewpoint.sh 3            far side
#     ./run_viewpoint.sh 4            left (+y)
#     ./run_viewpoint.sh 5            right (-y)
#     ./run_viewpoint.sh 3 --dry-run  plan and show, move nothing
#     ./run_viewpoint.sh 4 --tilt 15 --height 0.55 --centre 0.50 0.05
#
# Defaults: camera 0.50 m above the table, aimed at the workspace centre (0.50, 0.00
# in fr3_link0), side views tilted 20 deg; these keep all five reachable (viewpoints.py).
#
# Run a controller and its commander first (run_position_controller.sh + run_commander.sh
# --position, or run_controller.sh + run_commander.sh). Every leg goes through move_to_pose
# and its checks; an unreachable viewpoint is refused before anything moves.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

case "${1:-}" in
    -h|--help|"") awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
esac

# Warn if this build differs from the other machine's (tools/sync_check.sh).
source "$(dirname "${BASH_SOURCE[0]}")/tools/sync_check.sh" && sync_check

[ -f "$WS/devel/setup.bash" ] || die "$WS is not built. Run: $WS/build.sh"
set +u; source /opt/ros/noetic/setup.bash; source "$WS/devel/setup.bash"; set -u
timeout 3 rostopic list >/dev/null 2>&1 \
    || die "no ROS master. Start run_position_controller.sh first."
exec python3 "$(dirname "${BASH_SOURCE[0]}")/tools/goto_viewpoint.py" "$@"
