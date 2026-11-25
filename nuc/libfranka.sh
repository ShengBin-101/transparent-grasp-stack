#!/usr/bin/env bash
# Run a libfranka example (built by setup_franka.sh) against the arm.
#
#     ./libfranka.sh --list                    what is available, and what it does
#     ./libfranka.sh echo_robot_state          read state, touch nothing
#     ./libfranka.sh communication_test        MOVES THE ARM; asks first
#     ./libfranka.sh print_joint_poses 172.16.0.2
#     ./libfranka.sh --env                     print the export line for your shell
#
# The examples link a libfranka that is deliberately not on the loader
# path (so nothing picks up apt's 0.9.2), so run directly they fail with
# "libfranka.so.0.10: cannot open shared object file". This sets LD_LIBRARY_PATH,
# says which examples move the arm, and asks before running one that does.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PREFIX="$WS/local_libfranka_0_10_0"
LIB="$PREFIX/usr/lib"
BIN="$PREFIX/usr/bin"
ROBOT_IP_DEFAULT="${ROBOT_IP:-172.16.0.2}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

# What each example does, read from the libfranka sources; anything unlisted
# is treated as moving the arm.
classify() {
    case "$1" in
        echo_robot_state|print_joint_poses)          echo "read-only" ;;
        grasp_object|vacuum_object)                  echo "gripper-only" ;;
        cartesian_impedance_control|force_control)   echo "holds-position" ;;
        *)                                           echo "moves-arm" ;;
    esac
}

describe() {
    case "$(classify "$1")" in
        read-only)      echo "reads state; the arm is not touched" ;;
        gripper-only)   echo "moves the GRIPPER only; the arm is not touched" ;;
        holds-position) echo "torque control holding the start pose; back-drivable, no commanded motion" ;;
        moves-arm)      echo "MOVES THE ARM" ;;
    esac
}

[ -d "$BIN" ] || die "no libfranka at $PREFIX. Build it: ./setup_franka.sh --lib"

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0
fi

if [ "${1:-}" = "--env" ]; then
    echo "export LD_LIBRARY_PATH=\"$LIB:\${LD_LIBRARY_PATH:-}\""
    exit 0
fi

if [ "${1:-}" = "--list" ]; then
    say "examples in $BIN"
    for b in $(ls "$BIN" | sort); do
        case "$(classify "$b")" in
            moves-arm) c="$RED" ;;
            holds-position) c="$YEL" ;;
            *) c="$GRN" ;;
        esac
        printf "  %s%-36s%s %s\n" "$c" "$b" "$OFF" "$(describe "$b")"
    done
    echo
    warn "communication_test moves to franka's READY pose [0 -pi/4 0 -3pi/4 0 pi/2 pi/4],"
    warn "which is NOT the capture pose in config/home_pose.yaml. Re-home afterwards."
    exit 0
fi

ASSUME_YES=false
ARGS=()
for a in "$@"; do
    case "$a" in
        -y|--yes) ASSUME_YES=true ;;
        *) ARGS+=("$a") ;;
    esac
done
set -- "${ARGS[@]}"

EXAMPLE="$1"; shift
[ -x "$BIN/$EXAMPLE" ] || die "no example '$EXAMPLE'. Try: $0 --list"
IP="${1:-$ROBOT_IP_DEFAULT}"

say "$EXAMPLE -> $IP"
printf '    %s\n' "$(describe "$EXAMPLE")"

ping -c1 -W2 "$IP" >/dev/null 2>&1 || die "no ping to $IP. Is the arm on and the cable in?"
printf '    ok    reachable\n'
if timeout 3 bash -c "</dev/tcp/$IP/1337" 2>/dev/null; then
    printf '    ok    FCI port 1337 open\n'
else
    die "FCI port 1337 refused. Activate FCI in Desk at https://$IP"
fi

# FCI accepts one connection; franka_control would make this fail with an unclear error.
if pgrep -x franka_control_node >/dev/null 2>&1; then
    die "franka_control_node is running and holds the FCI connection.
       Only one client may connect at a time. Stop the controller first,
       and remember to restart it afterwards."
fi
printf '    ok    FCI is free\n'

# The link must sustain FCI's 1 kHz loop (tools/fci_link.sh); read-only examples only warn.
source "$WS/tools/fci_link.sh"
fci_link_check "$IP" "$([ "$(classify "$EXAMPLE")" = read-only ] && echo warn || echo "${FCI_LINK_CHECK:-refuse}")" || exit 1

if [ "$(classify "$EXAMPLE")" = "moves-arm" ] && [ "$ASSUME_YES" = false ]; then
    echo
    warn "THIS WILL MOVE THE ARM. Hand on the stop button."
    [ "$EXAMPLE" = "communication_test" ] && \
        warn "It moves to franka's ready pose first, NOT your capture pose."
    read -r -p "    type 'yes' to continue: " reply
    [ "$reply" = "yes" ] || die "aborted"
fi

exec env LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" "$BIN/$EXAMPLE" "$IP" "$@"
