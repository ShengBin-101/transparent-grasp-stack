#!/usr/bin/env bash
# Build this workspace (transparent-grasp-stack/nuc).
#
#     ./build.sh                     build everything that can build
#     ./build.sh --clean             wipe build/ and devel/ first
#     ./build.sh --check             verify the existing build, compile nothing
#     ./build.sh fr3_pose_commander  build just that package and its deps
#
# The workspace needs three non-default CMake settings (numbered below). CMake
# caches them in build/, so they vanish silently when build/ is deleted; this
# script passes them on every build.
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
FRANKA_PREFIX="$WS/local_libfranka_0_10_0"
FRANKA_DIR="$FRANKA_PREFIX/usr/lib/cmake/Franka"

# 1. gcc-9 (Noetic's compiler): newer GCC fails on franka_hw with
#    `'uint8_t' does not name a type`, as it never includes <cstdint>.
CC_BIN="${CC_BIN:-/usr/bin/gcc-9}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-9}"

# 2. franka_gazebo (simulation only) does not build against the installed protobuf.
BLACKLIST="${BLACKLIST:-franka_gazebo}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YEL" "$OFF" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

MODE=build
ONLY=""
for a in "$@"; do
    case "$a" in
        --clean) MODE=clean ;;
        --check) MODE=check ;;
        # Help is the header comment block, so it cannot drift from it.
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        -*) die "unknown option '$a' (try --help)" ;;
        *)  ONLY="$a" ;;
    esac
done

[ -d "$WS/src" ] || die "no $WS/src -- is WS set correctly?"
[ -d "$WS/src/franka_ros" ] || die "no $WS/src/franka_ros. Fetch it first: ./setup_franka.sh"
[ -f "$ROS_SETUP" ] || die "no ROS setup at $ROS_SETUP"

for c in "$CC_BIN" "$CXX_BIN"; do
    [ -x "$c" ] || die "$c is missing. Install it (apt install gcc-9 g++-9) or
       set CC_BIN/CXX_BIN. The default compiler breaks franka_hw."
done

# 3. libfranka 0.10.0, built by setup_franka.sh: apt's 0.9.2 is refused by the
#    FR3's FCI server (version 6) as soon as franka_control_node connects.
[ -f "$FRANKA_DIR/FrankaConfig.cmake" ] || die \
"no libfranka 0.10.0 at $FRANKA_PREFIX.
       The apt package (0.9.2) is too old for the FR3's FCI server, so this
       workspace builds its own copy. Build it with:
         ./setup_franka.sh --lib"

cd "$WS"

if [ "$MODE" = clean ]; then
    say "removing build/ and devel/"
    rm -rf build devel
fi

# -- verification, run after every build and on its own with --check ---------
verify() {
    local bad=0 devel="$WS/devel/lib"
    [ -d "$devel" ] || { warn "nothing built yet"; return 1; }

    say "checking what was produced"
    for n in franka_control/franka_control_node \
             franka_gripper/franka_gripper_node \
             libfranka_example_controllers.so \
             fr3_pose_commander/pose_commander_node.py; do
        if [ -e "$devel/$n" ]; then
            printf '    ok    %s\n' "$n"
        else
            printf '    %sMISSING%s  %s\n' "$RED" "$OFF" "$n"; bad=1
        fi
    done

    # Check the soname actually linked: a 0.9 link only shows when the robot refuses it.
    local old
    old=$(find "$devel" -type f -executable -exec sh -c \
        'ldd "$1" 2>/dev/null | grep -q "libfranka\.so\.0\.9" && echo "$1"' _ {} \; 2>/dev/null)
    if [ -n "$old" ]; then
        warn "these link libfranka 0.9, which the FR3 will refuse:"
        printf '      %s\n' $old
        bad=1
    else
        printf '    ok    every binary links libfranka 0.10\n'
    fi

    # The workspace must stand alone: nothing may link another catkin workspace's devel/ or install/.
    local leak
    leak=$(find "$devel" -type f -executable -exec sh -c \
        'ldd "$1" 2>/dev/null | grep -E "/(devel|install)/lib/" | grep -qv "$2/" && echo "$1"' _ {} "$WS" \; 2>/dev/null)
    if [ -n "$leak" ]; then
        warn "these reach outside this workspace:"
        printf '      %s\n' $leak
        bad=1
    else
        printf '    ok    self-contained: nothing links outside this workspace\n'
    fi

    # A cached whitelist silently limits every later build to those packages.
    local wl
    wl=$(grep -oP '(?<=^CATKIN_WHITELIST_PACKAGES:STRING=).*' build/CMakeCache.txt 2>/dev/null || true)
    if [ -n "$wl" ]; then
        warn "CATKIN_WHITELIST_PACKAGES is still set to '$wl'."
        warn "Only those packages will build. Clear it: ./build.sh"
        bad=1
    fi
    return $bad
}

if [ "$MODE" = check ]; then
    verify && say "build looks good" || die "see the warnings above"
    exit 0
fi

# ROS's setup.bash reads unset variables, which `set -u` would abort on.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

ARGS=(-DCMAKE_C_COMPILER="$CC_BIN"
      -DCMAKE_CXX_COMPILER="$CXX_BIN"
      -DFranka_DIR="$FRANKA_DIR"
      -DCATKIN_BLACKLIST_PACKAGES="$BLACKLIST")

if [ -n "$ONLY" ]; then
    say "building '$ONLY' and its dependencies"
    # catkin caches the whitelist, so this is always followed by the reset below.
    catkin_make "${ARGS[@]}" --only-pkg-with-deps "$ONLY"
    say "clearing the package whitelist so the next build is complete again"
    catkin_make "${ARGS[@]}" -DCATKIN_WHITELIST_PACKAGES="" >/dev/null
else
    say "building everything except: $BLACKLIST"
    catkin_make "${ARGS[@]}" -DCATKIN_WHITELIST_PACKAGES=""
fi

echo
verify || die "the build finished but the checks above failed"

# Stamp the commit this build came from; tools/sync_check.sh compares it with
# HEAD on every run_*.sh. "+dirty" means uncommitted edits were compiled in.
if git -C "$WS" rev-parse HEAD >/dev/null 2>&1; then
    stamp="$(git -C "$WS" rev-parse HEAD)"
    [ -n "$(git -C "$WS" status --porcelain -- src)" ] && stamp="$stamp +dirty"
    [ -n "$ONLY" ] && stamp="$stamp only:$ONLY"
    echo "$stamp $(date -Iseconds)" > "$WS/devel/.built_from"
fi
echo
say "done. Source it with:  source $WS/devel/setup.bash"
