#!/usr/bin/env bash
# Fetch the two Franka dependencies at the versions every reported run used.
#
#     ./setup_franka.sh          both, then run ./build.sh
#     ./setup_franka.sh --ros    franka_ros 0.10.1 into src/franka_ros, with the stiffness patch
#     ./setup_franka.sh --lib    libfranka 0.10.0, built from source into local_libfranka_0_10_0/usr
#
# franka_ros: tag 0.10.1 (commit 30e598a), plus patches/franka_ros-0.10.1-stiffness.patch, which
# raises two dynamic_reconfigure bounds so controller.launch's 1000 N/m and 50 N m/rad are applied
# (read the patch header). The tools and tests also read franka_description from here, so --ros is
# enough on a machine that only replays plans.
#
# libfranka: tag 0.10.0 (commit 4f9e3cc). apt's ros-noetic-libfranka is 0.9.2, which the FR3's FCI
# server (version 6) refuses. Needs: build-essential cmake git libpoco-dev libeigen3-dev, and gcc-9
# on Noetic (CC_BIN/CXX_BIN override it, as in build.sh).
set -euo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FRANKA_ROS_URL="${FRANKA_ROS_URL:-https://github.com/frankaemika/franka_ros.git}"
FRANKA_ROS_TAG=0.10.1
FRANKA_ROS_SHA=30e598aa6fb703cc80a203481e6427f397337b4c
LIBFRANKA_URL="${LIBFRANKA_URL:-https://github.com/frankaemika/libfranka.git}"
LIBFRANKA_TAG=0.10.0
LIBFRANKA_SHA=4f9e3cc666e42d267f1ab566869c4f4c552e5b57
PATCH="$WS/patches/franka_ros-0.10.1-stiffness.patch"
PREFIX="$WS/local_libfranka_0_10_0"
CC_BIN="${CC_BIN:-/usr/bin/gcc-9}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-9}"

RED=$'\033[31m'; GRN=$'\033[32m'; OFF=$'\033[0m'
say() { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
die() { printf '%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

DO_ROS=true; DO_LIB=true
case "${1:-}" in
    "") ;;
    --ros) DO_LIB=false ;;
    --lib) DO_ROS=false ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
esac

# Clone a tag into $2 and check that it is the expected commit.
fetch() {
    local url="$1" dest="$2" tag="$3" sha="$4" extra="${5:-}"
    if [ -d "$dest" ]; then
        local have
        have="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
        [ "$have" = "$sha" ] || die "$dest exists but is not $tag ($sha). Move it away and rerun."
        say "$dest is already $tag"
        return 1
    fi
    say "cloning $url at $tag"
    # shellcheck disable=SC2086
    git -c advice.detachedHead=false clone --quiet --depth 1 $extra -b "$tag" "$url" "$dest"
    [ "$(git -C "$dest" rev-parse HEAD)" = "$sha" ] || die "$tag in $url is not $sha"
}

if $DO_ROS; then
    if fetch "$FRANKA_ROS_URL" "$WS/src/franka_ros" "$FRANKA_ROS_TAG" "$FRANKA_ROS_SHA"; then
        git -C "$WS/src/franka_ros" apply "$PATCH"
        say "applied $(basename "$PATCH")"
    elif git -C "$WS/src/franka_ros" apply --reverse --check "$PATCH" 2>/dev/null; then
        say "the stiffness patch is already applied"
    else
        die "src/franka_ros is $FRANKA_ROS_TAG but the stiffness patch is not applied:
       git -C $WS/src/franka_ros apply $PATCH"
    fi
fi

if $DO_LIB; then
    for c in "$CC_BIN" "$CXX_BIN"; do
        [ -x "$c" ] || die "$c is missing: apt install gcc-9 g++-9, or set CC_BIN/CXX_BIN."
    done
    if [ -f "$PREFIX/usr/lib/cmake/Franka/FrankaConfig.cmake" ]; then
        say "libfranka is already installed in $PREFIX"
    else
        SRC="$(mktemp -d)"
        trap 'rm -rf "$SRC"' EXIT
        fetch "$LIBFRANKA_URL" "$SRC/libfranka" "$LIBFRANKA_TAG" "$LIBFRANKA_SHA" "--recursive --shallow-submodules" || true
        say "building libfranka $LIBFRANKA_TAG (a few minutes)"
        # LIBDIR=lib, not the multiarch default, because build.sh looks in usr/lib/cmake/Franka.
        cmake -S "$SRC/libfranka" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="$CC_BIN" -DCMAKE_CXX_COMPILER="$CXX_BIN" \
            -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
            -DCMAKE_INSTALL_PREFIX="$PREFIX/usr" -DCMAKE_INSTALL_LIBDIR=lib >/dev/null
        cmake --build "$SRC/build" -j"$(nproc)"
        cmake --install "$SRC/build" >/dev/null
        cp "$SRC/libfranka/LICENSE" "$SRC/libfranka/NOTICE" "$PREFIX/"
        say "installed libfranka $LIBFRANKA_TAG in $PREFIX"
    fi
fi

say "done. Next: ./build.sh"
