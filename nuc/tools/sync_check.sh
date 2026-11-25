#!/usr/bin/env bash
# Is the code about to run here committed, and built from what is checked out?
# Sourced by the run_*.sh launchers (build.sh stamps the commit it compares).
#
#     source tools/sync_check.sh && sync_check      from a launch script
#     tools/sync_check.sh                           by hand
#
# GitHub is the master: changes travel as commits (see "4. Keep the NUC
# current" in README.md). Offline, it checks for uncommitted edits under nuc/ and
# whether devel/.built_from (written by build.sh) matches HEAD, since a pull
# rebuilds nothing. It only warns and always returns 0: a robot session must never
# be blocked by git. It does not fetch; `git pull` before a session.

sync_check() {
    local root yel off
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    yel=$'\033[33m'; off=$'\033[0m'
    _sc_warn() { printf '%s!!!%s sync: %s\n' "$yel" "$off" "$*" >&2; }
    _sc_ok()   { printf '    ok    sync: %s\n' "$*"; }

    if ! git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        _sc_warn "$root is not a git checkout; nothing is tracking which version runs here"
        return 0
    fi

    # "-- ." keeps this to nuc/: Orin-side edits change nothing that runs here.
    local dirty
    dirty=$(git -C "$root" status --porcelain -- . 2>/dev/null | wc -l)
    if [ "$dirty" -gt 0 ]; then
        _sc_warn "$dirty uncommitted change(s) in this workspace, on this machine only."
        _sc_warn "Commit and push them, then git pull on the other machine:  git -C $root status"
    else
        _sc_ok "workspace clean at $(git -C "$root" rev-parse --short HEAD)"
    fi
    _sync_check_build "$root"
    return 0
}

# Is devel/ built from the checked-out source? Only C++, CMake, manifests,
# dynamic_reconfigure and message changes need a rebuild; catkin runs Python
# and launch files straight from src/.
_sync_check_build() {
    local root="$1" stamp built head changed
    [ -d "$root/devel" ] || return 0
    stamp="$root/devel/.built_from"
    if [ ! -f "$stamp" ]; then
        _sc_warn "no devel/.built_from: this build predates the stamp, so nothing says which"
        _sc_warn "source it came from. Run ./build.sh once to record it."
        return 0
    fi
    built=$(awk '{print $1}' "$stamp")
    head=$(git -C "$root" rev-parse HEAD)
    if [ "$built" = "$head" ]; then
        _sc_ok "build is from the checked-out source"
        return 0
    fi
    if ! git -C "$root" cat-file -e "$built^{commit}" 2>/dev/null; then
        _sc_warn "built from ${built:0:7}, which this repo does not know; run ./build.sh"
        return 0
    fi
    # --relative gives src/<pkg>/... paths (field 2 = package) and drops Orin-side changes.
    changed=$(git -C "$root" diff --relative --name-only "$built" HEAD -- src \
              | grep -E '\.(cpp|cc|h|hpp)$|CMakeLists\.txt$|package\.xml$|setup\.py$|\.cfg$|/msg/|/srv/|/action/' \
              | awk -F/ '{print $2}' | sort -u | tr '\n' ' ' || true)
    if [ -n "$changed" ]; then
        _sc_warn "built from ${built:0:7} but the source is at ${head:0:7}, and these need a"
        _sc_warn "rebuild:  $changed-> ./build.sh <package>"
    else
        _sc_ok "built from ${built:0:7}; nothing needing a rebuild changed since"
    fi
}

# Run directly rather than sourced.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    sync_check
fi
