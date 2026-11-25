#!/bin/bash
# Pre-flight for a trial session or block, run on the Orin HOST (not in the container):
#   scripts/preflight.sh            everything below; exit 1 if any check FAILs
#   scripts/preflight.sh --no-nuc   skip the NUC checks (arm off)
# Checks, each printed ok / WARN / FAIL:
#   code       the commit, and nothing uncommitted under ros scripts tmuxp docker third_party segmenter
#   camera     the L515 is on USB 3 (5000 Mb/s; on USB 2 it has no default profile)
#   NUC        reachable; FCI link at 1000 Mb/s (at 100 every controller aborts); disk >= 2 GB free (the trials
#              write tens of MB there); the NUC workspace's commit. NUC (ssh host, default nuc), NUC_IFACE (the
#              FCI interface, default eno1) and NUC_WS (default ~/transparent-grasp-stack/nuc) say where to look
#   pipeline   (inside the container, scripts/preflight_ros.py) colour >= 13 Hz and depth >= 25 Hz (15 and 30 configured), the colour
#              image not dark, depth coverage, the tool pose streaming from the NUC, plan_pick and
#              move_to_pose up, which pipeline label the orchestrator will record, and SBG's profile
# The pipeline checks need a running session (rg-adapted / rg-baseline / rg-raw); without one
# they say so and are skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${NAME:-remake_grasp}"
NUC="${NUC:-nuc}"
NUC_IFACE="${NUC_IFACE:-eno1}"
NUC_WS="${NUC_WS:-~/transparent-grasp-stack/nuc}"
fails=0
ok()   { printf '  ok    %-9s %s\n' "$1" "$2"; }
warn() { printf '  WARN  %-9s %s\n' "$1" "$2"; }
fail() { printf '  FAIL  %-9s %s\n' "$1" "$2"; fails=$((fails + 1)); }

echo "pre-flight $(date '+%Y-%m-%d %H:%M:%S')"
commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
dirty="$(git -C "$ROOT" status --porcelain -- ros scripts tmuxp docker third_party segmenter 2>/dev/null)"
if [ -z "$dirty" ]; then ok code "commit $commit, clean"
else fail code "commit $commit with uncommitted changes: $(echo "$dirty" | awk '{print $2}' | head -5 | tr '\n' ' ')"; fi

cam=""
for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] && [ "$(cat "$d/idVendor")" = 8086 ] && [ -f "$d/speed" ] && cam="$d"
done
if [ -z "$cam" ]; then fail camera "no Intel RealSense on USB: plug the L515 into a USB 3 port"
elif [ "$(cat "$cam/speed")" -ge 5000 ]; then ok camera "$(cat "$cam/product" 2>/dev/null || echo RealSense) on USB 3 ($(cat "$cam/speed") Mb/s)"
else fail camera "$(cat "$cam/product" 2>/dev/null || echo RealSense) on USB 2 ($(cat "$cam/speed") Mb/s): replug it into a USB 3 port, cable fully in"; fi

if [ "${1:-}" = --no-nuc ]; then
    warn NUC "skipped (--no-nuc)"
elif out="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$NUC" \
        "cat /sys/class/net/$NUC_IFACE/speed 2>/dev/null || echo ?; df -BM --output=avail /home | tail -1; cd $NUC_WS 2>/dev/null && git log -1 --format='%h %s' | cut -c1-70; git status --porcelain -- src tools *.sh 2>/dev/null | wc -l" 2>/dev/null)"; then
    speed="$(sed -n 1p <<<"$out")"; avail="$(sed -n 2p <<<"$out" | tr -dc 0-9)"
    head="$(sed -n 3p <<<"$out")"; ndirty="$(sed -n 4p <<<"$out" | tr -dc 0-9)"
    if [ "$speed" = 1000 ]; then ok NUC "FCI link $NUC_IFACE at 1000 Mb/s"
    else fail NUC "FCI link $NUC_IFACE at ${speed} Mb/s (needs 1000; at 100 every controller aborts): reseat or swap the cable"; fi
    # MB, not GB: df -BG rounds up, so 1.1 GB would read as 2
    gb="$(awk -v m="${avail:-0}" 'BEGIN { printf "%.1f", m / 1024 }')"
    if [ "${avail:-0}" -ge 2048 ]; then ok NUC "${gb} GB free on /home"
    else fail NUC "${gb} GB free on /home (< 2 GB): clear space in your account (Trash, old caches)"; fi
    if [ "${ndirty:-0}" = 0 ]; then ok NUC "workspace $head, clean"
    else warn NUC "workspace $head with $ndirty uncommitted file(s): write both on the log's cover"; fi
else
    fail NUC "no ssh to $NUC within 5 s: is it on, and the switch powered?"
fi

if [ "$(docker container inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
    fail pipeline "the $NAME container is not running (./start.sh)"
else
    docker exec "$NAME" bash -lc 'python3 /opt/remake_grasp/scripts/preflight_ros.py'
    rc=$?
    [ $rc -eq 1 ] && fails=$((fails + 1))
fi

echo
if [ $fails = 0 ]; then echo "pre-flight passed"; else echo "pre-flight: $fails check(s) FAILED"; fi
exit $((fails > 0))
