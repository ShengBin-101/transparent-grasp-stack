#!/usr/bin/env bash
# Is the network link to the robot fast enough for FCI's 1 kHz loop? Sourced by
# every script that opens an FCI control connection.
#
#     source tools/fci_link.sh && fci_link_check "$ROBOT_IP" || exit 1
#     tools/fci_link.sh 172.16.0.2                         by hand
#
# One bad cable pair drops the gigabit link to 100 Mb/s with no error anywhere,
# roughly doubling the round trip, and libfranka then aborts the controllers with
# communication_constraints_violation. Checked before a controller starts, that is
# one line naming the cable instead of a fault that looks like software.
#
# FCI_LINK_CHECK=warn turns the refusal into a warning, for diagnosis. The state
# stream alone (--state-only) is fine at 100 Mb/s; callers that only read state
# pass "warn".

# fci_link_check IP [warn|refuse]: 0 at >= 1000 Mb/s full duplex on the route's NIC,
# 0 with a warning when there is no route or in warn mode, else 1.
fci_link_check() {
    local ip="$1" mode="${2:-${FCI_LINK_CHECK:-refuse}}" dev speed duplex
    local red=$'\033[31m' yel=$'\033[33m' off=$'\033[0m'
    dev=$(ip route get "$ip" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)
    if [ -z "$dev" ]; then
        printf '%s!!!%s no route to %s, so the link speed cannot be checked\n' "$yel" "$off" "$ip" >&2
        return 0
    fi
    speed=$(cat "/sys/class/net/$dev/speed" 2>/dev/null || echo unknown)
    duplex=$(cat "/sys/class/net/$dev/duplex" 2>/dev/null || echo unknown)
    if [ "$speed" = 1000 ] || { [ "$speed" != unknown ] && [ "$speed" -gt 1000 ] 2>/dev/null; }; then
        if [ "$duplex" = full ]; then
            printf '    ok    link %s: %s Mb/s %s duplex\n' "$dev" "$speed" "$duplex"
            return 0
        fi
    fi
    local what="link $dev to the robot is at $speed Mb/s, $duplex duplex (need 1000, full)"
    if [ "$mode" = warn ]; then
        printf '%s!!!%s %s. Control connections will likely abort with\n' "$yel" "$off" "$what" >&2
        printf '        communication_constraints_violation; carrying on (FCI_LINK_CHECK=warn)\n' >&2
        return 0
    fi
    printf '%sERROR:%s %s.\n' "$red" "$off" "$what" >&2
    printf '       FCI'"'"'s 1 kHz loop does not survive it: both controllers were aborted by\n' >&2
    printf '       communication_constraints_violation on such a link (2026-09-26). A gigabit\n' >&2
    printf '       NIC falling back to 100 almost always means a broken pair: REPLACE the\n' >&2
    printf '       cable (reseating may hold only until the next boot), or try another\n' >&2
    printf '       switch port, then check  cat /sys/class/net/%s/speed  reads 1000.\n' "$dev" >&2
    printf '       FCI_LINK_CHECK=warn overrides this, for diagnosis only.\n' >&2
    return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    fci_link_check "${1:-172.16.0.2}"
fi
