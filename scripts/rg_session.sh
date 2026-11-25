#!/bin/bash
# One pipeline session at a time. Stops every pipeline process (stop_nodes.sh), then closes every
# remake_grasp* tmux session except KEEP. Run inside the container.
#   rg_session.sh KEEP   the tmuxp files' before_script: tmuxp has just created KEEP, so switching
#                        between adapted.yaml, baseline.yaml, raw.yaml and remake_grasp.yaml leaves one session
#   rg_session.sh        close them all (rg-down)
# Nodes go first: run from a pane of a session it closes, it still stops them.
KEEP="${1:-}"
"$(dirname "$0")/stop_nodes.sh"
for s in $(tmux list-sessions -F '#S' 2>/dev/null | grep '^remake_grasp'); do
    [ "$s" = "$KEEP" ] || tmux kill-session -t "$s"
done
exit 0
