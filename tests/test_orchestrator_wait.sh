#!/usr/bin/env bash
# run_chain waits for stage services that are still starting (ready_timeout), INSIDE the container:
#   docker exec remake_grasp bash /opt/remake_grasp/tests/test_orchestrator_wait.sh
# The real orchestrator, fake stage services; ROS_DOMAIN_ID 77, localhost only. There is no
# camera on 77, so every run ends at capture -- the wait is what is checked, and it comes first.
set -uo pipefail
WT="$(cd "$(dirname "$0")/.." && pwd)"
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
set +u; . /opt/ros/humble/setup.bash; . /opt/ws/install/local_setup.bash; set -u
export PYTHONPATH="$WT/ros/remake_grasp:$PYTHONPATH"
PY=/opt/venv/bin/python3
T=$(mktemp -d)
fail=0

run_case() {   # $1 label, $2 chain, $3 fake service delay (- = none), $4 expected wait min, $5 max, $6 expect "missing"
  local label=$1 chain=$2 delay=$3 lo=$4 hi=$5 want_missing=$6
  "$PY" -W ignore -m remake_grasp.orchestrator --ros-args -p live_dir:="$T/live" -p runs_dir:="$T/runs" \
      -p chain:="$chain" -p "archive:='off'" -p ready_timeout:=12.0 > "$T/orch_$label.log" 2>&1 &
  local orch=$!                                   # ('off' quoted: unquoted, YAML makes it the bool False)
  for i in $(seq 1 30); do ros2 service list 2>/dev/null | grep -qx /live_pipeline/run_chain && break
    kill -0 $orch 2>/dev/null || { echo "  FAIL  $label: the orchestrator died: $(tail -n 1 "$T/orch_$label.log")"; fail=1; return; }
    sleep 1; done
  local fake=""
  if [ "$delay" != - ]; then "$PY" -W ignore "$WT/tests/fake_trigger.py" segment_trans "$delay" > "$T/fake_$label.log" 2>&1 & fake=$!; fi
  local t0 t1 reply
  t0=$(date +%s.%N)
  reply=$(timeout 60 ros2 service call /live_pipeline/run_chain std_srvs/srv/Trigger 2>&1 | grep -oE "message='.{0,300}")
  t1=$(date +%s.%N)
  [ -n "$fake" ] && kill $fake 2>/dev/null
  kill -INT $orch 2>/dev/null; sleep 1; kill $orch 2>/dev/null; wait 2>/dev/null
  local took; took=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
  local waits; waits=$(grep -c "waiting up to" "$T/orch_$label.log")
  local missing; missing=$(grep -c "still not available" "$T/orch_$label.log")
  local ok=1
  awk "BEGIN{exit !($took >= $lo && $took <= $hi)}" || ok=0
  [ "$want_missing" = yes ] && [ "$missing" -eq 0 ] && ok=0
  [ "$want_missing" = no ] && [ "$missing" -ne 0 ] && ok=0
  echo "  $([ $ok = 1 ] && echo 'ok  ' || echo FAIL)  $label: run_chain took ${took:0:5}s (want $lo-$hi), 'waiting' logged $waits, 'still not available' $missing"
  echo "        reply: ${reply:0:160}"
  grep -E "waiting up to|still not available" "$T/orch_$label.log" | sed 's/^/        log: /' | cut -c1-150
  [ $ok = 1 ] || fail=1
}

# a stage service that appears 5 s into the call: run_chain waits ~5 s for it, nothing missing
run_case starting segment_trans 5 4 9 no
# a stage service that never appears: run_chain waits the full 12 s, then reports it and goes on
run_case absent "segment_trans,pointcloud?" - 11.5 16 yes
# already up (fake started 0 s, before the call settles): no wait to speak of
run_case up segment_trans 0 0 3 no
rm -rf "$T"
[ $fail = 0 ] && echo "all passed" || echo "FAILED"
exit $fail
