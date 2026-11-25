#!/usr/bin/env bash
# The pipeline without the robot or the camera: a sample capture through the real nodes, inside the container.
#   scripts/demo_offline.sh three_transparent adapted
#   scripts/demo_offline.sh sam3_missed_mug baseline
#   scripts/demo_offline.sh all                  # every sample x every preset it supports, then compare
# Presets: the trials' three conditions, read from pipeline.launch.py's PIPELINES:
#   adapted   the segmenter's mask + L_combo_both + prompt-free targets
#   baseline  SAM 3's mask + the released ReMake + the same targets (needs the sample's archived SAM 3 mask)
#   raw       the raw sensor depth into SBG: no mask, no ReMake, no target filter
# Each stage is the live node, started, called once through its /live_pipeline service, and stopped, in the
# order the orchestrator calls them -- on ROS_DOMAIN_ID 77, localhost only, so a live pipeline on domain 0
# never sees it. SAM 3 is not re-run: the sample's archived sam3_masks.npy stands in (samples/README.md).
# Output: data/demo/<sample>_<preset>/ (the run's files, node logs, summary.json). With samples/<s>/expected/
# <preset>.json present, the targets and grasp counts are compared with it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RG_ROOT="$ROOT" ROS_DOMAIN_ID="${DEMO_DOMAIN:-77}" ROS_LOCALHOST_ONLY=1
set +u; . /opt/ros/humble/setup.bash; [ -f /opt/ws/install/local_setup.bash ] && . /opt/ws/install/local_setup.bash; set -u
export PYTHONPATH="$ROOT/ros/remake_grasp:${PYTHONPATH:-}"
PY=${PY:-python3}

serve_call() {   # $1 module, $2 service, rest: node params -> start the node, call its service once, stop it
    local mod=$1 srv=$2; shift 2
    local args=(--ros-args -p live_dir:="$LIVE"); for kv in "$@"; do args+=(-p "$kv"); done
    "$PY" -W ignore -m "remake_grasp.$mod" "${args[@]}" > "$LIVE/node_$mod.log" 2>&1 &
    local pid=$! i r
    for i in $(seq 1 180); do ros2 service list 2>/dev/null | grep -qx "/live_pipeline/$srv" && break
        kill -0 $pid 2>/dev/null || break; sleep 1; done
    r=$(timeout 300 ros2 service call "/live_pipeline/$srv" std_srvs/srv/Trigger 2>&1 | grep -oE "success=[A-Za-z]+, message='[^']{0,200}")
    kill -INT $pid 2>/dev/null; sleep 2; kill $pid 2>/dev/null; wait $pid 2>/dev/null
    echo "  $srv: ${r:-NO REPLY (see $LIVE/node_$mod.log)}"
}

run() {   # $1 sample, $2 preset
    local sample=$1 preset=$2 mask remake targets ckpt fix mfile
    read -r mask remake targets < <("$PY" -c '
import ast, sys
tree = ast.parse(open("ros/remake_grasp/launch/pipeline.launch.py").read())
table = next(ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PIPELINES" for t in n.targets))
p = table.get(sys.argv[1])
print(*(p[k] for k in ("mask", "remake", "targets")) if p else "")
' "$preset")
    [ -n "$mask" ] || { echo "unknown preset $preset (adapted | baseline | raw)" >&2; return 1; }
    [ -f "samples/$sample/sample.json" ] || { echo "no sample samples/$sample" >&2; return 1; }
    LIVE="$ROOT/data/demo/${sample}_$preset"; rm -rf "$LIVE"
    "$PY" scripts/make_sample.py "samples/$sample" --unpack "$LIVE" || return 1
    if [ "$mask" = sam3 ] && [ ! -f "$LIVE/sam3_masks.npy" ]; then
        echo "== $sample / $preset: skipped -- this sample has no archived SAM 3 mask"; return 0; fi
    rm -f "$LIVE/sam3_masks_all.npy"                     # only TARGETS=prompt reads the grasp prompt's masks
    # mfile "" (no mask) goes to the node as the YAML empty string '': ROS 2 rejects an empty "-p name:="
    case "$mask" in seg) mfile=seg_masks.npy ;; sam3) mfile=sam3_masks.npy ;; *) mfile="" ;; esac
    echo "== $sample / $preset: mask $mask${mfile:+ ($mfile)}, remake $remake, targets $targets"
    if [ "$mask" = seg ] || [ "$targets" = depth ]; then serve_call seg_node segment_trans mask_name:=seg_masks.npy; fi
    local sdepth=remake_depth_mm.npy completed=true
    if [ "$remake" = none ]; then
        echo "  depth_remake: not run (raw depth)"; sdepth=depth_aligned_mm.npy; completed=false
    else
        read -r ckpt fix < <("$PY" - "$remake" <<'EOF'
import ast, sys
tree = ast.parse(open("ros/remake_grasp/launch/remake.launch.py").read())
table = next(ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "REMAKE" for t in n.targets))
f, fix = table[sys.argv[1]]
print(f, str(fix).lower())
EOF
)
        serve_call remake_node depth_remake mask_name:="$mfile" checkpoint:="$ROOT/checkpoints/$ckpt" rel_rgb_fix:="$fix"
    fi
    serve_call sbg_node grasps_sbg depth_name:="$sdepth" top_k:=50 num_points:=20000 max_aperture_mm:=70.0 \
        approach_cone_deg:=30.0 empty_check:=true
    serve_call grasp_pose_node grasp_poses source:=sbg base_frame:=camera_color_optical_frame targets:="$targets" \
        target_mask:=seg_masks.npy transparent_mask:="${mfile:-''}" depth_completed:="$completed"
    "$PY" - "$LIVE" "samples/$sample/expected/$preset.json" <<'EOF'
import json, sys
from pathlib import Path
live, expected = Path(sys.argv[1]), Path(sys.argv[2])
try:
    gp = json.loads((live / "grasp_poses.json").read_text())
except FileNotFoundError:
    print("  -> no grasp_poses.json: a stage failed (see the node logs above)"); sys.exit(0)
ex = [g for g in gp["grasps"] if g["executable"]]
objs = sorted(("transparent" if o["transparent"] else "opaque", o.get("source")) for o in gp.get("objects", []))
s = {"grasps": len(gp["grasps"]), "executable": len(ex),
     "executable_transparent": sum(bool(g["transparent"]) for g in ex),
     "executable_opaque": sum(g["transparent"] is False for g in ex), "targets": objs}
(live / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
print(f"  -> {s['grasps']} grasps, {s['executable']} executable ({s['executable_transparent']} on transparent, "
      f"{s['executable_opaque']} on opaque objects); targets {objs}")
if expected.exists():
    e = json.loads(expected.read_text())
    same_targets = [tuple(t) for t in e["targets"]] == objs
    print(f"  -> vs {expected}: targets {'the same' if same_targets else 'DIFFER: ' + str(e['targets'])}; "
          f"executable grasps {s['executable']} (expected {e['executable']})")
EOF
}

if [ "${1:-}" = all ]; then
    for d in samples/*/; do
        s=$(basename "$d")
        for p in $("$PY" -c "import json; print(' '.join(json.load(open('samples/$s/sample.json'))['presets']))"); do run "$s" "$p"; done
    done
elif [ $# -eq 2 ]; then
    run "$1" "$2"
else
    sed -n '2,14p' "$0"; exit 1
fi
