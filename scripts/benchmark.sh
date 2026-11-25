#!/usr/bin/env bash
# TransCG benchmark rows inside the container -- no robot, no camera (the mask zips are fetched on first use):
#   scripts/benchmark.sh L_combo_both dataset blanked nomask seg_d5
#   scripts/benchmark.sh official dataset
#   LIMIT=20 scripts/benchmark.sh official dataset          # a quick check on the first 20 frames
# The first argument is a ReMake name from remake.launch.py (official, L_combo_both, G_maskdrop, A_ft); the
# rest are conditions:
#   dataset  the benchmark protocol (the dataset's own transparent mask)
#   blanked  the dataset mask, with the sensor's depth inside it removed
#   nomask   an empty mask
#   seg_d0, seg_d5   the prompt-free segmenter's mask, undilated / dilated 5 px (made on first use)
#   sambox   SAM (v1) ViT-B masks prompted with a box around each region of the dataset's mask: datasets/sam_masks_box/, fetched
#            from the release (sam_masks_box.zip) on first use
#   sam3text SAM 3's masks from the text prompt "transparent object or bottle": datasets/sam3_masks_text/,
#            from the release (sam3_masks_text.zip) on first use (tools/sam3_text_masks.py --out datasets/sam3_masks_text
#            remakes them, not bit for bit)
# Each row: eval/predict_transcg.py, eval/score.py, eval/measure_concavity.py; the reports go to
# data/eval/results/<name>_<condition>/, and one line compares the row with the recorded one
# (eval/reference.json).
# Needs TransCG: the repo's datasets/transcg/transcg (eval/make_transcg_subset.py makes the 2.6 GB it reads),
# or a folder given as RG_DATASETS=<dir> ./start.sh, mounted at /datasets. ~20 min a row on the Orin.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ $# -ge 2 ] || { sed -n '2,20p' "$0"; exit 1; }
NAME=$1; shift
PY=${PY:-python3}
OUT="${RG_EVAL_OUT:-$ROOT/data/eval}"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
URL="${RG_WEIGHTS_URL:-https://github.com/ShengBin-101/transparent-grasp-stack/releases/download/v1.0}"
DATASETS=""              # the repo's datasets/ first, then a folder start.sh mounted at /datasets
for d in "$ROOT/datasets" /datasets; do [ -f "$d/transcg/transcg/metadata.json" ] && { DATASETS=$d; break; }; done
export RG_TRANSCG="${RG_TRANSCG:-${DATASETS:-$ROOT/datasets}/transcg/transcg}"
[ -f "$RG_TRANSCG/metadata.json" ] || {
    echo "TransCG not found at $RG_TRANSCG: python3 eval/make_transcg_subset.py <full TransCG> puts the 2.6 GB" \
         "the benchmark reads into datasets/ (or RG_DATASETS=<dir holding transcg/> ./start.sh --recreate)" >&2; exit 1; }

for cond in "$@"; do
    row="${NAME}_${cond}${LIMIT:+_first$LIMIT}"
    res="$OUT/results/$row"; mkdir -p "$res"
    case "$cond" in
        dataset|blanked|nomask) args=(--condition "$cond") ;;
        seg_d*) k=${cond#seg_d}; masks="$OUT/masks/seg_vits_rgb_d$k"
                if [ "$(ls "$masks" 2>/dev/null | wc -l)" -lt 1177 ] || [ -n "${LIMIT:-}" ]; then
                    $PY eval/make_seg_masks.py --dilate "$k" "${LIM[@]}" | tee "$res/make_masks.log"; fi
                args=(--condition dir --mask-dir "$masks") ;;
        sambox|sam3text)
                dir=$([ "$cond" = sambox ] && echo sam_masks_box || echo sam3_masks_text)
                masks="${DATASETS:-$ROOT/datasets}/$dir"
                if [ ! -d "$masks" ]; then
                    echo "   fetching $dir.zip from the release"
                    mkdir -p "$ROOT/datasets" && masks="$ROOT/datasets/$dir"
                    curl -fsSL "${URL%/}/$dir.zip" -o "$ROOT/datasets/$dir.zip" \
                        && $PY -m zipfile -e "$ROOT/datasets/$dir.zip" "$ROOT/datasets/" && rm -f "$ROOT/datasets/$dir.zip" \
                        || { echo "$cond: could not fetch $dir.zip (network?)" >&2; continue; }
                fi
                args=(--condition dir --mask-dir "$masks") ;;
        *) echo "unknown condition $cond" >&2; exit 1 ;;
    esac
    echo "== $row"
    if [ -z "${LIMIT:-}" ] && [ -z "${FRESH:-}" ] && grep -qs '"frames": 1177' "$OUT/predictions/$row/run.json"; then
        echo "   predictions complete from an earlier run; scoring them (FRESH=1 to predict again)"
    else
        $PY eval/predict_transcg.py --remake "$NAME" "${args[@]}" --name "$row" "${LIM[@]}" 2>&1 \
            | grep -v "xFormers\|UserWarning\|meshgrid\|return _VF" | tee "$res/predict.log"
    fi
    $PY eval/score.py --model "$row" "${LIM[@]}" > "$res/masked_rmse.txt"
    # concavity samples its frames at random under --limit, so a quick run would score almost none
    if [ -z "${LIMIT:-}" ]; then $PY eval/measure_concavity.py --model "$row" > "$res/concavity.txt"; else : > "$res/concavity.txt"; fi
    $PY - "$res" "$NAME|$cond" "${LIMIT:-}" <<'EOF'
import json, re, sys
from pathlib import Path
res, key, limit = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = (res / "masked_rmse.txt").read_text()
m = re.search(r"masked RMSE\s+mean\s+([\d.]+) mm", t)
l5 = re.search(r"L515: mean\s+([\d.]+) mm", t)
c = re.search(r"overall \(GT > 10 mm, n=\d+\): (\d+)% of concavity", (res / "concavity.txt").read_text())
mine = f"{float(m.group(1)):.1f} mm" + (f" (L515 {l5.group(1)})" if l5 else "") + (f", concavity {c.group(1)} %" if c else "")
ref = json.load(open("eval/reference.json"))["rows"].get(key)      # cwd is the repo root
if limit:
    print(f"   {key}: {mine}  [first {limit} frames only; not comparable with the recorded full rows]")
elif ref is None:
    print(f"   {key}: {mine}  [no recorded row to compare]")
else:
    d = float(m.group(1)) - ref["masked_rmse_mm"]
    dc = (int(c.group(1)) - ref["concavity_pct"]) if (c and ref.get("concavity_pct") is not None) else None
    ok = abs(d) <= 2.0 and (dc is None or abs(dc) <= 5)
    print(f"   {key}: {mine}; recorded {ref['masked_rmse_mm']:.1f} mm" +
          (f", {ref['concavity_pct']} %" if ref.get("concavity_pct") is not None else "") +
          f"  ->  {d:+.1f} mm" + (f", {dc:+d} points" if dc is not None else "") +
          ("  (within the ~2 mm / 5-point threshold)" if ok else "  ** OUTSIDE the ~2 mm / 5-point threshold **"))
EOF
done
