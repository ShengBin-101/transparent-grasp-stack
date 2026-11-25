#!/usr/bin/env bash
# Fill checkpoints/ and check every file against checkpoints/SHA256SUMS.
#   scripts/get_checkpoints.sh                    # check; download what is missing
#   scripts/get_checkpoints.sh --from <dir>       # copy files of the same names from a folder first
#   RG_WEIGHTS_URL=<base url> scripts/get_checkpoints.sh   # the project's weights from <base url>/<file>
# Where each file comes from (checkpoints/README.md):
#   authors  remake_official.tar (ReMake, Google Drive), depth_anything_v2_vits.pth (Hugging Face),
#            scale_balanced_full_model.tar (SBG, Google Drive folder), peggnet_jacquard_d_480.pt (GitHub)
#   project  L_combo_both_checkpoint43.tar, A_ft_checkpoint45.tar, G_maskdrop_checkpoint45.tar,
#            seg_vits_rgb_best.pt: this repository's GitHub Release v1.0
# Google Drive needs gdown (in the image; on a host: pip install gdown). SAM 3 is not here: it downloads into
# ~/.cache/huggingface on first use, after Meta approves access to facebook/sam3 (checkpoints/README.md).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$ROOT/checkpoints"
FROM=""
[ "${1:-}" = "--from" ] && { FROM="${2:?--from needs a folder}"; }
[ -z "${1:-}" ] || [ "$1" = "--from" ] || { sed -n '3,5p' "$0"; exit 1; }
mkdir -p "$DST"
want() { awk -v f="$1" '$2 == f {print $1}' "$DST/SHA256SUMS"; }
ok() { [ -f "$DST/$1" ] && [ "$(sha256sum "$DST/$1" | cut -d' ' -f1)" = "$(want "$1")" ]; }
gd() { command -v gdown >/dev/null 2>&1 || python3 -c "import gdown" 2>/dev/null; }
URL="${RG_WEIGHTS_URL:-https://github.com/ShengBin-101/transparent-grasp-stack/releases/download/v1.0}"

fetch() {   # $1 file -> --from, else its source
    local f=$1 tmp
    [ -n "$FROM" ] && [ -f "$FROM/$f" ] && cp -L "$FROM/$f" "$DST/$f" && ok "$f" && return 0
    case "$f" in
        remake_official.tar)
            gd && python3 -m gdown --quiet 1AF9sWyhEoNHnAlJ_y1AXsoxgrXq44333 -O "$DST/$f" ;;
        depth_anything_v2_vits.pth)
            curl -fsSL "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/$f" -o "$DST/$f" ;;
        scale_balanced_full_model.tar)   # the authors' folder; the full model is log_full_model/checkpoint.tar
            gd || return 1
            tmp=$(mktemp -d)
            python3 -m gdown --quiet --folder 1Y2o0uAbhS6-yZhPkKMAAnL0tCpbiJfk0 -O "$tmp" \
                && cp "$(find "$tmp" -path '*log_full_model/checkpoint.tar' | head -1)" "$DST/$f"
            rm -rf "$tmp" ;;
        peggnet_jacquard_d_480.pt)
            curl -fsSL "https://github.com/HZWang96/PEGG-Net/raw/be3ddfd/output/pre-trained_models/220205_1108_peggnet_jacquard_d_480/epoch_44_iou_0.94_statedict.pt" -o "$DST/$f" ;;
        *)  curl -fsSL "${URL%/}/$f" -o "$DST/$f" ;;
    esac
}

bad=0
for f in $(awk '{print $2}' "$DST/SHA256SUMS"); do
    if ok "$f"; then echo "ok       $f"; continue; fi
    [ -f "$DST/$f" ] && echo "checksum mismatch: $f -- fetching again" && rm -f "$DST/$f"
    if fetch "$f" && ok "$f"; then echo "fetched  $f"; continue; fi
    rm -f "$DST/$f"; bad=$((bad + 1))
    case "$f" in
        remake_official*|scale_balanced*) echo "MISSING  $f  (Google Drive: needs gdown, and network)" ;;
        *) echo "MISSING  $f  (download failed: network?)" ;;
    esac
done
chmod a+r "$DST"/* 2>/dev/null || true
[ "$bad" -eq 0 ] && echo "all $(wc -l < "$DST/SHA256SUMS") weights present and verified" || { echo "$bad missing"; exit 1; }
