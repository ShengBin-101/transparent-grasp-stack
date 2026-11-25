#!/usr/bin/env bash
# What can this setup run? Checks the weights, the GPU, SAM 3, the datasets and the disk, then says which of
# docs/REPRODUCE.md's steps will work. Run inside the container (the GPU check needs its torch):
#   scripts/check_setup.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=${PY:-python3}
pass() { printf '  ok    %s\n' "$*"; }
miss() { printf '  --    %s\n' "$*"; }
weights=0 gpu=0 sam3=0 transcg=0

echo "weights (checkpoints/SHA256SUMS)"
if [ -f checkpoints/SHA256SUMS ]; then
    n=0; while read -r sum f; do
        if [ -f "checkpoints/$f" ] && [ "$(sha256sum "checkpoints/$f" | cut -d' ' -f1)" = "$sum" ]; then n=$((n + 1)); pass "$f"
        else miss "$f  (scripts/get_checkpoints.sh)"; fi
    done < checkpoints/SHA256SUMS
    [ "$n" -eq "$(wc -l < checkpoints/SHA256SUMS)" ] && weights=1
else
    miss "checkpoints/SHA256SUMS not found (it is in git: is checkpoints/ mounted over?)"
fi

echo "GPU"
if [ "$(uname -m)" = aarch64 ]; then
    [ -d /usr/local/cuda-12.6 ] && pass "CUDA 12.6 from the host" || miss "/usr/local/cuda-12.6 missing: this image needs Jetson Linux R36.4 or later (CUDA 12.6)"
    ls /usr/lib/aarch64-linux-gnu/libcudnn.so.9* >/dev/null 2>&1 && pass "cuDNN 9 from the host" || miss "libcudnn.so.9 missing: this image needs the host's cuDNN 9"
fi
if out=$("$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null); then
    set -- $out
    [ "$2" = True ] && { gpu=1; pass "torch $1 sees ${*:3}"; } || miss "torch $1 sees no GPU (start the container with ./start.sh; on a Jetson it needs --runtime nvidia and the host's /dev)"
else miss "torch does not import here (run this inside the container)"; fi

echo "SAM 3 (PIPELINE=baseline live; not needed for the demo or the benchmark)"
if ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--facebook--sam3/snapshots/*/sam3.pt >/dev/null 2>&1; then sam3=1; pass "facebook/sam3 is in the Hugging Face cache"
else miss "not downloaded: request access at huggingface.co/facebook/sam3, then hf auth login (checkpoints/README.md)"; fi

echo "datasets (the repo's datasets/, else RG_DATASETS mounted at /datasets by start.sh)"
DS=""; for d in "$ROOT/datasets" /datasets; do [ -f "$d/transcg/transcg/metadata.json" ] && { DS=$d; break; }; done
TC="${RG_TRANSCG:-${DS:-$ROOT/datasets}/transcg/transcg}"
[ -f "$TC/metadata.json" ] && { transcg=1; pass "TransCG at $TC"; } \
    || miss "TransCG not found: python3 eval/make_transcg_subset.py <full TransCG> puts what the benchmark reads in datasets/"
DS=${DS:-$ROOT/datasets}
[ -d "$DS/cleargrasp/cleargrasp-dataset-test-val" ] && pass "ClearGrasp test/val" || miss "ClearGrasp not present (optional)"
[ -d "$DS/sam_masks_box" ] && pass "SAM-box masks (the sambox benchmark row)" || miss "SAM-box masks not present (optional: scripts/benchmark.sh <remake> sambox fetches them)"

echo "disk"
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 10 ] && pass "${free_gb} GB free" || miss "only ${free_gb} GB free (the benchmark writes ~1 GB a row; a rebuild needs ~20 GB)"

echo
echo "what will run here"
[ $weights = 1 ] && [ $gpu = 1 ] && echo "  yes   offline demo, every preset: scripts/demo_offline.sh all" || echo "  no    offline demo (needs the weights and the GPU)"
[ $weights = 1 ] && [ $gpu = 1 ] && [ $transcg = 1 ] && echo "  yes   TransCG benchmark: scripts/benchmark.sh <remake> <conditions>" || echo "  no    TransCG benchmark (needs the weights, the GPU and TransCG)"
[ $weights = 1 ] && [ $gpu = 1 ] && echo "  yes*  live PIPELINE=adapted and raw (* with the L515 connected: rg-adapted, rg-raw)" || echo "  no    live adapted pipeline"
[ $weights = 1 ] && [ $gpu = 1 ] && [ $sam3 = 1 ] && echo "  yes*  live PIPELINE=baseline (* with the L515: rg-baseline)" || echo "  no    live baseline (needs SAM 3)"
echo "  robot picks need the NUC planner/controller and the Franka (nuc/README.md)"
