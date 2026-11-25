#!/bin/bash
# Start the remake_grasp container (if needed) and open a shell in it.
#   ./start.sh              start and enter (run again from other terminals to get more shells)
#   ./start.sh --stop       stop the container; every node inside stops with it
#   ./start.sh --recreate   remove the container and start a fresh one (after ./build.sh)
# Optional environment, fixed when the container is created: DETECTOR=sbg|peggnet
#   PIPELINE=adapted|baseline|raw  MASK=sam3|seg|none  REMAKE=official|L_combo_both|G_maskdrop|A_ft|none
#   TARGETS=depth|prompt|transparent|none  ROS_DOMAIN_ID  NUC_IP  NUC_HOSTNAME  ROS_IP
#   RG_SSH=1 mounts ~/.ssh read-only and copies it into the container (to ssh to the NUC from inside).
# SAM 3's weights are gated: log in once with `hf auth login` (on the host or in the container; the
# Hugging Face cache is mounted), rather than passing a token on the docker command line.
# PIPELINE / MASK / REMAKE / TARGETS can also be set per rg-up (tmuxp/remake_grasp.yaml).
# RG_DATASETS=<dir> mounts a dataset kept outside the repo read-only at /datasets
# (<dir>/transcg/transcg/...); scripts/benchmark.sh looks in the repo's datasets/ first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${NAME:-remake_grasp}"
IMAGE="${IMAGE:-remake_grasp:latest}"

running() { [ "$(docker container inspect -f "{{.State.Running}}" "$NAME" 2>/dev/null)" = true ]; }
exists()  { docker container inspect "$NAME" >/dev/null 2>&1; }
enter() {
    if [ -t 0 ]; then exec docker exec -it -e TERM="${TERM:-xterm}" "$NAME" bash; fi
    echo "$NAME is running (no terminal here, so not entering it)"; exit 0
}

case "${1:-}" in
    --stop)     docker stop -t 15 "$NAME" >/dev/null && echo "stopped $NAME"; exit 0 ;;
    --recreate) docker rm -f "$NAME" >/dev/null 2>&1 || true ;;
    "") ;;
    *) sed -n '2,14p' "$0"; exit 1 ;;
esac

if running; then enter; fi
if exists; then
    if [ "$(docker container inspect -f "{{.Image}}" "$NAME")" != "$(docker image inspect -f '{{.Id}}' "$IMAGE")" ]; then
        echo "note: $IMAGE was rebuilt since this container was made; ./start.sh --recreate to use it"
    fi
    docker start "$NAME" >/dev/null
    enter
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE -- run ./build.sh first" >&2; exit 1; }
mkdir -p "$ROOT/data/live" "$ROOT/runs" "$ROOT/checkpoints" "${HF_HOME:-$HOME/.cache/huggingface}"

gpu=()
if [ "$(uname -m)" = aarch64 ]; then
    # Jetson: the image carries no CUDA; use the host's CUDA and cuDNN (read-only).
    if [ ! -d /usr/local/cuda-12.6 ] || ! ls /usr/lib/aarch64-linux-gnu/libcudnn.so.9* >/dev/null 2>&1; then
        echo "this image's torch needs CUDA 12.6 and cuDNN 9 from the host (Jetson Linux R36.4 or later);" \
             "/usr/local/cuda-12.6 or libcudnn.so.9 is missing (README.md, quick start)" >&2; exit 1
    fi
    gpu+=(--runtime nvidia)
    for d in /usr/local/cuda-12.6 /usr/lib/aarch64-linux-gnu/tegra; do
        [ -d "$d" ] && gpu+=(-v "$d:$d:ro")
    done
    for f in /usr/lib/aarch64-linux-gnu/libcudnn*.so.9*; do
        [ -e "$f" ] && gpu+=(-v "$f:$f:ro")
    done
else
    gpu+=(--gpus all)
fi

extra=()
[ "${RG_SSH:-0}" = 1 ] && [ -d "$HOME/.ssh" ] && extra+=(-v "$HOME/.ssh:/host-ssh:ro")
[ -f "$HOME/.tmux.conf" ] && extra+=(-v "$HOME/.tmux.conf:/root/.tmux.conf:ro")
[ -n "${NUC_HOSTNAME:-}" ] && extra+=(--add-host "$NUC_HOSTNAME:${NUC_IP:-172.16.0.1}")
if [ -n "${RG_DATASETS:-}" ]; then
    [ -d "$RG_DATASETS" ] || { echo "RG_DATASETS=$RG_DATASETS is not a directory" >&2; exit 1; }
    extra+=(-v "$(cd "$RG_DATASETS" && pwd):/datasets:ro")
fi

docker run -d --name "$NAME" --init --privileged --network host --ipc host \
    "${gpu[@]}" "${extra[@]}" \
    -v /dev:/dev -v /etc/localtime:/etc/localtime:ro \
    -v "$ROOT:/opt/remake_grasp" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -e DETECTOR="${DETECTOR:-sbg}" \
    -e PIPELINE="${PIPELINE:-adapted}" -e MASK="${MASK:-}" -e REMAKE="${REMAKE:-}" -e TARGETS="${TARGETS:-}" \
    -e NUC_IP="${NUC_IP:-172.16.0.1}" ${ROS_IP:+-e ROS_IP="$ROS_IP"} \
    "$IMAGE" >/dev/null
echo "started $NAME ($IMAGE)"
enter
