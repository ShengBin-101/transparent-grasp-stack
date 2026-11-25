#!/bin/bash
# Build the remake_grasp image for this machine: Jetson AGX Orin, Jetson Linux R36.5.2 (x86_64 + NVIDIA GPU is written but untested).
#   ./build.sh                 first build takes ~1 h on the Orin (RealSense SDK + ros1_bridge)
#   JOBS=8 ./build.sh          more compile jobs (needs ~3 GB RAM per job for ros1_bridge)
#   IMAGE=name:tag ./build.sh
#   NO_CACHE=1 ./build.sh      rebuild every layer, ignoring Docker's cache
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
IMAGE="${IMAGE:-remake_grasp:latest}"

case "$(uname -m)" in
    aarch64) echo "==> building for Jetson (aarch64)" ;;
    x86_64)  echo "==> building for x86_64" ;;
    *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

free_gb=$(df -BG --output=avail "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /)" | tail -1 | tr -dc 0-9)
if [ "${free_gb:-0}" -lt 25 ]; then
    echo "WARNING: only ${free_gb} GB free for Docker; the build needs ~20 GB. Try: docker system prune" >&2
fi
# third_party/remake, sbg and sam3 are git submodules: fetch them if this clone has not.
for d in remake sbg sam3; do
    if [ ! -f "third_party/$d/LICENSE" ]; then
        echo "==> fetching the third_party submodules"
        git submodule update --init
        break
    fi
done
[ -f checkpoints/remake_official.tar ] || echo "note: checkpoints/ is empty -- run scripts/get_checkpoints.sh before starting"

args=()
[ -n "${JOBS:-}" ] && args+=(--build-arg "JOBS=$JOBS")
[ -n "${NO_CACHE:-}" ] && args+=(--no-cache)
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile "${args[@]}" -t "$IMAGE" .
echo "==> built $IMAGE. Start it with ./start.sh"
