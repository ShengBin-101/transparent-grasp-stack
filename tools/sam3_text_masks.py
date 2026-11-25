#!/usr/bin/env python3
"""SAM 3's text-prompted masks for the frozen TransCG test subset, cached as PNGs.

    # in the container, with sam3_node serving on a scratch live folder and the rig's prompt:
    ros2 run remake_grasp sam3_node --ros-args -p live_dir:=/tmp/sam3_gen -p prompt:="transparent object or bottle" &
    python3 tools/sam3_text_masks.py --live-dir /tmp/sam3_gen --out data/eval/masks/sam3_text

For each of the 1,177 frames in eval/sample_list.json this copies the frame's rgb{camera}.png to the live folder,
calls /live_pipeline/segment_sam3 (the service the BASELINE condition uses, so the prompt under test is the
pipeline's own), and writes the mask as <key>.png (0/255, the format segmenter/mask_quality.py and
eval/predict_transcg.py --mask-dir read). A frame where SAM 3 finds nothing is written as an empty mask, not
dropped: an empty mask is the right record of that failure. index.json lists the instance count and mask share
per frame.

The masks are cached because the comparison they feed (eval/predict_transcg.py's dir condition: ReMake's error
with the masks a robot gets instead of the dataset's) needs every model to see the same masks, and SAM 3 is not
bit-deterministic across runs. These made the dissertation's SAM 3 rows: IoU 0.377, recall 0.381,
precision 0.914, 72 empty frames, ReMake 28.9 mm. Score them with
segmenter/mask_quality.py --mask-dir <out>.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "masks" / "sam3_text"))
    ap.add_argument("--live-dir", default="/tmp/sam3_gen", help="the live_dir sam3_node serves")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip keys already written")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger

    out, live = Path(args.out), Path(args.live_dir)
    out.mkdir(parents=True, exist_ok=True)
    live.mkdir(parents=True, exist_ok=True)
    samples = C.load_subset()["samples"]
    if args.limit:
        samples = samples[:args.limit]

    rclpy.init()
    node = Node("sam3_text_masks")
    cli = node.create_client(Trigger, "/live_pipeline/segment_sam3")
    if not cli.wait_for_service(timeout_sec=15.0):
        raise SystemExit("/live_pipeline/segment_sam3 is not advertised: start sam3_node first (see above)")

    stats, t0, done = [], time.time(), 0
    for i, s in enumerate(samples):
        key, c = C.sample_key(s), s["camera"]
        dst = out / f"{key}.png"
        if args.resume and dst.exists():
            continue
        shutil.copy(C.sample_dir(s) / f"rgb{c}.png", live / "rgb.png")
        for stale in ("sam3_masks.npy", "sam3_meta.json"):
            (live / stale).unlink(missing_ok=True)
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=180.0)
        p = live / "sam3_masks.npy"
        lab = np.load(p) if p.exists() else np.zeros((720, 1280), np.int32)   # zero instances: an empty mask
        cv2.imwrite(str(dst), ((lab > 0) * 255).astype(np.uint8))
        stats.append({"key": key, "camera": c, "instances": int(lab.max()), "frac": float((lab > 0).mean())})
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  {done}  {el:.0f}s  eta {(len(samples) - i - 1) * el / done:.0f}s", flush=True)

    meta = json.loads((live / "sam3_meta.json").read_text()) if (live / "sam3_meta.json").exists() else {}
    node.destroy_node()
    rclpy.shutdown()
    (out / "index.json").write_text(json.dumps(
        {"prompt": meta.get("prompt"), "frames": len(stats), "empty": sum(1 for r in stats if r["instances"] == 0),
         "mean_frac": float(np.mean([r["frac"] for r in stats])) if stats else 0.0,
         "seconds": round(time.time() - t0, 1), "rows": stats}, indent=1))
    print(f"  wrote {done} masks to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
