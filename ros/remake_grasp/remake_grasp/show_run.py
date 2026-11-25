"""Print a one-screen summary of the live directory (use with `watch`).

    ros2 run remake_grasp show_run [live_dir]
"""
import json
import sys
from pathlib import Path

from .common import LIVE_DIR


def _load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    live = Path(sys.argv[1]) if len(sys.argv) > 1 else LIVE_DIR
    rc = _load(live / "run_config.json")
    if not rc:
        print(f"{live}: no run yet (call /live_pipeline/run_chain)")
        return
    cap = rc.get("stages", {}).get("capture", {}).get("settings", {})
    print(f"run started {rc.get('started')}  camera {cap.get('camera')}  total {rc.get('total_seconds')}s"
          + (f"  pipeline {cap['pipeline']}" if cap.get("pipeline") else ""))
    for name, st in rc.get("stages", {}).items():
        extra = st.get("error") or ", ".join(f"{k}={st[k]}" for k in ("n_instances", "n_grasps", "n_accepted",
                                                                        "n_executable") if k in st)
        print(f"  {name:<18} {st.get('seconds', 0):6.2f}s  {extra}")
    poses = _load(live / "grasp_poses.json")
    if poses:
        best = poses.get("best")
        if best:
            p = best["grasp_pose"]["position"]
            w = best["width_m"]
            print(f"best grasp: {best['source']} label={best['label']} width="
                  f"{'?' if w is None else f'{w * 1000:.0f}mm'} at [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] "
                  f"in {best['frame_id']}")
        else:
            print("best grasp: none executable")
        for g in poses["grasps"][:8]:
            w = "?" if g["width_m"] is None else f"{g['width_m'] * 1000:.0f}mm"
            conf = g.get("score", g.get("quality", 0.0))
            print(f"  {'*' if g['executable'] else ' '} {str(g['label']):>4} {g['source']:<8} {w:>6} "
                  f"conf {conf:.2f}  {g['recommendation']}")


if __name__ == "__main__":
    main()
