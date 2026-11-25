#!/usr/bin/env python3
"""Make capture viewpoint 1 the home pose: write its joints into home_pose.yaml.

    python3 tools/home_from_viewpoint.py            print what would be written
    python3 tools/home_from_viewpoint.py --write    write it (the old value is kept
                                                    in a comment beside it)
    ... --centre 0.50 0.0 --height 0.50             the same knobs as run_viewpoint.sh

Moves nothing and needs no robot: viewpoint 1 is planned from the current home_pose.yaml joints
under the position controller's model (joint 3 held). Writing joints rather than a pose pins
the posture for run_home.sh, and the commander takes its home tool pose from the same joints.
After --write, run run_home.sh (before the controller); go_home, run_home.sh and
run_viewpoint.sh 1 then all end at the same posture.
"""
import argparse
import datetime
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "src", "fr3_pose_commander")
sys.path.insert(0, os.path.join(PKG, "src"))
from fr3_pose_commander import planning, viewpoints  # noqa: E402

HOME_YAML = os.path.join(PKG, "config", "home_pose.yaml")
LIMITS = os.path.join(HERE, "..", "src", "franka_ros", "franka_description", "robots", "fr3",
                      "joint_limits.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--centre", type=float, nargs=2, default=viewpoints.DEFAULT_CENTRE,
                    metavar=("X", "Y"))
    ap.add_argument("--height", type=float, default=viewpoints.DEFAULT_HEIGHT)
    ap.add_argument("--home-yaml", default=HOME_YAML)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg_yaml = yaml.safe_load(open(os.path.join(PKG, "config", "pose_commander.yaml")))
    ee_to_camera = yaml.safe_load(open(os.path.join(PKG, "config", "camera_extrinsics.yaml")))[
        "ee_to_camera"]
    table_z = float(cfg_yaml["planner"]["table_z"])
    cfg = planning.PlanConfig(
        ee_to_camera=tuple(ee_to_camera[:3]),
        rotation_max=float(cfg_yaml["limits"]["rotation_max"]),
        joint_margin=float(cfg_yaml["planner"]["joint_margin"]), table_z=table_z,
        linear_speed=float(cfg_yaml["ramp"]["linear_speed"]),
        angular_speed=float(cfg_yaml["ramp"]["angular_speed"]))
    limits = planning.load_limits(LIMITS)

    text = open(args.home_yaml).read()
    q_old = [float(v) for v in yaml.safe_load(text)["home_joints"]]
    plan = viewpoints.plan_viewpoint(1, q_old, viewpoints.home_pose(q_old), limits, cfg,
                                     ee_to_camera, tuple(args.centre), table_z, args.height,
                                     max_leg_deg=planning.leg_split(cfg)[0])
    if not plan["ok"]:
        sys.exit("viewpoint 1 is not reachable from the current home: %s" % plan["reasons"][0])
    q_new = plan["q_end"]
    old_tool, new_tool = viewpoints.home_pose(q_old), viewpoints.home_pose(q_new)
    print("viewpoint 1: camera at %s, looking at %s"
          % (np.round(plan["camera_position"], 3).tolist(), np.round(plan["target"], 3).tolist()))
    print("  old home_joints %s  tool %s" % (np.round(q_old, 4).tolist(),
                                            np.round(old_tool[0], 3).tolist()))
    print("  new home_joints %s  tool %s" % (np.round(q_new, 4).tolist(),
                                            np.round(new_tool[0], 3).tolist()))
    print("  largest joint change %.3f rad; wrist turn %.1f deg; worst joint margin %.2f rad; "
          "sigma_min %.3f" % (max(abs(a - b) for a, b in zip(q_old, q_new)), plan["turn_deg"],
                              plan["joint_margin"], plan["sigma_min"]))
    if not args.write:
        print("not written (add --write)")
        return
    line = "home_joints: [%s]" % ", ".join("%.9f" % v for v in q_new)
    stamp = datetime.date.today().isoformat()
    note = ("# VIEWPOINT 1 SINCE %s (tools/home_from_viewpoint.py): camera %.2f m above the\n"
            "# table, looking straight down at (%.2f, %.2f). Planned from the previous\n"
            "# home joints under the position controller's model; the previous value was\n"
            "#   [%s]\n"
            % (stamp, args.height, args.centre[0], args.centre[1],
               ", ".join("%.9f" % v for v in q_old)))
    new_text, n = re.subn(r"^home_joints:.*$", lambda _m: note + line, text, count=1, flags=re.M)
    if n != 1:
        sys.exit("no home_joints line in %s" % args.home_yaml)
    open(args.home_yaml, "w").write(new_text)
    print("wrote %s -- now run_home.sh to drive the arm there" % args.home_yaml)


if __name__ == "__main__":
    main()
