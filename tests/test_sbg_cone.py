"""Tests SBG's approach cone (sbg_core._cone_forward) against SBG's own ApproachNet.forward.

    tests/run_in_container.sh test_sbg_cone.py

With the cone off the wrapper must give upstream's outputs exactly; with it on, every seed's view
must lie in the cone, be the best-scoring view there, and agree with upstream wherever upstream's
own choice was already in the cone. Needs the SBG weights (scripts/get_checkpoints.sh) and a GPU.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp import sbg_core as SC  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


net = SC.ScaleBalancedGrasp(config=SC.SbgConfig(approach_cone_deg=None))
import modules  # noqa: E402  upstream models/modules.py, put on sys.path by ScaleBalancedGrasp
from loss_utils import generate_grasp_views  # noqa: E402

z = np.load(ROOT / "samples/three_transparent/arrays.npz")
cloud, _ = SC.cloud_from_depth(z["depth_aligned_mm"] / 1000.0, z["color_intrinsics"])
cloud = cloud[SC.sample_points(len(cloud), 20000, np.random.default_rng(0))]
pc = torch.from_numpy(cloud).unsqueeze(0).to(net.device)


def end_points(forward, cone_deg=None, axis=(0.0, 0.0, 1.0)):
    modules.ApproachNet.forward = forward
    if cone_deg:
        SC._CONE["axis"], SC._CONE["cos"] = list(axis), float(np.cos(np.radians(cone_deg)))
    else:
        SC._CONE["axis"], SC._CONE["cos"] = None, -2.0
    with torch.no_grad():
        out = net.net({"point_clouds": pc})
    modules.ApproachNet.forward = SC._cone_forward
    return {k: v.detach().clone() for k, v in out.items() if torch.is_tensor(v)}


up = end_points(SC._UPSTREAM_FORWARD)
off = end_points(SC._cone_forward)
ck("cone off: same end_points keys as upstream", up.keys() == off.keys())
ck("cone off: every end_points tensor equals upstream's", all(torch.equal(up[k], off[k]) for k in up))

views = generate_grasp_views(300).to(net.device)
for deg in (30.0, 15.0):
    on = end_points(SC._cone_forward, deg)
    allowed = (-views @ torch.tensor([0.0, 0.0, 1.0], device=net.device)) >= np.cos(np.radians(deg))
    ind = on["grasp_top_view_inds"]
    ck(f"{deg:.0f} deg: every chosen view is inside the cone", bool(allowed[ind].all()))
    best = on["view_score"].masked_fill(~allowed, float("-inf")).max(dim=2).values
    ck(f"{deg:.0f} deg: the chosen view is the best in the cone", torch.equal(on["grasp_top_view_score"], best))
    same = allowed[up["grasp_top_view_inds"]]
    ck(f"{deg:.0f} deg: agrees with upstream where upstream's view was in the cone",
       torch.equal(ind[same], up["grasp_top_view_inds"][same]), f"{int(same.sum())} seeds compared")
    ck(f"{deg:.0f} deg: rotation is the chosen view's", torch.allclose(-on["grasp_top_view_rot"][..., 0],
       on["grasp_top_view_xyz"], atol=1e-5))

print("all passed" if not fail else f"{len(fail)} failed: {fail}")
sys.exit(1 if fail else 0)
