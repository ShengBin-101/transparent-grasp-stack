"""align.composite_depth fill_holes: False (the default) keeps ReMake's depth to the mask; the sensor's holes elsewhere
stay empty. True fills them with ReMake's depth too.

    docker exec remake_grasp python3 /opt/remake_grasp/tests/test_composite_fill_holes.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp.align import composite_depth  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


H, W = 120, 160
model = np.full((H, W), 600.0, np.float32)          # ReMake: a surface at 600 mm everywhere
sensor = np.full((H, W), 800.0, np.float32)         # the sensor: the table at 800 mm
mask = np.zeros((H, W), bool)
mask[20:100, 30:80] = True
sensor[50:60, 40:50] = 0.0                          # a hole on the object (inside the mask)
sensor[0:30, 120:160] = 0.0                         # no return beyond the table's edge (outside the mask)

default, _ = composite_depth(model, sensor, mask)
on, _ = composite_depth(model, sensor, mask, fill_holes=True)
ck("fill_holes True: a hole outside the mask takes the model (the compositing V1 ran under)",
   np.all(on[0:30, 120:160] == 600.0))
off, st = composite_depth(model, sensor, mask, fill_holes=False)
ck("the default is fill_holes False", np.array_equal(default, off))
ck("fill_holes False: a hole outside the mask stays empty", np.all(off[0:30, 120:160] == 0.0))
ck("fill_holes False: inside the mask, the model (hole or not)", np.all(off[mask] == 600.0))
ck("fill_holes False: outside the mask, the sensor where it reads", np.all(off[~mask & (sensor > 0)] == 800.0))
ck("stats: the empty pixels are counted", abs(st["px_no_data"] - 30 * 40 / (H * W)) < 1e-9, st["px_no_data"])
cov = np.ones((H, W), bool)
cov[20:40, 30:40] = False
c, _ = composite_depth(model, sensor, mask, valid_region=cov, fill_holes=False)
ck("fill_holes False: nothing outside the depth camera's field of view, mask or not", np.all(c[20:40, 30:40] == 0.0))

print("%d failed" % len(fail) if fail else "all passed")
sys.exit(1 if fail else 0)
