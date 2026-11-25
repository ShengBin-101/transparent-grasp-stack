# tools/hand_eye

The hand-eye calibration of the wrist L515: where the camera's optical frame sits in the arm's
controlled end-effector frame (`fr3_EE`, what `O_T_EE` reports). The result is the seven numbers
`ee_to_camera` in `nuc/src/fr3_pose_commander/config/camera_extrinsics.yaml` and
`ros/remake_grasp/config/eye_in_hand.yaml`, which must hold the same values.

## The board

A calib.io AprilGrid lying fixed on the table: 9 x 6 tag36h11 tags, 30 mm, 9 mm gaps (39 mm
pitch). OpenCV needs `markerBorderBits = 2` for AprilGrid tags (`board.py`). The board's layout
(which cell each id is in) was found from the images (`board.find_layout`, which `plan_poses.py`
runs) rather than assumed, and `solve.py` and `loo.py` use the layout it found, `(9, 1, True)`.

## The method

The board does not move, so for every capture *i*

    T_base_board = T_base_EE(i) · X · T_cam_board(i)

where `X` is `ee_to_camera`. `solve.py`:

1. finds the board's pose in each image (`T_cam_board`, solvePnP and Levenberg-Marquardt, with the
   colour camera's distortion);
2. solves for `X` with OpenCV's five hand-eye methods (Tsai, Park, Horaud, Andreff, Daniilidis);
3. refines `X` and the board's pose in the base frame together, minimising the reprojection error
   of every tag corner in every image, from Park's solution.

The check that needs no ground truth is how consistently a given `X` puts the fixed board in one
place from every pose. `loo.py` is the held-out test: fit on all captures but one, and measure how
far the board lands from where the left-out image saw it.

## The captures

One folder with, for each pose `NN` (00, 01, ...):

- `NN.png`: the colour frame, still;
- `NN.json`: `{"tool": [[x, y, z], [qx, qy, qz, qw]]}`, the arm's `O_T_EE` in `fr3_link0` when the
  frame was taken;

and the colour intrinsics: `color_intrinsics_factory.npy` (the camera's own, which the pipeline
uses), and optionally `K_calibrated.npy` / `D_calibrated.npy` from calibrating the colour camera on
the same images. The script that recorded the 30 Sep captures is not in this repository; any way of
saving a still frame together with `O_T_EE` gives the same files. `plan_poses.py` plans the poses
around the board from one view of it, each from the one before (`poses.json` is the plan the 30 Sep
captures followed).

The 30 Sep 2026 captures are the release asset `handeye_captures_20260930.zip` (32 MB, CC BY 4.0).
Unpack it and pass the `handeye_20260930` folder as `CAPTURES`. Its JSON files also hold the arm's pose
just before and after the still (`tool_before`, `tool_after`) and the planned pose (`plan`).
`color_intrinsics.npy` there is a copy of `K_calibrated.npy`. `result.json` is the 30 Sep run's own
output with the recalibrated intrinsics; today's `solve.py --intrinsics calibrated` lands within 2 mm
of it, not exactly on it.

## Rerun the calibration

```bash
python3 tools/hand_eye/solve.py CAPTURES                 # factory intrinsics, as the pipeline uses
python3 tools/hand_eye/solve.py CAPTURES --intrinsics calibrated
python3 tools/hand_eye/loo.py CAPTURES                   # held-out error against the nominal mounting
python3 tools/hand_eye/validate.py CAPTURES/result_factory.json RUNS_DIR
```

They need numpy, scipy, PyYAML and OpenCV with the aruco module (4.5), and import the NUC
workspace's planner helpers from `nuc/tools/replay_plan.py` (no ROS). `validate.py` checks a result
on ordinary pipeline captures (the table's height and tilt in the raw depth, one mug seen from four
viewpoints); the captures it names are the rig's, which are not part of this repository.
`plan_poses.py` also reads the arm's joint limits from `nuc/src/franka_ros` (`cd nuc &&
./setup_franka.sh --ros`) and writes `poses.json` into the folder it is run from.

## The 30 Sep 2026 calibration

23 poses, all usable. From these captures the scripts give:

| | nominal mounting `[0.105, 0, -0.048]`, +90° about z | calibrated |
|---|---|---|
| board, held-out pose (`loo.py`) | 8.5 mm median, 11.3 mm rms | **4.5 mm median, 6.2 mm rms**; better on 18 of 23 |
| translation across the 23 leave-one-out fits | | x 109.4 ± 1.5, y 8.3 ± 0.7, z -31.5 ± 0.3 mm |

`solve.py` with the factory intrinsics gives exactly the installed value,
`[0.109234, 0.0087, -0.030235, 0.002532, -0.015076, 0.715291, 0.698659]`. With the recalibrated
intrinsics the result moves by under 2 mm. The remaining ~5 mm error changes sign between the two
sides of the workspace: it follows the arm's own kinematic accuracy, which no camera calibration
removes.
