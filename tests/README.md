# tests

Developer tests, kept as they were run. Some need nothing but this repo; others replay archived
runs, which are not in this repo. The groups below say which. A check whose archived run is missing
prints `skip` with the run it needs, and the rest of the test still runs.

For a check that needs only this repo and the weights, use the offline demo instead:
`scripts/demo_offline.sh all`. The benchmark, `scripts/benchmark.sh`, also needs the TransCG cut
([`docs/REPRODUCE.md`](../docs/REPRODUCE.md)).

## Run a test

In the container, `run_in_container.sh` runs a test with the live nodes' environment (ROS, the
workspace, CUDA's libraries):

```bash
tests/run_in_container.sh test_tmuxp.py
RG_RUNS=/opt/remake_grasp/runs tests/run_in_container.sh test_target_filter.py   # tests that read archived runs
RG_RUNS=/opt/remake_grasp/runs/trial tests/run_in_container.sh test_opaque_targets.py   # the trial archive's runs
```

The `test_*` files print one line per check, then `all passed` or the failures. The replays and the
equivalence checks print their comparison with the recorded output.

## What each test needs

**Nothing but the repo**

| Test | Checks |
|---|---|
| `test_tmuxp.py` | the four tmux session files agree; `rg_session.sh` against a fake tmux |

**The container**

| Test | Checks | Also needs |
|---|---|---|
| `test_sequencer_summary.py` | the pick summary text, refusals, the pick record, the weight check, one pick per grasp | |
| `test_remake_variants.py` | the `PIPELINE` presets and every `MASK` / `REMAKE` / `TARGETS` override; every launch parameter has its node's declared type; `rel_rgb_fix`; the segmenter service | the weights; `RG_RUNS` for its replay cases |
| `test_pins.py` | the pins node | |
| `test_composite_fill_holes.py` | `align.composite_depth`: ReMake's depth only inside the mask by default (`fill_holes` false, the trials' rule), every sensor hole with `fill_holes` true | |
| `test_sbg_cone.py` | SBG's approach cone: with the cone off the wrapper gives upstream's outputs exactly; with it on, every view is the best-scoring one inside the cone | the SBG weights and a GPU |
| `test_run_export.py` | run_export: the PCD/PLY writers, gripper drawing, capture pose and calibration choice, the executed grasp from a pick record | |
| `check_seg_overlay.py` | `seg_node` publishes its overlay on `/live_pipeline/seg_overlay`, identical to `seg_masks.png` | `seg_node` running on a live directory with an `rgb.png` (pass the directory) |
| `test_orchestrator_wait.sh` | `run_chain` waits for every stage's service before capturing | `ROS_DOMAIN_ID` 77 and `fake_trigger.py` |

**Archived runs** (the rig's `runs/`, not in git)

| Test | Checks | Also needs |
|---|---|---|
| `test_target_filter.py`, `test_opaque_targets.py` | which objects and targets the grasp-pose node picks | `RG_RUNS` |
| `replay_remake.py`, `replay_sbg.py`, `replay_sam3.py` | re-run one stage on an archived run with the settings it recorded (weights, mask, compositing, approach cone) and compare with its recorded output | a run folder, e.g. from the trial archive |
| `replay_peggnet.py` | the same for PEGG-Net; runs without a PEGG-Net stage, such as every trial run, are skipped | a run recorded with PEGG-Net |

The trial archive (release asset `trial_runs_v1.tar.zst`, unpacked into `runs/`, which gives `runs/trial/<run>`;
set `RG_RUNS` to `runs/trial` for the tests) works with the replays. On a fresh
build, a BASELINE and an ADAPTED trial run reproduced exactly: ReMake's depth to 0.000 mm and SBG's grasps row for row.

```bash
source /opt/ros/humble/setup.bash
python3 tests/replay_remake.py runs/trial/20261027_052859_auto runs/trial/20261027_062735_auto
python3 tests/replay_sbg.py runs/trial/20261027_062735_auto
python3 tests/replay_sam3.py runs/trial/20261027_052859_auto      # BASELINE runs carry SAM 3's mask
```

**Helpers**

| File | What it is |
|---|---|
| `run_in_container.sh` | runs a test with the live nodes' environment (above) |
| `fake_nuc.py` | stands in for the NUC's commander over ROS 1 (`move_to_pose`, `guarded_move_to_pose`, `release`, `go_home`), not its planner; needs the `ros1tools` image target ([below](#stand-in-nuc)) |
| `fake_trigger.py` | a stand-in stage service that comes up after a delay, for `test_orchestrator_wait.sh` |

## Stand-in NUC

To test the bridge without the robot, run `fake_nuc.py` in the ROS 1 tools image, then the bridge
in the container against it. It has no planner, so set `plan_pick: auto` in
`ros/remake_grasp/config/eye_in_hand.yaml` first (the shipped `required` refuses every pick without one):

```bash
docker build -f docker/Dockerfile --target ros1tools -t remake_grasp:ros1tools .
docker run -d --name fake_nuc --network host --entrypoint bash -v $PWD/tests:/tests \
    -e ROS_MASTER_URI=http://127.0.0.1:11311 -e ROS_IP=127.0.0.1 remake_grasp:ros1tools \
    -c "roscore & sleep 4; python3 /tests/fake_nuc.py"
# in the container:
ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1 ros2 launch remake_grasp bridge.launch.py
```
