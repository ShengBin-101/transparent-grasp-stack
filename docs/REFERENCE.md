# Reference

Commands, settings, outputs and fixes. Set-up and first runs are in the [README](../README.md).

## Commands in the container

| Command | What it does |
|---|---|
| `rg-adapted`, `rg-baseline`, `rg-raw` | start that trial condition's preset, in a tmux session (one session at a time). ADAPTED and BASELINE now run with the object-only fill; the V1 trials ran with full fill |
| `rg-up` | start `PIPELINE` with any `MASK` / `REMAKE` / `TARGETS` set in the shell |
| `rg-run` | one capture through every stage (`/live_pipeline/run_chain`) |
| `rg-pick` | pick the best grasp (`/live_pipeline/execute_pick`) |
| `rg-down` | stop every node and close the session |
| `rg-build` | rebuild the ROS package, once after adding a file under `ros/` |
| `ros2 service call /live_pipeline/<name> std_srvs/srv/Trigger` | `release` opens the fingers; `go_home` returns to the start pose; `go_v1`, `go_v2` move to the trials' top-down and side views ([`nuc/README.md`](../nuc/README.md#62-the-interface)) |
| `scripts/check_setup.sh` | what this setup can run |
| `scripts/preflight.sh` | on the host: code, camera, NUC and pipeline ready for a trial block |
| `scripts/demo_offline.sh`, `scripts/benchmark.sh` | the offline demo and the TransCG benchmark ([REPRODUCE](REPRODUCE.md)) |
| `trials/paired_replay.sh` | every condition on the same archived frames, offline |
| `tests/run_in_container.sh <test>.py` | one test with the nodes' environment ([`tests/README.md`](../tests/README.md)) |

The `rg-*` commands are defined in `docker/env.sh`.

## Settings

Each variable overrides one setting of the preset. Set it before `rg-up`, or pass it to the launch
files as `mask:=`, `remake:=`, `targets:=`. The `rg-<preset>` commands ignore all of them but `PROFILE`.

| Variable | Values |
|---|---|
| `PIPELINE` | the trial conditions: `adapted` (default), `baseline`, `raw` |
| `MASK` | `seg` (the segmenter), `sam3` (SAM 3, "transparent object"), `none` |
| `REMAKE` | `L_combo_both` (fine-tuned, ADAPTED's), `official` (the authors', BASELINE's), `G_maskdrop` and `A_ft` (the benchmark's and the paired replay's), `none` (SBG on the raw depth) |
| `TARGETS` | `depth` (the trials' rule: transparent from the segmenter, opaque from the raw depth: at least 2 cm tall, 9 cm², 12 mm wide, wholly in view), `transparent` (the mask's objects only, the rule before the trials), `prompt` (SAM 3's grasp prompt: PEGG-Net's input, only with `DETECTOR=peggnet`), `none` |
| `DETECTOR` | `sbg` (default; every trial) or `peggnet` (one top-down grasp per object, not used in the trials) |
| `PROFILE` | SBG's `picking` (30° approach cone, the trials' setting), `picking20`, `picking15`, or `perception` (upstream SBG) |

Every run records its preset and overrides in `run_config.json`.

## Container options

Set before `./start.sh`; they are fixed when the container is created (`./start.sh --recreate` to
change them, `./start.sh --stop` to stop it).

| Variable | Default | Sets |
|---|---|---|
| `RG_DATASETS` | none | a folder holding `transcg/transcg/`, mounted at `/datasets` |
| `ROS_DOMAIN_ID` | `0` | one per container on a shared network |
| `NUC_IP`, `NUC_HOSTNAME`, `ROS_IP` | `172.16.0.1`, none, found | how to reach the NUC's ROS 1 master |
| `RG_SSH` | `0` | `1` mounts `~/.ssh` read-only and copies it into the container, for `ssh nuc` from inside |
| `IMAGE`, `NAME` | `remake_grasp:latest`, `remake_grasp` | the image and the container's name |
| `HF_HOME` | `~/.cache/huggingface` | the host's Hugging Face cache, mounted for SAM 3's weights and login |

SAM 3's weights are gated on Hugging Face. Log in once with `hf auth login`, on the host or in the
container: the Hugging Face cache is mounted, so the login and the weights persist.

The repo is mounted at `/opt/remake_grasp`, so edits on the host are live after a node restart.
To add software, list it in `docker/apt-packages.txt`, `docker/ros-packages.txt` or
`docker/requirements.txt`, then `./build.sh` and `./start.sh --recreate`.

## Running the parts by hand

| Part | Launch |
|---|---|
| camera | `ros2 launch remake_grasp camera.launch.py` |
| mask and ReMake | `ros2 launch remake_grasp remake.launch.py pipeline:=adapted` |
| SBG / PEGG-Net | `ros2 launch remake_grasp sbg.launch.py pipeline:=adapted` / `peggnet.launch.py` |
| capture, chain, grasp poses, clouds | `ros2 launch remake_grasp pipeline.launch.py detector:=sbg pipeline:=adapted` |
| bridge and pick sequencer | `ros2 launch remake_grasp bridge.launch.py` (`dry_run:=true` never moves the arm) |
| Foxglove | `ros2 launch remake_grasp foxglove.launch.py` |

`remake.launch.py`, `pipeline.launch.py` and `sbg.launch.py` must get the same `pipeline:=` and overrides
(SBG picks its depth file from them), and the
detector must be running before the chain calls it.

## Connecting the NUC

1. The NUC must be reachable at `172.16.0.1`, with `export ROS_IP=172.16.0.1` on the NUC (or
   `NUC_HOSTNAME=<host> ./start.sh`).
2. Build the NUC side on the NUC: [`nuc/README.md`](../nuc/README.md).
3. Add SSH access from the Orin as host `nuc` in `~/.ssh/config`, then `RG_SSH=1 ./start.sh --recreate`
   (the container copies `~/.ssh` when created, and only with `RG_SSH=1`).
4. Check: `ros2 service list | grep move_to_pose` lists the NUC's services once its commander runs.

The Orin is on the robot's FCI subnet, so keep large copies to the NUC for when the arm is idle
([`nuc/README.md`](../nuc/README.md#the-network)).

## Output files

Each run writes to `data/live/`, then is archived in `runs/<date>_<time>_auto/`, or `runs/<date>_<time>_failed_<stage>/` when a
stage fails (`MANIFEST.txt` lists the files). Depth is float32 millimetres, 0 for no reading.

| File | Written by |
|---|---|
| `rgb.png`, `depth_aligned_mm.npy`, `depth_coverage.npy`, `color_intrinsics.npy` | capture |
| `seg_masks.npy` / `sam3_masks.npy`, with `.png` overlays | the segmenter / SAM 3 |
| `remake_depth_mm.npy`, `remake_depth_vis.png` | ReMake |
| `grasps_sbg.json`, `grasps_sbg_overlay.png` | SBG |
| `grasp_poses.json`, `target_objects.npy` | grasp poses: each grasp's object and whether it is executable |
| `capture_tool.json` | the arm's pose at capture |
| `pick_*.json` | each pick: the planner's reply, every leg, the load check |
| `cloud_sensor.pcd`, `cloud_completed.pcd`, `grasps.pcd`, `grasp_final.pcd` | point clouds in the robot base frame, written after archiving |
| `run_config.json` | every stage's settings and timing |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `start.sh`: "needs CUDA 12.6 and cuDNN 9 from the host" | the image's PyTorch uses the host's CUDA 12.6 and cuDNN 9: Jetson Linux R36.4 or later (JetPack 6), tested on R36.5.2 |
| a node dies loading weights | `scripts/check_setup.sh`, then `scripts/get_checkpoints.sh` |
| L515 depth at 0 Hz | `ros2 launch remake_grasp camera.launch.py initial_reset:=true`, then `pkill -9 -f realsense2_camera_node` and start again |
| "no camera frames received yet" | `rg-down` and start again (two camera drivers, or none) |
| SAM 3 401 on download | request access at huggingface.co/facebook/sam3, then `hf auth login` |
| "waiting ... for segment_sam3" | SAM 3 takes 30-40 s to load |
| bridge "waiting for the ROS 1 master" | the NUC's roscore is not up or not reachable |
| disk full while building | `docker image prune`, `docker builder prune` |
