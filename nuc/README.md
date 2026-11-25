# nuc: the arm's side

This folder is the ROS 1 Noetic workspace for the NUC, the PC that drives the Franka Research 3. It
moves the arm to the Cartesian poses the ROS 2 pipeline on the Orin requests over `ros1_bridge`, and
it can be watched from Foxglove on another machine. Only this folder runs on the NUC, and everything
here runs headless.

| Command | What it does |
|---|---|
| `./setup_franka.sh` | once: fetch `franka_ros` 0.10.1 with the stiffness patch, and build libfranka 0.10.0 |
| `./build.sh` | build the workspace (`--check`: verify the build) |
| `./run_home.sh` | before the controller: move to the saved capture pose. **Moves the arm.** `--check` only checks; `--save` saves where the arm is now |
| `./run_controller.sh` | terminal 1: the arm's impedance controller (the one the trials used) |
| `./run_position_controller.sh` | terminal 1, instead of `run_controller.sh` (never both: each claims the arm): the position controller, an alternative not used in the reported trials |
| `./run_commander.sh` | terminal 2: turns requested poses into motion (`--position` with the position controller, `--dry-run` to rehearse) |
| `./run_stiffness.sh` | show the controller's stiffness, set one gain, `--watch` the tracking error, `--reset` |
| `./run_viewpoint.sh <1-5>` | move to a capture viewpoint. **Moves the arm.** `--dry-run` plans and shows only |
| `./run_bridge.sh` | terminal 3, optional: foxglove_bridge, rosbridge and the URDF server, for Foxglove (`--no-description` without the server) |
| `./run_mock.sh` | no robot: a mock arm and both bridges, for Foxglove |

**Contents:** [1. Set up the NUC](#1-set-up-the-nuc) ·
[2. Run a session](#2-run-a-session) · [3. Without the robot](#3-without-the-robot) ·
[4. Keep the NUC current](#4-keep-the-nuc-current) · [5. Foxglove](#5-foxglove) ·
[6. Reference](#6-reference)

---

## 1. Set up the NUC

### What it needs

**The NUC.** Ubuntu 20.04 with ROS Noetic, and these, because libfranka's 1 kHz control loop must
finish every cycle inside 1 ms:

- a **PREEMPT_RT kernel**: `franka_control` runs with `realtime_config: enforce`, and libfranka then
  refuses to start unless `/sys/kernel/realtime` reads 1;
- real-time limits for the account that runs the controller (rtprio 99, for example through a `realtime` group in
  `/etc/security/limits.d/`);
- the CPU governor at `performance`;
- a **gigabit** Ethernet link to the robot. At 100 Mb/s every controller aborts with
  `communication_constraints_violation`; `tools/fci_link.sh` checks the link before each run.

**The robot.** A Franka Research 3 whose FCI server is version 6, the version libfranka 0.10.0
speaks (a mismatch fails at connect with *"Incompatible library version"*), with FCI activated in
Desk. In Desk:

- **The end-effector profile:** the Franka Hand with the camera bracket declared as part of the
  load, 1.03 kg in all (the trials used Desk's `end-effector-camera` profile; centre of mass
  (-0.010, 0, 0.030) m in the flange frame, see [6.6](#66-the-end-effector-load)). The hand's TCP
  is Desk's nominal one, (0, 0, 0.1034) m and -45° about z, which puts `fr3_EE` on `fr3_hand_tcp`.
  `rosrun fr3_pose_commander idle_wrench.py` reads back what the robot has.
- **Collision thresholds:** franka_control's defaults (`franka_control_node.yaml`: 20 N in
  translation). The contact guard ([6.3](#63-three-things-that-will-bite-you)) is set to stop
  the grasp leg before they trip.

### Install and build

```bash
git clone https://github.com/ShengBin-101/transparent-grasp-stack.git ~/transparent-grasp-stack
cd ~/transparent-grasp-stack/nuc
sudo apt install gcc-9 g++-9 libpoco-dev libeigen3-dev python3-scipy python3-yaml \
    ros-noetic-combined-robot-hw ros-noetic-controller-manager ros-noetic-dynamic-reconfigure \
    ros-noetic-robot-state-publisher ros-noetic-xacro ros-noetic-tf2-geometry-msgs \
    ros-noetic-foxglove-bridge ros-noetic-rosbridge-server    # the last two only for Foxglove
./setup_franka.sh          # franka_ros 0.10.1 + the stiffness patch, libfranka 0.10.0 from source
rosdep install --from-paths src --ignore-src -y --skip-keys libfranka    # whatever else franka_ros asks for
./build.sh                 # gcc-9, against that libfranka (6.7 says why)
./build.sh --check         # every line should say ok
```

The scripts find the workspace from their own location, so the clone can live anywhere (`WS=...`
overrides it). Leg logs go to `logs/` here, which is not in git.

### The network

| Machine | Address | Ports |
|---|---|---|
| the robot (FCI) | `172.16.0.2` (`ROBOT_IP`) | Desk on https; FCI on 1337 |
| the NUC | `172.16.0.1` | ROS master 11311 and each node's own port (`ROS_IP=172.16.0.1`); Foxglove 8765; rosbridge 9090 |
| the Orin | on `172.16.0.0/24` | runs `ros1_bridge` against the NUC's master |

These are Franka's default addresses and the ones every script defaults to. ufw is active on the
NUC, so open it to the subnet: `ros1_bridge` reaches every node's own port, not only the master.

**The Orin shares the robot's FCI link.** Keep transfers to the NUC small while a controller runs: on
27 Oct 2026 a 17 MB copy over ssh tripped the robot's communication reflex. Copy files with the arm
idle, or over another network.

```bash
sudo ufw allow from 172.16.0.0/24
```

The Orin's side of the connection is in [`docs/REFERENCE.md`](../docs/REFERENCE.md#connecting-the-nuc).

## 2. Run a session

On the NUC (or over `ssh nuc` from the Orin), keep a hand on the stop button:

1. **Home the arm**, before the controller starts (the controller holds the posture it starts in):
   ```bash
   cd ~/transparent-grasp-stack/nuc
   ./run_home.sh --check      # checks only; every line should say ok
   ./run_home.sh              # moves to the saved capture pose (asks first)
   ```
2. **Terminal 1:** `./run_controller.sh`. Leave it running.
3. **Terminal 2:** `./run_commander.sh`. Leave it running. If it warns that `ee_to_camera`
   differs from the Orin's copy, stop and fix the calibration first ([6.5](#65-the-camera-calibration)).
4. **Terminal 3, as needed:** `./run_stiffness.sh` to check the gains, `./run_viewpoint.sh <1-5>`
   to move to a capture viewpoint, or `./run_bridge.sh` for Foxglove.

Then, on the Orin, the pipeline's pick (`rg-pick`) drives the arm through this commander: the main
README's [Pick with the arm](../README.md#pick-with-the-arm).

- Every `run_*.sh` launcher except `run_stiffness.sh` first runs `tools/sync_check.sh`, which warns about uncommitted edits in this
  workspace and about C++ older than its source. It only warns.
- To use the position controller instead, start `./run_position_controller.sh` in terminal 1 (never
  together with `run_controller.sh`) and `./run_commander.sh --position` in terminal 2
  ([6.4](#64-the-position-controllers-elbow)).

## 3. Without the robot

**A mock arm, for laying out Foxglove panels:**

```bash
./run_mock.sh              # mock arm + both bridges + the commander in dry run
./run_mock.sh --still      # hold a posture instead of sweeping
```

This publishes a posture on `/joint_states` and lets robot_state_publisher build the same `/tf`
tree the real arm produces.

**It is not a simulator.** Nothing in it responds to a command. Trigger a grasp and the commander
resolves it, checks it and logs a ramp, but the drawn arm does not follow: nothing connects the
equilibrium setpoint back to the mock posture. The commander is forced into dry run here, as there is
no controller behind it, so nothing publishes the commanded motion either: its log shows each
resolved pose and ramp.

**Piece by piece.** Everything except `run_controller.sh` runs without the robot:

```bash
roscore &
./run_bridge.sh                 # both bridges and the URDF server
./run_commander.sh --dry-run    # poses resolved and checked, nothing published
```

In dry run every refusal and every resolved pose is real, and nothing is sent to the controller.
`--no-checks` is the opposite: refusals become warnings and the arm moves anyway. A refusal you
overrule was still true.

**The planner offline.** `tools/replay_plan.py` re-runs the planner on recorded captures, with no
ROS and no robot: [6.8](#68-replaying-the-planner-offline).

## 4. Keep the NUC current

The NUC runs from its own clone. After pulling:

- **A pull rebuilds nothing.** Python and launch files run straight from `src/`, but the C++
  (`fr3_position_control`) needs `./build.sh <package>` afterwards. `build.sh` records the commit it
  built in `devel/.built_from`, and `tools/sync_check.sh` compares it with the checkout.
- **A running node keeps the code it started with.** Restart the commander after a pull, and the
  controller too if its C++ was rebuilt.

## 5. Foxglove

Optional, for watching the arm. `foxglove_bridge` is installed with the **`assets`** capability, so
Foxglove loads the URDF's `package://` meshes through the bridge itself.

1. On the NUC: `./run_bridge.sh` (or `./run_mock.sh`).
2. In Foxglove: **Open connection → Foxglove WebSocket**, `ws://172.16.0.1:8765`.
3. Open a 3D panel and add the URDF from the `/robot_description` parameter. The meshes load
   themselves.
4. Set the 3D panel's **display frame** to `fr3_link0`. Transforms live in the panel's own
   Transforms section, not in the topic list.

`run_bridge.sh` also starts rosbridge on 9090, for websocket clients that are not Foxglove;
`serve_description.py` (`bridge.launch description:=true`, which `run_bridge.sh` passes unless
`--no-description`) serves the meshes over HTTP for such a client, which cannot resolve `package://` itself. Neither is needed by the pipeline.

**If Foxglove cannot connect,** check ufw first ([1](#the-network)), then that the bridge is
serving, from the NUC:

```bash
rosparam get /foxglove_bridge/port    # 8765
ss -tn state established '( sport = :8765 or sport = :9090 )'   # one line per connected client
```

The `ss` connection table is the reliable test. Node names are not: a Foxglove client on rosbridge
registers a `/foxglove_<digits>` node that can outlive the client (`rosnode cleanup` purges it),
and one on foxglove_bridge registers none. If Foxglove shows nothing while `/tf`, `/joint_states`
and `/robot_description` arrive, the fault is the panel's configuration (step 4).

---

## 6. Reference

### 6.1 What is in this folder

| Path | What it is |
|---|---|
| `src/fr3_pose_commander` | this workspace's own package: the commander, its planner, and the nodes below |
| `src/fr3_pose_commander/test/` | 15 test scripts, each run from `nuc/` as `python3 src/fr3_pose_commander/test/<name>.py`. `test_kinematics` and `test_leg_log` need only Python and NumPy, `test_planning`, `test_scene` and `test_viewpoints` also need `src/franka_ros` (`./setup_franka.sh --ros`), and the others import `rospy`, so run them on the NUC. None needs the robot |
| `src/fr3_position_control` | the position controller ([6.4](#64-the-position-controllers-elbow)), an alternative not used in the reported trials |
| `src/franka_ros` | not in git: `setup_franka.sh` clones upstream tag **0.10.1** (`30e598a`) and applies `patches/franka_ros-0.10.1-stiffness.patch`, which raises two ceilings in `franka_example_controllers/cfg/compliance_param.cfg`: `translational_stiffness` from 400 to 1000 N/m and `rotational_stiffness` from 30 to 100 N m/rad. dynamic_reconfigure clamps silently, so without the patch `controller.launch`'s 1000 and 50 would land as 400 and 30 |
| `local_libfranka_0_10_0/` | not in git: libfranka **0.10.0** (`4f9e3cc`), built from source by `setup_franka.sh` ([6.7](#67-why-buildsh-and-not-a-readme-line)) |
| `patches/` | the `franka_ros` stiffness patch, with its change notice |
| `tools/` | `replay_plan.py` (6.8), `goto_viewpoint.py` (behind `run_viewpoint.sh`) and `home_from_viewpoint.py`, `fci_link.sh`, `sync_check.sh`, and the diagnostics below |
| `run_*.sh`, `setup_franka.sh`, `build.sh` | one script per terminal, the Franka dependencies, and the build |

`fr3_pose_commander`'s nodes:

- **`pose_commander_node.py`** takes a desired pose, checks it, and walks the impedance
  controller's equilibrium to it ([6.2](#62-the-interface)).
- **`publish_camera_tf.py`** puts the wrist camera on the tf tree.
- **`publish_tool_pose.py`** streams the measured tool pose to the Orin.
- **`serve_description.py`** serves the URDF and meshes over HTTP (5. Foxglove).

**Diagnostics**, each run by hand with the controller up. Those marked **moves** drive the arm
(through the commander, so its checks apply). `stall_test.py` asks before each phase (`--yes` skips
that), and `measure_legs.py` starts moving at once:

| Script | What it measures |
|---|---|
| `scripts/idle_wrench.py` | the declared payload and the noise floor of the external-wrench estimate, arm still |
| `scripts/watch_tracking.py` | the arm tracking its equilibrium, live in a terminal |
| `scripts/stall_test.py` | **moves:** whether a standoff miss is a steady force (correctable) or a lack of authority, at two stiffnesses; `--standoff X Y Z` is rig-specific; writes `~/stall_tests/stall_test_<time>.json` |
| `scripts/gripper_test.py` | whether a successful grasp stays closed; moves the fingers only |
| `tools/measure_legs.py` | **moves:** ten small legs (±3 cm, ±15° and ±30° of roll), each followed by `go_home` |
| `tools/compare_legs.py` | the planner's predicted end posture against the measured one, from the leg log |

### 6.2 The interface

The topic and service names match the ROS 2 sequencer on the Orin
(`ros/remake_grasp/remake_grasp/sequencer.py`):

| Name | Type | Direction and meaning |
|---|---|---|
| `/live_pipeline/grasp_pose_best` | `PoseStamped` | in: the pose for the next leg; `frame_id` must be set |
| `/live_pipeline/grasp_width_best` | `Float64` | in: the fingers for that leg, in m; negative = leave them alone |
| `/live_pipeline/pick_candidates` | `String` | in: JSON, every candidate's legs in `fr3_link0`, for `plan_pick` |
| `/live_pipeline/plan_pick` | `Trigger` | service: choose the candidate the arm can follow; JSON reply |
| `/live_pipeline/move_to_pose` | `Trigger` | service: ramp to the pose, then apply the width |
| `/live_pipeline/guarded_move_to_pose` | `Trigger` | service: the same, but stop on contact |
| `/live_pipeline/measure_load` | `Trigger` | service: the standing external force, averaged (relaxing the impedance setpoint first); JSON reply |
| `/live_pipeline/release` | `Trigger` | service: open the fingers |
| `/live_pipeline/go_home` | `Trigger` | service: back to home, the forward kinematics of `home_pose.yaml` (viewpoint 1), in planned legs; fingers untouched |
| `/live_pipeline/go_v1` | `Trigger` | service: the trials' top-down view, `go_home` itself |
| `/live_pipeline/go_v2` | `Trigger` | service: the trials' side view (`views/v2` in `pose_commander.yaml`: `run_viewpoint.sh 4 --tilt 45 --height 0.35`), planned and run like `go_home`; no confirmation prompt; fingers untouched |
| `/live_pipeline/execution_status` | `String` | out: `ok: ...` or `refused: ...` |
| `/live_pipeline/tool_pose` | `PoseStamped` | out: the measured tool pose in `fr3_link0`, at most 30 Hz (`publish_tool_pose.py`) |
| `/live_pipeline/contact_pose` | `PoseStamped` | out: where a guarded leg touched, latched |
| `/live_pipeline/contact_wrench` | `WrenchStamped` | out: what it felt, minus the baseline, latched |
| `/franka_state_controller/franka_states` | `FrankaState` | in: the measured tool pose, joints and wrench |
| `/cartesian_impedance_example_controller/equilibrium_pose` | `PoseStamped` | out: the setpoint |
| `/cartesian_pose_tracking_controller/target_pose` | `PoseStamped` | out: the setpoint instead, with `run_commander.sh --position` |

**Each trigger is one leg, not a pick.** A pose sitting on `grasp_pose_best` does nothing until
`move_to_pose` or `guarded_move_to_pose` is called; then the arm ramps there and the fingers are
set. The staged pick (standoff, guarded grasp, retreat) is three of these, issued by the Orin's
sequencer.

**One command at a time.** A second trigger while one runs returns `already moving` and does not
queue. Publishing a new pose mid-motion does not redirect the arm: the command in flight finishes,
and the new pose waits for the next trigger.

**Transits are straight lines.** Every leg, the moves home and to a view included, is a straight
Cartesian line sampled and checked by the planner. Every reported pick used them.

### 6.3 Three things that will bite you

**The controller ignores `header.frame_id`.** It copies `msg.pose` straight into its target. A pose
handed over in the camera frame is accepted in silence and driven to as if it were in the base
frame. So `pose_commander_node` resolves the frame itself through tf, and **refuses** a pose whose
frame is empty or unresolvable rather than guessing.

**The setpoint is a force, not a destination.** The impedance law turns the offset between the
equilibrium and the arm into a force proportional to it: a 3 cm jump at the 1000 N/m this rig runs
asks for 30 N, past the 20 N collision reflex. So the setpoint is ramped to the goal at a bounded
speed (12 mm/s and 0.4 rad/s, `ramp/` in `pose_commander.yaml`). The last setpoint is the goal.

**The stock launch file fights you.** `franka_example_controllers`' own
`cartesian_impedance_example_controller.launch` starts `interactive_marker.py` with
`required="true"`, and it publishes to the same `equilibrium_pose` topic. Two writers to one
setpoint make an arm that argues with itself, and nothing reports it. Use `controller.launch` from
this package, which leaves it out (and rviz and rqt_reconfigure, which abort over SSH).

### 6.4 The position controller's elbow

`./run_position_controller.sh` (`fr3_position_control`) commands a Cartesian pose. It is an
alternative to the impedance controller and was not used in the reported trials. The arm has seven
joints and a pose fixes six numbers, so one is left over. libfranka lets the caller fix it with an
**elbow**: joint 3's angle, plus the flip sign of joint 4 (−1 on this arm).

With `hold_elbow: true` (the default, in `config/fr3_position_control.yaml`), the controller reads
the elbow the robot is already commanding (`elbow_c`) when it starts, and sends that same value
every cycle, so joint 3 is held at its start angle. That is what the commander's planner assumes
for this controller (model `elbow_locked`). `HOLD_ELBOW=false ./run_position_controller.sh` lets the
firmware choose joint 3.

The start-up log says which mode ran and the latched values
(`elbow HELD at joint 3 = ... rad, sign -1`). Firmware reflexes (safety stops) that an elbow problem
can trigger:

- `cartesian_motion_generator_start_elbow_invalid`: the first elbow sent is too far from the
  robot's. This is why the latch is `elbow_c`, not `q[2]`.
- `cartesian_motion_generator_elbow_sign_inconsistent`: the sign changed. It is never changed.
- `cartesian_motion_generator_elbow_limit_violation`: the elbow would move into a joint limit.
  With joint 3 held, the other six joints must reach the pose alone. The planner checks that from
  the measured joints before each leg, and refuses the leg when its mode is `enforce`.

A reflex stops the motion and franka_control prints it. It clears itself once the controller
disconnects.

### 6.5 The camera calibration

One camera, one calibration, in `config/camera_extrinsics.yaml`:

```
ee_to_camera: [0.109234, 0.008700, -0.030235, 0.002532, -0.015076, 0.715291, 0.698659]
```

- This is the camera's optical frame in the controlled end-effector frame (what `O_T_EE` reports,
  published to tf as `fr3_EE`), as [x, y, z, qx, qy, qz, qw].
- It was calibrated on 30 Sep 2026 against an AprilGrid from 23 arm poses, refined on every tag
  corner: a held-out board error of 4.5 mm median, against 8.5 mm for the nominal mounting
  `[0.105, 0.0, -0.048]` with exactly +90° about z. The solver and how to rerun it are in
  [`tools/hand_eye/`](../tools/hand_eye/).
- The camera sits on a bracket **about 109 mm to the side of the gripper** (not coaxial) and about
  30 mm above the fingertip TCP. Both are correct.
- `fr3_EE` itself is not in the URDF. It comes from `F_T_NE`/`NE_T_EE` in Desk, so changing the
  tool there moves every one of these offsets. With the robot running,
  `rosrun tf tf_echo fr3_link8 fr3_EE` prints where Desk puts it.

**There are two copies, and they must change together.** The Orin resolves every grasp with its
own, `ros/remake_grasp/config/eye_in_hand.yaml`; this file drives the NUC's tf tree.
`run_commander.sh` compares them on every start (where it can read the Orin's file) and warns if
they differ. A difference is a fixed bias on every commanded pose that nothing else reports.

The camera rides on the gripper, so there is no static camera-to-base transform anywhere. Without
`publish_camera_tf.py` running there is no chain from the camera frame to the base frame, and every
camera-frame pose is refused.

### 6.6 The end-effector load

Measured off the robot with `rosrun fr3_pose_commander idle_wrench.py` (arm still, brakes open,
touching nothing):

```
m_ee 1.030 kg   m_load 0.000 kg   m_total 1.030 kg
centre of mass (flange frame)  (-0.010, 0.000, 0.030) m
```

**The mass is right and the centre of mass is not.** 1.030 kg is Desk's end-effector-camera
profile, so the bracket is declared. But `(-0.010, 0, 0.030)` is the bare hand's centre of mass,
Franka's figure for the plain 0.73 kg hand: the mass was updated when the
camera went on, the centre of mass was not.

**Left as it is, deliberately.** With the calibrated mounting (6.5) the camera's optical centre is at
`(0.083, -0.071, 0.073)` in the flange frame (the y comes from the 45° rotation between `fr3_EE` and
`fr3_link8`, and the nominal mounting gives `(0.074, -0.074, 0.055)`). If the bracket's mass sat at
the camera, the combined centre would be `(+0.017, -0.021, +0.043)`; most of the bracket's 0.30 kg
is nearer the flange, so the true value lies between the two. That bounds the error at 36 mm,
**0.37 N m of unmodelled moment, or about 0.4° of angular droop** at 50 N m/rad.
Over a 100 mm approach, 0.4° displaces the far end by 0.7 mm.

**It does not affect the contact guard.** The guard thresholds the change from a reference taken
during each leg, and a centre-of-mass error produces a moment that depends only on orientation. The
grasp leg is a straight translation with one orientation throughout, so the error is constant
across it and subtracts out. The same holds for the estimate's standing offset (3.5 N measured with
the arm still).

**A trap for whoever fixes it.** `controller.launch`'s stated load centre `(0.093, 0, 0.030)` has
y = 0, which cannot be right for a camera the tf tree puts about 71 mm off-axis in y: it appears to have
been computed without the 45° flange-to-EE rotation.

**To get the right value** without CAD and without unmounting the camera (which would invalidate
`ee_to_camera`): hand-guide the arm to 10-15 well-spread wrist orientations, release it and let it
settle at each, record `O_F_ext_hat_K` and `O_T_EE`, and least-squares fit `tau = delta x g` for
`delta = m * (c_true - c_model)`. **Fit a constant sensor bias alongside it**, or the standing
offset aliases into the answer. Change it in Desk, never with `set_load`: applying a load restarts
the control loop.

### 6.7 Why build.sh, and not a README line

Three non-default CMake settings, all **cached** in `build/CMakeCache.txt`. Cached is the problem:
they vanish silently when `build/` is deleted, and none of the resulting failures looks like a
missing flag.

1. **gcc-9.** Under a newer GCC, `franka_hw` fails with `'uint8_t' does not name a type`:
   libstdc++ stopped including `<cstdint>` transitively, and `franka_hw` never included it itself.
2. **`franka_gazebo` blacklisted.** Its generated Ignition protobuf headers disagree with the
   installed protobuf. It is simulation only; nothing on the real arm needs it.
3. **libfranka 0.10.0, built from source** by `setup_franka.sh`. apt ships `ros-noetic-libfranka`
   0.9.2, and the FR3 speaks FCI server version 6, so linking it gives *"Incompatible library
   version (server version: 6, library version: 5)"* the moment `franka_control_node` touches the
   robot. `build.sh --check` verifies the soname that actually got linked.

### 6.8 Replaying the planner offline

`tools/replay_plan.py` runs the commander's planner on recorded captures (a run folder from the
Orin's `runs/`), with no ROS and no robot. It walks every candidate's full chain from home, against
the scene voxelised from the depth the detector used, and reports which are feasible, why the
others are refused, and which one each ranking would choose:

```bash
python3 tools/replay_plan.py <run_dir> [<run_dir> ...] --voxel-mm 10 --json out.json
```

- Every replay reported for the trials used `--voxel-mm 10` (the pose node's value, and the default)
  and the default `--model elbow_locked`
  (the posture model of the position controller). The trials themselves ran the impedance
  controller (`--model nullspace`); on the side-view check both models give the same feasible
  counts.
- The objects come from the run's `target_objects.npy` (the grasp-pose node's merged targets),
  else `seg_masks.npy`, else `sam3_masks.npy`; `--labels flags` uses one target for every
  candidate instead.
- The capture pose comes from `capture_tool.json`, else from a pick record in the run.
- It needs `src/franka_ros` (`./setup_franka.sh --ros` is enough) for the joint limits and the collision
  meshes, and numpy, scipy and PyYAML. Without the meshes the arm is checked as capsules, which refuses
  more grasps than the robot did, and the replay prints a warning.

### 6.9 Rig-specific values

These belong to one rig. Re-measure them before using another, or after moving the table:

| Value | Where | How it was measured |
|---|---|---|
| the table surface, z = −0.041 m in `fr3_link0` | `pose_commander.yaml` `planner/table_z` (the code's fallback is `planning.TABLE_Z`) | guarded contact |
| the floor limit, z = −0.036 m | `pose_commander.yaml` `limits/z_min` | 5 mm above the table |
| the capture pose (home) | `config/home_pose.yaml` | `./run_home.sh --save` or `tools/home_from_viewpoint.py` |
| the hand-eye | `config/camera_extrinsics.yaml` and the Orin's `eye_in_hand.yaml` | [6.5](#65-the-camera-calibration) |
| the end-effector profile | Desk | [6.6](#66-the-end-effector-load) |
