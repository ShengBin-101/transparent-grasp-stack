# Robot trials

The pre-registered grasping trials on the Franka Research 3, and their records.

**Outcome:** 54 attempts at the top-down view (V1), 18 per condition. Transparent objects held:
RAW 1 of 12 (8 %), BASELINE 8 of 12 (67 %), ADAPTED 7 of 12 (58 %) (`trial_totals.py`). The side view (V2) was not run
because a capture check found a reachable grasp on only 1 of 6 objects (`v2_capture_check.csv`).

## The conditions, and their names

| Condition | Run label in the records | Command | Preset |
|---|---|---|---|
| RAW | `raw` | `rg-raw` | `PIPELINE=raw` |
| BASELINE | `baseline (targets=depth)` | `rg-baseline` | `PIPELINE=baseline` |
| ADAPTED | `adapted` | `rg-adapted` | `PIPELINE=adapted` |

BASELINE's runs carry the label of their day, `baseline (targets=depth)`, which `paired_replay.py`
maps to today's `baseline`. ADAPTED was renamed after the trials: `trial_runs.csv`, the schedule,
the replay summaries, the view-debug results and the run folders were relabelled to match, and
nothing else in them changed.

## Files

| File | What it is |
|---|---|
| `trial_runs.csv` | every attempt: condition, object, outcome, failure code, the archived run and its pick records |
| `trial_schedule.csv` | the counterbalanced schedule, generated from a fixed seed (20261005) before the trials, by a development tool that is not in this release |
| `trial_totals.py` | held rates per condition and object kind, from `trial_runs.csv` |
| `timings.py` | the run times the README and the dissertation quote, with their definitions: capture to grasp pose (wall clock from the run request to the grasp pose, before planning or motion), each stage, planning, an executed pick (medians from the run folders) |
| `v2_capture_check.csv` | the side-view check: each object's SBG grasps, those under 70 mm, the executable ones, and the candidates the planner checked and found feasible from home |
| `make_trial_figures.py` | the results chart and the per-attempt detail for one view, from the records and the run folders |
| `build_trial_runs.py` | builds `trial_runs.csv` from the schedule, the session tracker and the run folders |
| `view_debug/` | the side-view analysis: ReMake's false surface where the sensor reads nothing, and the object-only fill that followed |
| `paired_replay.py`, `paired_replay.sh`, `paired_replay/` | every condition on the same archived frames, offline, and the summaries behind the reported replays |

The run folders the records name (`runs/trial/...`) are the release asset `trial_runs_v1.tar.zst`, which also holds
the six captures of the side-view check.

## Reading the records

- **The records.** `trial_runs.csv` and the run folders were written during the project and are kept
  as written, except that ADAPTED's records were relabelled (above). The pre-registration, the protocol with its later changes and the
  operator's procedure stay in the project's development records.
- **Which totals stand.** The plan was 162 attempts (6 objects x 3 views x 3 conditions x 3). The
  third view (V3) was dropped after its first five attempts (55 to 59), which were not counted and are not
  in these records. That left 108 at V1 and V2: the plan the
  dissertation reports. The side view (V2) was not run, after the capture check above. So 54 attempts at the top-down view (V1) ran and were counted.
- **Dates.** The trials ran on 27 Oct (V1, blocks 1 to 3, and V3's first five attempts) and on 4 Nov
  (the V2 capture check and the table clearing).
- **Compositing at V1.** Blocks 1 to 3 ran with ReMake's depth in every unread pixel, the rule
  then. After block 4's first attempts the rule was restricted to the transparency mask, and V1
  stands as run. At V1 the difference is small (0.2 % of pixels per frame), and `paired_replay.py`'s
  *-FULL-FILL conditions replay V1 as it ran.
- **Rates.** Success is reported as percentages with their counts (decided before the first
  attempt). Success rate is the only measure.
- **How results were entered.** During the sessions each outcome went into a tracker page, and
  `trial_runs.csv` was built from the tracker's export with `build_trial_runs.py`, and the five uncounted V3
  attempts were then removed from it. The tracker and its export are not published: `trial_runs.csv` is the record.
- **Outside the dissertation:** the V3 attempts, the table clearing, and the exploratory MASK-ONLY and
  dim-light robot blocks, which were never run. MASK-ONLY was replayed offline
  (`paired_replay/aft_20261029.csv`), and that replay is in the dissertation: the released weights on the
  segmenter's mask.

## Paired offline replay

On the robot each attempt runs one condition, so no two conditions see the same frame. Every capture
archives what all of them start from, so offline each frame can go through every condition. With
the pipeline down (`rg-down`; it needs the GPU), in the container:

```bash
trials/paired_replay.sh runs/trial/<run>... --out runs/paired/<name> [--conditions RAW,BASELINE,ADAPTED]
```

It writes `summary.csv` (SBG's counts and the executable grasps per run and condition, where
`executable_on_glass` and `best_on_glass` count grasps on any transparent object) and checks itself
against the robot's own record. Arm reachability is the NUC planner's replay, on the host:
`cd nuc && ./setup_franka.sh --ros` once (the planner reads franka_ros's joint limits and collision meshes), then
`python3 tools/replay_plan.py ../runs/paired/<name>/<run>/<condition> --voxel-mm 10`.

`paired_replay/` keeps the summaries the reported replays used, with the condition names of their
date. Up to 27 Oct (`keep_nearer`, `objects`, `views`) the bare BASELINE and ADAPTED ran full fill, today's
*-FULL-FILL, and *-OBJECT-FILL is today's default. From 29 Oct (`aft`, `cone`) the bare names are today's
default:

| File | What was replayed |
|---|---|
| `views_20261027_v1.csv` | 18 V1 trial captures (3 per object) under every condition; all 18 reproduce the robot's records |
| `views_20261027_v1_blank02.csv` | the same frames with random blanking: SBG's noise floor for the side-view analysis |
| `views_20261027_side.csv` | 22 side-view captures (V2, V3 and earlier oblique ones) |
| `objects_20261027.csv` | two captures of the trial objects before the first attempt (one is the RAW pipeline's end-to-end check) |
| `aft_20261029.csv` | 62 frames under every condition and the A_ft control (`view_debug/aft_compare.py`) |
| `cone_20261104.csv` | 31 frames with SBG's approach cone at 30°, 20° and 15° |
| `keep_nearer_20261023.csv` | the keep-nearer compositing rule, measured and then left off (the rule is no longer in the code) |

The trial asset holds the 60 run folders `trial_runs.csv` lists, the six captures of the side-view
check, and a copy of `trial_runs.csv` (the same rows, with Windows line endings). It covers both
`views_20261027_v1` files in full, 55 of `aft_20261029`'s 62 frames, 24 of `cone_20261104`'s 31 and 1
of `views_20261027_side`'s 22. The rest are not in this release: the five uncounted V3 attempts; in
`cone_20261104`, seven captures of the table clearing; the other side-view frames, and all of
`objects_20261027` and `keep_nearer_20261023`, which are development captures.

`v2_capture_check.csv` comes from each side-view check capture's `grasp_poses.json` (grasps with `width_m` under
0.070, and those marked `executable`) and from the planner's replay of the capture,
`cd nuc && python3 tools/replay_plan.py ../runs/trial/<run> --voxel-mm 10`, kept as
`view_debug/results/plan_v2check.json`. The box's first capture, made before it was re-placed, shows the
operator's arm and is left out.

To rerun `view_debug/analyse.py` or `view_debug/aft_compare.py` on these summaries, copy each
`paired_replay/<name>.csv` to `<dir>/<name>/summary.csv` and copy `view_debug/results/` to `<plans>`. Then

    python3 trials/view_debug/analyse.py <dir> <plans>
    python3 trials/view_debug/aft_compare.py <dir>/aft_20261029 <plans>/plan_aft.json <out.md>

The first prints its tables and rewrites `<plans>/view_debug_tables.json`, which equals the shipped file. The
second writes its tables to `<out.md>` (no copy of them ships).

`view_debug/` also holds three figures. `fig_views.jpg` shows a photo, the L515's height above the table and
the ReMake composite's for one V1, one V2 and two V3 captures (magenta: no depth). `fig_conditions.jpg` shows
RAW, BASELINE, ADAPTED and ADAPTED with the object-only fill on one V3 and one V2 capture. Both come from
`fig.py`. `fig_tall_sensor.jpg` marks in red where the L515's own depth reads more than 12 cm above the table on
four captures: the box and the glass cup at V3, the mug at V2 and a bottle at V1. Only the two V3 captures
show any red: on and around the box, which stands about 155 mm tall as placed, and in a small patch
inside the glass cup's mouth.
