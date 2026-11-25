# Reproducing the results

Set up first with the [README](../README.md)'s quick start. Everything below runs in the container
unless it says otherwise.

## 1. The pipeline on recorded captures

```bash
scripts/demo_offline.sh all                     # 4 captures x every preset they support, about 20 min
scripts/demo_offline.sh three_transparent adapted     # one capture, one preset
```

Each stage (the segmenter, ReMake, SBG, grasp poses) runs for real on the captures in `samples/`,
under the trials' three presets. SAM 3's masks are archived with the samples, so no Hugging Face
access is needed.

| Demo preset | What it is |
|---|---|
| `adapted` | the segmenter + `L_combo_both` + targets without a prompt: the trials' ADAPTED |
| `baseline` | SAM 3's mask + the released ReMake + the same targets: the trials' BASELINE (two of the samples have SAM 3's mask) |
| `raw` | the raw sensor depth into SBG, no mask and no target filter: the trials' RAW |

**Check:** each run ends with "targets the same" against `samples/<capture>/expected/<preset>.json`,
and prints its executable grasps beside the expected count. Outputs go to `data/demo/`. The samples
run without a robot, so the approach cone's "down" is the camera's optical axis, not the arm's tool
pose as on the rig.

## 2. The TransCG benchmark

Needs the 2.6 GB benchmark cut of TransCG: `python3 eval/make_transcg_subset.py <TransCG>/transcg`
writes it to `datasets/`.

```bash
scripts/benchmark.sh official dataset
scripts/benchmark.sh L_combo_both dataset blanked nomask seg_d5 sambox
scripts/benchmark.sh official sam3text                   # SAM 3's masks, prompt "transparent object or bottle"
LIMIT=20 scripts/benchmark.sh L_combo_both nomask         # a quick check
```

The conditions are in [`eval/README.md`](../eval/README.md). Each row ends by comparing itself with
the recorded row (`eval/reference.json`); rows rerun on the Orin matched to 0.1 mm. A row takes
about 20 minutes.

## 3. The robot trials

The attempt records are in [`trials/`](../trials/README.md). The archived
trial captures are the release asset `trial_runs_v1.tar.zst` (unpack it into `runs/`). Given them,
every condition can be replayed on the same frames, with the pipeline down (`trials/README.md` says
which of the kept replay summaries they cover):

```bash
trials/paired_replay.sh runs/trial/<run>... --out runs/paired/<name>
trials/paired_replay.sh runs/trial/<run>... --out runs/paired/<name> --conditions RAW,BASELINE-FULL-FILL,ADAPTED-FULL-FILL
```

The second line replays V1 as it ran, with every unread pixel filled. The replay reproduced the robot's own records on 18 of
18 frames. Arm reachability is then checked with the NUC planner offline, on the host:
`cd nuc && ./setup_franka.sh --ros` once (the planner reads franka_ros's joint limits and collision meshes), then
`python3 tools/replay_plan.py <paired out>/<run>/<condition> --voxel-mm 10`. Run with `--json` on the 48
trial V1 captures as recorded (`runs/trial/<run>`), it reproduces every verdict in
`trials/view_debug/results/plan_trial_v1.json`.

## Which result comes from where

| Result | Reproduced by | Needs |
|---|---|---|
| the pipeline on recorded captures | `scripts/demo_offline.sh` | the image and the weights |
| released and fine-tuned ReMake on TransCG, every mask | `scripts/benchmark.sh` | plus the TransCG cut |
| the segmenter's mask quality | `segmenter/predict.py`, `segmenter/mask_quality.py` | TransCG, ClearGrasp for the cross-domain rows |
| SAM 3's text-prompt masks (IoU 0.377) | the release's cached masks; `tools/sam3_text_masks.py --out datasets/sam3_masks_text` remakes them | SAM 3 access for a remake |
| training the fine-tunes and the segmenter | [`training/`](../training/README.md) and [`segmenter/`](../segmenter/README.md) | a 24 GB GPU, TransCG, the ReMake training code |
| the hand-eye calibration | [`tools/hand_eye/`](../tools/hand_eye/README.md) | the calibration captures (release asset `handeye_captures_20260930.zip`) |
| the side-view check (1 of 6 objects reachable) | `trials/v2_capture_check.csv`, from the captures' `grasp_poses.json` and `nuc/tools/replay_plan.py` | the trial asset, franka_ros for the planner |
| the run times | `trials/timings.py` | the trial asset |
| the robot trials | `trials/`, `nuc/` and the rig | the hardware, or the archived captures for replay |

## Not reproducible from this release

These findings of the dissertation have no record here. They were measured in development
environments whose records stay private, or on captures that are not shipped, and the dissertation
reports them as measured:

- the four-model comparison on the benchmark (DFNet, TDCNet, LingBot-Depth and the retrained
  ReMake), except the released ReMake's row, and the gap table's rows other than 11.2 to 28.9 mm;
- the ClearGrasp rows, scored in the development environments (ReMake's with its own test harness,
  TDCNet's with its own test code, DFNet's and LingBot-Depth's with scripts written for this project);
- the earlier retrain's error as its mask shrinks (10.5 to 55.8 mm, quoted in
  `training/results/G_maskdrop/RESULT.md`) and the curve drawn from it;
- the training runs other than the four shipped, among them the `F_blank` and `M_combo_scratch`
  rows of the training table (`F_blank`'s row is quoted in `training/results/L_combo_both/RESULT.md`),
  and the five-change retrain with its control (10.5 mm each, and the prior swap of about 0.5 mm);
- the prior bug's measured effect (its correlations and the 0.6 mm bound). The bug itself can be
  read in ReMake `4f56814` and in the patch's `align_rel_aug`;
- that a mask too small costs the released ReMake far more than one too large (Section 6.3), from an
  erosion and dilation sweep of its mask;
- that ReMake's colour augmentation changes most pixels even when set to make no change (Chapter 7);
- the second capture of the rig's mug, where only ten of the 13,700 depth readings over its footprint
  exceed 90 mm (Chapter 1);
- the 80 mm cavity test (4 % of the cavity kept inside the mask, 95 % outside) and the ten exploratory
  network variants;
- the Gemini 336L heights (within 6 and 8 mm against 23 mm);
- the rims read 47 to 57 mm too low on the tape-measured frames (one of the two frames ships, as
  `samples/rig_mug_box`);
- the segmenter's record on the rig (no transparent object missed in 15 captures, and the mug found in
  every archived run where SAM 3 found nothing; one such run ships, as `samples/sam3_missed_mug`), and
  the targets rule's count (104 of 300 archived runs);
- the comparison of three segmenters on ten rig scenes (the correct object count in 9, 5 and 0 of
  them) and PEGG-Net's 60 ms on the Orin;
- the targets rule's replay on 337 archived runs, and the early failed pick behind the planner's
  scene check with its replay (the approach met the mug 75 mm short, and the revised rule refuses that
  grasp for a finger within 9 mm);
- the comparison of approach cones made before the trials, and the planner's verdict on the 20° cone
  after them (a feasible grasp from the side on 3 of 6 objects: `trials/paired_replay/cone_20261104.csv`
  holds SBG's grasp counts only);
- the pair of runs with different seeds behind the noise margin (0.2 to 0.3 mm);
- the radar study's detector tests and REXO's result (Chapter 2). Three of its figures (the person
  views, Experiment 1's distances and Experiment 2) are redrawn by the
  [xwr_raw_ros](https://github.com/ShengBin-101/xwr_raw_ros) repository, which lists what cannot be
  rerun. Apart from three raw frames, no recordings are published;
- the drawing scripts of most figures. Their data ship for most of them (the release assets, the
  datasets and `eval/common.py`), but the scripts stay in the development records.

Never run: the x86_64 build path (written, untested).
