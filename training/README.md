# Training: the ReMake fine-tunes

The deployed depth-completion weights, `L_combo_both`, are ReMake's released checkpoint fine-tuned for six
more epochs with three changes to its training, and taken at epoch 43, the best on the held-out frames. This folder holds what is needed to check and re-run
that work: the code changes as one patch against public ReMake, the held-out frames used to pick each
run's epoch, and the config, provenance and scores of the deployed run and the runs it is compared with.

The weights themselves are in `checkpoints/` (`scripts/get_checkpoints.sh`). The transparency
segmenter's training code and its numbers are in [`segmenter/`](../segmenter/README.md).

## What is here

| Path | What it is |
|---|---|
| `remake_training_changes.patch` | every training-code change, as a `git diff` against ReMake `4f56814` (14 files). Applied to a clean checkout it reproduces the training tree's files byte for byte, except two wording changes that leave behaviour unchanged: a local variable in `run_tools/runner_train.py` renamed `is_best`, and a comment that names the trial mug as translucent |
| `val_list.json` | the held-out frames: 600 frames (stride 5) from 8 TransCG training scenes, 1, 19, 37, 54, 69, 92, 109 and 120. The trainer removes these scenes from training and picks the epoch with the lowest masked RMSE on them |
| `results/<run>/` | `official` (the released weights' scores: `provenance.txt` and the score files only), `A_ft` (the control), `G_maskdrop` and `L_combo_both`: `config.yaml` (what was trained; G_maskdrop's and L_combo_both's carry the pass rules written before the run in their header, and A_ft's gate is in its `RESULT.md`), `provenance.txt` (where the scored checkpoint came from: the commit, the checkpoint and how it was chosen, and for G_maskdrop and L_combo_both the seed and the starting checkpoint's md5), `masked_rmse*.txt` and `concavity*.txt` (the scores), `RESULT.md` (the verdict) |

### The changes in the patch

All are opt-in config settings, off by default, so an unchanged config trains as upstream does.

| Setting | What it does | Used by the runs here |
|---|---|---|
| `p_blank_in_mask` | with this probability, removes the sensor depth inside the transparent mask, so the network must fill the region from colour and the prior | L_combo_both |
| `p_mask_none`, `p_mask_erode`, `p_mask_dilate` | gives the network no mask, a shrunk mask or a grown mask, so it copes with an imperfect segmenter | G_maskdrop, L_combo_both |
| `rel_rgb_fix` | feeds the prior RGB. Upstream feeds it BGR (measured harmless) | L_combo_both |
| `align_rel_aug` | flips the Depth Anything prior's input with the same random draws as the rest of the sample. Upstream flips everything but the prior's input, so on 75 % of samples the prior sees a mirrored scene | none |
| `trainer.val_list` | the held-out validation pass above | G_maskdrop, L_combo_both |
| `--seed` | seeds torch, numpy and `random` (cuDNN is not pinned unless `--strict-determinism`) | G_maskdrop, L_combo_both, seed 1234 |
| `msgrad_beta`, `cameras`, `input_condition` | a multi-scale gradient loss, training on one TransCG camera, and blanked or mask-free test conditions on ClearGrasp | none |
| `smooth_type`, `mask_max_px`, `beta_full`, `beta_masked`, `uncertainty` (`uncertainty_tau`, `uncertainty_beta`), `msg_scales`, `msg_scale_invariant`, `input_mask_px`, `instance_masks` | further options of the earlier ladder: the smoothness loss's type, the mask-dropout kernel limit (15 px by default), loss weights, an uncertainty head, the gradient loss's scales, a mask-input width and per-instance masks. The configs here leave them at their defaults, except C_ft's `smooth_type: multiscale_gradient` | C_ft (`smooth_type`); G_maskdrop writes out the default `mask_max_px` |

The patch also adds `utils/hole_punching.py` (opaque objects' depth punched out as extra training
regions, `self_sup_holes`). No reported run used it.

## The runs

`L_combo_both` came out of a ladder of fine-tunes, each six epochs from `remake_official.tar` (epoch 39
to 45, learning rate 1e-4) with one change against the control `A_ft`. Two changes passed: blanking the
sensor's depth under the mask (`F_blank`: blanked 26.1 to 15.6 mm) and mask dropout (`G_maskdrop`), and
`L_combo_both` combines them. The other runs changed nothing or made things worse. Their records, and
those of the project's earlier retrains of ReMake, stay in the project's development records.

Masked RMSE in mm on the 1,177-frame TransCG test subset (`eval/sample_list.json`), one column per mask
ReMake is given (`eval/README.md` explains the conditions):

| Run | Change | GT mask | blanked | no mask | SAM box | Verdict |
|---|---|---|---|---|---|---|
| `official` | the authors' release, epoch 39 | 11.2 | | | | the starting point |
| `A_ft` | none: the control | 11.8 | 26.1 | 43.6 | 27.0 | the control holds |
| `G_maskdrop` | mask none / erode / dilate 0.25 / 0.15 / 0.15 | 11.1 | 26.8 | **17.0** | 23.1 | pass (SAM box a narrow miss) |
| **`L_combo_both`** | depth blanking + mask dropout + `rel_rgb_fix`, both cameras | 12.2 | **16.2** | **18.8** | **24.1** | **pass on every row: deployed** |

On the segmenter's mask grown 5 px, the mask the rig uses, `L_combo_both` scores 14.2 mm
(`results/L_combo_both/masked_rmse_seg_d5.txt`). Between two seeds of one recipe the masked RMSE moved
0.2 to 0.3 mm and concavity 5 points (measured on SGDCNet, an earlier depth network of this project).
Differences under 2 mm are read as ties.

`A_ft/RESULT.md` also reports `C_ft`, the multi-scale gradient loss in place of the normal loss
(config `A_ft/config_C_ft.yaml`): 76.9 mm, metric scale lost.

### How the runs differ in data and selection

- **`G_maskdrop` and `L_combo_both`** train on TransCG's training split minus the 8 held-out scenes
  (30,963 of 34,191 frames), with seed 1234, and are scored at the epoch with the lowest held-out masked
  RMSE.
- **`A_ft`, the control, ran earlier and differs on three points.** It trained on all 34,191 frames,
  is scored at its last epoch (45), and was not seeded. It ran from the training repository's
  `finetune-msgrad` branch (commit `bab3a81`), not the tree the patch rebuilds. So the comparisons
  with it are of one run each, under slightly different selection. Its training log and
  provenance are in `results/A_ft/` (`train_A_ft.log`: epochs 40 to 45 resumed from the released
  checkpoint's epoch 39; `provenance.txt`: commit `bab3a81`, last epoch evaluated).
- **The held-out scenes are not unseen by the starting weights.** Every fine-tune starts from the
  released checkpoint, which per the paper the authors trained on the whole training split (ReMake's release
  does not say). The held-out pass picks
  an epoch. It is not a clean validation score.

## Re-running a fine-tune

On a CUDA machine with 24 GB (the runs used an RTX 3090, Python 3.12.3 and torch 2.10.0+cu126). A
six-epoch fine-tune took about 9 hours.

```bash
REPO=/path/to/transparent-grasp-stack
git clone https://github.com/ChengYaofeng/ReMake.git remake && cd remake
git checkout 4f568148c8421544136bf49bb941149e0c990a34
git apply "$REPO/training/remake_training_changes.patch"
# environment: ReMake's own instructions (env.yaml)
mkdir -p ../eval/subset datasets/transcg checkpoints
cp "$REPO/training/val_list.json" ../eval/subset/              # the configs read ../eval/subset/val_list.json
ln -s /path/to/transcg datasets/transcg/transcg                 # TransCG (graspnet.net/transcg): metadata.json, scene1/, ...
cp "$REPO/checkpoints/depth_anything_v2_vits.pth" checkpoints/  # the prior's weights

RUN=L_combo_both
cp "$REPO/training/results/$RUN/config.yaml" configs/train/exp_$RUN.yaml
mkdir -p results/$RUN && cp "$REPO/checkpoints/remake_official.tar" results/$RUN/checkpoint.tar
python main.py -m train -c configs/train/exp_$RUN.yaml --reldepth_model depthanything \
    --logname $RUN --expname $RUN --seed 1234
```

A fine-tune resumes from the checkpoint copied into `results/<run>/checkpoint.tar` (epoch 39) and stops
at the config's `max_epoch`. The trainer writes `results/<run>/checkpoint<N>.tar` whenever the held-out
score improves, and its log (`results/<run>/train_<run>.log`) ends with `Min val masked RMSE: ... in
epoch N`, the epoch to score. Because cuDNN is not pinned, a re-run should match the record within the noise
floor, not bit for bit (no fine-tune was re-run for this release).

**To score a checkpoint**, use the evaluation harness in this repository: `eval/predict_transcg.py
--checkpoint <file>` (add `--rel-rgb-fix` for L_combo_both), then `eval/score.py`.
[`eval/README.md`](../eval/README.md) has the steps and the recorded rows.

## Notes on the records

- Config headers and some `RESULT.md` files name scripts and documents of the project's development
  repositories (`run_3090.sh`, `FINETUNE_BRIEF.md`, `REMAKE_TUNING_2026-09-30.md`, `baseline_report.md`,
  `SGDC_DESIGN.md`, `ORIN_TRANSFER.md`, the report's `.tex` chapters), and runs of the ladder whose
  records are not in this release (`E_flipfix`, `F_blank`, `J_l515`, `K_l515_combo`, and `M_combo_scratch`, the from-scratch run
  `L_combo_both/RESULT.md` names as the arbiter: it confirmed the mask gains but not the blanking gain,
  and missed its ClearGrasp guard by 2.6 mm). None of them is part
  of this release. The commands above are what `run_3090.sh` ran.
- In `provenance.txt`, the `final-epoch` line's note "test-selected above" is a leftover. Where a run
  has a `selected:` line naming the held-out pass, that is how its checkpoint was chosen.
- Machine paths in the records are shortened to paths inside the training repository. A few
  sentences were reworded for accuracy: `A_ft/RESULT.md` says "the pre-registration" for the plan
  written before its run and describes its planned grasp test without the statistic it named, and
  `L_combo_both/RESULT.md` names A_ft as the control (it was not the robot's weights). The configs'
  comments carry no dates. The project's earlier retrains of ReMake appear in `G_maskdrop` and
  `L_combo_both` as "an earlier retrain", and `official/provenance.txt` no longer compares the released weights with
  them. `A_ft/RESULT.md` has a dated note under C_ft's per-epoch line (its epoch-39 figure is on another
  sample set). `official/provenance.txt`'s training-data line said ClearGrasp-syn, a belief of the time, and
  now says TransCG, as the paper trains (the release does not say). One number changed:
  `L_combo_both/RESULT.md` gave its segmenter-mask row as 3.1 mm off its own GT-mask row, which is the gap to
  G_maskdrop's row. Against its own row it is 2.0 mm (14.2 against 12.2).
- Not included: weights other than the released ones, per-frame predictions and the raw training
  logs, except `A_ft/train_A_ft.log` for the control the others are compared with (so `train_C_ft.log`, which
  `A_ft/RESULT.md` cites, is not here). Also not included: the training code's shape metrics and its
  `eval/analyze.py`, which wrote the `shape.txt` files and which the score files' last line names.
