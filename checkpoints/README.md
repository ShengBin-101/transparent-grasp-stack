# checkpoints

The model weights. They are not in git; only this README and `SHA256SUMS` are.

## Fill this folder

```bash
scripts/get_checkpoints.sh                    # check every file; download what is missing
scripts/get_checkpoints.sh --from <folder>    # copy files of the same names from a folder first
scripts/check_setup.sh                        # in the container: reports anything still missing
```

Every file is checked against `SHA256SUMS`. Third-party weights come from their authors. The
project's own four come from this repository's GitHub Release
[`v1.0`](https://github.com/ShengBin-101/transparent-grasp-stack/releases/tag/v1.0).
`RG_WEIGHTS_URL=<base url>` fetches `<base url>/<file>` for those four from somewhere else.

## What each file is

| File | Model | Where it comes from | Licence |
|---|---|---|---|
| `remake_official.tar` | ReMake as its authors released it (epoch 39): `REMAKE=official`, the BASELINE preset | the ReMake README's Google Drive link, file `1AF9sWyhEoNHnAlJ_y1AXsoxgrXq44333` (gdown) | none stated |
| `depth_anything_v2_vits.pth` | Depth Anything V2 Small: ReMake's frozen relative-depth branch, and the segmenter's frozen encoder | `huggingface.co/depth-anything/Depth-Anything-V2-Small`, not gated | Apache-2.0 |
| `scale_balanced_full_model.tar` | Scale-Balanced Grasp | the SBG README's Google Drive folder `1Y2o0uAbhS6-yZhPkKMAAnL0tCpbiJfk0`, file `log_full_model/checkpoint.tar` (gdown) | none stated |
| `peggnet_jacquard_d_480.pt` | PEGG-Net, Jacquard depth, 480 px (`DETECTOR=peggnet`) | `github.com/HZWang96/PEGG-Net` at `be3ddfd`, `output/pre-trained_models/220205_1108_peggnet_jacquard_d_480/epoch_44_iou_0.94_statedict.pt` | BSD-3-Clause |
| `L_combo_both_checkpoint43.tar` | ReMake fine-tuned on TransCG (both cameras) with in-mask blanking, mask dropout and `rel_rgb_fix`, epoch 43: `REMAKE=L_combo_both`, the ADAPTED preset | release v1.0 | CC BY-NC 4.0 |
| `seg_vits_rgb_best.pt` | the prompt-free transparency segmenter's head (`MASK=seg`, `TARGETS=depth`), the ADAPTED preset | release v1.0 | CC BY-NC 4.0 |
| `A_ft_checkpoint45.tar` | ReMake fine-tuned with the same schedule as the other arms but the shipped data loader, epoch 45: the control in `training/`, and the comparison in the paired replay (`trials/`) | release v1.0 | CC BY-NC 4.0 |
| `G_maskdrop_checkpoint45.tar` | ReMake fine-tuned with mask dropout only, epoch 45 (the arm `L_combo_both` superseded): benchmark rows in `eval/README.md` | release v1.0 | CC BY-NC 4.0 |

The project's four are trained on TransCG, so they are released under its terms (CC BY-NC 4.0).
`A_ft` and `G_maskdrop` are for the benchmark and the replay; no preset uses them, and
`REMAKE=A_ft` or `REMAKE=G_maskdrop` selects them by hand.

How each scores on TransCG: `eval/README.md` (reproduced in the container with
`scripts/benchmark.sh`). **L_combo_both was trained with `rel_rgb_fix` but not `align_rel_aug`.**
`REMAKE=L_combo_both` sets `rel_rgb_fix` (from the `REMAKE` table in `remake.launch.py`, which
`eval/predict_transcg.py` reads too); at run time it changes the output by < 0.1 mm.

SAM 3 is not here: it downloads into `~/.cache/huggingface` (mounted into the container) the first
time `sam3_node` runs, once Meta has approved access to `facebook/sam3` and you have logged in with
`hf auth login` in the container. 3.45 GB.
