# eval: the TransCG benchmark, inside the container

The TransCG benchmark: it runs ReMake over a frozen 1,177-frame test subset and scores the depth it
repairs. Run it in the container with `scripts/benchmark.sh <remake> <conditions...>`; the conditions
and what to check are in [`docs/REPRODUCE.md`](../docs/REPRODUCE.md#2-the-transcg-benchmark).

| File | What it does | From |
|---|---|---|
| `predict_transcg.py` | ReMake over the frozen subset under one condition; loads the model with the live pipeline's `remake_node.load_remake`, reads frames with upstream's TransCG reader | written for this repo, after the training code's `eval/predict.py` |
| `score.py` | masked RMSE, per camera and per sensor region | the training code's `eval/score.py`, import path changed, its shape metrics left out |
| `measure_concavity.py` | the share of each open vessel's ground-truth concavity the prediction keeps | the training code's, import path changed, unreadable frames skipped and counted |
| `common.py` | data IO and the metric implementation | the training code's `eval/common.py`, paths changed, its shape metrics left out |
| `make_seg_masks.py` | the segmenter's masks for the subset (the live `seg_node` call) | written for this repo |
| `make_transcg_subset.py` | the 2.6 GB of TransCG the benchmark reads | written for this repo |
| `sample_list.json` | the 1,177 frames: every 20th test frame, all 52 test scenes, 592 D435 + 585 L515 | frozen 2026-09-07 (md5 `82e1feca`) |
| `reference.json` | the recorded rows `benchmark.sh` compares each run with | the training records (`training/results/`) and the Orin's runs |

**The metric.** Masked RMSE is computed per frame over the pixels inside the dataset's transparent
mask that have ground truth (more than 1 cm), at the native 1280 x 720, and the row is the mean over
frames. The prediction is nearest-upsampled to that size first, the same for every model.

The TransCG reader (`third_party/remake/datasets/transcg.py`) and the preprocessing it calls are
upstream ReMake's at `4f56814`, unchanged. The only test-time setting any fine-tune changed is
`rel_rgb_fix` (L_combo_both), which `predict_transcg.py` takes from `remake.launch.py`'s table.

## The conditions

A condition changes only ReMake's input: the mask it is given, and for `blanked` the depth inside it.
Scoring always uses the dataset's mask.

| Condition | The mask ReMake gets |
|---|---|
| `dataset` | the dataset's own transparent mask: the benchmark protocol |
| `blanked` | the dataset's mask, with the sensor's depth inside it removed |
| `nomask` | none |
| `seg_d0`, `seg_d5` | the segmenter's mask, as is and grown 5 px (made on first use with `make_seg_masks.py`) |
| `sambox` | SAM (v1) ViT-B (`sam_vit_b_01ec64.pth`) prompted with a box around each connected region of the dataset's mask (touching objects share one): IoU 0.628 |
| `sam3text` | SAM 3 with the rig's text prompt of 24 Sep 2026 ("transparent object or bottle"): IoU 0.377, 72 empty frames |

The SAM-box and SAM 3 masks are release assets (`sam_masks_box.zip`, `sam3_masks_text.zip`), fetched
by `benchmark.sh` on first use. `tools/sam3_text_masks.py --out datasets/sam3_masks_text`, run from the repository's root, remakes the SAM 3
masks where `benchmark.sh` reads them, though not bit for
bit (SAM 3 is not deterministic across runs). `segmenter/mask_quality.py --mask-dir <dir>` scores
any folder of masks against the dataset's.

## The recorded rows

`reference.json`, masked RMSE in mm (L515 in brackets) and the share of concavity kept:

| Weights | dataset | blanked | nomask | seg_d5 | sambox | sam3text |
|---|---|---|---|---|---|---|
| `official` (the authors') | 11.2 (13.4), 89 % | 25.9 (29.1), 34 % | 46.4 (64.2), 84 % | 14.4 (17.6), 82 % | | 28.9 (40.4) |
| `L_combo_both` (ADAPTED) | 12.2 (14.8), 87 % | 16.2 (17.4), 89 % | 18.8 (25.0), 63 % | 14.2 (17.6), 76 % | 24.1 (33.5), 76 % | |
| `G_maskdrop` | 11.1 (13.2), 87 % | 26.8 (30.0), 32 % | 17.0 (22.5), 66 % | 13.3 (16.3), 82 % | 23.1 (32.3), 79 % | |
| `A_ft` (the control) | 11.8 (13.9), 87 % | 26.1 (29.8), 32 % | 43.6 (60.2), 82 % | | 27.0 (37.0), 76 % | |

The training workstation (an RTX 3090) recorded most rows between 24 Sep and 2 Oct 2026 (the released weights'
dataset row on 19 Sep, on its CPU); the
released weights' blanked, no-mask and segmenter rows were recorded on the Orin on 5 Oct, and their SAM 3 row
on 24 Sep (no concavity recorded). On 23 Nov `benchmark.sh official sam3text` reproduced all 1,177 of
that row's per-frame scores from the release masks. Differences under
about 2 mm or 5 concavity points are not treated as results (the dissertation's Section 6.1). Reruns on the
Orin matched the recorded rows to 0.1 mm (below).

## Reproduced on the Orin, 2026-10-05

In the container, TransCG mounted read-only, against the training workstation's rows:

| Row | Orin: masked RMSE (L515), concavity | Recorded | Difference |
|---|---|---|---|
| L_combo_both, dataset mask | 12.2 mm (14.8), 87 % | 12.2 mm, 87 % | 0.0 mm, 0 points |
| L_combo_both, blanked | 16.2 mm (17.4), 89 % | 16.2 mm, 89 % | 0.0 mm, 0 points |
| L_combo_both, no mask | 18.8 mm (25.0), 63 % | 18.8 mm, 63 % | 0.0 mm, 0 points |
| L_combo_both, segmenter mask +5 px | 14.2 mm (17.6), 76 % | 14.2 mm, 76 % | 0.0 mm, 0 points |
| released (official), dataset mask | 11.2 mm (13.4), 89 % | 11.2 mm, 89 % | 0.0 mm, 0 points |

The segmenter's masks, regenerated here, have IoU 0.850 per frame (0.855 pooled) against the
dataset's mask; the training records give 0.85 for the 5 px-dilated mask. Every row matches to the
tenth of a millimetre.

Not reproduced here: G_maskdrop's rows and A_ft's rows other than its dataset-mask row, which matched
(11.8 mm, 87 %) when rerun for this release (their weights are released, and `benchmark.sh` runs them), the dissertation's ClearGrasp rows (scored in the development environments, ReMake's with its own test
harness, TDCNet's with its own test code and DFNet's and LingBot-Depth's with scripts written for this
project, not in this release), and the ladder's other arms, whose weights are not released.
