# segmenter

The prompt-free transparency segmenter: it marks the transparent pixels in an RGB frame, so ReMake
gets its mask without a text prompt. On the robot it runs as `seg_node` (`MASK=seg`, the ADAPTED
preset).

## What it is

A 1.28 M-parameter head on the frozen Depth Anything V2 ViT-S encoder, the encoder that ReMake's
relative-depth branch also runs (two separate passes on the robot).

- **Input:** RGB, resized to 518 x 686 and ImageNet-normalised.
- **Features:** the patch tokens of DINOv2 blocks 2, 5, 8 and 11 (the layers Depth Anything's DPT
  head reads), concatenated to 1536 channels on the 37 x 49 token grid.
- **Head:** a 1 x 1 convolution to 256 channels, two stages of bilinear x2 upsampling and a 3 x 3
  convolution, and a 1 x 1 logit. Its output grid is 148 x 196, 2/7 of the input, which is then
  resized bilinearly to the frame's size and thresholded at 0.5.
- **Training:** BCE plus soft Dice on TransCG's ground-truth transparency masks
  (`depth{c}-gt-mask.png`, both cameras).

## Files

| File | What it does |
|---|---|
| `model.py` | the head, its preprocessing, the loss, checkpoint save and load |
| `live_segment.py` | `predict_mask` and `segment` (what `seg_node` calls), and a runner for one live frame without ROS |
| `data.py` | TransCG (training, held-out validation, the test subset) and ClearGrasp-real (evaluation only) |
| `augment.py` | the photometric augmentations used in training |
| `train.py` | training and checkpoint selection |
| `predict.py` | masks for the TransCG test subset or ClearGrasp, with IoU, recall and precision |
| `mask_quality.py` | the mask metrics, and a scorer for any folder of `<key>.png` masks |
| `val_list.json` | the 8 TransCG training scenes held out for validation (600 frames) |

## Use it

The released head is `checkpoints/seg_vits_rgb_best.pt` (`scripts/get_checkpoints.sh`). In the
container:

```bash
# one live frame: data/live/rgb.png -> data/live/seg_masks.npy (int32 components, 0 = background)
python3 segmenter/live_segment.py --checkpoint checkpoints/seg_vits_rgb_best.pt \
    --backbone-weights checkpoints/depth_anything_v2_vits.pth --live-dir data/live
# the TransCG test subset's masks, for the benchmark's seg condition (eval/README.md)
python3 eval/make_seg_masks.py --dilate 5
```

## Train and evaluate it

TransCG is found as `eval/common.py` finds it (`$RG_TRANSCG`, `datasets/transcg/transcg`, or
`/datasets/transcg/transcg`); ClearGrasp's test-val release at `$RG_CLEARGRASP`, else
`datasets/cleargrasp/cleargrasp-dataset-test-val`.

```bash
python3 segmenter/train.py --name vits_rgb            # 6 epochs, batch 12, AdamW 3e-4, seed 1234, fp16
python3 segmenter/predict.py --checkpoint segmenter/runs/vits_rgb/best.pt                     # TransCG subset
python3 segmenter/predict.py --checkpoint segmenter/runs/vits_rgb/best.pt --dataset cleargrasp --cg-split real-test
python3 segmenter/predict.py --checkpoint segmenter/runs/vits_rgb/best.pt --dataset cleargrasp --cg-split real-val
```

The released head was trained on one RTX 3090 (24 GB) with Python 3.12 and torch 2.10.0+cu126:
30,963 TransCG training frames (the training split minus the 8 held-out scenes), 2,580 steps per
epoch, about 16 min per epoch, about 1.6 h in all. Augmentation: random resized crop (0.6 to 1.0),
flips, a brightness gain of 0.35x to 1.4x (the rig's first failed capture had a mean brightness of
15/255), HLS jitter, and noise or blur. The checkpoint is the epoch with the best held-out IoU
(0.9245, epoch 4 of 6), never chosen on the test subset. One run, one seed.

## Pre-registration

The acceptance thresholds were written on 30 Sep 2026, before the run, which started at 04:57
Singapore time. In short:

- IoU above **0.85** on the 1,177-frame TransCG test subset (in-domain);
- IoU above **0.60** on ClearGrasp-real, a different lab with different objects and cameras, never
  trained on (the cross-domain gate; below 0.45 the head does not transfer and is not deployed);
- a ReMake retrained by this project, given this mask: masked RMSE under **20 mm** (30.7 with SAM's box
  mask, 10.5 with the true mask);
- a second arm with the sensor-validity map as an extra input (`--use-valid`) is kept only if it
  gains more than 0.05 IoU on ClearGrasp;
- the ViT-B encoder is tried only if the ViT-S head fails the cross-domain gate.

## Results

Mask quality of the released head (`predict.py`, threshold 0.5):

| Set | Frames | IoU | Recall | Precision | Gate |
|---|---|---|---|---|---|
| TransCG test subset (in-domain) | 1,177 | **0.887** | 0.934 | 0.945 | > 0.85: pass |
| TransCG test subset, mask grown 5 px (the robot's setting) | 1,177 | 0.850 | 0.972 | 0.871 | |
| ClearGrasp real-test | 113 | **0.814** | 0.952 | 0.853 | > 0.60: pass |
| ClearGrasp real-val | 173 | **0.742** | 0.899 | 0.819 | > 0.60: pass |
| for comparison: SAM (v1) ViT-B prompted with a box around each region of the dataset's mask | 1,177 | 0.628 | 0.654 | 0.900 | |

The depth gate: the retrained ReMake given the head's mask scored 16.0 mm, and 13.7 mm with the mask
grown 5 px, under 20 mm: pass (the "earlier retrain" column of
[`training/results/G_maskdrop/RESULT.md`](../training/results/G_maskdrop/RESULT.md)).

The second arm with the sensor-validity channel reached 0.889 / 0.847 / 0.771: gains of 0.033 and
0.029 on ClearGrasp, under the 0.05 rule, so the RGB-only head was kept and the mask does not depend
on the depth camera. The ViT-B encoder was not needed. What these masks do to ReMake's depth error
on TransCG: [`eval/README.md`](../eval/README.md) and [`training/`](../training/). On ClearGrasp it is in the
dissertation (Section 6.6) and is not reproduced here.

## Licence

The code is MIT (the repository's `LICENSE`). The released head is CC BY-NC 4.0, because it is
trained on TransCG. Depth Anything V2 ViT-S, which it runs on, is Apache-2.0; the Base and Large
encoders are CC BY-NC 4.0, so `model.py` has no default weights for them.
