# G_maskdrop: the mask cliff becomes a slope -- PASS on no-mask, a narrow miss on SAM-box

2026-10-01. Six-epoch fine-tune of `remake_official.tar` (epoch 39 -> 45, lr 1e-4) with one
change: mask dropout on the mask the network sees (`p_mask_none 0.25 / p_mask_erode 0.15 /
p_mask_dilate 0.15`, 3-15 px; loss masks untouched). Control: `A_ft`. Selected on held-out val
masked RMSE (8.72 -> 6.50 mm; epoch 45 = the last epoch). Frozen 1,177-frame subset, native
resolution. Thresholds from the header of `config.yaml` in this folder.

| condition | A_ft (control) | **G_maskdrop** | threshold | |
|---|---|---|---|---|
| protocol (GT mask, sensor intact) | 11.8 mm / 87 % | **11.1 mm / 87 %** | <= A_ft + 1 mm | guard held |
| **no mask at all** | 43.6 mm (fail 75 %) | **17.0 mm (fail 16 %)** | < 30 mm | **PASS** |
| **SAM box mask (IoU 0.63)** | 27.0 mm | **23.1 mm** | < 22 mm | **miss by 1.1 mm** |
| sensor blanked under the mask | 26.1 mm / 32 % | 26.8 mm / 32 % | (F_blank's target) | unchanged |
| point F-score @ 5 mm | 58.0 % | 67.8 % | | up |
| normal angle vs mesh | 12.4 deg | 12.4 deg | | tie |

By camera, no mask: A_ft D435 27.2 / L515 60.2 mm; G_maskdrop 11.5 / 22.5 mm. SAM box:
17.2 / 37.0 -> 14.1 / 32.3 mm.

## Reading

**The no-mask row is the result.** ReMake with no mask went from 43.6 mm (an earlier
retrain: 55.8) to 17.0 mm -- below SGDCNet `base28`'s 18.8 mm, which the report holds up as
the deployment-grade number, and with the protocol row *better* than the control's. The
cliff the report describes (10.5 -> 55.8 the moment the mask is gone) is now a slope: 11.1 ->
17.0.

**SAM box misses its threshold by 1.1 mm** -- inside the 2 mm noise floor, so neither a pass
nor a refutation. The more telling comparison is inside the arm: no mask (17.0) beats SAM's
mask (23.1). The report recorded this for SGDCNet ("a bad mask is worse than no mask") and
noted ReMake did not show it because it had no inference to fall back on; after six epochs
of mask dropout it does. SAM's box masks miss whole objects (recall 0.65), and a
confidently-wrong "not transparent" hurts more than "unknown".

**The blanked row did not move**, as expected: this knob never removes the sensor reading.
That is F_blank's job (the depth-blanking run), and the two knobs are complementary by
construction.

**Single run, one seed;** the no-mask gain (26.6 mm) is more than ten floors wide, the SAM-box
gain (3.9 mm) about two.

## Follow-up queued

The deployment mask is the prompt-free segmenter's ([`segmenter/README.md`](../../../segmenter/README.md), IoU 0.85-0.89),
not SAM's. Scored 2026-10-01 01:13 (`results/G_maskdrop/masked_rmse_seg_d{0,5}.txt`):

| mask fed to the model | earlier retrain | G_maskdrop |
|---|---|---|
| segmenter, 0 px (IoU 0.89) | 16.0 mm | **14.6 mm** (D435 10.6 / L515 18.7, fail 9.2 %) |
| segmenter, +5 px (IoU 0.85) | 13.7 mm | **13.3 mm** (10.3 / 16.3, fail 6.5 %) |
| SGDCNet base28 on the same masks | 14.7 / 14.0 mm | |

With a deployable mask this arm scores 13.3 mm, 2.2 mm off its own GT-mask number, against
the 27 mm (SAM) and 43.6 mm (none) the control shows the robot today.

## Decisions

- `p_mask_none/erode/dilate` go into the combined from-scratch recipe with F_blank's knob.
- Deployment candidate: segmenter mask (5 px) + this knob + F_blank's knob, one 40-epoch run.
