# L_combo_both: both gains survive the combination -- PASS on every registered row, two at the edge

2026-10-02. Six-epoch fine-tune of `remake_official.tar` (epoch 39 -> 45, lr 1e-4) on both
TransCG cameras with the ladder's two passing knobs and its bug fix in one recipe:
`p_blank_in_mask 0.5`, `p_mask_none/erode/dilate 0.25/0.15/0.15`, `rel_rgb_fix`. Controls:
`A_ft` (the old stack's recipe), `F_blank`, `G_maskdrop` (each knob alone). Selected on held-out
val masked RMSE (8.43 -> 7.44 mm, epoch 43). Frozen 1,177-frame subset, native resolution.
Thresholds from the header of `config.yaml` in this folder, written before the run.

| condition (overall, n=1177) | A_ft | F_blank | G_maskdrop | **L_combo_both** | threshold | |
|---|---|---|---|---|---|---|
| protocol (GT mask) | 11.8 mm / 87 % | 11.9 / 89 % | 11.1 / 87 % | **12.2 / 87 %** | <= 12.8 | guard held |
| sensor blanked | 26.1 / 32 % | **15.6 / 92 %** | 26.8 / 32 % | **16.2 / 89 %** | <= 17.6 | **PASS** |
| no mask | 43.6 | 45.9 | **17.0** | **18.8** | <= 19.0 | **PASS** (0.2 to spare) |
| SAM box (IoU 0.63) | 27.0 | 28.8 | 23.1 | **24.1** | <= 24.1 | **PASS** (at the line) |

L515 rows (the rig's camera, n=585): protocol 14.8 mm, blanked 17.4 (A_ft 29.8), no-mask 25.0
(A_ft 60.2), SAM box 33.5 (A_ft 37.0). D435: 9.7 / 15.1 / 12.8 / 14.8.

## Reading

**The knobs do not interfere.** The registration's null was "either gain lost: blanked > 20 or
no-mask > 25". Neither happened: the blanking gain (26.1 -> 16.2) and the mask-dropout gain
(43.6 -> 18.8) are both present in one checkpoint, each within 0.6-1.8 mm of the single-knob
arm that produced it -- inside the 2 mm floor. One model now tolerates a wrong in-mask
reading, a missing mask and a bad mask at once, which no previous ReMake did.

**Two rows sit on their thresholds.** No-mask at 18.8 against 19.0 and SAM box at 24.1 against
24.1 are passes by the letter and ties by the floor; read them as "the combination costs about
a millimetre on the mask-robustness rows", which is the price of also carrying the blanking
behaviour. The protocol row paid 0.4 mm against A_ft and 1.1 against G_maskdrop, also inside
noise. The blanked concavity, 89 %, is three points under F_blank's 92 %: a tie.

**It is the rig candidate.** Against the control A_ft (the official checkpoint fine-tuned on the
same schedule with nothing changed), on the L515: blanked 29.8 -> 17.4 mm, no mask 60.2 -> 25.0, SAM box
37.0 -> 33.5, protocol 13.9 -> 14.8. The deployment pairing is this checkpoint with the
prompt-free segmenter's mask dilated 5 px. Scored 2026-10-02 11:19
(`results/L_combo_both/masked_rmse_seg_d{0,5}.txt`):

| mask fed to the model | earlier retrain | G_maskdrop | **L_combo_both** |
|---|---|---|---|
| segmenter, 0 px (IoU 0.89) | 16.0 mm | 14.6 | **15.4** (D435 11.2 / L515 19.6, fail 10.8 %) |
| segmenter, +5 px (IoU 0.85) | 13.7 mm | 13.3 | **14.2** (10.8 / 17.6, fail 8.3 %) |

Within a millimetre of G_maskdrop on the deployment mask, 2.0 mm off its own GT-mask row,
with the blanking behaviour G_maskdrop lacks. On the L515, 17.6 mm with a mask a robot can
make, against 37.0 for A_ft on SAM's box.

Single run, one seed; the from-scratch 40-epoch run of the same recipe is the arbiter for
the edge rows.

## Decisions

- `checkpoint43.tar` replaces G_maskdrop's as the new-stack weights for the robot trials
  (the development repository's transfer notes updated; not in this release).
- The combined recipe for the from-scratch run is confirmed: these three knobs plus
  `align_rel_aug` and the hygiene items.
