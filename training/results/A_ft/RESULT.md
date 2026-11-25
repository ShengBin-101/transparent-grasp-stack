# The multi-scale gradient fine-tune destroys metric scale

2026-09-26. `finetune-msgrad` branch, run in its own checkout, 6 epochs each from
the official checkpoint (epoch 39 to 45). Frozen 1,177-frame subset, native
resolution, this repository's own metric implementation.

| arm | smooth term | masked RMSE | D435 | L515 | fail | concavity |
|---|---|---|---|---|---|---|
| official (epoch 39) | shipped 1-px normal | 11.2 mm | 9.0 | 13.4 | 2.1 % | 89 % |
| **A_ft** control | shipped 1-px normal | **11.8 mm** | 9.7 | 13.9 | 2.1 % | 87 % |
| **C_ft** | multi-scale gradient | **76.9 mm** | 79.2 | 74.5 | **100 %** | 100 % |

## The control passes its gate; the arm does not fail gracefully

The pre-registered sanity gate asks A_ft to stay within about 1 mm of the official
checkpoint. It drifts 0.6 mm overall and 0.5 mm on L515, so the fine-tune recipe
is sound and C_ft is readable against it. Six extra epochs of TransCG bought
nothing on their own, which is exactly what the control exists to show.

The pre-registration also said that "masked RMSE may not improve, or may worsen
slightly. That is the pre-registered expectation, not a failure." **6.5x worse
with a 100 % failure rate is not "slightly".**

## What actually happened: it kept the shape and lost the metre

C_ft's *training* loss behaved exactly as the pre-registration predicted and never looked
wrong:

| step | total loss | smooth term |
|---|---|---|
| 1 | 0.478 | 4.72 |
| 100 | 0.102 | 0.584 |
| 2,000 | 0.044 | 0.245 |
| 20,000 | 0.032 | 0.152 |

The pre-registration said the smooth term would start at 5-10 and settle to 0.2-0.7 by step
100. It did. Meanwhile held-out validation went **10.3 mm to 60.8 mm inside the
first epoch** and never returned:

    epoch 39 (seed) 10.3 | 40: 60.8 | 41: 58.5 | 42: 54.6 | 43: 80.7 | 44: 76.8 | 45: 75.7

(Note added 23 Nov 2026: these are the trainer's per-epoch test passes, not a held-out
validation set. The epoch-39 figure is the metric stored with the released checkpoint, on 9,596
samples; epochs 40 to 45 are scored on all 23,524 TransCG test frames (`train_C_ft.log`, not in this release). So the
first step compares two sample sets. The like-for-like comparison is the table above: 11.2 mm
released, 76.9 mm C_ft, on the 1,177-frame subset.)

Two measurements name the failure:

* **Predicted median depth is 0.521 m where the truth is 0.456 m** -- a scale
  ratio of 1.148. A_ft's is 0.996. The model is uniformly 15 % too far away.
* Removing a per-frame constant offset takes masked RMSE from 73.9 to 53.9 mm,
  with a consistent 84 mm offset (10th-90th percentile 70 to 92 mm).

And concavity reads **100 %**, the best figure any model in this project has
produced, because concavity is a *ratio* of rim-to-interior differences and is
blind to both offset and scale. A model can reproduce every cavity perfectly in
relative terms while sitting 8 cm behind the object.

## Why: a scale-invariant term with no counterweight

A multi-scale gradient loss constrains the *derivative* of depth. It is
indifferent to absolute position by construction. In this config it replaces the
1-px normal term at weight 0.1, leaving L1 as the only anchor on absolute depth
-- and at step 1 the weighted smooth term is 0.472 against an L1 of roughly
0.006, an 80:1 imbalance. The optimiser takes the cheap route: satisfy the
gradient term, let the metre drift.

This is the same failure this project already documented and fixed once.
SGDCNet's design notes (section 5.2; a development document, not in this release) record the virtual-normal term taking **98.8 % of the
depth head's gradient at initialisation against L1's 0.01 %**, and the fix was
to set weights by measuring gradient share rather than by reading loss values.
That lesson did not travel to this branch.

## What this does and does not establish

* **Established:** at weight 0.1, substituted for the normal term, SPNet's
  multi-scale gradient loss is not a drop-in improvement to ReMake. It is a
  regression severe enough to make the checkpoint unusable.
* **Not established:** that the loss is worthless. The diagnosis points at a
  weighting failure, not at the term itself, and the obvious retry is to measure
  its gradient share against L1 and rebalance, or to keep the normal term
  alongside it rather than replacing it.
* **Not testable here:** the pre-registration's real keep rule needs a grasp test on the
  rig, which only the rig produces. The depth half
  is decisive enough that the grasp half is moot for this checkpoint.

## A method note

The training objective fell smoothly by a factor of 15 while the held-out metric
rose by a factor of 7. Nothing in the training log at any point looked wrong.
Any future arm on this branch should score held-out masked RMSE per epoch, not
only the training loss, because on this recipe the two moved in opposite
directions from the first epoch onwards.
