# samples

Four rig captures for the offline demo, with the results each gave. Run them in the container:

```bash
scripts/demo_offline.sh all                    # every sample under every preset it supports
scripts/demo_offline.sh three_transparent adapted    # one sample, one preset
```

How the demo works and what to check: the main README's
[quick start](../README.md#quick-start-no-camera-no-robot) and [`docs/REPRODUCE.md`](../docs/REPRODUCE.md#1-the-pipeline-on-recorded-captures).

## What a sample holds

Each sample is what the L515 capture stage wrote:

- `rgb.png`;
- `arrays.npz`, holding `depth_aligned_mm` (float32 mm, 0 = no reading), `depth_coverage`,
  `color_intrinsics`, and SAM 3's archived masks where the run had them (`sam3_masks`,
  `sam3_masks_all`), so `PIPELINE=baseline` runs without SAM 3;
- `sample.json`: where the capture came from and which presets it supports;
- `expected/<preset>.json`: the result the demo compares with.

`scripts/make_sample.py` packs an archived run into a sample, and unpacks it again (the arrays come
back bit-identical).

## The four samples

| Sample | Scene | Presets | From run |
|---|---|---|---|
| `three_transparent` | three transparent objects; SAM 3 found all three | adapted, raw, baseline | `20260928_171157_auto` |
| `bottle_plate_box` | a plastic bottle, an orange plate with a black foam handle and a cardboard box: transparent and opaque targets | adapted, raw (captured before SAM 3 was on the rig) | `20260918_162551_auto` |
| `sam3_missed_mug` | a translucent mug SAM 3's transparency mask missed; the segmenter finds it | adapted, raw, baseline | `20260930_021814_auto` |
| `rig_mug_box` | the tape-measured translucent mug (rim 110 mm) and box (90 mm); the segmenter also marks most of the box's top as transparent. Its depth is re-aligned without the capture code's rotation sign error of that day (`sample.json`, whose `source_run` names the re-aligned copy, `rig_mug_box_e2fixed`) | adapted, raw | `20260909_130411_auto` |

## What each gave

Recorded on the Orin on 2026-11-23, in `expected/<preset>.json`:

| Sample | Preset | Targets | Grasps | Executable (transparent / opaque) |
|---|---|---|---|---|
| three_transparent | adapted | 3 transparent (segmenter) | 50 | 11 (11 / 0) |
| three_transparent | baseline | 3 transparent (segmenter) | 50 | 15 (15 / 0) |
| three_transparent | raw | none: no target filter | 36 | 25 |
| bottle_plate_box | adapted | 2 transparent (segmenter) + 1 opaque (raw depth) | 50 | 22 (22 / 0) |
| bottle_plate_box | raw | none | 38 | 15 |
| sam3_missed_mug | adapted | 1 transparent (segmenter) | 11 | 4 (4 / 0) |
| sam3_missed_mug | baseline | 1 transparent (segmenter); SAM 3's mask is empty, so ReMake changes nothing | 25 | 17 (17 / 0) |
| sam3_missed_mug | raw | none | 25 | 17 |
| rig_mug_box | adapted | 2 "transparent" (segmenter): the mug, and the opaque box it marks as transparent | 32 | 2 (2 / 0) |
| rig_mug_box | raw | none | 30 | 11 |

Two things these show:

- **Executable is not correct.** "Executable" means SBG accepted the grasp, it fits the hand's
  70 mm grasp limit, it does not approach from the side the camera did not see and, where there are
  targets, it is on one. It does not mean the grasp is right. RAW's many grasps on transparent objects
  are on depth that reads them as the table, which is why it is the study's floor.
- **BASELINE depends on SAM 3 finding the object.** In `sam3_missed_mug` SAM 3's mask is empty, so
  ReMake leaves the raw depth as it is, and BASELINE's grasps are RAW's: the same 25 grasps and 17
  executable. The targets come from the segmenter in both ReMake presets, so they still mark the mug.

These expected results were recorded with ReMake's depth inside the mask only (`fill_holes`
false, the trials' rule after the first side-view attempts). Results recorded with every unread
pixel filled differ by a few grasps.
