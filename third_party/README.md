# third_party

Upstream network code, unchanged. The checkpoints depend on these exact definitions, so nothing here
is edited; the project's own code is in `ros/remake_grasp/` and `segmenter/`. Licences and
attributions: [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

| Folder | Upstream | Commit | Included as | What the project uses |
|---|---|---|---|---|
| `remake` | https://github.com/ChengYaofeng/ReMake | `4f56814` | git submodule | the ReMake model, the Depth Anything V2 encoder it wraps, `process_data` and what it imports, and the TransCG reader (`datasets/transcg.py`) for `eval/` |
| `sbg` | https://github.com/mahaoxiang822/Scale-Balanced-Grasp | `995d6f8` | git submodule | the GraspNet-MSCQ model, the collision detector and the Python side of pointnet2 |
| `sam3` | https://github.com/facebookresearch/sam3 | `8f0b7f4` | git submodule | the `sam3` package |
| `peggnet` | https://github.com/HZWang96/PEGG-Net | `be3ddfd` | copied: `models/peggnet.py` and `LICENSE`, byte-identical | the PEGG-Net model, for `DETECTOR=peggnet` |

After cloning, fetch the submodules (`./build.sh` does this when they are missing):

```bash
git submodule update --init
```

## What the project replaces at run time

No upstream file is edited. Two pieces of upstream behaviour are replaced when the project loads
the code, both in project code:

1. **SBG's choice of approach direction.** `ros/remake_grasp/remake_grasp/sbg_core.py`
   (`_cone_forward`) wraps SBG's `ApproachNet.forward` (`models/modules.py`). It runs upstream's
   forward unchanged and, when the approach cone is on (the `picking` profile, 30° about "down"),
   re-chooses each seed's view as the best-scoring one inside the cone, using upstream's
   `generate_grasp_views` and `batch_viewpoint_params_to_matrix`. Angle, depth, width and score
   are then predicted by upstream as before. With the cone off the outputs are upstream's, which
   `tests/test_sbg_cone.py` checks.
2. **SBG's compiled and optional dependencies.** `ros/remake_grasp/remake_grasp/stubs.py` registers
   stand-ins so that nothing needs compiling per GPU: `pointnet2_torch.py`, a pure-PyTorch
   version of the pointnet2 CUDA operators with the same outputs and index order
   (checked against the compiled extension on archived clouds during development); a `torch.cdist`
   version of the `knn` extension, which inference does not call; a NumPy `voxel_down_sample` in
   place of Open3D's, the only Open3D call SBG makes at inference (it gave Open3D 0.20's output
   exactly on 20 random clouds); and placeholders for modules that only ReMake's training and
   plotting code import (OpenEXR, Imath, pandas, pytorch_grad_cam).
