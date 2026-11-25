# Third-party notices

The project's own code is under the MIT licence in [`LICENSE`](LICENSE). Everything listed here
keeps its own licence. Git submodules and downloaded files are not redistributed by this
repository: they come from their authors, under their authors' terms, which you accept by fetching
them.

## Code

| Component | Version | How it is included | Licence |
|---|---|---|---|
| [ReMake](https://github.com/ChengYaofeng/ReMake) | `4f56814` | submodule `third_party/remake` | MIT, Copyright (c) 2025 ChengYaofeng; contains code under other licences (below) |
| [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) (SBG) | `995d6f8` | submodule `third_party/sbg` | MIT, Copyright (c) 2022 Haoxiang Ma; contains code from graspnet-baseline (below) |
| [SAM 3](https://github.com/facebookresearch/sam3) | `8f0b7f4` | submodule `third_party/sam3` | SAM License (Meta), in the submodule's `LICENSE` |
| [PEGG-Net](https://github.com/HZWang96/PEGG-Net) | `be3ddfd` | copied: `third_party/peggnet/models/peggnet.py` | BSD-3-Clause, Copyright (c) 2022 Haozhe Wang, in `third_party/peggnet/LICENSE` |
| [franka_ros](https://github.com/frankaemika/franka_ros) | tag `0.10.1` (`30e598a`) | fetched by `nuc/setup_franka.sh`, patched (below); one generated file is included, `nuc/src/fr3_pose_commander/test/data/fr3_hand.urdf` (xacro output of `franka_description`, unmodified apart from its header comment) | Apache-2.0, Copyright 2017 Franka Emika GmbH |
| [libfranka](https://github.com/frankaemika/libfranka) | tag `0.10.0` (`4f9e3cc`) | fetched and built by `nuc/setup_franka.sh` | Apache-2.0, Copyright 2017 Franka Emika GmbH |
| [librealsense](https://github.com/IntelRealSense/librealsense) | `v2.53.1` | built in `docker/Dockerfile`; its udev rule is copied as `docker/udev/99-realsense-libusb.rules` | Apache-2.0, Intel Corporation |
| [realsense-ros](https://github.com/IntelRealSense/realsense-ros) | `4.51.1` | built in `docker/Dockerfile` | Apache-2.0, Intel Corporation |
| [ros1_bridge](https://github.com/ros2/ros1_bridge) | `611755f` | built in `docker/Dockerfile` | Apache-2.0 |
| ROS 2 Humble, Ubuntu's ROS 1 packages (`ros-core-dev`), Python packages | as pinned in `docker/` | base image, apt and pip | their own licences |
| [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) | `master` of Nov 2026 | adapted: the design and code of `page/` | CC BY-SA 4.0, Eliahu Horwitz, itself adapted from the [Nerfies](https://nerfies.github.io) page |
| [three.js](https://github.com/mrdoob/three.js) | r128 | inlined in `page/static/viewer/point_clouds.html` | MIT, Copyright 2010-2021 three.js authors |

### Code inside the submodules

These files are in the upstream repositories, which this project uses through submodules and does
not copy. They are listed because their terms differ from the repositories' top-level licences.

- **LeReS** (ReMake, `relat_depth_models/leres/lib/`, 9 files): from
  [AdelaiDepth](https://github.com/aim-uofa/AdelaiDepth) under the Adobe Research License Terms
  (non-commercial research only, no redistribution). This project does not run LeReS, but ReMake's
  `relat_depth_models/__init__.py` imports it when the model loads.
- **Swin Transformer** (ReMake, `models/swin_transformer.py`): closely matches
  `pytorch_classification/swin_transformer/model.py` in
  [WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing),
  which is GPL-3.0. ReMake's model (`models/remake.py`) builds its image and relative-depth
  encoders from it, so every ReMake run in this project executes it.
- **Depth Anything V2 and DINOv2** (ReMake, `relat_depth_models/depth_anything_v2/`, 12 files):
  Apache-2.0 ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
  [DINOv2](https://github.com/facebookresearch/dinov2)). The segmenter (`segmenter/`) imports this
  encoder from the ReMake submodule.
- **ResNet** (ReMake, `models/resnet.py`): from torchvision, BSD-3-Clause.
- **graspnet-baseline** (SBG): `utils/collision_detector.py`, `utils/data_utils.py`,
  `knn/knn_modules.py` and `pointnet2/pytorch_utils.py` match
  [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) (`280c215`), and
  `models/{backbone,graspnet,modules}.py`, `utils/{label_generation,loss_utils}.py` and
  `pointnet2/{pointnet2_modules,pointnet2_utils}.py` derive from it. graspnet-baseline's licence
  is the GraspNet-Baseline Software License Agreement (academic or non-profit, non-commercial
  research use only). Fetching SBG means accepting it for those files.
- **PointNet++ operators** (SBG, `pointnet2/`): Copyright (c) Facebook, Inc. and its affiliates,
  MIT, from [votenet](https://github.com/facebookresearch/votenet), based on
  [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
- **CLIP tokenizer and vocabulary** (SAM 3, `sam3/model/tokenizer_ve.py`,
  `sam3/assets/bpe_simple_vocab_16e6.txt.gz`): adapted by Meta from open_clip and OpenAI CLIP,
  distributed as part of the SAM Materials.

### Project code that follows third-party code

None of these copies upstream source; they are listed for credit.

- `ros/remake_grasp/remake_grasp/sbg_core.py` calls SBG's `ApproachNet.forward` and re-chooses
  each seed's view inside an approach cone (see `third_party/README.md`). Its NMS follows the rule
  of [graspnetAPI](https://github.com/graspnet/graspnetAPI) and is written in NumPy.
- `ros/remake_grasp/remake_grasp/pointnet2_torch.py` is a pure-PyTorch version of the PointNet++
  CUDA operators above, written to give the same outputs and index order.
- `ros/remake_grasp/remake_grasp/peggnet_core.py` re-implements the pre- and post-processing that
  PEGG-Net and [GG-CNN](https://github.com/dougsm/ggcnn) use (depth inpainting, Gaussian filtering
  of the output maps).

### The franka_ros patch

`nuc/patches/franka_ros-0.10.1-stiffness.patch` modifies one file of franka_ros 0.10.1,
`franka_example_controllers/cfg/compliance_param.cfg`: the upper bound of
`translational_stiffness` goes from 400 to 1000 N/m and that of `rotational_stiffness` from 30 to
100 N m/rad, so that the values `nuc/src/fr3_pose_commander/launch/controller.launch` requests are
applied. The patched file carries a notice of the change (Apache-2.0, section 4(b)).

## Model weights

None of these is in the repository. `scripts/get_checkpoints.sh` downloads them.

| Weights | Source | Licence |
|---|---|---|
| ReMake, as released by its authors (`remake_official.tar`) | the ReMake README's download link | none stated by the authors |
| SBG (`scale_balanced_full_model.tar`) | the SBG README's download link | none stated by the authors |
| Depth Anything V2 ViT-S (`depth_anything_v2_vits.pth`) | Hugging Face, `depth-anything/Depth-Anything-V2-Small` | Apache-2.0 |
| PEGG-Net, Jacquard depth (`peggnet_jacquard_d_480.pt`) | the PEGG-Net repository at `be3ddfd` | BSD-3-Clause (repository licence) |
| SAM 3 (`sam3.pt`) | Hugging Face, `facebook/sam3` (gated: accept Meta's terms first) | SAM License |
| This project's weights: `L_combo_both`, `A_ft`, `G_maskdrop` (ReMake fine-tuned from the authors' weights) and the segmenter head | this repository's v1.0 release | CC BY-NC 4.0, because they are trained on TransCG |

The SAM-box masks in the v1.0 release (`sam_masks_box.zip`) were made with SAM (v1) ViT-B
(`sam_vit_b_01ec64.pth`, [segment-anything](https://github.com/facebookresearch/segment-anything),
Apache-2.0) prompted with a box around each connected region of TransCG's ground-truth mask. They are derived from TransCG and released
under CC BY-NC 4.0.

The SAM 3 masks in the v1.0 release (`sam3_masks_text.zip`) were made with SAM 3 (`sam3.pt`, SAM License)
prompted with the text "transparent object or bottle" on TransCG's test frames. They are derived from TransCG
and released under CC BY-NC 4.0.

## Datasets

Neither dataset is in the repository.

- **TransCG** (Fang et al., 2022, [graspnet.net/transcg](https://graspnet.net/transcg)):
  CC BY-NC 4.0. The project's ReMake weights and the segmenter are trained on it, and the
  benchmark in `eval/` runs on its test split.
- **ClearGrasp** (Sajjan et al., 2020, [sites.google.com/view/cleargrasp](https://sites.google.com/view/cleargrasp)):
  used only for evaluation. The authors do not state a licence for the dataset.

## Acknowledgement of SAM

This project uses SAM 3 from Meta (Carion et al., ICLR 2026, "SAM 3: Segment Anything with Concepts")
for text-prompted masks, and SAM from Meta (Kirillov et al., ICCV 2023, "Segment Anything") to make
the SAM-box masks.
