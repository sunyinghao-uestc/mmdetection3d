# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMDetection3D is an open-source 3D object detection toolbox based on PyTorch, part of the [OpenMMLab](https://openmmlab.com/) ecosystem. It supports LiDAR-based, camera-based (monocular/BEV), and multi-modal 3D object detection, plus LiDAR-based 3D semantic segmentation. This is the v1.x branch built on MMEngine.

## Common Commands

### Training

```bash
# Single GPU
python tools/train.py <config_file>

# Multi-GPU distributed
bash tools/dist_train.sh <config_file> <num_gpus>

# With config override
python tools/train.py <config> --cfg-options model.backbone.depth=50
```

### Testing / Evaluation

```bash
# Standard evaluation
python tools/test.py <config_file> <checkpoint_path>

# Multi-GPU
bash tools/dist_test.sh <config_file> <checkpoint_path> <num_gpus>

# Single GPU test (limits to one GPU for debug)
CUDA_VISIBLE_DEVICES=0 python tools/test.py <config_file> <checkpoint_path>
```

### Running Tests

```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_models/test_detectors/test_voxelnet.py

# Single test case
pytest tests/test_models/test_detectors/test_voxelnet.py::TestVoxelNet::test_voxelnet_v2

# With coverage
pytest tests/ --cov=mmdet3d --cov-report=term-missing
```

### Linting

```bash
flake8 mmdet3d/
isort --check-only mmdet3d/
yapf -r -d mmdet3d/
```

## Architecture

### Config System

Everything is config-driven. Configs are pure Python files that use `_base_` inheritance (MMEngine's `Config.fromfile()`). Configs live in `configs/<model_name>/` with `_base_` components in `configs/_base_/` (datasets/, models/, schedules/, default_runtime.py). Configs compose: dataset → pipeline transforms → model → schedule → runtime.

### Registry Pattern

All components register themselves via `mmdet3d.registry` registries (child registries of MMEngine's root registry). Key registries: `MODELS`, `DATASETS`, `TRANSFORMS`, `METRICS`, `TASK_UTILS`. Registration uses `@MODELS.register_module()` decorator. See `mmdet3d/registry.py:1` for the full list.

### Model Architecture (`mmdet3d/models/`)

Models follow the MMDetection pattern:
- **Detectors** (`detectors/`): Top-level detection models. `Base3DDetector` extends `mmdet.models.BaseDetector`. Implements `forward(inputs, data_samples, mode)` with modes: `'loss'`, `'predict'`, `'tensor'`.
- **Segmentors** (`segmentors/`): For LiDAR semantic segmentation tasks.
- **Backbones** (`backbones/`): Feature extractors (PointNet++, SECOND, DLA, MinkUNet, etc.).
- **Necks** (`necks/`): FPN variants for fusing multi-scale features.
- **Dense Heads** (`dense_heads/`): Anchor-based and anchor-free 3D detection heads.
- **ROI Heads** (`roi_heads/`): Two-stage detector second-stage heads.
- **Voxel Encoders** (`voxel_encoders/`): Point-to-voxel encoders (PillarFeatureNet, etc.).
- **Middle Encoders** (`middle_encoders/`): Sparse conv encoders for voxel processing.
- **Decode Heads** (`decode_heads/`): Segmentation heads.
- **Losses** (`losses/`): Custom 3D loss functions.
- **Data Preprocessors** (`data_preprocessors/`): Handles input normalization/augmentation at the model boundary, before the backbone.

### Data Flow

1. **Dataset** (`mmdet3d/datasets/`): Subclasses of `Det3DDataset` (KITTI, nuScenes, Waymo, etc.). Each dataset has a `pipeline` — a sequence of transform dicts composed at config level.
2. **Transforms** (`mmdet3d/datasets/transforms/`): Pipeline stages — loading (`LoadPointsFromFile`, `LoadAnnotations3D`), augmentation (`RandomFlip3D`, `GlobalRotScaleTrans`), filtering, and formatting (`Pack3DDetInputs`). Transforms are registered in the `TRANSFORMS` registry.
3. **Data Sample** (`mmdet3d/structures/det3d_data_sample.py`): `Det3DDataSample` (extends `DetDataSample`) is the standard data container. Holds `gt_instances_3d`, `pred_instances_3d`, `gt_pts_seg`, `pred_pts_seg`, etc. as `InstanceData` or `PointData` fields.
4. **3D Bounding Boxes** (`structures/bbox_3d/`): `BaseInstance3DBoxes` and subclasses for different box encodings (LiDAR, depth, camera coordinates).

### Evaluation (`mmdet3d/evaluation/`)

Per-dataset metrics: `KittiMetric`, `NuScenesMetric`, `WaymoMetric`, `LyftMetric`, `IndoorMetric`. Functional implementations in `evaluation/functional/` for offline evaluation. Metrics use the MMEngine `BaseMetric` interface.

### Projects (`projects/`)

Community-contributed model implementations (BEVFusion, DETR3D, PETR, DSVT, TPVFormer, etc.). Each project is self-contained with its own models, configs, and sometimes custom ops.

### Tools (`tools/`)

Entry points for training (`train.py`), testing (`test.py`), and data conversion (`create_data.py`, `dataset_converters/`). Analysis utilities in `tools/analysis_tools/`, deployment in `tools/deployment/`.

## Key Dependencies

- **mmengine**: Training framework (Runner, Config, Registry, logging, hooks)
- **mmcv**: Computer vision foundation (ops like voxelization, RoI align 3D, etc.)
- **mmdet**: 2D object detection base classes and components (backbones, necks, heads, ROI extractors)
- **PyTorch 1.8+**

Models inherit from mmdet base classes (e.g., `BaseDetector`, `BaseDenseHead`) and override/extend for 3D.

## Tests

Tests mirror the package structure under `tests/`. Test data fixtures are in `tests/data/`. Tests use pytest with `parameterized` for multi-config testing. Many model tests validate that a config builds correctly and runs a forward pass successfully.

When adding a new model, register it and add a test under the corresponding `tests/test_models/test_<component>/` directory.
