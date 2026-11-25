"""Stand-ins for modules the upstream code imports at load time but inference never needs.

- pointnet2._ext  -> pointnet2_torch (pure PyTorch, no per-GPU compile)
- knn_pytorch     -> torch.cdist version (only used in SBG training)
- open3d          -> numpy voxel_down_sample (the only open3d call SBG makes at inference)
- OpenEXR, Imath, pandas, pytorch_grad_cam -> placeholders (ReMake training/plotting only)

A real module that is installed always wins, except pointnet2._ext (see install()).
"""
import importlib.machinery
import importlib.util
import sys
import types

import numpy as np


class _Placeholder(types.ModuleType):
    """Any attribute is a dummy that fails loudly only if actually used."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def unavailable(*_a, **_k):
            raise RuntimeError(f"{self.__name__}.{name} is not installed in this image")
        return unavailable


def _add(name, module):
    """Put module in sys.modules under name and attach it to its parent package."""
    if getattr(module, "__spec__", None) is None:
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)   # some code calls find_spec()
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)
    return module


def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


# ---------------------------------------------------------------- open3d

def _voxel_down_sample(points, voxel):
    """Open3D's PointCloud.voxel_down_sample: mean of the points in each voxel (float64)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    lo = pts.min(axis=0) - voxel * 0.5
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    _, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inv.reshape(-1), pts)
    return sums / counts[:, None]


class _PointCloud:
    def __init__(self):
        self.points = np.zeros((0, 3))

    def voxel_down_sample(self, voxel_size):
        out = _PointCloud()
        out.points = _voxel_down_sample(self.points, voxel_size)
        return out


def _open3d():
    o3d = _Placeholder("open3d")
    o3d.geometry = _Placeholder("open3d.geometry")
    o3d.geometry.PointCloud = _PointCloud
    o3d.utility = _Placeholder("open3d.utility")
    o3d.utility.Vector3dVector = lambda a: np.asarray(a, dtype=np.float64)
    return o3d


# ---------------------------------------------------------------- knn

class _Knn:
    @staticmethod
    def knn(ref, query, inds):
        """Fill inds (B, k, M) with the k nearest ref columns of each query column."""
        import torch
        k = inds.shape[1]
        d = torch.cdist(query.transpose(1, 2).float(), ref.transpose(1, 2).float())
        inds.copy_(d.topk(k, dim=2, largest=False).indices.transpose(1, 2).to(inds.dtype))
        return 1


# ---------------------------------------------------------------- install

def install(pointnet2_dir=None, compiled_pointnet2=False):
    """Register the stand-ins for missing modules. `pointnet2_dir` is SBG's pointnet2 folder (package path).

    pointnet2._ext is always the PyTorch version unless compiled_pointnet2 or it is already imported.
    """
    if _missing("open3d"):
        _add("open3d", _open3d())
    for name in ("OpenEXR", "Imath", "pandas"):
        if _missing(name):
            _add(name, _Placeholder(name))
    if _missing("pytorch_grad_cam"):
        for name in ("pytorch_grad_cam", "pytorch_grad_cam.utils", "pytorch_grad_cam.utils.image"):
            _add(name, _Placeholder(name))
    if _missing("knn_pytorch"):
        _add("knn_pytorch", types.ModuleType("knn_pytorch")).knn_pytorch = _Knn()

    if compiled_pointnet2 or "pointnet2._ext" in sys.modules:
        return
    from . import pointnet2_torch
    pkg = types.ModuleType("pointnet2")
    pkg.__path__ = [str(pointnet2_dir)] if pointnet2_dir else []
    _add("pointnet2", pkg)
    _add("pointnet2._ext", pointnet2_torch)
