"""Pure-PyTorch replacement for SBG's compiled `pointnet2._ext` CUDA ops (inference only).

Same inputs, outputs and index order as the CUDA kernels, so nothing needs compiling
per GPU. Registered as `pointnet2._ext` by stubs.py. The operators are those of SBG's
pointnet2 extension (from votenet, Copyright (c) Facebook, Inc. and its affiliates, MIT,
based on erikwijmans/Pointnet2_PyTorch); this file is a new implementation of their behaviour.
"""
import math

import torch

_CHUNK = 8_000_000   # max query x point pairs held in memory at once


def _block_size(n):
    """Threads per block the CUDA FPS kernel uses (decides its tie-breaking)."""
    return max(min(1 << int(math.log(n) / math.log(2.0)), 512), 1)


def _bit_reverse(v, bits):
    out = torch.zeros_like(v)
    for b in range(bits):
        out |= ((v >> b) & 1) << (bits - 1 - b)
    return out


def _fps(xyz, npoint):
    """Farthest point sampling, one small step per sample (slow to launch, so graphed on CUDA)."""
    B, N, _ = xyz.shape
    dev = xyz.device
    valid = (xyz * xyz).sum(-1) > 1e-3
    # Invalid points: distance +inf and score -1, so they are never picked.
    pen = torch.where(valid, 0.0, float("inf"))
    temp = torch.where(valid, 1e10, -1.0)
    # CUDA reduces per-thread maxima with a tree that prefers the lower slot on ties;
    # this key reproduces that order: bit-reversed thread id, then point index.
    bs = _block_size(N)
    k = torch.arange(N, device=dev)
    key = (_bit_reverse(k % bs, int(math.log2(bs))) * N + k).expand(B, N)
    big = torch.iinfo(torch.int64).max
    rows = torch.arange(B, device=dev)
    old = torch.zeros(B, dtype=torch.long, device=dev)
    picks = [old]
    for _ in range(1, npoint):
        dx, dy, dz = (xyz - xyz[rows, old].unsqueeze(1)).unbind(-1)
        temp = torch.minimum(dx * dx + dy * dy + dz * dz + pen, temp)
        best = temp.max(dim=1, keepdim=True).values
        old = torch.where(temp == best, key, big).argmin(dim=1)
        picks.append(old)
    return torch.stack(picks, 1).to(torch.int32)


class _GraphedFps:
    """_fps captured once as a CUDA graph for a fixed (shape, npoint), then replayed."""

    def __init__(self, shape, npoint, device):
        self.inp = torch.zeros(shape, device=device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            _fps(self.inp, npoint)                                  # warm-up
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = _fps(self.inp, npoint)

    def __call__(self, xyz):
        self.inp.copy_(xyz)
        self.graph.replay()
        return self.out.clone()


_fps_graphs = {}


def furthest_point_sampling(xyz, npoint):
    """(B, N, 3) -> (B, npoint) int32. Starts at index 0; points with |p|^2 <= 1e-3 are skipped."""
    xyz = xyz.contiguous().float()
    if npoint <= 0:
        return torch.zeros(xyz.shape[0], 0, dtype=torch.int32, device=xyz.device)
    if not xyz.is_cuda:
        return _fps(xyz, npoint)
    key = (tuple(xyz.shape), int(npoint), xyz.device)
    if key not in _fps_graphs:
        _fps_graphs[key] = _GraphedFps(xyz.shape, npoint, xyz.device)
    return _fps_graphs[key](xyz)


def gather_points(features, idx):
    """(B, C, N), (B, M) -> (B, C, M)."""
    B, C, _ = features.shape
    return features.gather(2, idx.long().unsqueeze(1).expand(B, C, idx.shape[1]))


def group_points(features, idx):
    """(B, C, N), (B, npoint, nsample) -> (B, C, npoint, nsample)."""
    B, C, _ = features.shape
    _, P, S = idx.shape
    flat = idx.long().reshape(B, 1, P * S).expand(B, C, P * S)
    return features.gather(2, flat).reshape(B, C, P, S)


def _first_hits(hit, nsample):
    """First `nsample` True columns per row, in index order, padded with the first hit (0 if none)."""
    m, n = hit.shape
    cs = hit.cumsum(-1, dtype=torch.int32)
    r, c = (hit & (cs <= nsample)).nonzero(as_tuple=True)
    out = torch.full((m, nsample), n, dtype=torch.int32, device=hit.device)
    out[r, cs[r, c].long() - 1] = c.to(torch.int32)
    head = out[:, :1]
    out = torch.where(out == n, head, out)
    return torch.where(head == n, torch.zeros_like(out), out)


def _coords(t):
    return [c.contiguous() for c in t.unbind(-1)]


def _query(new_xyz, xyz, nsample, hit_fn):
    """Run hit_fn(query_coords, point_coords, rows) in chunks and keep the first hits."""
    B, M, _ = new_xyz.shape
    N = xyz.shape[1]
    out = torch.zeros(B, M, nsample, dtype=torch.int32, device=xyz.device)
    step = max(1, _CHUNK // max(N, 1))
    for b in range(B):
        q, p = _coords(new_xyz[b].float()), _coords(xyz[b].float())
        for s in range(0, M, step):
            rows = slice(s, s + step)
            out[b, rows] = _first_hits(hit_fn([c[rows, None] for c in q], p, b, rows), nsample)
    return out


def ball_query(new_xyz, xyz, radius, nsample):
    """Indices of the first `nsample` points within `radius` of each query: (B, M, nsample) int32."""
    r2 = float(radius) * float(radius)

    def hit(q, p, b, rows):
        dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        return dx * dx + dy * dy + dz * dz < r2

    return _query(new_xyz, xyz, nsample, hit)


def _cylinder_coords(q, p, R, b, rows):
    """Query-frame coordinates of every point: axial xr and squared radial distance yz2."""
    x, y, z = p[0] - q[0], p[1] - q[1], p[2] - q[2]
    r = [R[b, rows, i, None] for i in range(9)]
    xr = r[0] * x + r[3] * y + r[6] * z
    yr = r[1] * x + r[4] * y + r[7] * z
    zr = r[2] * x + r[5] * y + r[8] * z
    return xr, yr * yr + zr * zr


# SBG makes 16 cylinder queries (4 radii x 4 heights) on the same seeds and rotations, so the
# points inside this bound are found once and each query filters that list with the same comparisons.
_BOUND_R, _BOUND_XMIN, _BOUND_XMAX = 0.1, -0.05, 0.1
_cyl_cache = {}


def _cylinder_candidates(new_xyz, xyz, R):
    """Per batch: (row, col, xr, yz2) of every point inside the _BOUND cylinder, sorted by (row, col)."""
    B, M, _ = new_xyz.shape
    N = xyz.shape[1]
    step = max(1, _CHUNK // max(N, 1))
    per_batch = []
    for b in range(B):
        q, p = _coords(new_xyz[b].float()), _coords(xyz[b].float())
        parts = []
        for s in range(0, M, step):
            rows = slice(s, s + step)
            xr, yz2 = _cylinder_coords([c[rows, None] for c in q], p, R, b, rows)
            r, c = ((yz2 < _BOUND_R ** 2) & (xr > _BOUND_XMIN) & (xr < _BOUND_XMAX)).nonzero(as_tuple=True)
            parts.append((r + s, c, xr[r, c], yz2[r, c]))
        per_batch.append([torch.cat(t) for t in zip(*parts)])
    return per_batch


def cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample):
    """Points inside a cylinder along each query's rotated x axis: (B, M, nsample) int32.

    Queries within the _BOUND cylinder reuse cached candidates while the inputs are unchanged.
    """
    r2 = float(radius) * float(radius)
    R = rot.contiguous().float().reshape(rot.shape[0], rot.shape[1], 9)
    if not (radius <= _BOUND_R and hmin >= _BOUND_XMIN and hmax <= _BOUND_XMAX):
        def hit(q, p, b, rows):
            xr, yz2 = _cylinder_coords(q, p, R, b, rows)
            return (yz2 < r2) & (xr > hmin) & (xr < hmax)
        return _query(new_xyz, xyz, nsample, hit)

    key = tuple((t.data_ptr(), t._version, tuple(t.shape)) for t in (new_xyz, xyz, rot))
    if _cyl_cache.get("key") != key:
        _cyl_cache.clear()
        # Holding the tensors keeps their memory from being reused while the key is live.
        _cyl_cache.update(key=key, refs=(new_xyz, xyz, rot), cand=_cylinder_candidates(new_xyz, xyz, R))
    B, M, _ = new_xyz.shape
    N = xyz.shape[1]
    out = torch.zeros(B, M, nsample, dtype=torch.int32, device=xyz.device)
    for b, (rows, cols, xr, yz2) in enumerate(_cyl_cache["cand"]):
        sel = (yz2 < r2) & (xr > hmin) & (xr < hmax)
        rows, cols = rows[sel], cols[sel]                     # still sorted by (row, col)
        counts = torch.bincount(rows, minlength=M)
        rank = torch.arange(len(rows), device=xyz.device) - (counts.cumsum(0) - counts)[rows]
        keep = rank < nsample
        o = torch.full((M, nsample), N, dtype=torch.int32, device=xyz.device)
        o[rows[keep], rank[keep]] = cols[keep].to(torch.int32)
        head = o[:, :1]
        o = torch.where(o == N, head, o)
        out[b] = torch.where(head == N, torch.zeros_like(o), o)
    return out


def three_nn(unknown, known):
    """3 nearest `known` points for each `unknown` point: (dist2 (B, n, 3), idx int32 (B, n, 3))."""
    unknown, known = unknown.contiguous().float(), known.contiguous().float()
    B, n, _ = unknown.shape
    dist2 = torch.zeros(B, n, 3, device=unknown.device)
    idx = torch.zeros(B, n, 3, dtype=torch.int32, device=unknown.device)
    step = max(1, _CHUNK // max(known.shape[1], 1))
    for b in range(B):
        for s in range(0, n, step):
            dx, dy, dz = (unknown[b, s:s + step].unsqueeze(1) - known[b].unsqueeze(0)).unbind(-1)
            d = dx * dx + dy * dy + dz * dz
            for c in range(3):                     # argmin keeps the lowest index on ties, like CUDA
                i = d.argmin(dim=1)
                dist2[b, s:s + step, c] = d.gather(1, i[:, None])[:, 0]
                idx[b, s:s + step, c] = i.to(torch.int32)
                d.scatter_(1, i[:, None], float("inf"))
    return dist2, idx


def three_interpolate(features, idx, weight):
    """Weighted sum of 3 neighbours: (B, C, m), (B, n, 3), (B, n, 3) -> (B, C, n)."""
    B, C, _ = features.shape
    n = idx.shape[1]
    g = features.gather(2, idx.long().reshape(B, 1, n * 3).expand(B, C, n * 3)).reshape(B, C, n, 3)
    w = weight.unsqueeze(1)
    return g[..., 0] * w[..., 0] + g[..., 1] * w[..., 1] + g[..., 2] * w[..., 2]


def _no_grad(*_a, **_k):
    raise RuntimeError("pointnet2_torch is inference-only (no backward)")


gather_points_grad = group_points_grad = three_interpolate_grad = _no_grad
