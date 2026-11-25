"""The arm against the scene: FR3 sphere models versus the occupied voxels the Orin sends.

    scene = Scene(points_base, voxel_m)
    clearance, part = scene.clearance(arm_spheres(q, ee_to_camera))

SBG's collision check models only the gripper, so this checks the wrist, forearm, camera and (on labelled
scenes) fingers along the postures planning.py walks. The scene is base-frame voxel centres from
pick_candidates; the table is handled by planning as a plane, and the grasped object's voxels near the
grasp give way (Scene.without_box, Scene.for_instance). No ROS imports.
"""

import math

import numpy as np

from fr3_pose_commander import kinematics

# The URDF's self-collision capsules (fr3_linkN_sc): (name, frame, radius, end A, end B) in the frame named.
# Link frames are kinematics.forward()'s DH frames, equal to the URDF's (test_scene.py). "7sc" is
# fr3_link7 . Rz(+45 deg) (fr3_link7_sc_joint); "hand" is fr3_hand = flange . Rz(-45 deg).
CAPSULES = (
    ("link1", 1, 0.09, (0.0, 0.0, -0.05), (0.0, 0.0, -0.333)),
    ("link2", 2, 0.09, (0.0, 0.0, 0.06), (0.0, 0.0, -0.06)),
    ("link3", 3, 0.09, (0.0, 0.0, -0.07), (0.0, 0.0, -0.22)),
    ("link4", 4, 0.09, (0.0, 0.0, 0.06), (0.0, 0.0, -0.06)),
    ("link5", 5, 0.09, (0.0, 0.0, -0.21), (0.0, 0.0, -0.31)),
    ("link5b", 5, 0.055, (0.0, 0.08, -0.06), (0.0, 0.08, -0.2)),
    ("link6", 6, 0.08, (0.0, 0.0, 0.01), (0.0, 0.0, -0.07)),
    ("link7", "7sc", 0.07, (0.0, 0.0, 0.08), (0.0, 0.0, -0.06)),
    ("link7b", "7sc", 0.06, (0.065, 0.0, 0.082), (0.055, 0.0, 0.082)),
    ("hand", "hand", 0.07, (0.0, 0.05, 0.04), (0.0, -0.05, 0.04)),
)
# Franka Hand fingers in the TCP frame (y along the jaws, z the approach, tips at z = 0): 7 mm spheres
# on a 5 mm grid cover x +-12 mm, y from 2 mm inside each inner face to 12 mm outside, z -57..+2 mm.
FINGER_SPHERE_R = 0.007
_FX = (-0.005, 0.005)
_FZ = tuple(-0.005 * k for k in range(1, 11))


def finger_local(opening):
    """Finger sphere centres (Nx3, TCP frame) with the inner faces `opening` apart."""
    h = 0.5 * float(opening)
    pts = [(x, s * (h + 0.005), z) for s in (-1.0, 1.0) for x in _FX for z in _FZ]
    return np.array(pts, float)


# L515 (~61 mm across, ~26 mm deep, all behind the lens): a centre sphere and a ring of six, 13 mm each,
# 13 mm behind the optical frame.
CAMERA_SPHERE_R = 0.013
CAMERA_LOCAL = np.array([[0.0, 0.0, -0.013]] +
                        [[0.0175 * math.cos(a), 0.0175 * math.sin(a), -0.013]
                         for a in np.arange(6) * math.pi / 3])

# Mesh parts are surface samples MESH_SPACING apart, each a MESH_RADIUS sphere. The face grid leaves no
# surface point beyond spacing/sqrt(2) from a sample and the spacing/4 dedup moves one by at most
# (spacing/8)*sqrt(3), so the sum covers every surface point (over-reach <= ~14 mm).
MESH_SPACING = 0.015
MESH_RADIUS = MESH_SPACING / math.sqrt(2.0) + MESH_SPACING / 8.0 * math.sqrt(3.0)
# (name, frame, mesh file): the parts modelled from their collision meshes.
MESH_PARTS = (("link6", 6, "link6.stl"), ("link7", 7, "link7.stl"),
              ("hand", "hand", "hand.stl"))


def _rz(angle):
    """4x4 rotation about z."""
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4)
    m[:2, :2] = ((c, -s), (s, c))
    return m


_RZ_P45 = _rz(math.pi / 4)
_RZ_M45 = _rz(-math.pi / 4)


def _chain(a, b, radius):
    """Sphere centres along segment a-b, spaced at most radius/3 (a capsule as a sphere chain)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(math.ceil(np.linalg.norm(b - a) / (radius / 3.0))) + 1)
    return np.linspace(a, b, n)


# Sphere centres per capsule, in its own frame, precomputed once.
_LOCAL = [(name, frame, radius, _chain(a, b, radius)) for name, frame, radius, a, b in CAPSULES]


def _read_stl(path):
    """Triangles (N x 3 x 3) from an ASCII or binary STL."""
    import struct
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [[float(x) for x in line.split()[1:4]]
             for line in data.decode(errors="ignore").splitlines()
             if line.strip().startswith("vertex")]
        return np.asarray(v, float).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
        [("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].astype(float).reshape(-1, 3, 3)


def surface_samples(triangles, spacing):
    """Points over a triangle mesh, no surface point further than
    spacing/sqrt(2) from one of them (a grid of that pitch on every face)."""
    out = [triangles.reshape(-1, 3)]
    for a, b, c in triangles:
        n = int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a),
                              np.linalg.norm(c - b)) / spacing))
        if n < 2:
            continue
        for i in range(n + 1):
            for j in range(n + 1 - i):
                u, v = i / float(n), j / float(n)
                out.append((a + u * (b - a) + v * (c - a))[None, :])
    pts = np.vstack(out)
    return np.unique(np.round(pts / (spacing / 4)), axis=0) * (spacing / 4)


def _mesh_dir():
    """franka_description's collision meshes, in the same catkin workspace as this package."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # <ws>/src/fr3_pose_commander/src/fr3_pose_commander -> <ws>/src
    return os.path.join(here, "..", "..", "..", "franka_ros", "franka_description",
                        "meshes", "collision")


_MESH = None       # {name: local points}, or {} if the meshes are unavailable
MESH_ERROR = None


def _mesh_parts():
    """{name: local surface samples} for MESH_PARTS, loaded once.

    {} (with MESH_ERROR set) if a mesh cannot be read; those parts then fall back to their capsules,
    over-cautious but never optimistic.
    """
    global _MESH, MESH_ERROR
    if _MESH is None:
        import os
        _MESH = {}
        try:
            for name, _, fname in MESH_PARTS:
                _MESH[name] = surface_samples(_read_stl(os.path.join(_mesh_dir(), fname)),
                                              MESH_SPACING)
        except (IOError, OSError, ValueError) as exc:
            _MESH, MESH_ERROR = {}, "%s: %s" % (type(exc).__name__, exc)
    return _MESH


def arm_spheres(joints, ee_to_camera=None):
    """[(part name, centres Nx3 in the base frame, radius)] for posture `joints`.

    Links 6, 7 and the hand, which reach the objects, come from their collision meshes; links 1-5 from
    the URDF's fat capsules, harmless this far from the objects. The hand and link-7 capsules over-reach
    their real shape by ~45 mm toward the objects and refused a grasp the arm executed cleanly. The camera
    (with `ee_to_camera`) is a disc behind its lens. The fingers are separate (finger_spheres).
    """
    _, _, frames, flange, ee = kinematics.forward(joints)
    lookup = {i: frames[i - 1] for i in range(1, 8)}
    lookup["7sc"] = frames[6].dot(_RZ_P45)
    lookup["hand"] = flange.dot(_RZ_M45)
    meshes = _mesh_parts()
    meshed = {name for name, _, _ in MESH_PARTS} if meshes else set()
    out = []
    for name, frame, radius, local in _LOCAL:
        if name.rstrip("b") in meshed:
            continue                      # modelled from its mesh below
        t = lookup[frame]
        out.append((name, local.dot(t[:3, :3].T) + t[:3, 3], radius))
    radius = MESH_RADIUS
    for name, frame, _ in MESH_PARTS:
        if name in meshes:
            t = lookup[frame]
            out.append((name, meshes[name].dot(t[:3, :3].T) + t[:3, 3], radius))
    if ee_to_camera is not None:
        cam = np.eye(4)
        cam[:3, 3] = np.asarray(ee_to_camera[:3], float)
        x, y, z, w = [float(v) for v in ee_to_camera[3:7]] if len(ee_to_camera) >= 7 else (0, 0, 0, 1)
        cam[:3, :3] = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                       (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                       (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
        t = ee.dot(cam)
        out.append(("camera", CAMERA_LOCAL.dot(t[:3, :3].T) + t[:3, 3], CAMERA_SPHERE_R))
    return out


def finger_spheres(joints, opening):
    """[("finger", centres Nx3 in the base frame, radius)]: both fingers at posture `joints`, `opening` apart.

    The planner checks them only on labelled scenes, against every voxel except the grasped object, which
    is meant to be between them; they cover the ~20 mm of fingertip past the hand mesh.
    """
    ee = kinematics.forward(joints)[4]
    return [("finger", finger_local(opening).dot(ee[:3, :3].T) + ee[:3, 3], FINGER_SPHERE_R)]


class Scene(object):
    """Occupied voxel centres (Nx3, base frame) and their edge length, queried through a KD-tree.

    `target` (a Scene or None) holds the picked object's voxels apart, checked with
    PlanConfig.target_margin; `instance` is one SAM 3 label per point (0 = none) on a scene sent
    labelled, before for_instance splits it per candidate. Without scipy, a chunked numpy search is used.
    """

    def __init__(self, points, voxel_m, target=None, instance=None):
        self.points = np.asarray(points, float).reshape(-1, 3)
        self.voxel_m = float(voxel_m)
        self.target = target
        self.instance = None if instance is None else np.asarray(instance, int).reshape(-1)
        if self.instance is not None and len(self.instance) != len(self.points):
            raise ValueError("%d labels for %d points" % (len(self.instance), len(self.points)))
        # Set by for_instance: the keep-out box removes only the target, and the fingers are checked.
        self.box_target_only = False
        # The most a centre distance overstates the distance to a voxel's surface.
        self.half_diagonal = self.voxel_m * math.sqrt(3.0) / 2.0
        self._tree = None
        if len(self.points):
            try:
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self.points)
            except ImportError:
                self._tree = None

    def __len__(self):
        return len(self.points)

    @classmethod
    def labelled(cls, points, voxel_m, instance):
        """A scene whose points each carry a SAM 3 object label (0 = none)."""
        return cls(points, voxel_m, instance=instance)

    def for_instance(self, label):
        """This labelled scene as one candidate sees it: object `label` is the target, everything else
        (other objects, unlabelled voxels) stays an obstacle. `label` None or 0: no target at all.

        box_target_only is set, so the keep-out box at the grasp removes only the target and a neighbour
        inside it still counts (an unlabelled box once removed a mug handle a finger then hit)."""
        if self.instance is None:
            raise ValueError("for_instance needs a labelled scene")
        mine = (self.instance == int(label)) if label else np.zeros(len(self.points), bool)
        target = Scene(self.points[mine], self.voxel_m) if mine.any() else None
        out = Scene(self.points[~mine], self.voxel_m, target=target)
        out.box_target_only = True
        for key in ("table_removed", "table_seen"):
            if hasattr(self, key):
                setattr(out, key, getattr(self, key))
        return out

    @classmethod
    def split(cls, points, voxel_m, is_target):
        """A scene of the voxels NOT flagged, with the flagged ones as its .target."""
        pts = np.asarray(points, float).reshape(-1, 3)
        flags = np.asarray(is_target, bool).reshape(-1)
        if len(flags) != len(pts) or not flags.any():
            return cls(pts, voxel_m)
        return cls(pts[~flags], voxel_m, target=cls(pts[flags], voxel_m))

    def nearest(self, centres):
        """Distance from each centre to the nearest voxel centre (inf if empty)."""
        centres = np.asarray(centres, float).reshape(-1, 3)
        if not len(self.points):
            return np.full(len(centres), np.inf)
        if self._tree is not None:
            return self._tree.query(centres)[0]
        best = np.full(len(centres), np.inf)
        for i in range(0, len(self.points), 4096):
            chunk = self.points[i:i + 4096]
            d = np.sqrt(((centres[:, None, :] - chunk[None, :, :]) ** 2).sum(-1)).min(1)
            best = np.minimum(best, d)
        return best

    # m: a part whose bounding sphere clears the scene by this much reports that bound (see clearance).
    coarse_cutoff = 0.05

    def clearance(self, spheres):
        """(a lower bound on the smallest surface clearance in m, its part).

        Sphere radius and the voxel half-diagonal are both subtracted, so the true gap is never smaller.
        Coarse first: a mesh part is ~1,000-1,600 spheres, so each part's bounding sphere is tested and,
        when that clears by coarse_cutoff, its bound (still a lower bound) stands in. Only parts near the
        scene get the per-sphere query, so the value is exact where it can decide anything.
        """
        worst, part = np.inf, None
        for name, centres, radius in spheres:
            centroid = centres.mean(axis=0)
            reach = float(np.sqrt(((centres - centroid) ** 2).sum(1)).max()) + radius
            bound = float(self.nearest(centroid[None, :])[0]) - reach - self.half_diagonal
            if bound > self.coarse_cutoff:
                d = bound
            else:
                d = float(self.nearest(centres).min()) - radius - self.half_diagonal
            if d < worst:
                worst, part = d, name
        return worst, part

    def without_box(self, pose_matrix, half_extents, z_range):
        """This scene minus the voxels inside a box in the frame `pose_matrix`.

        The box is |x| <= half_extents[0], |y| <= half_extents[1], z_range[0] <= z <= z_range[1]: for a
        grasp pose, the gripper's own volume (y along the jaws, z the approach). The target always loses
        its voxels in the box; the obstacles only when not box_target_only.
        """
        target = (None if self.target is None
                  else self.target.without_box(pose_matrix, half_extents, z_range))
        if self.box_target_only or not len(self.points):
            # labelled: only the grasped object gives way; neighbours stay obstacles
            out = Scene(self.points, self.voxel_m, target=target)
            out.box_target_only = self.box_target_only
            return out
        local = (self.points - pose_matrix[:3, 3]).dot(pose_matrix[:3, :3])
        inside = ((np.abs(local[:, 0]) <= half_extents[0])
                  & (np.abs(local[:, 1]) <= half_extents[1])
                  & (local[:, 2] >= z_range[0]) & (local[:, 2] <= z_range[1]))
        return Scene(self.points[~inside], self.voxel_m, target=target)
