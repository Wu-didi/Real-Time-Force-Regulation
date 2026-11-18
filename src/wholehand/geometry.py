"""Object SDFs and geometric contact estimates. No physics-contact inputs."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator

@dataclass
class BoxSDF:
    half_size: np.ndarray

    def query(self, points):
        p = np.asarray(points, dtype=float)
        h = np.asarray(self.half_size)
        q = np.abs(p) - h
        outside = np.maximum(q, 0)
        norm = np.linalg.norm(outside, axis=1)
        distance = norm + np.minimum(np.max(q, axis=1), 0)
        normals = np.zeros_like(p)
        mask = norm > 1e-12
        normals[mask] = outside[mask] * np.sign(p[mask]) / norm[mask, None]
        inner = np.flatnonzero(~mask)
        axis = np.argmax(q[inner], axis=1)
        normals[inner, axis] = np.where(p[inner, axis] >= 0, 1., -1.)
        return distance, normals

@dataclass
class SphereSDF:
    radius: float

    def query(self, points):
        p = np.asarray(points, dtype=float)
        norm = np.linalg.norm(p, axis=1)
        normals = p / np.maximum(norm[:, None], 1e-12)
        normals[norm < 1e-12] = (1,0,0)
        return norm-self.radius, normals

class MeshSDF:
    """Canonical-frame precomputed signed distance grid, outside-positive.

    trimesh's signed-distance convention is opposite; negate at construction.
    Out-of-grid points are unknown (infinite distance) and never contacts.
    """
    def __init__(self, axes, distances):
        self.axes = tuple(np.asarray(x) for x in axes)
        self.distances = np.asarray(distances)
        self.distance_fn = RegularGridInterpolator(self.axes, self.distances,
                                                   bounds_error=False, fill_value=np.inf)
        gradients = np.gradient(self.distances, *self.axes, edge_order=2)
        self.gradient_fn = RegularGridInterpolator(self.axes,
                              np.stack(gradients, axis=-1), bounds_error=False, fill_value=0.)

    @classmethod
    def from_mesh(cls, mesh, voxel_size=0.002, padding=0.03):
        import trimesh
        if not mesh.is_watertight:
            raise ValueError("SDF requires a checked watertight mesh; do not silently repair")
        bounds = mesh.bounds + np.array([[-padding]*3, [padding]*3])
        axes = [np.linspace(lo, hi, int(np.ceil((hi-lo)/voxel_size))+1)
                for lo, hi in bounds.T]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        flat = grid.reshape(-1, 3)
        distances = np.empty(len(flat))
        for i in range(0, len(flat), 4000):
            distances[i:i+4000] = -trimesh.proximity.signed_distance(mesh, flat[i:i+4000])
        return cls(axes, distances.reshape(grid.shape[:-1]))

    def query(self, points):
        p = np.asarray(points)
        d = self.distance_fn(p)
        normals = self.gradient_fn(p)
        norm = np.linalg.norm(normals, axis=1)
        good = np.isfinite(d) & (norm > 1e-10)
        normals[good] /= norm[good, None]
        return d, normals

    def save(self, path):
        np.savez_compressed(path, x=self.axes[0], y=self.axes[1], z=self.axes[2],
                            distances=self.distances)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls((a["x"],a["y"],a["z"]),a["distances"])

def farthest_point_subsample(points, count):
    """Deterministic FPS after seeded surface sampling."""
    points = np.asarray(points)
    count = min(int(count), len(points))
    if count <= 0:
        return np.empty((0,3))
    selected = np.empty(count, dtype=int)
    selected[0] = np.argmax(np.sum((points-points.mean(axis=0))**2, axis=1))
    nearest = np.full(len(points), np.inf)
    for i in range(count):
        nearest = np.minimum(nearest, np.sum((points-points[selected[i]])**2, axis=1))
        if i+1 < count:
            selected[i+1] = np.argmax(nearest)
    return points[selected]

def estimate_contacts(points_world, links, active, object_position, object_rotation,
                      sdf, jacobian_fn, ndof, threshold=0.01, per_link=5):
    """Known mesh + supplied tracked object pose + hand FK only.

    Eligibility follows signed d<threshold. Top-K uses |d| for proximity to
    the surface, an explicitly recorded implementation choice.
    jacobian_fn(link, world_point) must use only robot kinematics.
    """
    from .allocator import ContactSet
    points_world = np.asarray(points_world)
    links = np.asarray(links, dtype=int)
    active = np.asarray(active, dtype=bool)
    R = np.asarray(object_rotation)
    points_object = (points_world-np.asarray(object_position)) @ R
    d, normals_object = sdf.query(points_object)
    chosen = []
    for link in np.unique(links):
        ids = np.flatnonzero((links == link) & np.isfinite(d) & (d < threshold) &
                             (np.linalg.norm(normals_object, axis=1) > 1e-10))
        ids = ids[np.argsort(np.abs(d[ids]), kind="stable")[:per_link]]
        chosen.extend(ids.tolist())
    ids = np.asarray(chosen, dtype=int)
    J = np.zeros((len(ids), 3, ndof))
    for i, sample in enumerate(ids):
        if active[sample]:
            J[i] = jacobian_fn(int(links[sample]), points_world[sample])
    return ContactSet(points_world[ids], normals_object[ids] @ R.T,
                      J, active[ids], links[ids], d[ids])
