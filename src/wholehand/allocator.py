"""Whole-hand quasi-static force QP. No simulator contact/force measurements."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import osqp

@dataclass(frozen=True)
class ForceSettings:
    friction: float = 0.7
    minimum_normal_force: float = 10.0
    torque_limit: float = 0.2
    nullspace_margin: float = 0.5
    rank_tolerance: float = 1e-6

@dataclass
class ContactSet:
    points: np.ndarray
    outward_normals: np.ndarray
    jacobians: np.ndarray  # WORLD translational Jacobians; robot-only columns
    active: np.ndarray
    links: np.ndarray
    distances: np.ndarray

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.outward_normals = np.asarray(self.outward_normals, dtype=float)
        self.jacobians = np.asarray(self.jacobians, dtype=float)
        self.active = np.asarray(self.active, dtype=bool)
        self.links = np.asarray(self.links, dtype=int)
        self.distances = np.asarray(self.distances, dtype=float)
        n = len(self.points)
        if self.points.shape != (n, 3) or self.outward_normals.shape != (n, 3):
            raise ValueError("contact position/normal shape")
        if self.jacobians.ndim != 3 or self.jacobians.shape[:2] != (n, 3):
            raise ValueError("contact Jacobian shape")
        if any(a.shape != (n,) for a in (self.active, self.links, self.distances)):
            raise ValueError("contact metadata shape")
        if any(not np.isfinite(a).all() for a in
               (self.points, self.outward_normals, self.jacobians, self.distances)):
            raise ValueError("nonfinite contact input")
        if n and np.any(np.linalg.norm(self.outward_normals, axis=1) < 1e-10):
            raise ValueError("zero normal")

def contact_bases(outward_normals):
    """Columns map [inward normal, tangent 1, tangent 2] to world force."""
    out = np.asarray(outward_normals, dtype=float)
    n = -out / np.linalg.norm(out, axis=1, keepdims=True)
    ref = np.eye(3)[np.argmin(np.abs(n), axis=1)]
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return np.stack((n, t1, t2), axis=2)

def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def matrices(contacts, center, rank_tolerance=1e-6, bases=None):
    n, _, ndof = contacts.jacobians.shape
    bases = contact_bases(contacts.outward_normals) if bases is None else bases
    G = np.zeros((6, 3*n))
    Jall = np.einsum("nji,njk->nik", bases, contacts.jacobians)
    for i in range(n):
        G[:3, 3*i:3*i+3] = bases[i]
        G[3:, 3*i:3*i+3] = skew(contacts.points[i] - center) @ bases[i]
    active_rows = (3*np.flatnonzero(contacts.active)[:, None] + np.arange(3)).ravel()
    Jactive = Jall[contacts.active].reshape(-1, ndof)
    T = np.zeros((ndof, 3*n))
    T[:, active_rows] = Jactive.T
    if len(active_rows):
        U, singular_values, _ = np.linalg.svd(Jactive, full_matrices=True)
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        null_local = U[:, rank:]
        N = np.zeros((null_local.shape[1], 3*n))
        N[:, active_rows] = null_local.T
    else:
        rank = 0
        N = np.zeros((0, 3*n))
    return G, T, N, bases, rank

@dataclass
class Allocation:
    torques: np.ndarray
    local_forces: np.ndarray
    status: str
    feasible: bool
    fallback: bool
    objective: float | None
    wrench: np.ndarray | None
    max_constraint_violation: float | None
    normal_force_sum: float
    contacts: int
    jacobian_rank: int
    compute_ms: float
    solver_iterations: int

class ForceAllocator:
    def __init__(self, ndof, settings=ForceSettings()):
        self.ndof = int(ndof)
        self.settings = settings
        self.last_feasible_torque = np.zeros(ndof)

    def solve(self, contacts, center):
        start = perf_counter()
        n = len(contacts.points)
        if contacts.jacobians.shape[2] != self.ndof:
            raise ValueError("wrong robot DOF")
        if not n:
            return self._fallback("no_contacts", n, 0, start)
        cfg = self.settings
        G, T, N, _, rank = matrices(contacts, np.asarray(center), cfg.rank_tolerance)
        normal = np.zeros((n, 3*n))
        normal[np.arange(n), 3*np.arange(n)] = 1
        friction = np.zeros((4*n, 3*n))
        for i in range(n):
            for k, (s1, s2) in enumerate(((1,1), (1,-1), (-1,1), (-1,-1))):
                friction[4*i+k, 3*i:3*i+3] = (-cfg.friction, s1, s2)
        A = np.vstack((normal, friction, normal.sum(axis=0, keepdims=True), T, N))
        lower = np.r_[np.zeros(n), np.full(4*n, -np.inf),
                      cfg.minimum_normal_force, np.full(self.ndof, -cfg.torque_limit),
                      np.full(len(N), -cfg.nullspace_margin)]
        upper = np.r_[np.full(n, np.inf), np.zeros(4*n), np.inf,
                      np.full(self.ndof, cfg.torque_limit),
                      np.full(len(N), cfg.nullspace_margin)]
        solver = osqp.OSQP()
        # Objective is exactly 1/2 ||G f||^2; no added force penalty.
        solver.setup(P=sp.triu(sp.csc_matrix(G.T @ G), format="csc"),
                     q=np.zeros(3*n), A=sp.csc_matrix(A), l=lower, u=upper,
                     verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                     max_iter=10000, polishing=False)
        result = solver.solve(raise_error=False)
        status = result.info.status
        if result.x is None or result.info.status_val not in (1, 2):
            return self._fallback(status, n, rank, start, result.info.iter)
        x = result.x
        ax = A @ x
        violation = float(max(0., np.max(lower-ax), np.max(ax-upper)))
        if not np.isfinite(x).all() or violation > 2e-4:
            return self._fallback(status + "_constraint_check_failed", n, rank,
                                  start, result.info.iter)
        tau = T @ x
        # Clip only numerical solver residual; report/check unrounded constraints above.
        tau = np.clip(tau, -cfg.torque_limit, cfg.torque_limit)
        self.last_feasible_torque = tau.copy()
        return Allocation(tau, x.reshape(n, 3), status, True, False,
                          float(0.5*np.linalg.norm(G @ x)**2), G @ x, violation,
                          float(normal.sum(axis=0) @ x), n, rank,
                          (perf_counter()-start)*1000, result.info.iter)

    def _fallback(self, status, n, rank, start, iterations=0):
        return Allocation(self.last_feasible_torque.copy(), np.zeros((n,3)),
                          status, False, True, None, None, None, 0., n, rank,
                          (perf_counter()-start)*1000, iterations)
