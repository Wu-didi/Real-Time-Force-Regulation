"""Shared closed-loop SDF contact estimation and force regulation."""
import time

import mujoco
import numpy as np

from .allocator import ForceAllocator, contact_bases
from .geometry import estimate_contacts
from .protocol import grasp_is_lost


def load_state(data, path):
    """Load a named, numeric MuJoCo state archive into a fresh MjData."""
    with np.load(path, allow_pickle=False) as state:
        for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            if name not in state:
                if name == "qpos":
                    raise ValueError("The state archive must contain qpos.")
                continue
            value = state[name]
            target = getattr(data, name)
            if value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} shape or values in {path}")
            target[:] = value


class Scene:
    def initialize(self, settings, method, period, threshold, candidates):
        self.settings = settings
        self.method = method
        self.period = period
        self.threshold = threshold
        self.candidates = candidates
        self.initial = {
            name: getattr(self.data, name).copy()
            for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")
        }
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for name, value in self.initial.items():
            getattr(self.data, name)[:] = value
        mujoco.mj_forward(self.model, self.data)
        self.allocator = ForceAllocator(len(self.dofs), self.settings)
        self.allocator.last_feasible_torque = self.initial["ctrl"][self.actuators].copy()
        self.next_control = 0.0
        self.contacts = self.allocation = None
        self.fixed = None
        self.fixed_targets = []
        self.control_updates = self.feasible_updates = self.fallback_updates = 0
        self.compute_ms = 0.0

    def jacobian(self, body, point):
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacobian, None, point, int(body))
        return jacobian

    def arm_control(self):
        pass

    def update_control(self):
        began = time.perf_counter()
        data = self.data
        mujoco.mj_forward(self.model, data)
        if self.fixed is None:
            points, links, active = self.world_surface()
            if self.method == "fingertip_only":
                keep = np.isin(links, self.fingertip_region)
                points, links, active = points[keep], links[keep], active[keep]
            self.contacts = estimate_contacts(
                points, links, active, data.xpos[self.object_body],
                data.xmat[self.object_body].reshape(3, 3), self.sdf,
                lambda link, point: self.jacobian(link, point)[:, self.dofs],
                len(self.dofs), threshold=self.threshold, per_link=self.candidates,
            )
            self.allocation = self.allocator.solve(self.contacts, data.xipos[self.object_body])
            torque = self.allocation.torques
            if self.method.startswith("fixed_"):
                self.fixed = torque.copy()
                if self.method == "fixed_impedance":
                    forces = np.einsum(
                        "nij,nj->ni", contact_bases(self.contacts.outward_normals),
                        self.allocation.local_forces,
                    )
                    for idx in np.flatnonzero(self.contacts.active):
                        body = int(self.contacts.links[idx])
                        point = self.contacts.points[idx]
                        rotation = data.xmat[body].reshape(3, 3)
                        local = rotation.T @ (point - data.xpos[body])
                        self.fixed_targets.append((body, local, point + forces[idx] / 500.0))
        else:
            torque = self.fixed
        if self.method == "fixed_impedance":
            torque = np.zeros(len(self.dofs))
            for body, local, target in self.fixed_targets:
                point = data.xmat[body].reshape(3, 3) @ local + data.xpos[body]
                jacobian = self.jacobian(body, point)
                force = 500.0 * (target - point) - jacobian @ data.qvel
                torque += jacobian[:, self.dofs].T @ force
            torque = np.clip(torque, -self.settings.torque_limit, self.settings.torque_limit)
        data.ctrl[self.actuators] = torque
        self.next_control = float(data.time) + self.period
        self.control_updates += 1
        self.feasible_updates += int(self.allocation.feasible)
        self.fallback_updates += int(self.allocation.fallback)
        self.compute_ms = (time.perf_counter() - began) * 1000.0

    def step(self, wrench=None, perturb=None):
        data = self.data
        if data.time >= self.next_control - 1e-10:
            self.update_control()
        data.xfrc_applied[:] = 0.0
        if perturb is not None and perturb.select == self.object_body and perturb.active:
            mujoco.mjv_applyPerturbForce(self.model, data, perturb)
        if wrench is not None:
            data.xfrc_applied[self.object_body] += wrench
        self.arm_control()
        mujoco.mj_step(self.model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Nonfinite MuJoCo state.")
        mujoco.mj_kinematics(self.model, data)

    def lost(self):
        if self.bounds is None:
            return None
        data = self.data
        return grasp_is_lost(self.hull_points(), data.xpos[self.object_body],
                             data.xmat[self.object_body].reshape(3, 3), self.bounds)
