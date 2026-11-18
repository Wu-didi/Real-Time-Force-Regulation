"""Finite directional-load and spring-rotation trials."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .perturbation import spring_wrench
from .protocol import Schedule


def run(scene, seconds, angle, seed, check_loss):
    schedule = Schedule()
    rotate_start = schedule.perturb_start
    model, data, body = scene.model, scene.data, scene.object_body
    axis = np.random.default_rng(seed).normal(size=3)
    axis /= np.linalg.norm(axis)
    reference = None
    max_rotation = 0.0
    max_torque = 0.0
    lost_at = None
    peak_candidates = 0
    while data.time < seconds - 1e-10:
        t = float(data.time)
        wrench = np.r_[schedule.load(t, model.body_mass[body]), np.zeros(3)]
        if angle and t >= rotate_start and reference is None:
            mujoco.mj_comPos(model, data)
            reference = data.xipos[body].copy(), data.xmat[body].reshape(3, 3).copy()
        if reference is not None and t < rotate_start + schedule.perturb_s:
            center, rotation = reference
            x = np.clip((t - rotate_start) / 0.8, 0.0, 1.0)
            fraction = x**3 * (10 - 15*x + 6*x*x)
            target = Rotation.from_rotvec(axis * np.deg2rad(angle) * fraction).as_matrix()
            wrench, _ = spring_wrench(model, data, body, center, target @ rotation)
        scene.step(wrench)
        max_torque = max(max_torque, float(np.max(np.abs(data.ctrl[scene.actuators]))))
        peak_candidates = max(peak_candidates, len(scene.contacts.points))
        if reference is not None:
            relative = data.xmat[body].reshape(3, 3) @ reference[1].T
            max_rotation = max(max_rotation, float(np.rad2deg(Rotation.from_matrix(relative).magnitude())))
        if check_loss and scene.lost():
            lost_at = float(data.time)
            break
    return {
        "method": scene.method,
        "seed": seed,
        "success": lost_at is None if check_loss else None,
        "loss_time_s": lost_at,
        "simulated_s": float(data.time),
        "target_rotation_deg": angle,
        "max_rotation_deg": max_rotation,
        "control_updates": scene.control_updates,
        "feasible_updates": scene.feasible_updates,
        "fallback_updates": scene.fallback_updates,
        "peak_contact_candidates": peak_candidates,
        "max_command_torque_Nm": max_torque,
    }
