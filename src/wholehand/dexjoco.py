"""Whole-hand control on the native bimanual DexJoCo assembly scene."""
from pathlib import Path

import mujoco
import numpy as np

from .geometry import MeshSDF
from .kinematics import HandKinematics, quaternion_matrix
from .native_geometry import RigidObjectGrid, collision_geoms, export_native_hand, geom_mesh
from .scene import Scene, load_state

FINGERS = ("ff", "mf", "rf", "th")
TARGETS = {
    "peg": ("right", "industreal_round_peg_8mm"),
    "socket": ("left", "industreal_tray_insert_round_peg_8mm"),
}


class DexScene(Scene):
    def __init__(self, folder, target, hand_path, sdf_path, **control):
        from dexjoco.sim.controllers.opspace import opspace

        self.opspace = opspace
        folder = Path(folder)
        self.model = model = mujoco.MjModel.from_binary_path(str(folder / "model.mjb"))
        changes_path = folder / "model_changes.npz"
        if changes_path.exists():
            with np.load(changes_path, allow_pickle=False) as changes:
                for name in changes.files:
                    array = getattr(model, name)
                    if not isinstance(array, np.ndarray) or array.shape != changes[name].shape:
                        raise ValueError(f"Invalid model field {name}")
                    array[:] = changes[name]
        model.opt.timestep = 0.001
        model.opt.gravity[:] = 0.0
        self.data = data = mujoco.MjData(model)
        load_state(data, folder / "initial_state.npz")
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        side, object_name = TARGETS[target]
        self.object_body = model.body(object_name).id

        def joints(hand_side):
            return [model.joint(f"{finger}j{k}_{hand_side}").id
                    for finger in FINGERS for k in range(4)]

        def actuators(hand_side):
            return np.array([model.actuator(f"{finger}a{k}_{hand_side}").id
                             for finger in FINGERS for k in range(4)])

        self.dofs = model.jnt_dofadr[joints(side)]
        self.actuators = act = actuators(side)
        other = "left" if side == "right" else "right"
        data.ctrl[actuators(other)] = data.qpos[model.jnt_qposadr[joints(other)]]
        if not (np.all(model.actuator_gear[act, 0] == 1.0) and
                np.all(model.actuator_dyntype[act] == 0)):
            raise ValueError("Expected native unit-gear Allegro actuators without dynamics.")
        model.actuator_gainprm[act] = 0.0
        model.actuator_gainprm[act, 0] = 1.0
        model.actuator_biasprm[act] = 0.0
        limit = control["settings"].torque_limit
        model.actuator_ctrlrange[act] = [-limit, limit]
        model.actuator_ctrllimited[act] = 1
        data.ctrl[act] = 0.0

        hand = HandKinematics.load(hand_path) if hand_path else export_native_hand(model, data)
        names = hand.metadata["link_names"]
        keep = np.isin(hand.point_link, [i for i, n in enumerate(names) if n.endswith("_" + side)])
        self.local = hand.points_local[keep]
        self.point_bodies = np.array([model.body(names[i]).id for i in hand.point_link[keep]])
        palm = model.body(f"allegro_palm_{side}").id
        tips = [model.body(f"{finger}_tip_{side}").id for finger in FINGERS]
        self.active = self.point_bodies != palm
        self.fingertip_region = tips + [model.body(f"{finger}_distal_{side}").id for finger in FINGERS]
        self.refs = [(body, self.local[self.point_bodies == body].mean(0)) for body in tips + [palm]]
        self.sdf = (MeshSDF.load(sdf_path) if sdf_path else
                    RigidObjectGrid.from_body(model, self.object_body).grid)
        vertices = np.concatenate([
            geom_mesh(model, geom).vertices @ quaternion_matrix(model.geom_quat[geom]).T +
            model.geom_pos[geom] for geom in collision_geoms(model, self.object_body)
        ])
        self.bounds = np.array([vertices.min(0), vertices.max(0)])
        affected = np.isin(model.geom_bodyid, [*np.unique(self.point_bodies), self.object_body])
        model.geom_friction[affected] = [control["settings"].friction, 0, 0]
        model.geom_condim[affected] = 3
        self.arms = []
        for arm in ("right", "left"):
            ids = [model.joint(f"joint{i}_{arm}").id for i in range(1, 8)]
            site = model.site(f"attachment_site_{arm}").id
            act = np.array([model.actuator(f"actuator{i}_{arm}").id for i in range(1, 8)])
            orientation = np.zeros(4)
            mujoco.mju_mat2Quat(orientation, data.site_xmat[site])
            self.arms.append((model.jnt_dofadr[ids], act, site, data.site_xpos[site].copy(),
                              orientation, data.qpos[model.jnt_qposadr[ids]].copy()))
        self.initialize(**control)

    def world_surface(self):
        rotation = self.data.xmat[self.point_bodies].reshape(-1, 3, 3)
        points = np.einsum("nij,nj->ni", rotation, self.local) + self.data.xpos[self.point_bodies]
        return points, self.point_bodies, self.active

    def hull_points(self):
        return np.array([
            self.data.xmat[body].reshape(3, 3) @ point + self.data.xpos[body]
            for body, point in self.refs
        ])

    def arm_control(self):
        for dofs, act, site, position, orientation, joints in self.arms:
            self.data.ctrl[act] = self.opspace(
                model=self.model, data=self.data, site_id=site, dof_ids=dofs,
                pos=position, ori=orientation, joint=joints, gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0), damping_ratio=4.0,
            )
