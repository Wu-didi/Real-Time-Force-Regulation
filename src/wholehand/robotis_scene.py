"""Robotis HX5-D20 and YCB object simulation using the original geometry estimator."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .geometry import BoxSDF, MeshSDF
from .robotis import build_model
from .scene import Scene, load_state
from .ycb import HandSurface, build_ycb, hand_hull_points


class RobotisScene(Scene):
    def __init__(self, robotis_xml, ycb, grasp, state, surface, **control):
        initial = json.loads(Path(grasp).read_text()) if grasp else {}
        position = initial.get("object_position", (0.055, 0, 0.14))
        if ycb:
            model, _, qpos, self.dofs, asset = build_ycb(
                robotis_xml, ycb, initial.get("mass_kg", 0.603), position,
                initial.get("object_quaternion", (1, 0, 0, 0)),
            )
            self.sdf = MeshSDF.load(Path(ycb) / "sdf.npz")
            self.bounds = np.asarray(asset["canonical_bounds_m"])
        else:
            half_size = np.array([0.025, 0.025, 0.025])
            model, _, qpos, self.dofs = build_model(
                robotis_xml, {"type": "box", "size": half_size, "position": position}
            )
            self.sdf = BoxSDF(half_size)
            self.bounds = np.array([-half_size, half_size])
        self.model = model
        self.data = mujoco.MjData(model)
        self.object_body = model.body("test_object").id
        self.actuators = np.arange(model.nu)
        limit = control["settings"].torque_limit
        model.actuator_ctrlrange[:] = [-limit, limit]
        collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_friction[collision] = [control["settings"].friction, 0, 0]
        if "qpos" in initial:
            self.data.qpos[qpos] = initial["qpos"]
        if state:
            load_state(self.data, state)
        self.surface = HandSurface.from_model(model, mode=surface)
        self.fingertip_region = [model.body(f"finger_r_link{i}").id
                                 for i in (3, 4, 7, 8, 11, 12, 15, 16, 19, 20)]
        self.initialize(**control)

    def world_surface(self):
        return self.surface.world(self.data)

    def hull_points(self):
        return hand_hull_points(self.model, self.data, self.surface)
