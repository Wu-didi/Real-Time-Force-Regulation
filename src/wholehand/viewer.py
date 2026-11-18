"""Native MuJoCo window with per-step picked-point perturbation."""
from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np


def show(scene, default_force):
    import mujoco.viewer

    keys = SimpleQueue()
    playing = False
    force = default_force
    pulse = np.zeros(6)
    expires = 0.0
    data, model = scene.data, scene.model
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.pert.select = scene.object_body
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key == 32:
                        playing = not playing
                    elif key == ord("R"):
                        scene.reset()
                        expires = 0.0
                        viewer.pert.active = 0
                        playing = False
                    elif key == ord("X"):
                        expires = 0.0
                        viewer.pert.active = 0
                        data.xfrc_applied[:] = 0
                    elif key in (ord("-"), ord("=")):
                        force = float(np.clip(force * (2 if key == ord("=") else 0.5), 0.01, 10))
                    elif ord("1") <= key <= ord("6"):
                        index = key - ord("1")
                        pulse[:] = 0.0
                        pulse[index // 2] = force * (-1 if index % 2 else 1)
                        expires = data.time + 0.5
                        playing = True
                dragging = viewer.pert.active and viewer.pert.select == scene.object_body
                if playing or dragging:
                    scene.step(pulse if data.time < expires - 1e-10 else None, viewer.pert)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - start)))
