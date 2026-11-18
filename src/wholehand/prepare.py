"""Build portable simulation inputs from public meshes and replay data."""
import json
from pathlib import Path

import mujoco
import numpy as np


def prepare_ycb(source, output, device):
    import coacd
    import trimesh

    from .warp_sdf import make_grid

    output = Path(output)
    mesh = trimesh.load_mesh(source, process=True)
    shift = mesh.bounds.mean(0)
    mesh.vertices -= shift
    if mesh.volume <= 0:
        raise ValueError("The mesh must have positive volume and outward winding.")
    output.mkdir(parents=True, exist_ok=False)
    make_grid(mesh, device=device).save(output / "sdf.npz")
    mesh.export(output / "canonical.obj")
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces), threshold=.001, real_metric=True,
        max_convex_hull=64, preprocess_mode="auto", resolution=1000,
        mcts_nodes=15, mcts_iterations=40, mcts_max_depth=3, merge=True,
        decimate=True, max_ch_vertex=96, seed=261116,
    )
    for i, (vertices, faces) in enumerate(parts):
        trimesh.Trimesh(vertices, faces, process=False).export(output / f"collision_{i:03d}.obj")
    record = {
        "canonical_shift_m": shift.tolist(),
        "canonical_bounds_m": mesh.bounds.tolist(),
        "volume_m3": float(mesh.volume),
        "COM_m": mesh.center_mass.tolist(),
        "inertia_density1": mesh.moment_inertia.tolist(),
        "collision_parts": len(parts),
    }
    (output / "READY.json").write_text(json.dumps(record, indent=2) + "\n")


def capture_dexjoco(replay, output, seed):
    import zarr
    from dexjoco.tasks.mappings import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import restore_initial_state

    source = zarr.open(str(replay), mode="r")["data"]
    actions = source["action"]
    if actions.ndim != 2 or actions.shape[1] != 46 or not len(actions):
        raise ValueError("Expected a nonempty DexJoCo bimanual (T, 46) replay.")
    config = CONFIG_MAPPING["bimanual_assembly"]()
    env = config.get_environment(
        policy_mode=True, render_mode="rgb_array", randomize=False,
        randomize_dynamics=False, image_obs=False, seed=seed,
    )
    try:
        env.reset(seed=seed)
        model, data = env.unwrapped.model, env.unwrapped.data
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        fields = {key: np.array(value, copy=True) for key in dir(model)
                  if isinstance(value := getattr(model, key, None), np.ndarray)}
        mujoco.mj_saveModel(model, str(output / "model.mjb"), None)
        restore_initial_state(env, "bimanual_assembly", config, np.asarray(source["state"][0]).ravel())
        mujoco.mj_forward(model, data)
        peg = model.body("industreal_round_peg_8mm").id
        start_height = float(data.xipos[peg, 2])
        qualified = False
        for frame, action in enumerate(actions):
            env.unwrapped.step({"right": action[:23], "left": action[23:]})
            mujoco.mj_forward(model, data)
            fingers = set()
            for contact in data.contact:
                if contact.dist > 0:
                    continue
                a, b = model.geom_bodyid[contact.geom]
                body = b if a == peg else a if b == peg else -1
                if body >= 0:
                    name = model.body(body).name
                    if name.endswith("_right") and name[:2] in ("ff", "mf", "rf", "th"):
                        fingers.add(name[:2])
            lift = float(data.xipos[peg, 2]) - start_height
            if lift >= .03 and len(fingers) >= 2:
                qualified = True
                break
        changes = {name: getattr(model, name) for name, value in fields.items()
                   if value.tobytes() != getattr(model, name).tobytes()}
        np.savez_compressed(output / "model_changes.npz", **changes)
        np.savez_compressed(output / "initial_state.npz",
                            **{name: getattr(data, name) for name in
                               ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")})
        result = {"frame": frame, "qualified": qualified, "peg_lift_m": lift,
                  "finger_groups": sorted(fingers), "seed": seed}
        (output / "READY.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
