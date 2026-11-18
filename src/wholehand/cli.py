"""Command-line entry points for asset preparation, viewing, and trials."""
import argparse
import json
from pathlib import Path


def controls(parser, force):
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--force", type=float, default=force, help="Viewer pulse amplitude in N")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--protocol", action="store_true", help="Enable loss checks and full load schedule")
    parser.add_argument("--rotation", type=float, choices=(0.0, 10.0, 30.0, 60.0, 90.0), default=0.0)
    parser.add_argument("--method", choices=("whole_hand", "fingertip_only", "fixed_feedforward",
                                            "fixed_impedance"), default="whole_hand")
    parser.add_argument("--period", type=float, default=.01, help="Control interval in simulated seconds")
    parser.add_argument("--threshold", type=float, default=.01, help="Candidate signed distance in meters")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--normal-force", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=261116)
    parser.add_argument("--output", type=Path, help="Write trial JSON here")


def main():
    parser = argparse.ArgumentParser(prog="wholehand", description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    robotis = modes.add_parser("mujoco", help="Robotis HX5-D20 with a box or YCB object")
    robotis.add_argument("--robotis-xml", required=True, type=Path)
    robotis.add_argument("--ycb", type=Path, help="Prepared YCB asset directory")
    robotis.add_argument("--grasp", type=Path, help="Initial grasp JSON")
    robotis.add_argument("--state", type=Path, help="Override the initial state with an NPZ")
    robotis.add_argument("--surface", choices=("visual", "collision_hull"), default="visual")
    controls(robotis, 1.0)
    dex = modes.add_parser("dexjoco", help="Native bimanual DexJoCo assembly")
    dex.add_argument("--grasp-dir", required=True, type=Path)
    dex.add_argument("--target", choices=("peg", "socket"), default="peg")
    dex.add_argument("--hand-samples", type=Path)
    dex.add_argument("--sdf", type=Path)
    controls(dex, 0.05)
    ycb = modes.add_parser("prepare-ycb", help="Prepare convex collider meshes and a canonical SDF")
    ycb.add_argument("--mesh", type=Path, required=True)
    ycb.add_argument("--output", type=Path, required=True)
    ycb.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
    capture = modes.add_parser("capture-dexjoco", help="Capture the first qualified grasp from a replay")
    capture.add_argument("--replay", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--seed", type=int, default=261118)
    args = parser.parse_args()
    if args.mode == "prepare-ycb":
        from .prepare import prepare_ycb
        prepare_ycb(args.mesh, args.output, args.device)
        return
    if args.mode == "capture-dexjoco":
        from .prepare import capture_dexjoco
        print(json.dumps(capture_dexjoco(args.replay, args.output, args.seed), indent=2))
        return
    from .allocator import ForceSettings
    if not (args.seconds > 0 and args.period > 0 and args.threshold > 0 and
            args.candidates > 0 and args.normal_force > 0 and args.force > 0):
        parser.error("Durations, thresholds, candidate count and forces must be positive.")
    control = dict(settings=ForceSettings(minimum_normal_force=args.normal_force),
                   method=args.method, period=args.period, threshold=args.threshold,
                   candidates=args.candidates)
    if args.mode == "mujoco":
        from .robotis_scene import RobotisScene
        scene = RobotisScene(args.robotis_xml, args.ycb, args.grasp, args.state,
                             args.surface, **control)
    else:
        try:
            from .dexjoco import DexScene
            scene = DexScene(args.grasp_dir, args.target, args.hand_samples, args.sdf, **control)
        except ModuleNotFoundError as error:
            if error.name == "dexjoco":
                parser.error("Install DexJoCo using the README before running this scene.")
            raise
    if args.viewer:
        from .viewer import show
        show(scene, args.force)
        return
    from .benchmark import run
    from .protocol import Schedule
    seconds = (Schedule().end if args.rotation else Schedule().perturb_start) if args.protocol else args.seconds
    result = run(scene, seconds, args.rotation, args.seed, args.protocol)
    result["backend"] = args.mode
    result["control_period_s"] = args.period
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as file:
            file.write(report + "\n")
    print(report)
