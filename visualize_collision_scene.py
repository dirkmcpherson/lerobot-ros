#!/usr/bin/env python3
"""Visualize the robot collision scene in meshcat (browser-based 3D viewer).

Displays the robot with collision geometries, the ground plane, and any
custom obstacles.  You can interactively set joint positions via sliders
or pass a configuration on the command line.

Usage:
    python visualize_collision_scene.py                    # default home config
    python visualize_collision_scene.py --joints 0 0.52 3.14 -1.22 0 -0.52 1.57
    python visualize_collision_scene.py --ground-z -0.05   # lower the ground plane
"""

import argparse
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from user_input.collision import CollisionChecker

# ---- defaults ----
URDF = "/home/james/lerobot-ros/gen3_7dof.urdf"
PKG_DIR = "/home/james/workspace/ros2_kortex_ws/src/ros2_kortex"
SRDF = (
    "/home/james/workspace/ros2_kortex_ws/src/ros2_kortex/"
    "kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_85_moveit_config/config/gen3.srdf"
)
HOME = [0.0, 0.52, 3.14, -1.22, 0.0, -0.52, 1.57]


def build_q(model, joint_positions):
    """Convert arm joint positions to a full pinocchio q vector."""
    q = pin.neutral(model)
    for i, pos in enumerate(joint_positions):
        joint = model.joints[i + 1]
        if joint.nq == 2:
            q[joint.idx_q] = np.cos(pos)
            q[joint.idx_q + 1] = np.sin(pos)
        else:
            q[joint.idx_q] = pos
    return q


def main():
    parser = argparse.ArgumentParser(description="Visualize collision scene")
    parser.add_argument(
        "--urdf", default=URDF, help="Path to robot URDF"
    )
    parser.add_argument(
        "--joints", nargs="+", type=float, default=HOME,
        help="Joint positions (radians)",
    )
    parser.add_argument(
        "--ground-z", type=float, default=0.0,
        help="Ground plane Z height (default: 0.0)",
    )
    parser.add_argument(
        "--no-ground", action="store_true",
        help="Disable ground plane",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep joint 2 through its range to test collisions",
    )
    args = parser.parse_args()

    # Load models
    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()

    # Visual geometry (for display)
    visual_model = pin.buildGeomFromUrdf(
        model, args.urdf, pin.VISUAL, package_dirs=[PKG_DIR]
    )

    # Collision geometry (for checking + display)
    collision_model = pin.buildGeomFromUrdf(
        model, args.urdf, pin.COLLISION, package_dirs=[PKG_DIR]
    )

    # Set up collision checker
    ground_z = None if args.no_ground else args.ground_z
    checker = CollisionChecker(
        urdf_path=args.urdf,
        package_dirs=[PKG_DIR],
        srdf_path=SRDF,
        ground_plane_z=ground_z,
    )
    checker.setup(model)

    # Create meshcat visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Draw collision geometries as wireframes
    viz.displayCollisions(True)
    viz.displayVisuals(True)

    # Add ground plane visualization manually (the collision checker's
    # ground is in a separate geom_model, so we draw it in meshcat directly)
    if ground_z is not None:
        import meshcat.geometry as mg
        import meshcat.transformations as mtf

        ground_mesh = mg.Box([4.0, 4.0, 0.01])
        ground_material = mg.MeshPhongMaterial(
            color=0x888888, opacity=0.4, transparent=True
        )
        transform = mtf.translation_matrix([0, 0, ground_z - 0.005])
        viz.viewer["ground_plane"].set_object(ground_mesh, ground_material)
        viz.viewer["ground_plane"].set_transform(transform)

    # Display initial configuration
    q = build_q(model, args.joints)
    viz.display(q)

    # Check collision
    has_collision = checker.check_collision(q)
    pairs = checker.get_colliding_pairs(q) if has_collision else []
    print(f"\nConfiguration: {args.joints}")
    print(f"Collision: {has_collision}")
    if pairs:
        print("Colliding pairs:")
        for g1, g2 in pairs:
            print(f"  {g1} <-> {g2}")

    if args.sweep:
        print("\nSweeping joint 2 from -2.2 to 2.2 rad...")
        joints = list(args.joints)
        for angle in np.linspace(-2.2, 2.2, 200):
            joints[1] = angle
            q = build_q(model, joints)
            viz.display(q)

            has_collision = checker.check_collision(q)
            if has_collision:
                pairs = checker.get_colliding_pairs(q)
                pair_str = ", ".join(f"{a}<->{b}" for a, b in pairs)
                print(f"  j2={angle:.2f}: COLLISION ({pair_str})")

            time.sleep(0.03)
        print("Sweep done.")

    print(f"\nViewer URL: {viz.viewer.url()}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
