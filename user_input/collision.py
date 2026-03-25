"""Pinocchio + hpp-fcl collision checker for robot arms.

Loads collision geometry from a URDF, applies SRDF exclusion rules for
self-collision pairs, and optionally adds environment obstacles (ground
plane, boxes).  The main entry point is :meth:`check_collision`, which
returns True if a given joint configuration would cause any collision.

Usage::

    checker = CollisionChecker(
        urdf_path="/tmp/gen3.urdf",
        package_dirs=["/path/to/kortex_description/.."],
        srdf_path="/path/to/gen3.srdf",         # optional
        ground_plane_z=0.0,                       # optional
    )
    checker.setup(pin_model)  # pass existing pinocchio model

    if checker.check_collision(q):
        print("collision!")
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CollisionChecker:
    """Self-collision and environment collision checking via Pinocchio."""

    def __init__(
        self,
        urdf_path: str,
        package_dirs: list[str] | None = None,
        srdf_path: str | None = None,
        ground_plane_z: float | None = None,
        obstacles: list[dict] | None = None,
        safety_margin: float = 0.02,
    ):
        """
        Parameters
        ----------
        urdf_path : str
            Path to the robot URDF (must contain <collision> tags).
        package_dirs : list[str] | None
            Directories to resolve ``package://`` mesh paths.
        srdf_path : str | None
            Path to SRDF file with ``<disable_collisions>`` entries.
            If None, all non-adjacent pairs are checked (conservative).
        ground_plane_z : float | None
            If set, adds a large thin box at this z height.
        obstacles : list[dict] | None
            Additional obstacles. Each dict has:
              - "type": "box"
              - "half_extents": [hx, hy, hz]
              - "position": [x, y, z]
              - "name": str (optional)
        safety_margin : float
            Minimum allowed distance (meters) between any collision pair.
            Default 0.02 (2cm). Set to 0.0 for pure intersection checking.
        """
        self._urdf_path = urdf_path
        self._package_dirs = package_dirs or []
        self._srdf_path = srdf_path
        self._ground_plane_z = ground_plane_z
        self._obstacles = obstacles or []
        self._safety_margin = safety_margin

        self._pin_model = None
        self._geom_model = None
        self._pin_data = None
        self._geom_data = None

        # Track environment objects for RViz publishing
        self._env_objects: list[dict] = []
        self._marker_pub = None
        self._marker_msg = None
        self._marker_timer = None

    def setup(self, pin_model) -> None:
        """Build collision geometry from the URDF, using the given kinematic model."""
        import pinocchio as pin

        self._pin_model = pin_model
        self._pin_data = pin_model.createData()

        # Load collision geometry
        self._geom_model = pin.buildGeomFromUrdf(
            pin_model,
            self._urdf_path,
            pin.COLLISION,
            package_dirs=self._package_dirs,
        )

        # Start with all possible collision pairs
        self._geom_model.addAllCollisionPairs()
        n_before = len(self._geom_model.collisionPairs)

        # Remove SRDF-excluded pairs (adjacent links, known never-collide)
        if self._srdf_path is not None:
            srdf_xml = Path(self._srdf_path).read_text()
            pin.removeCollisionPairsFromXML(
                pin_model, self._geom_model, srdf_xml, verbose=False
            )

        n_after = len(self._geom_model.collisionPairs)
        logger.info(
            f"Collision checker: {self._geom_model.ngeoms} geometries, "
            f"{n_before} pairs → {n_after} after SRDF exclusions"
        )

        # Add ground plane
        if self._ground_plane_z is not None:
            self._add_ground_plane(self._ground_plane_z)

        # Add custom obstacles
        for obs in self._obstacles:
            self._add_box_obstacle(
                half_extents=obs["half_extents"],
                position=obs["position"],
                name=obs.get("name", "obstacle"),
            )

        # Create geometry data (must be after all objects are added)
        self._geom_data = pin.GeometryData(self._geom_model)

        active_pairs = []
        for pair in self._geom_model.collisionPairs:
            g1 = self._geom_model.geometryObjects[pair.first].name
            g2 = self._geom_model.geometryObjects[pair.second].name
            active_pairs.append(f"{g1} <-> {g2}")
        logger.info(
            f"Active collision pairs ({len(active_pairs)}): "
            + ", ".join(active_pairs)
        )

    def _add_ground_plane(self, z: float) -> None:
        """Add a large thin box as a ground plane at height z."""
        import pinocchio as pin

        hppfcl = pin.hppfcl
        ground = hppfcl.Box(4.0, 4.0, 0.01)  # 4m x 4m x 1cm
        placement = pin.SE3.Identity()
        placement.translation = np.array([0.0, 0.0, z - 0.005])

        ground_obj = pin.GeometryObject(
            "ground_plane", 0, 0, placement, ground
        )
        ground_id = self._geom_model.addGeometryObject(ground_obj)

        # Pair ground with all moving robot links (skip base_link — it's
        # mounted on the table surface and always touches the ground plane)
        for i in range(ground_id):
            if self._geom_model.geometryObjects[i].parentJoint > 0:
                self._geom_model.addCollisionPair(pin.CollisionPair(i, ground_id))

        self._env_objects.append({
            "name": "ground_plane",
            "type": "box",
            "size": [4.0, 4.0, 0.01],
            "position": [0.0, 0.0, z - 0.005],
        })
        logger.info(f"Added ground plane at z={z}")

    def _add_box_obstacle(
        self,
        half_extents: list[float],
        position: list[float],
        name: str = "obstacle",
    ) -> None:
        """Add a box obstacle to the collision scene."""
        import pinocchio as pin

        hppfcl = pin.hppfcl
        box = hppfcl.Box(*(2.0 * np.array(half_extents)))
        placement = pin.SE3.Identity()
        placement.translation = np.array(position)

        box_obj = pin.GeometryObject(name, 0, 0, placement, box)
        box_id = self._geom_model.addGeometryObject(box_obj)

        # Pair with all moving robot links (parentJoint > 0)
        for i in range(box_id):
            if self._geom_model.geometryObjects[i].parentJoint > 0:
                self._geom_model.addCollisionPair(pin.CollisionPair(i, box_id))

        full_size = (2.0 * np.array(half_extents)).tolist()
        self._env_objects.append({
            "name": name,
            "type": "box",
            "size": full_size,
            "position": list(position),
        })
        logger.info(f"Added box obstacle '{name}' at {position}")

    def check_collision(self, q: np.ndarray) -> bool:
        """Return True if any collision pair is closer than the safety margin."""
        import pinocchio as pin

        pin.computeDistances(
            self._pin_model, self._pin_data,
            self._geom_model, self._geom_data,
            q,
        )
        for k in range(len(self._geom_model.collisionPairs)):
            if self._geom_data.distanceResults[k].min_distance < self._safety_margin:
                return True
        return False

    def get_colliding_pairs(self, q: np.ndarray) -> list[tuple[str, str, float]]:
        """Return list of (name1, name2, distance) for pairs within the safety margin."""
        import pinocchio as pin

        pin.computeDistances(
            self._pin_model, self._pin_data,
            self._geom_model, self._geom_data,
            q,
        )
        pairs = []
        for k in range(len(self._geom_model.collisionPairs)):
            dist = self._geom_data.distanceResults[k].min_distance
            if dist < self._safety_margin:
                pair = self._geom_model.collisionPairs[k]
                g1 = self._geom_model.geometryObjects[pair.first].name
                g2 = self._geom_model.geometryObjects[pair.second].name
                pairs.append((g1, g2, dist))
        return pairs

    def publish_to_rviz(self, node) -> None:
        """Publish environment collision objects as RViz markers.

        Creates a timer that re-publishes at 1 Hz so markers are always
        visible regardless of when RViz connects.

        Parameters
        ----------
        node : rclpy.node.Node
            A ROS2 node to create the publisher on.
        """
        from builtin_interfaces.msg import Time
        from visualization_msgs.msg import Marker, MarkerArray

        self._marker_pub = node.create_publisher(
            MarkerArray, "/collision_objects", 10
        )

        marker_array = MarkerArray()
        for i, obj in enumerate(self._env_objects):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = Time(sec=0, nanosec=0)  # use RViz time
            m.ns = "collision_checker"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = obj["position"][0]
            m.pose.position.y = obj["position"][1]
            m.pose.position.z = obj["position"][2]
            m.pose.orientation.w = 1.0
            m.scale.x = obj["size"][0]
            m.scale.y = obj["size"][1]
            m.scale.z = obj["size"][2]
            # Semi-transparent red
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 0.4
            # Keep marker alive indefinitely
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            marker_array.markers.append(m)

            # Safety margin shell
            ms = Marker()
            ms.header.frame_id = "base_link"
            ms.header.stamp = Time(sec=0, nanosec=0)
            ms.ns = "collision_checker_margin"
            ms.id = i
            ms.type = Marker.CUBE
            ms.action = Marker.ADD
            ms.pose.position.x = obj["position"][0]
            ms.pose.position.y = obj["position"][1]
            ms.pose.position.z = obj["position"][2]
            ms.pose.orientation.w = 1.0
            ms.scale.x = obj["size"][0] + 2.0 * self._safety_margin
            ms.scale.y = obj["size"][1] + 2.0 * self._safety_margin
            ms.scale.z = obj["size"][2] + 2.0 * self._safety_margin
            ms.color.r = 1.0
            ms.color.g = 0.8
            ms.color.b = 0.0
            ms.color.a = 0.15
            ms.lifetime.sec = 0
            ms.lifetime.nanosec = 0
            marker_array.markers.append(ms)

        self._marker_msg = marker_array

        # Re-publish at 1 Hz so markers survive RViz restarts
        self._marker_timer = node.create_timer(1.0, self._publish_markers)
        self._publish_markers()

        logger.info(
            f"Publishing {len(self._env_objects)} collision objects "
            f"to /collision_objects at 1 Hz (with {self._safety_margin:.3f}m margin)"
        )

    def _publish_markers(self) -> None:
        if self._marker_pub is not None and self._marker_msg is not None:
            self._marker_pub.publish(self._marker_msg)

    @property
    def geom_model(self):
        return self._geom_model

    @property
    def pin_model(self):
        return self._pin_model
