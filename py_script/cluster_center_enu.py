#!/usr/bin/env python3
"""
Convert LiDAR cluster centres from body FLU coordinates to global ENU coordinates.

Inputs:
- /lidar/cluster_bboxes (visualization_msgs/MarkerArray)
- /fmu/out/vehicle_local_position_v1 (px4_msgs/VehicleLocalPosition)
- /fmu/out/vehicle_attitude (px4_msgs/VehicleAttitude)

Outputs:
- /lidar/obstacle_centers_enu (geometry_msgs/PoseArray)
- /lidar/obstacle_centers_enu_markers (visualization_msgs/MarkerArray)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


MARKER_LIFETIME_NS = 500_000_000
TEXT_Z_OFFSET = 0.4


class ClusterCenterEnuNode(Node):
    """Transform cluster centres from LiDAR-relative FLU to world ENU."""

    def __init__(self) -> None:
        super().__init__("cluster_center_enu")

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # PX4 topics use BEST_EFFORT QoS.
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.current_local_position: Optional[VehicleLocalPosition] = None
        self.current_attitude: Optional[VehicleAttitude] = None

        self.cluster_sub = self.create_subscription(
            MarkerArray,
            "/lidar/cluster_bboxes",
            self.cluster_callback,
            marker_qos,
        )
        self.localpos_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.vehicle_local_position_callback,
            px4_qos,
        )
        self.att_sub = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            px4_qos,
        )

        self.pub_enu_pose = self.create_publisher(
            PoseArray,
            "/lidar/obstacle_centers_enu",
            marker_qos,
        )
        self.pub_enu_markers = self.create_publisher(
            MarkerArray,
            "/lidar/obstacle_centers_enu_markers",
            marker_qos,
        )

        self.get_logger().info("Cluster center ENU converter ready")

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self.current_local_position = msg

    def vehicle_attitude_callback(self, msg: VehicleAttitude) -> None:
        self.current_attitude = msg

    def flu_to_frd(self, vector_flu: np.ndarray) -> np.ndarray:
        return np.array([vector_flu[0], -vector_flu[1], -vector_flu[2]], dtype=np.float64)

    def body_to_ned(self, vector_frd: np.ndarray, att: VehicleAttitude) -> np.ndarray:
        w, x, y, z = att.q[0], att.q[1], att.q[2], att.q[3]
        dcm = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        return dcm @ vector_frd

    def add_vehicle_pos(self, ned_rel: np.ndarray, pos: VehicleLocalPosition) -> np.ndarray:
        return np.array([pos.x, pos.y, pos.z], dtype=np.float64) + ned_rel

    def ned_to_enu(self, ned: np.ndarray) -> np.ndarray:
        return np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)

    def cluster_callback(self, msg: MarkerArray) -> None:
        if self.current_local_position is None or self.current_attitude is None:
            self.get_logger().warn("Waiting for PX4 local position and attitude before ENU conversion.")
            return

        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        if msg.markers:
            pose_array.header.stamp = msg.markers[0].header.stamp

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        valid_markers = [
            marker for marker in msg.markers
            if marker.action == Marker.ADD and marker.type == Marker.CUBE
        ]

        for idx, marker in enumerate(valid_markers):
            cluster_center_flu = np.array([
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z,
            ], dtype=np.float64)

            cluster_center_frd = self.flu_to_frd(cluster_center_flu)
            cluster_center_ned_rel = self.body_to_ned(cluster_center_frd, self.current_attitude)
            cluster_center_ned_abs = self.add_vehicle_pos(
                cluster_center_ned_rel,
                self.current_local_position,
            )
            cluster_center_enu = self.ned_to_enu(cluster_center_ned_abs)

            pose = Pose()
            pose.position.x = float(cluster_center_enu[0])
            pose.position.y = float(cluster_center_enu[1])
            pose.position.z = float(cluster_center_enu[2])
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            sphere_marker = Marker()
            sphere_marker.header = pose_array.header
            sphere_marker.ns = "obstacle_centers_enu"
            sphere_marker.id = idx
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose = pose
            sphere_marker.scale.x = 0.3
            sphere_marker.scale.y = 0.3
            sphere_marker.scale.z = 0.3
            sphere_marker.color.r = 0.1
            sphere_marker.color.g = 0.8
            sphere_marker.color.b = 1.0
            sphere_marker.color.a = 0.95
            sphere_marker.lifetime.nanosec = MARKER_LIFETIME_NS
            marker_array.markers.append(sphere_marker)

            text_marker = Marker()
            text_marker.header = pose_array.header
            text_marker.ns = "obstacle_centers_enu_text"
            text_marker.id = idx + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = pose.position.x
            text_marker.pose.position.y = pose.position.y
            text_marker.pose.position.z = pose.position.z + TEXT_Z_OFFSET
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.25
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.95
            text_marker.text = (
                f"ID {marker.id}\n"
                f"ENU({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
            )
            text_marker.lifetime.nanosec = MARKER_LIFETIME_NS
            marker_array.markers.append(text_marker)

        if pose_array.poses:
            self.pub_enu_pose.publish(pose_array)
            self.pub_enu_markers.publish(marker_array)
            self.get_logger().info(
                f"Published {len(pose_array.poses)} obstacle centres in ENU frame."
            )


def main() -> None:
    rclpy.init()
    node = ClusterCenterEnuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
