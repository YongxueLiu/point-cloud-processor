#!/usr/bin/env python3
"""
Industrial LiDAR Perception Pipeline (ROS2 + Open3D)
with PX4 Global ENU Coordinate Transformation

Pipeline:
1)  PointCloud2 -> numpy
2)  Voxel downsample (adaptive)
3)  SOR filter
4)  Ground plane removal (RANSAC)
5)  Wall removal (iterative RANSAC)
6)  DBSCAN clustering
7)  FLU cluster center -> ENU global coordinate transform
8)  Bounding box estimation + publish (ENU frame)
9)  Multi-object tracking (Kalman Filter, ENU frame)
10) Velocity estimation
11) Dynamic/static classification

Coordinate Transform Chain:
  FLU (sensor/body-front-left-up)
    -> FRD (body-front-right-down)
    -> NED (world-north-east-down, via quaternion DCM)
    -> NED_global (add vehicle NED position)
    -> ENU (world-east-north-up)

Enhanced Logging:
- Coordinate ranges & unit detection
- Wall removal count & plane normals
- Cluster coordinates in FLU and ENU
- Track velocity & classification (ENU)
"""

import time
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

# PX4 message types
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXEL_SIZE             = 0.03    # metres
SOR_NEIGHBOURS         = 20
SOR_STD_RATIO          = 2.0
GROUND_DIST_THRESH     = 0.05
GROUND_RANSAC_N        = 3
GROUND_ITERATIONS      = 500
WALL_DIST_THRESH       = 0.08
WALL_ITERATIONS        = 300
WALL_PASSES            = 3       # max number of wall planes to remove
DBSCAN_EPS             = 0.4
DBSCAN_MIN_PTS         = 10
CLUSTER_MIN_SIZE       = 5
TRACK_GATE_DIST        = 2.0     # metres – max association distance
TRACK_MAX_MISSED       = 5
SPEED_THRESHOLD        = 0.15    # m/s – dynamic vs static
MARKER_LIFETIME_NS     = 500_000_000
WALL_NORMAL_THRESHOLD  = 0.3     # |z_component| < 0.3 indicates vertical wall
wall_inliers_threshold = 300     # 增加最小墙面尺寸约束（防小平面误删）


# ---------------------------------------------------------------------------
# Helper – vectorised XYZ+RGB -> PointCloud2
# ---------------------------------------------------------------------------

def xyzrgb_to_pc2(
    points: np.ndarray,
    colors: np.ndarray,
    frame_id: str,
    stamp,
) -> PointCloud2:
    """
    Vectorised conversion.  `points` (N,3) float32, `colors` (N,3) float32 [0,1].
    """
    header = Header()
    header.frame_id = frame_id
    header.stamp    = stamp

    cols_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint32)
    packed  = (cols_u8[:, 0] << 16) | (cols_u8[:, 1] << 8) | cols_u8[:, 2]
    rgb_f   = packed.view(np.float32)

    cloud_arr = np.column_stack([
        points[:, 0].astype(np.float32),
        points[:, 1].astype(np.float32),
        points[:, 2].astype(np.float32),
        rgb_f,
    ])

    fields = [
        PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    return point_cloud2.create_cloud(header, fields, cloud_arr.tolist())


# ---------------------------------------------------------------------------
# Kalman Track  (operates in ENU global frame)
# ---------------------------------------------------------------------------

class Track:
    """
    Constant-velocity Kalman filter track.
    State: [x, y, z, vx, vy, vz]  — all in global ENU frame.
    """

    def __init__(self, track_id: int, position: np.ndarray, timestamp: float) -> None:
        self.id          = track_id
        self.x           = np.zeros(6, dtype=np.float64)
        self.x[:3]       = position
        self.P           = np.eye(6) * 0.1
        self.last_update = timestamp
        self.age         = 0
        self.missed      = 0
        self.hit_count   = 0

    def predict(self, dt: float) -> None:
        F      = np.eye(6)
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt

        Q         = np.eye(6) * 0.01
        Q[3,3]    = Q[4,4] = Q[5,5] = 0.1

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray) -> None:
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)

        R = np.eye(3) * 0.05

        innov = z - H @ self.x
        S     = H @ self.P @ H.T + R
        K     = self.P @ H.T @ np.linalg.inv(S)

        self.x         = self.x + K @ innov
        self.P         = (np.eye(6) - K @ H) @ self.P
        self.missed    = 0
        self.age      += 1
        self.hit_count += 1
        self.last_update = time.time()

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[3:6]))

    @property
    def classification(self) -> str:
        return "DYNAMIC" if self.speed > SPEED_THRESHOLD else "STATIC"


# ---------------------------------------------------------------------------
# Multi-Object Tracker  (greedy nearest-neighbour, ENU frame)
# ---------------------------------------------------------------------------

class MultiObjectTracker:

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.next_id = 0

    def step(
        self,
        detections: list[np.ndarray],
        dt: float,
    ) -> list[Track]:
        """
        detections: list of ENU global positions (np.ndarray shape (3,))
        """
        for t in self.tracks:
            t.predict(dt)

        assigned_track_ids: set[int] = set()

        for det in detections:
            best_track = None
            best_dist  = TRACK_GATE_DIST

            for t in self.tracks:
                if t.id in assigned_track_ids:
                    continue
                dist = float(np.linalg.norm(t.position - det))
                if dist < best_dist:
                    best_dist  = dist
                    best_track = t

            if best_track is not None:
                best_track.update(det)
                assigned_track_ids.add(best_track.id)
            else:
                new_track = Track(self.next_id, det, time.time())
                self.tracks.append(new_track)
                assigned_track_ids.add(self.next_id)
                self.next_id += 1

        for t in self.tracks:
            if t.id not in assigned_track_ids:
                t.missed += 1

        self.tracks = [t for t in self.tracks if t.missed < TRACK_MAX_MISSED]

        return self.tracks


# ---------------------------------------------------------------------------
# Main ROS2 Node
# ---------------------------------------------------------------------------

class IndustrialLidarNode(Node):

    def __init__(self) -> None:
        super().__init__("industrial_lidar_perception")

        # ── LiDAR QoS ────────────────────────────────────────────────────
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── PX4 QoS (BEST_EFFORT + VOLATILE) ─────────────────────────────
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── LiDAR subscription ────────────────────────────────────────────
        self.sub = self.create_subscription(
            PointCloud2,
            "/lidar_points/points",
            self._callback,
            lidar_qos,
        )

        # ── PX4 state subscriptions ───────────────────────────────────────
        self.localpos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback,
            px4_qos,
        )
        self.att_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            px4_qos,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_cloud  = self.create_publisher(PointCloud2, "/lidar/processed_points",  lidar_qos)
        self.pub_bbox   = self.create_publisher(MarkerArray, "/lidar/cluster_bboxes",     lidar_qos)
        self.pub_tracks = self.create_publisher(MarkerArray, "/lidar/tracks",             lidar_qos)
        self.pub_raw    = self.create_publisher(PointCloud2, "/debug/raw",                lidar_qos)
        self.pub_voxel  = self.create_publisher(PointCloud2, "/debug/voxel",              lidar_qos)
        self.pub_sor    = self.create_publisher(PointCloud2, "/debug/sor",                lidar_qos)
        self.pub_ground = self.create_publisher(PointCloud2, "/debug/ground_removed",     lidar_qos)
        self.pub_wall   = self.create_publisher(PointCloud2, "/debug/wall_removed",       lidar_qos)

        # ── Tracker & state ───────────────────────────────────────────────
        self.tracker   = MultiObjectTracker()
        self.last_time: Optional[float] = None
        self.frame_count = 0

        # ── PX4 vehicle state cache ───────────────────────────────────────
        self._vehicle_local_pos: Optional[VehicleLocalPosition] = None
        self._vehicle_attitude:  Optional[VehicleAttitude]      = None

        self.get_logger().info("Industrial LiDAR Pipeline Ready (ENU global frame enabled)")

    # ═══════════════════════════════════════════════════════════════════════
    #  PX4 state callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self._vehicle_local_pos = msg

    def vehicle_attitude_callback(self, msg: VehicleAttitude) -> None:
        self._vehicle_attitude = msg

    # ═══════════════════════════════════════════════════════════════════════
    #  坐标变换 – FLU → FRD → NED_body → NED_global → ENU
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def flu_to_frd(v: np.ndarray) -> np.ndarray:
        """
        FLU (Front-Left-Up) → FRD (Front-Right-Down)
        Flip Y and Z axes.
        """
        return np.array([v[0], -v[1], -v[2]], dtype=np.float64)

    @staticmethod
    def body_to_ned(v_frd: np.ndarray, att: VehicleAttitude) -> np.ndarray:
        """
        Rotate a FRD body-frame vector to NED world frame using
        the quaternion from VehicleAttitude.
        att.q = [w, x, y, z]
        """
        #w, x, y, z = float(att.q[0]), float(att.q[1]), float(att.q[2]), float(att.q[3])
        w, x, y, z = 0.7071, 0, 0, 0.7071
        dcm = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        return dcm @ v_frd

    @staticmethod
    def add_vehicle_pos(ned_rel: np.ndarray, pos: VehicleLocalPosition) -> np.ndarray:
        """
        Add vehicle NED local position to a relative NED vector.
        VehicleLocalPosition: x=North, y=East, z=Down
        """
        return np.array([pos.x, pos.y, pos.z], dtype=np.float64) + ned_rel

    @staticmethod
    def ned_to_enu(ned: np.ndarray) -> np.ndarray:
        """
        NED (North-East-Down) → ENU (East-North-Up)
        """
        return np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)

    def flu_center_to_enu(self, center_flu: np.ndarray) -> Optional[np.ndarray]:
        """
        Full transform chain:
          FLU -> FRD -> NED_body_relative -> NED_global -> ENU_global

        Returns None if PX4 state is not yet available.
        """
        if self._vehicle_attitude is None or self._vehicle_local_pos is None:
            return None

        # Step 1: FLU -> FRD
        v_frd = self.flu_to_frd(center_flu)

        # Step 2: FRD body frame -> NED world frame (relative to vehicle)
        v_ned_rel = self.body_to_ned(v_frd, self._vehicle_attitude)

        # Step 3: Add vehicle NED position -> global NED
        v_ned_global = self.add_vehicle_pos(v_ned_rel, self._vehicle_local_pos)

        # Step 4: NED -> ENU
        v_enu = self.ned_to_enu(v_ned_global)

        return v_enu

    # ═══════════════════════════════════════════════════════════════════════
    #  Open3D helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pcd

    @staticmethod
    def _remove_dominant_plane(
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray, int]:
        """Returns (remaining_pcd, plane_coefficients, inlier_count)."""
        plane, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        remaining = pcd.select_by_index(inliers, invert=True)
        return remaining, np.asarray(plane), len(inliers)

    @staticmethod
    def _classify_plane(plane_coeff: np.ndarray) -> str:
        """
        Classify plane as ground / wall / oblique using angle between
        plane normal and world Z axis.
        plane: ax + by + cz + d = 0
        """
        GROUND_ANGLE_THRESHOLD = 20.0
        WALL_ANGLE_THRESHOLD   = 70.0
        a, b, c, _ = plane_coeff

        normal = np.array([a, b, c], dtype=np.float64)
        normal /= (np.linalg.norm(normal) + 1e-9)

        z_axis    = np.array([0.0, 0.0, 1.0])
        cos_theta = np.clip(np.dot(normal, z_axis), -1.0, 1.0)
        angle     = np.degrees(np.arccos(abs(cos_theta)))

        if angle < GROUND_ANGLE_THRESHOLD:
            return "HORIZONTAL (Ground)"
        elif angle > WALL_ANGLE_THRESHOLD:
            return "VERTICAL (Wall)"
        else:
            return "OBLIQUE"

    # ═══════════════════════════════════════════════════════════════════════
    #  Main LiDAR callback
    # ═══════════════════════════════════════════════════════════════════════

    def _callback(self, msg: PointCloud2) -> None:  # noqa: C901
        self.frame_count += 1
        t0  = time.time()
        now = t0

        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"FRAME #{self.frame_count} START")
        self.get_logger().info(f"{'='*80}")

        # Check PX4 state availability
        px4_ready = (
            self._vehicle_attitude is not None
            and self._vehicle_local_pos is not None
        )
        if not px4_ready:
            self.get_logger().warning(
                "[PX4] Vehicle attitude or local position not yet received. "
                "ENU transform unavailable – FLU coordinates will be used as fallback."
            )

        # ================================================================
        # STEP 1: Parse Point Cloud
        # ================================================================
        raw = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )

        if raw.dtype.names:
            pts = np.column_stack([raw[f] for f in ("x", "y", "z")]).astype(np.float32)
        else:
            pts = raw.reshape(-1, 3).astype(np.float32)

        if len(pts) == 0:
            self.get_logger().warning("Empty point cloud received – skipping frame.")
            return

        # ================================================================
        # STEP 1B: Coordinate Range & Unit Detection
        # ================================================================
        pts_min, pts_max = pts.min(axis=0), pts.max(axis=0)
        pts_range  = pts_max - pts_min
        pts_extent = float(np.max(pts_range))

        self.get_logger().info(f"[PARSE] Raw point cloud: {len(pts)} points")
        self.get_logger().info(
            f"  X range: [{pts_min[0]:8.3f}, {pts_max[0]:8.3f}] (span: {pts_range[0]:8.3f})"
        )
        self.get_logger().info(
            f"  Y range: [{pts_min[1]:8.3f}, {pts_max[1]:8.3f}] (span: {pts_range[1]:8.3f})"
        )
        self.get_logger().info(
            f"  Z range: [{pts_min[2]:8.3f}, {pts_max[2]:8.3f}] (span: {pts_range[2]:8.3f})"
        )
        self.get_logger().info(f"  Max extent: {pts_extent:.3f} m")

        if 0.0 < pts_extent < 1.0:
            self.get_logger().warning(
                f"[UNIT_DETECT] Extent {pts_extent:.3f} m < 1.0 m – "
                "likely MILLIMETRE data. Converting to METRES..."
            )
            pts = pts / 1000.0
            pts_min, pts_max = pts.min(axis=0), pts.max(axis=0)
            pts_range  = pts_max - pts_min
            pts_extent = float(np.max(pts_range))
            self.get_logger().info(
                f"[UNIT_CONVERT] After conversion: extent = {pts_extent:.3f} m"
            )

        # ================================================================
        # STEP 1C: Raw Point Filtering
        # ================================================================
        mask_valid   = np.isfinite(pts).all(axis=1)
        mask_nonzero = np.linalg.norm(pts, axis=1) > 1e-3
        dist_xy      = np.linalg.norm(pts[:, :2], axis=1)
        mask_self    = dist_xy > 0.3           # remove self-returns (sensor radius)
        mask_range   = np.linalg.norm(pts, axis=1) < 15.0
        mask_front   = pts[:, 0] > 0           # FLU: forward = +X

        mask         = mask_valid & mask_nonzero & mask_self & mask_range & mask_front
        pts          = pts[mask]

        self.get_logger().info(
            f"[RAW_FILTER] After filtering: {len(pts)} points remain"
        )

        if len(pts) == 0:
            self.get_logger().warning("[RAW_FILTER] No points after raw filter – skipping frame.")
            return

        # ================================================================
        # STEP 2: Voxel Downsampling (Industrial Adaptive)
        # ================================================================
        pcd = self._to_pcd(pts)

        x_span     = np.max(pts[:, 0]) - np.min(pts[:, 0])
        y_span     = np.max(pts[:, 1]) - np.min(pts[:, 1])
        pts_extent = max(x_span, y_span)

        adaptive_voxel = float(np.clip(pts_extent * 0.003, 0.03, 0.12))

        try:
            pcd   = pcd.voxel_down_sample(adaptive_voxel)
            ratio = len(pcd.points) / len(pts)
            self.get_logger().info(
                f"[VOXEL] {len(pts):6d} → {len(pcd.points):6d} "
                f"(voxel={adaptive_voxel:.3f} m | extent={pts_extent:.1f} m | ratio={ratio*100:.1f}%)"
            )
        except RuntimeError as e:
            self.get_logger().error(f"[VOXEL_ERROR] {e}, fallback to 0.05")
            pcd = pcd.voxel_down_sample(0.05)

        # ================================================================
        # STEP 3: Statistical Outlier Removal (SOR)
        # ================================================================
        pts_before_sor = len(pcd.points)
        try:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=SOR_NEIGHBOURS,
                std_ratio=SOR_STD_RATIO,
            )
            removed = pts_before_sor - len(pcd.points)
            self.get_logger().info(
                f"[SOR] Removed {removed:6d} outliers "
                f"({removed/pts_before_sor*100:.1f}%) | {len(pcd.points):6d} points remain"
            )
        except RuntimeError as e:
            self.get_logger().warning(f"[SOR_warning] SOR failed: {e}. Continuing without SOR.")

        # ================================================================
        # STEP 4: Ground Plane Removal
        # ================================================================
        pts_before_ground = len(pcd.points)
        ground_inliers    = 0
        try:
            pcd, ground_plane, ground_inliers = self._remove_dominant_plane(
                pcd,
                distance_threshold=GROUND_DIST_THRESH,
                ransac_n=GROUND_RANSAC_N,
                num_iterations=GROUND_ITERATIONS,
            )
            a, b, c, d = ground_plane
            normal      = np.array([a, b, c]) / (np.linalg.norm([a, b, c]) + 1e-9)
            plane_type  = self._classify_plane(ground_plane)

            self.get_logger().info(
                f"[GROUND] Removed {ground_inliers:6d} points "
                f"({ground_inliers/pts_before_ground*100:.1f}%)"
            )
            self.get_logger().info(
                f"  Plane: {a:7.4f}x + {b:7.4f}y + {c:7.4f}z + {d:7.4f} = 0"
            )
            self.get_logger().info(
                f"  Normal: [{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}] "
                f"(type: {plane_type}) | Remaining: {len(pcd.points):6d}"
            )
        except RuntimeError as e:
            self.get_logger().warning(f"[GROUND_warning] Ground removal failed: {e}.")

        # ================================================================
        # STEP 5: Iterative Wall Removal
        # ================================================================
        wall_count   = 0
        wall_details = []

        for wall_pass in range(WALL_PASSES):
            if len(pcd.points) < 50:
                self.get_logger().info(
                    f"[WALL] Stopping: only {len(pcd.points)} points remain (< 50)"
                )
                break
            try:
                pcd, wall_plane, wall_inliers = self._remove_dominant_plane(
                    pcd,
                    distance_threshold=WALL_DIST_THRESH,
                    ransac_n=3,
                    num_iterations=WALL_ITERATIONS,
                )
                if wall_inliers < 10:
                    self.get_logger().info(
                        f"[WALL] Pass {wall_pass+1}: insufficient inliers ({wall_inliers}), stopping."
                    )
                    break

                a, b, c, d = wall_plane
                normal     = np.array([a, b, c]) / (np.linalg.norm([a, b, c]) + 1e-9)
                plane_type = self._classify_plane(wall_plane)

                wall_count += 1
                wall_details.append({
                    "pass":    wall_pass + 1,
                    "inliers": wall_inliers,
                    "normal":  normal,
                    "plane":   wall_plane,
                    "type":    plane_type,
                })

                self.get_logger().info(
                    f"[WALL] Pass {wall_pass+1}: Removed {wall_inliers:6d} points | "
                    f"Type: {plane_type} | Remaining: {len(pcd.points):6d}"
                )
            except RuntimeError as e:
                self.get_logger().debug(
                    f"[WALL_DEBUG] Wall pass {wall_pass+1} failed: {e}. Stopping."
                )
                break

        self.get_logger().info(f"[WALL_SUMMARY] Total walls removed: {wall_count}")

        # ================================================================
        # STEP 6: Safety guard
        # ================================================================
        remain_pts = np.asarray(pcd.points, dtype=np.float32)

        if len(remain_pts) < DBSCAN_MIN_PTS:
            self.get_logger().warning(
                f"[SAFETY] Too few points ({len(remain_pts)} < {DBSCAN_MIN_PTS}) – "
                "skipping DBSCAN."
            )
            return

        # ================================================================
        # STEP 7: DBSCAN Clustering
        # ================================================================
        labels = np.asarray(
            pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS),
            dtype=np.int32,
        )

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        valid_mask   = counts >= CLUSTER_MIN_SIZE
        valid_labels = unique_labels[valid_mask]

        # clusters_enu: list of ENU positions used for tracking
        clusters_enu: list[np.ndarray] = []
        bbox_markers = MarkerArray()

        self.get_logger().info(
            f"[CLUSTER] DBSCAN: {len(unique_labels)} potential, "
            f"{len(valid_labels)} valid (≥{CLUSTER_MIN_SIZE} pts)"
        )

        for idx, lab in enumerate(valid_labels, 1):
            cluster_pts = remain_pts[labels == lab]

            # ── Cluster center in FLU (sensor frame) ──────────────────
            center_flu = cluster_pts.mean(axis=0).astype(np.float64)

            mn   = cluster_pts.min(axis=0)
            mx   = cluster_pts.max(axis=0)
            dims = mx - mn

            # ── Transform FLU -> ENU global ────────────────────────────
            center_enu = self.flu_center_to_enu(center_flu)
            enu_available = center_enu is not None

            if enu_available:
                clusters_enu.append(center_enu)
                log_pos_str = (
                    f"FLU=({center_flu[0]:7.3f}, {center_flu[1]:7.3f}, {center_flu[2]:7.3f}) | "
                    f"ENU=({center_enu[0]:7.3f}, {center_enu[1]:7.3f}, {center_enu[2]:7.3f})"
                )
                # Use ENU for marker position
                marker_x, marker_y, marker_z = float(center_enu[0]), float(center_enu[1]), float(center_enu[2])
                marker_frame = "map"   # ENU global frame
            else:
                # Fallback: use FLU if PX4 state unavailable
                clusters_enu.append(center_flu.copy())
                log_pos_str = (
                    f"FLU=({center_flu[0]:7.3f}, {center_flu[1]:7.3f}, {center_flu[2]:7.3f}) | "
                    f"ENU=N/A (PX4 not ready)"
                )
                marker_x, marker_y, marker_z = float(center_flu[0]), float(center_flu[1]), float(center_flu[2])
                marker_frame = msg.header.frame_id

            self.get_logger().info(
                f"[CLUSTER_{idx}] ID={int(lab)} | Points={len(cluster_pts):5d} | {log_pos_str}"
            )
            self.get_logger().info(
                f"            Bounds: X[{mn[0]:7.3f},{mx[0]:7.3f}] "
                f"Y[{mn[1]:7.3f},{mx[1]:7.3f}] Z[{mn[2]:7.3f},{mx[2]:7.3f}]"
            )
            self.get_logger().info(
                f"            Dims: {dims[0]:.3f}×{dims[1]:.3f}×{dims[2]:.3f} m | "
                f"Vol: {np.prod(dims):.4f} m³"
            )

            # Bounding box marker (published in ENU / map frame when available)
            bbox_m                    = Marker()
            bbox_m.header.frame_id    = marker_frame
            bbox_m.header.stamp       = msg.header.stamp
            bbox_m.ns                 = "cluster_bbox"
            bbox_m.id                 = int(lab)
            bbox_m.type               = Marker.CUBE
            bbox_m.action             = Marker.ADD
            bbox_m.pose.position.x    = marker_x
            bbox_m.pose.position.y    = marker_y
            bbox_m.pose.position.z    = marker_z
            bbox_m.pose.orientation.w = 1.0
            bbox_m.scale.x            = float(max(dims[0], 0.05))
            bbox_m.scale.y            = float(max(dims[1], 0.05))
            bbox_m.scale.z            = float(max(dims[2], 0.05))
            bbox_m.color.r            = 0.9
            bbox_m.color.g            = 0.9
            bbox_m.color.b            = 0.1
            bbox_m.color.a            = 0.4
            bbox_m.lifetime           = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)
            bbox_markers.markers.append(bbox_m)

        self.pub_bbox.publish(bbox_markers)

        # ================================================================
        # STEP 8: Publish Processed Cloud
        # ================================================================
        if len(remain_pts) > 0:
            colors = np.tile([0.2, 0.8, 1.0], (len(remain_pts), 1))
            self.pub_cloud.publish(
                xyzrgb_to_pc2(remain_pts, colors, msg.header.frame_id, msg.header.stamp)
            )
            self.get_logger().info(f"[PUBLISH] Published {len(remain_pts)} processed points")

        # ================================================================
        # STEP 9: Multi-Object Tracking (ENU global frame)
        # ================================================================
        dt             = (now - self.last_time) if self.last_time is not None else 0.1
        self.last_time = now

        tracks     = self.tracker.step(clusters_enu, dt)
        track_msgs = MarkerArray()

        self.get_logger().info(
            f"[TRACKING] dt={dt:.4f}s | Active tracks: {len(tracks)}"
        )

        track_marker_frame = "map" if px4_ready else msg.header.frame_id

        for track_idx, t in enumerate(tracks, 1):
            pos            = t.position   # ENU global (or FLU fallback)
            vel            = t.velocity
            speed          = t.speed
            classification = t.classification

            frame_label = "ENU" if px4_ready else "FLU(fallback)"
            self.get_logger().info(
                f"[TRACK_{track_idx}] ID={t.id:3d} | Age={t.age:3d} | Hits={t.hit_count:3d} | "
                f"Pos[{frame_label}]=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m"
            )
            self.get_logger().info(
                f"            Vel=({vel[0]:7.4f}, {vel[1]:7.4f}, {vel[2]:7.4f}) m/s | "
                f"Speed={speed:7.4f} m/s | [{classification}]"
            )

            m                    = Marker()
            m.header.frame_id    = track_marker_frame
            m.header.stamp       = msg.header.stamp
            m.ns                 = "track"
            m.id                 = t.id
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = float(pos[0])
            m.pose.position.y    = float(pos[1])
            m.pose.position.z    = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x            = m.scale.y = m.scale.z = 0.4
            m.color.a            = 0.9
            m.lifetime           = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)

            if classification == "DYNAMIC":
                m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.2   # Red
            else:
                m.color.r, m.color.g, m.color.b = 0.2, 0.8, 0.2   # Green

            track_msgs.markers.append(m)

        self.pub_tracks.publish(track_msgs)

        # ================================================================
        # STEP 10: Summary & Timing
        # ================================================================
        elapsed_ms = (time.time() - t0) * 1000
        total_wall_pts = sum(d['inliers'] for d in wall_details)

        self.get_logger().info(f"{'='*80}")
        self.get_logger().info(f"FRAME #{self.frame_count} SUMMARY")
        self.get_logger().info(f"{'='*80}")
        self.get_logger().info(f"  Raw points:           {len(pts):8d}")
        self.get_logger().info(f"  Ground removed:       {ground_inliers:8d}")
        self.get_logger().info(
            f"  Walls removed:        {wall_count:8d} (total pts: {total_wall_pts:d})"
        )
        self.get_logger().info(f"  Final points:         {len(remain_pts):8d}")
        self.get_logger().info(f"  Clusters detected:    {len(clusters_enu):8d}")
        self.get_logger().info(f"  Active tracks:        {len(tracks):8d}")
        self.get_logger().info(f"  PX4 ENU available:    {'YES' if px4_ready else 'NO (FLU fallback)'}")
        self.get_logger().info(f"  Frame processing:     {elapsed_ms:8.2f} ms")
        self.get_logger().info(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    node = IndustrialLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
