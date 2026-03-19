#!/usr/bin/env python3
"""
Industrial LiDAR Perception Pipeline (ROS2 + Open3D)

Pipeline:
1)  PointCloud2 -> numpy
2)  Voxel downsample (adaptive)
3)  SOR filter
4)  Ground plane removal (RANSAC)
5)  Wall removal (iterative RANSAC)
6)  DBSCAN clustering
7)  Bounding box estimation + publish
8)  Multi-object tracking (Kalman Filter)
9)  Velocity estimation
10) Dynamic/static classification

Enhanced Logging:
- Coordinate ranges & unit detection
- Wall removal count & plane normals
- Cluster coordinates & dimensions
- Track velocity & classification
"""

import time
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXEL_SIZE          = 0.03   # metres
SOR_NEIGHBOURS      = 20
SOR_STD_RATIO       = 2.0
GROUND_DIST_THRESH  = 0.05
GROUND_RANSAC_N     = 3
GROUND_ITERATIONS   = 500
WALL_DIST_THRESH    = 0.08
WALL_ITERATIONS     = 300
WALL_PASSES         = 3     # max number of wall planes to remove
DBSCAN_EPS          = 0.4
DBSCAN_MIN_PTS      = 10
CLUSTER_MIN_SIZE    = 5
TRACK_GATE_DIST     = 2.0    # metres – max association distance
TRACK_MAX_MISSED    = 5
SPEED_THRESHOLD     = 0.15    # m/s – dynamic vs static
MARKER_LIFETIME_NS  = 500_000_000
WALL_NORMAL_THRESHOLD = 0.3  # |z_component| < 0.3 indicates vertical wall
wall_inliers_threshold = 300 # 增加最小墙面尺寸约束（防小平面误删）
BBOX_TEXT_Z_OFFSET = 0.35
TRACK_TEXT_Z_OFFSET = 0.45
VELOCITY_ARROW_SCALE = 0.35
DEBUG_PUBLISH_EVERY_N_FRAMES = 5
 
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


def solid_color(count: int, rgb: tuple[float, float, float]) -> np.ndarray:
    """Create an (N, 3) RGB array for PointCloud2 coloring."""
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.tile(np.asarray(rgb, dtype=np.float32), (count, 1))


# ---------------------------------------------------------------------------
# Kalman Track
# ---------------------------------------------------------------------------

class Track:
    """
    Constant-velocity Kalman filter track.
    State: [x, y, z, vx, vy, vz]
    """

    def __init__(self, track_id: int, position: np.ndarray, timestamp: float) -> None:
        self.id          = track_id
        self.x           = np.zeros(6, dtype=np.float64)
        self.x[:3]       = position
        self.P           = np.eye(6) * 0.1
        self.last_update = timestamp
        self.age         = 0
        self.missed      = 0
        self.hit_count   = 0  # consecutive successful updates

    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        F      = np.eye(6)
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt

        # Process noise – higher on velocity states
        Q         = np.eye(6) * 0.01
        Q[3,3]    = Q[4,4] = Q[5,5] = 0.1

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    def update(self, z: np.ndarray) -> None:
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)

        R = np.eye(3) * 0.05

        innov = z - H @ self.x
        S     = H @ self.P @ H.T + R
        K     = self.P @ H.T @ np.linalg.inv(S)

        self.x        = self.x + K @ innov
        self.P        = (np.eye(6) - K @ H) @ self.P
        self.missed   = 0
        self.age     += 1
        self.hit_count += 1
        self.last_update = time.time()

    # ------------------------------------------------------------------
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
        """Dynamic or Static"""
        return "DYNAMIC" if self.speed > SPEED_THRESHOLD else "STATIC"


# ---------------------------------------------------------------------------
# Multi-Object Tracker  (greedy nearest-neighbour)
# ---------------------------------------------------------------------------

class MultiObjectTracker:

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.next_id = 0

    # ------------------------------------------------------------------
    def step(
        self,
        detections: list[np.ndarray],
        dt: float,
    ) -> list[Track]:
        """
        1. Predict all tracks forward by `dt`.
        2. Greedily match detections to tracks (nearest unassigned).
        3. Spawn new tracks for unmatched detections.
        4. Increment `missed` counter for unmatched tracks and prune stale.
        """

        # --- predict -------------------------------------------------------
        for t in self.tracks:
            t.predict(dt)

        assigned_track_ids: set[int] = set()

        # Sort detections once; iterate in deterministic order
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

        # --- increment missed / prune --------------------------------------
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
        self.declare_parameter("rviz_debug_publish_every_n_frames", DEBUG_PUBLISH_EVERY_N_FRAMES)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(
            PointCloud2,
            "/lidar_points/points",
            self._callback,
            qos,
        )

        self.pub_cloud  = self.create_publisher(PointCloud2,   "/lidar/processed_points", qos)
        self.pub_bbox   = self.create_publisher(MarkerArray,   "/lidar/cluster_bboxes",   qos)
        self.pub_tracks = self.create_publisher(MarkerArray,   "/lidar/tracks",            qos)
        self.pub_raw      = self.create_publisher(PointCloud2, "/debug/raw_points", qos)
        self.pub_voxel    = self.create_publisher(PointCloud2, "/debug/voxel_points", qos)
        self.pub_sor      = self.create_publisher(PointCloud2, "/debug/sor_points", qos)
        self.pub_ground   = self.create_publisher(PointCloud2, "/debug/ground_removed_points", qos)
        self.pub_wall     = self.create_publisher(PointCloud2, "/debug/wall_removed_points", qos)

        self.tracker   = MultiObjectTracker()
        self.last_time: Optional[float] = None
        self.frame_count = 0
        self.debug_publish_every_n_frames = max(
            1,
            int(self.get_parameter("rviz_debug_publish_every_n_frames").value),
        )

        self.get_logger().info(
            "Industrial LiDAR Pipeline Ready | "
            f"RViz debug clouds every {self.debug_publish_every_n_frames} frame(s)"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

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
        """
        Returns (remaining_pcd, plane_coefficients, inlier_count).
        """
        plane, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        remaining = pcd.select_by_index(inliers, invert=True)
        return remaining, np.asarray(plane), len(inliers)

    def _publish_debug_cloud(
        self,
        publisher,
        points: np.ndarray,
        frame_id: str,
        stamp,
        color: tuple[float, float, float],
    ) -> None:
        """Publish a colored debug cloud for RViz2 visualisation."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return
        publisher.publish(
            xyzrgb_to_pc2(points, solid_color(len(points), color), frame_id, stamp)
        )

    def _should_publish_debug_clouds(self) -> bool:
        """Throttle intermediate RViz2 point cloud topics to reduce queue pressure."""
        return (self.frame_count % self.debug_publish_every_n_frames) == 0

    #@staticmethod
    # def _classify_plane(plane_coeff: np.ndarray) -> str:
    #     """
    #     Classify plane as horizontal (ground) or vertical (wall) based on normal.
    #     """
    #     a, b, c, _ = plane_coeff
    #     normal = np.array([a, b, c])
    #     normal_norm = np.linalg.norm(normal) + 1e-9
    #     normal = normal / normal_norm

    #     z_component = abs(normal[2])

    #     if z_component > 0.7:
    #         return "HORIZONTAL (Ground)"
    #     elif z_component < WALL_NORMAL_THRESHOLD:
    #         return "VERTICAL (Wall)"
    #     else:
    #         return "OBLIQUE"


    @staticmethod
    def _classify_plane(plane_coeff: np.ndarray) -> str:
        """
        Classify plane as ground / wall / oblique using angle between
        plane normal and world Z axis.

        plane: ax + by + cz + d = 0
        normal = [a,b,c]
        """
        GROUND_ANGLE_THRESHOLD = 20.0   # degrees
        WALL_ANGLE_THRESHOLD   = 70.0   # degrees
        a, b, c, _ = plane_coeff

        normal = np.array([a, b, c], dtype=np.float64)
        normal /= (np.linalg.norm(normal) + 1e-9)

        z_axis = np.array([0.0, 0.0, 1.0])

        # angle between normal and z-axis
        cos_theta = np.dot(normal, z_axis)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle = np.degrees(np.arccos(abs(cos_theta)))

        if angle < GROUND_ANGLE_THRESHOLD:
            return "HORIZONTAL (Ground)"

        elif angle > WALL_ANGLE_THRESHOLD:
            return "VERTICAL (Wall)"

        else:
            return "OBLIQUE"

    # -----------------------------------------------------------------------
    # Callback
    # -----------------------------------------------------------------------

    def _callback(self, msg: PointCloud2) -> None:  # noqa: C901
        self.frame_count += 1
        t0  = time.time()
        now = t0

        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"FRAME #{self.frame_count} START")
        self.get_logger().info(f"{'='*80}")

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
            self.get_logger().warn("Empty point cloud received – skipping frame.")
            return

        publish_debug_clouds = self._should_publish_debug_clouds()
        if publish_debug_clouds:
            self._publish_debug_cloud(
                self.pub_raw,
                pts,
                msg.header.frame_id,
                msg.header.stamp,
                (0.8, 0.8, 0.8),
            )

        # ================================================================
        # STEP 1B: Diagnose Coordinate Range & Unit Detection
        # ================================================================
        pts_min, pts_max = pts.min(axis=0), pts.max(axis=0)
        pts_range = pts_max - pts_min
        pts_extent = float(np.max(pts_range))

        self.get_logger().info(
            f"[PARSE] Raw point cloud: {len(pts)} points"
        )
        self.get_logger().info(
            f"  X range: [{pts_min[0]:8.3f}, {pts_max[0]:8.3f}] (span: {pts_range[0]:8.3f})"
        )
        self.get_logger().info(
            f"  Y range: [{pts_min[1]:8.3f}, {pts_max[1]:8.3f}] (span: {pts_range[1]:8.3f})"
        )
        self.get_logger().info(
            f"  Z range: [{pts_min[2]:8.3f}, {pts_max[2]:8.3f}] (span: {pts_range[2]:8.3f})"
        )
        self.get_logger().info(
            f"  Max extent: {pts_extent:.3f} m"
        )

        # Auto-detect millimetre data and convert to metres
        unit_conversion_factor = 1.0
        if 0.0 < pts_extent < 1.0:
            self.get_logger().warn(
                f"[UNIT_DETECT] Point cloud extent {pts_extent:.3f} m < 1.0 m – "
                f"likely MILLIMETRE data. Converting to METRES..."
            )
            pts = pts / 1000.0
            unit_conversion_factor = 0.001
            pts_min, pts_max = pts.min(axis=0), pts.max(axis=0)
            pts_range = pts_max - pts_min
            pts_extent = float(np.max(pts_range))

            self.get_logger().info(
                f"[UNIT_CONVERT] After conversion: extent = {pts_extent:.3f} m"
                
            )

        # ================================================================
        # STEP 1C: Raw Point Filtering (INVALID + SELF MASK + RANGE)
        # ================================================================

        # ---------- remove NaN / Inf ----------
        mask_valid = np.isfinite(pts).all(axis=1)

        # ---------- remove zero points ----------
        mask_nonzero = np.linalg.norm(pts, axis=1) > 1e-3

        # ---------- remove points near sensor (self mask) ----------
        sensor_radius = 0.3  # UAV radius (meters)
        dist = np.linalg.norm(pts[:, :2], axis=1)
        mask_self = dist > sensor_radius

        # ---------- remove far noise ----------
        max_range = 15.0
        mask_range = np.linalg.norm(pts, axis=1) < max_range

        # ========== NEW: FILTER OUT UAV BACKSIDE POINTS ==========
        # Assumption: UAV's forward direction is X+ axis (global coordinate system)
        # We only keep points in front of UAV (X > 0)
        mask_front = pts[:, 0] > 0  # Only keep points with X > 0 (front of UAV)
       # ========================================================


        # ---------- combine ----------
        mask = mask_valid & mask_nonzero & mask_self & mask_range & mask_front

        pts_filtered = pts[mask]

        self.get_logger().info(
            f"[RAW_FILTER] {len(pts)} → {len(pts_filtered)} points "
            f"(removed {len(pts)-len(pts_filtered)})"
        )

        pts = pts_filtered

        # # ================================================================
        # # STEP 2: Voxel Downsampling (Adaptive)
        # # ================================================================
        
        # # ================================================================

        
        # pcd = self._to_pcd(pts)

        # # Adaptive voxel size: 1% of point cloud extent, clamped to [0.005, 0.5]
        # adaptive_voxel = np.clip(pts_extent * 0.01, 0.005, 0.5)

        # try:
        #     pcd = pcd.voxel_down_sample(adaptive_voxel)
        #     self.get_logger().info(
        #         f"[VOXEL] Downsampled {len(pts):6d} → {len(pcd.points):6d} points "
        #         f"(voxel_size={adaptive_voxel:.5f} m, ratio={len(pcd.points)/len(pts)*100:.1f}%)"
        #     )
        # except RuntimeError as e:
        #     self.get_logger().error(
        #         f"[VOXEL_ERROR] Voxel downsampling failed (voxel_size={adaptive_voxel}): {e}. "
        #         f"Retrying with 2x size..."
        #     )
        #     try:
        #         pcd = pcd.voxel_down_sample(adaptive_voxel * 2)
        #         self.get_logger().info(
        #             f"[VOXEL_RETRY] Success with 2x voxel size: {len(pcd.points):6d} points"
        #         )
        #     except RuntimeError as e2:
        #         self.get_logger().error(f"[VOXEL_FATAL] Fallback voxel also failed: {e2}. Skipping frame.")
        #         return

        # ================================================================
        # STEP 2: Voxel Downsampling (Industrial Adaptive)
        # ================================================================

        pcd = self._to_pcd(pts)

        # compute XY extent (robust)
        x_span = np.max(pts[:,0]) - np.min(pts[:,0])
        y_span = np.max(pts[:,1]) - np.min(pts[:,1])
        pts_extent = max(x_span, y_span)

        # industrial adaptive voxel
        adaptive_voxel = np.clip(pts_extent * 0.003, 0.03, 0.12)

        try:
            pcd = pcd.voxel_down_sample(adaptive_voxel)

            ratio = len(pcd.points) / len(pts)

            self.get_logger().info(
                f"[VOXEL] Downsampled {len(pts):6d} → {len(pcd.points):6d} "
                f"(voxel={adaptive_voxel:.3f} m | extent={pts_extent:.1f} m | ratio={ratio*100:.1f}%)"
            )

        except RuntimeError as e:

            self.get_logger().error(
                f"[VOXEL_ERROR] {e}, fallback to 0.05"
            )

            pcd = pcd.voxel_down_sample(0.05)

        voxel_pts = np.asarray(pcd.points, dtype=np.float32)
        if publish_debug_clouds:
            self._publish_debug_cloud(
                self.pub_voxel,
                voxel_pts,
                msg.header.frame_id,
                msg.header.stamp,
                (1.0, 0.7, 0.2),
            )







        # ================================================================
        # STEP 3: Statistical Outlier Removal (SOR)
        # ================================================================
        pts_before_sor = len(pcd.points)
        try:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=SOR_NEIGHBOURS,
                std_ratio=SOR_STD_RATIO,
            )
            pts_after_sor = len(pcd.points)
            removed = pts_before_sor - pts_after_sor
            self.get_logger().info(
                f"[SOR] Removed {removed:6d} outliers "
                f"({removed/pts_before_sor*100:.1f}%) | {pts_after_sor:6d} points remain"
            )
        except RuntimeError as e:
            self.get_logger().warn(f"[SOR_WARN] SOR filter failed: {e}. Continuing without SOR.")

        sor_pts = np.asarray(pcd.points, dtype=np.float32)
        if publish_debug_clouds:
            self._publish_debug_cloud(
                self.pub_sor,
                sor_pts,
                msg.header.frame_id,
                msg.header.stamp,
                (0.4, 1.0, 0.4),
            )

        # ================================================================
        # STEP 4: Ground Plane Removal
        # ================================================================
        pts_before_ground = len(pcd.points)
        ground_inliers = 0
        try:
            pcd, ground_plane, ground_inliers = self._remove_dominant_plane(
                pcd,
                distance_threshold=GROUND_DIST_THRESH,
                ransac_n=GROUND_RANSAC_N,
                num_iterations=GROUND_ITERATIONS,
            )

            a, b, c, d = ground_plane
            normal = np.array([a, b, c])
            normal_norm = np.linalg.norm(normal) + 1e-9
            normal = normal / normal_norm

            plane_type = self._classify_plane(ground_plane)

            self.get_logger().info(
                f"[GROUND] Removed {ground_inliers:6d} ground points "
                f"({ground_inliers/pts_before_ground*100:.1f}%)"
            )
            self.get_logger().info(
                f"  Plane equation: {a:7.4f}x + {b:7.4f}y + {c:7.4f}z + {d:7.4f} = 0"
            )
            self.get_logger().info(
                f"  Normal vector: [{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}] "
                f"(type: {plane_type})"
            )
            self.get_logger().info(
                f"  Remaining: {len(pcd.points):6d} points"
            )
        except RuntimeError as e:
            self.get_logger().warn(f"[GROUND_WARN] Ground removal failed: {e}. Skipping ground removal.")

        ground_removed_pts = np.asarray(pcd.points, dtype=np.float32)
        if publish_debug_clouds:
            self._publish_debug_cloud(
                self.pub_ground,
                ground_removed_pts,
                msg.header.frame_id,
                msg.header.stamp,
                (0.2, 0.9, 0.9),
            )

        # ================================================================
        # STEP 5: Iterative Wall Removal
        # ================================================================
        wall_count = 0
        wall_details = []

        for wall_pass in range(WALL_PASSES):
            if len(pcd.points) < 50:
                self.get_logger().info(
                    f"[WALL] Stopping wall removal: only {len(pcd.points)} points remain (< 50)"
                )
                break

            try:
                pcd, wall_plane, wall_inliers = self._remove_dominant_plane(
                    pcd,
                    distance_threshold=WALL_DIST_THRESH,
                    ransac_n=3,
                    num_iterations=WALL_ITERATIONS,
                )

                if wall_inliers < 10:  # too few points for a valid wall
                    self.get_logger().info(
                        f"[WALL] Pass {wall_pass+1}: insufficient inliers ({wall_inliers}), stopping."
                    )
                    break

                a, b, c, d = wall_plane
                normal = np.array([a, b, c])
                normal_norm = np.linalg.norm(normal) + 1e-9
                normal = normal / normal_norm

                plane_type = self._classify_plane(wall_plane)

                wall_count += 1
                wall_details.append({
                    "pass": wall_pass + 1,
                    "inliers": wall_inliers,
                    "normal": normal,
                    "plane": wall_plane,
                    "type": plane_type,
                })

                self.get_logger().info(
                    f"[WALL] Pass {wall_pass+1}: Removed {wall_inliers:6d} wall points "
                    f"({wall_inliers/max(len(pcd.points)+wall_inliers, 1)*100:.1f}%)"
                )
                self.get_logger().info(
                    f"  Plane: {a:7.4f}x + {b:7.4f}y + {c:7.4f}z + {d:7.4f} = 0"
                )
                self.get_logger().info(
                    f"  Normal: [{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}] "
                    f"(type: {plane_type})"
                )
                self.get_logger().info(
                    f"  Remaining: {len(pcd.points):6d} points"
                )

            except RuntimeError as e:
                self.get_logger().debug(f"[WALL_DEBUG] Wall removal pass {wall_pass+1} failed: {e}. Stopping.")
                break

        self.get_logger().info(
            f"[WALL_SUMMARY] Total walls removed: {wall_count}"
        )

        wall_removed_pts = np.asarray(pcd.points, dtype=np.float32)
        if publish_debug_clouds:
            self._publish_debug_cloud(
                self.pub_wall,
                wall_removed_pts,
                msg.header.frame_id,
                msg.header.stamp,
                (0.9, 0.2, 0.9),
            )

        # ================================================================
        # STEP 6: Safety guard
        # ================================================================
        remain_pts = np.asarray(pcd.points, dtype=np.float32)

        if len(remain_pts) < DBSCAN_MIN_PTS:
            self.get_logger().warn(
                f"[SAFETY] Too few points after filtering ({len(remain_pts)} < {DBSCAN_MIN_PTS}) "
                f"– skipping DBSCAN."
            )
            return

        # ================================================================
        # STEP 7: DBSCAN Clustering  (vectorised)
        # ================================================================
        labels = np.asarray(
            pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS),
            dtype=np.int32,
        )

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        valid_mask    = counts >= CLUSTER_MIN_SIZE
        valid_labels  = unique_labels[valid_mask]

        clusters: list[np.ndarray] = []
        bbox_markers              = MarkerArray()
        clear_bbox_markers = Marker()
        clear_bbox_markers.header.frame_id = msg.header.frame_id
        clear_bbox_markers.header.stamp = msg.header.stamp
        clear_bbox_markers.action = Marker.DELETEALL
        bbox_markers.markers.append(clear_bbox_markers)

        self.get_logger().info(
            f"[CLUSTER] DBSCAN found {len(unique_labels)} potential clusters, "
            f"{len(valid_labels)} valid (size >= {CLUSTER_MIN_SIZE})"
        )

        for idx, lab in enumerate(valid_labels, 1):
            cluster_pts = remain_pts[labels == lab]
            center      = cluster_pts.mean(axis=0)
            clusters.append(center)

            # Compute bounding box dimensions
            mn   = cluster_pts.min(axis=0)
            mx   = cluster_pts.max(axis=0)
            dims = mx - mn
            volume = np.prod(dims)

            # Log cluster details
            self.get_logger().info(
                f"[CLUSTER_{idx}] ID={int(lab)} | Points={len(cluster_pts):5d} | "
                f"Center=({center[0]:7.3f}, {center[1]:7.3f}, {center[2]:7.3f}) m"
            )
            self.get_logger().info(
                f"            Bounds: X[{mn[0]:7.3f}, {mx[0]:7.3f}] "
                f"Y[{mn[1]:7.3f}, {mx[1]:7.3f}] Z[{mn[2]:7.3f}, {mx[2]:7.3f}]"
            )
            self.get_logger().info(
                f"            Dimensions: {dims[0]:7.3f} × {dims[1]:7.3f} × {dims[2]:7.3f} m | "
                f"Volume: {volume:8.4f} m³"
            )

            # Bounding box marker
            bbox_m                       = Marker()
            bbox_m.header.frame_id       = msg.header.frame_id
            bbox_m.header.stamp          = msg.header.stamp
            bbox_m.id                    = int(lab)
            bbox_m.type                  = Marker.CUBE
            bbox_m.action                = Marker.ADD
            bbox_m.pose.position.x       = float(center[0])
            bbox_m.pose.position.y       = float(center[1])
            bbox_m.pose.position.z       = float(center[2])
            bbox_m.pose.orientation.w    = 1.0
            bbox_m.scale.x               = float(max(dims[0], 0.05))
            bbox_m.scale.y               = float(max(dims[1], 0.05))
            bbox_m.scale.z               = float(max(dims[2], 0.05))
            bbox_m.color.r               = 0.9
            bbox_m.color.g               = 0.9
            bbox_m.color.b               = 0.1
            bbox_m.color.a               = 0.4
            bbox_m.lifetime              = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)
            bbox_markers.markers.append(bbox_m)

            label_m                     = Marker()
            label_m.header.frame_id     = msg.header.frame_id
            label_m.header.stamp        = msg.header.stamp
            label_m.ns                  = "cluster_labels"
            label_m.id                  = 1000 + int(lab)
            label_m.type                = Marker.TEXT_VIEW_FACING
            label_m.action              = Marker.ADD
            label_m.pose.position.x     = float(center[0])
            label_m.pose.position.y     = float(center[1])
            label_m.pose.position.z     = float(mx[2] + BBOX_TEXT_Z_OFFSET)
            label_m.pose.orientation.w  = 1.0
            label_m.scale.z             = 0.28
            label_m.color.r             = 1.0
            label_m.color.g             = 1.0
            label_m.color.b             = 1.0
            label_m.color.a             = 0.95
            label_m.text                = (
                f"Cluster {int(lab)} | {len(cluster_pts)} pts\n"
                f"{dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m"
            )
            label_m.lifetime            = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)
            bbox_markers.markers.append(label_m)

        self.pub_bbox.publish(bbox_markers)

        # ================================================================
        # STEP 8: Publish Processed Cloud
        # ================================================================
        if len(remain_pts) > 0:
            colors = np.tile([0.2, 0.8, 1.0], (len(remain_pts), 1))
            self.pub_cloud.publish(
                xyzrgb_to_pc2(remain_pts, colors, msg.header.frame_id, msg.header.stamp)
            )
            self.get_logger().info(
                f"[PUBLISH] Published {len(remain_pts)} processed points"
            )

        # ================================================================
        # STEP 9: Multi-Object Tracking
        # ================================================================
        dt             = (now - self.last_time) if self.last_time is not None else 0.1
        self.last_time = now

        tracks     = self.tracker.step(clusters, dt)
        track_msgs = MarkerArray()
        clear_track_markers = Marker()
        clear_track_markers.header.frame_id = msg.header.frame_id
        clear_track_markers.header.stamp = msg.header.stamp
        clear_track_markers.action = Marker.DELETEALL
        track_msgs.markers.append(clear_track_markers)

        self.get_logger().info(
            f"[TRACKING] dt={dt:.4f}s | Active tracks: {len(tracks)}"
        )

        for track_idx, t in enumerate(tracks, 1):
            pos     = t.position
            vel     = t.velocity
            speed   = t.speed
            classification = t.classification

            self.get_logger().info(
                f"[TRACK_{track_idx}] ID={t.id:3d} | Age={t.age:3d} | Hits={t.hit_count:3d} | "
                f"Pos=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m"
            )
            self.get_logger().info(
                f"            Velocity: ({vel[0]:7.4f}, {vel[1]:7.4f}, {vel[2]:7.4f}) m/s | "
                f"Speed: {speed:7.4f} m/s | [{classification}]"
            )

            m                       = Marker()
            m.header.frame_id       = msg.header.frame_id
            m.header.stamp          = msg.header.stamp
            m.id                    = t.id
            m.type                  = Marker.SPHERE
            m.action                = Marker.ADD
            m.pose.position.x       = float(pos[0])
            m.pose.position.y       = float(pos[1])
            m.pose.position.z       = float(pos[2])
            m.pose.orientation.w    = 1.0
            m.scale.x               = m.scale.y = m.scale.z = 0.4
            m.color.a               = 0.9
            m.lifetime              = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)

            if classification == "DYNAMIC":
                m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.2  # Red
            else:
                m.color.r, m.color.g, m.color.b = 0.2, 0.8, 0.2  # Green

            track_msgs.markers.append(m)

            arrow = Marker()
            arrow.header.frame_id      = msg.header.frame_id
            arrow.header.stamp         = msg.header.stamp
            arrow.ns                   = "track_velocity"
            arrow.id                   = 1000 + t.id
            arrow.type                 = Marker.ARROW
            arrow.action               = Marker.ADD
            arrow.pose.orientation.w   = 1.0
            arrow.scale.x              = 0.08
            arrow.scale.y              = 0.14
            arrow.scale.z              = 0.18
            arrow.color.a              = 0.95
            arrow.color.r              = m.color.r
            arrow.color.g              = m.color.g
            arrow.color.b              = m.color.b
            arrow.lifetime             = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)
            start_point = Point()
            start_point.x = float(pos[0])
            start_point.y = float(pos[1])
            start_point.z = float(pos[2])
            end_point = Point()
            end_point.x = float(pos[0] + vel[0] * VELOCITY_ARROW_SCALE)
            end_point.y = float(pos[1] + vel[1] * VELOCITY_ARROW_SCALE)
            end_point.z = float(pos[2] + vel[2] * VELOCITY_ARROW_SCALE)
            arrow.points = [start_point, end_point]
            track_msgs.markers.append(arrow)

            text_marker = Marker()
            text_marker.header.frame_id    = msg.header.frame_id
            text_marker.header.stamp       = msg.header.stamp
            text_marker.ns                 = "track_labels"
            text_marker.id                 = 2000 + t.id
            text_marker.type               = Marker.TEXT_VIEW_FACING
            text_marker.action             = Marker.ADD
            text_marker.pose.position.x    = float(pos[0])
            text_marker.pose.position.y    = float(pos[1])
            text_marker.pose.position.z    = float(pos[2] + TRACK_TEXT_Z_OFFSET)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z            = 0.3
            text_marker.color.r            = 1.0
            text_marker.color.g            = 1.0
            text_marker.color.b            = 1.0
            text_marker.color.a            = 0.95
            text_marker.text               = (
                f"ID {t.id} | {classification}\n"
                f"{speed:.2f} m/s"
            )
            text_marker.lifetime           = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)
            track_msgs.markers.append(text_marker)

        self.pub_tracks.publish(track_msgs)

        # ================================================================
        # STEP 10: Summary & Timing
        # ================================================================
        elapsed_ms = (time.time() - t0) * 1000

        self.get_logger().info(f"{'='*80}")
        self.get_logger().info(f"FRAME #{self.frame_count} SUMMARY")
        self.get_logger().info(f"{'='*80}")
        self.get_logger().info(
            f"  Raw points:           {len(pts):8d}"
        )
        self.get_logger().info(
            f"  After voxel:          {len(pcd.points) + sum(d['inliers'] for d in wall_details) + ground_inliers:8d}"
        )
        self.get_logger().info(
            f"  Ground removed:       {ground_inliers:8d}"
        )
        self.get_logger().info(
            f"  Walls removed:        {wall_count:8d} (total points: {sum(d['inliers'] for d in wall_details):d})"
        )
        self.get_logger().info(
            f"  Final processing:     {len(remain_pts):8d}"
        )
        self.get_logger().info(
            f"  Clusters detected:    {len(clusters):8d}"
        )
        self.get_logger().info(
            f"  Active tracks:        {len(tracks):8d}"
        )
        self.get_logger().info(
            f"  Frame processing:     {elapsed_ms:8.2f} ms"
        )
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
