# Industrial LiDAR Perception Pipeline (ROS 2 + Open3D)

A complete, real-time 3D perception pipeline for industrial mobile robots or UAVs operating in structured indoor/outdoor environments. Designed to reliably detect, track, and classify moving and static objects from raw LiDAR point clouds.

Built with **ROS 2 (Humble/Foxy+)** and **Open3D**, this node processes point clouds through a sequence of geometric filters and clustering algorithms to output:
- Cleaned and segmented point clouds
- 3D bounding boxes of detected objects
- Persistent object tracks with velocity estimation
- Dynamic vs. static classification

Ideal for AGVs, warehouse robots, inspection drones, or any application requiring robust obstacle awareness in cluttered industrial scenes.

---

## 🔧 Pipeline Overview

The system performs the following steps in real time:

1. **Input**: Raw `sensor_msgs/PointCloud2`  
2. **Preprocessing**: Unit auto-detection (mm → m), NaN/self/far-point filtering  
3. **Adaptive voxel downsampling**  
4. **Statistical Outlier Removal (SOR)**  
5. **Ground plane removal** (RANSAC)  
6. **Iterative vertical wall removal** (multi-pass RANSAC with normal-angle validation)  
7. **DBSCAN clustering** for object segmentation  
8. **3D bounding box estimation** and visualization  
9. **Multi-object tracking** using constant-velocity Kalman filters  
10. **Velocity-based dynamic/static classification**

All intermediate stages are published as debug topics for easy tuning and validation.

---

## 🚀 Features

- ✅ **Robust to unit errors**: Auto-detects millimeter-scale data and converts to meters  
- ✅ **Industrial-adaptive voxelization**: Voxel size scales with scene extent  
- ✅ **Smart plane classification**: Distinguishes ground, walls, and oblique surfaces by normal angle  
- ✅ **Front-hemisphere filtering**: Assumes forward-facing sensor (X > 0) to ignore drone body  
- ✅ **Persistent tracking**: Handles occlusion with missed-frame tolerance  
- ✅ **Comprehensive logging**: Coordinate ranges, cluster dimensions, wall normals, track velocities  
- ✅ **Debug-friendly**: Publishes every processing stage (`/debug/raw`, `/debug/voxel`, etc.)

---

