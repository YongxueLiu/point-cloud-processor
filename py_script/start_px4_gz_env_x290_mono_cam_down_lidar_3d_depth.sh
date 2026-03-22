#!/bin/bash
# =========================================================
# 自动化启动无人机仿真环境脚本
# 启动顺序：
# 1. Gazebo 仿真世界
# 2. PX4 SITL
# 3. ROS-GZ 图像桥接
# 4. MicroXRCEAgent
# 5. QGroundControl
# =========================================================

# === 可配置路径 ===
WORLD_SCRIPT_PATH="$HOME/.simulation-gazebo/worlds"
PX4_PATH="$HOME/PX4-Autopilot"
PYTHON_ENV_PATH="$HOME/myenv/bin/activate"
WORLD_NAME="mod" # <world name="mod">

# ===激活 Python 虚拟环境（如果存在） ===
if [ -f "$PYTHON_ENV_PATH" ]; then
    echo "🔹 Activating Python environment..."
    source "$PYTHON_ENV_PATH"
else
    echo "⚠️ 未找到 Python 虚拟环境（$PYTHON_ENV_PATH），跳过。"
fi

# === 2️⃣ 启动 Gazebo 仿真 ===
gnome-terminal --tab --title="Gazebo Sim" -- bash -c "
cd $WORLD_SCRIPT_PATH;
echo '🚀 启动 Gazebo 世界：$WORLD_NAME';
python3 simulation-gazebo --world $WORLD_NAME;
exec bash
"

# === 3️⃣ 启动 PX4 SITL ===
sleep 4
gnome-terminal --tab --title="PX4 SITL" -- bash -c "
cd $PX4_PATH;
echo '🛫 启动 PX4 SITL';
PX4_GZ_WORLD=$WORLD_NAME PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 \
PX4_GZ_MODEL_POSE='0.0, 0.0' PX4_SIM_MODEL=x290_mono_cam_down_lidar_3d_depth \
./build/px4_sitl_default/bin/px4 -i 0;
exec bash
"


# # === 4️⃣ 启动 ROS-GZ Bridge ===
sleep 3
gnome-terminal --tab --title="ROS-GZ Bridge" -- bash -c "
cd $PX4_PATH;
echo '🔄 启动 ROS2 <-> Gazebo 图像与激光雷达桥接';
ros2 run ros_gz_bridge parameter_bridge \
/rgb_camera@sensor_msgs/msg/Image@gz.msgs.Image \
/depth_camera@sensor_msgs/msg/Image@gz.msgs.Image \
/lidar_points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
/mono_cam/image@sensor_msgs/msg/Image@gz.msgs.Image \
/lidar_scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan;
exec bash
"

# === 5️⃣ 启动 MicroXRCEAgent ===
sleep 2
gnome-terminal --tab --title="MicroXRCEAgent" -- bash -c "
cd $PX4_PATH;
echo '📡 启动 MicroXRCEAgent (UDP port 8888)';
MicroXRCEAgent udp4 -p 8888;
exec bash
"

# === 6️⃣ 启动 QGroundControl ===
sleep 2
gnome-terminal --tab --title="QGroundControl" -- bash -c "
echo '🛰️ 启动 QGroundControl';
~/bin/QGroundControl-x86_64.AppImage;
exec bash
"

echo "✅ 所有仿真组件已启动！"
