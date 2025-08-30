# NXP AIM India 2025 – ROS2 Warehouse Exploration

This repository contains ROS 2 packages and configuration files developed for the **NXP AIM India 2025 Challenge**, focused on autonomous warehouse exploration, object recognition, and navigation.

The system integrates **SLAM, navigation, object detection (YOLOv5 / TFLite), and behavior trees** for exploring warehouse environments with a buggy platform.

🔗 **More Resources & Reference**: [NXPHoverGames/NXP\_AIM\_INDIA\_2025](https://github.com/NXPHoverGames/NXP_AIM_INDIA_2025)


## 📂 Repository Structure

### 🔹 Navigation & SLAM

* **`slam.yaml`** – SLAM configuration file.
* **`nav2.yaml`** – Nav2 (navigation2) parameters for global & local planners.
* **`nav_to_pose_bt.xml`** – Behavior Tree for navigating to a single pose with replanning and recovery actions.
* **`nav_through_poses_bt.xml`** – Behavior Tree for navigating through multiple waypoints with replanning and recovery actions.

### 🔹 ROS2 Nodes

* **`b3rb_ros_object_recog.py`** – Object recognition node using YOLOv5 (TFLite) and optional QR code detection.
* **`b3rb_ros_warehouse.py`** – Core warehouse exploration logic:

  * SLAM-based map exploration.
  * Shelf/object identification using clustering & PCA.
  * QR code detection & object counting.
  * Goal planning and frontier exploration.
* **`b3rb_ros_model_remove.py`** – Gazebo model remover node. Dynamically removes curtains/shelves when QR codes are identified.
* **`b3rb_ros_draw_map.py`** – Visualization node that subscribes to `/map` and displays the occupancy grid using Matplotlib.


## 🚀 Features

* **Autonomous Navigation**: Nav2 with custom behavior trees for robust replanning & recovery.
* **SLAM Support**: Builds and maintains an occupancy grid of the warehouse.
* **Object Detection**: Runs YOLOv5/TFLite for detecting shelf objects (banana, cup, teddy bear, etc.).
* **QR Code Scanning**: Reads shelf QR codes (via pyzbar or OpenCV fallback).
* **Shelf Exploration**: Identifies rectangular objects (shelves) in the map, plans approach goals, and scans objects.
* **Dynamic Environment**: Removes Gazebo curtains as shelves are discovered to simulate progressive exploration.
* **Visualization**: Real-time map visualization.


## ⚙️ Dependencies

* ROS 2 (Humble / Iron recommended)
* `nav2_bringup`
* `rclpy`
* `cv2`, `numpy`, `torch`, `torchvision`
* `tflite_runtime`
* `pyzbar` (optional, for QR codes)
* `scikit-learn`, `scipy` (for clustering & PCA)
* `matplotlib` (for map visualization)


## ▶️ Running the System

1. **Build your workspace**:

   ```bash
   colcon build --packages-select b3rb_ros_aim_india
   source install/setup.bash
   ```

2. **Launch SLAM & Navigation**:

   ```bash
   ros2 launch nav2_bringup tb3_simulation_launch.py params_file:=slam.yaml
   ```

3. **Run Object Recognition**:

   ```bash
   ros2 run b3rb_ros_aim_india b3rb_ros_object_recog.py
   ```

4. **Run Warehouse Exploration**:

   ```bash
   ros2 run b3rb_ros_aim_india b3rb_ros_warehouse.py --ros-args -p shelf_count:=4
   ```

5. **Enable Dynamic Shelf Removal (Gazebo)**:

   ```bash
   ros2 run b3rb_ros_aim_india b3rb_ros_model_remove.py --ros-args -p warehouse_id:=1
   ```

6. **Visualize Map**:

   ```bash
   ros2 run b3rb_ros_aim_india b3rb_ros_draw_map.py
   ```



