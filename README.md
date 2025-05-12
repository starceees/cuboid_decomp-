# Cuboid Decomposition & Direct-Path Planning  
A ROS2‐based Python tool for decomposing a 3D point cloud into free cuboidal volumes, constructing a connectivity graph between them, and planning a collision‐free direct path through specified waypoints. It also publishes visualization topics for RViz.

---

## 🚀 Features

- **3D Occupancy Grid Construction**  
  Builds a voxel occupancy grid from a point cloud and applies a configurable safety margin.

- **Region Growing Cuboid Extraction**  
  Identifies maximal free-space cuboids via a 3D region‐growing algorithm with adjustable maximum height.

- **Connectivity Graph**  
  Converts cuboids into world‐coordinate boxes, checks overlap/line‐of‐sight, and builds an A*‐searchable graph on the fly.

- **A* Path Planning**  
  Finds an optimal sequence of adjacent cuboids between start/end waypoints, then refines into straight‐line segments via intersection midpoints.

- **ROS2 RViz Visualization**  
  Publishes PointCloud2, MarkerArray, and Path messages for visualizing:
  - the original point cloud
  - all cuboids
  - the subset of cuboids used in planning
  - the planned direct‐path line and PoseStamped path
  - waypoint markers
  - a moving drone mesh along the path

- **Metrics Logging**  
  Measures decomposition time, planning time, path length, graph size, and saves to CSV.

---

## 📋 Prerequisites

- Ubuntu 20.04 / 22.04  
- ROS2 (Foxy / Galactic / Humble)  
- Python 3.8+  
- NumPy, SciPy, tqdm, `sensor_msgs_py`  
- A compiled ROS2 workspace with `rclpy`, `std_msgs`, `sensor_msgs`, `visualization_msgs`, `nav_msgs`, `geometry_msgs`

---

## ⚙️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/starceees/cuboid_decomp-.git
   cd cuboid_decomp-
   ```

2. **Install Python dependencies**  
   ```bash
   pip3 install numpy scipy tqdm
   ```

3. **Build your ROS2 workspace** (if not already)  
   ```bash
   colcon build
   source install/setup.bash
   ```

---

## 🔧 Configuration

All parameters live in the `CONFIG` dictionary inside `main()`:

| Key                  | Description                                  | Default           |
|----------------------|----------------------------------------------|-------------------|
| `POINT_CLOUD_FILE`   | Path to `.npy` point cloud (x, y, z)         | `../pointcloud/.../point_cloud_gq.npy` |
| `MIN_POINT_Z`        | Minimum Z cutoff for filtering               | `-1.0`            |
| `MAX_POINT_Z`        | Maximum Z cutoff for filtering               | `100.0`           |
| `RESOLUTION`         | Voxel side length (meters)                   | `0.2`             |
| `SAFETY_VOXELS`      | Number of dilation iterations (safety margin)| `1`               |
| `MAX_Z_THICKNESS`    | Max vertical growth (voxels) in region grow  | `1`               |
| `CUBOIDS_FILE`       | Path to cache cuboids (pickle)               | `my_cuboids_gq.pkl` |
| `WAYPOINTS_FILE`     | Text file containing waypoints (x, y, z)     | `../waypoints/.../waypoints_gq.txt` |
| `DRONE_MESH_RESOURCE`| URI for the drone mesh (Marker.MESH_RESOURCE)| `file://.../race2.stl` |

---

## 📖 Waypoint File Format

A plain `.txt` file:

\`\`\`
<NUM_WAYPOINTS>
x₁
y₁
z₁
qx₁
qy₁
qz₁
qw₁
x₂
...
\`\`\`

Only the first three lines of each 7‐line block (x, y, z) are used.

---

## 🚗 Usage

\`\`\`bash
source /opt/ros/<distro>/setup.bash
ros2 run <your_package> cuboid_decomp_node
\`\`\`

- The node will load the point cloud, perform decomposition (or load cached cuboids), plan segment-by-segment between successive waypoints, log metrics to \`metrics_gq_new.csv\`, then spin an RViz publisher under \`/free_cuboids\`, \`/path_cuboids\`, \`/line_path\`, \`/planned_path\`, \`/waypoints\`, and \`/drone_mesh\`.

- Launch RViz and add:
  - **PointCloud2** on \`/point_cloud\`
  - **MarkerArray** on \`/free_cuboids\` & \`/path_cuboids\`
  - **Marker** on \`/line_path\`
  - **Path** on \`/planned_path\`
  - **MarkerArray** on \`/waypoints\`
  - **Marker** on \`/drone_mesh\`

---

## 📐 How It Works

1. **Occupancy Grid**  
   Converts each point to a voxel index then marks occupied voxels.

2. **Obstacle Expansion**  
   Binary‐dilates the occupancy grid to add safety padding.

3. **Region Growing**  
   Finds maximal axis‐aligned empty cuboids up to a vertical thickness, marking visited voxels to avoid overlap.

4. **World Cuboids & Connectivity**  
   Each block is converted to real‐world bounds; pairwise overlap + line‐of‐sight checks define graph edges.

5. **A***  
   Uses Euclidean‐distance heuristic on cuboid centers to find a shortest path in the connectivity graph.

6. **Direct Path Construction**  
   Between each cuboid pair, computes the midpoint of their intersection (or center fallback) to create smoother waypoints.

7. **Visualization**  
   A ROS2 node continuously publishes all elements for live inspection in RViz; also animates a drone mesh along the computed straight‐line path.

---

## 📝 Metrics Output

On completion, a CSV (\`metrics_gq_new.csv\`) is generated with columns:

- Size of point cloud  
- Number of cuboids  
- Number of cuboids in path  
- Total decomposition time (s)  
- Total connectivity/planning time (s)  
- Total path length (m)  
- Graph edge count  

---

## 🛠️ Extending & Troubleshooting

- **Adjust safety margin** via \`SAFETY_VOXELS\`.  
- **Increase \`MAX_Z_THICKNESS\`** to allow taller cuboids.  
- **Tweak \`step\`** in visibility checks (\`build_cuboids_with_connectivity\`, \`line_of_sight_in_cuboids\`) for more/less strict LOS.  
- **Cache reuse**: delete the pickle file to force a rebuild of cuboids.  
- **Profiling**: wrap key sections (e.g., region growing) with timers to identify bottlenecks.

---

## 📄 License

This project is released under the MIT License. Feel free to adapt and extend for your own research and robotic applications.
