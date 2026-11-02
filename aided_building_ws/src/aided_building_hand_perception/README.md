# aided_building_hand_perception

Hand detection with a camera (MediaPipe Hand Landmarker), with the position expressed
in the robot base frame (`fr3_link0`).

The laptop camera is monocular (no depth). Each point of the hand becomes a ray, and that
ray is intersected with a plane parallel to the workbench (`z = plane_height` in the
`workbench` frame). The position is right as long as the hands work on the table, which is
the case when soldering.

```text
fr3_link0 ──(workbench.yaml: PROVISIONAL)──► workbench ──(ArUco calibration)──► camera_optical_frame
```

## Dependencies

```bash
sudo apt install ros-jazzy-v4l2-camera ros-jazzy-cv-bridge ros-jazzy-tf2-ros ros-jazzy-camera-calibration
pip install --user --break-system-packages --no-deps mediapipe absl-py flatbuffers sounddevice
wget -O ~/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Use `--no-deps`, because without it pip installs `numpy` 2.x and `opencv-contrib-python`
in `~/.local`, and that breaks ROS's `cv_bridge` (`KeyError: 16`).

## Usage

```bash
colcon build --symlink-install --packages-select aided_building_hand_perception
source install/setup.bash

# 1. Intrinsic calibration of the camera (once per camera)
#    result: ~/.ros/camera_info/notebook_cam.yaml

# 2. Print the ArUco marker (100 mm, at 100%, without fitting to the page)
ros2 run aided_building_hand_perception generate_marker -- --output marker.svg

# 3. Camera -> workbench calibration (with the camera already in its final position and the marker on the table)
ros2 run v4l2_camera v4l2_camera_node --ros-args --params-file \
  $(ros2 pkg prefix aided_building_hand_perception)/share/aided_building_hand_perception/config/webcam.yaml
ros2 run aided_building_hand_perception workbench_calibration   # another terminal
#    result: ~/.ros/aided_building/workbench_camera.yaml

# 4. Detection
ros2 launch aided_building_hand_perception hand_perception.launch.py
#    with the group's simulation: use_sim_time:=true
#    with the camera driver already running: start_camera:=false
```

Without the file from step 3, the launch uses an approximate camera pose
(`fallback_camera` in `config/workbench.yaml`). In that case the positions are only
good for testing the flow.

**When the robot is available:** measure where the center of the marker is relative to
the robot base and update `robot_base_to_workbench` in `config/workbench.yaml`.

## Published topics

### Hand zone (for contact analysis)

Seen from above, each hand takes up a **zone**: the smallest oriented rectangle around
the 21 points of the hand projected on the table, enlarged by `zone_margin`
(default 3 cm) on each side. The zone goes from the table (z = 0 in `workbench`)
up to `zone_height` (default 10 cm). The margin covers the skin beyond the detected
joints and the tracking error.

| Topic | Type | Content |
| --- | --- | --- |
| `/hands/zones` | `vision_msgs/Detection3DArray` | one oriented box per hand: `bbox.center` (pose) + `bbox.size`, `results[0].hypothesis.class_id` = `left`/`right`, `score` = confidence |
| `/hands/{left,right}/zone` | `geometry_msgs/PolygonStamped` | the 4 corners of the zone's footprint (seen from above), at table level |
| `/hands/tool_distance` | `std_msgs/Float32` | shortest distance between the origin of `tool_frame` and the zones (0 = tool inside the zone) |
| `/hands/min_distance` | `std_msgs/Float32` | shortest distance between the origins of links `fr3_link0..8` and the zones |
| `/hands/top_view` | `sensor_msgs/Image` | top view: zones, robot (with its 855 mm reach), tool and camera |

`bbox` uses the same format (pose + size) as a MoveIt BOX `SolidPrimitive`, so the zone
can be inserted directly as a collision object.

### Position and motion

| Topic | Type | Content |
| --- | --- | --- |
| `/hands/poses` | `geometry_msgs/PoseArray` | center of the palm of each hand (empty without hands) |
| `/hands/{left,right}/palm` | `geometry_msgs/PointStamped` | filtered palm (One Euro), by the person's side |
| `/hands/{left,right}/velocity` | `geometry_msgs/Vector3Stamped` | velocity of the palm (m/s) |
| `/hands/markers` | `visualization_msgs/MarkerArray` | skeleton, zone box and label for RViz |
| `/hands/debug_image` | `sensor_msgs/Image` | camera image with the skeleton and the zone box |

The `/hands/debug_image` and `/hands/top_view` images are only generated when there is a
subscriber. To see them:
`ros2 run rqt_image_view rqt_image_view /hands/top_view`.

All of them use the image's `header.stamp` and the `fr3_link0` frame. With
`use_sim_time:=true`, the stamp comes from the node's clock, because the camera uses the
real clock and Gazebo uses the simulated clock.

These types are provisional, until the group defines `aided_building_interfaces/HumanState`.

## Known issues

- **Use `v4l2_camera`, not `usb_cam` 0.8.1.** `usb_cam` stamps the images 0 to 1 s late,
  and that error keeps growing by about 1 ms/s (a bug in `utils.hpp` in the `tv_usec`
  conversion).
- **Subscribe to images with RELIABLE QoS.** With BEST_EFFORT, the 900 kB images arrived
  at 9–19 Hz instead of 30 Hz.
- **VS Code installed as a Snap:** windows (calibrator, RViz) break with
  `symbol lookup error ... libpthread`. Run `unset GTK_PATH GTK_EXE_PREFIX
  GIO_MODULE_DIR GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR GTK_IM_MODULE_FILE
  LOCPATH GSETTINGS_SCHEMA_DIR` first.
- **Shared network:** with `ROS_DOMAIN_ID` 0 and `SUBNET` discovery, the TF of other
  simulations on the network gets mixed with yours. Use
  `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` or a `ROS_DOMAIN_ID` of your own.
- MediaPipe's Left/Right label assumes a mirrored image (selfie). The node swaps the
  labels (`image_is_mirrored: false`). Check it with your own hands.
- The marker needs a white margin of at least one cell. The generated SVG already has
  two, so do not cut it flush with the black.

## Tests

```bash
colcon test --packages-select aided_building_hand_perception
colcon test-result --verbose
```
