# Move and Pick Object in Rosmaster x3 Plus

This repository contains a ROS Catkin workspace for a mecanum-wheeled robot with integrated YOLO-based object detection and manipulation. The main logic is implemented in [`YOLOv11/src/yolov11/scripts/ROS_YOLO.py`](YOLOv11/src/yolov11/scripts/ROS_YOLO.py).

## Main Script

### [`ROS_YOLO.py`](YOLOv11/src/yolov11/scripts/ROS_YOLO.py)

This script implements the `RosYoloDetector` class, which manages:

- **YOLOv11 object detection** using two models:
  - `model_detect`: For initial detection and 2D alignment.
  - `model_box`: For horizontal-only correction after gripping.
- **Robot movement** with encoder-based distance tracking and bounding box correction.
- **Manipulator control** for picking and releasing objects.
- **Threaded image processing** and inference for real-time performance.
- **State management** for switching between detection and manipulation phases.

### Key Features

- **Dual-model YOLO inference**: Switches between full 2D and horizontal-only correction based on task phase.
- **Thread-safe robot and state control** using Python threading and locks.
- **High FPS inference** ~20FPS while running with the YOLOv11n and NCNN format.
- **Real-time display** with OpenCV, including bounding boxes, FPS, and status overlays.
- **Clean shutdown** and resource management.

### Example Workflow

1. **Move forward** to the search area.
2. **Detect and align** with the target object using YOLO.
3. **Pick the object** with the manipulator.
4. **Switch to box model** and move backward with horizontal correction.
5. **Return to original position** and release the object.

## Running the Main Script

```sh
cd YOLOv11
source devel/setup.bash
roslaunch yolov11 yolov11.launch
```

## File Structure

- [`YOLOv11/src/yolov11/scripts/ROS_YOLO.py`](YOLOv11/src/yolov11/scripts/ROS_YOLO.py): Main logic for detection, movement, and manipulation.
- Other scripts in `scripts/` provide alternative or experimental logic.

## Requirements

- ROS (tested with Melodic/Noetic)
- Python 3.8
- OpenCV, NumPy, ultralytics YOLO, NCNN (Can easily download the dependencies with the [`requirements.sh`](YOLOv11/src/yolov11/scripts/requirements.sh) )
- Custom robot libraries: `Rosmaster_Lib`, `MecanumRobot`, `Robot`

## Notes

- This project is creates from the main configuration from [Yahboom repotory](http://www.yahboom.net/study/ROSMASTER-X3-PLUS)
- The script expects a compatible camera publishing to `/camera/rgb/image_raw`.
- YOLO model weights should be placed in `YOLO_weights/` as referenced in the script.
- For details on the robot API, see [`MecanumRobot`](YOLOv11/src/yolov11/scripts/MecanumRobot.py).

---

For more details, see the code and docstrings in [`ROS_YOLO.py`](YOLOv11/src/yolov11/scripts/ROS_YOLO.py).
