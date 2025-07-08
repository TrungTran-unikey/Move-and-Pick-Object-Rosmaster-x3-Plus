#!/usr/bin/env python3.8

import os
import sys
import math
import time
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
from threading import Lock

from Rosmaster_Lib import Rosmaster
from MecanumRobot import MecanumRobot, ManipulatorController
# Legacy imports for compatibility
from Robot import P1, P2, P3, P4

CLASS = {0: "Blue", 
         1: "Green", 
         2: "Red", 
         3: "Yellow"}

# CLASS_ID = {v.lower(): k for k, v in CLASS.items()}

class RosYoloDetector:
    """
    ROS YOLO Detector with dual model system for object detection and manipulation.
    
    MODEL SWITCHING LOGIC:
    1. Initially uses 'model_detect' for object detection and approach/alignment
       - Provides full 2D correction (both X and Y axis)
       - Used during initial search, approach, and precise alignment phases
    
    2. After gripping object, switches to 'model_box' for backward movement
       - Provides horizontal-only correction (X axis only)
       - Used only during backward movement after successful grip
       - Ensures better alignment during post-grip maneuvers
    
    3. Only ONE model performs inference at any given time
       - Thread-safe model switching with state protection
       - Clear state tracking prevents incorrect model usage
    
    4. Model switching is automatic based on operation phase:
       - Detect -> Box: After successful object grip
       - Box -> Detect: After backward movement completion
    """
    
    def __init__(self):
        # Initialize node
        self.target_cube_class = 0
        self.speed = 30.0
        self.correction_speed = 30
        self.horizontal_weight = 50
        rospy.init_node('yolov11', anonymous=True)

        # Create YOLO model
        self.model_detect = YOLO('/root/YOLOv11/src/yolov11/scripts/YOLO_weights/best_ncnn_model_256', task='detect')
        self.model_detect_conf = 0.3
        self.model_box = YOLO('/root/YOLOv11/src/yolov11/scripts/YOLO_weights/best_ncnn_model_box', task='detect')
        self.model_box_conf = 0.3
        
        self.IMG_SIZE = 256
        self.IMG_ORG_H = 480
        self.IMG_ORG_W = 640

        # Initialize the bot with new API
        print("Initialize the bot ...")
        self.bot = Rosmaster()
        self.bot.create_receive_threading()
        self.bot.set_car_type(0x02)
        self.bot.clear_auto_report_data()
        self.bot.set_auto_report_state(True, forever=False)
        self.bot.reset_flash_value()
        self.bot.set_motor(0, 0, 0, 0)
        time.sleep(12.5)

        self.bot.set_uart_servo_angle_array(P1, run_time=4000)
        
        # Create robot controller with new API
        self.robot = MecanumRobot(self.bot, kp=5.0, ki=1.0, kd=0.0, debug=True)
        self.manipulator = ManipulatorController(self.bot, self.robot.config)
        
        # Initialize pose
        _, _, self.yaw = self.bot.get_imu_attitude_data(ToAngle=True)
        self.robot.initialize_pose(self.yaw)
        print(f"Initial Yaw: {self.yaw:.2f}°")
        
        # Set up bounding box correction callback
        # self.robot.set_bounding_box_correction(self.get_bb_error)
        
        # Distance tracking for new movement system
        self.target_distance = 0.0  # Target distance for current movement
        self.initial_encoders = [0.0, 0.0, 0.0, 0.0]  # Initial encoder values
        self.movement_active = False  # Flag to track if movement is in progress
        self.movement_direction = 0  # 1 for forward, -1 for backward, 0 for stopped
        
        # Encoder configuration (same as Robot.py EncoderPositionController)
        self.wheel_diameter = 97.0  # mm
        self.wheel_circumference = math.pi * self.wheel_diameter
        self.encoder_resolution = (205/60) * 11  # placeholder, adjust based on actual spec
        self.gear_ratio = 56  # gear ratio
        self.total_encoder_per_revolution = self.encoder_resolution * self.gear_ratio
        self.mm_per_pulse = self.wheel_circumference / self.total_encoder_per_revolution if self.total_encoder_per_revolution != 0 else 0
        print(f"Distance tracking setup: {self.mm_per_pulse:.4f} mm per pulse")

        # Thread synchronization
        self.robot_lock = Lock()  # Protect robot control operations
        self.model_detect_lock = Lock()  # Protect YOLO model access
        self.state_lock = Lock()  # Protect shared state variables
        self.running = True
        
        # FPS calculation
        self.prev_time = time.perf_counter()
        self.fps = [0] * 20

        # Image scaling coefficients
        self.height_coef = self.IMG_ORG_H / self.IMG_SIZE
        self.width_coef = self.IMG_ORG_W / self.IMG_SIZE

        # Bounding box setup
        self.bounding_box = None
        self.set_default_bounding_box(280, 370, 275, 240)
        self.move_to_box = False
        
        # Model selection for bounding box correction
        # True for model_box (horizontal only), False for model_detect (full 2D)
        self.use_box_model_correction = False  # Start with model_detect for initial alignment

        # Gripper variable
        self.move_gripper = False

        # State management for model switching and movement
        self.moving_backward = False  # Flag to indicate if we are moving backward after gripping
        self.object_gripped = False   # Flag to track if object has been gripped
        self.current_model = "detect"  # Current active model: "detect" or "box"

        # Thread-safe queues - optimized for low latency
        self.raw_image_queue = queue.Queue(maxsize=1)  # Raw images from callback - keep only latest
        self.processed_image_queue = queue.Queue(maxsize=1)  # Preprocessed images with timestamps - keep only latest
        self.result_queue = queue.Queue(maxsize=1)  # Inference results with frame data - keep only latest

        # Frame synchronization
        self.frame_counter = 0
        self.frame_lock = Lock()

        # Video capture for backward movement
        self.cap = cv2.VideoCapture(1)
        self.capture_lock = Lock()
        
        # Latest synchronized frame and results for display
        self.latest_frame_data = None  # Contains (image, results, frame_id, timestamp)
        self.frame_data_lock = Lock()
        self.result_timestamp = 0

        # Start processing threads
        self.preprocess_thread = threading.Thread(target=self.preprocess_worker, daemon=True)
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        
        self.preprocess_thread.start()
        self.inference_thread.start()

        self.count_correct = 0 # Count of correct alignments
        self.target_correct = self.correction_speed // 5 # Number of correct alignments before stopping

        # Subscribe to image topic
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback,
                         queue_size=1, buff_size=2**24)
        rospy.loginfo("yolov11 node started, waiting for images...")

    def get_encoder_distances(self):
        """Get movement distance from encoders (mm) - same logic as Robot.py"""
        encoder_data = self.bot.get_motor_encoder()
        distances = []
        for enc in encoder_data:
            distance_mm = enc * self.mm_per_pulse
            distances.append(distance_mm)
        return distances
    
    def get_current_travel_distance(self):
        """Calculate the current traveled distance since movement started"""
        current_encoders = self.get_encoder_distances()
        
        # Calculate distance traveled from each wheel
        distances = []
        for i in range(len(current_encoders)):
            distance = abs(current_encoders[i] - self.initial_encoders[i])
            distances.append(distance)
        
        # Use average of all wheels for more accuracy
        if distances:
            return sum(distances) / len(distances)
        return 0.0
    
    def start_movement(self, target_distance, direction_deg, speed):
        """Start a new movement with distance tracking"""
        with self.robot_lock:
            # Store initial encoder values
            self.initial_encoders = self.get_encoder_distances()
            self.target_distance = abs(target_distance)
            self.movement_direction = 1 if direction_deg == 0 else -1  # 0=forward, 180=backward
            self.movement_active = True
            
            # Calculate movement vector based on direction
            if direction_deg == 0:  # Forward
                vx, vy = target_distance, 0
                dx_target, dy_target = target_distance, 0
            elif direction_deg == 180:  # Backward
                vx, vy = -target_distance, 0
                dx_target, dy_target = target_distance, 0  # Distance is always positive
            else:
                rospy.logwarn(f"Unsupported direction: {direction_deg}°. Using forward.")
                vx, vy = target_distance, 0
                dx_target, dy_target = target_distance, 0
            
            # Start the movement with the new API
            result = self.robot.move_distance(
                vx=vx, vy=vy, 
                dx_target=dx_target, dy_target=dy_target,
                current_distance=0.0,  # Starting distance
                max_speed=speed
            )
            
            rospy.loginfo(f"Started movement: target={target_distance}mm, direction={direction_deg}°")
    
    def update_movement(self, speed = None, debug=False):
        """Update movement progress and stop when target is reached"""
        if speed == None:
            speed = self.speed

        if not self.movement_active:
            return True  # Movement complete
            
        with self.robot_lock:
            current_distance = self.get_current_travel_distance()
            
            # Continue movement with updated distance
            if self.movement_direction == 1:  # Forward
                vx, vy = self.target_distance, 0
                dx_target, dy_target = self.target_distance, 0
            else:  # Backward
                vx, vy = -self.target_distance, 0
                dx_target, dy_target = self.target_distance, 0

            # Apply horizontal correction when using box model
            if self.current_model == "box":
                # Get horizontal pixel error for correction
                try:
                    horizontal_correction = self.get_horizontal_correction()
                    # Keep vx the same, apply horizontal correction to vy
                    corrected_vy = vy + horizontal_correction
                    print(f"Box model: applying horizontal correction {horizontal_correction:.2f} to vy")
                except Exception as e:
                    rospy.logwarn(f"Failed to get horizontal correction: {e}")
                    corrected_vy = vy
                
                self.display_worker()
                movement_complete = self.robot.move_distance(
                    vx=vx, vy=corrected_vy,
                    dx_target=dx_target, dy_target=dy_target,
                    current_distance=current_distance,
                    max_speed=speed
                )
            else:
                # For detect model, use original movement without correction
                movement_complete = self.robot.move_distance(
                    vx=vx, vy=vy,
                    dx_target=dx_target, dy_target=dy_target,
                    current_distance=current_distance,
                    max_speed=speed
                )
            
            if movement_complete and not debug:
                # Target reached, stop movement
                self.robot.stop()
                self.movement_active = False
                rospy.loginfo(f"Movement complete: traveled {current_distance:.1f}mm of {self.target_distance:.1f}mm")
                return True
            else:
                # Continue movement
                progress = (current_distance / self.target_distance) * 100 if self.target_distance > 0 else 0
                if int(progress) % 20 == 0:  # Log every 20% progress
                    rospy.logdebug(f"Movement progress: {progress:.1f}% ({current_distance:.1f}/{self.target_distance:.1f}mm)")
                return False

    def get_horizontal_correction(self):
        """
        Calculate horizontal correction based on current detection results.
        Returns the horizontal velocity correction for box model alignment.
        """
        if self.latest_frame_data is None:
            return 0.0
        
        # Get the latest frame data
        frame_data = self.latest_frame_data
        results = frame_data['results']
        
        if results is None:
            return 0.0
        
        # YOLO results are returned as a list, get the first result
        for result in results:
            
            # Check if result has boxes
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            # Extract detection data
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Filter for target class
            target_id = self.target_cube_class if self.current_model == "detect" else 0
            target_detections = [(box, score) for box, score, class_id in zip(boxes, scores, class_ids) 
                            if class_id == target_id]
            
            # Get best detection
            best_detection = max(target_detections, key=lambda x: x[1])
            box, _ = best_detection
            x1, y1, x2, y2 = map(int, box)
            
            # Convert to original image coordinates
            x1_org = int(x1 * self.width_coef)
            x2_org = int(x2 * self.width_coef)
            
            # Calculate YOLO detection center
            cx_yolo = (x1_org + x2_org) / 2
            
            # Calculate horizontal center of image for box model alignment
            cx_center = (self.IMG_ORG_W - 150) / 2
            
            # Calculate pixel error (positive = target is right of center)
            dx_pixel = cx_center - cx_yolo
            
            # Convert pixel error to velocity correction (K_x gain from moving method)
            K_x = self.horizontal_weight
            horizontal_correction = K_x * dx_pixel
            
        print(f"Horizontal correction: dx_pixel={dx_pixel:.1f}, correction={horizontal_correction:.2f}")
        return horizontal_correction

    def switch_to_model(self, model_name):
        """
        Switch between YOLO models and update correction mode.
        
        LOGIC:
        - Initially uses "detect" model for full 2D correction during approach and alignment
        - After gripping object, switches to "box" model for horizontal-only correction during backward movement
        - Only one model performs inference at any given time
        - Box model can only be used after object is gripped
        
        Args:
            model_name: "detect" for initial detection/alignment, "box" for backward movement
        """
        if model_name == "box" and not self.object_gripped:
            rospy.logwarn("Cannot switch to box model before object is gripped")
            return False
        
        previous_model = self.current_model
        
        if model_name == "detect":
            self.current_model = "detect"
            self.use_box_model_correction = False  # Full 2D correction
            rospy.loginfo(f"Switched from {previous_model} to detect model with full 2D correction")
        elif model_name == "box":
            self.current_model = "box"
            self.use_box_model_correction = True   # Horizontal only correction
            rospy.loginfo(f"Switched from {previous_model} to box model with horizontal-only correction")
        else:
            rospy.logerr(f"Unknown model name: {model_name}")
            return False
        
        return True

    def get_current_model_status(self):
        """Get current model and state information for debugging."""
        with self.state_lock:
            return {
                'current_model': self.current_model,
                'object_gripped': self.object_gripped,
                'moving_backward': self.moving_backward,
                'use_box_model_correction': self.use_box_model_correction
            }

    def set_default_bounding_box(self, xmin, xmax, ymin, ymax):
        self.bounding_box = (xmin, ymin, xmax, ymax)
        x_min, y_min, x_max, y_max = self.bounding_box
        cx_default = (x_min + x_max) / 2
        cy_default = (y_min + y_max) / 2
        self.default_bb_center = (cx_default, cy_default)

    def put_tilted_text(self, img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(0, 255, 0), thickness=2, angle=45):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        canvas_size = int(np.sqrt(text_width**2 + text_height**2)) + 10
        mask_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        text_x = (canvas_size - text_width) // 2
        text_y = (canvas_size - text_height) // 2 + text_height
        cv2.putText(mask_canvas, text, (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)
        rotation_center = (canvas_size // 2, canvas_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        rotated_mask = cv2.warpAffine(mask_canvas, rotation_matrix, (canvas_size, canvas_size),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pos_x = org[0] - canvas_size // 2
        pos_y = org[1] - canvas_size // 2
        pos_x = max(0, min(pos_x, img.shape[1] - canvas_size))
        pos_y = max(0, min(pos_y, img.shape[0] - canvas_size))
        roi = img[pos_y:pos_y + canvas_size, pos_x:pos_x + canvas_size]
        alpha = rotated_mask.astype(np.float32) / 255.0
        for c in range(3):
            roi[:, :, c] = np.clip((1 - alpha) * roi[:, :, c] + alpha * color[c], 0, 255).astype(np.uint8)

    def image_callback(self, img_msg):
        """Put raw image into queue for processing"""
        with self.state_lock:
            if not self.running or self.moving_backward:
                return
            
        try:
            # Non-blocking put - if queue is full, drop the oldest frame
            if self.raw_image_queue.full():
                try:
                    self.raw_image_queue.get_nowait()  # Remove oldest
                except queue.Empty:
                    pass
            self.raw_image_queue.put_nowait(img_msg)
        except queue.Full:
            pass  # Queue is full, drop this frame

    def preprocess_worker(self):
        """Thread to handle image preprocessing"""
        while self.running and not rospy.is_shutdown():
            cv_img = None

            if self.moving_backward:
                try:
                    if self.cap is not None and self.cap.isOpened():
                        ret, cv_img = self.cap.read()
                        if not ret:
                            cv_img = None
                            rospy.logwarn("Failed to capture frame from USB camera")
                            import sys; sys.exit("USB camera not ready, exiting...")
                    else:
                        # If camera not ready, wait a bit
                        time.sleep(0.01)
                        continue
                except queue.Empty:
                    continue
            else:
                try:
                    # Get image from callback queue with timeout
                    img_msg = self.raw_image_queue.get(timeout=0.1)
                    
                    # Convert ROS image to OpenCV format
                    if img_msg.encoding == 'bgr8':
                        cv_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, 3))
                    elif img_msg.encoding == 'rgb8':
                        cv_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, 3))
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    else:
                        rospy.logerr(f'Unsupported encoding: {img_msg.encoding}')
                        continue
                except queue.Empty:
                    # No ROS image, loop again
                    continue

            if cv_img is None:
                time.sleep(0.01) # Avoid busy-waiting if no image is available
                continue

            # Generate unique frame ID
            frame_id = self.frame_counter
            self.frame_counter += 1
            
            # Resize for YOLO inference
            frame_rsz = cv2.resize(cv_img, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Create frame package with timestamp
            timestamp = time.time()
            frame_package = {
                'frame_id': frame_id,
                'original_image': cv_img.copy(),
                'processed_image': frame_rsz,
                'timestamp': timestamp
            }
            
            # Put into processed queue
            try:
                if self.processed_image_queue.full():
                    try:
                        self.processed_image_queue.get_nowait()  # Remove oldest
                    except queue.Empty:
                        pass
                self.processed_image_queue.put_nowait(frame_package)
            except queue.Full:
                pass  # Queue is full, drop this frame

    def inference_worker(self):
        """Thread to perform YOLO inference with dynamic model selection"""
        while self.running and not rospy.is_shutdown():
            try:
                # Get preprocessed frame package with timeout
                frame_package = self.processed_image_queue.get(timeout=0.1)
                
                # Run YOLO inference with thread safety and appropriate model
                if self.current_model == "box":
                    # Use box model for backward movement after gripping
                    # rospy.loginfo("Using box model for inference")
                    results = self.model_box.track(frame_package['processed_image'], conf=self.model_box_conf, verbose=False,
                                                    persist=True, tracker="bytetrack.yaml")
                else:
                    # Use detect model for initial detection and alignment
                    results = self.model_detect.track(frame_package['processed_image'], conf=self.model_detect_conf, verbose=False,
                                                    persist=True, tracker="bytetrack.yaml")
                
                # Create synchronized result package
                current_time = time.time()
                result_package = {
                    'frame_id': frame_package['frame_id'],
                    'original_image': frame_package['original_image'],
                    'results': results,
                    'timestamp': current_time,
                    'processing_delay': current_time - frame_package['timestamp'],
                    'model_used': self.current_model  # Track which model was used
                }
                
                # Store latest synchronized frame data
                self.latest_frame_data = result_package
                self.result_timestamp = current_time
                
                # Put results into result queue
                try:
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()  # Remove oldest
                        except queue.Empty:
                            pass
                    self.result_queue.put_nowait(result_package)
                except queue.Full:
                    pass  # Drop result if queue is full
                    
            except queue.Empty:
                continue  # No new processed images, continue loop
            except Exception as e:
                rospy.logerr(f"Error in inference_worker: {e}")

    def moving(self, cx_yolo, cy_yolo, cx_bb, cy_bb):
        """Thread-safe robot movement using new API"""
        with self.robot_lock:
            dx = cx_bb - cx_yolo
            dy = cy_bb - cy_yolo
            K_x, K_y = 0.2, 0.075

            vy = K_x * dx
            vx = K_y * dy
            
            # Calculate pixel error
            error = abs((dx + dy)**2 / 2)
            
            aligned = False
            try:
                # Use new API with bounding box correction
                aligned = self.robot.move_with_bounding_box_correction(
                    vx_pixel=vx, 
                    vy_pixel=vy, 
                    speed=self.correction_speed, 
                    pixel_error=error, 
                    tolerance_pixel=1.0
                )
                
                if aligned:
                    with self.state_lock:
                        self.count_correct += 1
                        rospy.loginfo(f"Aligned with box: {self.count_correct}/{self.target_correct}")
                        if self.count_correct >= self.target_correct:
                            self.robot.stop()
                            self.move_to_box = aligned
                            return aligned
                
            except Exception as e:
                rospy.logwarn(f"Robot movement error: {e}")
                self.move_to_box = False
                return False

    def display_worker(self):
        """Process results and display (synchronized frame-result pairs)"""
        try:
            # Get the latest synchronized frame data
            frame_data = None
            current_time = time.time()
            
            # Try to get fresh results from queue first
            try:
                frame_data = self.result_queue.get_nowait()
                # Update latest frame data if we got fresh ones
                with self.frame_data_lock:
                    self.latest_frame_data = frame_data
                    self.result_timestamp = current_time
            except queue.Empty:
                # Use cached latest frame data
                with self.frame_data_lock:
                    frame_data = self.latest_frame_data
                    # Check if frame data is too old (more than 200ms)
                    if frame_data is None or (current_time - self.result_timestamp) > 0.2:
                        frame_data = None
            
            # If no synchronized data available, skip this frame
            if frame_data is None:
                # Show a blank or loading frame
                loading_img = np.zeros((self.IMG_ORG_H, self.IMG_ORG_W, 3), dtype=np.uint8)
                cv2.putText(loading_img, "Waiting for synchronized frames...", (50, self.IMG_ORG_H//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('ROS YOLO Detector', loading_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                    cv2.destroyAllWindows()
                    return False
                return True
            
            # Use synchronized frame and results
            cv_img = frame_data['original_image']
            results = frame_data['results']
            frame_id = frame_data['frame_id']
            processing_delay = frame_data['processing_delay']
            
            # Convert to display size
            display_img = cv2.resize(cv_img, (self.IMG_ORG_W, self.IMG_ORG_H))
            
            # Process detection results
            target_detected = False
            
            if results is not None:
                for result in results:
                    # Extract YOLO detections
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    target_id = self.target_cube_class if self.current_model == "detect" else 0

                    # Filter detections for target class
                    target_detections = [(box, score) for box, score, class_id in zip(boxes, scores, class_ids) 
                                       if class_id == target_id]

                    if target_detections:
                        target_detected = True
                        # Select detection with highest score
                        best_detection = max(target_detections, key=lambda x: x[1])
                        box, score = best_detection
                        x1, y1, x2, y2 = map(int, box)
                        x1_org = int(x1 * self.width_coef)
                        y1_org = int(y1 * self.height_coef)
                        x2_org = int(x2 * self.width_coef)
                        y2_org = int(y2 * self.height_coef)

                        # Compute YOLO bounding box center
                        cx_yolo = (x1_org + x2_org) / 2
                        cy_yolo = (y1_org + y2_org) / 2

                        # Compute default bounding box center and move robot
                        if self.bounding_box and not self.move_to_box:
                            if self.current_model == "box":
                                # For model_box: horizontal correction only
                                cx_default = self.IMG_ORG_W / 2
                                cy_default = cy_yolo  # Keep current Y position
                            else:
                                # For model_detect: full 2D correction to default bounding box center
                                cx_default, cy_default = self.default_bb_center

                            self.moving(cx_yolo, cy_yolo, cx_default, cy_default)

                        # Draw YOLO bounding box
                        label = f'Class {CLASS[self.target_cube_class]} {score:.2f}'
                        cv2.rectangle(display_img, (x1_org, y1_org), (x2_org, y2_org), (0, 255, 0), 2)
                        cv2.putText(display_img, label, (x1_org, y1_org-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if not target_detected and not self.move_to_box:
                # No target detected, stop the robot safely
                with self.robot_lock:
                    try:
                        self.bot.set_motor(0, 0, 0, 0)
                    except Exception as e:
                        rospy.logwarn(f"Failed to stop robot: {e}")

            # Draw default bounding box
            if self.bounding_box:
                x_min, y_min, x_max, y_max = self.bounding_box
                cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Calculate and display FPS
            now = time.perf_counter()
            dt = now - self.prev_time
            self.prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            self.fps.pop(0)
            self.fps.append(fps)
            fps_text = f"FPS: {sum(self.fps)/len(self.fps):.2f}"
            cv2.putText(display_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add status and synchronization info
            current_model = self.current_model
            object_gripped = self.object_gripped
            moving_backward = self.moving_backward
            
            status_text = f"Move to box: {self.move_to_box}, Move gripper: {self.move_gripper}"
            cv2.putText(display_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display model and state information
            model_text = f"Model: {current_model}, Gripped: {object_gripped}, Backward: {moving_backward}"
            cv2.putText(display_img, model_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            sync_text = f"Frame ID: {frame_id}, Delay: {processing_delay*1000:.1f}ms"
            model_used = frame_data.get('model_used', 'unknown')
            sync_text += f", Used: {model_used}"
            cv2.putText(display_img, sync_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display the result
            cv2.imshow('ROS YOLO Detector', display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.running = False
                cv2.destroyAllWindows()
                return False
            return True
                    
        except Exception as e:
            rospy.logerr(f"Error in display_worker: {e}")
            return True

    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down detector...")
        self.running = False
        
        # Release USB camera if it's open
        with self.capture_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        # Stop robot first
        with self.robot_lock:
            try:
                self.robot.stop()
                self.bot.set_motor(0, 0, 0, 0)
            except Exception as e:
                rospy.logwarn(f"Error stopping robot during shutdown: {e}")
        
        # Wait for threads to finish
        if hasattr(self, 'preprocess_thread') and self.preprocess_thread.is_alive():
            self.preprocess_thread.join(timeout=1.0)
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        
        # Clean up OpenCV
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Final robot cleanup
        try:
            self.robot.reset_controllers()
            self.bot.set_auto_report_state(False, forever=False)
            self.bot.clear_auto_report_data()
        except Exception as e:
            rospy.logwarn(f"Error during final robot cleanup: {e}")

    def spin(self):
        """Main detection and control loop with movement handling"""
        try:
            while not rospy.is_shutdown() and self.running:
                # Update any active movement
                if self.movement_active:
                    movement_complete = self.update_movement()
                    if not movement_complete:
                        # Movement still in progress, continue updating
                        rospy.sleep(0.01)
                        continue
                
                if not self.move_to_box:
                    # Continue detecting & moving
                    if not self.display_worker():
                        break  # User pressed 'q' or error occurred
                elif not self.move_gripper:
                    # Alignment just achieved - perform pick operation
                    try:
                        # Stop robot defensively
                        with self.robot_lock:
                            self.robot.stop()
                        
                        # Close display during pick operation
                        cv2.destroyAllWindows()
                        
                        # Call pick operation
                        rospy.loginfo("Starting pick operation...")
                        self.manipulator.pick_object()
                        
                        # Mark object as gripped and switch to box model for backward movement
                        with self.state_lock:
                            self.object_gripped = True
                        
                        self.move_gripper = True
                        rospy.loginfo("Pick operation completed")
                        
                    except Exception as e:
                        rospy.logerr(f"Failed pick operation: {e}")
                        self.move_gripper = True  # Prevent retry loop
                else:
                    # Already picked - exit detection loop
                    rospy.loginfo("Pick completed, exiting detection loop")
                    break
                    
                # Small delay to prevent CPU overload
                rospy.sleep(0.005)  # Reduced from 0.01 for better responsiveness
                
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        except Exception as e:
            rospy.logerr(f"Error in spin loop: {e}")
        finally:
            pass


if __name__ == '__main__':
    detector = None
    distance = 4000
    step_back = 1000  # Distance to move back after pick
    try:
        # Create detector
        detector = RosYoloDetector()
        
        # Store initial yaw for later use
        initial_yaw = detector.yaw
        
        # Move forward to search area using new movement system
        rospy.loginfo("Moving forward to search area...")
        detector.start_movement(distance, 0, detector.speed)
        
        # Wait for forward movement to complete in a loop
        while detector.movement_active and not rospy.is_shutdown():
            detector.update_movement()
            rospy.sleep(0.01)
        
        # Start detection and pick sequence
        rospy.loginfo("Starting object detection...")
        detector.spin()
        # detector.move_gripper = True
        
        # After successful pick, continue with navigation
        if detector.move_gripper:
            rospy.loginfo("Object picked successfully, continuing navigation...")
            
            # Mark that we are moving backward for box model usage
            with detector.state_lock:
                detector.moving_backward = True
            
            # Move backward with box model correction (horizontal alignment only)
            rospy.loginfo("Moving backward with box model correction...")
            try:
                # Use regular movement system with integrated correction
                detector.start_movement(step_back, 180, 30)  # Move back slightly
                
                # Wait for backward movement to complete with integrated correction
                while detector.movement_active and not rospy.is_shutdown():
                    detector.update_movement(30)
                    rospy.sleep(0.01)
                
                rospy.loginfo("Backward movement with box model correction completed")
            except Exception as e:
                rospy.logerr(f"Error during backward movement: {e}")

            if detector.switch_to_model("box"):
                rospy.loginfo("Switched to box model for backward movement")

            # Move back to original position
            rospy.loginfo("Returning to original position...")
            detector.start_movement(distance - step_back + 200, 180, 30)

            # Wait for return movement to complete
            while detector.movement_active and not rospy.is_shutdown():
            # while True:
                detector.update_movement(30)
                rospy.sleep(0.01)
            
            # Mark that backward movement is complete
            with detector.state_lock:
                detector.moving_backward = False
            
            # Stop at drop location
            rospy.loginfo("Arrived at drop location...")
            detector.robot.stop()
            
            # Release object
            rospy.loginfo("Releasing object...")
            detector.manipulator.release_object()
            rospy.loginfo("Mission completed successfully!")
        else:
            rospy.logwarn("Object not picked, mission incomplete")
            
    except KeyboardInterrupt:
        if detector:
            detector.running = False
            detector.robot.stop()
        rospy.loginfo("Keyboard interrupt received, shutting down...")
    finally:
        print("Cleaning up...")
        if detector:
            try:
                detector.shutdown()
                time.sleep(0.1)
                del detector
            except Exception as e:
                rospy.logerr(f"Error during cleanup: {e}")
        print("Cleanup completed")
