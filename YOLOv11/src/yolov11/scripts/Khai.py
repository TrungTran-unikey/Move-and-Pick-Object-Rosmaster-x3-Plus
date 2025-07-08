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
from Robot import *

CLASS = {0: "Blue", 
         1: "Green", 
         2: "Red", 
         3: "Yellow"}

# CLASS_ID = {v.lower(): k for k, v in CLASS.items()}

class RosYoloDetector:
    def __init__(self):
        # Initialize node
        self.target_box_class = 2
        self.speed = 35.0
        rospy.init_node('yolov11', anonymous=True)

        # Create YOLO model
        self.model = YOLO('/root/YOLOv11/src/yolov11/scripts/YOLO_weights/best_ncnn_model', task='detect')
        self.IMG_SIZE = 256
        self.IMG_ORG_H = 480
        self.IMG_ORG_W = 640

        # Initialize the bot
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
        self.controller = MecanumPositionController(self.bot, kp=5.0, ki=1.0, kd=0.0)
        _, _, self.yaw = self.bot.get_imu_attitude_data(ToAngle=True)
        self.controller.set_initial_pose(self.yaw)
        print(f"Initial Yaw: {self.yaw:.2f}°")

        # Thread synchronization
        self.robot_lock = Lock()  # Protect robot control operations
        self.model_lock = Lock()  # Protect YOLO model access
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

        # Gripper variable
        self.move_gripper = False

        # Thread-safe queues - optimized for low latency
        self.raw_image_queue = queue.Queue(maxsize=1)  # Raw images from callback - keep only latest
        self.processed_image_queue = queue.Queue(maxsize=1)  # Preprocessed images - keep only latest
        self.result_queue = queue.Queue(maxsize=1)  # Inference results - keep only latest

        # Current image for display (thread-safe)
        self.current_image = None
        self.image_lock = Lock()
        
        # Latest detection results for immediate display
        self.latest_results = None
        self.results_lock = Lock()
        self.result_timestamp = 0

        # Start processing threads
        self.preprocess_thread = threading.Thread(target=self.preprocess_worker, daemon=True)
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        
        self.preprocess_thread.start()
        self.inference_thread.start()

        self.count_correct = 0 # Count of correct alignments
        self.target_correct = 10 # Number of correct alignments before stopping

        # Subscribe to image topic
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback,
                         queue_size=1, buff_size=2**24)
        rospy.loginfo("yolov11 node started, waiting for images...")

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
        if not self.running:
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
                
                # Resize for YOLO inference
                frame_rsz = cv2.resize(cv_img, (self.IMG_SIZE, self.IMG_SIZE))
                
                # Store current image for display (thread-safe)
                with self.image_lock:
                    self.current_image = cv_img.copy()
                
                # Put into processed queue
                try:
                    if self.processed_image_queue.full():
                        try:
                            self.processed_image_queue.get_nowait()  # Remove oldest
                        except queue.Empty:
                            pass
                    self.processed_image_queue.put_nowait(frame_rsz)
                except queue.Full:
                    pass  # Drop frame if queue is full
                    
            except queue.Empty:
                continue  # No new images, continue loop
            except Exception as e:
                rospy.logerr(f"Error in preprocess_worker: {e}")

    def inference_worker(self):
        """Thread to perform YOLO inference"""
        while self.running and not rospy.is_shutdown():
            try:
                # Get preprocessed image with timeout
                frame_rsz = self.processed_image_queue.get(timeout=0.1)
                
                # Run YOLO inference with thread safety
                with self.model_lock:
                    results = self.model.track(frame_rsz, conf=0.4, verbose=False, # stream=True,
                                             persist=True, tracker="bytetrack.yaml")
                
                # Store latest results with timestamp for immediate access
                current_time = time.time()
                with self.results_lock:
                    self.latest_results = results
                    self.result_timestamp = current_time
                
                # Put results into result queue
                try:
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()  # Remove oldest
                        except queue.Empty:
                            pass
                    self.result_queue.put_nowait(results)
                except queue.Full:
                    pass  # Drop result if queue is full
                    
            except queue.Empty:
                continue  # No new processed images, continue loop
            except Exception as e:
                rospy.logerr(f"Error in inference_worker: {e}")

    def moving(self, cx_yolo, cy_yolo, cx_bb, cy_bb):
        """Thread-safe robot movement"""
        with self.robot_lock:
            dx = cx_bb - cx_yolo
            dy = cy_bb - cy_yolo
            K_x, K_y = 0.2, 0.2
            vy = K_x * dx
            vx = K_y * dy
            error = abs((dx + dy)**2 / 2)
            aligned = False
            try:
                aligned = self.controller.move_with_correction(vx, vy, speed=self.speed, pixel_error=error, tolerance_pixel=1)
                if aligned:
                    with self.state_lock:
                        self.count_correct += 1
                        rospy.loginfo(f"Aligned with box: {self.count_correct}/{self.target_correct}")
                        if self.count_correct >= self.target_correct:
                            self.controller.stop_all_motors()
                            self.move_to_box = aligned
                            return aligned
                
            except Exception as e:
                rospy.logwarn(f"Robot movement error: {e}")
                self.move_to_box = False
                return False

    def display_worker(self):
        """Process results and display (optimized for low latency)"""
        try:
            # Get current image
            with self.image_lock:
                if self.current_image is None:
                    return True
                cv_img = self.current_image.copy()
            
            # Convert to display size
            display_img = cv2.resize(cv_img, (self.IMG_ORG_W, self.IMG_ORG_H))
            
            # Use latest results directly for minimal latency
            target_detected = False
            current_time = time.time()
            
            # Try to get fresh results from queue first
            try:
                results = self.result_queue.get_nowait()
                # Update latest results if we got fresh ones
                with self.results_lock:
                    self.latest_results = results
                    self.result_timestamp = current_time
            except queue.Empty:
                # Use cached latest results
                with self.results_lock:
                    results = self.latest_results
                    # Check if results are too old (more than 100ms)
                    if results is None or (current_time - self.result_timestamp) > 0.1:
                        results = None
                
            if results is not None:
                for result in results:
                    # Extract YOLO detections
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    # Filter detections for target class
                    target_detections = [(box, score) for box, score, class_id in zip(boxes, scores, class_ids) 
                                       if class_id == self.target_box_class]

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
                            cx_default, cy_default = self.default_bb_center
                            self.moving(cx_yolo, cy_yolo, cx_default, cy_default)

                        # Draw YOLO bounding box
                        label = f'Class {CLASS[self.target_box_class]} {score:.2f}'
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

            # Add status text
            status_text = f"Move to box: {self.move_to_box}, Move gripper: {self.move_gripper}"
            cv2.putText(display_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
        
        # Stop robot first
        with self.robot_lock:
            try:
                self.controller.stop_all_motors()
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
            self.controller.reset_controllers()
            self.bot.set_auto_report_state(False, forever=False)
            self.bot.clear_auto_report_data()
        except Exception as e:
            rospy.logwarn(f"Error during final robot cleanup: {e}")

    def spin(self):
        """Main detection and control loop"""
        try:
            while not rospy.is_shutdown() and self.running:
                if not self.move_to_box:
                    # Continue detecting & moving
                    if not self.display_worker():
                        break  # User pressed 'q' or error occurred
                elif not self.move_gripper:
                    # Alignment just achieved - perform pick operation
                    try:
                        # Stop robot defensively
                        with self.robot_lock:
                            self.controller.stop_all_motors()
                        
                        # Close display during pick operation
                        cv2.destroyAllWindows()
                        
                        # Call pick operation
                        rospy.loginfo("Starting pick operation...")
                        pick_object(self.bot)
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
            # self.shutdown()
            pass


if __name__ == '__main__':
    detector = None
    distance = 1500
    try:
        # Create detector
        detector = RosYoloDetector()
        
        # Store initial yaw for later use
        initial_yaw = detector.yaw
        
        # Move forward to search area
        rospy.loginfo("Moving forward to search area...")
        move_forward_distance(detector.controller, distance, speed=detector.speed)
        
        # Start detection and pick sequence
        rospy.loginfo("Starting object detection...")
        detector.spin()
        
        # After successful pick, continue with navigation
        if detector.move_gripper:
            rospy.loginfo("Object picked successfully, continuing navigation...")
            
            # Rotate 180 degrees
            rospy.loginfo("Rotating 180 degrees...")
            rotate_to_absolute(detector.controller, angle_deg=initial_yaw + 180, speed=detector.speed)
            
            # Move forward to drop location
            rospy.loginfo("Moving to drop location...")
            detector.controller.target_yaw = initial_yaw + 180
            move_forward_distance(detector.controller, distance*2+300, speed=detector.speed)
            
            # Release object
            rospy.loginfo("Releasing object...")
            release_object(detector.bot)
            rospy.loginfo("Mission completed successfully!")
        else:
            rospy.logwarn("Object not picked, mission incomplete")
            
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
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
