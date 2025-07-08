#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue

CLASS = {0: "Blue", 1: "Green", 2: "Red", 3: "Yellow"}

class YoloErrorPrinter:
    def __init__(self):
        # Initialize node
        rospy.init_node('yolo_error_printer', anonymous=True)

        # Create YOLO model (exported format, no fuse)
        self.model = YOLO(
            '/root/YOLOv11/src/yolov11/scripts/YOLO_weights/best_ncnn_model',
            task='detect'
        )
        # FUSE removed: not supported for exported models

        self.IMG_SIZE = 128
        self.IMG_ORG_H = 480
        self.IMG_ORG_W = 640

        # Image scaling coefficients
        self.height_coef = self.IMG_ORG_H / self.IMG_SIZE
        self.width_coef = self.IMG_ORG_W / self.IMG_SIZE

        # Initialize bounding box and target class
        self.init_bounding_box = None
        self.init_center = None
        self.set_init_bounding_box(300, 370, 275, 240)  # xmin, xmax, ymin, ymax
        self.target_class = 0  # Blue object

        # Thread-safe queues
        self.raw_image_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        # Start worker threads
        self.preprocess_thread = threading.Thread(
            target=self.preprocess_worker, daemon=True
        )
        self.preprocess_thread.start()

        self.inference_thread = threading.Thread(
            target=self.inference_worker, daemon=True
        )
        self.inference_thread.start()

        # Subscribe to image topic
        rospy.Subscriber(
            '/camera/rgb/image_raw', Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Timer for printing data
        self.pub_timer = rospy.Timer(
            rospy.Duration(0.1), self.print_data
        )  # 10 Hz

        rospy.loginfo("YOLO Error Printer node started")
        rospy.loginfo(
            f"Initial center: ({self.init_center[0]:.1f}, {self.init_center[1]:.1f})"
        )

    def set_init_bounding_box(self, xmin, xmax, ymin, ymax):
        """Set the initial target bounding box and calculate its center"""
        self.init_bounding_box = (xmin, ymin, xmax, ymax)
        self.init_center = [
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0
        ]
        rospy.loginfo(
            f"Init bounding box set: ({xmin}, {ymin}, {xmax}, {ymax})"
        )
        rospy.loginfo(
            f"Init center: ({self.init_center[0]:.1f}, {self.init_center[1]:.1f})"
        )

    def image_callback(self, img_msg):
        """Fast image callback - minimal processing"""
        try:
            if img_msg.encoding in ('bgr8', 'rgb8'):
                cv_img = np.frombuffer(
                    img_msg.data, dtype=np.uint8
                ).reshape(
                    (img_msg.height, img_msg.width, 3)
                )
                if img_msg.encoding == 'rgb8':
                    cv_img = cv2.cvtColor(
                        cv_img, cv2.COLOR_RGB2BGR
                    )
            else:
                return
            # Drop old frames, keep latest
            while not self.raw_image_queue.empty():
                self.raw_image_queue.get_nowait()
            self.raw_image_queue.put_nowait(cv_img)
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def preprocess_worker(self):
        """Thread to handle image preprocessing"""
        while not rospy.is_shutdown():
            try:
                if not self.raw_image_queue.empty():
                    cv_img = self.raw_image_queue.get_nowait()
                    frame_rsz = cv2.resize(
                        cv_img, (self.IMG_SIZE, self.IMG_SIZE)
                    )
                    # Keep only latest
                    while not self.image_queue.empty():
                        self.image_queue.get_nowait()
                    self.image_queue.put_nowait(frame_rsz)
                else:
                    rospy.sleep(0.001)
            except Exception as e:
                rospy.logerr(f"Preprocessing error: {e}")

    def inference_worker(self):
        """Thread to perform YOLO inference"""
        while not rospy.is_shutdown():
            try:
                if not self.image_queue.empty():
                    frame_rsz = self.image_queue.get_nowait()
                    results = self.model(
                        frame_rsz, conf=0.5, verbose=False, device='cpu'
                    )
                    while not self.result_queue.empty():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait((frame_rsz, results))
                else:
                    rospy.sleep(0.001)
            except Exception as e:
                rospy.logerr(f"Inference error: {e}")

    def get_best_detection(self):
        """Get the best detection for the target class"""
        if self.result_queue.empty():
            return None, None
        try:
            frame_rsz, results = self.result_queue.get_nowait()
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                dets = [
                    (box, score)
                    for box, score, c in zip(boxes, scores, cls_ids)
                    if c == self.target_class
                ]
                if dets:
                    box, score = max(dets, key=lambda x: x[1])
                    x1, y1, x2, y2 = box
                    x1 *= self.width_coef; x2 *= self.width_coef
                    y1 *= self.height_coef; y2 *= self.height_coef
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    return [cx, cy], score
            return None, None
        except queue.Empty:
            return None, None

    def print_data(self, event):
        """Timer callback to print detection data"""
        obj_center, conf = self.get_best_detection()
        print(f"[Init]    Center: ({self.init_center[0]:.1f}, {self.init_center[1]:.1f})")
        if obj_center is not None:
            ex = self.init_center[0] - obj_center[0]
            ey = self.init_center[1] - obj_center[1]
            print(f"[Detect] Center: ({obj_center[0]:.1f}, {obj_center[1]:.1f}), conf={conf:.2f}")
        else:
            print("[Detect] No detection!")
        print("-" * 60)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = YoloErrorPrinter()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo("Shutting down YOLO Error Printer")
