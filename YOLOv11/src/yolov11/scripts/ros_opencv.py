#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np

class ImageDisplayNode:
    def __init__(self):
        """Initialize the ROS node and set up subscribers and display."""
        # Initialize the ROS node
        rospy.init_node('image_display_node', anonymous=True)

        # Define the image topics to subscribe to
        self.rgb_topic = '/camera/rgb/image_raw'
        self.depth_topic = '/camera/depth/image_raw'
        self.ir_topic = '/camera/ir/image_raw'
        self.depth_to_color_topic = '/camera/depth_to_color/image_raw'
        self.usb_cam = cv2.VideoCapture(0)

        # Initialize the display image (960x1280 for a 2x2 grid of 640x480 images)
        self.display_img = np.zeros((960, 1280, 3), dtype=np.uint8)

        # Set up subscribers for each topic
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.ir_topic, Image, self.ir_callback)
        rospy.Subscriber(self.depth_to_color_topic, Image, self.depth_to_color_callback)

        # Set up a timer to refresh the display at ~30 FPS (every 0.033 seconds)
        rospy.Timer(rospy.Duration(0.033), self.display_callback)

        rospy.loginfo("Image display node started, waiting for images...")

    def rgb_callback(self, img_msg):
        """Process RGB image messages and update the display."""
        # Convert ROS Image message to OpenCV BGR image
        if img_msg.encoding == 'bgr8':
            cv_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, 3))
        elif img_msg.encoding == 'rgb8':
            cv_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, 3))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        else:
            rospy.logwarn(f"Unsupported encoding for RGB image: {img_msg.encoding}")
            return

        # Resize to 640x480 if necessary
        if cv_img.shape[:2] != (480, 640):
            cv_img = cv2.resize(cv_img, (640, 480))

        # Place in top-left quadrant
        self.display_img[0:480, 0:640, :] = cv_img

    def depth_callback(self, img_msg):
        """Process depth image messages and update the display."""
        # Convert ROS Image message to OpenCV image based on encoding
        if img_msg.encoding == '16UC1':
            depth_img = np.frombuffer(img_msg.data, dtype=np.uint16).reshape((img_msg.height, img_msg.width))
            # Normalize to 0-255 for display
            depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
        else:
            rospy.logwarn(f"Unsupported encoding for depth image: {img_msg.encoding}")
            return

        # Resize to 640x480 if necessary
        if depth_bgr.shape[:2] != (480, 640):
            depth_bgr = cv2.resize(depth_bgr, (640, 480))

        # Place in top-right quadrant
        self.display_img[0:480, 640:1280, :] = depth_bgr

    def ir_callback(self, img_msg):
        """Process infrared image messages and update the display."""
        # Convert ROS Image message to OpenCV image based on encoding
        if img_msg.encoding == 'mono8':
            ir_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width))
            ir_bgr = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
        elif img_msg.encoding == 'mono16':
            ir_img = np.frombuffer(img_msg.data, dtype=np.uint16).reshape((img_msg.height, img_msg.width))
            # Normalize to 0-255 for display
            ir_norm = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ir_bgr = cv2.cvtColor(ir_norm, cv2.COLOR_GRAY2BGR)
        else:
            rospy.logwarn(f"Unsupported encoding for IR image: {img_msg.encoding}")
            return

        # Resize to 640x480 if necessary
        if ir_bgr.shape[:2] != (480, 640):
            ir_bgr = cv2.resize(ir_bgr, (640, 480))

        # Place in bottom-left quadrant
        self.display_img[480:960, 0:640, :] = ir_bgr

    def depth_to_color_callback(self, img_msg):
        """Process registered depth image messages and update the display."""
        # Convert ROS Image message to OpenCV image based on encoding
        if self.usb_cam is not None and self.usb_cam.isOpened():
            ret, usb_img = self.usb_cam.read()
            if not ret:
                rospy.logwarn("Failed to read from USB camera.")
                return
            usb_img = cv2.resize(usb_img, (640, 480))

        # Place in bottom-right quadrant
        self.display_img[480:960, 640:1280, :] = usb_img

    def display_callback(self, event):
        """Display the combined image with labels and separators."""
        # Create a copy of the display image to draw overlays
        disp = self.display_img.copy()

        # Draw vertical and horizontal lines to separate quadrants
        cv2.line(disp, (640, 0), (640, 960), (255, 255, 255), 2)
        cv2.line(disp, (0, 480), (1280, 480), (255, 255, 255), 2)

        # Add labels to each quadrant
        cv2.putText(disp, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(disp, "Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(disp, "IR", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(disp, "Depth to Color", (650, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the combined image
        cv2.imshow('Four Images', disp)
        cv2.waitKey(1)

    def spin(self):
        """Keep the node running until shutdown."""
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ImageDisplayNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass