#!/usr/bin/env python3
"""
Professional Mecanum Robot API
A clean, well-organized interface for controlling mecanum wheel robots
with position feedback and precise movement control.
"""

import time
import math
from typing import Tuple, Optional, List, Callable
from Rosmaster_Lib import Rosmaster
from Robot import *


class PIDController:
    """
    A PID controller implementation for closed-loop control systems.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-100, 100)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            output_limits: Tuple of (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error: float) -> float:
        """
        Update PID controller with current error.
        
        Args:
            error: Current error value
            
        Returns:
            Control output value
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        self.integral = max(-50, min(50, self.integral))  # Clamp integral
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits with deadband
        if output > 0:
            if output < 10:
                output = 0
            elif output > self.output_limits[1]:
                output = self.output_limits[1]
        else:
            if output > -10:
                output = 0
            elif output < self.output_limits[0]:
                output = self.output_limits[0]
        
        # Update for next iteration
        self.previous_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class RobotConfiguration:
    """
    Robot hardware configuration parameters.
    """
    
    def __init__(self, debug: bool = False):
        # Debug flag
        self.debug = debug
        
        # Wheel parameters
        self.wheel_diameter_mm = 97.0
        self.wheel_circumference_mm = math.pi * self.wheel_diameter_mm
        
        # Encoder parameters
        self.encoder_resolution = (205/60) * 11  # Adjust based on actual specs
        self.gear_ratio = 56
        self.total_encoder_per_revolution = self.encoder_resolution * self.gear_ratio
        self.mm_per_pulse = self.wheel_circumference_mm / self.total_encoder_per_revolution if self.total_encoder_per_revolution != 0 else 0
        
        # Control limits
        self.max_motor_speed = 100
        self.default_speed_limit = 27
        
        # Bounding box correction parameters
        self.pixel_to_speed_ratio = 40.0  # Normalization factor for pixel error to speed
        self.bb_correction_kx = 0.8  # Gain for X direction correction
        self.bb_correction_ky = 0.8  # Gain for Y direction correction
        
        # Servo positions for manipulator
        self.servo_positions = {
            'home': P1,
            'approach': P2,
            'grasp': P3,
            'carry': P4,
            'release': P5
        }


class SensorInterface:
    """
    Interface for robot sensors including encoders and IMU.
    """
    
    def __init__(self, bot: Rosmaster, config: RobotConfiguration):
        self.bot = bot
        self.config = config
    
    def get_encoder_distances(self) -> List[float]:
        """
        Get distances traveled by each wheel from encoders.
        
        Returns:
            List of distances in mm for each wheel [FL, RL, FR, RR]
        """
        encoder_data = self.bot.get_motor_encoder()
        distances = [enc * self.config.mm_per_pulse for enc in encoder_data]
        return distances
    
    def reset_encoders(self):
        """Reset encoder values to zero."""
        self.bot.reset_flash_value()
    
    def get_current_yaw(self) -> float:
        """
        Get current yaw angle from IMU.
        
        Returns:
            Yaw angle in degrees
        """
        _, _, yaw = self.bot.get_imu_attitude_data(ToAngle=True)
        return yaw
    
    def get_imu_data(self) -> Tuple[float, float, float]:
        """
        Get full IMU attitude data.
        
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        return self.bot.get_imu_attitude_data(ToAngle=True)


class MecanumKinematics:
    """
    Mecanum wheel kinematics calculations.
    """
    
    @staticmethod
    def calculate_wheel_speeds(vx: float, vy: float, vz: float) -> Tuple[float, float, float, float]:
        """
        Calculate individual wheel speeds from desired velocities.
        
        Args:
            vx: Forward/backward velocity
            vy: Left/right velocity  
            vz: Rotational velocity
            
        Returns:
            Tuple of wheel speeds (FL, RL, FR, RR)
        """
        # Mecanum wheel kinematics
        fl = vx - vy - vz  # Front Left
        rl = vx + vy - vz  # Rear Left
        fr = vx + vy + vz  # Front Right
        rr = vx - vy + vz  # Rear Right
        
        return fl, rl, fr, rr
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-180, 180] range.
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Normalized angle in degrees
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle


class MotorController:
    """
    Low-level motor control interface.
    """
    
    def __init__(self, bot: Rosmaster, config: RobotConfiguration):
        self.bot = bot
        self.config = config
    
    def limit_speed(self, speed: float, speed_limit: Optional[float] = None) -> int:
        """
        Apply speed limits to motor commands (matching Robot.py exactly).
        
        Args:
            speed: Desired speed
            speed_limit: Optional speed limit override
            
        Returns:
            Limited speed as integer
        """
        if speed_limit is None:
            speed_limit = self.config.default_speed_limit
            
        # Apply Robot.py logic exactly - corrected version
        if speed >= 0:
            if speed <= speed_limit: 
                speed = speed_limit
            elif speed >= 100:
                speed = 100
        elif speed < 0:
            if speed >= -speed_limit:   
                speed = -speed_limit
            elif speed <= -100:
                speed = -100
                
        return int(speed)
    
    def set_motor_speeds(self, fl: float, rl: float, fr: float, rr: float, speed_limit: Optional[float] = None):
        """
        Set motor speeds with limits applied.
        
        Args:
            fl, rl, fr, rr: Motor speeds for each wheel
            speed_limit: Optional speed limit override
        """
        limited_speeds = [
            self.limit_speed(fl, speed_limit),
            self.limit_speed(rl, speed_limit),
            self.limit_speed(fr, speed_limit),
            self.limit_speed(rr, speed_limit)
        ]
        
        # Debug output
        if self.config.debug:
            print(f"Motor speeds: FL={limited_speeds[0]}, RL={limited_speeds[1]}, FR={limited_speeds[2]}, RR={limited_speeds[3]}")
        
        self.bot.set_motor(*limited_speeds)
    
    def stop_all_motors(self):
        """Stop all motors immediately."""
        self.bot.set_motor(0, 0, 0, 0)


class MecanumRobot:
    """
    Main robot control interface providing high-level movement commands.
    """
    
    def __init__(self, bot: Rosmaster, kp: float = 2.5, ki: float = 0.1, kd: float = 0.8, debug: bool = False):
        """
        Initialize the mecanum robot controller.
        
        Args:
            bot: Rosmaster instance
            kp, ki, kd: PID gains for heading control
            debug: Enable debug output
        """
        self.bot = bot
        self.config = RobotConfiguration(debug=debug)
        self.debug = debug
        self.sensors = SensorInterface(bot, self.config)
        self.motors = MotorController(bot, self.config)
        self.kinematics = MecanumKinematics()
        
        # Control state
        self.target_yaw = 0.0
        self.initial_yaw = None
        self.is_initialized = False
        
        # PID controller for heading
        self.pid_yaw = PIDController(kp, ki, kd, output_limits=(-100, 100))
        
        # Bounding box correction state
        self.bb_correction_enabled = False
        self.bb_error_callback = None  # Function to get bounding box error
        
        if self.debug:
            print(f"Mecanum Robot initialized - Encoder: {self.config.mm_per_pulse:.4f} mm/pulse")
            print(f"Debug mode: ENABLED")
            print(f"PID gains: kp={kp}, ki={ki}, kd={kd}")
    
    def initialize_pose(self, yaw: Optional[float] = None):
        """
        Initialize robot pose reference.
        
        Args:
            yaw: Optional yaw angle, if None uses current yaw
        """
        if not self.is_initialized:
            if yaw is None:
                yaw = self.sensors.get_current_yaw()
            
            self.initial_yaw = yaw
            self.target_yaw = yaw
            self.is_initialized = True
            print(f"Robot pose initialized: Yaw = {yaw:.2f}°")
    
    def _calculate_yaw_error(self, current_yaw: float) -> float:
        """Calculate yaw error with proper angle wrapping."""
        error = self.target_yaw - current_yaw
        return self.kinematics.normalize_angle(error)    
    
    def move(self, vx: float, vy: float, speed: float = 30, use_imu_correction: bool = True, use_bb_correction: bool = False) -> bool:
        """
        Core movement function - move by vx, vy with optional corrections.
        
        Args:
            vx: Movement in x direction (forward/backward in pixels or normalized)
            vy: Movement in y direction (left/right in pixels or normalized)  
            speed: Movement speed (0-100)
            use_imu_correction: Whether to apply IMU heading correction
            use_bb_correction: Whether to apply bounding box correction
            
        Returns:
            True if movement executed successfully
        """
        try:
            # Ensure pose is initialized
            if not self.is_initialized:
                current_yaw = self.sensors.get_current_yaw()
                self.initialize_pose(current_yaw)
            
            # Normalize velocities (matching Robot.py approach exactly)
            vx_norm = vx / 40
            vy_norm = vy / 40
            
            # Base speeds (matching Robot.py exactly)
            base_speed = speed
            vx_cmd = int(base_speed * vx_norm)
            vy_cmd = int(base_speed * vy_norm)
            
            vz_cmd = 0
            
            # Apply IMU heading correction if enabled  
            if use_imu_correction:
                current_yaw = self.sensors.get_current_yaw()
                yaw_error = self._calculate_yaw_error(current_yaw)
                vz_cmd = self.pid_yaw.update(yaw_error)  # Correction term
                
                if self.debug:
                    print(f"IMU correction: yaw={current_yaw:.2f}°, target yaw={self.target_yaw}, error={yaw_error:.2f}°, correction={vz_cmd:.2f}")
            
            # Apply bounding box correction if enabled
            if use_bb_correction:
                bb_vx, bb_vy, bb_error = self._get_bounding_box_correction()
                vx_cmd += int(base_speed * bb_vx)
                vy_cmd += int(base_speed * bb_vy)
                
                if self.debug:
                    print(f"BB correction: dx={bb_vx:.2f}, dy={bb_vy:.2f}, error={bb_error:.2f}")
            
            # Mecanum kinematics with yaw correction (matching Robot.py exactly)
            fl = vx_cmd - vy_cmd - vz_cmd  # Front Left
            fr = vx_cmd + vy_cmd + vz_cmd  # Front Right 
            rl = vx_cmd + vy_cmd - vz_cmd  # Rear Left
            rr = vx_cmd - vy_cmd + vz_cmd  # Rear Right
            
            if self.debug:
                print(f"Movement: vx={vx:.2f}, vy={vy:.2f}, speed={speed}")
                print(f"Commands: vx={vx_cmd}, vy={vy_cmd}, vz={vz_cmd}")
                print(f"Wheels: FL={fl}, RL={rl}, FR={fr}, RR={rr}")
            
            # Limit and apply speeds (matching Robot.py order exactly)
            speeds = [self.motors.limit_speed(v) for v in (fl, rl, fr, rr)]
            self.bot.set_motor(*speeds)
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Movement error: {e}")
            self.bot.set_motor(0, 0, 0, 0)
            return False
    
    def move_with_bounding_box_correction(self, vx_pixel: float, vy_pixel: float, speed: float, 
                                        pixel_error: float, tolerance_pixel: float = 2) -> bool:
        """
        Move robot with bounding box correction (exactly matching Robot.py interface).
        
        Args:
            vx_pixel: Y direction pixel error (forward/backward movement)
            vy_pixel: X direction pixel error (left/right movement)  
            speed: Movement speed (0-100)
            pixel_error: Total pixel error magnitude
            tolerance_pixel: Pixel error tolerance for alignment
            
        Returns:
            True if aligned within tolerance, False if still moving
        """
        try:            
            # Check if already aligned (matching Robot.py exactly)
            if pixel_error <= tolerance_pixel:
                if self.debug:
                    print("Robot have moved to the target!")
                return True
            
            # Normalize pixel movements (matching Robot.py exactly)
            vx_norm = vx_pixel / 40
            vy_norm = vy_pixel / 40
            
            # Compute yaw correction
            yaw = self.sensors.get_current_yaw()
            error = self._calculate_yaw_error(yaw)
            correction = self.pid_yaw.update(error)
            
            # Base speeds
            base_speed = speed
            vx_cmd = int(base_speed * vx_norm)
            vy_cmd = int(base_speed * vy_norm)
            
            # Mecanum kinematics with yaw correction (matching Robot.py exactly)
            fl = vx_cmd - vy_cmd - correction
            fr = vx_cmd + vy_cmd + correction
            rl = vx_cmd + vy_cmd - correction
            rr = vx_cmd - vy_cmd + correction
            
            # Limit and send (matching Robot.py exactly)
            speeds = [self.motors.limit_speed(v) for v in (fl, rl, fr, rr)]
            self.bot.set_motor(*speeds)
            
            if self.debug:
                print(f"Moving with speeds:{speeds} | pixel error:{pixel_error:.2f} | yaw error:{error:.2f}")
            
            return False  # Still moving, not aligned yet
            
        except Exception as e:
            if self.debug:
                print(f"Bounding box correction error: {e}")
            self.bot.set_motor(0, 0, 0, 0)
            return False
    
    def move_distance(self, vx: float, vy: float, dx_target: float, dy_target: float, 
                     current_distance: float, max_speed: float = 30, tolerance_mm: float = 15.0,
                     use_imu_correction: bool = True) -> bool:
        """
        Move robot by vx, vy velocities with distance tracking (simplified version like move()).
        
        Args:
            vx: Movement velocity in x direction (forward/backward)
            vy: Movement velocity in y direction (left/right)
            dx_target: Target distance in x direction (mm)
            dy_target: Target distance in y direction (mm)
            current_distance: Current distance traveled (mm)
            max_speed: Maximum speed (0-100)
            tolerance_mm: Distance tolerance for completion
            use_imu_correction: Whether to apply IMU heading correction
            
        Returns:
            True if target reached within tolerance, False if still moving
        """
        try:
            # Ensure pose is initialized
            if not self.is_initialized:
                current_yaw = self.sensors.get_current_yaw()
                self.initialize_pose(current_yaw)
            
            # Calculate target total distance
            target_distance = math.sqrt(dx_target**2 + dy_target**2)
            
            # Check if target reached
            if current_distance >= (target_distance - tolerance_mm):
                self.bot.set_motor(0, 0, 0, 0)
                if self.debug:
                    print(f"Target distance reached: {current_distance:.1f}mm >= {target_distance:.1f}mm")
                return True
            
            # Normalize velocities (matching Robot.py approach exactly)
            vx_norm = vx / target_distance
            vy_norm = vy / target_distance
            
            # Base speeds (matching Robot.py exactly)
            base_speed = max_speed
            vx_cmd = int(base_speed * vx_norm)
            vy_cmd = int(base_speed * vy_norm)
            
            vz_cmd = 0
            
            # Apply IMU heading correction if enabled  
            if use_imu_correction:
                current_yaw = self.sensors.get_current_yaw()
                yaw_error = self._calculate_yaw_error(current_yaw)
                vz_cmd = self.pid_yaw.update(yaw_error)  # Correction term
                
                if self.debug:
                    print(f"IMU correction: yaw={current_yaw:.2f}°, error={yaw_error:.2f}°, correction={vz_cmd:.2f}")
            
            # Mecanum kinematics with yaw correction (matching Robot.py exactly)
            fl = vx_cmd - vy_cmd - vz_cmd  # Front Left
            fr = vx_cmd + vy_cmd + vz_cmd  # Front Right 
            rl = vx_cmd + vy_cmd - vz_cmd  # Rear Left
            rr = vx_cmd - vy_cmd + vz_cmd  # Rear Right
            
            if self.debug:
                print(f"Distance movement: vx={vx:.2f}, vy={vy:.2f}, distance={current_distance:.1f}/{target_distance:.1f}mm")
                print(f"Commands: vx={vx_cmd}, vy={vy_cmd}, vz={vz_cmd}")
                print(f"Wheels: FL={fl}, RL={rl}, FR={fr}, RR={rr}")
            
            # Limit and apply speeds (matching Robot.py order exactly)
            speeds = [self.motors.limit_speed(v) for v in (fl, rl, fr, rr)]
            self.bot.set_motor(*speeds)
            
            return False  # Still moving, not reached target yet
            
        except Exception as e:
            if self.debug:
                print(f"Distance movement error: {e}")
            self.bot.set_motor(0, 0, 0, 0)
            return False
    
    def rotate_to_angle(self, target_angle: float, max_speed: float = 20, tolerance_deg: float = 1.0) -> bool:
        """
        Rotate to absolute target angle.
        
        Args:
            target_angle: Target absolute angle in degrees
            max_speed: Maximum rotation speed
            tolerance_deg: Angular tolerance in degrees
            
        Returns:
            True if rotation completed
        """
        self.target_yaw = target_angle
        
        while True:
            current_yaw = self.sensors.get_current_yaw()
            yaw_error = self._calculate_yaw_error(current_yaw)
            
            if abs(yaw_error) <= tolerance_deg:
                self.bot.set_motor(0, 0, 0, 0)
                if self.debug:
                    print(f"Rotation complete: {current_yaw:.1f}° (target: {target_angle:.1f}°)")
                return True
            
            # Calculate rotation command
            rotation_cmd = self.pid_yaw.update(yaw_error)
            rotation_speed = max(-max_speed, min(max_speed, rotation_cmd))
            
            # Mecanum rotation kinematics (only rotation, no translation)
            fl = -rotation_speed
            fr = rotation_speed  
            rl = -rotation_speed
            rr = rotation_speed
            
            # Apply speeds
            speeds = [self.motors.limit_speed(v) for v in (fl, rl, fr, rr)]
            self.bot.set_motor(*speeds)
            
            time.sleep(0.02)  # 50Hz control loop
    
    def rotate_relative(self, angle_deg: float, max_speed: float = 20, tolerance_deg: float = 1.0) -> bool:
        """
        Rotate relative to current heading.
        
        Args:
            angle_deg: Relative angle in degrees (positive = CCW)
            max_speed: Maximum rotation speed
            tolerance_deg: Angular tolerance in degrees
            
        Returns:
            True if rotation completed
        """
        current_yaw = self.sensors.get_current_yaw()
        target = self.kinematics.normalize_angle(current_yaw + angle_deg)
        return self.rotate_to_angle(target, max_speed, tolerance_deg)
    
    def stop(self):
        """Stop all robot motion."""
        self.motors.stop_all_motors()
    
    def reset_controllers(self):
        """Reset all control state."""
        self.pid_yaw.reset()
        self.is_initialized = False


class ManipulatorController:
    """
    Robot arm/manipulator control interface.
    """
    
    def __init__(self, bot: Rosmaster, config: RobotConfiguration):
        self.bot = bot
        self.config = config
    
    def move_to_position(self, position_name: str, run_time: int = 4000):
        """
        Move manipulator to named position.
        
        Args:
            position_name: Name of position from config
            run_time: Movement time in ms
        """
        if position_name in self.config.servo_positions:
            angles = self.config.servo_positions[position_name]
            self.bot.set_uart_servo_angle_array(angles, run_time=run_time)
            print(f"Moving manipulator to '{position_name}' position")
        else:
            print(f"Unknown position: {position_name}")
    
    def pick_object(self):
        """Execute object picking sequence."""
        print("Executing pick sequence...")
        
        self.move_to_position('home', 4000)
        time.sleep(2.0)
        
        self.move_to_position('approach', 4000)
        time.sleep(2.0)
        
        # Close gripper
        self.bot.set_uart_servo_angle(6, 135, run_time=4000)
        time.sleep(2.0)

        self.move_to_position('carry', 4000)
        time.sleep(2.0)

        self.bot.set_uart_servo_angle(2, 80, run_time=4000)

        print("Pick sequence complete")
    
    def release_object(self):
        """Execute object release sequence."""
        print("Executing release sequence...")
        
        self.move_to_position('release', 4000)
        time.sleep(5.0)
        
        self.bot.set_uart_servo_angle(6, 30, run_time=500)
        time.sleep(1.0)

        self.move_to_position('home', 4000)
        
        print("Release sequence complete")


# High-level convenience functions
def create_robot(kp: float = 5.0, ki: float = 1.0, kd: float = 0.0, debug: bool = False) -> Tuple[MecanumRobot, ManipulatorController]:
    """
    Create and initialize a complete robot system.
    
    Args:
        kp, ki, kd: PID gains for heading control
        debug: Enable debug output
        
    Returns:
        Tuple of (robot_controller, manipulator_controller)
    """
    # Initialize hardware
    bot = Rosmaster()
    bot.create_receive_threading()
    bot.set_car_type(0x02)
    bot.clear_auto_report_data()
    bot.set_auto_report_state(True, forever=False)
    bot.reset_flash_value()
    
    # Stop motors and wait for stabilization
    bot.set_motor(0, 0, 0, 0)
    time.sleep(12.5)
    
    # Create controllers with debug support
    robot = MecanumRobot(bot, kp=kp, ki=ki, kd=kd, debug=debug)
    manipulator = ManipulatorController(bot, robot.config)
    
    # Initialize pose
    robot.initialize_pose()
    
    print("Robot system initialized and ready")
    if debug:
        print("Debug mode: ENABLED")
    return robot, manipulator


def cleanup_robot(robot: MecanumRobot):
    """
    Safely shutdown robot system.
    
    Args:
        robot: Robot controller instance
    """
    print("Shutting down robot system...")
    robot.stop()
    robot.reset_controllers()
    robot.bot.set_auto_report_state(False, forever=False)
    robot.bot.clear_auto_report_data()
    print("Robot system shutdown complete")


if __name__ == "__main__":
    # Example usage with both IMU and Bounding Box corrections
    try:
        # Create robot with debug enabled
        robot, manipulator = create_robot(debug=True)
        
        print("\n=== Enhanced Robot API Demo with Debug ===")
        
        # Example bounding box error function (replace with actual implementation)
        def get_bb_error():
            # This should return (dx_pixel, dy_pixel, total_error) from your vision system
            # For demo purposes, returning dummy values
            return 0.0, 0.0, 0.0
        
        # Enable bounding box correction
        robot.set_bounding_box_correction(get_bb_error)
        
        # Example 1: Move forward with IMU correction only
        print("1. Moving forward 4000mm with IMU correction only...")
        robot.move_distance(4000, 0, max_speed=35, use_imu_correction=True, use_bb_correction=False)
        time.sleep(2)
        
        # Example 2: Move backward to starting position
        print("2. Moving backward 4000mm to return to start...")
        robot.move_distance(-4000, 0, max_speed=35, use_imu_correction=True, use_bb_correction=False)
        time.sleep(2)
        
        # Example 2: Move with both IMU and bounding box correction
        # print("2. Moving with both IMU and BB correction...")
        # robot.move_distance(0, 300, max_speed=35, use_imu_correction=True, use_bb_correction=True)
        # time.sleep(1)
        
        # Example 3: Pure bounding box alignment (visual servoing)
        # print("3. Aligning with bounding box target...")
        # robot.move_to_target_with_bb_correction(max_speed=25, bb_tolerance_pixel=2.0)
        # time.sleep(1)
        
        # Example 4: Legacy bounding box correction interface
        # print("4. Using legacy BB correction interface...")
        # success = robot.move_with_bounding_box_correction(
        #     vx_pixel=10.0, vy_pixel=5.0, speed=30, pixel_error=15.0, tolerance_pixel=2.0
        # )
        time.sleep(1)
        
        # print("5. Rotation (no corrections)...")
        # robot.rotate_relative(90, max_speed=30)
        time.sleep(1)
        
        print("6. Manipulator operations...")
        manipulator.pick_object()
        time.sleep(1)
        manipulator.release_object()
        
        print("\n=== Enhanced Demo Complete ===")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        
    finally:
        if 'robot' in locals():
            cleanup_robot(robot)
