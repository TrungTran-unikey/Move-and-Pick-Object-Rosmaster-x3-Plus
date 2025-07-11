#!/usr/bin/env python3
import time
from Rosmaster_Lib import Rosmaster
import math

P1 = [90, 180, 0, 0, 90, 30]
P2 = [90, 36, 61, 55, 90, 30]
P3 = [90, 19, 80, 60, 90, 30]
P4 = [90, 180, 0, 0, 90, 135]
P5 = [1, 0, 90, 90, 90, 135]

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def update(self, error: float):
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        # Prevent integral windup
        if self.integral > 50:
            self.integral = 50
        elif self.integral < -50:
            self.integral = -50
            
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply custom output limits with deadband: (-100, -10) + (10, 100)
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
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

class EncoderPositionController:
    def __init__(self, bot: Rosmaster):
        self.bot = bot
        self.wheel_diameter = 97.0  # mm (đường kính bánh xe)
        self.wheel_circumference = math.pi * self.wheel_diameter  # chu vi bánh xe
        # encoder_resolution: pulses per motor shaft revolution before gearbox
        # Example: if encoder gives 205 RPM at 60V? Adjust based on actual spec
        self.encoder_resolution = (205/60) * 11  # placeholder, cần hiệu chỉnh theo thực tế
        self.gear_ratio = 56  # tỷ số truyền
        self.total_encoder_per_revolution = self.encoder_resolution * self.gear_ratio
        
        # Tính mm per encoder pulse
        self.mm_per_pulse = self.wheel_circumference / self.total_encoder_per_revolution if self.total_encoder_per_revolution != 0 else 0
        
        print(f"Encoder setup: {self.mm_per_pulse:.4f} mm per pulse")
        
    def get_encoder_distances(self):
        """Lấy khoảng cách di chuyển từ encoder (mm)"""
        encoder_data = self.bot.get_motor_encoder()
        distances = []
        for enc in encoder_data:
            distance_mm = enc * self.mm_per_pulse
            distances.append(distance_mm)
        return distances

class MecanumPositionController:
    def __init__(self, bot: Rosmaster, kp=2.5, ki=0.1, kd=0.8):
        self.bot = bot
        self.target_yaw = 0.0
        self.initial_yaw = None
        self.is_initialized = False
        
        # PID controller cho yaw
        self.pid_yaw = PIDController(kp, ki, kd, output_limits=(-100, 100))
        
        # Encoder controller
        self.encoder_controller = EncoderPositionController(bot)
        
    def normalize_angle(self, angle: float):
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def calculate_yaw_error(self, current_yaw: float):
        """Calculate yaw error with proper angle wrapping"""
        error = self.target_yaw - current_yaw
        return self.normalize_angle(error)
    
    def limit_speed(self, speed, speed_limit=25):
        """Limit speed to range (-100, 100)"""
        if speed >= 0:
            if speed <= speed_limit: 
                speed = speed_limit
            elif speed >=100:
                speed = 100
        elif speed < 0:
            if speed >= -speed_limit:   
                speed = -speed_limit
            elif speed <= -100:
                speed = -100
        return speed
    
    def set_initial_pose(self, current_yaw):
        """Set the initial pose of the robot as the target reference"""
        if not self.is_initialized:
            self.initial_yaw = current_yaw
            self.target_yaw = current_yaw
            self.is_initialized = True
            print(f"Initial robot pose captured: Yaw = {current_yaw:.2f}°")
    
    def get_current_yaw(self):
        """Return current yaw from IMU"""
        _, _, yaw = self.bot.get_imu_attitude_data(ToAngle=True)
        return yaw

    def move_with_correction(self, vx_pixel: float, vy_pixel: float, speed: float, pixel_error: float = None, tolerance_pixel=2):
        """
        Move the robot with given velocity components (vx_mm, vy_mm) at a percent speed,
        applying yaw correction to maintain initial heading.
        """
        # Ensure initial pose is set
        # current_yaw = self.get_current_yaw()
        # self.set_initial_pose(current_yaw)
        # distances = self.encoder_controller.get_encoder_distances()
        # differences = [diff/max(distances) for diff in distances]

        vx_norm = vx_pixel / 40
        vy_norm = vy_pixel / 40

        try:
            # Check duration
            if pixel_error <= tolerance_pixel:
                # print("Robot have moved to the target!")
                return True

            # Compute yaw correction
            yaw = self.get_current_yaw()
            error = self.calculate_yaw_error(yaw)
            correction = self.pid_yaw.update(error)

            # Base speeds
            base_speed = speed
            vx_cmd = int(base_speed * vx_norm)
            vy_cmd = int(base_speed * vy_norm)

            # Mecanum kinematics with yaw correction
            fl = vx_cmd - vy_cmd - correction
            fr = vx_cmd + vy_cmd + correction
            rl = vx_cmd + vy_cmd - correction
            rr = vx_cmd - vy_cmd + correction

            # Limit and send
            speeds = [self.limit_speed(v) for v in (fl, rl, fr, rr)]
            self.bot.set_motor(*speeds)
            print(f"Moving with speeds:{speeds} | pixel error:{pixel_error:.2f} | yaw error:{error:.2f}")
            # time.sleep(0.07)
            # self.bot.set_motor(0, 0, 0, 0)
        except:
            self.bot.set_motor(0, 0, 0, 0)
            return False
    
    def move_distance(self, vx_mm, vy_mm, target_distance_mm, max_speed=30, tolerance_mm=15.0):
        """
        Di chuyển robot một khoảng cách cụ thể
        """
        print(f"Moving to distance: {target_distance_mm}mm (tolerance: ±{tolerance_mm}mm)")
        
        # Tính tỷ lệ vận tốc
        motion_magnitude = math.sqrt(vx_mm**2 + vy_mm**2)
        if motion_magnitude == 0:
            print("No movement commanded")
            return
            
        # Normalize direction vectors
        vx_norm = vx_mm / motion_magnitude
        vy_norm = vy_mm / motion_magnitude
        
        start_time = time.time()
        # reached_target = False
        
        while True:
            # Lấy dữ liệu encoder hiện tại
            distances = self.encoder_controller.get_encoder_distances()
            differences = [diff/max(distances) if max(distances) > 0 else 1 for diff in distances]
            
            # Tính khoảng cách trung bình đã di chuyển
            avg_distance = sum(abs(d) for d in distances) / len(distances)
            
            # Tính khoảng cách còn lại
            remaining_distance = target_distance_mm - avg_distance
            
            # Kiểm tra đã đến đích chưa
            if abs(remaining_distance) <= tolerance_mm:
                # reached_target = True
                self.bot.set_motor(0, 0, 0, 0)
                break
            
            # Điều chỉnh tốc độ dựa trên khoảng cách còn lại
            if abs(remaining_distance) < 50:  # Giảm tốc khi gần đích
                speed_factor = max(0.3, abs(remaining_distance) / 50.0)
            else:
                speed_factor = 1.0
                
            current_speed = max_speed * speed_factor
            
            # Tính vận tốc cho từng hướng
            base_vx = int(current_speed * vx_norm)
            base_vy = int(current_speed * vy_norm)
            
            # Lấy góc yaw hiện tại và điều chỉnh
            yaw = self.get_current_yaw()
            yaw_error = self.calculate_yaw_error(yaw)
            yaw_correction = self.pid_yaw.update(yaw_error)
            
            # Tính tốc độ bánh xe theo kinematic mecanum
            wheel_fl = base_vx / differences[0] - base_vy - yaw_correction
            wheel_fr = base_vx / differences[1] + base_vy + yaw_correction
            wheel_rl = base_vx / differences[2] + base_vy - yaw_correction
            wheel_rr = base_vx / differences[3] - base_vy + yaw_correction
            
            # Giới hạn tốc độ
            wheel_speeds = [
                self.limit_speed(wheel_fl),
                self.limit_speed(wheel_rl),
                self.limit_speed(wheel_fr),
                self.limit_speed(wheel_rr)
            ]
            
            # Áp dụng tốc độ cho motors
            self.bot.set_motor(*wheel_speeds)
            
            # Debug info
            elapsed = time.time() - start_time
            if int(elapsed * 10) % 5 == 0:  # Print every 0.5 seconds
                print(f"Distance: {avg_distance:.1f}/{target_distance_mm}mm | "
                      f"Remaining: {remaining_distance:.1f}mm | "
                      f"Speed: {current_speed:.1f} | "
                      f"Yaw: {yaw:.1f}° (error: {yaw_error:.1f}°)")
            
            # time.sleep(0.02)  # 50Hz control loop
        
        # Dừng robot
        self.bot.set_motor(0, 0, 0, 0)
        
        # Final status
        final_distances = self.encoder_controller.get_encoder_distances()
        final_avg_distance = sum(abs(d) for d in final_distances) / len(final_distances)
        print(f"Movement complete! Final distance: {final_avg_distance:.1f}mm "
              f"(target: {target_distance_mm}mm, error: {final_avg_distance - target_distance_mm:.1f}mm)")
    
    def rotate_to_angle(self, target_angle_deg, max_speed=20, tolerance_deg=1):
        """
        Xoay robot đến góc cụ thể (absolute yaw)
        """
        print(f"Rotating to angle: {target_angle_deg}° (tolerance: ±{tolerance_deg}°)")
        
        self.target_yaw = target_angle_deg
        start_time = time.time()
        reached_target = False
        
        while not reached_target:
            # Lấy góc hiện tại
            yaw = self.get_current_yaw()
            yaw_error = self.calculate_yaw_error(yaw)
            
            # Kiểm tra đã đến góc mục tiêu chưa
            if abs(yaw_error) <= tolerance_deg:
                reached_target = True
                break
            
            # Điều chỉnh tốc độ dựa trên góc còn lại
            if abs(yaw_error) < 10:  # Giảm tốc khi gần đích
                speed_factor = max(0.3, abs(yaw_error) / 10.0)
            else:
                speed_factor = 1.0
                
            rotation_speed = max_speed * speed_factor
            
            # Xác định hướng xoay: CCW positive
            if yaw_error > 0:
                vz = rotation_speed  # CCW
            else:
                vz = -rotation_speed  # CW
            
            # Áp dụng chỉ chuyển động xoay
            wheel_fl = -vz
            wheel_fr = vz
            wheel_rl = -vz
            wheel_rr = vz
            
            wheel_speeds = [
                self.limit_speed(wheel_fl),
                self.limit_speed(wheel_rl),
                self.limit_speed(wheel_fr),
                self.limit_speed(wheel_rr)
            ]
            
            self.bot.set_motor(*wheel_speeds)
            
            # Debug info
            elapsed = time.time() - start_time
            if int(elapsed * 10) % 5 == 0:
                print(f"Current: {yaw:.1f}° | Target: {target_angle_deg:.1f}° | "
                      f"Error: {yaw_error:.1f}° | Speed: {rotation_speed:.1f}")
            
            time.sleep(0.01)
        
        # Dừng robot
        self.stop_all_motors()

        time.sleep(1.0)
        
        # Final status
        final_yaw = self.get_current_yaw()
        final_error = self.calculate_yaw_error(final_yaw)
        print(f"Rotation complete! Final angle: {final_yaw:.1f}° "
              f"(target: {target_angle_deg:.1f}°, error: {final_error:.1f}°)")
    
    def rotate_relative(self, angle_deg, max_speed=20, tolerance_deg=0.0):
        """
        Xoay robot tương đối (relative) so với yaw hiện tại
        """
        current_yaw = self.get_current_yaw()
        target = current_yaw + angle_deg
        # Normalize to [-180,180]
        target = self.normalize_angle(target)
        self.rotate_to_angle(target, max_speed=max_speed, tolerance_deg=tolerance_deg)

    def rotate_in_place(self, direction: str, speed=20, duration=None):
        """
        Quay robot tại chỗ liên tục hoặc trong thời gian nhất định.
        direction: 'cw' hoặc 'ccw'
        speed: tốc độ xoay (0-100)
        duration: thời gian quay (giây). Nếu None, quay đến khi có lệnh dừng.
        """
        if direction.lower() not in ['cw', 'ccw']:
            print("Invalid direction! Use 'cw' or 'ccw'.")
            return
        vz = -speed if direction.lower() == 'cw' else speed
        wheel_fl = -vz
        wheel_fr = vz
        wheel_rl = -vz
        wheel_rr = vz
        print(f"Rotating in place {'clockwise' if vz<0 else 'counter-clockwise'} at speed {speed}%{'' if duration is None else f' for {duration}s'}")
        start = time.time()
        try:
            while True:
                self.bot.set_motor(
                    self.limit_speed(wheel_fl),
                    self.limit_speed(wheel_rl),
                    self.limit_speed(wheel_fr),
                    self.limit_speed(wheel_rr)
                )
                if duration is not None and (time.time() - start) >= duration:
                    break
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_motors()
            print("In-place rotation stopped.")
    
    def stop_all_motors(self):
        """Stop all motors"""
        self.bot.set_motor(0, 0, 0, 0)
        
    def reset_controllers(self):
        """Reset PID controller"""
        self.pid_yaw.reset()

# Convenience movement functions with distance control

def move_forward_distance(controller, distance_mm, speed=30):
    """Move forward by specified distance"""
    controller.move_distance(distance_mm, 0, distance_mm, speed)

def move_backward_distance(controller, distance_mm, speed=30):
    """Move backward by specified distance"""
    controller.move_distance(-distance_mm, 0, distance_mm, speed)

def move_right_distance(controller, distance_mm, speed=30):
    """Move right by specified distance"""
    controller.move_distance(0, distance_mm, distance_mm, speed)

def move_left_distance(controller, distance_mm, speed=30):
    """Move left by specified distance"""
    controller.move_distance(0, -distance_mm, distance_mm, speed)

def move_diagonal_distance(controller, vx_mm, vy_mm, speed=30):
    """Move diagonally by specified distances"""
    target_distance = math.sqrt(vx_mm**2 + vy_mm**2)
    controller.move_distance(vx_mm, vy_mm, target_distance, speed)

# Convenience rotation functions

def rotate_to_absolute(controller, angle_deg, speed=20):
    """Rotate to an absolute yaw angle"""
    controller.rotate_to_angle(angle_deg, max_speed=speed)

def rotate_by(controller, angle_deg, speed=20):
    """Rotate relative by angle_deg"""
    controller.rotate_relative(angle_deg, max_speed=speed)

def rotate_continuous(controller, direction='ccw', speed=20, duration=None):
    """Rotate in place continuously or for duration"""
    controller.rotate_in_place(direction, speed=speed, duration=duration)


def pick_object(bot: Rosmaster):
    bot.set_uart_servo_angle_array(P1, run_time=4000)
    time.sleep(2.0)

    bot.set_uart_servo_angle_array(P2, run_time=4000)
    time.sleep(2.0)

    bot.set_uart_servo_angle(6, 140, run_time=4000)
    time.sleep(2.0)

    bot.set_uart_servo_angle_array(P4, run_time=4000)
    time.sleep(2.0)


def release_object(bot: Rosmaster):
    bot.set_uart_servo_angle(1, 1, run_time=4000)
    time.sleep(2.0)
    bot.set_uart_servo_angle(6, 30, run_time=500)


def main():
    bot = Rosmaster()
    bot.create_receive_threading()
    bot.set_car_type(0x02)
    bot.clear_auto_report_data()
    bot.set_auto_report_state(True, forever=False)
    bot.reset_flash_value()

    bot.set_motor(0, 0, 0, 0)
    time.sleep(15)

    # Tạo controller với encoder
    controller = MecanumPositionController(bot, kp=5.0, ki=1.0, kd=0.0)

    print("Mecanum Position Control with Encoder Started")
    print("Robot will use encoder feedback for precise positioning")
    print("-" * 80)

    # Khởi tạo pose ban đầu
    roll, pitch, yaw = bot.get_imu_attitude_data(ToAngle=True)
    controller.set_initial_pose(yaw)
    print(f"Initial Yaw: {yaw:.2f}°")

    speed = 35.0

    try:
        print("\n=== Starting Position-Based Movement Demonstration ===")
        
        # # Di chuyển bằng khoảng cách cụ thể
        # #print("\n1. Moving forward 3000mm...")
        # move_forward_distance(controller, 4000, speed=speed)
        # time.sleep(5)
        
        # print("\n2. Picking object...")
        # pick_object(bot)
        
        # print("\n3. Rotating 90° clockwise in place...")
        # #rotate_to_absolute(controller, angle_deg=90, speed=speed)  # ví dụ quay 90° approx tùy tốc độ
        # #time.sleep(2)
        # rotate_to_absolute(controller, angle_deg=180, speed=speed) 
        # time.sleep(2)
        # # rotate_to_absolute(controller, angle_deg=0, speed=speed)
        # # time.sleep(2)
        
        # # print("\n4. Moving backward 6000mm...")
        # # # Điều chỉnh target yaw nếu cần
        # controller.target_yaw = yaw+180  # giữ hướng ban đầu nếu muốn
        move_backward_distance(controller, 4000, speed=-speed)
        time.sleep(5)
        
        print("\n5. Releasing object...")
        release_object(bot)
        
        print("\n=== Movement Demonstration Complete! ===")
            
    except KeyboardInterrupt:
        print("\nStopping motors...")
        controller.stop_all_motors()
        
    finally:
        print("Cleaning up...")
        time.sleep(0.1)
        controller.reset_controllers()
        controller.stop_all_motors()
        bot.set_auto_report_state(False, forever=False)
        bot.clear_auto_report_data()
        del bot

if __name__ == "__main__":
    import sys
    main()
