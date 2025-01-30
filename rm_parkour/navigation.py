import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from nav_msgs.msg import Odometry

class ObstacleMap:
    def __init__(self):
        self.obstacles = []
        self.map_size = 500
        self.map_resolution = 0.02
        self.map_center = (self.map_size // 2, self.map_size // 2)

    def add_obstacle(self, x, y):
        self.obstacles.append((x, y))

    def plot_map(self, robot_pose=None):
        map_image = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for obstacle in self.obstacles:
            x = int((obstacle[0] / self.map_resolution) + self.map_center[0])
            y = int((obstacle[1] / self.map_resolution) + self.map_center[1])
            cv2.circle(map_image, (x, y), 5, (0, 0, 255), -1)

        if robot_pose is not None:
            robot_x = int((robot_pose.position.x / self.map_resolution) + self.map_center[0])
            robot_y = int((robot_pose.position.y / self.map_resolution) + self.map_center[1])
            cv2.circle(map_image, (robot_x, robot_y), 8, (0, 255, 0), -1)
            cv2.line(map_image, (robot_x, robot_y),
                     (int(robot_x + 20 * np.cos(self.get_yaw(robot_pose.orientation))),
                      int(robot_y + 20 * np.sin(self.get_yaw(robot_pose.orientation)))),
                     (0, 255, 0), 2)

        return map_image

    def get_yaw(self, orientation):
        _, _, yaw = self.euler_from_quaternion(orientation)
        return yaw

    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-0.7, 0.7), sample_time=0.01, alpha=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.sample_time = sample_time
        self.alpha = alpha  #low pass filter coefficient
        self.integral = 0
        self.previous_error = 0
        self.previous_output = 0
        self.last_time = None
        self.obstacle_map = ObstacleMap()

    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_output = 0
        self.last_time = None

    def update(self, error):
        current_time = time.time()

        if self.last_time is None:
            self.last_time = current_time
            return 0

        dt = current_time - self.last_time

        proportional = self.kp * error
        self.integral += self.ki * error * dt
        derivative = self.kd * (error - self.previous_error) / dt

        output = proportional + self.integral + derivative
        output = max(min(output, self.output_limits[1]), self.output_limits[0])

        #low pass filter
        filtered_output = self.alpha * output + (1 - self.alpha) * self.previous_output

        self.previous_error = error
        self.previous_output = filtered_output
        self.last_time = current_time

        return filtered_output
    
class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(Image, '/RoboMaster/camera/image_color', self.image_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, '/RoboMaster/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/RoboMaster/odom', self.odom_callback, 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.2, self.action_callback)
        self.state = 0
        self.cam_image = None
        self.mask_green = None
        self.mask_red = None
        self.green_area = 0
        self.red_area = 0
        self.green_height = None
        self.red_height = None
        self.obstacle_map = ObstacleMap()
        self.after_search_turn = (0,0,1) # turning sign, clock counter, clock limit counter
        self.robot_pose = None

        # SYSTEM PARAMATERS
        self.EXISTENCE_THRESHOLD = 50
        self.AREA_PROXIMITY_THRESHOLD = 5000
        self.TEMP_PROXIMITY_THRESHOLD = 5000
        self.AREA_SEARCH_THRESHOLD = 500 
        self.GREEN_LOWER_BOUND = np.array([48, 40, 100])
        self.GREEN_UPPER_BOUND = np.array([55, 255, 255])
        self.RED_LOWER_BOUND = np.array([174, 200, 100])
        self.RED_UPPER_BOUND = np.array([180, 255, 200])
        self.CENTERING_THRESHOLD = 50
        self.CORNER_THRESHOLD = 100
        self.TOP_BOTTOM_DIFF_THRESHOLD = 20
        self.AFTER_SEARCH_CLOCK_LIMIT = 12

        
        self.STATES = {
            0: 'SEARCH',
            1: 'MOVE_CLOSER',
            2: 'OCR_INIT', 
            3: 'OCR_SEARCH', 
            4: 'OCR_1',
            5: 'OCR_2',
            6: 'PASS_THROUGH',
            7: 'AFTER_SEARCH'
        }

        #pid part:
        self.dt = 0.01
        self.x_controller = PIDController(kp=0.15, ki=0.00, kd=0.05)
        self.y_controller = PIDController(kp=0.001, ki=0.00, kd=0.0)
        self.theta_controller = PIDController(kp=0.2, ki=0.02, kd=0.02)

        #ocr2 pid part, to control better the aligning movement: 
        self.x_controller_ocr2 = PIDController(kp=0.00005, ki=0.0, kd=0.0001)
        self.theta_controller_ocr2 = PIDController(kp=0.005, ki=0.0, kd=0.0001)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    
    def pub_vel(self, x=0,y=0,theta=0):
        vel_msg = Twist()
        vel_msg.linear.x = float(x)
        vel_msg.linear.y = float(y)
        vel_msg.angular.z = float(theta)
        self.velocity_publisher.publish(vel_msg)

    def search(self):
        if any([v > self.EXISTENCE_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.pub_vel(0,0,0)
            self.state = 1
        else:
            self.pub_vel(0,0,0.2)

    def move_closer(self):
        self.log_message(f'Green area: {self.green_area}, Red area: {self.red_area}')
        if any([self.green_area > self.AREA_PROXIMITY_THRESHOLD, self.red_area > self.AREA_PROXIMITY_THRESHOLD]):
            self.pub_vel(0,0,0)
            self.state = 2
            self.x_controller.reset()
            self.theta_controller.reset()
        else:
            largest_area = max((self.green_area, self.mask_green), (self.red_area, self.mask_red), key=lambda x: x[0])
            left_x = np.min(np.where(largest_area[1])[1])
            right_x = np.max(np.where(largest_area[1])[1])
            center_x = (left_x + right_x) // 2
            true_center = largest_area[1].shape[1] // 2
            diff = true_center - center_x
            theta = self.theta_controller.update(diff) #if robot at the right of the target center then rotate to the left in proportion to pid of the error 
            area_error = self.AREA_PROXIMITY_THRESHOLD - max(self.green_area, self.red_area) # we calculate the error in the proximity of the robot to the object (too far from the object, area_error would be positive, and if it's too close, area_error would be negative)
            x = self.x_controller.update(area_error)
            self.pub_vel(x, 0, theta)

    
    def ocr_init(self):
        if all([v  > self.AREA_PROXIMITY_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.pub_vel(0,0,0)
            self.state = 5
        else:
            self.pub_vel(0,0,0)
            self.state = 3

    def ocr_search(self):
        if all([v  > self.AREA_SEARCH_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.pub_vel(0,0,0)
            self.state = 5
        else:
            (orientation, area, mask) = max((-1, self.green_area, self.mask_green), (1, self.red_area, self.mask_red), key=lambda x: x[1])
            x = np.min(np.where(mask)[1]) if orientation == -1 else mask.shape[1] - np.max(np.where(mask)[1])
            if x < self.CORNER_THRESHOLD:
                self.log_message('In corner and didnt find')
                self.pub_vel(0,0,0)
                self.TEMP_PROXIMITY_THRESHOLD = area # change
                self.state = 4
            else:
                color = 'green' if orientation == -1 else 'red'
                self.log_message(f'Putting {color} to corner, area: {area}')
                self.pub_vel(0,0,orientation*0.2)
    
    def ocr_1(self):
        if all([v > self.TEMP_PROXIMITY_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.pub_vel(0, 0, 0)
            self.state = 5
            self.x_controller.reset()
            self.theta_controller.reset()
        else:
            (orientation, area, mask) = max((-1, self.green_area, self.mask_green), (1, self.red_area, self.mask_red), key=lambda x: x[1])
            self.log_message(f'Orientation: {orientation}, Area: {area}, Mask shape: {mask.shape}, green_area: {self.green_area}, red_area: {self.red_area}')

            area_error = self.TEMP_PROXIMITY_THRESHOLD - area
            x_threshold = np.min(np.where(mask)[1]) if orientation == -1 else mask.shape[1] - np.max(np.where(mask)[1])
            x = self.x_controller.update(area_error)
            x_threshold_error = self.CORNER_THRESHOLD - x_threshold
            theta = self.theta_controller.update(x_threshold_error)
            y = -0.1 * orientation

            # we check for the last object if it exist and make a little movement in the opposite direction: 
            if self.robot_pose is not None and self.obstacle_map.obstacles:
                last_obstacle_x, last_obstacle_y = self.obstacle_map.obstacles[-1]
                robot_x = self.robot_pose.position.x
                robot_y = self.robot_pose.position.y
                if abs(robot_x - last_obstacle_x) < 0.1 and abs(robot_y - last_obstacle_y) < 0.1:
                    if orientation == -1:
                        y = 0.1  
                    else:
                        y = -0.1

            focus = 'green' if orientation == -1 else 'red'
            self.log_message(f'x: {x}, y: {y}, theta: {theta}, x_threshold: {x_threshold}, focus: {focus}')
            self.pub_vel(x, y, theta)

    def ocr_2(self):
        self.log_message(f"OBSTACLES ON: {self.obstacle_map.obstacles} AND ODOM: {self.robot_pose}")
        top_g, top_r, bottom_g, bottom_r = self.get_top_bottom()
        top = top_g - top_r
        bottom = bottom_g - bottom_r

        if np.any(self.mask_green):
            green_most_right = np.max(np.where(self.mask_green)[1])
        else:
            green_most_right = 0

        if np.any(self.mask_red):
            red_most_left = np.min(np.where(self.mask_red)[1])
        else:
            red_most_left = self.mask_green.shape[1]  # Assuming the image width

        current_middle = (green_most_right + red_most_left) / 2
        ang_diff = self.mask_green.shape[1] / 2 - current_middle

        cond1 = all([v > self.AREA_SEARCH_THRESHOLD for v in [self.green_area, self.red_area]])
        cond2 = all([abs(v) < 50 for v in [top, bottom]])
        cond3 = abs(ang_diff) < self.CENTERING_THRESHOLD

        if all([cond1, cond2, cond3]):
            self.pub_vel(0, 0, 0)
            self.state = 6
            self.green_height = bottom_g
            self.red_height = bottom_r
            self.x_controller.reset()
            self.theta_controller.reset()
        else:
            green_height = bottom_g - top_g
            red_height = bottom_r - top_r
            theta = self.theta_controller_ocr2.update(ang_diff)
            height_diff_error = red_height - green_height
            y = self.y_controller.update(height_diff_error)
            area_error = self.AREA_PROXIMITY_THRESHOLD - max(self.green_area, self.red_area)
            x = self.x_controller_ocr2.update(area_error)

            # we check for the last object if it exist and make a little movement in the opposite direction: 
            if self.robot_pose is not None and self.obstacle_map.obstacles:
                self.log_message("avoiding punching the previous holes")
                last_obstacle_x, last_obstacle_y = self.obstacle_map.obstacles[-1]
                robot_x = self.robot_pose.position.x
                robot_y = self.robot_pose.position.y
                distance = np.sqrt((robot_x - last_obstacle_x)**2 + (robot_y - last_obstacle_y)**2) #euclidean distance to the obstacle
                if distance < 0.1:  #threshold to adjust, distance to the obstacle
                    obstacle_angle = np.arctan2(last_obstacle_y - robot_y, last_obstacle_x - robot_x)  #and we check orientation to it
                    robot_angle = self.obstacle_map.get_yaw(self.robot_pose.orientation)
                    angle_diff = obstacle_angle - robot_angle
                    if abs(angle_diff) < np.pi / 2:  #in front of robot
                        x = -0.1  # we go backwards
                    else:  # behind
                        x = 0.1  

            self.log_message(f'x: {x}, y: {y}, theta: {theta}, green height {green_height}, red height {red_height}, ang diff {ang_diff}')
            self.pub_vel(x, y, theta)

    def get_top_bottom(self):
        if np.any(self.mask_green):
            top_green = np.min(np.where(self.mask_green)[0])
            bottom_green = np.max(np.where(self.mask_green)[0])
        else:
            top_green = 0
            bottom_green = 0
        
        if np.any(self.mask_red):
            top_red = np.min(np.where(self.mask_red)[0])
            bottom_red = np.max(np.where(self.mask_red)[0])
        else:
            top_red = 0
            bottom_red = 0
        
        return top_green, top_red, bottom_green, bottom_red

        
    def pass_through(self):
        if not hasattr(self, 'pass_through_start_time'):
            self.pass_through_start_time = time.time()
            self.pass_through_duration = 4.5

        elapsed_time = time.time() - self.pass_through_start_time

        if elapsed_time <= self.pass_through_duration:
            green_most_right = np.max(np.where(self.mask_green)[1]) if np.any(self.mask_green) else 0
            red_most_left = np.min(np.where(self.mask_red)[1]) if np.any(self.mask_red) else self.mask_green.shape[1]
            current_middle = (green_most_right + red_most_left) / 2
            ang_diff = self.mask_green.shape[1] / 2 - current_middle
            
            if abs(ang_diff) > 5:  # to double check we are centered
                self.theta_controller.reset()
                theta = self.theta_controller.update(ang_diff)
                self.pub_vel(0.2, 0, theta)
            else:
                self.pub_vel(0.2, 0, 0)  
        else:
            self.pub_vel(0, 0, 0)
            self.state = 7
            self.green_height = None
            self.red_height = None
            self.x_controller.reset()
            self.theta_controller.reset()

            # we map the obstacle
            if self.robot_pose is not None:
                x = self.robot_pose.position.x
                y = self.robot_pose.position.y
                self.obstacle_map.add_obstacle(x, y)
                map_image = self.obstacle_map.plot_map(self.robot_pose)
                cv2.imshow("Obstacle Map", map_image)
                cv2.waitKey(1)

            
            delattr(self, 'pass_through_start_time')

    def after_search(self):
        if any([v > self.EXISTENCE_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.pub_vel(0,0,0)
            self.state = 1
            self.after_search_turn = (0,0,1)
        else:
            a,b,c = self.after_search_turn
            if a == 0:
                self.after_search_turn = (1,0,1)
            else:
                if b < self.AFTER_SEARCH_CLOCK_LIMIT:
                    self.after_search_turn = (a, b+1, c)

                else:
                    if c == 0:
                        self.after_search_turn = (a,0,c+1)
                    else:
                        self.after_search_turn = (-a,0,0)
            self.pub_vel(0,0,self.after_search_turn[0]*0.2)

    def log_message(self, message):
        self.get_logger().info(message)

    def action_callback(self):
        if self.cam_image is None:
            self.log_message('No image received')
            return

        self.green_area = np.sum(self.mask_green)
        self.red_area = np.sum(self.mask_red)

        # plot robots real time position and orientation
        if self.robot_pose is not None:
            robot_x = int((self.robot_pose.position.x / self.obstacle_map.map_resolution) + self.obstacle_map.map_center[0])
            robot_y = int((self.robot_pose.position.y / self.obstacle_map.map_resolution) + self.obstacle_map.map_center[1])
            map_image = self.obstacle_map.plot_map()  
            cv2.circle(map_image, (robot_x, robot_y), 8, (0, 255, 0), -1)
            cv2.line(map_image, (robot_x, robot_y),
                    (int(robot_x + 20 * np.cos(self.obstacle_map.get_yaw(self.robot_pose.orientation))),
                    int(robot_y + 20 * np.sin(self.obstacle_map.get_yaw(self.robot_pose.orientation)))),
                    (0, 255, 0), 2)
            cv2.imshow("Obstacle Map", map_image)
            cv2.waitKey(1)

        if self.state == 0:
            self.log_message('SEARCH')
            self.search()
        elif self.state == 1:
            self.log_message('MOVE_CLOSER')
            self.move_closer()
        elif self.state == 2:
            self.log_message('OCR_INIT')
            self.ocr_init()
        elif self.state == 3:
            self.log_message('OCR_SEARCH')
            self.ocr_search()
        elif self.state == 4:
            self.log_message('OCR_1')
            self.ocr_1()
        elif self.state == 5:
            self.log_message('OCR_2')
            self.ocr_2()
        elif self.state == 6:
            self.log_message('PASS_THROUGH')
            self.pass_through()
        elif self.state == 7:
            self.log_message("AFTER_SEARCH")
            self.after_search()
         

    def image_callback(self, msg):
        ts = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.cam_image = cv_image
        self.mask_green, self.mask_red = self.get_masks()
        cv2.imshow('Camera Frame', cv_image)
        cv2.imshow('Green Mask', self.mask_green.astype(np.uint8)*255)
        cv2.imshow('Red Mask', self.mask_red.astype(np.uint8)*255)
        cv2.waitKey(1)
        te = time.time()
        self.dt = te - ts

    
    def is_proximity(self, bool_mask):
        return np.sum(bool_mask) > self.AREA_PROXIMITY_THRESHOLD
    

    def get_masks(self): 
        hsv_image = cv2.cvtColor(self.cam_image, cv2.COLOR_BGR2HSV)
        mask_green = np.all((hsv_image >= self.GREEN_LOWER_BOUND) & (hsv_image <= self.GREEN_UPPER_BOUND), axis=-1)
        mask_red = np.all((hsv_image >= self.RED_LOWER_BOUND) & (hsv_image <= self.RED_UPPER_BOUND), axis=-1)
        
        cc_green = self.get_components(mask_green)
        cc_red = self.get_components(mask_red)
        
        if cc_green:
            maxx_cc_green = self.get_largest_component(cc_green) if self.state != 6 else self.get_outer_components(cc_green)
            mask_green = np.zeros_like(mask_green)
            for i, j in maxx_cc_green:
                mask_green[i, j] = True
        else:
            mask_green = np.zeros_like(mask_green)
        
        if cc_red:
            maxx_cc_red = self.get_largest_component(cc_red) if self.state != 6 else self.get_outer_components(cc_red)
            mask_red = np.zeros_like(mask_red)
            for i, j in maxx_cc_red:
                mask_red[i, j] = True
        else:
            mask_red = np.zeros_like(mask_red)

        return mask_green, mask_red

    def get_components(self, mask):
        rows, cols = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components = []

        def dfs(row, col):
            stack = [(row, col)]
            component = []

            while stack:
                curr_row, curr_col = stack.pop()
                if curr_row < 0 or curr_row >= rows or curr_col < 0 or curr_col >= cols or visited[curr_row, curr_col]:
                    continue

                visited[curr_row, curr_col] = True
                component.append((curr_row, curr_col))

                neighbors = [(curr_row + 1, curr_col), (curr_row - 1, curr_col),
                            (curr_row, curr_col + 1), (curr_row, curr_col - 1)]
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and mask[neighbor]:
                        stack.append(neighbor)

            return component

        for pixel in np.argwhere(mask):
            if not visited[tuple(pixel)]:
                component = dfs(pixel[0], pixel[1])
                components.append(component)

        return components

    def get_largest_component(self, components):
        if not components:
            return []
        largest_component = max(components, key=len)
        return largest_component
    
    def get_outer_components(self, components):
        if not components:
            return []
        outest_component = min(components, key=lambda x: min(x, key=lambda x: x[0]))
        return outest_component



def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

