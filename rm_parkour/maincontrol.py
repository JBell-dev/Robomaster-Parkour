import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, Vector3, PointStamped, Pose
from robomaster_msgs.action import GripperControl
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from nav_msgs.msg import Odometry

class CubePickupAndGateNavigation(Node):
    def __init__(self):
        super().__init__('cube_pickup_and_gate_navigation')
        # Subscriptions
        self.camera_subscription = self.create_subscription(Image, '/RoboMaster/camera/image_color', self.camera_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/RoboMaster/odom', self.odom_callback, 10)

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/RoboMaster/cmd_vel', 10)
        self.arm_publisher = self.create_publisher(Vector3, '/RoboMaster/cmd_arm', 10)

        # Action Client for the Gripper
        self.gripper_action_client = ActionClient(self, GripperControl, '/RoboMaster/gripper')

        # ros to cv2 bridge
        self.bridge = CvBridge()

        # Initial state and parameters
        self.cube_position = None
        self.cube_area = 0
        self.cube_bounding_box = None
        self.image_width = None
        self.image_height = None
        self.state = 0  
        self.approach_threshold = 0.017  # Threshold for cube approach
        self.accommodate_threshold = 0.0172  # Threshold for gripper accommodation
        self.contour = None
        self.joint_angles = None
        self.arm_position = None
        self.gripper_target_state = None
        self.gripper_command_in_progress = False

        # Camera controller parameters
        self.cam_image = None
        self.mask_green = None
        self.mask_red = None
        self.green_area = 0
        self.red_area = 0
        self.green_height = None
        self.red_height = None
        self.obstacle_map = ObstacleMap()
        self.after_search_turn = (0, 0, 1)  # turning sign, clock counter, clock limit counter
        self.robot_pose = None

        # SYSTEM PARAMETERS
        self.EXISTENCE_THRESHOLD = 50
        self.AREA_PROXIMITY_THRESHOLD = 5000
        self.TEMP_PROXIMITY_THRESHOLD = 5000
        self.AREA_SEARCH_THRESHOLD = 500
        self.GREEN_LOWER_BOUND = np.array([48, 40, 100])
        self.GREEN_UPPER_BOUND = np.array([55, 255, 255])
        self.RED_LOWER_BOUND = np.array([174, 200, 100])
        self.RED_UPPER_BOUND = np.array([180, 255, 200])
        self.CENTERING_THRESHOLD = 10
        self.CORNER_THRESHOLD = 100
        self.TOP_BOTTOM_DIFF_THRESHOLD = 10
        self.AFTER_SEARCH_CLOCK_LIMIT = 12

        # PID controllers
        self.dt = 0.01
        self.x_controller = PIDController(kp=0.15, ki=0.00, kd=0.05)
        self.y_controller = PIDController(kp=0.001, ki=0.00, kd=0.0)
        self.theta_controller = PIDController(kp=0.2, ki=0.02, kd=0.02)

        # OCR2 PID controllers
        self.x_controller_ocr2 = PIDController(kp=0.00005, ki=0.0, kd=0.00001)
        self.theta_controller_ocr2 = PIDController(kp=0.001, ki=0.0, kd=0.00000)

        self.mask_green = np.zeros((480, 640), dtype=bool)  
        self.mask_red = np.zeros((480, 640), dtype=bool)

    def camera_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_height, self.image_width = self.cv_image.shape[:2]

        if self.state < 5:  # Gripper implementation : cube detection (states 0-4)
            self.detect_cube(self.cv_image)
        else:  # Camera controller implementation: color masks for gate navigation (states 5 and above)
            self.cam_image = self.cv_image
            self.mask_green, self.mask_red = self.get_masks()
            #self.get_logger().info(f" Green area - {self.green_area}, Red area - {self.red_area}")

        self.display_camera_feed() #visualize the camera feed


    def odom_callback(self, msg):
        "updates robot pose and plot it in the obstacle map"
        self.robot_pose = msg.pose.pose
        self.obstacle_map.plot_map(self.robot_pose)

    def detect_cube(self, cv_image):
        # Convert image to HSV and threshold to detect the cube
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            self.contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(self.contour)
            # We compute the centroid of the rectangle around the "cube" (started with a cube but deformed it a bit to fit in the gripper)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.cube_position = (cx, cy)
                self.cube_area = cv2.contourArea(self.contour)
                self.cube_bounding_box = cv2.boundingRect(self.contour)
        else:
            self.contour = None
            self.cube_position = None
            self.cube_area = 0
            self.cube_bounding_box = None

    def display_camera_feed(self):
        if self.state < 5:  # Gripper implementation: draws the cube contour 
            if self.contour is not None:
                cv_image_with_contour = cv2.drawContours(self.cv_image.copy(), [self.contour], -1, (0, 255, 0), 2)
                x, y, w, h = self.cube_bounding_box
                cv_image_with_contour = cv2.rectangle(cv_image_with_contour, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                cv_image_with_contour = self.cv_image.copy()
        else:  # Camera controller implementation: display masks 
            cv_image_with_contour = self.cam_image.copy()
            cv2.imshow('Green Mask', self.mask_green.astype(np.uint8) * 255)
            cv2.imshow('Red Mask', self.mask_red.astype(np.uint8) * 255)

        cv2.imshow("Camera Feed", cv_image_with_contour)
        cv2.waitKey(1)

    def move_robot(self):
        """MAIN STATE MACHINE: that is call periodically by the a timer callback """
        # State machine of the gripping part
        #self.get_logger().info(f"Current state: {self.state}")
        if self.state < 5:  # Gripper implementation
            if self.state == 0:
                self.search_cube()
            elif self.state == 1:
                self.approach_cube()
            elif self.state == 2:
                self.accommodate_gripper()
            elif self.state == 3:
                if not self.gripper_command_in_progress:
                    self.grasp_cube()
            elif self.state == 4:
                self.rotate_in_place()
        else:  # Camera controller implementation
            if self.state == 5:
                self.log_message('SEARCH')
                self.search()
            elif self.state == 6:
                self.log_message('MOVE_CLOSER')
                self.move_closer()
            elif self.state == 7:
                self.log_message('OCR_INIT')
                self.ocr_init()
            elif self.state == 8:
                self.log_message('OCR_SEARCH')
                self.ocr_search()
            elif self.state == 9:
                self.log_message('OCR_1')
                self.ocr_1()
            elif self.state == 10:
                self.log_message('OCR_2')
                self.ocr_2()
            elif self.state == 11:
                self.log_message('PASS_THROUGH')
                self.pass_through()
            elif self.state == 12:
                self.log_message("AFTER_SEARCH")
                self.after_search()

    def search_cube(self):
        self.get_logger().info("State: Searching for cube")
        twist = Twist()
        twist.angular.z = 0.2
        self.cmd_vel_publisher.publish(twist)

        if self.cube_position is not None:
            self.state = 1

    def approach_cube(self):
        # In a similar fashion as the camera controller, we try to accommodate the robot base on the error of centroid image vs object
        if self.cube_position is None:
            self.state = 0
            return

        cx, cy = self.cube_position
        image_center_x = self.image_width // 2
        image_center_y = self.image_height // 2

        error_x = cx - image_center_x #error between cube centroid and the image center

        twist = Twist()
        frame_area = self.image_width * self.image_height
        cube_area_ratio = self.cube_area / frame_area #compute the cube area over the frame 

        if cube_area_ratio < self.approach_threshold: #if under threshold we continue advancing proportional to error. -> we could extend it with a PID
            self.get_logger().info(f"State: Approaching cube, cube_area_ratio: {cube_area_ratio}")
            twist.linear.x = 0.1
            twist.angular.z = -0.005 * error_x
        else:
            self.get_logger().info("State: Cube approached")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.state = 2 #done approaching, lets acommodate 

        self.cmd_vel_publisher.publish(twist)

    def accommodate_gripper(self):
        # Arm accommodation
        if self.cube_position is None:
            self.state = 0
            return

        frame_area = self.image_width * self.image_height
        cube_area_ratio = self.cube_area / frame_area

        if cube_area_ratio > self.accommodate_threshold:
            self.get_logger().info(f"State: Accommodating gripper, cube_area_ratio: {cube_area_ratio}")
            arm_cmd = Vector3()
            arm_cmd.x = 0.02
            arm_cmd.y = 0.0
            arm_cmd.z = 0.02
            self.arm_publisher.publish(arm_cmd)
        else:
            self.get_logger().info("State: Gripper accommodated")
            self.state = 3

    def grasp_cube(self):
        self.get_logger().info('Stopping robot')
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)

        self.get_logger().info('Opening gripper')
        self.gripper_target_state = GripperControl.Goal.OPEN
        self.send_gripper_command(self.gripper_target_state)

    def feedback_callback(self, feedback_msg):
        # Feedback of the action of the gripper
        if self.state < 5:  
            feedback = feedback_msg.feedback
            self.get_logger().info(f'Feedback: Current state - {feedback.current_state}')
            if feedback.current_state == self.gripper_target_state: #check if the last ordered gripper state match its actual state
                self.gripper_command_in_progress = False
                if self.gripper_target_state == GripperControl.Goal.OPEN:
                    # Delay to allow for the actuation to happen -> if its open then it send to close 
                    self.get_logger().info('Waiting for gripper to open')
                    time.sleep(1.0)
                    self.close_gripper()
                elif self.gripper_target_state == GripperControl.Goal.CLOSE:
                    self.lift_cube()
        else:
            # State is 5 or higher, exit the function immediately
            return

    def send_gripper_command(self, command):
        #send a command to the grip action server -> creating a goal state and power 
        goal_msg = GripperControl.Goal()
        goal_msg.target_state = command
        goal_msg.power = 0.5 

        self.gripper_command_in_progress = True #flag for command in progress
        self.gripper_action_client.wait_for_server()
        self._send_goal_future = self.gripper_action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """when we receive a response from the gripper action server"""
        goal_handle = future.result()
        if not goal_handle.accepted: #check if the action was done or not
            self.get_logger().info('Gripper command rejected')
            self.state = 2
            self.gripper_command_in_progress = False
            return

        self.get_logger().info('Gripper command accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback) #if it was accepted, kicks another callback

    def get_result_callback(self, future):
        result = future.result().result #logs duration of the action
        self.get_logger().info(f'Gripper action result: Duration - {result.duration}')

    def close_gripper(self):
        #sends the target command of closing the gripper
        self.get_logger().info('Closing gripper')
        self.gripper_target_state = GripperControl.Goal.CLOSE
        self.send_gripper_command(self.gripper_target_state)

    def lift_cube(self):
        # after cube is grasp, is lifted a bit
        self.get_logger().info('Lifting cube')
        arm_cmd = Vector3()
        arm_cmd.x = 0.0
        arm_cmd.y = 0.0
        arm_cmd.z = 0.2
        self.arm_publisher.publish(arm_cmd)

        self.get_logger().info('Waiting for arm to lift')
        time.sleep(2.0)

        self.get_logger().info('Moving robot in reverse')
        twist = Twist()
        twist.linear.x = -0.2
        self.cmd_vel_publisher.publish(twist)

        self.get_logger().info('Waiting for robot to move in reverse')
        time.sleep(5.0)

        self.get_logger().info('Stopping robot')
        twist.linear.x = 0.0
        self.cmd_vel_publisher.publish(twist)

        self.lower_arm()

    def lower_arm(self):
        # Lowering the arm to not disturb camera vision
        self.get_logger().info('Lowering arm to the desired position')
        arm_cmd = Vector3()
        arm_cmd.x = 0.0
        arm_cmd.y = 0.0
        arm_cmd.z = -0.1
        self.arm_publisher.publish(arm_cmd)

        self.get_logger().info('Waiting for arm to lower')
        time.sleep(2.0)

        self.get_logger().info('Cube grasping complete')
        self.state = 5

    def rotate_in_place(self):
        # Transition to the camera controller implementation
        self.get_logger().info('Transitioning to gate navigation')
        self.state = 5
        self.x_controller.reset()
        self.theta_controller.reset()
        self.x_controller_ocr2.reset()
        self.theta_controller_ocr2.reset()

    def pub_vel(self, x=0, y=0, theta=0):
        vel_msg = Twist()
        vel_msg.linear.x = float(x)
        vel_msg.linear.y = float(y)
        vel_msg.angular.z = float(theta)
        self.cmd_vel_publisher.publish(vel_msg)

    def search(self):
        self.get_logger().info("Executing search in state 5")

        # logs for masks 
        self.get_logger().info(f"Green area: {self.green_area}, Red area: {self.red_area}")
        self.get_logger().info(f"Mask green shape: {self.mask_green.shape}, Mask red shape: {self.mask_red.shape}")

        # Check if any area is above the threshold
        if any([v > self.EXISTENCE_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.get_logger().info("Detection threshold met, stopping and transitioning to state 6")
            self.pub_vel(0, 0, 0)
            self.state = 6 #if detected, it goes to move closer phase 
        else:
            self.get_logger().info("Detection threshold not met, continuing search")
            self.pub_vel(0, 0, 0.2)


    def move_closer(self):
        self.log_message(f'Green area: {self.green_area}, Red area: {self.red_area}')
        if any([self.green_area > self.AREA_PROXIMITY_THRESHOLD, self.red_area > self.AREA_PROXIMITY_THRESHOLD]):
            #if exceed the area proximity threshold when moving close, it stops and transition to ocr intialization 
            self.pub_vel(0, 0, 0)
            self.state = 7
            self.x_controller.reset()
            self.theta_controller.reset()
        else: #if does not, it needs to still move closer
            #compare and get the max area among the green and red masks
            largest_area = max((self.green_area, self.mask_green), (self.red_area, self.mask_red), key=lambda x: x[0])
            left_x = np.min(np.where(largest_area[1])[1]) #leftmost
            right_x = np.max(np.where(largest_area[1])[1]) #and rightmost x coordinate of the largest area (max x coordinate of the 1's values)
            center_x = (left_x + right_x) // 2 #then we get the center
            true_center = largest_area[1].shape[1] // 2 #take the center of the image
            diff = true_center - center_x #if the difference is positive, it means the largest area is to the left of the true center.
            theta = self.theta_controller.update(diff)  # If robot at the right of the target center then rotate to the left in proportion to PID of the error
            area_error = self.AREA_PROXIMITY_THRESHOLD - max(self.green_area, self.red_area)  # We calculate the error in the proximity of the robot to the object (too far from the object, area_error would be positive, and if it's too close, area_error would be negative)
            x = self.x_controller.update(area_error)
            self.pub_vel(x, 0, theta)

    def ocr_init(self):
        if all([v > self.AREA_PROXIMITY_THRESHOLD for v in [self.green_area, self.red_area]]):
            #if both red and green exceeed the threshold then we can go to OCR2
            self.pub_vel(0, 0, 0)
            self.state = 10
        else:
            #if not, to search since we need to make the other visible too
            self.pub_vel(0, 0, 0)
            self.state = 8

    def ocr_search(self):
        if all([v > self.AREA_SEARCH_THRESHOLD for v in [self.green_area, self.red_area]]): #if both are visible, go to ocr2
            self.pub_vel(0, 0, 0)
            self.state = 10
        else:
            #if they are not found then we determine the orientation by checking which area is the larget -> nearer
            (orientation, area, mask) = max((-1, self.green_area, self.mask_green), (1, self.red_area, self.mask_red), key=lambda x: x[1])
            x = np.min(np.where(mask)[1]) if orientation == -1 else mask.shape[1] - np.max(np.where(mask)[1]) #check the leftmost coordinate
            if x < self.CORNER_THRESHOLD: #if is close to a corner -> for example, down - left part of the image frame -> very close to the edge of the image frame
                self.log_message('In corner and didn\'t find')
                self.pub_vel(0, 0, 0)
                self.TEMP_PROXIMITY_THRESHOLD = area  # it adds a temporary proximity threshold equal to the largest area of the perceived mask
                self.state = 9 #and goes to ocr1
            else:
                #If the x-coordinate is greater than or equal to the corner threshold, it means the robot is not in a corner and needs to continue searching.
                #It determines the color of the larger gate based on the orientation (-1 for green, 1 for red).
                color = 'green' if orientation == -1 else 'red'
                self.log_message(f'Putting {color} to corner, area: {area}')
                self.pub_vel(0, 0, orientation * 0.2)

    def ocr_1(self):
        if all([v > self.TEMP_PROXIMITY_THRESHOLD for v in [self.green_area, self.red_area]]):
            #checks if both the green and red areas exceed the temporary proximity threshold.
            self.pub_vel(0, 0, 0)
            self.state = 10 #if its, then goes to the final allignment oc2
            self.x_controller.reset()
            self.theta_controller.reset()
        else:
            (orientation, area, mask) = max((-1, self.green_area, self.mask_green), (1, self.red_area, self.mask_red), key=lambda x: x[1])
            self.log_message(f'Orientation: {orientation}, Area: {area}, Mask shape: {mask.shape}, green_area: {self.green_area}, red_area: {self.red_area}')
            
            #if not, it computes the error between the temp area and the gates areas
            area_error = self.TEMP_PROXIMITY_THRESHOLD - area
            x_threshold = np.min(np.where(mask)[1]) if orientation == -1 else mask.shape[1] - np.max(np.where(mask)[1])
            x = self.x_controller.update(area_error)
            x_threshold_error = self.CORNER_THRESHOLD - x_threshold #if the targget pos is 100 pixels and you are on 50 -> move to the right
            theta = self.theta_controller.update(x_threshold_error)
            y = -0.1 * orientation

            # We check for the last object if it exists and make a little movement in the opposite direction:
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
        top_g, top_r, bottom_g, bottom_r = self.get_top_bottom() #top and bottom coordinates of the masks
        top = top_g - top_r
        bottom = bottom_g - bottom_r

        #compute the middle point between the rightmost of the green mask and the leftmost of the red mask (inner points)
        if np.any(self.mask_green):
            green_most_right = np.max(np.where(self.mask_green)[1])
        else:
            green_most_right = 0

        if np.any(self.mask_red):
            red_most_left = np.min(np.where(self.mask_red)[1])
        else:
            red_most_left = self.mask_green.shape[1]  # Assuming the image width

        current_middle = (green_most_right + red_most_left) / 2
        ang_diff = self.mask_green.shape[1] / 2 - current_middle #check the difference to the center of the frame 

        cond1 = all([v > self.AREA_SEARCH_THRESHOLD for v in [self.green_area, self.red_area]]) #first check if areas are over the threshold
        cond2 = all([abs(v) < 50 for v in [top, bottom]]) #if the top and bottom  differences are also in the threshold
        cond3 = abs(ang_diff) < self.CENTERING_THRESHOLD #distance of centering less than threshold

        if all([cond1, cond2, cond3]): #if all are met, congrats, lets pass thourhg it
            self.pub_vel(0, 0, 0)
            self.state = 11
            self.green_height = bottom_g
            self.red_height = bottom_r
            self.x_controller.reset()
            self.theta_controller.reset()
        else:
            #if are not met, we compute the respective error and pass the feedback to the pid
            green_height = bottom_g - top_g
            red_height = bottom_r - top_r
            theta = self.theta_controller_ocr2.update(ang_diff)
            height_diff_error = red_height - green_height
            y = self.y_controller.update(height_diff_error)
            area_error = self.AREA_PROXIMITY_THRESHOLD - max(self.green_area, self.red_area)
            x = self.x_controller_ocr2.update(area_error)

            # We check for the last object if it exists and make a little movement in the opposite direction:
            if self.robot_pose is not None and self.obstacle_map.obstacles:
                self.log_message("Avoiding punching the previous holes")
                last_obstacle_x, last_obstacle_y = self.obstacle_map.obstacles[-1]
                robot_x = self.robot_pose.position.x
                robot_y = self.robot_pose.position.y
                distance = np.sqrt((robot_x - last_obstacle_x) ** 2 + (robot_y - last_obstacle_y) ** 2)  # Euclidean distance to the obstacle
                if distance < 0.1:  # Threshold to adjust, distance to the obstacle
                    obstacle_angle = np.arctan2(last_obstacle_y - robot_y, last_obstacle_x - robot_x)  # And we check orientation to it
                    robot_angle = self.obstacle_map.get_yaw(self.robot_pose.orientation)
                    angle_diff = obstacle_angle - robot_angle
                    if abs(angle_diff) < np.pi / 2:  # In front of robot
                        x = -0.1  # We go backwards
                    else:  # Behind
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
            self.pass_through_start_time = time.time() #start time
            self.pass_through_duration = 5  #duration

        elapsed_time = time.time() - self.pass_through_start_time #elapsed pass through phase

        if elapsed_time <= self.pass_through_duration:
            #double check centered to not hit
            green_most_right = np.max(np.where(self.mask_green)[1]) if np.any(self.mask_green) else 0
            red_most_left = np.min(np.where(self.mask_red)[1]) if np.any(self.mask_red) else self.mask_green.shape[1]
            current_middle = (green_most_right + red_most_left) / 2
            ang_diff = self.mask_green.shape[1] / 2 - current_middle

            if abs(ang_diff) > 5:  # To double-check we are centered
                self.theta_controller.reset()
                theta = self.theta_controller.update(ang_diff)
                self.pub_vel(0.2, 0, theta)
            else:
                self.pub_vel(0.2, 0, 0)
        else:
            self.pub_vel(0, 0, 0)
            self.state = 12 
            self.green_height = None
            self.red_height = None
            self.x_controller.reset()
            self.theta_controller.reset()

            # We map the obstacle
            if self.robot_pose is not None:
                x = self.robot_pose.position.x
                y = self.robot_pose.position.y
                self.obstacle_map.add_obstacle(x, y)
                map_image = self.obstacle_map.plot_map(self.robot_pose)
                #cv2.imshow("Obstacle Map", map_image)
                #cv2.waitKey(1)

            delattr(self, 'pass_through_start_time')

    def after_search(self):
        #check if the masks are existent now that it passed the last gate
        if any([v > self.EXISTENCE_THRESHOLD for v in [self.green_area, self.red_area]]):
            self.pub_vel(0, 0, 0)
            self.state = 6 #if it is, we start again
            self.after_search_turn = (0, 0, 1)
        else:
            #if not, we need to search it but turning with a lmited degree since we dont want to punch the gates we wheere going through
            a, b, c = self.after_search_turn
            if a == 0:
                self.after_search_turn = (1, 0, 1)
            else:
                if b < self.AFTER_SEARCH_CLOCK_LIMIT:
                    self.after_search_turn = (a, b + 1, c)
                else:
                    if c == 0:
                        self.after_search_turn = (a, 0, c + 1)
                    else:
                        self.after_search_turn = (-a, 0, 0)
            self.pub_vel(0, 0, self.after_search_turn[0] * 0.2)

    def log_message(self, message):
        self.get_logger().info(message)

    def get_masks(self):
        hsv_image = cv2.cvtColor(self.cam_image, cv2.COLOR_BGR2HSV)
        mask_green = np.all((hsv_image >= self.GREEN_LOWER_BOUND) & (hsv_image <= self.GREEN_UPPER_BOUND), axis=-1)
        mask_red = np.all((hsv_image >= self.RED_LOWER_BOUND) & (hsv_image <= self.RED_UPPER_BOUND), axis=-1)

        # mask based areas 
        self.green_area = np.sum(mask_green)
        self.red_area = np.sum(mask_red)

        
        cc_green = self.get_components(mask_green)
        cc_red = self.get_components(mask_red)

        if cc_green:
            maxx_cc_green = self.get_largest_component(cc_green) if self.state != 11 else self.get_outer_components(cc_green)
            mask_green = np.zeros_like(mask_green)
            for i, j in maxx_cc_green:
                mask_green[i, j] = True
        else:
            mask_green = np.zeros_like(mask_green)

        if cc_red:
            maxx_cc_red = self.get_largest_component(cc_red) if self.state != 11 else self.get_outer_components(cc_red)
            mask_red = np.zeros_like(mask_red)
            for i, j in maxx_cc_red:
                mask_red[i, j] = True
        else:
            mask_red = np.zeros_like(mask_red)

        # Update the class variables
        self.mask_green = mask_green
        self.mask_red = mask_red

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

    def run(self):
        #the main timer for the state machine
        self.timer = self.create_timer(0.1, self.move_robot)
        rclpy.spin(self)


class ObstacleMap:
    """mantain a list of the obstacle coordinates and plot them in a map with the additional real time robot position"""
    def __init__(self):
        self.obstacles = []
        self.map_size = 500
        self.map_resolution = 0.02
        self.map_center = (self.map_size // 2, self.map_size // 2)

    def add_obstacle(self, x, y):
        self.obstacles.append((x, y))

    def plot_map(self, robot_pose=None):
        self.map_image = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for obstacle in self.obstacles:
            x = int((obstacle[0] / self.map_resolution) + self.map_center[0])
            y = int((obstacle[1] / self.map_resolution) + self.map_center[1])
            cv2.circle(self.map_image, (x, y), 5, (0, 0, 255), -1)

        if robot_pose is not None:
            robot_x = int((robot_pose.position.x / self.map_resolution) + self.map_center[0])
            robot_y = int((robot_pose.position.y / self.map_resolution) + self.map_center[1])
            cv2.circle(self.map_image, (robot_x, robot_y), 8, (0, 255, 0), -1)
            cv2.line(self.map_image, (robot_x, robot_y),
                    (int(robot_x + 20 * np.cos(self.get_yaw(robot_pose.orientation))),
                    int(robot_y + 20 * np.sin(self.get_yaw(robot_pose.orientation)))),
                    (0, 255, 0), 2)

        cv2.imshow("Obstacle Map", self.map_image)
        cv2.waitKey(1)

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
        self.alpha = alpha  # Low pass filter coefficient
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

        # Low pass filter
        filtered_output = self.alpha * output + (1 - self.alpha) * self.previous_output

        self.previous_error = error
        self.previous_output = filtered_output
        self.last_time = current_time

        return filtered_output


def main(args=None):
    rclpy.init(args=args)
    cube_pickup_and_gate_navigation = CubePickupAndGateNavigation()
    cube_pickup_and_gate_navigation.run()
    cube_pickup_and_gate_navigation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
