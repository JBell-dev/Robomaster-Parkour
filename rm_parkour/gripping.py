import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, Vector3, PointStamped
from robomaster_msgs.action import GripperControl
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class CubePickup(Node):
    def __init__(self):
        super().__init__('cube_pickup')
        # Subscriptions
        self.camera_subscription = self.create_subscription(Image, '/RoboMaster/camera/image_color', self.camera_callback, 10)
        #self.joint_state_subscription = self.create_subscription(JointState, '/RoboMaster/joint_states', self.joint_state_callback, 10)
        #self.arm_position_subscription = self.create_subscription(PointStamped, '/RoboMaster/arm_position', self.arm_position_callback, 10)

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
        self.state = 0  # 0: search, 1: approach, 2: accommodate, 3: grasp, 4: rotate
        self.approach_threshold = 0.017  # Threshold for cube approach
        self.accommodate_threshold = 0.0172  # Threshold for gripper accommodation
        self.contour = None
        self.joint_angles = None
        self.arm_position = None
        self.gripper_target_state = None
        self.gripper_command_in_progress = False

    def camera_callback(self, msg):
        # ros2 to cv2 
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_height, self.image_width = self.cv_image.shape[:2]
        self.detect_cube(self.cv_image)
        self.display_camera_feed()

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
            #we compute the centroid of the rectangle around the "cube" (started with a cube but deformed it a bit to fit in the gripperx) 
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
        if self.contour is not None:
            cv_image_with_contour = cv2.drawContours(self.cv_image.copy(), [self.contour], -1, (0, 255, 0), 2)
            x, y, w, h = self.cube_bounding_box
            cv_image_with_contour = cv2.rectangle(cv_image_with_contour, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv_image_with_contour = self.cv_image.copy()

        cv2.imshow("Camera Feed", cv_image_with_contour)
        cv2.waitKey(1)

    def move_robot(self):
        # state machine of the robot gripper
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

    def search_cube(self):
        self.get_logger().info("State: Searching for cube")
        twist = Twist()
        twist.angular.z = 0.2
        self.cmd_vel_publisher.publish(twist)

        if self.cube_position is not None:
            self.state = 1

    def approach_cube(self):
        # in a similar fashion as the camera controller we tried to accomodate the robot base on error of centroid image vs object
        if self.cube_position is None:
            self.state = 0
            return

        cx, cy = self.cube_position
        image_center_x = self.image_width // 2
        image_center_y = self.image_height // 2

        error_x = cx - image_center_x

        twist = Twist()
        frame_area = self.image_width * self.image_height
        cube_area_ratio = self.cube_area / frame_area

        if cube_area_ratio < self.approach_threshold:
            self.get_logger().info(f"State: Approaching cube, cube_area_ratio: {cube_area_ratio}")
            twist.linear.x = 0.1
            twist.angular.z = -0.005 * error_x
        else:
            self.get_logger().info("State: Cube approached")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.state = 2

        self.cmd_vel_publisher.publish(twist)

    def accommodate_gripper(self):
        # arm accomodation 
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
        # feedback of the action of the gripper
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: Current state - {feedback.current_state}')
        if feedback.current_state == self.gripper_target_state:
            self.gripper_command_in_progress = False
            if self.gripper_target_state == GripperControl.Goal.OPEN:
                # delay to allow for the actuation to happen
                self.get_logger().info('Waiting for gripper to open')
                time.sleep(1.0)
                self.close_gripper()
            elif self.gripper_target_state == GripperControl.Goal.CLOSE:
                self.lift_cube()

    def send_gripper_command(self, command):
        goal_msg = GripperControl.Goal()
        goal_msg.target_state = command
        goal_msg.power = 0.5

        self.gripper_command_in_progress = True
        self.gripper_action_client.wait_for_server()
        self._send_goal_future = self.gripper_action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Gripper command rejected')
            self.state = 2
            self.gripper_command_in_progress = False
            return

        self.get_logger().info('Gripper command accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Gripper action result: Duration - {result.duration}')

    def close_gripper(self):
        self.get_logger().info('Closing gripper')
        self.gripper_target_state = GripperControl.Goal.CLOSE
        self.send_gripper_command(self.gripper_target_state)

    def lift_cube(self):
        # main robot movements
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
        # lowering the arm to not disturb camera vision
        self.get_logger().info('Lowering arm to the desired position')
        arm_cmd = Vector3()
        arm_cmd.x = 0.0
        arm_cmd.y = 0.0
        arm_cmd.z = -0.1
        self.arm_publisher.publish(arm_cmd)

        self.get_logger().info('Waiting for arm to lower')
        time.sleep(2.0)

        self.get_logger().info('Cube grasping complete')
        self.state = 4

    def rotate_in_place(self):
        # PLACEHOLDER STATE -> HERE WILL  PASS TO SEARCH OF GATE TO START THE MAIN WORK
        self.get_logger().info('Rotating in place')
        twist = Twist()
        twist.angular.z = 0.2
        self.cmd_vel_publisher.publish(twist)

    def run(self):
        self.timer = self.create_timer(0.1, self.move_robot)
        rclpy.spin(self)

def main(args=None):
    rclpy.init(args=args)
    cube_pickup = CubePickup()
    cube_pickup.run()
    cube_pickup.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
