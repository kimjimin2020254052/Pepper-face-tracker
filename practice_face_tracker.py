import rclpy
from rclpy.node import Node
import cv2
import time, math
import mediapipe as mp
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from sensor_msgs.msg import Range
from std_msgs.msg import String
from .pepper_config import dic_joints, joint_order

class MotionTrackNode(Node):
    def __init__(self):
        super().__init__('motion_track_node')
        # basic name master
        self.joint_master_names = [dic_joints[k]["name"] for k in joint_order]
        # basic angle master
        self.default_angles = [float(dic_joints[k]["default"]) for k in joint_order]
        # Check different moving
        self.is_moving = False
        # for movement following my finger
        self.w = 0
        self.h = 0
        self.cx = 0
        self.cy = 0
        self.k_p = 0.0005
        self.HS_degree = 0
        self.HUD_degree = -0.15
        self.ratio = 0.0
        # to check the movement
        self.has_front_moved = False
        self.has_back_moved = False
        # face mesh => really good detection. 468 points in the face.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            # more accurate in surrounding eye
            refine_landmarks=True, 
            # confidence percentage
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        # hand for tracking my wrist
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_tracking_confidence=0.5,
            # confidence percentage
            min_detection_confidence=0.5
        )
        # subscription for image
        self.bridge = CvBridge()
        self.track_sub = self.create_subscription(
            Image,
            '/camera/front/image_raw', 
            self.image_callback,
            10
        )
        # subscription for sonar 
        self.back_dist = 2.5
        self.sonar_sub = self.create_subscription(
            Range,
            '/sonar/back',
            self.sonar_callback,
            10
        )
        self.joint_pub = self.create_publisher(JointAnglesWithSpeed, '/joint_angles', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/speech', 10)
    
    def image_callback(self, msg):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)

            results = self.face_mesh.process(rgb_frame)
            results_hands = self.hands.process(rgb_frame)

            # detect the face 
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # nose tip => 1 / cheek left, right => 234, 454
                    nose_tip = face_landmarks.landmark[1]
                    # for checking distance of the face
                    left_face = face_landmarks.landmark[234].x
                    right_check = face_landmarks.landmark[454].x
                    face_width = abs(left_face - right_check)
                    
                    self.h, self.w, c = cv_frame.shape
                    # convert to pixel data
                    self.cx, self.cy = int(nose_tip.x * self.w), int(nose_tip.y * self.h)
                    self.face_width = int(face_width * self.w)
                    self.ratio = float(self.face_width / self.w)
                    
                    # ratio limit
                    if self.ratio < 0.05 and self.ratio > 0.4:
                        self.ratio = None

                    #cv2.circle(cv_frame, (self.cx, self.cy), 10, (0, 255, 0), cv2.FILLED)
                    self.calculate_degree()
                    self.move_joint()
            else:
                self.ratio = None

            # detect the hand (expecially wrist)
            wrist_x_max = 0
            wrist_x_min = 1
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # wirst number
                    self.wrist_x = hand_landmarks.landmark[0].x
                    if self.wrist_x > wrist_x_max:
                        wrist_x_max = self.wrist_x
                    if self.wrist_x < wrist_x_min:
                        wrist_x_min = self.wrist_x
                    check_Hi = self.w*(wrist_x_max-wrist_x_min)
                    if check_Hi > 50 and self.is_moving == False:
                        self.get_logger().info(f'Hi_pixel : {check_Hi:.3f}')
                        self.GiveMeChocolate()
                        wrist_x_max = 0
                        wrist_x_min = 1

            # show in the laptop 
            cv2.imshow("Pepper_camera", cv_frame)
            cv2.waitKey(1) 
        
        # you can put specific error reason in e
        except Exception as e:
            # ros2 official log system as "error_level"
            self.get_logger().error(f'Processing Error: {e}')

    # pepper's saying function
    def say(self, text):
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    # for prevent back crushing
    def sonar_callback(self, msg):
        try:
            self.back_dist = msg.range

        except Exception as e:
            # ros2 official log system as "error_level"
            self.get_logger().error(f'Processing Error: {e}')

    # for calculation of degree
    def calculate_degree(self):
        center_x = self.w//2 
        center_y = self.h//2
        offset_x = center_x - self.cx 
        offset_y = center_y - self.cy 
        if abs(offset_x)<20: offset_x = 0
        if abs(offset_y)<20: offset_y = 0
        # convert to the radian unit
        self.HS_degree += offset_x * self.k_p
        self.HUD_degree -= offset_y * self.k_p
        self.HS_degree = max(min(self.HS_degree, 2.0), -2.0)
        self.HUD_degree = max(min(self.HUD_degree, 0.6), -0.7)

    # for movement of joint
    def move_joint(self):
        msg2 = Twist()
        if self.is_moving:
            msg2.linear.x = 0.0
            self.vel_pub.publish(msg2)
            return
        # head movement
        msg = JointAnglesWithSpeed()
        msg.joint_names = ["HeadYaw", "HeadPitch"]
        msg.joint_angles = [float(self.HS_degree), float(self.HUD_degree)]
        msg.speed = 0.1
        msg.relative = 0

        # base movement
        msg2.angular.z = 0.0
        if self.back_dist > 0.5 and self.is_moving == False:
            if 0.2 < self.ratio:
                if self.ratio < 0.22:
                    msg2.linear.x = float(0.0)
                    self.vel_pub.publish(msg2)
                else :
                    msg2.linear.x = float(-0.2)
                    self.has_front_moved = True
                    self.has_back_moved = False
                    self.joint_pub.publish(msg)
                    self.vel_pub.publish(msg2)
            elif 0.1 > self.ratio:
                if self. ratio > 0.08:
                    msg2.linear.x = float(0.0)
                    self.vel_pub.publish(msg2)
                else : 
                    msg2.linear.x = float(0.2)
                    self.has_front_moved = False
                    self.has_back_moved = True    
                    self.joint_pub.publish(msg)
                    self.vel_pub.publish(msg2)
            else :
                if self.has_front_moved:
                    self.get_logger().info(f'Shygirl ratio : {self.ratio:.3f}')
                    self.vel_pub.publish(msg2)
                    self.has_front_moved = False
                    time.sleep(1.0)
                    self.move_shyGirl()
                elif self.has_back_moved:
                    self.get_logger().info(f'Lovelygirl ratio : {self.ratio:.3f}')
                    self.vel_pub.publish(msg2)
                    self.has_back_moved = False
                    time.sleep(1.0)
                    self.move_lovelyGirl()
                else : 
                    self.get_logger().info(f'ratio : {self.ratio:.3f}')
                    self.joint_pub.publish(msg)
                    
        else:
            msg2.linear.x = float(0.0)
            self.vel_pub.publish(msg2)
            return

    # movement of the shy girl
    def move_shyGirl(self):
        if self.is_moving:
            return
        self.is_moving = True
        self.say("Oh, you are too close!")
        msg = JointAnglesWithSpeed()
        msg.speed = 0.1
        msg.relative = 0

        msg.joint_names = self.joint_master_names
        for i in [x*0.1 for x in range(3, 0, -1)]:
            current_angles = list(self.default_angles)
            current_angles[1] = float(0.6-i)
            # left
            current_angles[2] = float(i)
            current_angles[3] = float(0.24)
            current_angles[4] = float(-0.97)
            current_angles[5] = float(-1.51)
            current_angles[6] = float(-0.88)
            current_angles[7] = float(1)
            # right
            current_angles[8] = float(i)
            current_angles[9] = float(-0.24)
            current_angles[10] = float(0.97)
            current_angles[11] = float(1.51)
            current_angles[12] = float(0.88)
            current_angles[13] = float(1)

            msg.joint_angles = current_angles
            self.joint_pub.publish(msg)
            time.sleep(0.02)

        time.sleep(4.5)
        self.is_moving = False
        self.basic_setting()
    
    # movement of the lovely girl
    def move_lovelyGirl(self):
        if self.is_moving:
            return
        self.is_moving = True
        self.say("I missed you!! come here!")
        msg = JointAnglesWithSpeed()
        msg.joint_names = self.joint_master_names
        msg.speed = 0.1
        msg.relative = 0
        for i in [x*0.1 for x  in range(1,11)]:   
            current_angles = list(self.default_angles) 
            # left
            current_angles[2] = float(1.0)
            current_angles[3] = float(1.0)
            current_angles[4] = float(-1.0)
            current_angles[6] = float(-1.0)
            current_angles[7] = float(i)
            # right
            current_angles[8] = float(1.0)
            current_angles[9] = float(-1.0)
            current_angles[10] = float(1.0)
            current_angles[12] = float(1.0)
            current_angles[13] = float(i)
            # base
            current_angles[14] 

            msg.joint_angles = current_angles
            self.joint_pub.publish(msg)
            time.sleep(0.02)
        time.sleep(4.5)
        self.is_moving = False
        self.basic_setting()

    # for setting movement initilization
    def basic_setting(self):
        if self.is_moving:
            return
        self.is_moving = True
        msg = JointAnglesWithSpeed()
        msg.speed = 0.1
        msg.relative = 0
        msg.joint_names = self.joint_master_names
        msg.joint_angles = self.default_angles
        self.joint_pub.publish(msg)
        time.sleep(2.0)
        self.is_moving = False

    def interpolation(self, start_angle, target_angel):
        self.t = float((target_angel-start_angle)/self.steps)

    def GiveMeChocolate(self):
        if self.is_moving:
            return
        self.is_moving = True
        self.say("Hello. Today is Valentine's day. Give me a chocolate!!")
        msg = JointAnglesWithSpeed()
        msg.speed = 0.1
        msg.relative = 0 
        msg.joint_names = self.joint_master_names
        self.steps = 10
        sign = 1

        for num in range (4):
            for i in range(self.steps+1):
                current_angles = list(self.default_angles)
                # degree to radian
                # left
                current_angles[2] = float(math.radians(34.9))
                current_angles[3] = float(math.radians(5.4))
                current_angles[4] = float(math.radians(-58.9))
                current_angles[5] = float(math.radians(-43.4))
                current_angles[6] = float(math.radians(-87.8))
                self.interpolation(0, 1)
                current_angles[7] = float(self.t*i)
                # right
                current_angles[8] = float(math.radians(34.9))
                current_angles[9] = float(math.radians(-5.4))
                current_angles[10] = float(math.radians(58.9))
                current_angles[11] = float(math.radians(43.4))
                current_angles[12] = float(math.radians(87.8))
                current_angles[13] = float(self.t*i)      
                # base
                self.interpolation(0, 0.14*sign)
                current_angles[14] = float(self.t*i)    
                
                msg.joint_angles = current_angles
                self.joint_pub.publish(msg)
                time.sleep(0.02)  
            sign *= -1
            time.sleep(1.0)
        time.sleep(2.0)
        self.is_moving = False
        self.basic_setting()    

def main(args=None):
    rclpy.init(args=args)
    node = MotionTrackNode()
    
    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
