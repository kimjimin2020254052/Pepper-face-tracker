import rclpy
from rclpy.node import Node
import cv2
import mediapipe as mp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed

dic_joints = {
# Head
"HS":{"name":"HeadYaw", "min":-2.0857, "max":2.0857},
"HUD":{"name":"HeadPitch", "min":-0.7068, "max":0.6371}
}

class MotionTrackNode(Node):
    def __init__(self):
        super().__init__('motion_track_node')
        # for movement following my finger
        self.w = 0
        self.h = 0
        self.cx = 0
        self.cy = 0
        self.k_p = 0.0005
        self.HS_degree = 0
        self.HUD_degree = -0.15
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_tracking_confidence=0.5,
            # confidence percentage
            min_detection_confidence=0.5
        )

        self.bridge = CvBridge()
        self.track_sub = self.create_subscription(
            Image,
            '/camera/front/image_raw', 
            self.image_callback,
            10
        )
        self.joint_pub = self.create_publisher(JointAnglesWithSpeed, '/joint_angles', 10)

    def image_callback(self, msg):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # point finger data
                    index_point_finger_tip = hand_landmarks.landmark[8]
                    self.h, self.w, c = cv_frame.shape
                    # convert to pixel data
                    self.cx, self.cy = int(index_point_finger_tip.x * self.w), int(index_point_finger_tip.y * self.h)
                    cv2.circle(cv_frame, (self.cx, self.cy), 10, (0, 255, 0), cv2.FILLED)
                    self.calculate_degree()
                    self.move_joint()

            cv2.imshow("Pepper_camera", cv_frame)
            cv2.waitKey(1) 
        
        # you can put specific error reason in e
        except Exception as e:
            # ros2 official log system as "error_level"
            self.get_logger().error(f'Processing Error: {e}')

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

    def move_joint(self):
        msg = JointAnglesWithSpeed()
        msg.joint_names = ["HeadYaw", "HeadPitch"]
        msg.joint_angles = [float(self.HS_degree), float(self.HUD_degree)]
        msg.speed = 0.05
        msg.relative = 0
        self.joint_pub.publish(msg)


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