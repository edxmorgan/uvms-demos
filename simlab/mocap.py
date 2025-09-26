# path_builder.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path
from mocap4r2_msgs.msg import RigidBodies
import numpy as np
import tf2_ros
from tf2_ros import TransformException


# multiply two quaternions (Hamilton product)
def quat_mul(q1, q2):
    r1 = R.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = R.from_quat([q2.x, q2.y, q2.z, q2.w])
    r_out = r1 * r2
    x, y, z, w = r_out.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)

# inverse of a quaternion
def quat_inv(q):
    r = R.from_quat([q.x, q.y, q.z, q.w])
    rinv = r.inv()
    x, y, z, w = rinv.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)

def transform_optitrack_to_enu(pose_opt: Pose) -> Pose:
    # position, [Xe, Yn, Zu] = [Zf, Xr, Yu]
    px, py, pz = pose_opt.position.x, pose_opt.position.y, pose_opt.position.z
    pos_enu = Point(x=pz, y=px, z=py)

    # orientation, pre multiply by q_perm = rotation that maps X->Y, Y->Z, Z->X
    q_perm = Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
    q_in = pose_opt.orientation
    q_enu = quat_mul(q_perm, q_in)

    return Pose(position=pos_enu, orientation=q_enu)


def is_valid_pose(p: Pose, pos_eps=1e-6) -> bool:
    # many OptiTrack drivers publish a zeroed first sample
    if abs(p.position.x) < pos_eps and abs(p.position.y) < pos_eps and abs(p.position.z) < pos_eps:
        return False
    # orientation should be near unit length
    nq = p.orientation.x**2 + p.orientation.y**2 + p.orientation.z**2 + p.orientation.w**2
    return 0.5 < nq < 1.5


class MocapPathBuilder(Node):
    def __init__(self):
        super().__init__('path_builder')

        self.declare_parameter('frame_id', 'robot_1_map')
        self.declare_parameter('max_points', 2000)
        self.declare_parameter('pose_topic', 'mocap_pose')
        self.declare_parameter('path_topic', 'mocap_path')

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.max_points = self.get_parameter('max_points').get_parameter_value().integer_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value

        # Subscriber QoS for mocap
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # Publishers QoS for RViz
        pose_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        path_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, pose_pub_qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, path_pub_qos)

        self.path_msg = Path()
        self.path_msg.header = Header(frame_id=self.frame_id)

        self.subscription = self.create_subscription(
            RigidBodies, '/rigid_bodies', self.cb_rigid_bodies, sub_qos
        )

        # store the first valid pose, in ENU, as offset
        self.origin_offset = None
        self.origin_q = None   # <--- add this

        self.get_logger().info(
            f'Publishing Pose on "{self.pose_topic}" and Path on "{self.path_topic}", frame="{self.frame_id}"'
        )

    def cb_rigid_bodies(self, msg: RigidBodies):
        if not msg.rigidbodies:
            return

        rb = msg.rigidbodies[0]

        # convert OptiTrack pose into ENU first
        pose_enu = transform_optitrack_to_enu(rb.pose)

        # skip zeroed or invalid first samples
        if not is_valid_pose(pose_enu):
            return

        # record the first valid ENU pose as origin
        if self.origin_offset is None:
            self.origin_offset = pose_enu.position
        if self.origin_q is None:
            self.origin_q = pose_enu.orientation

        # Build pose relative to initial offset
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.frame_id

        ps.pose.position.x = pose_enu.position.x - self.origin_offset.x
        ps.pose.position.y = pose_enu.position.y - self.origin_offset.y
        ps.pose.position.z = pose_enu.position.z - self.origin_offset.z
        ps.pose.orientation = quat_mul(pose_enu.orientation, quat_inv(self.origin_q))

        self.pose_pub.publish(ps)

        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.header.frame_id = self.frame_id
        self.path_msg.poses.append(ps)

        if self.max_points > 0 and len(self.path_msg.poses) > self.max_points:
            del self.path_msg.poses[0:len(self.path_msg.poses) - self.max_points]

        self.path_pub.publish(self.path_msg)


def main():
    rclpy.init()
    node = MocapPathBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
