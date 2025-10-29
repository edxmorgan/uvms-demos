#!/usr/bin/env python3
import rclpy, numpy as np, trimesh, fcl
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory
import os

RESET = "\033[0m"; BOLD="\033[1m"; GRN="\033[32m"; CYN="\033[36m"

def fcl_bvh_from_mesh(path):
    scene_or_mesh = trimesh.load(path, force='scene')
    mesh = scene_or_mesh.dump(concatenate=True) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    bvh = fcl.BVHModel()
    bvh.beginModel(V.shape[0], F.shape[0]); bvh.addSubModel(V, F); bvh.endModel()
    return bvh, V.shape[0], F.shape[0]

class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.tf_buf = Buffer(); self.tf = TransformListener(self.tf_buf, self)
        self.marker_pub = self.create_publisher(Marker, 'contact_markers', 10)

        pkg_share = get_package_share_directory('ros2_control_blue_reach_5')

        robot_mesh_path = os.path.join(pkg_share, 'blue/meshes/bluerov2_heavy_reach/bluerov2_heavy_reach.dae')
        robot_frame = 'robot_1_base_link'

        seafloor_mesh_path = os.path.join(pkg_share, 'Bathymetry/meshes/hawaii_cropped.stl')
        seafloor_frame = 'bathymetry_seafloor'

        shipwreck_mesh_path = os.path.join(pkg_share, 'Bathymetry/meshes/manhomansett.stl')
        shipwreck_frame = 'bathymetry_shipwreck'

        self.get_logger().info(f'Loading robot mesh, {robot_mesh_path}')
        self.get_logger().info(f'Loading seafloor mesh, {seafloor_mesh_path}')
        self.get_logger().info(f'Loading shipwreck mesh, {shipwreck_mesh_path}')

        robot_bvh, nV_r, nF_r = fcl_bvh_from_mesh(robot_mesh_path)
        seafloor_bvh, nV_s, nF_s = fcl_bvh_from_mesh(seafloor_mesh_path)
        shipwreck_bvh, nV_w, nF_w = fcl_bvh_from_mesh(shipwreck_mesh_path)

        self.robot_obj = fcl.CollisionObject(robot_bvh)
        self.seafloor_obj = fcl.CollisionObject(seafloor_bvh)
        self.shipwreck_obj = fcl.CollisionObject(shipwreck_bvh)

        self.get_logger().info(f'{GRN}Robot BVH built{RESET}, vertices {BOLD}{nV_r}{RESET}, faces {BOLD}{nF_r}{RESET}')
        self.get_logger().info(f'{CYN}Seafloor BVH built{RESET}, vertices {BOLD}{nV_s}{RESET}, faces {BOLD}{nF_s}{RESET}')
        self.get_logger().info(f'{CYN}Shipwreck BVH built{RESET}, vertices {BOLD}{nV_w}{RESET}, faces {BOLD}{nF_w}{RESET}')

        self.robot_frame = robot_frame
        self.seafloor_frame = seafloor_frame
        self.shipwreck_frame = shipwreck_frame

        self.timer = self.create_timer(0.05, self.tick)

    def set_from_tf(self, obj, target_frame):
        try:
            t: TransformStamped = self.tf_buf.lookup_transform('world', target_frame, rclpy.time.Time())
            q = t.transform.rotation; p = t.transform.translation
            obj.setTransform(fcl.Transform([q.w, q.x, q.y, q.z], [p.x, p.y, p.z]))
            return True
        except Exception:
            return False

    def tick(self):
        ok_robot = self.set_from_tf(self.robot_obj, self.robot_frame)
        ok_floor = self.set_from_tf(self.seafloor_obj, self.seafloor_frame)
        ok_wreck = self.set_from_tf(self.shipwreck_obj, self.shipwreck_frame)
        if not (ok_robot and ok_floor and ok_wreck):
            return

        req = fcl.CollisionRequest(num_max_contacts=32, enable_contact=True)
        res = fcl.CollisionResult()
        in_contact_floor = fcl.collide(self.robot_obj, self.seafloor_obj, req, res)
        
        self.get_logger().info(f'in_contact_floor : {in_contact_floor}')

        # clear old markers
        clear = Marker(); clear.header.frame_id = 'world'; clear.action = Marker.DELETEALL
        self.marker_pub.publish(clear)

        CONTACT_MARKER_SIZE = 0.08  # meters

        if in_contact_floor > 0:
            for i, c in enumerate(res.contacts):
                m = Marker()
                m.header.frame_id = 'world'
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.scale.x = m.scale.y = m.scale.z = CONTACT_MARKER_SIZE
                m.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
                m.pose.position.x, m.pose.position.y, m.pose.position.z = c.pos
                m.id = i
                self.marker_pub.publish(m)

def main():
    rclpy.init(); node = CollisionNode(); rclpy.spin(node); rclpy.shutdown()

if __name__ == '__main__':
    main()
