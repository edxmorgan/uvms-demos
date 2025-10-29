#!/usr/bin/env python3
import os, math
import numpy as np
import rclpy, trimesh, fcl
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

RESET="\033[0m"; BOLD="\033[1m"; GRN="\033[32m"; CYN="\033[36m"; MAG="\033[35m"


def color(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c

def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], float)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], float)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], float)
    return Rz @ Ry @ Rx

def fcl_bvh_from_mesh(path, scale=1.0, rpy=(0.0, 0.0, 0.0)):
    scene_or_mesh = trimesh.load(path, force='scene')
    mesh = scene_or_mesh.dump(concatenate=True) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh
    if scale != 1.0:
        mesh.apply_scale(float(scale))
    if any(abs(a) > 1e-12 for a in rpy):
        R = rpy_to_R(*rpy)
        T = np.eye(4)
        T[:3, :3] = R
        mesh.apply_transform(T)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if V.size == 0 or F.size == 0:
        raise RuntimeError(f'Empty mesh after transforms, check file, {path}')
    bvh = fcl.BVHModel()
    bvh.beginModel(V.shape[0], F.shape[0])
    bvh.addSubModel(V, F)
    bvh.endModel()
    return bvh, V.shape[0], F.shape[0]

def make_mesh_marker(ns, mid, frame, package_uri, scale_xyz, color):
    m = Marker()
    m.ns = ns
    m.id = mid
    m.header.frame_id = frame
    m.type = Marker.MESH_RESOURCE
    m.action = Marker.ADD
    m.mesh_resource = package_uri
    m.mesh_use_embedded_materials = True
    m.scale.x, m.scale.y, m.scale.z = scale_xyz
    m.color = color
    return m

class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        self.contact_pub = self.create_publisher(Marker, 'contact_markers', 10)
        self.mesh_pub = self.create_publisher(Marker, 'mesh_debug', 10)

        pkg = 'ros2_control_blue_reach_5'
        pkg_share = get_package_share_directory(pkg)

        robot_mesh_path = os.path.join(pkg_share, 'blue/meshes/bluerov2_heavy_reach/bluerov2_heavy_reach.dae')
        seafloor_mesh_path = os.path.join(pkg_share, 'Bathymetry/meshes/hawaii_cropped.stl')
        shipwreck_mesh_path = os.path.join(pkg_share, 'Bathymetry/meshes/manhomansett.stl')

        self.robot_frame = 'robot_1_base_link'
        self.seafloor_frame = 'bathymetry_seafloor'
        self.shipwreck_frame = 'bathymetry_shipwreck'

        robot_scale = 0.025
        robot_rpy = (0.0, 0.0, 0.0)

        seafloor_scale = 0.4
        seafloor_rpy = (0.0, 0.0, 0.0)

        shipwreck_scale = 5.0
        shipwreck_rpy = (math.pi/2.0, 0.0, 0.0)

        self.get_logger().info(f'Loading robot mesh, {robot_mesh_path}')
        self.get_logger().info(f'Loading seafloor mesh, {seafloor_mesh_path}')
        self.get_logger().info(f'Loading shipwreck mesh, {shipwreck_mesh_path}')

        robot_bvh, nV_r, nF_r = fcl_bvh_from_mesh(robot_mesh_path, robot_scale, robot_rpy)
        seafloor_bvh, nV_s, nF_s = fcl_bvh_from_mesh(seafloor_mesh_path, seafloor_scale, seafloor_rpy)
        shipwreck_bvh, nV_w, nF_w = fcl_bvh_from_mesh(shipwreck_mesh_path, shipwreck_scale, shipwreck_rpy)

        self.robot_obj = fcl.CollisionObject(robot_bvh)
        self.seafloor_obj = fcl.CollisionObject(seafloor_bvh)
        self.shipwreck_obj = fcl.CollisionObject(shipwreck_bvh)

        self.get_logger().info(f'{GRN}Robot BVH built{RESET}, vertices {BOLD}{nV_r}{RESET}, faces {BOLD}{nF_r}{RESET}')
        self.get_logger().info(f'{CYN}Seafloor BVH built{RESET}, vertices {BOLD}{nV_s}{RESET}, faces {BOLD}{nF_s}{RESET}')
        self.get_logger().info(f'{MAG}Shipwreck BVH built{RESET}, vertices {BOLD}{nV_w}{RESET}, faces {BOLD}{nF_w}{RESET}')

        robot_uri = f'package://{pkg}/blue/meshes/bluerov2_heavy_reach/bluerov2_heavy_reach.dae'
        seafloor_uri = f'package://{pkg}/Bathymetry/meshes/hawaii_cropped.stl'
        shipwreck_uri = f'package://{pkg}/Bathymetry/meshes/manhomansett.stl'

        self.mesh_markers = [
            make_mesh_marker('debug', 1, self.robot_frame,    robot_uri,    (robot_scale,)*3,    color(1.0, 1.0, 1.0, 0.95)),
            make_mesh_marker('debug', 2, self.seafloor_frame, seafloor_uri, (seafloor_scale,)*3, color(0.6, 0.7, 0.9, 0.7)),
            make_mesh_marker('debug', 3, self.shipwreck_frame, shipwreck_uri, (shipwreck_scale,)*3, color(0.9, 0.7, 0.7, 0.7)),
        ]


        self.timer = self.create_timer(0.05, self.tick)

    def set_from_tf(self, obj, target_frame):
        try:
            t: TransformStamped = self.tf_buf.lookup_transform('world', target_frame, rclpy.time.Time())
            q = t.transform.rotation
            p = t.transform.translation
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

        # publish debug meshes
        for m in self.mesh_markers:
            self.mesh_pub.publish(m)

        # one request, fresh result objects
        req = fcl.CollisionRequest(num_max_contacts=64, enable_contact=True)

        res_floor = fcl.CollisionResult()
        count_floor = fcl.collide(self.robot_obj, self.seafloor_obj, req, res_floor)

        res_wreck = fcl.CollisionResult()
        count_wreck = fcl.collide(self.robot_obj, self.shipwreck_obj, req, res_wreck)

        # clear and draw
        clear = Marker()
        clear.header.frame_id = 'world'
        clear.action = Marker.DELETEALL
        self.contact_pub.publish(clear)

        CONTACT_MARKER_SIZE = 0.08

        if count_floor > 0:
            for i, c in enumerate(res_floor.contacts):
                m = Marker()
                m.header.frame_id = 'world'
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.scale.x = m.scale.y = m.scale.z = CONTACT_MARKER_SIZE
                m.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)  # red
                m.pose.position.x, m.pose.position.y, m.pose.position.z = c.pos
                m.id = i
                self.contact_pub.publish(m)

        if count_wreck > 0:
            base_id = 1000
            for i, c in enumerate(res_wreck.contacts):
                m = Marker()
                m.header.frame_id = 'world'
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.scale.x = m.scale.y = m.scale.z = CONTACT_MARKER_SIZE
                m.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0)  # green
                m.pose.position.x, m.pose.position.y, m.pose.position.z = c.pos
                m.id = base_id + i
                self.contact_pub.publish(m)

        self.get_logger().info(f'contacts, seafloor {count_floor}, shipwreck {count_wreck}')


def main():
    rclpy.init()
    node = CollisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
