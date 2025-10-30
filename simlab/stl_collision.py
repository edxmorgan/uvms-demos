#!/usr/bin/env python3
import numpy as np
import rclpy, trimesh, fcl
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF
import re
from urdf_parser_py.urdf import Mesh
RESET="\033[0m"; BOLD="\033[1m"; GRN="\033[32m"; CYN="\033[36m"; MAG="\033[35m"


def color(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c

def se3_from_rpy_xyz(rpy, xyz):
    """Return a 4x4 homogeneous transform from rpy and xyz."""
    roll, pitch, yaw = rpy
    tx, ty, tz = xyz
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T

def fcl_bvh_from_mesh(path, scale=[1.0, 1.0, 1.0], rpy=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 0.0)):
    scene_or_mesh = trimesh.load(path, force='scene')
    mesh = scene_or_mesh.dump(concatenate=True) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh
    if scale != 1.0:
        mesh.apply_scale(np.array(scale))
    if any(abs(v) > 1e-12 for v in rpy) or any(abs(v) > 1e-12 for v in xyz):
        T = se3_from_rpy_xyz(rpy, xyz)  # 4x4
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


_ROS2_CTRL_RE = re.compile(r"<\s*ros2_control[\s\S]*?</\s*ros2_control\s*>", re.MULTILINE)

def _parse_urdf_no_ros2_control(urdf_string: str) -> URDF:
    urdf_clean = _ROS2_CTRL_RE.sub("", urdf_string)
    return URDF.from_xml_string(urdf_clean)

def _strip_file_prefix(uri: str) -> str:
    # Remove leading file:// if present
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri

def log_all_visual_meshes(self, urdf_string: str):
    model = _parse_urdf_no_ros2_control(urdf_string)
    visuals_info = []

    for link in model.links or []:
        for idx, vis in enumerate(getattr(link, "visuals", []) or []):
            geom = getattr(vis, "geometry", None)
            if not isinstance(geom, Mesh):
                continue

            origin = getattr(vis, "origin", None)

            xyz = list(getattr(origin, "xyz", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
            rpy = list(getattr(origin, "rpy", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
            scale = list(getattr(geom, "scale", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0])

            raw_uri = getattr(geom, "filename", "")
            uri = _strip_file_prefix(raw_uri)

            info = {
                "link": link.name,
                "xyz": xyz,
                "rpy": rpy,
                "scale": scale,
                "uri": uri,
            }
            visuals_info.append(info)

            # self.get_logger().info(
            #     f"link {link.name} uri {uri} xyz {xyz} rpy {rpy} scale {scale}"
            # )

    return visuals_info

class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.declare_parameter('robot_description', '')

        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        if not urdf_string:
            self.get_logger().error('robot_description param is empty. Did you load it into the param server in launch')
            raise RuntimeError('no robot_description')
    
        # walk all links and collect meshes
        meshes_info = log_all_visual_meshes(self, urdf_string)

        # split links into robot links and environment links
        robot_links = []
        env_links = []

        for m in meshes_info:
            link_name = m['link']
            if link_name.startswith('robot_'):
                robot_links.append(m)
            elif link_name.startswith('bathymetry_'):
                env_links.append(m)

        self.get_logger().info(f'robot_links { [x["link"] for x in robot_links] }')
        self.get_logger().info(f'env_links { [x["link"] for x in env_links] }')

        self.bodies_robot = self.build_bodies(robot_links, "robot")
        self.bodies_env   = self.build_bodies(env_links,   "env")

        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        self.contact_pub = self.create_publisher(Marker, 'contact_markers', 10)
        self.mesh_pub = self.create_publisher(Marker, 'mesh_debug', 10)

        self.timer = self.create_timer(0.05, self.tick)

    def tick(self):
        # sync transforms for every body
        ok_list = list(map(self.set_from_tf, self.bodies_robot + self.bodies_env))
        if not np.all(ok_list):
            return

        CONTACT_MARKER_SIZE = 0.15
        NEAR_THRESH = 1.0

        # clear old markers once per tick
        clear = Marker()
        clear.header.frame_id = 'world'
        clear.action = Marker.DELETEALL
        self.contact_pub.publish(clear)

        req = fcl.CollisionRequest(num_max_contacts=200, enable_contact=True)
        dreq = fcl.DistanceRequest(enable_nearest_points=True)

        marker_id_counter = 0

        # loop every robot link vs every env object
        for rob in self.bodies_robot:
            for env in self.bodies_env:
                # distance first
                dres = fcl.DistanceResult()
                fcl.distance(rob["fcl_obj"], env["fcl_obj"], dreq, dres)

                # draw yellow near miss if within NEAR_THRESH and not actually colliding
                if 0.0 < dres.min_distance < NEAR_THRESH:
                    px, py, pz = dres.nearest_points[1]  # point on env surface

                    m = Marker()
                    m.header.frame_id = 'world'
                    m.ns = 'near'
                    m.id = marker_id_counter
                    marker_id_counter += 1
                    m.type = Marker.SPHERE
                    m.action = Marker.ADD
                    m.scale.x = m.scale.y = m.scale.z = CONTACT_MARKER_SIZE
                    m.color = ColorRGBA(r=1.0, g=1.0, b=0.1, a=1.0)  # yellow
                    m.pose.position.x = px
                    m.pose.position.y = py
                    m.pose.position.z = pz
                    self.contact_pub.publish(m)

                    # log near miss distance
                    # self.get_logger().info(
                    #     f'near {rob["name"]} vs {env["name"]}, dist {dres.min_distance:.3f} m'
                    # )

                # now collide
                cres = fcl.CollisionResult()
                hit_count = fcl.collide(rob["fcl_obj"], env["fcl_obj"], req, cres)

                if hit_count > 0:
                    # red markers for contact points
                    for c in cres.contacts:
                        m = Marker()
                        m.header.frame_id = 'world'
                        m.ns = 'contact'
                        m.id = marker_id_counter
                        marker_id_counter += 1
                        m.type = Marker.SPHERE
                        m.action = Marker.ADD
                        m.scale.x = m.scale.y = m.scale.z = CONTACT_MARKER_SIZE
                        m.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)  # red
                        m.pose.position.x, m.pose.position.y, m.pose.position.z = c.pos
                        self.contact_pub.publish(m)

                    # log collision
                    # self.get_logger().info(
                    #     f'collision {rob["name"]} vs {env["name"]}, contacts {hit_count}'
                    # )


    def build_bodies(self, link_list, kind):
        """
        link_list is a list of dicts from meshes_info
        kind is just a string for logging, like "robot" or "env"
        returns a list of {name, frame, fcl_obj}
        """
        out = []
        for m in link_list:
            path_abs  = m['uri']
            scale_vec = m['scale']
            xyz       = tuple(m['xyz'])
            rpy       = tuple(m['rpy'])

            bvh, nV, nF = fcl_bvh_from_mesh(path_abs, scale_vec, rpy, xyz)
            obj = fcl.CollisionObject(bvh)

            out.append({
                "name":  m['link'],
                "frame": m['link'],   # assume TF frame matches link name
                "fcl_obj": obj,
            })

            self.get_logger().info(f'{kind} body {m["link"]} verts {nV} faces {nF}')

        return out

    def set_from_tf(self, body):
        try:
            t: TransformStamped = self.tf_buf.lookup_transform(
                'world',
                body["frame"],
                rclpy.time.Time()
            )
            q = t.transform.rotation
            p = t.transform.translation
            body["fcl_obj"].setTransform(
                fcl.Transform([q.w, q.x, q.y, q.z], [p.x, p.y, p.z])
            )
            return True
        except Exception:
            return False
        

def main():
    rclpy.init()
    node = CollisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
