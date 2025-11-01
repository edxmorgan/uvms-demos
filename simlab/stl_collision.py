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
from geometry_msgs.msg import TransformStamped, WrenchStamped
from tf2_ros import TransformBroadcaster

RESET="\033[0m"; BOLD="\033[1m"; GRN="\033[32m"; CYN="\033[36m"; MAG="\033[35m"


def color(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def quat_from_vec_align_x(v):
    """
    Build a quaternion [w, x, y, z] whose +X axis aligns with v.
    If v is ~0, return identity.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    d = v / n
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    rot, _ = R.align_vectors(d.reshape(1,3), x_axis.reshape(1,3))
    q_xyzw = rot.as_quat()  # [x,y,z,w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    return q_wxyz

def quat_to_rotmat(q_wxyz):
    """
    q_wxyz is [w,x,y,z].
    returns 3x3 rotation matrix world_from_quat.
    """
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    w, x, y, z = q_wxyz
    q_xyzw = np.array([x, y, z, w], dtype=float)
    rot = R.from_quat(q_xyzw)
    return rot.as_matrix()



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

        self.tf_broadcaster = TransformBroadcaster(self)
        self.contact_wrench_debug_pub = self.create_publisher(WrenchStamped, 'contact_wrench_contacts', 10)
        self.contact_wrench_pub = self.create_publisher(WrenchStamped, 'contact_wrench_body', 10)

        self.timer = self.create_timer(0.05, self.tick)


    def tick(self):
        ok_list = list(map(self.set_from_tf, self.bodies_robot + self.bodies_env))
        if not np.all(ok_list):
            return

        CONTACT_MARKER_SIZE = 0.15
        MAX_CONTACT_FRAMES = 20
        BASE_LINK = 'robot_1_base_link'


        STIFFNESS = 1.0    # N per m
        DAMPING   = 0.0    # N per (m per s)


        # clear old marker spheres
        clear = Marker()
        clear.header.frame_id = 'world'
        clear.action = Marker.DELETEALL
        self.contact_pub.publish(clear)

        # per tick accumulators
        self.contact_wrenches_world = {}
        contact_list_this_tick = []  # each item will be dict with p_world, f_lin_world, contact_frame_name

        req  = fcl.CollisionRequest(num_max_contacts=200, enable_contact=True)
        dreq = fcl.DistanceRequest(enable_nearest_points=True)

        marker_id_counter = 0

        for rob in self.bodies_robot:
            for env in self.bodies_env:
                # collision query
                cres = fcl.CollisionResult()
                hit_count = fcl.collide(rob["fcl_obj"], env["fcl_obj"], req, cres)
                if hit_count <= 0:
                    continue

                for c in cres.contacts:
                    p_world = np.array(c.pos, dtype=float)

                    n_world_env_on_robot = -np.array(c.normal, dtype=float)
                    depth = float(c.penetration_depth)

                    v_rel_n   = 0.0
                    F_n = STIFFNESS * depth + DAMPING * v_rel_n
                    if F_n < 0.0:
                        F_n = 0.0
 
                    f_lin_world = F_n * n_world_env_on_robot  # force on robot in world frame

                    # torque about link origin in world frame
                    link_pos_world = rob["last_world_pos"]
                    r_world = p_world - link_pos_world
                    tau_world = np.cross(r_world, f_lin_world)

                    # accumulate net wrench for this link in world frame
                    wrench_world_vec = np.concatenate([f_lin_world, tau_world], axis=0)
                    link_key = rob["name"]
                    if link_key not in self.contact_wrenches_world:
                        self.contact_wrenches_world[link_key] = np.zeros(6, dtype=float)
                    self.contact_wrenches_world[link_key] += wrench_world_vec

                    # visualize contact point as red sphere (optional but helpful)
                    sphere = Marker()
                    sphere.header.frame_id = 'world'
                    sphere.ns = 'contact'
                    sphere.id = marker_id_counter
                    marker_id_counter += 1
                    sphere.type = Marker.SPHERE
                    sphere.action = Marker.ADD
                    sphere.scale.x = sphere.scale.y = sphere.scale.z = CONTACT_MARKER_SIZE
                    sphere.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)  # red
                    sphere.pose.position.x = float(p_world[0])
                    sphere.pose.position.y = float(p_world[1])
                    sphere.pose.position.z = float(p_world[2])
                    self.contact_pub.publish(sphere)

                    # save contact info for later TF and wrench publishing
                    if len(contact_list_this_tick) < MAX_CONTACT_FRAMES:
                        contact_list_this_tick.append({
                            "p_world": p_world,
                            "f_lin_world": f_lin_world
                        })

        # publish per contact TF frames and per contact wrenches
        # reuse a fixed pool of names: contact_0 ... contact_(MAX_CONTACT_FRAMES-1)
        now_stamp = self.get_clock().now().to_msg()

        for i in range(MAX_CONTACT_FRAMES):
            frame_id = f"contact_{i}"

            if i < len(contact_list_this_tick):
                p_world = contact_list_this_tick[i]["p_world"]
                f_lin_world = contact_list_this_tick[i]["f_lin_world"]

                # orientation so +X lines up with force
                q_wxyz = quat_from_vec_align_x(f_lin_world)

                # broadcast TF for this contact frame
                tf_msg = TransformStamped()
                tf_msg.header.stamp = now_stamp
                tf_msg.header.frame_id = 'world'
                tf_msg.child_frame_id = frame_id
                tf_msg.transform.translation.x = float(p_world[0])
                tf_msg.transform.translation.y = float(p_world[1])
                tf_msg.transform.translation.z = float(p_world[2])
                tf_msg.transform.rotation.w = float(q_wxyz[0])
                tf_msg.transform.rotation.x = float(q_wxyz[1])
                tf_msg.transform.rotation.y = float(q_wxyz[2])
                tf_msg.transform.rotation.z = float(q_wxyz[3])
                self.tf_broadcaster.sendTransform(tf_msg)

                # express force in contact frame
                # quat_to_rotmat returns world_from_frame
                R_world_from_contact = quat_to_rotmat(q_wxyz)
                R_contact_from_world = R_world_from_contact.T
                f_contact = R_contact_from_world @ f_lin_world

                wmsg = WrenchStamped()
                wmsg.header.stamp = now_stamp
                wmsg.header.frame_id = frame_id
                wmsg.wrench.force.x = float(f_contact[0])
                wmsg.wrench.force.y = float(f_contact[1])
                wmsg.wrench.force.z = float(f_contact[2])
                wmsg.wrench.torque.x = 0.0
                wmsg.wrench.torque.y = 0.0
                wmsg.wrench.torque.z = 0.0
                self.contact_wrench_debug_pub.publish(wmsg)

            else:
                # no contact to show in this slot this tick
                # move frame to origin and publish zero wrench so RViz Wrench shows nothing meaningful
                tf_msg = TransformStamped()
                tf_msg.header.stamp = now_stamp
                tf_msg.header.frame_id = 'world'
                tf_msg.child_frame_id = frame_id
                tf_msg.transform.translation.x = 0.0
                tf_msg.transform.translation.y = 0.0
                tf_msg.transform.translation.z = -9999.0  # bury it far below view
                tf_msg.transform.rotation.w = 1.0
                tf_msg.transform.rotation.x = 0.0
                tf_msg.transform.rotation.y = 0.0
                tf_msg.transform.rotation.z = 0.0
                self.tf_broadcaster.sendTransform(tf_msg)

                wmsg = WrenchStamped()
                wmsg.header.stamp = now_stamp
                wmsg.header.frame_id = frame_id
                wmsg.wrench.force.x = 0.0
                wmsg.wrench.force.y = 0.0
                wmsg.wrench.force.z = 0.0
                wmsg.wrench.torque.x = 0.0
                wmsg.wrench.torque.y = 0.0
                wmsg.wrench.torque.z = 0.0
                self.contact_wrench_debug_pub.publish(wmsg)

        # now compute and publish total wrench on the base link
        if BASE_LINK in self.contact_wrenches_world:
            wrench_world_sum = self.contact_wrenches_world[BASE_LINK]
        else:
            wrench_world_sum = np.zeros(6, dtype=float)

        f_world_sum   = wrench_world_sum[0:3]
        tau_world_sum = wrench_world_sum[3:6]

        base_quat_wxyz = self.get_quat_for_link(BASE_LINK)
        R_world_from_base = quat_to_rotmat(base_quat_wxyz)
        R_base_from_world = R_world_from_base.T

        f_body   = R_base_from_world @ f_world_sum
        tau_body = R_base_from_world @ tau_world_sum

        w_total = WrenchStamped()
        w_total.header.stamp = now_stamp
        w_total.header.frame_id = BASE_LINK
        w_total.wrench.force.x  = float(f_body[0])
        w_total.wrench.force.y  = float(f_body[1])
        w_total.wrench.force.z  = float(f_body[2])
        w_total.wrench.torque.x = float(tau_body[0])
        w_total.wrench.torque.y = float(tau_body[1])
        w_total.wrench.torque.z = float(tau_body[2])

        self.contact_wrench_pub.publish(w_total)


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

            # cache latest world pose for this link
            body["last_world_pos"] = np.array([p.x, p.y, p.z], dtype=float)
            body["last_world_quat"] = np.array([q.w, q.x, q.y, q.z], dtype=float)  # wxyz

            # update FCL object transform for collision math
            body["fcl_obj"].setTransform(
                fcl.Transform([q.w, q.x, q.y, q.z], [p.x, p.y, p.z])
            )
            return True
        except Exception:
            return False

    def get_link_position_world(self, link_name):
        for b in (self.bodies_robot + self.bodies_env):
            if b["name"] == link_name and "last_world_pos" in b:
                return b["last_world_pos"]
        return np.array([0.0, 0.0, 0.0], dtype=float)

    def get_quat_for_link(self, link_name):
        for b in (self.bodies_robot + self.bodies_env):
            if b["name"] == link_name and "last_world_quat" in b:
                return b["last_world_quat"]
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def main():
    rclpy.init()
    node = CollisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
