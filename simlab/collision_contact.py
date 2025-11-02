#!/usr/bin/env python3
import numpy as np
import rclpy, fcl
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from mesh_utils import fcl_bvh_from_mesh, make_marker, collect_env_meshes, color



class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.declare_parameter('robot_description', '')

        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        if not urdf_string:
            self.get_logger().error('robot_description param is empty. Did you load it into the param server in launch')
            raise RuntimeError('no robot_description')
    
        # walk all links and collect meshes
        robot_links, env_links = collect_env_meshes(urdf_string)

        self.get_logger().info(f'robot_links { [x["link"] for x in robot_links] }')
        self.get_logger().info(f'env_links { [x["link"] for x in env_links] }')

        self.bodies_robot = self.build_fcl_bodies(robot_links, "robot")
        self.bodies_env   = self.build_fcl_bodies(env_links,   "env")

        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)

        self.contact_pub = self.create_publisher(Marker, 'contact_markers', 10)

        self.timer = self.create_timer(0.05, self.tick)


    def tick(self):
        ok_list = list(map(self.set_from_tf, self.bodies_robot + self.bodies_env))
        if not np.all(ok_list):
            return

        CONTACT_MARKER_SIZE = 0.05
        NEAR_THRESH = 1.0

        # clear old marker spheres
        clear = Marker()
        clear.header.frame_id = 'world'
        clear.action = Marker.DELETEALL
        self.contact_pub.publish(clear)

        req  = fcl.CollisionRequest(num_max_contacts=200, enable_contact=True)
        dreq = fcl.DistanceRequest(enable_nearest_points=True)

        marker_id_counter = 0

        for rob in self.bodies_robot:
            for env in self.bodies_env:

                # shortest Euclidean distance between the robot object and the environment object.
                dres = fcl.DistanceResult()
                fcl.distance(rob["fcl_obj"], env["fcl_obj"], dreq, dres)
                rob_px, rob_py, rob_pz = dres.nearest_points[0] # point on robot surface
                env_px, env_py, env_pz = dres.nearest_points[1] # point on env surface

                # collision query
                cres = fcl.CollisionResult()
                hit_count = fcl.collide(rob["fcl_obj"], env["fcl_obj"], req, cres)
                if hit_count <= 0:
                    continue

                for c in cres.contacts:
                    p_world = np.array(c.pos, dtype=float)

                    n_world_env_on_robot = -np.array(c.normal, dtype=float)
                    depth = float(c.penetration_depth)

                    # visualize contact point as red sphere (optional but helpful)
                    red = color(r=1.0, g=0.1, b=0.1, a=1.0) # red
                    sphere = make_marker('contact', marker_id_counter, 'world', CONTACT_MARKER_SIZE, p_world, red)
                    marker_id_counter += 1
                    self.contact_pub.publish(sphere)

    def build_fcl_bodies(self, link_list, kind):
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

            # update FCL object transform for collision math
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
