import numpy as np
import trimesh, fcl
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R
import re
from urdf_parser_py.urdf import URDF, Mesh as URDFMesh


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

def make_marker(ns, mid, frame, scale_xyz, pos_world, color):
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = scale_xyz
    m.color = color
    m.pose.position.x = float(pos_world[0])
    m.pose.position.y = float(pos_world[1])
    m.pose.position.z = float(pos_world[2])
    return m


def _parse_urdf_no_ros2_control(urdf_string: str) -> URDF:
    _ROS2_CTRL_RE = re.compile(r"<\s*ros2_control[\s\S]*?</\s*ros2_control\s*>", re.MULTILINE)
    urdf_clean = _ROS2_CTRL_RE.sub("", urdf_string)
    return URDF.from_xml_string(urdf_clean)

def _strip_file_prefix(uri: str) -> str:
    # Remove leading file:// if present
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri

def collect_env_meshes(urdf_string: str):
    """
    Parse the URDF and return 2 lists:
      robot_out: meshes from links whose name starts with 'robot_'
      env_out:   meshes from links whose name starts with 'bathymetry_'
    """

    model = _parse_urdf_no_ros2_control(urdf_string)

    robot_out = []
    env_out = []

    # walk all links in the URDF
    for link in model.links or []:

        link_name = link.name

        # only care about robot_... and bathymetry_...
        is_robot_link = link_name.startswith("robot_")
        is_env_link   = link_name.startswith("bathymetry_")

        if not (is_robot_link or is_env_link):
            continue  # skip early

        # walk all visuals in this link
        for vis in getattr(link, "visuals", []) or []:
            geom = getattr(vis, "geometry", None)
            if not isinstance(geom, URDFMesh):
                continue  # skip non-mesh visuals, like primitives etc if any

            origin = getattr(vis, "origin", None)

            xyz = list(
                getattr(origin, "xyz", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            )
            rpy = list(
                getattr(origin, "rpy", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            )
            scale = list(
                getattr(geom, "scale", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0]
            )

            raw_uri = getattr(geom, "filename", "")
            uri = _strip_file_prefix(raw_uri)

            entry = {
                "link":  link_name,
                "uri":   uri,
                "scale": scale,
                "xyz":   xyz,
                "rpy":   rpy,
            }

            # push to the appropriate list
            if is_robot_link:
                robot_out.append(entry)
            if is_env_link:
                env_out.append(entry)

    return robot_out, env_out


def conc_env_trimesh(env_mesh_infos):
    meshes = []

    for info in env_mesh_infos:
        path_abs = info["uri"]
        scale_xyz = np.array(info["scale"], dtype=float)
        xyz = np.array(info["xyz"], dtype=float)
        rpy = np.array(info["rpy"], dtype=float)

        scene_or_mesh = trimesh.load(path_abs, force='scene')
        mesh = (
            scene_or_mesh.dump(concatenate=True)
            if isinstance(scene_or_mesh, trimesh.Scene)
            else scene_or_mesh
        )

        # scale
        mesh.apply_scale(scale_xyz)

        # apply local rpy, xyz into mesh
        T_local = trimesh.transformations.euler_matrix(
            rpy[0], rpy[1], rpy[2], axes="sxyz"
        )
        T_local[0:3, 3] = xyz
        mesh.apply_transform(T_local)

        meshes.append(mesh)

    if len(meshes) == 0:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)