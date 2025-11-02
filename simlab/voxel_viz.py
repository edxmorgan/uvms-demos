#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node

import trimesh
from ament_index_python.packages import get_package_share_directory

from mesh_utils import collect_env_meshes, conc_env_trimesh


class VoxelVizNode(Node):
    def __init__(self):
        super().__init__("voxel_viz_node")
        self.declare_parameter('robot_description', '')

        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        if not urdf_string:
            self.get_logger().error('robot_description param is empty. Did you load it into the param server in launch')
            raise RuntimeError('no robot_description')

        # collect meshes
        robot_mesh_infos, env_mesh_infos = collect_env_meshes(urdf_string)
        if len(env_mesh_infos) == 0:
            self.get_logger().warn("No env meshes with prefix bathymetry_ found")
        else:
            self.get_logger().info(
                f"env links {[x['link'] for x in env_mesh_infos]}"
            )

        # merge meshes into one Trimesh in world frame
        env_mesh = conc_env_trimesh(env_mesh_infos)
        if env_mesh is None:
            self.get_logger().error("No environment mesh could be built")
            raise RuntimeError("empty env mesh")

        self.get_logger().info(
            f"concatenated env meshes, {len(env_mesh.faces)} faces total"
        )

        # choose voxel resolution
        self.voxel_size = 0.10  # meters per cell

        # load cached voxel centers from ros2_control_blue_reach_5 share
        # or build them and save there
        self.centers = self.load_or_build_voxels(env_mesh, self.voxel_size)

        self.get_logger().info(
            f"env voxel grid ready, {self.centers.shape[0]} occupied voxels at "
            f"{self.voxel_size} m"
        )

        # timer for future publishing of octomap_full
        self.timer = self.create_timer(0.05, self.tick)


    def tick(self):
        # next step is to convert self.centers to octomap and publish
        self.get_logger().info(
            f"tick, {self.centers.shape[0]} voxels cached in ros2_control_blue_reach_5 share, "
            "octomap publish TODO"
        )

    def get_cache_path(self, voxel_size: float):
        pkg_share = get_package_share_directory('ros2_control_blue_reach_5')

        # Put voxel cache in Bathymetry/voxels under that share directory
        voxels_dir = os.path.join(pkg_share, 'Bathymetry', 'voxels')
        os.makedirs(voxels_dir, exist_ok=True)

        fname = f"env_voxels_{voxel_size:.3f}m.npy"
        cache_path = os.path.join(voxels_dir, fname)

        return cache_path

    def load_or_build_voxels(self,
                             mesh: trimesh.Trimesh,
                             voxel_size: float):
        """
        Try to load cached centers from ros2_control_blue_reach_5 share.
        If not present, voxelize, save there, then return.
        """
        cache_path = self.get_cache_path(voxel_size)

        if os.path.exists(cache_path):
            self.get_logger().info(
                f"loading cached voxel centers from {cache_path}"
            )
            centers = np.load(cache_path)
            return centers

        # cache miss case
        self.get_logger().info(
            f"no cache found, voxelizing at {voxel_size} m and saving to {cache_path}"
        )

        centers, _ = self.voxelize_mesh(
            mesh,
            voxel_size,
            solid=False
        )

        # write cache file
        np.save(cache_path, centers)
        self.get_logger().info(
            f"saved {centers.shape[0]} voxel centers to {cache_path}"
        )

        return centers

    def voxelize_mesh(self,
                      mesh: trimesh.Trimesh,
                      voxel_size: float,
                      solid: bool = False):
        """
        Voxelize with trimesh.
        solid False gives surface shell voxels.
        solid True fills interior.
        """
        v = mesh.voxelized(pitch=voxel_size, method="subdivide")
        if solid:
            v = v.fill()

        centers = v.points.copy()
        return centers, voxel_size

def main():
    rclpy.init()
    node = VoxelVizNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
